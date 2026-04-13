"""
SFT on reasoning traces (data/reasoning_traces.jsonl).

Trains CodingSSM to produce <think>...</think> + code format.
Run AFTER gen_traces.sh generates the trace dataset.

Usage:
    python -m train.sft_reasoning \
        --traces data/reasoning_traces.jsonl \
        --output-dir checkpoints/sft \
        --model-size 700m \
        --epochs 3

    Multi-GPU: pass --data-parallel only if each GPU has enough VRAM; the default is
    single-GPU training when several devices are visible (avoids OOM on 2×T4).
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers.optimization import Adafactor

from arch import CodingSSM
from arch.config import ModelConfig700M, ModelConfig3B


# ─── Dataset ─────────────────────────────────────────────────────────────────

class ReasoningTraceDataset(Dataset):
    """Load reasoning traces from JSONL and tokenize into ChatML format."""

    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        print(f"[DATA] Loaded {len(self.records)} reasoning traces from {jsonl_path}")

    def __len__(self):
        return len(self.records)

    _SYSTEM = (
        "You are an expert Python programmer and software engineer. "
        "Think carefully step by step.\n"
        "Use <think> tags to show your reasoning, then provide a clear solution.\n"
        "Format:\n<think>\n[your reasoning]\n</think>\n[solution]"
    )

    def __getitem__(self, idx):
        rec = self.records[idx]
        if "chatml" in rec:
            text = rec["chatml"]
        else:
            # Build ChatML from reasoning trace fields
            thinking = (rec.get("thinking") or "").strip()
            solution = (rec.get("solution") or "").strip()
            prompt   = (rec.get("prompt")   or rec.get("question") or "").strip()
            assistant = f"<think>\n{thinking}\n</think>\n```python\n{solution}\n```" if thinking else solution
            text = (
                f"<|im_start|>system\n{self._SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant}<|im_end|>"
            )

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = ids[: self.max_length]

        # Build labels: mask everything up to (and including) "assistant\n"
        # Find the last <|im_start|>assistant\n token boundary
        assistant_tag = "<|im_start|>assistant\n"
        assistant_ids = self.tokenizer.encode(assistant_tag, add_special_tokens=False)
        a_len = len(assistant_ids)

        # Find position of last occurrence
        mask_until = 0
        for i in range(len(ids) - a_len, -1, -1):
            if ids[i : i + a_len] == assistant_ids:
                mask_until = i + a_len
                break

        labels = ids[:]
        for i in range(mask_until):
            labels[i] = -100

        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch):
    input_ids_list, labels_list = zip(*batch)
    max_len = max(x.shape[0] for x in input_ids_list)

    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (ids, lbls) in enumerate(zip(input_ids_list, labels_list)):
        L = ids.shape[0]
        padded_ids[i, :L] = ids
        padded_labels[i, :L] = lbls

    return {"input_ids": padded_ids, "labels": padded_labels}


# ─── Trainer ─────────────────────────────────────────────────────────────────

def train(
    model: CodingSSM,
    dataloader: DataLoader,
    model_cfg,
    output_dir: Path,
    epochs: int = 3,
    lr: float = 3e-4,
    grad_accum: int = 16,
    warmup_steps: int = 100,
    max_grad_norm: float = 1.0,
    save_every: int = 500,
    log_every: int = 10,
    device: torch.device = torch.device("cpu"),
    resume_from: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_path_in_repo: str = "training/sft_latest.pt",
    hf_token: Optional[str] = None,
):
    raw = model.module if isinstance(model, nn.DataParallel) else model
    raw.enable_gradient_checkpointing()
    model.to(device)

    # fp16 AMP: halves activation + intermediate tensor memory on GPU.
    # Without this the 1.65B model barely fits on T4 (14.56 GB) — weights+grads
    # alone are ~13.2 GB in float32, leaving almost nothing for activations.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        free, total = torch.cuda.mem_get_info(device)
        print(f"[AMP] fp16 enabled. GPU free: {free/1e9:.1f}/{total/1e9:.1f} GB", flush=True)

    optimizer = Adafactor(
        model.parameters(),
        lr=lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )

    global_step = 0
    best_loss = float("inf")

    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        global_step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", best_loss)
        print(f"[RESUME] Loaded from {resume_from} (step={global_step})")

    total_steps = len(dataloader) * epochs // grad_accum

    def get_lr(step):
        if step < warmup_steps:
            return lr * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad()
    accum_loss = 0.0
    accum_steps_done = 0

    print(f"[SFT] {total_steps} optimizer steps, {epochs} epochs, lr={lr}, grad_accum={grad_accum}")

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits, aux_loss = model(input_ids)
                # DataParallel gathers per-device MoE aux as a vector — reduce to scalar
                if isinstance(aux_loss, torch.Tensor):
                    aux_loss = aux_loss.mean()
                B, L, V = logits.shape
                loss_ce = F.cross_entropy(logits.view(B * L, V), labels.view(B * L), ignore_index=-100)
                loss = loss_ce + 0.01 * aux_loss
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
            loss = loss / grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item() * grad_accum
            accum_steps_done += 1

            if accum_steps_done >= grad_accum:
                current_lr = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1
                avg_loss = accum_loss / accum_steps_done
                accum_loss = 0.0
                accum_steps_done = 0

                if global_step % log_every == 0:
                    print(
                        f"epoch={epoch+1} step={global_step:05d}/{total_steps} "
                        f"loss={avg_loss:.4f} lr={current_lr:.2e}",
                        flush=True,
                    )

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    _save(
                        model, optimizer, model_cfg, global_step, best_loss, output_dir, "best",
                        hf_repo=hf_repo, hf_path_in_repo=hf_path_in_repo, hf_token=hf_token,
                    )

                if global_step % save_every == 0:
                    _save(
                        model, optimizer, model_cfg, global_step, best_loss, output_dir, f"step_{global_step:05d}",
                        hf_repo=hf_repo, hf_path_in_repo=hf_path_in_repo, hf_token=hf_token,
                    )

    _save(
        model, optimizer, model_cfg, global_step, best_loss, output_dir, "final",
        hf_repo=hf_repo, hf_path_in_repo=hf_path_in_repo, hf_token=hf_token,
    )
    print(f"[SFT] Done. Best loss: {best_loss:.4f}")


def _save(
    model,
    optimizer,
    model_cfg,
    step,
    best_loss,
    output_dir,
    tag,
    hf_repo: Optional[str] = None,
    hf_path_in_repo: str = "training/sft_latest.pt",
    hf_token: Optional[str] = None,
):
    path = output_dir / f"sft_{tag}.pt"
    raw = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "step": step,
        "best_loss": best_loss,
        "model_state": raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model_cfg.__dict__,
    }, path)
    print(f"[SAVE] {path}", flush=True)

    if hf_repo and hf_path_in_repo:
        from ssm.hf_checkpoint_sync import upload_checkpoint

        upload_checkpoint(path, hf_repo, hf_path_in_repo, hf_token)
        # When SFT completes, also push a persistent "final" marker so a new
        # Kaggle session can detect SFT is done and skip re-running it.
        if tag == "final":
            upload_checkpoint(path, hf_repo, "training/sft_final.pt", hf_token)

    # Keep only last 3 step checkpoints
    ckpts = sorted(output_dir.glob("sft_step_*.pt"), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-3]:
        old.unlink()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    # ── Single-instance lock ──────────────────────────────────────────────────
    # Kaggle GPU T4 x2 spawns two parallel papermill workers that both reach
    # this subprocess.  Only one should actually run on GPU 0; the other exits
    # cleanly so it doesn't compete for VRAM.
    import fcntl
    _lock_path = "/tmp/ssm_sft_train.lock"
    _lock_fh = open(_lock_path, "w")
    try:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("[SFT] Another worker already holds the GPU — this worker exits.", flush=True)
        return  # exit with code 0, not 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="data/reasoning_traces.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/sft")
    parser.add_argument("--model-size", default="700m", choices=["700m", "3b"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Load model weights only (e.g. pretrain_checkpoint.pt), fresh optimizer — paper §4 after Phase 1 pretrain",
    )
    parser.add_argument("--device", default=None, help="Device: cpu, cuda, cuda:0, etc.")
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help=(
            "Use nn.DataParallel when multiple CUDA devices are visible. "
            "Off by default: each GPU holds a full model copy and gradients reduce on GPU 0, "
            "which often causes OOM on 2×T4-class GPUs for this model size."
        ),
    )
    parser.add_argument(
        "--hf-repo",
        default=os.environ.get("HF_MODEL_REPO"),
        help=(
            "Hugging Face model repo id (your weights repo, not GitHub). "
            "Uploads each save to --hf-sft-path. Defaults to HF_MODEL_REPO env."
        ),
    )
    parser.add_argument(
        "--hf-sft-path",
        default="training/sft_latest.pt",
        help="Path inside the model repo for the latest SFT checkpoint (overwritten each save).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hub token (defaults to HF_TOKEN / HUGGINGFACE_HUB_TOKEN).",
    )
    args = parser.parse_args()

    if args.resume and args.init_checkpoint:
        print("[ERROR] Use only one of --resume or --init-checkpoint", file=sys.stderr)
        sys.exit(1)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_cfg = ModelConfig700M() if args.model_size == "700m" else ModelConfig3B()
    model = CodingSSM(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model params: {n_params:,} ({n_params/1e9:.2f}B)")

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[INIT] Loaded weights from {args.init_checkpoint} "
            f"(missing_keys={len(missing)}, unexpected_keys={len(unexpected)})",
            flush=True,
        )

    # Multi-GPU: DataParallel is opt-in — it replicates the full model per GPU and
    # reduces grads on device 0, which spikes memory during backward (common OOM on 2×T4).
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    batch_size = args.batch_size
    if n_gpus > 1 and args.data_parallel:
        print(f"[INFO] Using {n_gpus} GPUs via DataParallel")
        model = nn.DataParallel(model)
        batch_size = args.batch_size * n_gpus
    elif n_gpus > 1:
        print(
            f"[INFO] {n_gpus} CUDA devices visible; training on a single GPU "
            f"({device}). Pass --data-parallel to use DataParallel (needs headroom).",
            flush=True,
        )
    print(f"[INFO] Effective batch size: {batch_size} (x{args.grad_accum} grad_accum = {batch_size * args.grad_accum})")

    ds = ReasoningTraceDataset(args.traces, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    hf_tok = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    train(
        model=model,
        dataloader=loader,
        model_cfg=model_cfg,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        lr=args.lr,
        grad_accum=args.grad_accum,
        save_every=args.save_every,
        device=device,
        resume_from=args.resume,
        hf_repo=args.hf_repo,
        hf_path_in_repo=args.hf_sft_path,
        hf_token=hf_tok,
    )  # --init-checkpoint only loads weights; does not restore optimizer step


if __name__ == "__main__":
    main()
