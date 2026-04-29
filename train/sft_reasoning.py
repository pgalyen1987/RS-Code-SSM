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
from arch.config import ModelConfig700M, ModelConfig3B, ModelConfigCPU


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

    # fp16 training: store weights + gradients in fp16 (3.3 GB each vs 6.6 GB).
    # GradScaler is NOT used — it requires fp32 gradients, but our model is fp16.
    # Adafactor keeps its own fp32 second moments internally, so it handles fp16
    # parameters correctly without loss scaling.
    # autocast is still used to keep activations in fp16 (saves activation memory).
    use_amp = device.type == "cuda"
    if use_amp:
        model = model.to(torch.float16)
        free, total = torch.cuda.mem_get_info(device)
        print(f"[AMP] fp16 model + autocast enabled. GPU free: {free/1e9:.1f}/{total/1e9:.1f} GB", flush=True)

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
        # Load to CPU first — the fp32 checkpoint is 6+ GB.  Loading directly
        # to GPU would put 6 GB fp32 tensors on a 15.6 GB T4 alongside the
        # fp16 model (3.3 GB) + gradients (3.3 GB) + CUDA overhead (~3.8 GB),
        # leaving no room for the optimizer step.
        ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        # Skip restoring optimizer state: Adafactor fp32 second moments are
        # ~3 GB on GPU and we're only at step 78/20954 — barely warmed up.
        # Fresh state costs one slightly noisier step; OOM costs the whole run.
        global_step = ckpt.get("step", 0)
        best_loss = ckpt.get("best_loss", best_loss)
        del ckpt
        torch.cuda.empty_cache()
        print(f"[RESUME] Loaded model weights from {resume_from} (step={global_step}, optimizer state reset)")

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
                # Cast to fp32 for numerical stability — cross_entropy in fp16 can NaN
                loss_ce = F.cross_entropy(logits.float().view(B * L, V), labels.view(B * L), ignore_index=-100)
                loss = loss_ce + 0.01 * aux_loss.float()
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
            loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item() * grad_accum
            accum_steps_done += 1

            if accum_steps_done >= grad_accum:
                current_lr = get_lr(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                avg_loss = accum_loss / accum_steps_done
                accum_loss = 0.0
                accum_steps_done = 0

                if global_step % log_every == 0:
                    if torch.cuda.is_available():
                        free, total = torch.cuda.mem_get_info(device)
                        print(
                            f"epoch={epoch+1} step={global_step:05d}/{total_steps} "
                            f"loss={avg_loss:.4f} lr={current_lr:.2e} "
                            f"gpu={free/1e9:.1f}/{total/1e9:.1f}GB free",
                            flush=True,
                        )
                    else:
                        print(
                            f"epoch={epoch+1} step={global_step:05d}/{total_steps} "
                            f"loss={avg_loss:.4f} lr={current_lr:.2e}",
                            flush=True,
                        )

                if global_step % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if avg_loss < best_loss:
                    best_loss = avg_loss

                # Save + HF upload only on save_every cadence — not on every best_loss
                # improvement. Early training loss drops every few steps, causing
                # constant 3 GB HF uploads that fill GPU memory.
                if global_step % save_every == 0:
                    _save(
                        model, optimizer, model_cfg, global_step, best_loss, output_dir, "latest",
                        hf_repo=hf_repo, hf_path_in_repo=hf_path_in_repo, hf_token=hf_token,
                    )
                    torch.cuda.empty_cache()

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
    # Always write to a single file — overwrite previous checkpoint.
    # No accumulation of sft_best/sft_step_*/sft_latest copies.
    path = output_dir / "sft_latest.pt"
    raw = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({
        "step": step,
        "best_loss": best_loss,
        "model_state": raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model_cfg.__dict__,
    }, path)
    print(f"[SAVE] {path}  step={step}  loss={best_loss:.4f}", flush=True)

    # Delete any leftover multi-copy files from previous runs
    for old in output_dir.glob("sft_step_*.pt"):
        old.unlink(missing_ok=True)
    for old in output_dir.glob("sft_best.pt"):
        old.unlink(missing_ok=True)

    if hf_repo and hf_token:
        from ssm.hf_checkpoint_sync import upload_checkpoint
        upload_checkpoint(path, hf_repo, hf_path_in_repo, hf_token)
        if tag == "final":
            upload_checkpoint(path, hf_repo, "training/sft_final.pt", hf_token)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    # ── Single-instance lock ──────────────────────────────────────────────────
    # Kaggle GPU T4 x2 launches two parallel papermill workers (separate /tmp,
    # but shared /kaggle/working).  Use atomic O_CREAT|O_EXCL on /kaggle/working
    # so only one worker trains; the other exits cleanly (code 0).
    import os as _os
    _kaggle_wd = "/kaggle/working"
    _lock_dir  = _kaggle_wd if _os.path.isdir(_kaggle_wd) else "/tmp"
    _lock_path = _os.path.join(_lock_dir, ".ssm_sft_train.lock")
    _lock_acquired = False
    try:
        # O_CREAT|O_EXCL is atomic: exactly one process creates the file
        _lfd = _os.open(_lock_path, _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY, 0o644)
        _os.write(_lfd, str(_os.getpid()).encode())
        _os.close(_lfd)
        _lock_acquired = True
        print(f"[SFT] Lock acquired: {_lock_path}", flush=True)
    except FileExistsError:
        print(f"[SFT] Another worker already training ({_lock_path} exists) — exiting.", flush=True)
        return  # exit code 0

    try:
        _main()
    finally:
        if _lock_acquired:
            try:
                _os.unlink(_lock_path)
            except Exception:
                pass


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="data/reasoning_traces.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/sft")
    parser.add_argument("--model-size", default="auto", choices=["auto", "cpu", "700m", "3b"])
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

    # Auto-select model size: use CPU-safe config when no GPU available
    model_size = args.model_size
    if model_size == "auto":
        model_size = "700m" if torch.cuda.is_available() else "cpu"
    if model_size == "cpu":
        model_cfg = ModelConfigCPU()
        print("[INFO] No GPU detected — using CPU-compatible 150M config (seq_len will be capped at 512)", flush=True)
    elif model_size == "700m":
        model_cfg = ModelConfig700M()
    else:
        model_cfg = ModelConfig3B()
    model = CodingSSM(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: {model_size}  params={n_params:,} ({n_params/1e9:.2f}B)", flush=True)

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
