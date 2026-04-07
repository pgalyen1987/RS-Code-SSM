"""
Phase 1 pretraining on multi-language code from The Stack / GitHub Code.

Streams data directly from HuggingFace — no local data required.
Supports resuming across multiple Kaggle sessions.

Languages: Python, C++, JavaScript, TypeScript, Java, Go, Rust, Kotlin
           (round-robin interleaving for balanced representation)

Usage:
    python -m train.pretrain \
        --output-dir checkpoints/pretrain \
        --model-size 700m \
        --max-tokens 2_000_000_000 \
        --seq-len 1024 \
        --batch-size 1 \
        --grad-accum 16 \
        --lr 1e-3 \
        --save-every 1000 \
        --device cuda
"""

import argparse
import math
import os
import time
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.optimization import Adafactor

from arch import CodingSSM
from arch.config import ModelConfig700M, ModelConfig3B


# ─── Data streaming ──────────────────────────────────────────────────────────

# Languages to train on — round-robin interleaved for balanced representation.
# The Stack data_dir names and their comment markers.
LANGUAGES = [
    ("python",     "data/python",      "# Python\n"),
    ("kotlin",     "data/kotlin",      "// Kotlin\n"),
    ("cpp",        "data/c++",         "// C++\n"),
    ("javascript", "data/javascript",  "// JavaScript\n"),
    ("typescript", "data/typescript",  "// TypeScript\n"),
    ("java",       "data/java",        "// Java\n"),
    ("go",         "data/go",          "// Go\n"),
    ("rust",       "data/rust",        "// Rust\n"),
]


def _iter_the_stack_lang(tokenizer, lang_name: str, data_dir: str, marker: str) -> Iterator[list[int]]:
    """Stream one language from bigcode/the-stack with a language marker prefix."""
    from datasets import load_dataset
    ds = load_dataset(
        "bigcode/the-stack",
        data_dir=data_dir,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    for sample in ds:
        content = sample.get("content", "") or ""
        if not content.strip():
            continue
        ids = tokenizer.encode(content, add_special_tokens=False)
        if ids:
            yield marker_ids + ids


def _iter_github_code_lang(tokenizer, lang_name: str, marker: str) -> Iterator[list[int]]:
    """Fallback: stream one language from codeparrot/github-code."""
    from datasets import load_dataset
    # Map our names to codeparrot language strings
    lang_map = {
        "python": "Python", "kotlin": "Kotlin", "cpp": "C++",
        "javascript": "JavaScript", "typescript": "TypeScript",
        "java": "Java", "go": "Go", "rust": "Rust",
    }
    gh_lang = lang_map.get(lang_name, lang_name)
    ds = load_dataset(
        "codeparrot/github-code",
        streaming=True, split="train", trust_remote_code=True,
        filters=[("language", gh_lang)],
    )
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    for sample in ds:
        content = sample.get("code", "") or ""
        if not content.strip():
            continue
        ids = tokenizer.encode(content, add_special_tokens=False)
        if ids:
            yield marker_ids + ids


def _make_lang_stream(tokenizer, lang_name: str, data_dir: str, marker: str) -> Iterator[list[int]]:
    """Try the-stack for one language; fall back to github-code."""
    try:
        gen = _iter_the_stack_lang(tokenizer, lang_name, data_dir, marker)
        first = next(gen)   # peek to confirm
        yield first
        yield from gen
    except Exception as e:
        print(f"[DATA] the-stack/{lang_name} failed ({e}), trying github-code", flush=True)
        try:
            yield from _iter_github_code_lang(tokenizer, lang_name, marker)
        except Exception as e2:
            print(f"[DATA] github-code/{lang_name} failed ({e2}), skipping", flush=True)


def make_token_stream(tokenizer) -> Iterator[list[int]]:
    """
    Round-robin interleave all languages so the model sees balanced multi-language data.
    Kotlin is listed first to ensure it gets early exposure.
    """
    print(f"[DATA] Streaming {len(LANGUAGES)} languages: {[l[0] for l in LANGUAGES]}", flush=True)
    generators = [
        _make_lang_stream(tokenizer, name, data_dir, marker)
        for name, data_dir, marker in LANGUAGES
    ]
    # Round-robin: advance each generator one sample at a time
    active = list(range(len(generators)))
    while active:
        next_active = []
        for i in active:
            try:
                yield next(generators[i])
                next_active.append(i)
            except StopIteration:
                print(f"[DATA] {LANGUAGES[i][0]} stream exhausted", flush=True)
        active = next_active


def chunk_stream(token_stream: Iterator[list[int]], seq_len: int) -> Iterator[torch.Tensor]:
    """
    Pack a stream of token lists into fixed-length chunks of seq_len+1 tokens
    (the +1 is so we can shift to get input/target pairs).
    """
    buf: list[int] = []
    for ids in token_stream:
        buf.extend(ids)
        while len(buf) >= seq_len + 1:
            chunk = buf[: seq_len + 1]
            buf = buf[seq_len + 1 :]
            yield torch.tensor(chunk, dtype=torch.long)


def batched_chunks(
    chunk_iter: Iterator[torch.Tensor],
    batch_size: int,
    skip_chunks: int = 0,
) -> Iterator[torch.Tensor]:
    """
    Group chunks into batches of shape (batch_size, seq_len+1).
    If skip_chunks > 0, fast-forward past already-seen data (for resume).
    """
    skipped = 0
    batch: list[torch.Tensor] = []
    for chunk in chunk_iter:
        if skipped < skip_chunks:
            skipped += 1
            if skipped % 50_000 == 0:
                print(f"[RESUME] Fast-forwarding... {skipped}/{skip_chunks} chunks", flush=True)
            continue
        batch.append(chunk)
        if len(batch) == batch_size:
            yield torch.stack(batch, dim=0)
            batch = []


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _save_checkpoint(
    model,
    optimizer,
    model_cfg,
    step: int,
    tokens_seen: int,
    output_dir: Path,
    tag: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"pretrain_{tag}.pt"
    torch.save({
        "step": step,
        "tokens_seen": tokens_seen,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model_cfg.__dict__,
    }, path)
    print(f"[SAVE] {path}  (tokens_seen={tokens_seen:,})", flush=True)

    # Keep only last 3 step checkpoints (not "final")
    ckpts = sorted(output_dir.glob("pretrain_step_*.pt"), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-3]:
        old.unlink()

    return path


def _upload_to_hf(local_path: Path, hf_token: str):
    """Upload checkpoint to HF Hub if token is available."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        repo_id = "pgalyen1987/RS-Code-SSM-1.6B"
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo="pretrain_checkpoint.pt",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"[HF] Uploaded {local_path.name} → {repo_id}/pretrain_checkpoint.pt", flush=True)
    except Exception as e:
        print(f"[HF] Upload failed: {e}", flush=True)


# ─── Training loop ────────────────────────────────────────────────────────────

def train(
    model: CodingSSM,
    model_cfg,
    tokenizer,
    output_dir: Path,
    max_tokens: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seq_len: int,
    save_every: int,
    device: torch.device,
    resume_from: Optional[str],
    hf_token: str,
):
    model.to(device)
    model.enable_gradient_checkpointing()

    optimizer = Adafactor(
        model.parameters(),
        lr=lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )

    step = 0
    tokens_seen = 0

    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        step = ckpt.get("step", 0)
        tokens_seen = ckpt.get("tokens_seen", 0)
        print(f"[RESUME] Loaded from {resume_from} (step={step}, tokens_seen={tokens_seen:,})", flush=True)

    # Warmup: 1000 optimizer steps
    warmup_steps = 1000

    def get_lr(s: int) -> float:
        if s < warmup_steps:
            return lr * max(s, 1) / warmup_steps
        # Cosine decay to lr/10
        total_opt_steps = max_tokens // (seq_len * batch_size * grad_accum)
        progress = (s - warmup_steps) / max(total_opt_steps - warmup_steps, 1)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    # How many chunks to skip on resume
    tokens_per_chunk = seq_len + 1
    chunks_seen = tokens_seen // tokens_per_chunk

    print(f"[PRETRAIN] max_tokens={max_tokens:,}  batch={batch_size}  grad_accum={grad_accum}  lr={lr}  seq_len={seq_len}", flush=True)
    print(f"[PRETRAIN] Starting from step={step}, tokens_seen={tokens_seen:,}", flush=True)

    token_gen = make_token_stream(tokenizer)
    chunks = chunk_stream(token_gen, seq_len)
    batches = batched_chunks(chunks, batch_size, skip_chunks=chunks_seen * batch_size)

    optimizer.zero_grad()
    accum_loss = 0.0
    accum_count = 0
    t0 = time.time()
    tokens_at_last_log = tokens_seen

    for batch in batches:
        if tokens_seen >= max_tokens:
            break

        batch = batch.to(device)            # (B, seq_len+1)
        input_ids = batch[:, :-1]           # (B, seq_len)
        targets    = batch[:, 1:]           # (B, seq_len)

        logits, aux_loss = model(input_ids)
        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.mean()
        B, L, V = logits.shape
        loss_ce = F.cross_entropy(logits.reshape(B * L, V), targets.reshape(B * L))
        loss = loss_ce + 0.01 * aux_loss
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()
        loss = loss / grad_accum
        loss.backward()

        tokens_this_batch = B * L
        tokens_seen += tokens_this_batch
        accum_loss += loss.item() * grad_accum
        accum_count += 1

        if accum_count >= grad_accum:
            current_lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            avg_loss = accum_loss / accum_count
            accum_loss = 0.0
            accum_count = 0

            if step % 100 == 0:
                elapsed = time.time() - t0
                toks_delta = tokens_seen - tokens_at_last_log
                tok_per_sec = toks_delta / max(elapsed, 1e-6)
                pct = 100 * tokens_seen / max_tokens
                print(
                    f"step={step:06d}  loss={avg_loss:.4f}  lr={current_lr:.2e}"
                    f"  tokens={tokens_seen/1e6:.1f}M/{max_tokens/1e9:.1f}B ({pct:.1f}%)"
                    f"  tok/s={tok_per_sec:.0f}",
                    flush=True,
                )
                t0 = time.time()
                tokens_at_last_log = tokens_seen

            if step % save_every == 0:
                ckpt_path = _save_checkpoint(
                    model, optimizer, model_cfg, step, tokens_seen,
                    output_dir, f"step_{step:06d}",
                )
                if hf_token:
                    _upload_to_hf(ckpt_path, hf_token)

    # Final save
    ckpt_path = _save_checkpoint(
        model, optimizer, model_cfg, step, tokens_seen,
        output_dir, "final",
    )
    if hf_token:
        _upload_to_hf(ckpt_path, hf_token)

    print(f"[PRETRAIN] Done. Total tokens: {tokens_seen:,}  Steps: {step}", flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CodingSSM Phase 1 pretraining on Python code")
    parser.add_argument("--output-dir",  default="checkpoints/pretrain")
    parser.add_argument("--model-size",  default="700m", choices=["700m", "3b"])
    parser.add_argument("--max-tokens",  type=int, default=2_000_000_000)
    parser.add_argument("--batch-size",  type=int, default=1)
    parser.add_argument("--grad-accum",  type=int, default=16)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--seq-len",     type=int, default=1024)
    parser.add_argument("--save-every",  type=int, default=1000)
    parser.add_argument("--resume",      default=None, help="Path to checkpoint to resume from (or empty string)")
    parser.add_argument("--device",      default=None, help="Device: cpu, cuda, cuda:0, etc.")
    args = parser.parse_args()

    # Treat empty string resume as None
    resume = args.resume if args.resume else None

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
    print(f"[INFO] Model: {args.model_size}  params={n_params:,} ({n_params/1e9:.2f}B)", flush=True)

    hf_token = os.environ.get("HF_TOKEN", "")

    train(
        model=model,
        model_cfg=model_cfg,
        tokenizer=tokenizer,
        output_dir=Path(args.output_dir),
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        seq_len=args.seq_len,
        save_every=args.save_every,
        device=device,
        resume_from=resume,
        hf_token=hf_token,
    )


if __name__ == "__main__":
    main()
