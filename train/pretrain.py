"""
Phase 1 pretraining on multi-language code from The Stack / GitHub Code.

Streams data directly from HuggingFace — no local data required.
Supports resuming across multiple Kaggle sessions (default --save-every 5 so short runs still checkpoint).

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
        --save-every 5 \
        --device cuda
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.optimization import Adafactor

from arch import CodingSSM
from arch.config import ModelConfig700M, ModelConfig3B


# ─── Data streaming ──────────────────────────────────────────────────────────
#
# Data source priority (all public Parquet, no loading scripts, no gating):
#
#  Raw code (~80% of tokens):
#   1. bigcode/the-stack-smol  — 10B token deduplicated subset, openly accessible
#   2. bigcode/starcoderdata   — StarCoder training data (per-language configs)
#   3. HuggingFaceTB/smollm-corpus (cosmopedia-v2) — has inline `text` (python-edu has only blob_id; needs S3)
#
#  Novel problems (~20% of tokens, 1 per 4 code samples):
#   - deepmind/code_contests   — competitive programming + solutions
#   - greengerong/leetcode     — LeetCode problems + Python solutions
#   - iamtarun/python_code_instructions_18k_alpaca — instruction-style problems
#   - TokenBender/code_instructions_122k_alpaca_style — broader coverage

# (lang_name, stack_smol_name, starcoder_name, marker)
def _encode(tokenizer, text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)


# Language markers injected by Magicoder stream
_LANG_MARKERS = {
    "Python": "# Python\n", "C++": "// C++\n", "JavaScript": "// JavaScript\n",
    "TypeScript": "// TypeScript\n", "Java": "// Java\n", "Go": "// Go\n",
    "Rust": "// Rust\n", "Kotlin": "// Kotlin\n",
}

# Human-readable source list for startup log
CODE_SOURCES = [
    "HuggingFaceTB/smollm-corpus (cosmopedia-v2, text)",
    "ise-uiuc/Magicoder-OSS-Instruct-75K (all languages)",
    "theblackcat102/evol-codealpaca-v1 (fallback)",
]

NOVEL_PROBLEM_DATASETS = [
    ("deepmind/code_contests", "train"),
    ("greengerong/leetcode", "train"),
    ("iamtarun/python_code_instructions_18k_alpaca", "train"),
    ("TokenBender/code_instructions_122k_alpaca_style", "train"),
]


def _iter_smollm_cosmopedia(tokenizer) -> Iterator[list[int]]:
    """
    SmolLM cosmopedia-v2: synthetic educational text with an inline `text` field.

    Do NOT use config `python-edu` here: it only has blob_id/repo metadata; actual code
    must be fetched from Software Heritage S3 (see dataset README), so streaming yields
    no usable text without a separate download step.
    """
    from datasets import load_dataset
    marker_ids = tokenizer.encode("# SmolLM cosmopedia-v2\n", add_special_tokens=False)
    ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
    for sample in ds:
        content = sample.get("text", "") or ""
        ids = _encode(tokenizer, content)
        if ids:
            yield marker_ids + ids


def _iter_magicoder_all(tokenizer) -> Iterator[list[int]]:
    """All languages mixed — no per-language scan. Starts instantly."""
    from datasets import load_dataset
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True)
    for item in ds:
        lang = item.get("lang", "")
        problem = item.get("problem", "") or ""
        solution = item.get("solution", "") or ""
        if not solution.strip():
            continue
        marker = _LANG_MARKERS.get(lang, f"// {lang}\n")
        text = f"{marker}# Problem:\n{problem}\n\n# Solution:\n{solution}\n" \
               if problem.strip() else f"{marker}{solution}\n"
        ids = _encode(tokenizer, text)
        if ids:
            yield ids


def _iter_evol_code_all(tokenizer) -> Iterator[list[int]]:
    """Mixed-language fallback: evol-codealpaca-v1 (~110K examples, open)."""
    from datasets import load_dataset
    ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train", streaming=True)
    for item in ds:
        instruction = item.get("instruction", "") or ""
        output = item.get("output", "") or ""
        if not output.strip():
            continue
        ids = _encode(tokenizer, f"{instruction}\n{output}\n")
        if ids:
            yield ids


def _iter_novel_problems(tokenizer) -> Iterator[list[int]]:
    """
    Stream novel problem datasets interleaved into pretraining.
    All use standard Parquet (no loading scripts, no gating).
    """
    from datasets import load_dataset
    import json as _json

    sources = [
        # Competitive programming: full problem statement + accepted solutions
        ("deepmind/code_contests", "train"),
        # LeetCode problems + Python solutions
        ("greengerong/leetcode", "train"),
        # Instruction-style coding problems (18K)
        ("iamtarun/python_code_instructions_18k_alpaca", "train"),
        # Broader instruction coverage (122K)
        ("TokenBender/code_instructions_122k_alpaca_style", "train"),
    ]

    for dataset_id, split in sources:
        try:
            ds = load_dataset(dataset_id, split=split, streaming=True)
            for item in ds:
                text = ""
                if dataset_id == "deepmind/code_contests":
                    problem = item.get("description", "") or ""
                    solutions = item.get("solutions", {})
                    py_sols = solutions.get("solution", []) if isinstance(solutions, dict) else []
                    if not problem.strip():
                        continue
                    text = f"# Competitive Programming Problem\n\"\"\"\n{problem}\n\"\"\"\n"
                    if py_sols:
                        text += f"\n# Solution:\n{py_sols[0]}\n"

                elif dataset_id == "greengerong/leetcode":
                    title = item.get("title", "") or ""
                    content = item.get("content", "") or ""
                    solution = item.get("python", "") or ""
                    if not content.strip():
                        continue
                    text = f"# LeetCode: {title}\n\"\"\"\n{content}\n\"\"\"\n"
                    if solution.strip():
                        text += f"\n# Python Solution:\n{solution}\n"

                elif dataset_id in (
                    "iamtarun/python_code_instructions_18k_alpaca",
                    "TokenBender/code_instructions_122k_alpaca_style",
                ):
                    instruction = item.get("instruction", "") or ""
                    output = item.get("output", "") or ""
                    if not instruction.strip():
                        continue
                    text = f"# Problem: {instruction}\n\n# Solution:\n{output}\n"

                ids = _encode(tokenizer, text)
                if ids:
                    yield ids

        except Exception as e:
            print(f"[DATA] Novel problems {dataset_id} failed: {e}", flush=True)


NOVEL_PROBLEM_DATASETS = [
    ("deepmind/code_contests", "train"),
    ("greengerong/leetcode", "train"),
    ("iamtarun/python_code_instructions_18k_alpaca", "train"),
    ("TokenBender/code_instructions_122k_alpaca_style", "train"),
]


def make_token_stream(tokenizer) -> Iterator[list[int]]:
    """
    Infinite cycling mixed-language code stream + novel problems.

    All three sources cycle (restart from scratch when exhausted) so
    training never stalls regardless of dataset size vs token target:
      1. smollm-corpus cosmopedia-v2 — synthetic educational text (inline `text`)
      2. Magicoder-OSS-Instruct-75K — all languages mixed, real GitHub code
      3. evol-codealpaca-v1         — mixed-lang fallback

    Every 4 code samples → 1 problem sample (competitive / LeetCode).
    """
    print(f"[DATA] Code sources (cycling): {CODE_SOURCES}", flush=True)
    print(f"[DATA] Problem datasets: {[d[0] for d in NOVEL_PROBLEM_DATASETS]}", flush=True)

    # Factory functions so we can restart each source on exhaustion
    source_factories = [
        ("smollm-cosmopedia-v2", lambda: _iter_smollm_cosmopedia(tokenizer)),
        ("magicoder-all", lambda: _iter_magicoder_all(tokenizer)),
        ("evol-codealpaca", lambda: _iter_evol_code_all(tokenizer)),
    ]

    def _cycling(label, factory):
        """Yield from source indefinitely, restarting when exhausted."""
        cycle = 0
        while True:
            count = 0
            try:
                for item in factory():
                    yield item
                    count += 1
                print(f"[DATA] {label} cycle {cycle} done ({count} docs), restarting", flush=True)
            except Exception as e:
                print(f"[DATA] {label} error at cycle {cycle} doc {count}: {e}", flush=True)
            cycle += 1

    # Round-robin across all cycling sources — never exhausted
    gens = [(label, _cycling(label, fn)) for label, fn in source_factories]
    problem_gen = _cycling("novel-problems", lambda: _iter_novel_problems(tokenizer))
    code_count = 0
    n = len(gens)
    idx = 0

    while True:
        label, gen = gens[idx % n]
        idx += 1
        try:
            yield next(gen)
            code_count += 1
            if code_count % 4 == 0:
                yield next(problem_gen)
        except StopIteration:
            # Should never happen with _cycling, but just in case
            print(f"[DATA] {label} unexpectedly stopped — this is a bug", flush=True)
            break


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


def _upload_to_hf(local_path: Path, hf_token: str, hf_repo: str = "pgalyen1987/RS-Code-SSM-1.6B"):
    """Upload checkpoint to HF Hub if token is available."""
    try:
        from huggingface_hub import HfApi

        sz_gb = local_path.stat().st_size / (1024**3)
        print(
            f"[HF] Starting upload: {local_path.name} ({sz_gb:.2f} GiB) → "
            f"{hf_repo}/training/pretrain_latest.pt (may take several minutes)…",
            flush=True,
        )
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo="training/pretrain_latest.pt",
            repo_id=hf_repo,
            repo_type="model",
        )
        print(f"[HF] Upload finished successfully.", flush=True)
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
    hf_repo: str = "pgalyen1987/RS-Code-SSM-1.6B",
    time_limit_minutes: int = 0,   # 0 = no limit; >0 = stop gracefully after N minutes
):
    model.to(device)
    model.enable_gradient_checkpointing()

    # fp16: halves weight+grad memory (3.3 GB each vs 6.6 GB) on T4.
    use_amp = device.type == "cuda"
    if use_amp:
        model = model.to(torch.float16)
        free, total = torch.cuda.mem_get_info(device)
        print(f"[AMP] fp16 model. GPU free: {free/1e9:.1f}/{total/1e9:.1f} GB", flush=True)

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
        # Load to CPU: the fp32 checkpoint is 6+ GB; loading to GPU alongside
        # the fp16 model + CUDA overhead would OOM the T4.
        ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        # Skip optimizer state to save ~3 GB GPU memory; pretrain Adafactor
        # re-warms quickly since LR is computed from tokens_seen.
        step = ckpt.get("step", 0)
        tokens_seen = ckpt.get("tokens_seen", 0)
        del ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[RESUME] Loaded model from {resume_from} (step={step}, tokens_seen={tokens_seen:,})", flush=True)

    deadline = time.time() + time_limit_minutes * 60 if time_limit_minutes > 0 else None

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
        if deadline and time.time() >= deadline:
            print(f"[PRETRAIN] Time limit reached ({time_limit_minutes} min). Saving and exiting.", flush=True)
            break

        batch = batch.to(device)            # (B, seq_len+1)
        input_ids = batch[:, :-1]           # (B, seq_len)
        targets    = batch[:, 1:]           # (B, seq_len)

        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with amp_ctx:
            logits, aux_loss = model(input_ids)
        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.mean()
        B, L, V = logits.shape
        loss_ce = F.cross_entropy(logits.float().reshape(B * L, V), targets.reshape(B * L))
        loss = loss_ce + 0.01 * aux_loss.float()
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

            # Save at step 1 (first optimizer step) so slow CPU/GPU runs checkpoint before save_every
            if step >= 1 and (step == 1 or step % save_every == 0):
                ckpt_path = _save_checkpoint(
                    model, optimizer, model_cfg, step, tokens_seen,
                    output_dir, f"step_{step:06d}",
                )
                if hf_token:
                    _upload_to_hf(ckpt_path, hf_token, hf_repo)

    # Final save
    ckpt_path = _save_checkpoint(
        model, optimizer, model_cfg, step, tokens_seen,
        output_dir, "final",
    )
    if hf_token:
        _upload_to_hf(ckpt_path, hf_token, hf_repo)

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
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save locally and upload to HF (if HF_TOKEN) every N optimizer steps. Default 5.",
    )
    parser.add_argument("--resume",      default=None, help="Path to checkpoint to resume from (or empty string)")
    parser.add_argument("--device",      default=None, help="Device: cpu, cuda, cuda:0, etc.")
    parser.add_argument("--hf-repo",     default="pgalyen1987/RS-Code-SSM-1.6B", help="HF model repo for checkpoint uploads")
    parser.add_argument("--time-limit-minutes", type=int, default=0, help="Stop gracefully after N minutes (0=no limit)")
    args = parser.parse_args()

    # Treat empty string resume as None
    resume = args.resume if args.resume else None

    if args.device:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(
                "[WARN] --device requests CUDA but this PyTorch build has no CUDA; using CPU.",
                flush=True,
            )
            device = torch.device("cpu")
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

    hf_token = (os.environ.get("HF_TOKEN") or "").strip()

    if hf_token:
        print(
            f"[HF] Token present — will upload checkpoints to {args.hf_repo}/training/pretrain_latest.pt",
            flush=True,
        )
    else:
        print(
            "[HF] HF_TOKEN not set — checkpoints are saved only under --output-dir (no Hub upload).",
            flush=True,
        )

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
        hf_repo=args.hf_repo,
        time_limit_minutes=args.time_limit_minutes,
    )


if __name__ == "__main__":
    main()
