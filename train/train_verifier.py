"""
Train the verifier model on (problem, solution, label) triples.

Positive examples: solutions that passed tests (from reasoning traces)
Negative examples: solutions that failed tests (from GRPO rollouts or synthetic hard negatives)

Usage:
    python -m train.train_verifier \
        --traces data/all_traces.jsonl \
        --output-dir checkpoints/verifier \
        --epochs 5 \
        --lr 1e-4
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer


# ── Dataset ───────────────────────────────────────────────────────────────────

def _corrupt_solution(solution: str, rng: random.Random) -> str:
    """
    Generate a hard negative by lightly corrupting a correct solution.
    Strategies:
      1. Delete a random line (most common bugs are missing lines).
      2. Shuffle two random lines (logic reordering bugs).
      3. Replace return value with None.
    """
    lines = solution.splitlines()
    if not lines:
        return solution

    strategy = rng.choice(["delete", "shuffle", "return_none"])

    if strategy == "delete" and len(lines) > 1:
        idx = rng.randrange(len(lines))
        lines.pop(idx)

    elif strategy == "shuffle" and len(lines) > 1:
        i, j = rng.sample(range(len(lines)), 2)
        lines[i], lines[j] = lines[j], lines[i]

    elif strategy == "return_none":
        # Replace the first `return X` (where X is not None) with `return None`
        new_lines = []
        replaced = False
        for line in lines:
            stripped = line.lstrip()
            if not replaced and stripped.startswith("return ") and "None" not in stripped:
                indent = len(line) - len(stripped)
                line = " " * indent + "return None"
                replaced = True
            new_lines.append(line)
        lines = new_lines

    return "\n".join(lines)


def _format_input(prompt: str, solution: str) -> str:
    """Format a (problem, solution) pair as a chat-style string."""
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{solution}<|im_end|>"
    )


class VerifierDataset(Dataset):
    """
    Loads JSONL traces and creates (input_ids, label) pairs.

    Positive examples (label=1): solutions from traces where status is 'pass'
      or the trace has a 'solution' field.
    Negative examples (label=0): synthetic hard negatives created by corrupting
      positive solutions.

    Each positive solution generates exactly one hard negative to keep the
    dataset balanced.
    """

    def __init__(
        self,
        traces_path: str,
        tokenizer,
        max_length: int = 2048,
        neg_ratio: float = 1.0,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id or 0

        rng = random.Random(seed)
        self.samples = []  # list of (text, label)

        with open(traces_path) as f:
            raw_lines = [l.strip() for l in f if l.strip()]

        for line in raw_lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = (rec.get("prompt") or "").strip()
            solution = (rec.get("solution") or rec.get("response") or "").strip()

            if not prompt or not solution:
                continue

            # Positive example
            self.samples.append((_format_input(prompt, solution), 1.0))

            # Hard negative
            if rng.random() < neg_ratio:
                corrupted = _corrupt_solution(solution, rng)
                if corrupted != solution:  # only add if actually corrupted
                    self.samples.append((_format_input(prompt, corrupted), 0.0))

        # Shuffle
        rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)   # (L,)
        return input_ids, torch.tensor(label, dtype=torch.float32)


# ── Training loop ─────────────────────────────────────────────────────────────

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute accuracy, precision, recall from sigmoid scores and binary labels."""
    preds = (logits >= 0.5).float()
    correct = (preds == labels).float()
    acc = correct.mean().item()

    tp = ((preds == 1) & (labels == 1)).float().sum().item()
    fp = ((preds == 1) & (labels == 0)).float().sum().item()
    fn = ((preds == 0) & (labels == 1)).float().sum().item()

    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)

    return {"accuracy": acc, "precision": precision, "recall": recall}


def train(args):
    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[train_verifier] Device: {device}", flush=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Dataset
    print(f"[train_verifier] Loading traces from {args.traces} ...", flush=True)
    full_dataset = VerifierDataset(
        traces_path=args.traces,
        tokenizer=tokenizer,
        max_length=2048,
    )
    print(f"[train_verifier] Total samples: {len(full_dataset)}", flush=True)

    # Train / val split (90% / 10%)
    n_val = max(1, int(0.1 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"[train_verifier] Train: {n_train}  Val: {n_val}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Model
    from arch.verifier import VerifierModel, VerifierConfig100M
    cfg = VerifierConfig100M()
    model = VerifierModel(cfg).to(device)
    n_params = model.num_parameters()
    print(f"[train_verifier] VerifierModel params: {n_params:,} ({n_params/1e6:.1f}M)", flush=True)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.05)

    loss_fn = nn.BCELoss()

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_ckpt_path = output_dir / "verifier_best.pt"

    print(f"[train_verifier] Training for {args.epochs} epochs ...", flush=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            scores = model(input_ids)       # (B,)
            loss = loss_fn(scores, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / n_batches
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(
                    f"  epoch={epoch} step={batch_idx+1}/{len(train_loader)} "
                    f"loss={avg_loss:.4f} lr={lr_now:.2e} ({elapsed:.1f}s)",
                    flush=True,
                )

        avg_train_loss = total_loss / max(n_batches, 1)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                scores = model(input_ids)
                loss = loss_fn(scores, labels)
                val_loss += loss.item()
                all_scores.append(scores.cpu())
                all_labels.append(labels.cpu())

        avg_val_loss = val_loss / max(len(val_loader), 1)
        all_scores_t = torch.cat(all_scores)
        all_labels_t = torch.cat(all_labels)
        metrics = compute_metrics(all_scores_t, all_labels_t)

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
            f"rec={metrics['recall']:.4f}",
            flush=True,
        )

        # Save best checkpoint
        if metrics["accuracy"] > best_val_acc:
            best_val_acc = metrics["accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_accuracy": best_val_acc,
                    "config": cfg.__dict__,
                },
                best_ckpt_path,
            )
            print(f"  [SAVE] Best checkpoint -> {best_ckpt_path} (val_acc={best_val_acc:.4f})", flush=True)

        # Save epoch checkpoint
        epoch_path = output_dir / f"verifier_epoch{epoch:02d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_accuracy": metrics["accuracy"],
                "config": cfg.__dict__,
            },
            epoch_path,
        )

        # Prune old epoch checkpoints (keep last 2)
        epoch_ckpts = sorted(output_dir.glob("verifier_epoch*.pt"), key=lambda p: p.stat().st_mtime)
        for old in epoch_ckpts[:-2]:
            old.unlink(missing_ok=True)

    print(f"\n[train_verifier] Training complete. Best val_acc={best_val_acc:.4f}", flush=True)
    print(f"[train_verifier] Best checkpoint: {best_ckpt_path}", flush=True)

    # ── Optional HF upload ─────────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        _upload_to_hub(best_ckpt_path, hf_token)

    return best_ckpt_path


def _upload_to_hub(ckpt_path: Path, token: str):
    """Upload verifier checkpoint to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi  # type: ignore

        repo_id = "pgalyen1987/RS-Code-SSM"
        filename = "verifier.pt"

        api = HfApi(token=token)
        # Ensure the repo exists (create if needed)
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True)
        except Exception:
            pass

        print(f"[train_verifier] Uploading {ckpt_path} -> {repo_id}/{filename} ...", flush=True)
        api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo=filename,
            repo_id=repo_id,
            token=token,
        )
        print(f"[train_verifier] Upload complete: https://huggingface.co/{repo_id}", flush=True)
    except Exception as e:
        print(f"[train_verifier] HF upload failed: {e}", flush=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train CodingSSM verifier model")
    parser.add_argument(
        "--traces", default="data/all_traces.jsonl",
        help="Path to JSONL traces file (all_traces.jsonl or similar)",
    )
    parser.add_argument(
        "--output-dir", default="checkpoints/verifier",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--device", default=None,
        help="Device: cpu, cuda, cuda:0, etc. Auto-detects if omitted.",
    )
    args = parser.parse_args()

    # Validate traces file
    if not Path(args.traces).exists():
        print(f"[ERROR] Traces file not found: {args.traces}", file=sys.stderr)
        print("  Run: bash scripts/gen_traces.sh  to generate traces first.", file=sys.stderr)
        sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
