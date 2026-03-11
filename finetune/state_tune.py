"""
RWKV State Priming — fast, CPU-feasible domain adaptation.

How it works:
  - RWKV is an RNN. Its hidden "state" summarizes everything seen so far.
  - Instead of gradient-based fine-tuning (requires training infrastructure),
    we run the model forward through high-quality coding examples and save
    the resulting hidden state as the starting point for all future queries.
  - The model then begins every conversation already "thinking about code".
  - Training time: minutes on CPU (one forward pass, no backprop).

This is the approach RWKV practitioners use for domain state initialization.

Usage:
    python finetune/state_tune.py \\
        --model $MODEL_DIR/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth \\
        --data  data/train.jsonl \\
        --examples 50 \\
        --out   $CONFIG_DIR/state.pt

After running, ssm cli automatically loads state.pt from CONFIG_DIR at startup.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_examples(jsonl_path: str, max_examples: int = 200) -> list[str]:
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(obj["text"])
            if len(examples) >= max_examples:
                break
    return examples


def pick_best(examples: list[str], n: int) -> list[str]:
    """
    Heuristic: prefer examples with actual code (contain ``` or def/function).
    Falls back to random sample.
    """
    code_heavy = [
        ex for ex in examples
        if "```" in ex or "def " in ex or "function " in ex or "class " in ex
    ]
    pool = code_heavy if len(code_heavy) >= n else examples
    return random.sample(pool, min(n, len(pool)))


def build_primer(examples: list[str], max_chars: int = 16000) -> str:
    """Concatenate examples into a single coding context block."""
    parts = [
        "The following are high-quality coding examples:\n"
    ]
    total = len(parts[0])
    for ex in examples:
        segment = ex.strip() + "\n\n"
        if total + len(segment) > max_chars:
            break
        parts.append(segment)
        total += len(segment)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Core: run model forward and capture state
# ---------------------------------------------------------------------------

def prime_state(model, pipeline, primer_text: str, chunk_size: int = 64) -> list:
    """
    Process primer_text token-by-token through the model, accumulating state.
    Returns the final hidden state after seeing all the coding context.
    """
    tokens = pipeline.encode(primer_text)
    total = len(tokens)
    print(f"  Primer: {total} tokens across {(total + chunk_size - 1) // chunk_size} chunks")

    state = None
    processed = 0

    with torch.no_grad():
        while processed < total:
            chunk = tokens[processed: processed + chunk_size]
            for tok in chunk:
                logits, state = model.forward([tok], state)
            processed += len(chunk)

            pct = 100 * processed // total
            bar = "#" * (pct // 5) + "." * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%  ({processed}/{total} tokens)", end="", flush=True)

    print()  # newline after progress bar
    return state


# ---------------------------------------------------------------------------
# Average states across multiple examples
# ---------------------------------------------------------------------------

def average_states(states: list[list]) -> list:
    """Element-wise mean of a list of RWKV states."""
    if not states:
        raise ValueError("No states to average")
    if len(states) == 1:
        return states[0]

    averaged = []
    for tensors in zip(*states):
        stacked = torch.stack(list(tensors), dim=0)
        averaged.append(stacked.mean(dim=0))
    return averaged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RWKV state priming — fast domain adaptation via context."
    )
    parser.add_argument("--model",    required=True, help="Path to RWKV .pth model file.")
    parser.add_argument("--strategy", default="cpu fp32", help="Inference strategy.")
    parser.add_argument("--data",     required=True, help="Path to train.jsonl.")
    parser.add_argument("--examples", type=int, default=50,
                        help="Number of coding examples to prime with.")
    parser.add_argument("--runs",     type=int, default=3,
                        help="Forward passes to average (more = smoother state).")
    parser.add_argument(
        "--out",
        default=os.path.join(os.environ.get("CONFIG_DIR") or str(Path.home() / ".ssm"), "state.pt"),
        help="Output state file path (default: CONFIG_DIR/state.pt)",
    )
    args = parser.parse_args()

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE

    model = RWKV(model=args.model, strategy=args.strategy)
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

    # Load and pick examples
    print(f"\nLoading examples from: {args.data}")
    all_examples = load_examples(args.data, max_examples=500)
    print(f"  {len(all_examples)} total examples available")

    states = []
    for run in range(1, args.runs + 1):
        selected = pick_best(all_examples, args.examples)
        primer = build_primer(selected)
        print(f"\n[Run {run}/{args.runs}] Priming with {len(selected)} examples "
              f"({len(primer):,} chars) ...")

        state = prime_state(model, pipeline, primer)
        states.append([s.clone() for s in state])
        print(f"  Run {run} complete. State tensors: {len(state)}")

    # Average all runs
    print(f"\nAveraging {len(states)} states ...")
    final_state = average_states(states)

    # Save
    torch.save(final_state, out_path)
    size_mb = out_path.stat().st_size / 1_000_000
    print(f"\nDone. State saved to: {out_path}  ({size_mb:.1f} MB)")
    print(
        "\nThe ssm CLI will automatically load this state on next startup.\n"
        "The model will now begin every coding session with domain-primed context."
    )


if __name__ == "__main__":
    main()
