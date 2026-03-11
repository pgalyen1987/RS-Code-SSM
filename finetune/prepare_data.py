"""
Prepare a coding fine-tuning dataset from local files or The Stack (HuggingFace).

Usage:
    python finetune/prepare_data.py --source local --dir /path/to/code --out data/train.jsonl
    python finetune/prepare_data.py --source stack --lang python --samples 5000 --out data/train.jsonl

Output format (one JSON object per line):
    {"text": "<full training example in RWKV World format>"}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# RWKV World instruction format
INSTRUCTION_TEMPLATE = "\nUser: {instruction}\n\nAssistant: {response}"

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".go", ".rs", ".c", ".cpp", ".h",
    ".java", ".rb", ".sh", ".sql",
}

IGNORE_DIRS = {"__pycache__", "node_modules", ".venv", "venv", ".git", "dist", "build"}


# ---------------------------------------------------------------------------
# Local source
# ---------------------------------------------------------------------------

def _extract_docstring(text: str) -> tuple[str, str]:
    """Try to extract the first docstring as the 'instruction' from a Python function."""
    m = re.search(r'def\s+\w+[^:]+:\s*"""(.*?)"""', text, re.DOTALL)
    if m:
        return m.group(1).strip(), text
    return "", text


def examples_from_directory(directory: str, min_lines: int = 5, max_lines: int = 150) -> list[str]:
    """
    Extract training examples from a local codebase.
    Each file becomes one or more instruction-response pairs.
    """
    dir_path = Path(directory).resolve()
    examples = []

    for file_path in dir_path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix not in CODE_EXTENSIONS:
            continue
        if any(part in IGNORE_DIRS for part in file_path.parts):
            continue

        try:
            text = file_path.read_text(errors="ignore").strip()
        except Exception:
            continue

        lines = text.splitlines()
        if len(lines) < min_lines:
            continue

        rel = str(file_path.relative_to(dir_path))

        # Strategy 1: whole-file completion
        # "Write the contents of <file>" -> actual file
        if len(lines) <= max_lines:
            instruction = f"Write the complete contents of {rel}."
            response = f"```{file_path.suffix.lstrip('.')}\n{text}\n```"
            examples.append(INSTRUCTION_TEMPLATE.format(
                instruction=instruction, response=response
            ))

        # Strategy 2: prefix -> suffix (fill-in-the-middle style)
        if len(lines) >= min_lines * 2:
            split = len(lines) // 2
            prefix = "\n".join(lines[:split])
            suffix = "\n".join(lines[split:])
            instruction = (
                f"Continue this {file_path.suffix.lstrip('.')} code from {rel}:\n\n"
                f"```{file_path.suffix.lstrip('.')}\n{prefix}\n```"
            )
            response = f"```{file_path.suffix.lstrip('.')}\n{suffix}\n```"
            examples.append(INSTRUCTION_TEMPLATE.format(
                instruction=instruction, response=response
            ))

    return examples


# ---------------------------------------------------------------------------
# HuggingFace The Stack (small subset)
# ---------------------------------------------------------------------------

def examples_from_stack(language: str, n_samples: int = 2000) -> list[str]:
    """
    Load coding instruction examples from sahil2801/CodeAlpaca-20k (public, no login).
    Falls back to iamtarun/python_code_instructions_18k_alpaca for Python.
    Each sample has: instruction, input, output.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets", file=sys.stderr)
        sys.exit(1)

    # Use CodeAlpaca-20k — public coding instruction dataset, no gating
    print("Loading sahil2801/CodeAlpaca-20k from HuggingFace...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)

    examples = []
    for i, sample in enumerate(ds):
        if i >= n_samples:
            break

        instruction = sample.get("instruction", "").strip()
        inp = sample.get("input", "").strip()
        output = sample.get("output", "").strip()

        if not instruction or not output:
            continue

        # Combine instruction + input if present
        full_instruction = instruction
        if inp:
            full_instruction = f"{instruction}\n\nInput:\n```\n{inp}\n```"

        examples.append(INSTRUCTION_TEMPLATE.format(
            instruction=full_instruction,
            response=output,
        ))

    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare RWKV fine-tuning data.")
    parser.add_argument("--source", choices=["local", "stack"], default="local")
    parser.add_argument("--dir", default=".", help="Local codebase directory (--source local)")
    parser.add_argument("--lang", default="python", help="Language for The Stack (--source stack)")
    parser.add_argument("--samples", type=int, default=2000, help="Max examples from The Stack")
    parser.add_argument("--out", default="data/train.jsonl", help="Output .jsonl file")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "local":
        print(f"Extracting examples from {args.dir} ...")
        examples = examples_from_directory(args.dir)
    else:
        print(f"Downloading {args.samples} {args.lang} samples from The Stack ...")
        examples = examples_from_stack(args.lang, args.samples)

    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps({"text": ex}) + "\n")

    print(f"Wrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
