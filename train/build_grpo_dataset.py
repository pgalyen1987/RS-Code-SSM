"""
Build a GRPO-ready dataset from HumanEval + MBPP.

Each record has:
  prompt     - the problem description
  test_code  - the test suite (for execution reward)
  language   - "python"
  source     - "humaneval" | "mbpp"

Output: data/grpo_problems.jsonl

Usage:
    python -m train.build_grpo_dataset --output data/grpo_problems.jsonl
"""

import argparse
import json
from pathlib import Path


def load_humaneval():
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    records = []
    for ex in ds:
        prompt = ex["prompt"].strip()
        test_code = ex["test"] + f"\ncheck({ex['entry_point']})"
        records.append({
            "prompt": prompt,
            "test_code": test_code,
            "language": "python",
            "source": "humaneval",
            "entry_point": ex["entry_point"],
        })
    return records


def load_mbpp():
    from datasets import load_dataset
    ds = load_dataset("mbpp", split="test")
    records = []
    for ex in ds:
        prompt = ex["text"].strip()
        test_code = "\n".join(ex["test_list"])
        records.append({
            "prompt": prompt,
            "test_code": test_code,
            "language": "python",
            "source": "mbpp",
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/grpo_problems.jsonl")
    parser.add_argument("--no-humaneval", action="store_true")
    parser.add_argument("--no-mbpp", action="store_true")
    args = parser.parse_args()

    records = []
    if not args.no_humaneval:
        he = load_humaneval()
        records.extend(he)
        print(f"HumanEval: {len(he)} problems")
    if not args.no_mbpp:
        mb = load_mbpp()
        records.extend(mb)
        print(f"MBPP: {len(mb)} problems")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} problems to {out}")


if __name__ == "__main__":
    main()
