"""
Build the GRPO problem set: Python problems with executable tests.

GRPO's reward is "do the tests pass", so every problem needs runnable
`test_code`. Our SFT sources (Magicoder / CodeFeedback) have no tests, so we
use MBPP (Mostly Basic Python Problems), whose `test_list` is exactly a set of
asserts against an entry-point function.

Output JSONL (one per line):
    {"prompt": ..., "test_code": "assert ...\nassert ...", "entry_point": ...,
     "language": "python", "source": "mbpp"}

Usage:
    python scripts/gen_grpo_problems.py                         # full MBPP train
    python scripts/gen_grpo_problems.py --sample 32 \
        --output data/grpo_problems_small.jsonl                 # local smoke
"""
import argparse
import json
import re
from pathlib import Path

ENTRY_RE = re.compile(r"assert\s+(\w+)\s*\(")


def entry_point_from_tests(test_list: list[str]) -> str:
    for t in test_list:
        m = ENTRY_RE.search(t)
        if m:
            return m.group(1)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/grpo_problems.jsonl")
    ap.add_argument("--sample", type=int, default=None,
                    help="Limit to N problems (local smoke test)")
    ap.add_argument("--split", default="train",
                    help="MBPP split: train (374) / validation / test / prompt")
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"[mbpp] Loading google-research-datasets/mbpp ({args.split}) ...", flush=True)
    ds = load_dataset("google-research-datasets/mbpp", split=args.split)

    records = []
    skipped = 0
    for ex in ds:
        prompt = (ex.get("text") or "").strip()
        test_list = ex.get("test_list") or []
        setup = (ex.get("test_setup_code") or "").strip()
        if not prompt or not test_list:
            skipped += 1
            continue
        entry = entry_point_from_tests(test_list)
        test_code = "\n".join(([setup] if setup else []) + list(test_list))
        records.append({
            "prompt": f"{prompt}\nYour function must be named `{entry}`." if entry else prompt,
            "test_code": test_code,
            "entry_point": entry,
            "language": "python",
            "source": "mbpp",
        })
        if args.sample and len(records) >= args.sample:
            break

    out = Path(args.output)
    if not out.is_absolute():
        out = Path(__file__).parent.parent / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    has_entry = sum(1 for r in records if r["entry_point"])
    print(f"[DONE] {len(records)} problems → {out}  (skipped {skipped})", flush=True)
    print(f"[AUDIT] with entry_point: {has_entry}/{len(records)}", flush=True)
    if records:
        r = records[0]
        print(f"[SAMPLE] prompt={r['prompt'][:80]!r}", flush=True)
        print(f"[SAMPLE] test={r['test_code'][:120]!r}", flush=True)


if __name__ == "__main__":
    main()
