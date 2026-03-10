#!/usr/bin/env python3
"""
Evaluate CodingSSM on HumanEval benchmark.

Measures pass@1, pass@4, pass@8, pass@16 using test-time compute.

Usage:
    source .venv/bin/activate
    python scripts/eval_humaneval.py
    python scripts/eval_humaneval.py --n-samples 8 --n-problems 50
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-samples", type=int, default=4, help="Samples per problem (pass@k)")
    parser.add_argument("--n-problems", type=int, default=164, help="HumanEval has 164 problems")
    parser.add_argument("--output", default="data/eval_humaneval.jsonl")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    print(f"[EVAL] HumanEval — pass@{args.n_samples} — {args.n_problems} problems")

    # Load HumanEval
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
        problems = list(ds)[:args.n_problems]
    except Exception as e:
        print(f"[ERROR] Could not load HumanEval: {e}")
        sys.exit(1)

    from ssm.test_time_compute import TTCInference
    ttc = TTCInference(
        checkpoint=args.checkpoint,
        n_samples=args.n_samples,
        temperature=args.temperature,
    )

    results = []
    n_pass = 0
    n_total = 0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as out:
        for i, prob in enumerate(problems):
            task_id = prob["task_id"]
            prompt = prob["prompt"].strip()
            test = prob.get("test", "")
            entry = prob.get("entry_point", "")

            # HumanEval test format needs the entry point
            test_code = test if test else ""

            print(f"\n[{i+1:03d}/{len(problems)}] {task_id}")
            result = ttc.solve(prompt, test_code=test_code, n_samples=args.n_samples)

            n_total += 1
            if result.passed:
                n_pass += 1

            status = "✓" if result.passed else ("?" if result.passed is None else "✗")
            print(f"  {status} attempt={result.attempt}/{result.n_samples} "
                  f"think={len(result.thinking)}c code={len(result.solution)}c")

            rec = {
                "task_id": task_id,
                "passed": result.passed,
                "attempt": result.attempt,
                "n_samples": result.n_samples,
                "solution": result.solution,
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()

            # Running accuracy
            if n_total % 10 == 0:
                acc = n_pass / n_total * 100
                print(f"\n  === Running pass@{args.n_samples}: {n_pass}/{n_total} = {acc:.1f}% ===\n")

    acc = n_pass / n_total * 100 if n_total else 0
    print(f"\n{'='*50}")
    print(f"FINAL pass@{args.n_samples}: {n_pass}/{n_total} = {acc:.1f}%")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
