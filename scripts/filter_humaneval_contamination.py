"""
Filter a ChatML training JSONL to remove rows that contain HumanEval test
problems. Used to build a clean SFT dataset that is publishable on HumanEval.

A row is considered contaminated if its chatml field contains either:
    1. a HumanEval function definition line  ``def <entry_point>(``  whose
       entry_point matches the official HumanEval test set, or
    2. the first 200 normalized characters of any HumanEval prompt.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _norm(s: str) -> str:
    return " ".join(s.split())


def load_humaneval_signatures(humaneval_path: Path):
    entry_points: set[str] = set()
    prompt_prefixes: set[str] = set()
    with humaneval_path.open() as f:
        for line in f:
            d = json.loads(line)
            ep = (d.get("entry_point") or "").strip()
            if ep:
                entry_points.add(ep)
            p = (d.get("prompt") or "").strip()
            if p:
                prompt_prefixes.add(_norm(p)[:200])
    return entry_points, prompt_prefixes


def is_contaminated(chatml: str, entry_points: set[str], prompt_prefixes: set[str]) -> bool:
    for ep in entry_points:
        if not ep:
            continue
        if re.search(rf"\bdef\s+{re.escape(ep)}\s*\(", chatml):
            return True
    cmn = _norm(chatml)
    return any(p in cmn for p in prompt_prefixes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--humaneval", default="data/grpo_humaneval.jsonl")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    eps, pps = load_humaneval_signatures(Path(args.humaneval))
    print(f"[filter] HumanEval signatures: entry_points={len(eps)} prompt_prefixes={len(pps)}")

    n_in = n_out = n_contam = 0
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            d = json.loads(line)
            cm = d.get("chatml", "")
            if is_contaminated(cm, eps, pps):
                n_contam += 1
                continue
            fout.write(json.dumps(d) + "\n")
            n_out += 1

    print(f"[filter] {args.input} -> {args.output}: in={n_in} kept={n_out} contam_dropped={n_contam}")


if __name__ == "__main__":
    main()
