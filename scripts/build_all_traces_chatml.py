"""
Merge every available trace source into a single ChatML JSONL file.

Reads from a configurable list of sources, normalizes each row to ChatML via
``scripts.normalize_traces_chatml.normalize``, deduplicates by hash of the
chatml content, and writes the result to ``--output``.

Output rows have the schema:
    {chatml: str, source: str, ...preserved keys}

Usage:
    python -m scripts.build_all_traces_chatml \
        --output data/all_traces_chatml.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Make sibling script importable when invoked from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_traces_chatml import normalize  # type: ignore

DEFAULT_SOURCES: list[str] = [
    "data/qwen_sft_solutions.jsonl",
    "data/qwen_sft_solutions_chatml.jsonl",
    "data/external_reasoning.jsonl",
    "data/mbpp_reasoning.jsonl",
    "data/sft_fix.jsonl",
    "data/claude_code_traces.jsonl",
    "data/claude_opus_traces.jsonl",
    "data/hermes_traces.jsonl",
]


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def merge(sources: list[Path], out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    stats = {
        "files": [],
        "total_in": 0,
        "total_out": 0,
        "total_skip": 0,
        "total_dup": 0,
    }
    with out_path.open("w") as fout:
        for src in sources:
            if not src.exists():
                print(f"[merge] SKIP missing: {src}", file=sys.stderr)
                continue
            n_in = n_out = n_skip = n_dup = 0
            with src.open() as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    n_in += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        n_skip += 1
                        continue
                    norm = normalize(rec)
                    if norm is None:
                        n_skip += 1
                        continue
                    h = _hash(norm["chatml"])
                    if h in seen:
                        n_dup += 1
                        continue
                    seen.add(h)
                    if "source" not in norm:
                        norm["source"] = src.stem
                    fout.write(json.dumps(norm) + "\n")
                    n_out += 1
            stats["files"].append(
                {"path": str(src), "in": n_in, "out": n_out, "skip": n_skip, "dup": n_dup}
            )
            stats["total_in"] += n_in
            stats["total_out"] += n_out
            stats["total_skip"] += n_skip
            stats["total_dup"] += n_dup
            print(
                f"[merge] {src}: in={n_in} kept={n_out} dup={n_dup} skipped={n_skip}",
                file=sys.stderr,
            )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge all trace sources into one ChatML JSONL.")
    parser.add_argument("--output", default="data/all_traces_chatml.jsonl")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=DEFAULT_SOURCES,
        help="Override default source list.",
    )
    args = parser.parse_args()

    sources = [Path(s) for s in args.sources]
    stats = merge(sources, Path(args.output))
    print(
        "[merge] DONE total_in={ti} kept={ko} dup={du} skipped={sk} -> {out}".format(
            ti=stats["total_in"], ko=stats["total_out"],
            du=stats["total_dup"], sk=stats["total_skip"],
            out=args.output,
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
