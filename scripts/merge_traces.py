"""
Merge all trace sources into data/all_traces.jsonl.

Sources (in priority order — first seen wins on dedup):
  Epichat (dedup by eu_id):
    epichat_traces.jsonl          (top-level, newest/largest)
    data/epichat_traces.jsonl
    data/all_traces.jsonl         (old epichat subset)

  Reasoning (dedup by problem_id):
    reasoning_traces_r1.jsonl     (top-level, from Kaggle)
    data/reasoning_traces_r1.jsonl
    data/external_reasoning.jsonl, data/mbpp_reasoning.jsonl
    data/public_supplement.jsonl  (convert_external_traces supplement)
    data/kaggle_supplement.jsonl  (convert_external_traces kaggle)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssm.reasoning_trace_format import build_chatml_from_parts

EPICHAT_SOURCES = [
    ROOT / "epichat_traces.jsonl",
    ROOT / "data" / "epichat_traces.jsonl",
    ROOT / "data" / "all_traces.jsonl",
]

REASONING_SOURCES = [
    ROOT / "reasoning_traces_r1.jsonl",
    ROOT / "data" / "reasoning_traces_r1.jsonl",
    ROOT / "data" / "external_reasoning.jsonl",  # convert_external_traces humaneval
    ROOT / "data" / "mbpp_reasoning.jsonl",  # convert_external_traces mbpp
    ROOT / "data" / "public_supplement.jsonl",  # convert_external_traces supplement
    ROOT / "data" / "kaggle_supplement.jsonl",  # convert_external_traces kaggle
]

OUTPUT = ROOT / "data" / "all_traces.jsonl"


def build_reasoning_chatml(rec: dict) -> str:
    return build_chatml_from_parts(
        rec.get("prompt", ""),
        rec.get("thinking", ""),
        rec.get("solution", ""),
    )


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def merge():
    # ── Epichat traces (dedup by eu_id) ──────────────────────────────────────
    seen_eu = set()
    epichat_traces = []
    for src in EPICHAT_SOURCES:
        records = load_jsonl(src)
        added = 0
        for rec in records:
            eu_id = rec.get("eu_id")
            if eu_id and eu_id not in seen_eu:
                seen_eu.add(eu_id)
                epichat_traces.append(rec)
                added += 1
        print(f"  {src.name}: {len(records)} read, {added} new  (epichat total: {len(epichat_traces)})")

    # ── Reasoning traces (dedup by problem_id, add chatml) ───────────────────
    seen_pid = set()
    reasoning_traces = []
    for src in REASONING_SOURCES:
        records = load_jsonl(src)
        added = 0
        for rec in records:
            pid = rec.get("problem_id")
            if pid and pid not in seen_pid:
                seen_pid.add(pid)
                if "chatml" not in rec:
                    rec = dict(rec, chatml=build_reasoning_chatml(rec))
                reasoning_traces.append(rec)
                added += 1
        print(f"  {src.name}: {len(records)} read, {added} new  (reasoning total: {len(reasoning_traces)})")

    # ── Write output ──────────────────────────────────────────────────────────
    all_traces = epichat_traces + reasoning_traces
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for rec in all_traces:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(all_traces)} traces → {OUTPUT}")
    print(f"  epichat:   {len(epichat_traces)}")
    print(f"  reasoning: {len(reasoning_traces)}")


if __name__ == "__main__":
    merge()
