#!/usr/bin/env python3
"""
Convert public code / reasoning datasets into JSONL for train.sft_reasoning and merge_traces.

Target record shape (per line):
  - chatml (optional if prompt + thinking + solution present — we build chatml)
  - prompt, thinking, solution
  - problem_id (for dedup in merge_traces)
  - test_code (optional, for GRPO when using HumanEval-style rows)

Recommended Hugging Face datasets (install: pip install datasets):
  - openai_humaneval — 164 canonical solutions + tests (great for GRPO test harness)
  - mbpp — 500 Python tasks with reference code and tests
  - ise-uiuc/Magicoder-OSS-Instruct-75K — 75K diverse coding problems
  - nickrosh/Evol-Instruct-Code-80k-v1 — 80K evolved instruction/code pairs
  - codeparrot/apps — 10K Python problems with test cases (great for GRPO)
  - bigcode/humanevalpack — multilingual HumanEval (Python/JS/Java/Go/C++/Rust)
  - m-a-p/CodeFeedback-Filtered — instruction/output style

Kaggle (API; needs ~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY):
  python scripts/convert_external_traces.py kaggle -d owner/dataset-slug -o data/kaggle_supplement.jsonl \\
      --prompt-field instruction --solution-field response
  Auto-discovers *.jsonl / *.csv / *.json under the downloaded archive unless --data-file narrows it.

Examples:
  python scripts/convert_external_traces.py humaneval --output data/external_reasoning.jsonl
  python scripts/convert_external_traces.py mbpp --output data/mbpp_reasoning.jsonl
  python scripts/convert_external_traces.py apps --output data/apps_traces.jsonl
  python scripts/convert_external_traces.py humanevalpack --output data/humanevalpack.jsonl
  python scripts/convert_external_traces.py supplement --output data/public_supplement.jsonl
  python scripts/convert_external_traces.py jsonl -i weird.jsonl --prompt-field instruction \\
      --solution-field output --thinking-field reasoning --output data/external_reasoning.jsonl

supplement pulls (HF): CodeAlpaca-20k, bigcode/self-oss-instruct, Magicoder-OSS-Instruct,
  Evol-Instruct-Code (Python rows by default), optional CodeFeedback.
  Total: ~200K+ problems with no generation needed.

Then: python scripts/merge_traces.py  → data/all_traces.jsonl (includes public_supplement + kaggle_supplement when present).

Upload train bundle: all_traces.jsonl and optionally public_supplement.jsonl / kaggle_supplement.jsonl to the dataset
repo; kaggle_train builds data/train_full.jsonl from downloaded JSONL files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssm.reasoning_trace_format import record_for_sft


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {len(records)} rows → {path}")


def cmd_humaneval(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    rows = []
    for ex in ds:
        pid = ex["task_id"]
        prompt = ex["prompt"]
        solution = ex["canonical_solution"]
        test_code = ex["test"] + f"\ncheck({ex['entry_point']})"
        rows.append(
            record_for_sft(
                prompt=prompt,
                thinking="",
                solution=solution,
                source="openai_humaneval",
                problem_id=pid,
                test_code=test_code,
            )
        )
    write_jsonl(Path(args.output), rows)


def cmd_mbpp(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    if args.sanitized:
        ds = load_dataset("mbpp", "sanitized", split=args.split)
    else:
        ds = load_dataset("mbpp", split=args.split)
    rows = []
    for ex in ds:
        tid = ex.get("task_id", len(rows))
        prompt = (ex.get("text") or ex.get("prompt") or "").strip()
        solution = (ex.get("code") or "").strip()
        if not prompt or not solution:
            continue
        setup = ex.get("test_setup_code") or ""
        if ex.get("test_imports"):
            setup = setup + "\n" + "\n".join(ex["test_imports"])
        tests = "\n".join(ex.get("test_list") or [])
        test_code = (setup + "\n\n" + tests).strip() if setup.strip() else tests
        rows.append(
            record_for_sft(
                prompt=prompt,
                thinking="",
                solution=solution,
                source="mbpp_sanitized" if args.sanitized else "mbpp",
                problem_id=f"MBPP/{tid}",
                test_code=test_code or None,
            )
        )
    write_jsonl(Path(args.output), rows)


def _get_nested(obj: dict, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if cur is None:
            return None
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def _get_field(rec: dict, name: str | None):
    if not name:
        return ""
    val = _get_nested(rec, name) if "." in name else rec.get(name)
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val).strip()


def load_records_from_file(path: Path, max_rows: int | None = None) -> list[dict]:
    """Load tabular / JSONL / JSON-array records from a single file."""
    suf = path.suffix.lower()
    if suf == ".csv":
        rows: list[dict] = []
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for i, rec in enumerate(reader):
                if max_rows is not None and i >= max_rows:
                    break
                rows.append({k: (v or "") for k, v in rec.items()})
        return rows
    if suf in (".jsonl", ".ndjson"):
        rows = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if max_rows is not None and i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except json.JSONDecodeError:
                    pass
        return rows
    if suf == ".json":
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, list):
            out = [x for x in data if isinstance(x, dict)]
            if max_rows is not None:
                out = out[:max_rows]
            return out
        if isinstance(data, dict):
            for key in ("train", "test", "validation", "data", "rows", "examples"):
                if key in data and isinstance(data[key], list):
                    out = [x for x in data[key] if isinstance(x, dict)]
                    if max_rows is not None:
                        out = out[:max_rows]
                    return out
            return [data]
        return []
    return []


def cmd_apps(args: argparse.Namespace) -> None:
    """
    codeparrot/apps — 10,000 Python programming problems with test cases.
    Difficulty: introductory / interview / competition.
    test_code is populated from the input/output test cases → usable by GRPO execution reward.
    """
    from datasets import load_dataset

    splits = args.splits.split(",")
    rows: list[dict] = []
    for split in splits:
        try:
            ds = load_dataset("codeparrot/apps", split=split.strip())
        except Exception as e:
            print(f"  APPS split {split!r} failed: {e}")
            continue
        for i, ex in enumerate(ds):
            if args.max_rows and len(rows) >= args.max_rows:
                break
            if args.difficulty and ex.get("difficulty", "") not in args.difficulty.split(","):
                continue
            problem = (ex.get("question") or "").strip()
            if not problem:
                continue
            # solutions is a JSON-string array of accepted solutions
            raw_sols = ex.get("solutions") or "[]"
            try:
                sols = json.loads(raw_sols) if isinstance(raw_sols, str) else raw_sols
            except json.JSONDecodeError:
                sols = []
            solution = sols[0].strip() if sols else ""
            if not solution:
                continue
            # Build a minimal test harness from input_output
            test_code = None
            raw_io = ex.get("input_output") or "{}"
            try:
                io = json.loads(raw_io) if isinstance(raw_io, str) else raw_io
                inputs = io.get("inputs", [])
                outputs = io.get("outputs", [])
                if inputs and outputs:
                    # Use the first test case as a smoke-test assertion
                    test_code = (
                        f"# APPS auto-generated test (first case)\n"
                        f"import io, sys\n"
                        f"_in = {repr(str(inputs[0]))}\n"
                        f"_expected = {repr(str(outputs[0]).strip())}\n"
                        f"sys.stdin = io.StringIO(_in)\n"
                        f"import io as _io; _buf = _io.StringIO(); sys.stdout = _buf\n"
                        f"# (solution output captured above)\n"
                        f"_out = _buf.getvalue().strip()\n"
                        f"assert _out == _expected, f'got {{_out!r}}, expected {{_expected!r}}'"
                    )
            except Exception:
                pass
            rows.append(
                record_for_sft(
                    prompt=problem,
                    thinking="",
                    solution=solution,
                    source="apps",
                    problem_id=f"apps_{split}_{i}",
                    test_code=test_code,
                )
            )
    write_jsonl(Path(args.output), rows)


def cmd_humanevalpack(args: argparse.Namespace) -> None:
    """
    bigcode/humanevalpack — multilingual HumanEval in Python, JavaScript, Java, Go, C++, Rust.
    Each row has the problem prompt, canonical solution, and test code — perfect for multilingual GRPO.
    """
    from datasets import load_dataset

    LANG_MAP = {
        "python": "python",
        "js": "javascript",
        "java": "java",
        "go": "go",
        "cpp": "cpp",
        "rust": "rust",
    }
    requested = [l.strip() for l in args.languages.split(",")]

    rows: list[dict] = []
    for hf_lang, canonical_lang in LANG_MAP.items():
        if canonical_lang not in requested:
            continue
        try:
            ds = load_dataset("bigcode/humanevalpack", hf_lang, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  humanevalpack {hf_lang} skipped: {e}")
            continue
        for ex in ds:
            task_id = ex.get("task_id", f"{hf_lang}_{len(rows)}")
            prompt = (ex.get("prompt") or ex.get("declaration") or "").strip()
            solution = (ex.get("canonical_solution") or "").strip()
            test = (ex.get("test") or "").strip()
            entry = ex.get("entry_point", "")
            test_code = (test + (f"\ncheck({entry})" if entry else "")).strip() or None
            if not prompt or not solution:
                continue
            rows.append(
                record_for_sft(
                    prompt=prompt,
                    thinking="",
                    solution=solution,
                    source="humanevalpack",
                    problem_id=f"hep_{task_id}",
                    test_code=test_code,
                    language=canonical_lang,
                )
            )
        print(f"  humanevalpack {hf_lang}: {sum(1 for r in rows if r.get('language') == canonical_lang)} rows")
    write_jsonl(Path(args.output), rows)


def cmd_magicoder(args: argparse.Namespace) -> None:
    """
    ise-uiuc/Magicoder-OSS-Instruct-75K — 75K diverse coding problems synthesised
    from real open-source code snippets. High quality, diverse languages.
    """
    from datasets import load_dataset

    rows: list[dict] = []
    try:
        ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    except Exception as e:
        print(f"  Magicoder load failed: {e}")
        write_jsonl(Path(args.output), rows)
        return

    for i, ex in enumerate(ds):
        if args.max_rows and i >= args.max_rows:
            break
        problem = (ex.get("problem") or ex.get("instruction") or "").strip()
        solution = (ex.get("solution") or ex.get("response") or ex.get("output") or "").strip()
        lang = (ex.get("lang") or ex.get("language") or "python").lower()
        if not problem or not solution:
            continue
        rows.append(
            record_for_sft(
                prompt=problem,
                thinking="",
                solution=solution,
                source="magicoder",
                problem_id=f"magicoder_{i}",
                language=lang,
            )
        )
    write_jsonl(Path(args.output), rows)


def cmd_evol(args: argparse.Namespace) -> None:
    """
    nickrosh/Evol-Instruct-Code-80k-v1 — 80K evolved coding instruction/response pairs.
    WizardCoder-style data: problems are progressively rewritten to be more complex.
    """
    from datasets import load_dataset

    rows: list[dict] = []
    for did in (
        "nickrosh/Evol-Instruct-Code-80k-v1",
        "theblackcat102/evol-codealpaca-v1",
    ):
        try:
            ds = load_dataset(did, split="train")
            print(f"  Evol-Instruct using: {did}")
            break
        except Exception as e:
            print(f"  skip {did}: {e}")
    else:
        print("  Evol-Instruct: no dataset loaded")
        write_jsonl(Path(args.output), rows)
        return

    for i, ex in enumerate(ds):
        if args.max_rows and i >= args.max_rows:
            break
        instruction = (ex.get("instruction") or ex.get("prompt") or "").strip()
        output = (ex.get("output") or ex.get("response") or ex.get("code") or "").strip()
        if not instruction or not output:
            continue
        rows.append(
            record_for_sft(
                prompt=instruction,
                thinking="",
                solution=output,
                source="evol_instruct",
                problem_id=f"evol_{i}",
            )
        )
    write_jsonl(Path(args.output), rows)


def cmd_supplement(args: argparse.Namespace) -> None:
    """
    Pull instruction→code traces from popular HF datasets (Python-focused where applicable).
    Does not duplicate HumanEval/MBPP — use merge_traces with all_traces + this file.
    """
    from datasets import load_dataset

    rows: list[dict] = []

    # ── CodeAlpaca-style (instruction / input / output) ───────────────────────
    n_cap = args.max_codealpaca
    loaded = False
    for did in ("sahil2801/CodeAlpaca-20k", "codealpaca/CodeAlpaca-20k"):
        try:
            ds = load_dataset(did, split="train")
            loaded = True
            print(f"  Using dataset: {did}")
            break
        except Exception as e:
            print(f"  skip {did}: {e}")
    if loaded:
        for i, ex in enumerate(ds):
            if i >= n_cap:
                break
            instr = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            out = (ex.get("output") or ex.get("code") or "").strip()
            if not instr or not out:
                continue
            prompt = f"{instr}\n\n{inp}" if inp else instr
            rows.append(
                record_for_sft(
                    prompt=prompt,
                    thinking="",
                    solution=out,
                    source="codealpaca",
                    problem_id=f"codealpaca_{i}",
                )
            )
    print(f"  CodeAlpaca rows: {len(rows)}")

    # ── bigcode/self-oss-instruct (optionally Python-only) ────────────────────
    before_self = len(rows)
    ds = None
    for did, split_spec in (
        ("bigcode/self-oss-instruct-sc2-exec-filter-50k", "train"),
        ("bigcode/self-oss-instruct", f"train[:{max(args.max_self_oss * 4, 50000)}]"),
    ):
        try:
            ds = load_dataset(did, split=split_spec)
            print(f"  Using self-oss dataset: {did}")
            break
        except Exception as e:
            print(f"  skip {did}: {e}")

    def _is_python_row(ex_row: dict) -> bool:
        seen = False
        for k in ("programming_language", "lang", "language"):
            v = ex_row.get(k)
            if v is None:
                continue
            seen = True
            s = str(v).lower()
            if "python" in s or s in ("py", "python3"):
                return True
            return False
        return not seen

    if ds is not None:
        added_s = 0
        for ex in ds:
            if added_s >= args.max_self_oss:
                break
            if getattr(args, "self_oss_python_only", True) and not _is_python_row(ex):
                continue
            instr = (ex.get("instruction") or ex.get("prompt") or "").strip()
            out = (ex.get("response") or ex.get("output") or ex.get("code") or "").strip()
            if not instr or not out:
                continue
            rows.append(
                record_for_sft(
                    prompt=instr,
                    thinking="",
                    solution=out,
                    source="self_oss_instruct",
                    problem_id=f"selfoss_{added_s}",
                )
            )
            added_s += 1
    print(f"  self-oss-instruct rows (+{len(rows) - before_self}), total: {len(rows)}")

    # ── Magicoder-OSS-Instruct-75K ────────────────────────────────────────────
    before_mag = len(rows)
    if args.max_magicoder > 0:
        try:
            ds_mag = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
            added_mag = 0
            for i, ex in enumerate(ds_mag):
                if added_mag >= args.max_magicoder:
                    break
                problem = (ex.get("problem") or ex.get("instruction") or "").strip()
                solution = (ex.get("solution") or ex.get("response") or ex.get("output") or "").strip()
                lang = (ex.get("lang") or ex.get("language") or "python").lower()
                if not problem or not solution:
                    continue
                rows.append(record_for_sft(
                    prompt=problem, thinking="", solution=solution,
                    source="magicoder", problem_id=f"magicoder_{i}", language=lang,
                ))
                added_mag += 1
        except Exception as e:
            print(f"  Magicoder-OSS-Instruct skipped: {e}")
    print(f"  Magicoder rows (+{len(rows) - before_mag}), total: {len(rows)}")

    # ── Evol-Instruct-Code-80K ────────────────────────────────────────────────
    before_evol = len(rows)
    if args.max_evol > 0:
        ds_evol = None
        for did in ("nickrosh/Evol-Instruct-Code-80k-v1", "theblackcat102/evol-codealpaca-v1"):
            try:
                ds_evol = load_dataset(did, split="train")
                print(f"  Evol-Instruct: {did}")
                break
            except Exception as e:
                print(f"  skip {did}: {e}")
        if ds_evol is not None:
            added_evol = 0
            for i, ex in enumerate(ds_evol):
                if added_evol >= args.max_evol:
                    break
                instruction = (ex.get("instruction") or ex.get("prompt") or "").strip()
                output = (ex.get("output") or ex.get("response") or ex.get("code") or "").strip()
                if not instruction or not output:
                    continue
                rows.append(record_for_sft(
                    prompt=instruction, thinking="", solution=output,
                    source="evol_instruct", problem_id=f"evol_{i}",
                ))
                added_evol += 1
    print(f"  Evol-Instruct rows (+{len(rows) - before_evol}), total: {len(rows)}")

    # ── Optional: m-a-p/CodeFeedback (English, instruction/output) ───────────
    if args.max_codefeedback > 0:
        n_cf = len(rows)
        try:
            cfsplit = f"train[:{args.max_codefeedback}]"
            ds_cf = load_dataset("m-a-p/CodeFeedback-Filtered", split=cfsplit, trust_remote_code=True)
            for i, ex in enumerate(ds_cf):
                instr = (ex.get("instruction") or ex.get("query") or "").strip()
                out = (ex.get("answer") or ex.get("output") or "").strip()
                if not instr or not out:
                    continue
                rows.append(
                    record_for_sft(
                        prompt=instr,
                        thinking="",
                        solution=out,
                        source="codefeedback_filtered",
                        problem_id=f"codefeedback_{i}",
                    )
                )
        except Exception as e:
            print(f"  CodeFeedback-Filtered skipped: {e}")
        print(f"  CodeFeedback rows (+{len(rows) - n_cf}), total: {len(rows)}")

    write_jsonl(Path(args.output), rows)


def cmd_jsonl(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    raw = load_records_from_file(inp)
    rows = [_row_from_mapping(d, args, i) for i, d in enumerate(raw)]
    rows = [r for r in rows if r.get("prompt") and r.get("solution")]
    write_jsonl(Path(args.output), rows)


def _row_from_mapping(d: dict, args: argparse.Namespace, index: int) -> dict:
    prompt = _get_field(d, args.prompt_field)
    solution = _get_field(d, args.solution_field)
    thinking = _get_field(d, args.thinking_field) if args.thinking_field else ""
    pid = _get_field(d, args.id_field) if args.id_field else f"row_{index}"
    test_code = _get_field(d, args.test_field) if args.test_field else None
    return record_for_sft(
        prompt=prompt,
        thinking=thinking,
        solution=solution,
        source=args.source_name,
        problem_id=pid or f"row_{index}",
        test_code=test_code or None,
    )


def _discover_kaggle_data_files(root: Path, data_files: list[str] | None) -> list[Path]:
    skip_parts = {"__MACOSX", ".ipynb_checkpoints", "__pycache__"}
    skip_names = {"dataset-metadata.json", "kernel-metadata.json"}
    if data_files:
        found: list[Path] = []
        for name in data_files:
            matches = [p for p in root.rglob(name) if p.is_file() and not any(s in p.parts for s in skip_parts)]
            if matches:
                found.append(sorted(matches)[0])
            else:
                print(f"  WARN: no file matching {name!r} under {root}")
        return found
    out: list[Path] = []
    for pat in ("*.jsonl", "*.ndjson", "*.csv", "*.json"):
        for p in root.rglob(pat):
            if not p.is_file() or any(s in p.parts for s in skip_parts):
                continue
            if p.name.lower() in skip_names:
                continue
            out.append(p)
    pri = {".jsonl": 0, ".ndjson": 0, ".csv": 1, ".json": 2}
    return sorted(out, key=lambda p: (pri.get(p.suffix.lower(), 9), str(p)))


def cmd_kaggle(args: argparse.Namespace) -> None:
    """Download Kaggle dataset(s) and map columns into SFT JSONL (same flags as jsonl)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Install the Kaggle CLI package: pip install kaggle", file=sys.stderr)
        sys.exit(1)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(
            f"Kaggle authentication failed: {e}\n"
            "  Use ~/.kaggle/kaggle.json (username + key) or env KAGGLE_USERNAME and KAGGLE_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    cache = Path(args.cache)
    rows: list[dict] = []
    data_files = getattr(args, "data_files", None) or None
    global_i = 0

    for slug in args.datasets:
        if args.max_total is not None and len(rows) >= args.max_total:
            break
        dest = cache / slug.replace("/", "__")
        dest.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {slug} → {dest}")
        try:
            api.dataset_download_files(
                slug,
                path=str(dest),
                unzip=True,
                quiet=False,
                force=args.force_download,
            )
        except Exception as e:
            print(f"  ERROR {slug}: {e}")
            continue

        files = _discover_kaggle_data_files(dest, data_files)
        if not files:
            print(f"  No tabular/json files found under {dest}")
            continue

        slug_key = slug.replace("/", "_")
        for fp in files:
            if args.max_total is not None and len(rows) >= args.max_total:
                break
            try:
                rel = fp.relative_to(dest)
            except ValueError:
                rel = fp.name
            print(f"    Reading {rel} ({fp.stat().st_size // 1024} KiB)")
            raw = load_records_from_file(fp, max_rows=args.max_rows_per_file)
            for j, rec in enumerate(raw):
                if args.max_total is not None and len(rows) >= args.max_total:
                    break
                if not isinstance(rec, dict):
                    continue
                mapped = _row_from_mapping(rec, args, global_i)
                mapped["problem_id"] = f"kaggle_{slug_key}_{fp.stem}_{j}"
                mapped["source"] = args.source_name
                rows.append(mapped)
                global_i += 1

    rows = [r for r in rows if r.get("prompt") and r.get("solution")]
    write_jsonl(Path(args.output), rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert external datasets to SFT JSONL")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("humaneval", help="openai_humaneval (HF)")
    h.add_argument("--output", "-o", default="data/external_reasoning.jsonl")
    h.set_defaults(func=cmd_humaneval)

    m = sub.add_parser("mbpp", help="google-research mbpp (HF)")
    m.add_argument(
        "--sanitized",
        action="store_true",
        help="Use sanitized config (smaller, cleaner prompts)",
    )
    m.add_argument("--split", default="test")
    m.add_argument("--output", "-o", default="data/external_reasoning.jsonl")
    m.set_defaults(func=cmd_mbpp)

    j = sub.add_parser("jsonl", help="Map columns from JSONL or CSV")
    j.add_argument("--input", "-i", required=True)
    j.add_argument("--output", "-o", default="data/external_reasoning.jsonl")
    j.add_argument("--prompt-field", default="prompt")
    j.add_argument("--solution-field", default="solution")
    j.add_argument("--thinking-field", default="")
    j.add_argument("--id-field", default="")
    j.add_argument("--test-field", default="")
    j.add_argument("--source-name", default="external_jsonl")
    j.set_defaults(func=cmd_jsonl)

    kg = sub.add_parser(
        "kaggle",
        help="Download Kaggle dataset(s) via API; map columns like jsonl (needs pip install kaggle + credentials)",
    )
    kg.add_argument(
        "-d",
        "--dataset",
        action="append",
        dest="datasets",
        required=True,
        metavar="OWNER/SLUG",
        help="Dataset id (repeat for multiple). Example: -d username/my-code-dataset",
    )
    kg.add_argument("--output", "-o", default="data/kaggle_supplement.jsonl")
    kg.add_argument(
        "--cache",
        default="data/kaggle_cache",
        help="Directory for downloaded archives (reused across runs)",
    )
    kg.add_argument(
        "--data-file",
        action="append",
        dest="data_files",
        help="Only read file(s) with this basename (rglob). Repeatable. Default: all *.jsonl/*.csv/*.json",
    )
    kg.add_argument(
        "--max-rows-per-file",
        type=int,
        default=250_000,
        help="Stop after this many rows per file (safety for huge JSON arrays)",
    )
    kg.add_argument("--max-total", type=int, default=None, help="Cap total rows in the output JSONL")
    kg.add_argument(
        "--force-download",
        action="store_true",
        help="Pass force=True to Kaggle API (re-fetch archive)",
    )
    kg.add_argument("--prompt-field", default="prompt")
    kg.add_argument("--solution-field", default="solution")
    kg.add_argument("--thinking-field", default="")
    kg.add_argument("--id-field", default="")
    kg.add_argument("--test-field", default="")
    kg.add_argument("--source-name", default="kaggle")
    kg.set_defaults(func=cmd_kaggle)

    s = sub.add_parser(
        "supplement",
        help="HF: CodeAlpaca + self-oss-instruct (+ optional CodeFeedback); merge with all_traces",
    )
    s.add_argument("--output", "-o", default="data/public_supplement.jsonl")
    s.add_argument("--max-codealpaca", type=int, default=20000)
    s.add_argument("--max-self-oss", type=int, default=25000)
    s.add_argument("--max-magicoder", type=int, default=50000, help="Rows from Magicoder-OSS-Instruct-75K (0=skip)")
    s.add_argument("--max-evol", type=int, default=40000, help="Rows from Evol-Instruct-Code-80K (0=skip)")
    s.add_argument("--max-codefeedback", type=int, default=0, help="0=skip m-a-p/CodeFeedback-Filtered")
    s.add_argument(
        "--all-languages",
        action="store_true",
        help="For self-oss-instruct: do not filter to Python rows",
    )
    s.set_defaults(func=cmd_supplement, self_oss_python_only=True)

    ap = sub.add_parser("apps", help="codeparrot/apps — 10K Python problems with test cases (GRPO-ready)")
    ap.add_argument("--output", "-o", default="data/apps_traces.jsonl")
    ap.add_argument("--splits", default="train,test", help="Comma-separated dataset splits")
    ap.add_argument("--difficulty", default="", help="Comma-separated: introductory,interview,competition (empty=all)")
    ap.add_argument("--max-rows", type=int, default=0, help="Cap total rows (0=all)")
    ap.set_defaults(func=cmd_apps)

    hep = sub.add_parser("humanevalpack", help="bigcode/humanevalpack — multilingual HumanEval with tests")
    hep.add_argument("--output", "-o", default="data/humanevalpack.jsonl")
    hep.add_argument("--languages", default="python,javascript,java,go,cpp,rust",
                     help="Comma-separated languages to include")
    hep.set_defaults(func=cmd_humanevalpack)

    mag = sub.add_parser("magicoder", help="ise-uiuc/Magicoder-OSS-Instruct-75K")
    mag.add_argument("--output", "-o", default="data/magicoder_traces.jsonl")
    mag.add_argument("--max-rows", type=int, default=0, help="Cap rows (0=all)")
    mag.set_defaults(func=cmd_magicoder)

    ev = sub.add_parser("evol", help="nickrosh/Evol-Instruct-Code-80k-v1")
    ev.add_argument("--output", "-o", default="data/evol_traces.jsonl")
    ev.add_argument("--max-rows", type=int, default=0, help="Cap rows (0=all)")
    ev.set_defaults(func=cmd_evol)

    args = p.parse_args()
    if args.cmd == "supplement" and args.all_languages:
        args.self_oss_python_only = False
    args.func(args)


if __name__ == "__main__":
    main()
