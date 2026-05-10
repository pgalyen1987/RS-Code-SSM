"""
Prepare clean SFT training data from:
  1. qwen_sft_solutions.jsonl  (coding problems with passing solutions — primary)
     OR auto-downloaded from HF: pgalyen1987/rs-code-ssm-traces / all_traces.jsonl
  2. claude_opus_traces.jsonl  (general Q&A, all have assistant turns)
  3. hermes_traces.jsonl       (agent/reasoning, all have assistant turns)

Every output record carries:
  - chatml:     full ChatML string (for SFT label masking)
  - prompt:     raw problem text (for GRPO prompt building)
  - test_code:  test harness string (for GRPO execution reward)
  - language:   "python" / "cpp" / etc.
  - source:     origin dataset

Usage:
    python scripts/prepare_sft_data.py [--output data/sft_clean.jsonl]
    python scripts/prepare_sft_data.py --hf-repo pgalyen1987/rs-code-ssm-traces
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from train.grpo import build_system_prompt  # single source of truth for system prompts


def download_from_hf(repo_id: str, filename: str, dest: Path) -> bool:
    """Download a file from a HuggingFace dataset repo. Returns True on success."""
    try:
        from huggingface_hub import hf_hub_download
        print(f"[HF] Downloading {repo_id}/{filename} → {dest} ...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(dest.parent),
        )
        # hf_hub_download may save to a cache subdir; copy to dest if different
        downloaded = Path(path)
        if downloaded != dest:
            import shutil
            shutil.copy2(downloaded, dest)
        print(f"[HF] Downloaded → {dest} ({dest.stat().st_size // 1024 // 1024} MB)")
        return True
    except Exception as e:
        print(f"[HF] Download failed: {e}")
        return False


def qwen_solutions_to_records(path: Path) -> list[dict]:
    """
    Convert qwen_sft_solutions.jsonl to unified records.

    Expected input fields (any subset):
        problem / prompt / question  — the problem statement
        solution / code              — the code solution
        thinking / think             — optional chain-of-thought
        test_code                    — test harness (for GRPO reward)
        language                     — defaults to "python"
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            problem = rec.get("problem") or rec.get("prompt") or rec.get("question") or ""
            solution = rec.get("solution") or rec.get("code") or ""
            thinking = (rec.get("thinking") or rec.get("think") or "").strip()
            test_code = rec.get("test_code") or ""
            language = rec.get("language") or "python"
            if not problem or not solution:
                continue

            system = build_system_prompt(language)
            if thinking:
                assistant = f"<think>\n{thinking}\n</think>\n```{language}\n{solution.strip()}\n```"
            else:
                assistant = f"```{language}\n{solution.strip()}\n```"

            chatml = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{problem.strip()}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant}<|im_end|>"
            )
            records.append({
                "source": "qwen_solutions",
                "chatml": chatml,
                "prompt": problem.strip(),
                "test_code": test_code,
                "language": language,
            })
    return records


def filter_existing_chatml(path: Path) -> list[dict]:
    """Keep only records with proper assistant turns; add empty prompt/test_code if missing."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chatml = rec.get("chatml", "")
            if "<|im_start|>assistant" not in chatml:
                continue
            idx = chatml.rfind("<|im_start|>assistant\n")
            if idx == -1:
                continue
            after = chatml[idx + len("<|im_start|>assistant\n"):]
            after = after.replace("<|im_end|>", "").strip()
            if len(after) < 10:
                continue
            # Ensure unified schema — GRPO needs prompt/test_code/language
            records.append({
                "source": rec.get("source", "unknown"),
                "chatml": chatml,
                "prompt": rec.get("prompt", ""),
                "test_code": rec.get("test_code", ""),
                "language": rec.get("language", "python"),
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sft_clean.jsonl")
    parser.add_argument("--qwen-solutions", default="data/qwen_sft_solutions.jsonl")
    parser.add_argument("--hf-repo", default="pgalyen1987/rs-code-ssm-traces",
                        help="HF dataset repo to download all_traces.jsonl from if qwen file absent")
    parser.add_argument("--max-per-source", type=int, default=10000)
    parser.add_argument("--no-download", action="store_true",
                        help="Disable auto-download from HuggingFace")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"
    all_records = []

    # 1. Coding solutions (highest priority — has test_code for GRPO)
    qwen_path = Path(args.qwen_solutions)
    if not qwen_path.is_absolute():
        qwen_path = Path(__file__).parent.parent / args.qwen_solutions

    if not qwen_path.exists() and not args.no_download:
        # Auto-download all_traces.jsonl from HF as the primary coding solutions file
        hf_all = data_dir / "hf_all_traces.jsonl"
        if not hf_all.exists():
            download_from_hf(args.hf_repo, "all_traces.jsonl", hf_all)
        if hf_all.exists():
            # Re-format with our canonical system prompt and filter for coding records
            print(f"[hf_traces] Processing {hf_all} as coding solutions...")
            coding = qwen_solutions_to_records(hf_all)
            print(f"[hf_traces] {len(coding)} records (re-formatted with canonical system prompt)")
            all_records.extend(coding)
        else:
            print(f"[coding_data] NOT FOUND — GRPO will use format-only rewards")
    elif qwen_path.exists():
        qwen = qwen_solutions_to_records(qwen_path)
        print(f"[qwen_solutions] {len(qwen)} records")
        all_records.extend(qwen)
    else:
        print(f"[qwen_solutions] NOT FOUND at {qwen_path} — skipping (use --hf-repo to auto-download)")

    # 2. Claude Opus traces (general Q&A — no test_code but good format variety)
    opus_path = data_dir / "claude_opus_traces.jsonl"
    if not opus_path.exists() and not args.no_download and args.hf_repo:
        download_from_hf(args.hf_repo, "claude_opus_traces.jsonl", opus_path)
    if opus_path.exists():
        opus = filter_existing_chatml(opus_path)
        random.shuffle(opus)
        opus = opus[:args.max_per_source]
        print(f"[claude_opus] {len(opus)} records after filtering")
        all_records.extend(opus)

    # 3. Hermes agent traces (has <think> blocks — teaches reasoning format)
    hermes_path = data_dir / "hermes_traces.jsonl"
    if not hermes_path.exists() and not args.no_download and args.hf_repo:
        download_from_hf(args.hf_repo, "hermes_traces.jsonl", hermes_path)
    if hermes_path.exists():
        hermes = filter_existing_chatml(hermes_path)
        random.shuffle(hermes)
        hermes = hermes[:args.max_per_source]
        print(f"[hermes] {len(hermes)} records after filtering")
        all_records.extend(hermes)

    random.shuffle(all_records)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent.parent / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    # Audit
    total = len(all_records)
    bad = sum(1 for r in all_records if "<|im_start|>assistant" not in r.get("chatml", ""))
    has_think = sum(1 for r in all_records if "<think>" in r.get("chatml", ""))
    has_test = sum(1 for r in all_records if r.get("test_code"))
    has_code = sum(1 for r in all_records if "```" in r.get("chatml", ""))
    print(f"\n[DONE] {total} clean records → {out_path}")
    print(f"[AUDIT] missing assistant turn: {bad} (must be 0)")
    print(f"[AUDIT] has <think> blocks:     {has_think} ({100*has_think/max(total,1):.1f}%)")
    print(f"[AUDIT] has code fences:        {has_code} ({100*has_code/max(total,1):.1f}%)")
    print(f"[AUDIT] has test_code (GRPO):   {has_test} ({100*has_test/max(total,1):.1f}%)")
    if has_test == 0:
        print("[WARN]  No test_code found! GRPO will only get format rewards (max 0.1/step).")
        print("        Ensure qwen_sft_solutions.jsonl is present and has test_code fields.")


if __name__ == "__main__":
    main()
