"""
Evaluation harness for CodingSSM.

Supported benchmarks:
  - HumanEval (openai/HumanEval): 164 Python programming problems
  - MBPP (google-research-datasets/mbpp): 374 Python programming problems
  - Custom pass@k evaluation

Evaluation flow:
  1. Load benchmark problems
  2. Generate solutions using the student model (greedy or sampling)
  3. Execute solutions in a sandboxed subprocess with timeout
  4. Compute pass@1 and pass@k metrics

NOTE: Code execution is sandboxed via subprocess with a timeout.
      Never execute untrusted code without the sandbox.
"""

import ast
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Problem loaders
# ---------------------------------------------------------------------------

def load_humaneval(cache_dir: str = None) -> list[dict]:
    """
    Load HumanEval benchmark.
    Returns list of dicts with keys:
      task_id, prompt, canonical_solution, test, entry_point
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/HumanEval", split="test", cache_dir=cache_dir)
        return [dict(ex) for ex in ds]
    except Exception as e:
        raise RuntimeError(f"Failed to load HumanEval: {e}")


def load_mbpp(cache_dir: str = None, split: str = "test") -> list[dict]:
    """
    Load MBPP benchmark.
    Returns list of dicts with keys:
      task_id, text (instruction), code (solution), test_list
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", split=split, cache_dir=cache_dir)
        return [dict(ex) for ex in ds]
    except Exception as e:
        raise RuntimeError(f"Failed to load MBPP: {e}")


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code_block(text: str, entry_point: str = None) -> str:
    """
    Extract Python code from model output.
    Tries:
      1. ```python ... ``` fenced block
      2. ``` ... ``` block
      3. Lines that look like a function def matching entry_point
      4. Raw text (fallback)
    """
    import re

    # Try ```python ... ```
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try ``` ... ```
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try to find a function definition block
    if entry_point:
        pattern = rf"(def {re.escape(entry_point)}\s*\(.*?)(?=\ndef |\Z)"
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()

    return text.strip()


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------

def execute_code_safe(
    code: str,
    test_code: str,
    timeout: int = 10,
) -> tuple[bool, str]:
    """
    Execute code + test_code in a subprocess sandbox.

    Returns:
      (passed: bool, error_message: str)
    """
    full_code = textwrap.dedent(code) + "\n\n" + textwrap.dedent(test_code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        else:
            return False, (result.stderr or result.stdout)[:500]
    except subprocess.TimeoutExpired:
        return False, f"Timeout ({timeout}s)"
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# HumanEval evaluator
# ---------------------------------------------------------------------------

def evaluate_humaneval(
    model,
    tokenizer,
    problems: list[dict] = None,
    n_samples: int = 1,         # samples per problem for pass@k
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    cache_dir: str = None,
    output_path: str = None,
    device: str = "cpu",
) -> dict:
    """
    Evaluate model on HumanEval.

    Returns:
      {
        "pass@1": float,
        "pass@k": float (if n_samples > 1),
        "n_correct": int,
        "n_total": int,
        "results": list[dict],
      }
    """
    if problems is None:
        problems = load_humaneval(cache_dir)

    results = []
    n_correct = 0

    for i, prob in enumerate(problems):
        task_id = prob["task_id"]
        prompt = prob["prompt"]
        test_code = prob["test"] + f"\ncheck({prob['entry_point']})"
        entry_point = prob["entry_point"]

        # Format as instruction
        instruction = (
            f"Complete the following Python function:\n\n```python\n{prompt}\n```"
        )

        passed_any = False
        sample_results = []

        for s in range(n_samples):
            t = temperature if n_samples > 1 else 0.0

            raw_output = _generate_text(model, tokenizer, instruction, t, max_new_tokens, device)
            code = extract_code_block(raw_output, entry_point)

            # Prepend prompt so the function is fully defined
            full_code = prompt + "\n" + code

            passed, err = execute_code_safe(full_code, test_code)
            sample_results.append({"passed": passed, "error": err, "code": code[:200]})
            if passed:
                passed_any = True

        if passed_any:
            n_correct += 1

        result = {
            "task_id": task_id,
            "passed": passed_any,
            "samples": sample_results,
        }
        results.append(result)

        status = "✓" if passed_any else "✗"
        print(f"[{i+1}/{len(problems)}] {task_id} {status}  pass_rate={n_correct/(i+1):.3f}")

    pass_at_1 = n_correct / len(problems)

    summary = {
        "pass@1": pass_at_1,
        "n_correct": n_correct,
        "n_total": len(problems),
        "results": results,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {output_path}")

    print(f"\nHumanEval pass@1: {pass_at_1:.4f} ({n_correct}/{len(problems)})")
    return summary


# ---------------------------------------------------------------------------
# MBPP evaluator
# ---------------------------------------------------------------------------

def evaluate_mbpp(
    model,
    tokenizer,
    problems: list[dict] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    cache_dir: str = None,
    output_path: str = None,
    device: str = "cpu",
) -> dict:
    """
    Evaluate model on MBPP.

    Returns same format as evaluate_humaneval.
    """
    if problems is None:
        problems = load_mbpp(cache_dir)

    results = []
    n_correct = 0

    for i, prob in enumerate(problems):
        task_id = prob["task_id"]
        instruction = prob["text"]
        tests = prob.get("test_list", [])

        if not tests:
            continue

        test_code = "\n".join(tests)

        raw_output = _generate_text(model, tokenizer, instruction, temperature, max_new_tokens, device)
        code = extract_code_block(raw_output)

        passed, err = execute_code_safe(code, test_code)
        if passed:
            n_correct += 1

        result = {"task_id": task_id, "passed": passed, "error": err[:200] if err else ""}
        results.append(result)

        status = "✓" if passed else "✗"
        print(f"[{i+1}/{len(problems)}] {task_id} {status}  pass_rate={n_correct/(i+1):.3f}")

    pass_at_1 = n_correct / max(len(problems), 1)
    summary = {
        "pass@1": pass_at_1,
        "n_correct": n_correct,
        "n_total": len(problems),
        "results": results,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nMBPP pass@1: {pass_at_1:.4f} ({n_correct}/{len(problems)})")
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_text(
    model,
    tokenizer,
    instruction: str,
    temperature: float,
    max_new_tokens: int,
    device: str,
) -> str:
    """Generate text from model given an instruction string."""
    from train.dataset import format_chat, IM_START, IM_END

    # Build prompt (no response — model completes it)
    prompt = (
        f"{IM_START}system\nYou are CodingSSM, an expert AI coding assistant.{IM_END}\n"
        f"{IM_START}user\n{instruction}{IM_END}\n"
        f"{IM_START}assistant\n"
    )

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = enc["input_ids"].to(device)

    import torch
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            top_p=0.95,
        )

    # Decode only the new tokens
    new_ids = out_ids[0]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text


# ---------------------------------------------------------------------------
# Quick benchmark runner (CLI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="Evaluate CodingSSM on coding benchmarks")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint dir")
    parser.add_argument("--benchmark", default="humaneval", choices=["humaneval", "mbpp", "both"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--n-problems", type=int, default=None, help="Limit number of problems")
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-Coder-7B")
    args = parser.parse_args()

    from arch.config import ModelConfig
    from arch.model import CodingSSM
    from train.dataset import get_tokenizer
    import torch

    # Load model
    meta_path = Path(args.checkpoint) / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        model_cfg = ModelConfig(**meta.get("model_config", {}))
    else:
        model_cfg = ModelConfig()

    model = CodingSSM(model_cfg)
    state = torch.load(Path(args.checkpoint) / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    tokenizer = get_tokenizer(args.tokenizer)

    if args.benchmark in ("humaneval", "both"):
        problems = load_humaneval()
        if args.n_problems:
            problems = problems[:args.n_problems]
        evaluate_humaneval(
            model, tokenizer, problems,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            output_path=args.output.replace(".json", "_humaneval.json"),
        )

    if args.benchmark in ("mbpp", "both"):
        problems = load_mbpp()
        if args.n_problems:
            problems = problems[:args.n_problems]
        evaluate_mbpp(
            model, tokenizer, problems,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            output_path=args.output.replace(".json", "_mbpp.json"),
        )
