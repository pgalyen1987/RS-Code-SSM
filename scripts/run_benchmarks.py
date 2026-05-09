"""
Unified benchmark runner for CodingSSM.

Loads a checkpoint saved by ``train.sft_reasoning`` / ``train.grpo``
(``{model_state, model_config, ...}`` in a single ``.pt`` file) and runs
HumanEval, MBPP, and BigCodeBench pass@1 evaluations via subprocess-sandboxed
Python execution. Writes a single JSON results file.

Usage:
    python scripts/run_benchmarks.py \\
        --checkpoint checkpoints/sft_clean/sft_latest.pt \\
        --benchmarks humaneval mbpp bigcodebench \\
        --output runs/bench_sft_clean.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arch import CodingSSM  # noqa: E402
from arch.config import ModelConfig3B, ModelConfig700M, ModelConfigCPU  # noqa: E402


# ---------------------------------------------------------------------------
# Code execution sandbox
# ---------------------------------------------------------------------------

_HARNESS = """\
import sys, traceback
try:
{body}
{tests}
    print("PASS")
except Exception as e:
    print(f"FAIL: {{type(e).__name__}}: {{e}}")
    sys.exit(1)
"""


def _indent(text: str, n: int = 4) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines())


def execute(code: str, tests: str, timeout: int = 12) -> tuple[bool, str]:
    src = _HARNESS.format(body=_indent(code), tests=_indent(tests))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        out = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        ok = out.returncode == 0 and "PASS" in out.stdout
        err = "" if ok else (out.stdout + out.stderr)[:500]
        return ok, err
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Code extraction from model output
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def extract_code(text: str, must_include: str = "") -> str:
    matches = _FENCE_RE.findall(text)
    if matches:
        for m in matches:
            if not must_include or must_include in m:
                return m.strip()
        return matches[0].strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


# ---------------------------------------------------------------------------
# Model loading + generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert Python programmer and software engineer. "
    "Think carefully step by step.\n"
    "Use <think> tags to show your reasoning, then provide a clear solution.\n"
    "Format:\n<think>\n[your reasoning]\n</think>\n[solution]"
)


def load_model(checkpoint: str, model_size: str, device: torch.device):
    if model_size == "3b":
        cfg = ModelConfig3B()
    elif model_size == "700m":
        cfg = ModelConfig700M()
    else:
        cfg = ModelConfigCPU()

    model = CodingSSM(cfg)
    print(f"[load] checkpoint={checkpoint}", flush=True)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ck.get("model_state", ck) if isinstance(ck, dict) else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    model.eval()
    if device.type == "cuda":
        model = model.to(torch.float16).to(device)
    else:
        model = model.to(device)
    return model, cfg


def build_prompt_ids(tokenizer, instruction: str) -> torch.Tensor:
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor([ids], dtype=torch.long)


@torch.no_grad()
def generate(model, tokenizer, instruction: str, device, max_new_tokens: int = 768, temperature: float = 0.0) -> str:
    ids = build_prompt_ids(tokenizer, instruction).to(device)
    if hasattr(model, "generate"):
        try:
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-3),
                top_p=0.95,
                eos_token_id=eos_id,
            )
            # CodingSSM.generate returns only newly sampled ids: (B, new_len)
            new_ids = out[0] if out.dim() == 2 else out
            return tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)
        except Exception as e:  # noqa: BLE001
            print(f"[gen] model.generate failed ({e}); falling back to manual loop", flush=True)

    # Greedy / argmax fallback when no .generate is available.
    cur = ids
    eos = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|im_end|>")
    out_tokens: list[int] = []
    for _ in range(max_new_tokens):
        logits, _ = model(cur)
        next_logits = logits[0, -1].float()
        if temperature > 0:
            probs = torch.softmax(next_logits / max(temperature, 1e-3), dim=-1)
            tok = int(torch.multinomial(probs, 1).item())
        else:
            tok = int(torch.argmax(next_logits).item())
        if eos is not None and tok == eos:
            break
        out_tokens.append(tok)
        cur = torch.cat([cur, torch.tensor([[tok]], device=device, dtype=cur.dtype)], dim=1)
    return tokenizer.decode(out_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Benchmark loaders
# ---------------------------------------------------------------------------

def load_humaneval(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    out = []
    for ex in ds:
        out.append({
            "task_id": ex["task_id"],
            "prompt": ex["prompt"],
            "test_code": ex["test"] + f"\ncheck({ex['entry_point']})",
            "entry_point": ex["entry_point"],
        })
    return out[:limit] if limit else out


def load_mbpp(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("mbpp", split="test")
    out = []
    for ex in ds:
        out.append({
            "task_id": f"MBPP/{ex['task_id']}",
            "prompt": ex["text"],
            "test_code": "\n".join(ex["test_list"]),
            "entry_point": "",
        })
    return out[:limit] if limit else out


def load_bigcodebench(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    # Use the standard BigCodeBench split. The dataset id has flipped over time;
    # ``bigcode/bigcodebench`` is the canonical one as of 2025.
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    out = []
    for ex in ds:
        # Each row carries: task_id, complete_prompt, instruct_prompt, code_prompt,
        # canonical_solution, test, entry_point
        out.append({
            "task_id": ex.get("task_id"),
            "prompt": ex.get("complete_prompt") or ex.get("instruct_prompt") or "",
            "test_code": ex.get("test", ""),
            "entry_point": ex.get("entry_point", ""),
        })
    return out[:limit] if limit else out


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------

def evaluate(name: str, problems: list[dict], model, tokenizer, device, max_new_tokens: int, temperature: float) -> dict:
    print(f"\n[{name}] {len(problems)} problems")
    n_pass = 0
    rows = []
    t0 = time.time()
    for i, p in enumerate(problems):
        if name == "humaneval":
            instruction = f"Complete the following Python function:\n\n```python\n{p['prompt']}\n```"
        elif name == "mbpp":
            instruction = (
                f"Write a Python function for the task. The tests below must pass.\n\n"
                f"Task: {p['prompt']}\n\nTests:\n{p['test_code']}"
            )
        else:
            instruction = p["prompt"]

        text = generate(model, tokenizer, instruction, device, max_new_tokens, temperature)
        code = extract_code(text, must_include=p.get("entry_point", "") or "")

        if name == "humaneval":
            full_code = p["prompt"] + "\n" + code
        else:
            full_code = code

        ok, err = execute(full_code, p["test_code"], timeout=12)
        if ok:
            n_pass += 1
        rows.append({
            "task_id": p["task_id"],
            "passed": ok,
            "error": err,
            "code_snippet": code[:200],
        })
        rate = n_pass / (i + 1)
        elapsed = time.time() - t0
        mark = "PASS" if ok else "FAIL"
        print(
            f"  [{i+1}/{len(problems)}] {p['task_id']} {mark}  pass@1={rate:.3f}  ({elapsed:.0f}s)",
            flush=True,
        )

    return {
        "benchmark": name,
        "n_total": len(problems),
        "n_passed": n_pass,
        "pass@1": n_pass / max(len(problems), 1),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-size", default="3b", choices=["3b", "700m", "cpu"])
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--benchmarks", nargs="+", default=["humaneval", "mbpp"], choices=["humaneval", "mbpp", "bigcodebench"])
    ap.add_argument("--limit", type=int, default=None, help="Limit problems per benchmark for smoke tests")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output", default="runs/benchmarks.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model, _ = load_model(args.checkpoint, args.model_size, device)

    summary: dict = {"checkpoint": args.checkpoint, "results": {}}

    if "humaneval" in args.benchmarks:
        problems = load_humaneval(args.limit)
        summary["results"]["humaneval"] = evaluate("humaneval", problems, model, tokenizer, device, args.max_new_tokens, args.temperature)

    if "mbpp" in args.benchmarks:
        problems = load_mbpp(args.limit)
        summary["results"]["mbpp"] = evaluate("mbpp", problems, model, tokenizer, device, args.max_new_tokens, args.temperature)

    if "bigcodebench" in args.benchmarks:
        problems = load_bigcodebench(args.limit)
        summary["results"]["bigcodebench"] = evaluate("bigcodebench", problems, model, tokenizer, device, args.max_new_tokens, args.temperature)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for name, res in summary["results"].items():
        print(f"  {name}: pass@1 = {res['pass@1']:.4f} ({res['n_passed']}/{res['n_total']})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
