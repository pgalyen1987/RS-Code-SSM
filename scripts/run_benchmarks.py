"""
Unified benchmark runner for CodingSSM.

Loads a checkpoint saved by ``train.sft_reasoning`` / ``train.grpo``
(``{model_state, model_config, ...}`` in a single ``.pt`` file) and runs
HumanEval, MBPP, and BigCodeBench pass@1 evaluations via subprocess-sandboxed
Python execution. Writes a single JSON results file.

Usage:
    python scripts/run_benchmarks.py \\
        --checkpoint checkpoints/grpo/grpo_best.pt \\
        --model-size 700m \\
        --benchmarks humaneval mbpp bigcodebench \\
        --output runs/bench_grpo.json
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

# Must match GRPO training's build_system_prompt("python") exactly.
SYSTEM_PROMPT = (
    "You are an expert Python programmer. Think carefully step by step before writing code.\n"
    "Use <think> tags to show your reasoning, then provide the final solution.\n"
    "Format:\n<think>\n[your chain-of-thought reasoning here]\n</think>\n```python\n[your solution here]\n```"
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


def build_prompt_text(tokenizer, instruction: str, entry_point: str = "") -> tuple[str, str]:
    """Build the ChatML prompt with the GRPO-style forced prefix.

    Returns (full_text, forced_prefix) where forced_prefix is what was
    appended after <|im_start|>assistant\n so extract_code can find ```python.
    """
    func_name = entry_point if entry_point else "solution"
    forced_prefix = f"```python\ndef {func_name}("
    full_text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{forced_prefix}"
    )
    return full_text, forced_prefix


@torch.no_grad()
def generate(
    model,
    tokenizer,
    instruction: str,
    device,
    entry_point: str = "",
    max_new_tokens: int = 768,
    temperature: float = 0.0,
) -> str:
    """Generate using state-passing (depth=1) with repetition penalty.

    Returns forced_prefix + decoded_tokens so extract_code finds the ```python block.
    """
    full_text, forced_prefix = build_prompt_text(tokenizer, instruction, entry_point)
    ids = tokenizer.encode(full_text, add_special_tokens=False)

    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for t in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)

    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    states = [None] * model.config.n_layers
    generated: list[int] = []

    # Process full prompt at once to warm up SSM states (depth=1, matches inference)
    logits, _, states = model(input_tensor, states=states, return_states=True)
    next_logits = logits[0, -1, :].float()

    for _ in range(max_new_tokens):
        # Repetition penalty prevents degenerate token loops
        if generated:
            window = torch.tensor(generated[-64:], device=device)
            counts = torch.bincount(window, minlength=next_logits.shape[0]).float()
            next_logits = next_logits - counts * 5.0

        if temperature <= 0:
            nid = int(torch.argmax(next_logits).item())
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            sorted_p, sorted_idx = probs.sort(descending=True)
            cum = sorted_p.cumsum(0)
            mask = (cum - sorted_p) > 0.95
            sorted_p[mask] = 0.0
            sorted_p /= sorted_p.sum().clamp(min=1e-8)
            nid = int(sorted_idx[torch.multinomial(sorted_p, 1)].item())

        if nid in stop_ids:
            break
        generated.append(nid)

        logits, _, states = model(
            torch.tensor([[nid]], dtype=torch.long, device=device),
            states=states,
            return_states=True,
        )
        next_logits = logits[0, 0, :].float()

    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    return forced_prefix + decoded


# ---------------------------------------------------------------------------
# Benchmark loaders
# ---------------------------------------------------------------------------

def load_humaneval(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    out = []
    for ex in ds:
        test = ex["test"]
        # Conditionally append check() only if the dataset version omits it
        if f"check({ex['entry_point']})" not in test:
            test = test + f"\ncheck({ex['entry_point']})"
        out.append({
            "task_id": ex["task_id"],
            "prompt": ex["prompt"],
            "test_code": test,
            "entry_point": ex["entry_point"],
        })
    return out[:limit] if limit else out


def load_mbpp(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("mbpp", split="test")
    out = []
    for ex in ds:
        # Extract entry_point from first assert in test_list
        ep = ""
        for test_line in ex.get("test_list", []):
            m = re.search(r"assert\s+(\w+)\s*\(", test_line)
            if m:
                ep = m.group(1)
                break
        out.append({
            "task_id": f"MBPP/{ex['task_id']}",
            "prompt": ex["text"],
            "test_code": "\n".join(ex["test_list"]),
            "entry_point": ep,
        })
    return out[:limit] if limit else out


def load_bigcodebench(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    # BigCodeBench split name varies by dataset version; try in order of likelihood.
    splits_to_try = ["v0.1.4", "complete", "v0.1.0_hf", "v0.1.1_hf", "v0.1.2_hf", "v0.1.3_hf", "train"]
    ds = None
    for split in splits_to_try:
        try:
            ds = load_dataset("bigcode/bigcodebench", split=split)
            print(f"[BCB] Loaded bigcode/bigcodebench split='{split}' ({len(ds)} problems)", flush=True)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[BCB] split='{split}' failed: {exc}", flush=True)
    if ds is None:
        raise RuntimeError(
            f"Could not load bigcode/bigcodebench with any of: {splits_to_try}"
        )
    out = []
    for ex in ds:
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

def evaluate(
    name: str,
    problems: list[dict],
    model,
    tokenizer,
    device,
    max_new_tokens: int,
    temperature: float,
) -> dict:
    print(f"\n[{name}] {len(problems)} problems", flush=True)
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

        entry_point = p.get("entry_point", "") or ""
        text = generate(
            model, tokenizer, instruction, device,
            entry_point=entry_point,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        code = extract_code(text, must_include=entry_point)

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
    ap.add_argument("--model-size", default="700m", choices=["3b", "700m", "cpu"])
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument(
        "--benchmarks",
        nargs="+",
        default=["humaneval", "mbpp"],
        choices=["humaneval", "mbpp", "bigcodebench"],
    )
    ap.add_argument("--limit", type=int, default=None, help="Limit problems per benchmark (smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0, help="0=greedy (default), >0=sampling")
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
        summary["results"]["humaneval"] = evaluate(
            "humaneval", problems, model, tokenizer, device, args.max_new_tokens, args.temperature
        )

    if "mbpp" in args.benchmarks:
        problems = load_mbpp(args.limit)
        summary["results"]["mbpp"] = evaluate(
            "mbpp", problems, model, tokenizer, device, args.max_new_tokens, args.temperature
        )

    if "bigcodebench" in args.benchmarks:
        problems = load_bigcodebench(args.limit)
        summary["results"]["bigcodebench"] = evaluate(
            "bigcodebench", problems, model, tokenizer, device, args.max_new_tokens, args.temperature
        )

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
