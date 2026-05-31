"""
SFT quality check — does the model generate runnable Python after SFT?

Prompts the model EXACTLY like the SFT data (scripts/prepare_sft_data.py) and
GRPO (train/grpo.py): same system prompt, opening ```python fence only, no
forced `def name(` body prefix (which would break SSM generation), no <think>.

Usage:
    python scripts/check_sft_quality.py                       # 700m on cuda
    python scripts/check_sft_quality.py --model-size tiny --device cpu \
        --checkpoint checkpoints/local_sft/sft_latest.pt
"""
import argparse
import re
import subprocess
import sys
import textwrap

import torch
from transformers import AutoTokenizer

from arch import CodingSSM
from arch.config import (
    ModelConfig700M, ModelConfig3B, ModelConfigCPU, ModelConfigTiny,
)

# Must match scripts/prepare_sft_data.py SYSTEM_PROMPT and grpo.build_system_prompt.
SYSTEM = (
    "You are an expert Python programmer. "
    "Write clean, correct, well-structured Python code."
)

# Must match train/grpo.py and train/sft_reasoning.py.
TOKENIZER = "Qwen/Qwen2.5-Coder-7B-Instruct"

PROBLEMS = [
    {
        "name": "add",
        "prompt": "Write a Python function add(a, b) that returns the sum of two numbers.",
        "entry_point": "add",
        "test": "assert add(1, 2) == 3\nassert add(-1, 5) == 4\nprint('PASS')",
    },
    {
        "name": "is_palindrome",
        "prompt": "Write a Python function is_palindrome(s) that returns True if s is a palindrome.",
        "entry_point": "is_palindrome",
        "test": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nprint('PASS')",
    },
    {
        "name": "factorial",
        "prompt": "Write a Python function factorial(n) that computes n! recursively.",
        "entry_point": "factorial",
        "test": "assert factorial(0) == 1\nassert factorial(5) == 120\nprint('PASS')",
    },
]

_CONFIGS = {
    "tiny": ModelConfigTiny,
    "cpu": ModelConfigCPU,
    "700m": ModelConfig700M,
    "3b": ModelConfig3B,
}


def build_prefix(problem: dict) -> str:
    """Build the same prompt SFT/GRPO use: system + user + opening fence."""
    user_msg = problem["prompt"]
    if problem.get("test"):
        user_msg += f"\n\nYour solution must pass these tests:\n```python\n{problem['test']}\n```"
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n```python\n"
    )


def generate(model, tok, prefix: str, device, stop_ids, max_new: int = 256) -> str:
    """Top-p sampling + repetition penalty (mirrors GRPO rollout settings)."""
    ids = torch.tensor([tok.encode(prefix, add_special_tokens=False)], device=device)
    generated: list[int] = []
    states = [None] * model.config.n_layers

    with torch.no_grad():
        logits, _, states = model(ids, states=states, return_states=True)
        next_logits = logits[0, -1, :].float()

        for _ in range(max_new):
            if generated:
                window = torch.tensor(generated[-64:], device=device)
                counts = torch.bincount(window, minlength=next_logits.shape[0]).float()
                next_logits = next_logits - counts * 5.0

            probs = torch.softmax(next_logits / 0.7, dim=-1)
            sorted_p, sorted_idx = probs.sort(descending=True)
            cum = sorted_p.cumsum(0)
            mask = (cum - sorted_p) > 0.9
            sorted_p[mask] = 0.0
            sorted_p /= sorted_p.sum().clamp(min=1e-8)
            nid = sorted_idx[torch.multinomial(sorted_p, 1)].item()

            if nid in stop_ids:
                break
            generated.append(nid)

            logits, _, states = model(
                torch.tensor([[nid]], device=device), states=states, return_states=True
            )
            next_logits = logits[0, 0, :].float()

    return prefix + tok.decode(generated)


def extract_code(text: str) -> str:
    """Pull the Python body out of the (possibly unclosed) ```python block."""
    m = re.search(r"```python\s*(.*?)(?:```|<\|im_end\|>|$)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    idx = text.rfind("```python")
    return text[idx + 9:].strip() if idx >= 0 else ""


def run_code(code: str, test: str) -> bool:
    full = textwrap.dedent(code) + "\n" + test
    try:
        result = subprocess.run(
            [sys.executable, "-c", full],
            capture_output=True, text=True, timeout=10,
        )
        return "PASS" in result.stdout
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/sft/sft_latest.pt")
    ap.add_argument("--model-size", default="700m", choices=list(_CONFIGS))
    ap.add_argument("--device", default=None, help="cpu, cuda, cuda:0 ...")
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[CHECK] Device: {device}", flush=True)

    print(f"[CHECK] Loading {args.checkpoint} ...", flush=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CodingSSM(_CONFIGS[args.model_size]())
    model.load_state_dict(ckpt["model_state"])
    # bf16 on GPU (matches training), fp32 on CPU (no bf16 matmul kernels there).
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    model = model.to(device).eval()
    step = ckpt.get("step", "?")
    best_loss = ckpt.get("best_loss", float("nan"))
    del ckpt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[CHECK] Loaded step={step} best_loss={best_loss:.4f}", flush=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    stop_ids = {tok.eos_token_id}
    for t in ["<|im_end|>", "<|endoftext|>"]:
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid != tok.unk_token_id:
            stop_ids.add(tid)

    passed_gen = 0
    passed_exec = 0
    for p in PROBLEMS:
        print(f"\n[CHECK] --- {p['name']} ---", flush=True)
        prefix = build_prefix(p)
        response = generate(model, tok, prefix, device, stop_ids)
        print(response[len(prefix):][:400], flush=True)

        code = extract_code(response)
        has_return = "return" in code
        has_def = "def " in code
        print(f"[CHECK] has_def={has_def} has_return={has_return}", flush=True)

        if has_return or has_def:
            passed_gen += 1
            if code and run_code(code, p["test"]):
                passed_exec += 1
                print("[CHECK] EXEC: PASS", flush=True)
            else:
                print("[CHECK] EXEC: FAIL (wrong answer or syntax error)", flush=True)

    print(f"\n[CHECK] Generates code: {passed_gen}/{len(PROBLEMS)}", flush=True)
    print(f"[CHECK] Passes tests:   {passed_exec}/{len(PROBLEMS)}", flush=True)

    if passed_gen == 0:
        print("[CHECK] FAIL: model generates no code at all — SFT collapsed.", flush=True)
        sys.exit(1)
    elif passed_exec == 0:
        print("[CHECK] WARN: code generated but nothing passes — proceeding with GRPO.", flush=True)
    else:
        print(f"[CHECK] PASS: {passed_exec}/{len(PROBLEMS)} correct — SFT healthy.", flush=True)


if __name__ == "__main__":
    main()
