"""
Use Qwen2.5-Coder-3B-Instruct to generate solutions for HumanEval problems.
Execute each against tests. Save all results; keep passing ones for SFT.
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
DATA_PATH = "/workspace/RS-Code-SSM/data/grpo_humaneval.jsonl"
OUT_SFT   = "/workspace/RS-Code-SSM/data/qwen_sft_solutions.jsonl"
OUT_ALL   = "/workspace/RS-Code-SSM/data/qwen_all_solutions.jsonl"
N_SAMPLES = 8   # attempts per problem
TIMEOUT   = 10  # seconds per test

def load_problems():
    problems = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems

def build_prompt(problem):
    ep = problem.get("entry_point", "solution")
    return (
        f"<|im_start|>system\nYou are an expert Python programmer.<|im_end|>\n"
        f"<|im_start|>user\n{problem['prompt']}<|im_end|>\n"
        f"<|im_start|>assistant\n```python\ndef {ep}("
    )

def extract_code(text, entry_point):
    """Extract python code from model output (handles closed and unclosed fences)."""
    import re
    # Model generated after "def ep(" so prepend it
    full = f"def {entry_point}(" + text
    # Try to find closing fence
    m = re.search(r"```", full)
    if m:
        full = full[:m.start()]
    return full.strip()

def run_code(code, test_code, timeout=TIMEOUT):
    harness = textwrap.dedent(f"""
import sys, traceback
try:
{textwrap.indent(code, '    ')}
{textwrap.indent(test_code, '    ')}
    print("PASS")
except Exception as e:
    print(f"FAIL: {{e}}")
    sys.exit(1)
""")
    try:
        r = subprocess.run(
            [sys.executable, "-c", harness],
            capture_output=True, text=True, timeout=timeout
        )
        return r.returncode == 0 and "PASS" in r.stdout
    except Exception:
        return False

def main():
    print(f"Loading {MODEL_ID}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE
    )
    model.eval()
    print("Model loaded.", flush=True)

    problems = load_problems()
    print(f"Problems: {len(problems)}", flush=True)

    sft_records = []
    all_records = []
    total_pass = 0

    for i, prob in enumerate(problems):
        ep = prob.get("entry_point", "solution")
        prompt_text = build_prompt(prob)
        inputs = tok(prompt_text, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]

        passes = 0
        for attempt in range(N_SAMPLES):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=384,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tok.eos_token_id,
                )
            gen_ids = out[0][prompt_len:]
            gen_text = tok.decode(gen_ids, skip_special_tokens=True)
            code = extract_code(gen_text, ep)
            passed = run_code(code, prob.get("test_code", ""))

            rec = {
                "prompt": prob["prompt"],
                "solution": code,
                "entry_point": ep,
                "passed": passed,
                "source": "qwen_humaneval",
            }
            all_records.append(rec)
            if passed:
                passes += 1
                # Format as SFT trace with <think> prefix
                sft_rec = {
                    "instruction": prob["prompt"],
                    "output": (
                        f"<think>\nLet me solve this step by step.\n</think>\n\n"
                        f"```python\n{code}\n```"
                    ),
                    "source": "qwen_humaneval_verified",
                }
                sft_records.append(sft_rec)

        total_pass += passes
        print(f"  [{i+1}/{len(problems)}] {ep}: {passes}/{N_SAMPLES} passed", flush=True)

    print(f"\nTotal: {total_pass}/{len(problems)*N_SAMPLES} passed", flush=True)
    print(f"Unique passing solutions: {len(sft_records)}", flush=True)

    with open(OUT_ALL, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    with open(OUT_SFT, "w") as f:
        for r in sft_records:
            f.write(json.dumps(r) + "\n")
    print(f"Written: {OUT_SFT} ({len(sft_records)} records)", flush=True)
    print(f"Written: {OUT_ALL} ({len(all_records)} records)", flush=True)

if __name__ == "__main__":
    main()
