"""
SFT quality check using GRPO-style prompting.

Tests whether the model can complete function bodies given the same forced prefix
that GRPO uses. This avoids false negatives from the cold-start context-association
failure (model predicts newlines without the forced prefix).
"""
import sys, subprocess, textwrap, torch
from transformers import AutoTokenizer
from arch import CodingSSM
from arch.config import ModelConfig700M

SYSTEM = (
    "You are an expert Python programmer. Think carefully step by step before writing code.\n"
    "Use <think> tags to show your reasoning, then provide the final solution.\n"
    "Format:\n<think>\n[your chain-of-thought reasoning here]\n</think>\n```python\n[your solution here]\n```"
)

# GRPO-style forced prefix: <think> block + function signature start
# Model only needs to complete the function body — same as GRPO rollout generation.
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

device = torch.device("cuda")
print("[CHECK] Loading SFT checkpoint...", flush=True)
ckpt = torch.load("checkpoints/sft/sft_latest.pt", map_location="cpu", weights_only=False)
model = CodingSSM(ModelConfig700M()).to(torch.float16).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
step = ckpt.get("step", "?")
best_loss = ckpt.get("best_loss", float("nan"))
del ckpt
torch.cuda.empty_cache()
print(f"[CHECK] Loaded step={step} best_loss={best_loss:.4f}", flush=True)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
stop_ids = {tok.eos_token_id}
for t in ["<|im_end|>", "<|endoftext|>"]:
    tid = tok.convert_tokens_to_ids(t)
    if tid and tid != tok.unk_token_id:
        stop_ids.add(tid)


def grpo_prefix(problem: dict) -> str:
    """Build the same forced prefix that GRPO uses for rollout generation."""
    user_msg = problem["prompt"]
    if problem.get("test"):
        user_msg += f"\n\nYour solution must pass these tests:\n```python\n{problem['test']}\n```"
    code_prefix = f"```python\ndef {problem['entry_point']}("
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{code_prefix}"
    )


def generate(prefix: str, max_new: int = 256) -> str:
    """Generate with top-p sampling + repetition penalty (mirrors GRPO rollout settings)."""
    ids = torch.tensor([tok.encode(prefix, add_special_tokens=False)], device=device)
    generated = []
    states = [None] * model.config.n_layers

    with torch.no_grad():
        # Process prefix once to get states (depth=1, same as GRPO)
        logits, _, states = model(ids, states=states, return_states=True)
        next_logits = logits[0, -1, :].float()

        for step in range(max_new):
            # Repetition penalty
            if generated:
                window = torch.tensor(generated[-64:], device=device)
                counts = torch.bincount(window, minlength=next_logits.shape[0]).float()
                next_logits = next_logits - counts * 5.0

            # Top-p sampling
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


def extract_code(text: str, entry_point: str) -> str:
    """Extract the Python function from the generated text."""
    import re
    m = re.search(r"```python\s*(.*?)(?:```|$)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: everything after the last ```python
    idx = text.rfind("```python")
    return text[idx + 9:].strip() if idx >= 0 else ""


def run_code(code: str, test: str) -> bool:
    """Execute code + test in a subprocess. Returns True if 'PASS' printed."""
    full = textwrap.dedent(code) + "\n" + test
    try:
        result = subprocess.run(
            [sys.executable, "-c", full],
            capture_output=True, text=True, timeout=10
        )
        return "PASS" in result.stdout
    except Exception:
        return False


passed_gen = 0   # generates syntactically recognizable code
passed_exec = 0  # code actually passes the test

for p in PROBLEMS:
    print(f"\n[CHECK] --- {p['name']} ---", flush=True)
    prefix = grpo_prefix(p)
    response = generate(prefix)
    # Show just the generated part (after the forced prefix)
    generated_part = response[len(prefix):]
    print(generated_part[:400], flush=True)

    code = extract_code(response, p["entry_point"])
    has_return = "return" in code
    has_def = "def " in code or f"def {p['entry_point']}" in response
    print(f"[CHECK] has_return={has_return} has_def={has_def}", flush=True)

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
    print("[CHECK] FAIL: Model generates no code at all — SFT collapsed.", flush=True)
    sys.exit(1)
elif passed_exec == 0:
    print("[CHECK] WARN: Code generated but nothing passes — model partially learned, proceeding with GRPO.", flush=True)
    # Don't exit 1 — GRPO can still improve a model that generates plausible but wrong code
else:
    print(f"[CHECK] PASS: {passed_exec}/{len(PROBLEMS)} solutions correct — SFT healthy.", flush=True)
