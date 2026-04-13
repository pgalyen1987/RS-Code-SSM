"""
Shared ChatML layout for SFT (train.sft_reasoning.ReasoningTraceDataset).

Records may use either:
  - "chatml" (full string), or
  - "prompt" / "question" + "thinking" + "solution" (built into ChatML here).
"""

# Matches scripts/merge_traces.py reasoning traces (kaggle_traces notebook style)
REASONING_SYSTEM_PROMPT = (
    "You are an expert Python programmer. Think step by step before writing code.\n"
    "Use <redacted_thinking> tags to show your reasoning, then provide the solution.\n"
    "Format:\n"
    "<redacted_thinking>\n[your chain-of-thought reasoning]\n</redacted_thinking>\n"
    "```python\n[your solution]\n```"
)


def build_chatml_from_parts(
    prompt: str,
    thinking: str,
    solution: str,
) -> str:
    """Build full ChatML string for one (prompt, thinking, solution) example."""
    prompt = (prompt or "").strip()
    thinking = (thinking or "").strip()
    solution = (solution or "").strip()

    think_block = f"<redacted_thinking>\n{thinking}\n</redacted_thinking>\n" if thinking else ""
    assistant_content = f"{think_block}```python\n{solution}\n```"

    return (
        f"<|im_start|>system\n{REASONING_SYSTEM_PROMPT}<|redacted_im_end|>\n"
        f"<|im_start|>user\n{prompt}<|redacted_im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}<|redacted_im_end|>"
    )


def record_for_sft(
    *,
    prompt: str,
    thinking: str = "",
    solution: str = "",
    source: str = "external",
    problem_id: str | None = None,
    test_code: str | None = None,
    language: str | None = None,
    extra: dict | None = None,
) -> dict:
    """One JSONL row compatible with SFT + merge_traces + GRPO (optional test_code)."""
    chatml = build_chatml_from_parts(prompt, thinking, solution)
    rec: dict = {
        "source": source,
        "prompt": prompt,
        "thinking": thinking,
        "solution": solution,
        "chatml": chatml,
    }
    if problem_id is not None:
        rec["problem_id"] = problem_id
    if test_code:
        rec["test_code"] = test_code
    if language:
        rec["language"] = language
    if extra:
        rec.update(extra)
    return rec
