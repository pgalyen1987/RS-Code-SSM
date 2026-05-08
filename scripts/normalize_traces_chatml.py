"""
Universal trace normalizer.

Reads a JSONL of training traces in any of the supported schemas and emits a
JSONL where every row has a `chatml` field shaped:

    <|im_start|>system
    {system}<|im_end|>
    <|im_start|>user
    {user}<|im_end|>
    <|im_start|>assistant
    {assistant}<|im_end|>

Supported input schemas (auto-detected per row):
    1. {chatml: "..."}                              — passthrough (validated)
    2. {messages: [{role, content}, ...]}           — OpenAI ChatCompletion form
    3. {instruction|prompt|question, output|completion|response}
    4. {prompt|question, thinking?, solution}       — reasoning trace
    5. {prompt|question, output}                    — generic instruct

Records lacking both a user prompt and an assistant response are skipped.
Rows that already have a valid ``chatml`` field are passed through verbatim
so we never re-wrap something that was already correct.

Usage:
    python -m scripts.normalize_traces_chatml INPUT.jsonl OUTPUT.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_SYSTEM = (
    "You are an expert Python programmer and software engineer. "
    "Think carefully step by step.\n"
    "Use <think> tags to show your reasoning, then provide a clear solution.\n"
    "Format:\n<think>\n[your reasoning]\n</think>\n[solution]"
)


def _wrap(system: str, user: str, assistant: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


def _from_messages(msgs: list[dict]) -> str | None:
    """Convert OpenAI-style messages list into ChatML."""
    parts: list[str] = []
    has_assistant = False
    for m in msgs:
        if not isinstance(m, dict):
            return None
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not role or not content:
            continue
        if role not in ("system", "user", "assistant"):
            role = "user"
        if role == "assistant":
            has_assistant = True
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    if not has_assistant:
        return None
    return "\n".join(parts)


def _assistant_from_thinking_solution(rec: dict) -> str:
    thinking = (rec.get("thinking") or "").strip()
    solution = (rec.get("solution") or "").strip()
    if not solution:
        return ""
    if not thinking:
        return solution
    if solution.lstrip().startswith("```"):
        body = solution
    else:
        body = f"```python\n{solution}\n```"
    return f"<think>\n{thinking}\n</think>\n{body}"


def normalize(rec: dict) -> dict | None:
    """Return ``rec`` augmented with a valid ``chatml`` field, or ``None`` to skip."""
    chatml = rec.get("chatml")
    if isinstance(chatml, str) and "<|im_start|>assistant" in chatml and "<|im_end|>" in chatml:
        return rec

    msgs = rec.get("messages")
    if isinstance(msgs, list):
        cm = _from_messages(msgs)
        if cm:
            rec["chatml"] = cm
            return rec

    user = (
        rec.get("prompt")
        or rec.get("instruction")
        or rec.get("question")
        or ""
    ).strip()

    assistant = (
        rec.get("output")
        or rec.get("completion")
        or rec.get("response")
        or _assistant_from_thinking_solution(rec)
        or (rec.get("solution") or "").strip()
    )
    assistant = assistant.strip() if isinstance(assistant, str) else ""

    if not user or not assistant:
        return None

    system = (rec.get("system") or DEFAULT_SYSTEM).strip()
    rec["chatml"] = _wrap(system, user, assistant)
    return rec


def convert_file(in_path: Path, out_path: Path) -> tuple[int, int, int]:
    n_in = n_out = n_skip = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_skip += 1
                continue
            out = normalize(rec)
            if out is None:
                n_skip += 1
                continue
            fout.write(json.dumps(out) + "\n")
            n_out += 1
    return n_in, n_out, n_skip


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a trace JSONL into ChatML form.")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    n_in, n_out, n_skip = convert_file(Path(args.input), Path(args.output))
    print(f"[normalize] {args.input} -> {args.output}: in={n_in} out={n_out} skipped={n_skip}", file=sys.stderr)


if __name__ == "__main__":
    main()
