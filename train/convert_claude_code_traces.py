"""
Convert nlile/misc-merged-claude-code-traces-v1 to SFT trace format.

Strategy:
  - Skip rows with empty assistant responses
  - Use pre-built `chatml` field if present
  - Otherwise build from messages_json (+ gitdiff injected into user turn)

Usage:
    python -m train.convert_claude_code_traces
"""
import json
import re
from pathlib import Path


def _build_chatml(system: str, messages: list[dict], gitdiff: str = "") -> str:
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        # Inject gitdiff into the first user turn
        if role == "user" and i == 0 and gitdiff:
            content = f"{content}\n\n<gitdiff>\n{gitdiff}\n</gitdiff>"

        if role in ("user", "human"):
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role in ("assistant", "gpt"):
            # Preserve <think> blocks if present, otherwise wrap bare response
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

    return "\n".join(parts)


def convert(output_path: str = "data/claude_code_traces.jsonl", max_gitdiff_lines: int = 60) -> int:
    from datasets import load_dataset

    ds = load_dataset("nlile/misc-merged-claude-code-traces-v1", split="train")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped_empty = 0
    skipped_no_content = 0

    with open(out, "w", encoding="utf-8") as f:
        for ex in ds:
            if ex.get("has_empty_response"):
                skipped_empty += 1
                continue

            assistant = (ex.get("assistant_response") or "").strip()
            if not assistant:
                skipped_empty += 1
                continue

            _gd = (ex.get("gitdiff") or "").strip()
            if _gd:
                _gd_lines = _gd.splitlines()
                if len(_gd_lines) > max_gitdiff_lines:
                    _gd = "\n".join(_gd_lines[:max_gitdiff_lines]) + f"\n... ({len(_gd_lines) - max_gitdiff_lines} lines truncated)"
            gitdiff = _gd
            model = (ex.get("model") or "").strip()

            # Use pre-built chatml if available
            chatml_raw = (ex.get("chatml") or "").strip()
            if chatml_raw:
                # Inject gitdiff into the chatml if not already present
                if gitdiff and "<gitdiff>" not in chatml_raw:
                    chatml_raw = chatml_raw.replace(
                        "<|im_end|>",
                        f"\n\n<gitdiff>\n{gitdiff}\n</gitdiff><|im_end|>",
                        1,  # only first user turn
                    )
                chatml = chatml_raw
            else:
                # Build from messages_json
                messages_raw = (ex.get("messages_json") or "").strip()
                if messages_raw:
                    try:
                        messages = json.loads(messages_raw)
                    except Exception:
                        skipped_no_content += 1
                        continue
                elif ex.get("user_prompt") and ex.get("assistant_response"):
                    messages = [
                        {"role": "user", "content": ex["user_prompt"]},
                        {"role": "assistant", "content": ex["assistant_response"]},
                    ]
                else:
                    skipped_no_content += 1
                    continue

                system = (ex.get("system_prompt") or "").strip()
                chatml = _build_chatml(system, messages, gitdiff)

            if not chatml.strip():
                skipped_no_content += 1
                continue

            f.write(json.dumps({
                "source": "claude_code_traces",
                "model": model,
                "chatml": chatml,
            }) + "\n")
            count += 1

    print(
        f"[claude_code] Converted {count} examples "
        f"(skipped {skipped_empty} empty, {skipped_no_content} no-content) -> {out}"
    )
    return count


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/claude_code_traces.jsonl")
    args = p.parse_args()
    convert(args.output)
