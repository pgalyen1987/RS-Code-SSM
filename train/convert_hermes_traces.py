"""
Convert lambda/hermes-agent-reasoning-traces (kimi split) to SFT trace format.

Takes: system + first human turn + first gpt turn (includes <think> + <tool_call>).
This teaches the model to plan and emit structured tool calls without the
multi-turn hallucination risk of full trajectories.

Usage:
    python -m train.convert_hermes_traces
    python -m train.convert_hermes_traces --config glm-5.1 --output data/hermes_glm_traces.jsonl
"""
import json
import re
from pathlib import Path


def convert(output_path: str = "data/hermes_traces.jsonl", config: str = "kimi") -> int:
    from datasets import load_dataset

    ds = load_dataset("lambda/hermes-agent-reasoning-traces", config, split="train")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0

    with open(out, "w", encoding="utf-8") as f:
        for ex in ds:
            conversations = ex.get("conversations", [])
            tools_raw = ex.get("tools", "")

            system = human = gpt = ""
            for msg in conversations:
                role = msg.get("from", "")
                value = msg.get("value", "")
                if role == "system" and not system:
                    system = value
                elif role == "human" and not human:
                    human = value
                elif role == "gpt" and not gpt:
                    gpt = value
                    break

            if not human or not gpt:
                skipped += 1
                continue

            # Append tool schemas to system prompt so the model knows what's available
            sys_content = system
            if tools_raw:
                try:
                    tools_obj = json.loads(tools_raw)
                    tools_str = json.dumps(tools_obj, indent=2)
                except Exception:
                    tools_str = tools_raw
                sys_content = (sys_content + f"\n\nAvailable tools:\n{tools_str}").strip()

            # Extract <think> from gpt response; keep rest (tool_call + follow-up) as response
            thinking = ""
            response = gpt
            m = re.search(r"<think>(.*?)</think>\s*(.*)", gpt, re.DOTALL)
            if m:
                thinking = m.group(1).strip()
                response = m.group(2).strip()

            sys_block = f"<|im_start|>system\n{sys_content}<|im_end|>\n" if sys_content else ""
            asst_part = f"<think>\n{thinking}\n</think>\n{response}" if thinking else response
            chatml = (
                sys_block
                + f"<|im_start|>user\n{human}<|im_end|>\n"
                + f"<|im_start|>assistant\n{asst_part}<|im_end|>"
            )

            f.write(json.dumps({
                "source": f"hermes_agent_{config.replace('-', '_')}",
                "category": ex.get("category", ""),
                "subcategory": ex.get("subcategory", ""),
                "chatml": chatml,
            }) + "\n")
            count += 1

    print(f"[hermes] Converted {count} examples, skipped {skipped} -> {out}")
    return count


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Convert Hermes agent traces to SFT format")
    p.add_argument("--output", default="data/hermes_traces.jsonl")
    p.add_argument("--config", default="kimi", choices=["kimi", "glm-5.1"])
    args = p.parse_args()
    convert(args.output, args.config)
