"""
Convert Roman1111111/claude-opus-4.6-10000x to SFT trace format.
Run once before SFT: python -m train.convert_opus_traces
"""
import json, re
from pathlib import Path


def convert(output_path="data/claude_opus_traces.jsonl"):
    from datasets import load_dataset
    ds = load_dataset("Roman1111111/claude-opus-4.6-10000x", split="train")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out, "w") as f:
        for ex in ds:
            messages = ex.get("messages", [])
            system = user = assistant = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    system = content
                elif role == "user":
                    user = content
                elif role == "assistant":
                    assistant = content

            if not user or not assistant:
                continue

            thinking = ""
            response = assistant
            for pat in [r"<thinking>(.*?)</thinking>\s*(.*)", r"<think>(.*?)</think>\s*(.*)"]:
                m = re.search(pat, assistant, re.DOTALL)
                if m:
                    thinking = m.group(1).strip()
                    response = m.group(2).strip()
                    break

            sys_block = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
            asst_part = f"<think>\n{thinking}\n</think>\n{response}" if thinking else response
            chatml = (
                sys_block
                + f"<|im_start|>user\n{user}<|im_end|>\n"
                + f"<|im_start|>assistant\n{asst_part}<|im_end|>"
            )

            f.write(json.dumps({"source": "claude_opus_46", "chatml": chatml}) + "\n")
            count += 1

    print(f"Converted {count} examples -> {out}")
    return count


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/claude_opus_traces.jsonl")
    args = p.parse_args()
    convert(args.output)
