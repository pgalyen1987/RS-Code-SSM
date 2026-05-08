"""
Convert qwen_sft_solutions.jsonl ({instruction, output, source})
into the schema train.sft_reasoning expects: a `chatml` field that holds
the full ChatML conversation. The output already contains <think>...</think>
blocks and ```python``` fences, so we wrap it verbatim as the assistant turn.
"""
import json
import sys
from pathlib import Path

SYSTEM = (
    "You are an expert Python programmer and software engineer. "
    "Think carefully step by step.\n"
    "Use <think> tags to show your reasoning, then provide a clear solution.\n"
    "Format:\n<think>\n[your reasoning]\n</think>\n[solution]"
)


def convert(in_path: Path, out_path: Path) -> None:
    n_in = n_out = n_skip = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            instruction = (
                rec.get("instruction")
                or rec.get("prompt")
                or rec.get("question")
                or ""
            ).strip()
            output = (
                rec.get("output")
                or rec.get("completion")
                or rec.get("response")
                or rec.get("solution")
                or ""
            ).strip()
            if not instruction or not output:
                n_skip += 1
                continue
            chatml = (
                f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n{output}<|im_end|>"
            )
            fout.write(
                json.dumps(
                    {
                        "chatml": chatml,
                        "prompt": instruction,
                        "solution": output,
                        "source": rec.get("source", "qwen"),
                    }
                )
                + "\n"
            )
            n_out += 1
    print(f"[convert] {in_path} -> {out_path}: in={n_in} out={n_out} skipped={n_skip}")


if __name__ == "__main__":
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    convert(in_path, out_path)
