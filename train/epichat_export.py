"""
Convert EpiChat's EpistemicUnit knowledge graph into SFT training traces.

Produces ~2K–3K structured training examples from the existing 3,500 EUs in EpiChat,
covering algorithms, data structures, design patterns, complexity, and code best practices.

No teacher model required — knowledge is already structured and justified.

Output format (data/epichat_traces.jsonl):
    {
        "source": "epichat",
        "eu_id": str,
        "domain": str,
        "confidence": float,
        "chatml": str  -- <think>...</think> + code/explanation
    }

Usage:
    python -m train.epichat_export \\
        --epichat-dir $EPICHAT_DIR \\
        --output data/epichat_traces.jsonl
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Optional


# ─── EpiChat loader (no import — load raw JSON to avoid venv conflicts) ──────

def load_epichat_units(epichat_dir: Path) -> list[dict]:
    """Load serialized EUs from EpiChat's episteme_data/units.json."""
    epichat_dir = Path(epichat_dir).resolve()
    units_path = epichat_dir / "episteme_data" / "units.json"
    if not units_path.exists():
        print(f"[ERROR] units.json not found at {units_path}", file=sys.stderr, flush=True)
        print(f"[HINT] EPICHAT_DIR={epichat_dir} — ensure generate_eus completed and saved.", file=sys.stderr, flush=True)
        return []
    with open(units_path) as f:
        data = json.load(f)
    # units.json is a dict {eu_id: eu_dict} or a list
    if isinstance(data, dict):
        units = list(data.values())
    else:
        units = data
    print(f"[INFO] Loaded {len(units)} EpistemicUnits from EpiChat", file=sys.stderr)
    return units


# ─── Trace builders ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Python programmer and software engineer. Think carefully step by step.
Use <think> tags to show your reasoning, then provide a clear explanation and code example.
Format:
<think>
[your reasoning and analysis]
</think>
[explanation and code]"""


def _s(v) -> str:
    """Safe string: converts None → ''."""
    return (v or "").strip()


def build_algorithm_trace(eu: dict) -> Optional[dict]:
    """
    For algorithm/data_structure EUs: build an "explain + implement" trace.
    """
    prop = _s(eu.get("proposition"))
    if not prop or len(prop) < 20:
        return None

    domain = _s(eu.get("domain"))
    confidence = eu.get("confidence", 0.5) or 0.5
    code_snippet = _s(eu.get("code_snippet"))
    time_complexity = _s(eu.get("time_complexity"))
    space_complexity = _s(eu.get("space_complexity"))
    when_to_use = _s(eu.get("when_to_use"))
    tradeoffs = _s(eu.get("tradeoffs"))

    # Build the reasoning block
    thinking_parts = [f"The proposition is: {prop}"]

    if time_complexity:
        thinking_parts.append(f"Time complexity: {time_complexity}")
    if space_complexity:
        thinking_parts.append(f"Space complexity: {space_complexity}")
    if when_to_use:
        thinking_parts.append(f"When to use: {when_to_use}")
    if tradeoffs:
        thinking_parts.append(f"Key tradeoffs: {tradeoffs}")

    thinking_parts.append(
        "Let me think through the core logic, key invariants, and a clean implementation."
    )
    thinking = "\n".join(thinking_parts)

    # Build the response block
    response_parts = [prop]
    if time_complexity or space_complexity:
        response_parts.append("")
        if time_complexity:
            response_parts.append(f"**Time complexity**: {time_complexity}")
        if space_complexity:
            response_parts.append(f"**Space complexity**: {space_complexity}")
    if when_to_use:
        response_parts.append(f"\n**When to use**: {when_to_use}")
    if tradeoffs:
        response_parts.append(f"\n**Tradeoffs**: {tradeoffs}")

    if code_snippet:
        response_parts.append(f"\n```python\n{code_snippet}\n```")
    elif domain in ("algorithms", "data_structures"):
        # Generate a placeholder — traces without code are still useful
        pass

    response = "\n".join(response_parts)

    # Determine the question framing
    ktype = eu.get("knowledge_type", "")
    if "ALGORITHM" in ktype or domain == "algorithms":
        question = f"Explain the algorithm: {_extract_name(prop)}"
    elif "DATA" in ktype or domain == "data_structures":
        question = f"Explain the data structure: {_extract_name(prop)}"
    elif "DESIGN" in ktype or domain == "design_patterns":
        question = f"Explain the design pattern: {_extract_name(prop)}"
    elif "BEST_PRACTICE" in ktype:
        question = f"What is the best practice for: {_extract_name(prop)}"
    else:
        question = f"Explain: {_extract_name(prop)}"

    chatml = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n{thinking}\n</think>\n"
        f"{response}"
        f"<|im_end|>"
    )

    return {
        "source": "epichat",
        "eu_id": eu.get("id", ""),
        "domain": domain,
        "knowledge_type": ktype,
        "confidence": confidence,
        "question": question,
        "thinking": thinking,
        "response": response,
        "chatml": chatml,
    }


def build_best_practice_trace(eu: dict) -> Optional[dict]:
    """
    For best practice / tradeoff EUs: build a "when to use X and why" trace.
    """
    prop = _s(eu.get("proposition"))
    if not prop or len(prop) < 20:
        return None

    ktype = _s(eu.get("knowledge_type"))
    if not any(t in ktype for t in ("BEST_PRACTICE", "TRADEOFF", "CODE_EXAMPLE", "DESIGN_PATTERN")):
        return None

    domain = _s(eu.get("domain"))
    confidence = eu.get("confidence", 0.5) or 0.5
    code_snippet = _s(eu.get("code_snippet"))
    tradeoffs = _s(eu.get("tradeoffs"))
    when_to_use = _s(eu.get("when_to_use"))

    thinking_parts = [f"Analyzing best practice: {prop}"]
    if tradeoffs:
        thinking_parts.append(f"Tradeoffs to consider: {tradeoffs}")
    if when_to_use:
        thinking_parts.append(f"When this applies: {when_to_use}")
    thinking_parts.append(
        "I should give a concrete recommendation with reasoning and a code example if applicable."
    )
    thinking = "\n".join(thinking_parts)

    response_parts = [prop]
    if tradeoffs:
        response_parts.append(f"\n**Tradeoffs**: {tradeoffs}")
    if when_to_use:
        response_parts.append(f"\n**When to use**: {when_to_use}")
    if code_snippet:
        response_parts.append(f"\n```python\n{code_snippet}\n```")
    response = "\n".join(response_parts)

    question = f"What is the best practice for {_extract_name(prop)}?"

    chatml = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n{thinking}\n</think>\n"
        f"{response}"
        f"<|im_end|>"
    )

    return {
        "source": "epichat",
        "eu_id": eu.get("id", ""),
        "domain": domain,
        "knowledge_type": ktype,
        "confidence": confidence,
        "question": question,
        "thinking": thinking,
        "response": response,
        "chatml": chatml,
    }


def build_complexity_trace(eu: dict) -> Optional[dict]:
    """
    For EUs with complexity info: build a "what is the complexity of X and why" trace.
    """
    time_c = _s(eu.get("time_complexity"))
    space_c = _s(eu.get("space_complexity"))
    if not (time_c or space_c):
        return None

    prop = _s(eu.get("proposition"))
    if not prop:
        return None

    name = _extract_name(prop)
    confidence = eu.get("confidence", 0.5) or 0.5
    code_snippet = _s(eu.get("code_snippet"))

    thinking = (
        f"I need to analyze the complexity of {name}.\n"
        f"The proposition states: {prop}\n"
    )
    if time_c:
        thinking += f"Time complexity is {time_c}. "
    if space_c:
        thinking += f"Space complexity is {space_c}. "
    thinking += "\nLet me explain why these complexities arise from the algorithm's structure."

    response_parts = [f"**{name}** complexity analysis:"]
    if time_c:
        response_parts.append(f"- **Time**: {time_c}")
    if space_c:
        response_parts.append(f"- **Space**: {space_c}")
    if code_snippet:
        response_parts.append(f"\n```python\n{code_snippet}\n```")

    response = "\n".join(response_parts)
    question = f"What is the time and space complexity of {name}? Explain why."

    chatml = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n{thinking}\n</think>\n"
        f"{response}"
        f"<|im_end|>"
    )

    return {
        "source": "epichat_complexity",
        "eu_id": eu.get("id", ""),
        "domain": eu.get("domain", ""),
        "knowledge_type": eu.get("knowledge_type", ""),
        "confidence": confidence,
        "question": question,
        "thinking": thinking,
        "response": response,
        "chatml": chatml,
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_name(proposition: str) -> str:
    """Extract short name from proposition string."""
    # First N words
    words = proposition.split()
    return " ".join(words[:6]) if len(words) > 6 else proposition


def _should_include(eu: dict, min_confidence: float = 0.4) -> bool:
    """Filter EUs worth including in training."""
    if (eu.get("confidence") or 0) < min_confidence:
        return False
    prop = _s(eu.get("proposition"))
    if len(prop) < 20:
        return False
    # Skip raw Wikipedia sentence dumps with no structure
    if _s(eu.get("knowledge_type")) == "EMPIRICAL" and not eu.get("code_snippet"):
        if len(prop) < 50:
            return False
    return True


# ─── Main export pipeline ─────────────────────────────────────────────────────

def export(epichat_dir: Path, output_path: Path, min_confidence: float = 0.4) -> int:
    units = load_epichat_units(epichat_dir)
    if not units:
        print("[ERROR] No units loaded. Run EpiChat seeding first.", file=sys.stderr, flush=True)
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set = set()
    n_written = 0

    # Domain priority: structured code knowledge first
    priority_domains = {"algorithms", "data_structures", "design_patterns", "oop", "best_practices"}

    def sort_key(eu):
        domain = eu.get("domain", "")
        return (0 if domain in priority_domains else 1, -eu.get("confidence", 0))

    units_sorted = sorted(units, key=sort_key)

    with open(output_path, "w") as f:
        for eu in units_sorted:
            if not _should_include(eu, min_confidence):
                continue

            eu_id = eu.get("id", "")
            if eu_id in seen_ids:
                continue

            traces_generated = []

            # Try all trace builders
            t1 = build_algorithm_trace(eu)
            if t1:
                traces_generated.append(t1)

            t2 = build_best_practice_trace(eu)
            if t2 and t2 != t1:
                traces_generated.append(t2)

            t3 = build_complexity_trace(eu)
            if t3:
                traces_generated.append(t3)

            for trace in traces_generated:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                n_written += 1

            if traces_generated:
                seen_ids.add(eu_id)

    print(f"[DONE] {n_written} training traces written to {output_path}", file=sys.stderr, flush=True)
    print(f"       from {len(seen_ids)} unique EpistemicUnits", file=sys.stderr, flush=True)
    return n_written


def main():
    parser = argparse.ArgumentParser(description="Export EpiChat knowledge as SFT training traces")
    parser.add_argument(
        "--epichat-dir",
        default=os.environ.get("EPICHAT_DIR", ""),
        help="EpiChat root directory (default: EPICHAT_DIR env)",
    )
    parser.add_argument("--output", default="data/epichat_traces.jsonl", help="Output JSONL file")
    parser.add_argument("--min-confidence", type=float, default=0.4, help="Minimum EU confidence")
    args = parser.parse_args()

    epichat_dir = args.epichat_dir or os.environ.get("EPICHAT_DIR")
    if not epichat_dir:
        try:
            from ssm.paths import EPICHAT_DIR
            epichat_dir = str(EPICHAT_DIR)
        except ImportError:
            print("[ERROR] EPICHAT_DIR not set and ssm.paths unavailable.", file=sys.stderr, flush=True)
            sys.exit(1)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    n = export(Path(epichat_dir), output_path, args.min_confidence)
    print(f"[INFO] Total: {n} traces", file=sys.stderr, flush=True)
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[ERROR] Trace export failed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
