"""
Generate <think>...</think> chain-of-thought reasoning traces from teacher model.

Usage:
    python -m train.reasoning_data \\
        --model $MODEL_DIR/Qwen3.5-35B-A3B-Q4_K_M.gguf \\
        --output data/reasoning_traces.jsonl \\
        --n-problems 5000

Output JSONL format (one JSON per line):
    {
        "source": "humaneval"|"mbpp"|"codealpaca",
        "problem_id": str,
        "prompt": str,           # original problem statement
        "thinking": str,         # content of <think>...</think>
        "solution": str,         # final code solution
        "full_response": str,    # full raw teacher output
        "chatml": str            # formatted for SFT training
    }
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# ─── EpiChat knowledge retriever ─────────────────────────────────────────────

class EpiChatRetriever:
    """
    Lightweight EpiChat retriever — loads FAISS + units.json directly,
    no dependency on EpiChat's venv. Used to inject relevant structured
    knowledge into teacher prompts.
    """

    def __init__(self, epichat_dir: str):
        self._dir = Path(epichat_dir)
        self._units: dict = {}
        self._index = None
        self._id_map: list = []
        self._embedder = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer

            units_path = self._dir / "episteme_data" / "units.json"
            faiss_path = self._dir / "episteme_data" / "faiss.index"
            map_path   = self._dir / "episteme_data" / "faiss_map.json"

            with open(units_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._units = data
            else:
                self._units = {u["id"]: u for u in data}

            self._index = faiss.read_index(str(faiss_path))

            with open(map_path) as f:
                self._id_map = json.load(f)

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._loaded = True
            print(f"[EpiChat] Loaded {len(self._units)} EUs for context injection", file=sys.stderr)
        except Exception as e:
            print(f"[EpiChat] Disabled — {e}", file=sys.stderr)
            self._loaded = True  # mark as attempted, won't retry

    def query(self, text: str, top_k: int = 3, min_confidence: float = 0.55) -> list[dict]:
        """Return top-k most relevant EpistemicUnits for a given text."""
        self._load()
        if self._index is None or self._embedder is None:
            return []
        try:
            import numpy as np
            vec = self._embedder.encode([text], normalize_embeddings=False).astype("float32")
            D, I = self._index.search(vec, top_k * 3)  # over-fetch, then filter
            results = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self._id_map):
                    continue
                eu_id = self._id_map[idx]
                eu = self._units.get(str(eu_id)) or self._units.get(eu_id)
                if eu is None:
                    continue
                confidence = eu.get("confidence", 0)
                if confidence < min_confidence:
                    continue
                sim = 1.0 / (1.0 + float(dist))
                results.append((sim, eu))
            results.sort(key=lambda x: -x[0])
            return [eu for _, eu in results[:top_k]]
        except Exception as e:
            print(f"[EpiChat] query error: {e}", file=sys.stderr)
            return []

    def format_context(self, eus: list[dict]) -> str:
        """Format EUs as a structured context block for injection into prompts."""
        if not eus:
            return ""
        _s = lambda v: (v or "").strip()
        lines = ["Relevant knowledge (use this to inform your reasoning):"]
        for eu in eus:
            prop = _s(eu.get("proposition"))
            tc = _s(eu.get("time_complexity"))
            sc = _s(eu.get("space_complexity"))
            when = _s(eu.get("when_to_use"))
            trade = _s(eu.get("tradeoffs"))
            code = _s(eu.get("code_snippet"))

            entry = f"• {prop}"
            if tc:
                entry += f" [Time: {tc}]"
            if sc:
                entry += f" [Space: {sc}]"
            if when:
                entry += f"\n  When to use: {when}"
            if trade:
                entry += f"\n  Tradeoffs: {trade}"
            if code and len(code) < 300:
                entry += f"\n  Example:\n  ```python\n  {code}\n  ```"
            lines.append(entry)
        return "\n".join(lines)


# ─── Problem loaders ────────────────────────────────────────────────────────

def load_humaneval() -> list[dict]:
    """Load HumanEval problems from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        problems = []
        for row in ds:
            problems.append({
                "source": "humaneval",
                "problem_id": row["task_id"],
                "prompt": row["prompt"].strip(),
                "test_code": row.get("test", ""),
                "entry_point": row.get("entry_point", ""),
            })
        return problems
    except Exception as e:
        print(f"[WARN] HumanEval load failed: {e}", file=sys.stderr)
        return []


def load_mbpp() -> list[dict]:
    """Load MBPP problems."""
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        problems = []
        for i, row in enumerate(ds):
            problems.append({
                "source": "mbpp",
                "problem_id": f"mbpp_{row.get('task_id', i)}",
                "prompt": row.get("text", row.get("prompt", "")).strip(),
                "test_code": "\n".join(row.get("test_list", row.get("test_cases", []))),
                "entry_point": "",
            })
        return problems
    except Exception as e:
        print(f"[WARN] MBPP load failed: {e}", file=sys.stderr)
        return []


def load_codealpaca(n: int = 2000) -> list[dict]:
    """Load CodeAlpaca problems (instruction-only, no test harness)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        problems = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            instruction = row["instruction"].strip()
            inp = row.get("input", "").strip()
            prompt = instruction if not inp else f"{instruction}\n\nInput: {inp}"
            problems.append({
                "source": "codealpaca",
                "problem_id": f"codealpaca_{i}",
                "prompt": prompt,
                "test_code": "",
                "entry_point": "",
            })
        return problems
    except Exception as e:
        print(f"[WARN] CodeAlpaca load failed: {e}", file=sys.stderr)
        return []


# ─── Prompt building ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Python programmer. Think carefully step by step before writing code.
Use <think> tags to show your reasoning, then provide the final solution.
Format:
<think>
[your chain-of-thought reasoning here]
</think>
```python
[your solution here]
```"""


def build_prompt(problem: dict, epichat_context: str = "") -> str:
    """Build a ChatML prompt for teacher model, optionally injecting EpiChat knowledge."""
    user_msg = problem["prompt"]
    if problem.get("test_code"):
        user_msg += f"\n\nYour solution must pass these tests:\n```python\n{problem['test_code']}\n```"
    if epichat_context:
        user_msg += f"\n\n{epichat_context}"

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_sft_chatml(problem: dict, thinking: str, solution: str) -> str:
    """Build the final SFT training string."""
    user_msg = problem["prompt"]
    if problem.get("test_code"):
        user_msg += f"\n\nYour solution must pass these tests:\n```python\n{problem['test_code']}\n```"

    assistant_response = f"<think>\n{thinking}\n</think>\n```python\n{solution}\n```"

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_response}<|im_end|>"
    )


# ─── Response parsing ─────────────────────────────────────────────────────────

def parse_response(raw: str) -> tuple[str, str]:
    """
    Extract (thinking, solution) from raw teacher output.
    Returns ("", "") if parsing fails — caller should skip.
    """
    # Extract <think>...</think>
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""

    # Extract ```python ... ``` block
    code_match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    if not code_match:
        code_match = re.search(r"```\s*(.*?)```", raw, re.DOTALL)
    solution = code_match.group(1).strip() if code_match else ""

    return thinking, solution


# ─── Code execution (for rejection sampling) ──────────────────────────────────

def _execute_solution(code: str, test_code: str, timeout: int = 10) -> bool:
    """Run code + test_code in a subprocess. Returns True if all tests pass."""
    import subprocess, tempfile
    if not test_code.strip():
        return True  # no tests → accept (can't verify)
    script = f"{code}\n\n{test_code}"
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        import os; os.unlink(fname)


# ─── Ollama HTTP helper ───────────────────────────────────────────────────────

def _ollama_generate(
    prompt: str,
    model: str = "llama3.1:8b",
    host: str = "http://localhost:11434",
    max_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
    timeout: int = 1200,
) -> str:
    """Call Ollama /api/generate and return the response text."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": ["<|im_end|>"],
        },
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "").strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        raise RuntimeError(f"Ollama HTTP {e.code}: {body}") from e


def _test_ollama_model(model: str, host: str = "http://localhost:11434") -> bool:
    """Return True if the model can actually generate a response."""
    try:
        payload = json.dumps({"model": model, "prompt": "hi", "stream": False,
                              "options": {"num_predict": 1}}).encode()
        req = urllib.request.Request(f"{host}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return "response" in data
    except Exception:
        return False


def _pick_ollama_model(preferred: str = "qwen35-teacher", fallback: str = "llama3.1:8b",
                       host: str = "http://localhost:11434") -> str:
    """Return whichever Ollama model can actually generate, preferring preferred."""
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            names = {m["name"].split(":")[0] for m in data.get("models", [])}
    except Exception:
        names = set()

    for model in (preferred, fallback):
        if model.split(":")[0] in names:
            print(f"[INFO] Testing {model}...", file=sys.stderr)
            if _test_ollama_model(model, host):
                return model
            print(f"[WARN] {model} listed but failed to load — trying next", file=sys.stderr)
    return fallback


# ─── Generator ──────────────────────────────────────────────────────────────

class ReasoningDataGenerator:
    def __init__(
        self,
        model_path: str = "",           # kept for CLI compat; ignored (uses Ollama)
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.95,
        n_threads: int = 8,             # kept for CLI compat; ignored
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "",         # auto-select if empty
        epichat_dir: Optional[str] = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.ollama_host = ollama_host

        if ollama_model:
            self.ollama_model = ollama_model
        else:
            self.ollama_model = _pick_ollama_model()
        print(f"[INFO] Using teacher via Ollama: {self.ollama_model} @ {self.ollama_host}", file=sys.stderr)

        # EpiChat retriever for knowledge-grounded prompts
        self.epichat: Optional[EpiChatRetriever] = None
        if epichat_dir:
            self.epichat = EpiChatRetriever(epichat_dir)
            print(f"[INFO] EpiChat context injection enabled from {epichat_dir}", file=sys.stderr)

    def generate_trace(self, problem: dict, n_samples: int = 1) -> dict | None:
        """
        Generate a reasoning trace, optionally with rejection sampling (best-of-N).
        With n_samples > 1, generates N candidates and returns the first that passes tests.
        Falls back to the longest thinking trace if none pass.
        """
        # Retrieve relevant EpiChat EUs to inject as context
        epichat_context = ""
        if self.epichat:
            eus = self.epichat.query(problem["prompt"], top_k=3)
            epichat_context = self.epichat.format_context(eus)

        prompt = build_prompt(problem, epichat_context)
        test_code = problem.get("test_code", "")

        candidates = []
        for attempt in range(max(1, n_samples)):
            # Use higher temperature for diversity in multi-sample mode
            temp = self.temperature if n_samples == 1 else min(1.0, self.temperature + 0.2 * attempt)
            try:
                raw = _ollama_generate(
                    prompt,
                    model=self.ollama_model,
                    host=self.ollama_host,
                    max_tokens=self.max_new_tokens,
                    temperature=temp,
                    top_p=self.top_p,
                )
            except Exception as e:
                print(f"[ERROR] generation failed for {problem['problem_id']}: {e}", file=sys.stderr)
                continue

            thinking, solution = parse_response(raw)
            if not solution:
                continue

            passed = _execute_solution(solution, test_code) if test_code else None
            candidates.append((raw, thinking, solution, passed))

            if passed:
                print(f"  [RFT] ✓ sample {attempt+1}/{n_samples} passes tests", file=sys.stderr)
                break  # found a passing solution

        if not candidates:
            return None

        # Pick: first passing, else longest thinking (most reasoning effort)
        passing = [c for c in candidates if c[3]]
        chosen = passing[0] if passing else max(candidates, key=lambda c: len(c[1]))
        raw, thinking, solution, passed = chosen

        if not thinking:
            print(f"[WARN] no <think> block for {problem['problem_id']}", file=sys.stderr)

        if not solution:
            print(f"[SKIP] no code block found for {problem['problem_id']}", file=sys.stderr)
            return None

        return {
            "source": problem["source"],
            "problem_id": problem["problem_id"],
            "prompt": problem["prompt"],
            "thinking": thinking,
            "solution": solution,
            "tests_passed": passed,
            "full_response": raw,
            "chatml": build_sft_chatml(problem, thinking, solution),
        }

    def generate_dataset(
        self,
        problems: list[dict],
        output_path: Path,
        resume: bool = True,
        n_samples: int = 1,
        verified_only: bool = False,
    ) -> int:
        """
        Write traces to JSONL, resuming from last completed problem.

        n_samples:     generate N candidates per problem, keep first that passes tests (RFT)
        verified_only: if True, skip traces where tests didn't pass (stricter but slower)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build set of already-done IDs
        done_ids: set[str] = set()
        if resume and output_path.exists():
            with open(output_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        done_ids.add(rec["problem_id"])
                    except Exception:
                        pass
            print(f"[RESUME] {len(done_ids)} problems already done", file=sys.stderr)

        pending = [p for p in problems if p["problem_id"] not in done_ids]
        print(f"[INFO] {len(pending)} problems to generate (n_samples={n_samples})", file=sys.stderr)

        n_written = len(done_ids)
        n_verified = 0
        with open(output_path, "a") as f:
            for i, problem in enumerate(pending):
                t0 = time.time()
                trace = self.generate_trace(problem, n_samples=n_samples)
                elapsed = time.time() - t0

                if trace is None:
                    continue

                if verified_only and trace.get("tests_passed") is False:
                    print(f"[SKIP] {problem['problem_id']} — no passing solution found", file=sys.stderr)
                    continue

                f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                f.flush()
                n_written += 1
                if trace.get("tests_passed"):
                    n_verified += 1

                status = "✓" if trace.get("tests_passed") else ("?" if trace.get("tests_passed") is None else "✗")
                print(
                    f"[{n_written:04d}] {status} {problem['problem_id']} "
                    f"think={len(trace['thinking'])}c "
                    f"code={len(trace['solution'])}c "
                    f"({elapsed:.1f}s)",
                    file=sys.stderr,
                )

        return n_written


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate reasoning traces via Ollama teacher model")
    parser.add_argument(
        "--model",
        default="",
        help="Ollama model name (e.g. llama3.1:8b). Auto-selected if empty.",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama server URL",
    )
    parser.add_argument(
        "--output",
        default="data/reasoning_traces.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=5000,
        help="Max total problems to generate (across all sources)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["humaneval", "mbpp", "codealpaca"],
        choices=["humaneval", "mbpp", "codealpaca"],
        help="Which problem sources to use",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--n-threads", type=int, default=8, help="Unused (Ollama handles threading)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--n-samples", type=int, default=1,
        help="Rejection sampling: generate N candidates per problem, keep first that passes tests",
    )
    parser.add_argument(
        "--verified-only", action="store_true",
        help="Only write traces where the solution passes unit tests (requires test_code)",
    )
    parser.add_argument(
        "--epichat-dir",
        default=os.environ.get("EPICHAT_DIR", ""),
        help="EpiChat directory for knowledge-grounded context injection (set to '' to disable)",
    )
    args = parser.parse_args()

    # Load problems
    all_problems: list[dict] = []
    if "humaneval" in args.sources:
        he = load_humaneval()
        print(f"[INFO] HumanEval: {len(he)} problems", file=sys.stderr)
        all_problems.extend(he)
    if "mbpp" in args.sources:
        mb = load_mbpp()
        print(f"[INFO] MBPP: {len(mb)} problems", file=sys.stderr)
        all_problems.extend(mb)
    if "codealpaca" in args.sources:
        ca = load_codealpaca(n=max(0, args.n_problems - len(all_problems)))
        print(f"[INFO] CodeAlpaca: {len(ca)} problems", file=sys.stderr)
        all_problems.extend(ca)

    # Trim to budget
    all_problems = all_problems[: args.n_problems]
    print(f"[INFO] Total problems: {len(all_problems)}", file=sys.stderr)

    epichat_dir = args.epichat_dir or os.environ.get("EPICHAT_DIR")
    if not epichat_dir:
        try:
            from ssm.paths import EPICHAT_DIR
            if (EPICHAT_DIR / "episteme_data" / "units.json").exists():
                epichat_dir = str(EPICHAT_DIR)
        except Exception:
            pass
    gen = ReasoningDataGenerator(
        ollama_model=args.model,
        ollama_host=args.ollama_host,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        epichat_dir=epichat_dir,
    )

    n = gen.generate_dataset(
        all_problems,
        output_path=Path(args.output),
        resume=not args.no_resume,
        n_samples=args.n_samples,
        verified_only=args.verified_only,
    )
    print(f"\n[DONE] {n} traces written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
