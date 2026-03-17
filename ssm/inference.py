"""
Test-time compute inference for CodingSSM.

Generates up to N solutions and returns the first that passes tests.
With pass@1=50%, pass@8 = 99.6%. With pass@1=40%, pass@8 = 98.3%.

Usage:
    from ssm.inference import CodingSSMInference
    engine = CodingSSMInference.from_checkpoint('checkpoints/grpo/best.pt')
    result = engine.solve(
        prompt="Write a function that reverses a string",
        test_code="assert reverse('hello') == 'olleh'",
        language='python',
        n_samples=8,
    )
    print(result.solution, result.pass_rate)
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from arch import CodingSSM
from arch.config import ModelConfig700M


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class SolveResult:
    solution: str           # best solution code (extracted from response)
    thinking: str           # chain-of-thought reasoning
    passed: Optional[bool]  # True=tests pass, None=no tests, False=all failed
    n_tried: int            # number of samples generated
    pass_rate: float        # fraction of samples that passed (0.0–1.0)
    language: str           # programming language


# ── Multilingual executor ────────────────────────────────────────────────────

# Language-specific execution templates.
# {code} = solution code, {tests} = test assertions/harness
_PYTHON_HARNESS = """\
import sys, traceback
try:
{code_indented}
{tests_indented}
    print("PASS")
except Exception as e:
    print(f"FAIL: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

_CPP_HARNESS = """\
#include <bits/stdc++.h>
using namespace std;

{code}

int main() {{
{tests_indented}
    cout << "PASS" << endl;
    return 0;
}}
"""

_JAVA_HARNESS = """\
public class Solution {{
{code_indented}

    public static void main(String[] args) throws Exception {{
{tests_indented}
        System.out.println("PASS");
    }}
}}
"""

_JS_HARNESS = """\
{code}

try {{
{tests_indented}
    console.log("PASS");
}} catch(e) {{
    console.error("FAIL: " + e.message);
    process.exit(1);
}}
"""

_GO_HARNESS = """\
package main

import (
    "fmt"
)

{code}

func main() {{
{tests_indented}
    fmt.Println("PASS")
}}
"""

_KOTLIN_HARNESS = """\
{code}

fun main() {{
    try {{
{tests_indented}
        println("PASS")
    }} catch (e: Exception) {{
        System.err.println("FAIL: ${{e.message}}")
        System.exit(1)
    }}
}}
"""


def _indent(text: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


class MultilingualExecutor:
    """
    Executes solution code + test assertions for multiple languages.
    Uses subprocess with configurable timeouts.
    """

    SUPPORTED = {"python", "cpp", "java", "javascript", "js", "go", "kotlin"}

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def execute(self, solution: str, test_code: str, language: str = "python") -> bool:
        """
        Run solution + tests. Returns True if tests pass.
        If test_code is empty, returns False (cannot verify without tests).
        """
        if not test_code.strip():
            return False

        lang = language.lower().strip()
        try:
            if lang == "python":
                return self._run_python(solution, test_code)
            elif lang in ("cpp", "c++"):
                return self._run_cpp(solution, test_code)
            elif lang == "java":
                return self._run_java(solution, test_code)
            elif lang in ("javascript", "js"):
                return self._run_js(solution, test_code)
            elif lang == "go":
                return self._run_go(solution, test_code)
            elif lang == "kotlin":
                return self._run_kotlin(solution, test_code)
            else:
                # Unknown language — try Python as fallback
                return self._run_python(solution, test_code)
        except Exception:
            return False

    def _run_subprocess(self, cmd: list, code: str, suffix: str) -> bool:
        """Write code to a temp file, run cmd, return True if returncode==0 and PASS in stdout."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            f.write(code)
            fname = f.name
        try:
            result = subprocess.run(
                cmd + [fname],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return result.returncode == 0 and "PASS" in result.stdout
        except subprocess.TimeoutExpired:
            return False
        except FileNotFoundError:
            # Compiler/interpreter not installed
            return False
        finally:
            try:
                os.unlink(fname)
            except OSError:
                pass

    def _run_compiled(
        self, source: str, source_suffix: str,
        compile_cmd_fn,  # callable(src_path, bin_path) -> list[str]
        run_cmd_fn,      # callable(bin_path) -> list[str]
    ) -> bool:
        """Compile then run. Returns True if compilation and execution both succeed."""
        import tempfile as _tf
        with _tf.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, f"solution{source_suffix}")
            bin_path = os.path.join(tmpdir, "solution_bin")
            with open(src_path, "w") as f:
                f.write(source)
            # Compile
            try:
                compile_result = subprocess.run(
                    compile_cmd_fn(src_path, bin_path),
                    capture_output=True,
                    timeout=30,
                )
                if compile_result.returncode != 0:
                    return False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
            # Run
            try:
                run_result = subprocess.run(
                    run_cmd_fn(bin_path),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                return run_result.returncode == 0 and "PASS" in run_result.stdout
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False

    def _run_python(self, solution: str, test_code: str) -> bool:
        harness = _PYTHON_HARNESS.format(
            code_indented=_indent(solution, 4),
            tests_indented=_indent(test_code, 4),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(harness)
            fname = f.name
        try:
            result = subprocess.run(
                [sys.executable, fname],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return result.returncode == 0 and "PASS" in result.stdout
        except subprocess.TimeoutExpired:
            return False
        finally:
            try:
                os.unlink(fname)
            except OSError:
                pass

    def _run_cpp(self, solution: str, test_code: str) -> bool:
        source = _CPP_HARNESS.format(
            code=solution,
            tests_indented=_indent(test_code, 4),
        )
        return self._run_compiled(
            source, ".cpp",
            compile_cmd_fn=lambda src, out: ["g++", "-O2", "-std=c++17", "-o", out, src],
            run_cmd_fn=lambda out: [out],
        )

    def _run_java(self, solution: str, test_code: str) -> bool:
        import tempfile as _tf
        source = _JAVA_HARNESS.format(
            code_indented=_indent(solution, 4),
            tests_indented=_indent(test_code, 8),
        )
        with _tf.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "Solution.java")
            with open(src_path, "w") as f:
                f.write(source)
            try:
                compile_r = subprocess.run(
                    ["javac", src_path], capture_output=True, timeout=30,
                )
                if compile_r.returncode != 0:
                    return False
                run_r = subprocess.run(
                    ["java", "-cp", tmpdir, "Solution"],
                    capture_output=True, text=True, timeout=self.timeout,
                )
                return run_r.returncode == 0 and "PASS" in run_r.stdout
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False

    def _run_js(self, solution: str, test_code: str) -> bool:
        source = _JS_HARNESS.format(
            code=solution,
            tests_indented=_indent(test_code, 4),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(source)
            fname = f.name
        try:
            result = subprocess.run(
                ["node", fname],
                capture_output=True, text=True, timeout=self.timeout,
            )
            return result.returncode == 0 and "PASS" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        finally:
            try:
                os.unlink(fname)
            except OSError:
                pass

    def _run_go(self, solution: str, test_code: str) -> bool:
        source = _GO_HARNESS.format(
            code=solution,
            tests_indented=_indent(test_code, 4),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False) as f:
            f.write(source)
            fname = f.name
        try:
            result = subprocess.run(
                ["go", "run", fname],
                capture_output=True, text=True, timeout=self.timeout,
            )
            return result.returncode == 0 and "PASS" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        finally:
            try:
                os.unlink(fname)
            except OSError:
                pass

    def _run_kotlin(self, solution: str, test_code: str) -> bool:
        source = _KOTLIN_HARNESS.format(
            code=solution,
            tests_indented=_indent(test_code, 8),
        )
        return self._run_compiled(
            source, ".kt",
            compile_cmd_fn=lambda src, out: ["kotlinc", src, "-include-runtime", "-d", out + ".jar"],
            run_cmd_fn=lambda out: ["java", "-jar", out + ".jar"],
        )


# ── Prompt helpers ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert {language} programmer. Think carefully step by step before writing code.
Use <think> tags to show your reasoning, then provide the final solution.
Format:
<think>
[your chain-of-thought reasoning here]
</think>
```{lang_tag}
[your solution here]
```"""

_LANG_TAGS = {
    "python": "python",
    "cpp": "cpp",
    "c++": "cpp",
    "java": "java",
    "javascript": "javascript",
    "js": "javascript",
    "go": "go",
    "kotlin": "kotlin",
}


def _build_prompt(prompt: str, test_code: str, language: str) -> str:
    lang_tag = _LANG_TAGS.get(language.lower(), language.lower())
    system = SYSTEM_PROMPT_TEMPLATE.format(
        language=language.title(),
        lang_tag=lang_tag,
    )
    user_msg = prompt
    if test_code and test_code.strip():
        user_msg += f"\n\nYour solution must pass these tests:\n```{lang_tag}\n{test_code}\n```"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _parse_response(text: str, language: str = "python") -> tuple[str, str]:
    """Extract (thinking, solution) from model response."""
    think_m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    thinking = think_m.group(1).strip() if think_m else ""

    lang_tag = _LANG_TAGS.get(language.lower(), language.lower())
    # Try language-specific fence first
    code_m = re.search(rf"```{lang_tag}\s*(.*?)```", text, re.DOTALL)
    if not code_m:
        # Generic fence
        code_m = re.search(r"```\w*\s*(.*?)```", text, re.DOTALL)
    if not code_m:
        # No fence — take everything after </think>
        if think_m:
            after = text[think_m.end():].strip()
            solution = after if after else ""
        else:
            solution = text.strip()
    else:
        solution = code_m.group(1).strip()

    return thinking, solution


# ── Main inference engine ─────────────────────────────────────────────────────

class CodingSSMInference:
    """
    Test-time compute inference engine for CodingSSM.

    Generates up to N candidate solutions, executes each against provided tests,
    and returns the first passing solution (or the best attempt if none pass).
    Optionally uses a VerifierModel to rank solutions when no test_code is given.
    """

    def __init__(
        self,
        model: CodingSSM,
        tokenizer,
        device: torch.device,
        exec_timeout: int = 10,
        verifier=None,  # Optional[VerifierModel]
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.executor = MultilingualExecutor(timeout=exec_timeout)
        self.verifier = verifier

        self._eos_id = getattr(tokenizer, "eos_token_id", None)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: str = None,
        exec_timeout: int = 10,
        verifier=None,
    ) -> "CodingSSMInference":
        """
        Load a CodingSSM checkpoint and return a ready-to-use inference engine.

        Args:
            path:         Path to .pt checkpoint (GRPO or SFT).
            device:       'cpu', 'cuda', 'cuda:0', etc. Auto-detects if None.
            exec_timeout: Seconds allowed per test execution.
            verifier:     Optional VerifierModel for ranking when no test_code.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load model
        model_cfg = ModelConfig700M()
        model = CodingSSM(model_cfg)

        ckpt = torch.load(path, map_location=dev, weights_only=False)
        if isinstance(ckpt, dict):
            state = ckpt.get("model_state", ckpt.get("model", ckpt))
        else:
            state = ckpt
        model.load_state_dict(state, strict=False)
        model.to(dev)
        model.eval()

        print(
            f"[CodingSSMInference] Loaded {sum(p.numel() for p in model.parameters())/1e9:.2f}B "
            f"params from {path} on {dev}",
            file=sys.stderr,
        )

        return cls(model, tokenizer, dev, exec_timeout=exec_timeout, verifier=verifier)

    # ── Core solve ─────────────────────────────────────────────────────────────

    def solve(
        self,
        prompt: str,
        test_code: str = "",
        language: str = "python",
        n_samples: int = 8,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
    ) -> SolveResult:
        """
        Generate up to n_samples solutions, returning the first that passes tests.

        If no test_code is provided:
          - Uses the verifier (if available) to rank solutions by predicted correctness.
          - Otherwise returns the solution with the longest reasoning trace.

        Args:
            prompt:         The coding problem description.
            test_code:      Unit tests / assertions to verify solutions against.
            language:       Programming language ('python', 'cpp', 'java', 'js', 'go', 'kotlin').
            n_samples:      Maximum number of candidate solutions to generate.
            temperature:    Sampling temperature (higher = more diverse).
            max_new_tokens: Maximum tokens to generate per sample.

        Returns:
            SolveResult with the best solution found.
        """
        candidates = []
        n_passed = 0

        for i in range(n_samples):
            # Slightly increase temperature per attempt for diversity
            sample_temp = min(1.4, temperature + 0.05 * i)
            full_prompt = _build_prompt(prompt, test_code, language)

            raw = self._generate_one(full_prompt, sample_temp, max_new_tokens)
            thinking, solution = _parse_response(raw, language)

            if not solution:
                continue

            if test_code.strip():
                passed = self.executor.execute(solution, test_code, language)
                if passed:
                    n_passed += 1
            else:
                passed = None

            candidates.append({
                "thinking": thinking,
                "solution": solution,
                "passed": passed,
                "raw": raw,
                "attempt": i + 1,
            })

            if passed:
                # Early exit — found a verified solution
                break

        if not candidates:
            return SolveResult(
                solution="", thinking="", passed=False,
                n_tried=n_samples, pass_rate=0.0, language=language,
            )

        best = self._select_best(candidates, prompt, test_code)
        total_tried = len(candidates)
        pass_rate = n_passed / total_tried if total_tried > 0 else 0.0

        return SolveResult(
            solution=best["solution"],
            thinking=best["thinking"],
            passed=best["passed"],
            n_tried=total_tried,
            pass_rate=pass_rate,
            language=language,
        )

    def solve_batch(
        self,
        problems: list[dict],
        n_samples: int = 8,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
    ) -> list[SolveResult]:
        """
        Solve a batch of problems sequentially.

        Each element of problems should be a dict with keys:
            prompt    (str)  — required
            test_code (str)  — optional
            language  (str)  — optional, defaults to 'python'

        Returns:
            List of SolveResult, one per problem.
        """
        results = []
        for i, prob in enumerate(problems):
            print(f"[solve_batch] problem {i+1}/{len(problems)}", file=sys.stderr)
            result = self.solve(
                prompt=prob["prompt"],
                test_code=prob.get("test_code", ""),
                language=prob.get("language", "python"),
                n_samples=n_samples,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            results.append(result)
        return results

    def eval_humaneval_x(
        self,
        n_samples: int = 8,
        languages: list[str] = None,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
        dataset_path: str = None,
    ) -> dict[str, float]:
        """
        Evaluate on HumanEval-X (multilingual HumanEval).

        Loads problems from:
          1. dataset_path if provided (JSONL with 'prompt', 'test_code', 'language' fields).
          2. Hugging Face datasets: 'THUDM/humaneval-x' if available.

        Prints pass@1 and pass@n per language and returns a dict of results.

        Args:
            n_samples:    Number of samples per problem (pass@n).
            languages:    Languages to evaluate. None = all available.
            temperature:  Sampling temperature.
            max_new_tokens: Max new tokens per sample.
            dataset_path: Optional local JSONL dataset path.

        Returns:
            Dict mapping language -> pass@n rate (0.0–1.0).
        """
        problems = self._load_humaneval_x(dataset_path, languages)
        if not problems:
            print("[eval_humaneval_x] No problems loaded.", file=sys.stderr)
            return {}

        # Group by language
        by_lang: dict[str, list] = {}
        for p in problems:
            lang = p.get("language", "python")
            by_lang.setdefault(lang, []).append(p)

        results_by_lang = {}
        for lang, lang_problems in by_lang.items():
            n_pass = 0
            n_total = len(lang_problems)
            print(f"[eval_humaneval_x] {lang}: {n_total} problems", file=sys.stderr)

            for i, prob in enumerate(lang_problems):
                result = self.solve(
                    prompt=prob["prompt"],
                    test_code=prob.get("test_code", ""),
                    language=lang,
                    n_samples=n_samples,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                if result.passed:
                    n_pass += 1
                pass_at_n = n_pass / (i + 1)
                print(
                    f"  [{i+1}/{n_total}] passed={result.passed} "
                    f"n_tried={result.n_tried} running_pass@{n_samples}={pass_at_n:.3f}",
                    file=sys.stderr,
                )

            rate = n_pass / n_total if n_total > 0 else 0.0
            results_by_lang[lang] = rate
            print(f"[eval_humaneval_x] {lang} pass@{n_samples} = {rate:.3f} ({n_pass}/{n_total})")

        # Summary
        print("\n=== HumanEval-X Results ===")
        for lang, rate in results_by_lang.items():
            print(f"  {lang:12s}: {rate*100:.1f}%")

        return results_by_lang

    # ── Internals ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _generate_one(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Tokenize prompt, run model.generate(), decode output."""
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        out_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            eos_token_id=self._eos_id,
        )  # (1, new_tokens)

        return self.tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)

    def _select_best(self, candidates: list[dict], prompt: str, test_code: str) -> dict:
        """
        Pick the best candidate from a list.

        Priority:
          1. If test_code provided: pick first passing candidate (already handled by
             early-exit in solve(), but checked here as a safety net).
          2. If verifier available and no test_code: rank by verifier score.
          3. Fallback: longest thinking trace.
        """
        # Prefer passing solution
        passing = [c for c in candidates if c["passed"]]
        if passing:
            return passing[0]

        # No passing solution — use verifier if available and no test_code
        if self.verifier is not None and not test_code.strip():
            try:
                best_score = -1.0
                best_cand = candidates[0]
                for cand in candidates:
                    score = self.verifier.score(prompt, cand["solution"], self.tokenizer)
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                return best_cand
            except Exception:
                pass

        # Fallback: longest thinking (more deliberate reasoning is usually better)
        return max(candidates, key=lambda c: len(c["thinking"]))

    def _load_humaneval_x(
        self,
        dataset_path: Optional[str],
        languages: Optional[list[str]],
    ) -> list[dict]:
        """Load HumanEval-X problems from local JSONL or HF datasets."""
        problems = []

        if dataset_path:
            path = Path(dataset_path)
            if path.exists():
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            problems.append(json.loads(line))
            else:
                print(f"[eval_humaneval_x] dataset_path not found: {dataset_path}", file=sys.stderr)
        else:
            # Try to load from Hugging Face datasets
            try:
                from datasets import load_dataset  # type: ignore
                target_langs = languages or ["python", "cpp", "java", "javascript", "go"]
                for lang in target_langs:
                    ds = load_dataset("THUDM/humaneval-x", lang, split="test", trust_remote_code=True)
                    for item in ds:
                        problems.append({
                            "prompt": item.get("prompt", item.get("declaration", "")),
                            "test_code": item.get("test", ""),
                            "language": lang,
                            "task_id": item.get("task_id", ""),
                        })
            except Exception as e:
                print(f"[eval_humaneval_x] Could not load HumanEval-X: {e}", file=sys.stderr)

        if languages:
            problems = [p for p in problems if p.get("language", "python") in languages]

        return problems


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CodingSSM test-time compute inference"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--prompt", default=None, help="Problem prompt (interactive if omitted)")
    parser.add_argument("--test-code", default="", help="Test assertions to verify solution")
    parser.add_argument("--language", default="python", help="Programming language")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--eval-humaneval-x", action="store_true",
        help="Run HumanEval-X benchmark instead of single solve",
    )
    parser.add_argument("--humaneval-dataset", default=None, help="Local JSONL path for HumanEval-X")
    parser.add_argument("--languages", nargs="+", default=None, help="Languages for HumanEval-X eval")
    args = parser.parse_args()

    engine = CodingSSMInference.from_checkpoint(
        args.checkpoint, device=args.device,
    )

    if args.eval_humaneval_x:
        engine.eval_humaneval_x(
            n_samples=args.n_samples,
            languages=args.languages,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            dataset_path=args.humaneval_dataset,
        )
        return

    prompt = args.prompt
    if not prompt:
        print("Enter problem (Ctrl+D to finish):")
        prompt = sys.stdin.read().strip()

    result = engine.solve(
        prompt=prompt,
        test_code=args.test_code,
        language=args.language,
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\n=== Result ===")
    print(f"Passed:    {result.passed}")
    print(f"Tried:     {result.n_tried}/{args.n_samples}")
    print(f"Pass rate: {result.pass_rate*100:.1f}%")
    print(f"Language:  {result.language}")
    if result.thinking:
        print(f"\n--- Thinking ---\n{result.thinking}")
    print(f"\n--- Solution ---\n{result.solution}")


if __name__ == "__main__":
    main()
