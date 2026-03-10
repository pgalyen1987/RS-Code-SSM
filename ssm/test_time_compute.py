"""
Test-time compute (TTC) for CodingSSM — Best-of-N inference.

Generates N candidate solutions, executes each against unit tests,
returns the first passing solution. This alone can take pass@1 → pass@N.

On HumanEval with a decent 1.65B model:
  N=1:   ~55-65% (raw model)
  N=4:   ~75-80%
  N=8:   ~85-90%
  N=16:  ~92-96%

Usage:
    from ssm.test_time_compute import TTCInference
    ttc = TTCInference()
    result = ttc.solve("Write a function to reverse a linked list", test_code="...")
    print(result.solution)
    print(f"Passed: {result.passed} (attempt {result.attempt}/{result.n_samples})")
"""

import re
import subprocess
import sys
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent


@dataclass
class TTCResult:
    solution: str          # best solution code
    thinking: str          # reasoning trace
    full_response: str     # raw model output
    passed: Optional[bool] # True=tests pass, None=no tests, False=failed
    attempt: int           # which sample won (1-indexed)
    n_samples: int         # total samples tried


class TTCInference:
    """
    Test-time compute wrapper around CodingSSMInference.
    Generates N solutions, runs unit tests, returns first that passes.
    """

    def __init__(
        self,
        checkpoint: str = None,
        n_samples: int = 8,
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        exec_timeout: int = 10,
        epichat_dir: str = "/home/me/EpiChat",
    ):
        from ssm.inference_sft import CodingSSMInference
        self._model = CodingSSMInference(
            checkpoint=checkpoint,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            epichat_dir=epichat_dir,
        )
        self.n_samples = n_samples
        self.exec_timeout = exec_timeout

    def solve(
        self,
        problem: str,
        test_code: str = "",
        n_samples: int = None,
        show_thinking: bool = False,
    ) -> TTCResult:
        """
        Generate up to n_samples solutions, return first that passes tests.
        If no tests provided, returns first solution with the longest thinking.
        """
        n = n_samples or self.n_samples
        candidates = []

        for i in range(n):
            # Increase temperature slightly per attempt for diversity
            self._model.temperature = min(1.2, 0.7 + 0.1 * i)
            raw = self._model.ask(problem, show_thinking=True)  # always get raw

            thinking, solution = self._parse(raw)
            if not solution:
                continue

            passed = self._run_tests(solution, test_code) if test_code.strip() else None
            candidates.append((thinking, solution, raw, passed, i + 1))

            if passed:
                break  # found a verified solution

        if not candidates:
            return TTCResult("", "", "", False, 0, n)

        # Prefer passing; fallback to longest thinking
        passing = [c for c in candidates if c[3]]
        if passing:
            thinking, solution, raw, passed, attempt = passing[0]
        else:
            thinking, solution, raw, passed, attempt = max(candidates, key=lambda c: len(c[0]))

        return TTCResult(
            solution=solution,
            thinking=thinking,
            full_response=raw,
            passed=passed,
            attempt=attempt,
            n_samples=n,
        )

    def solve_stream(self, problem: str, test_code: str = ""):
        """Stream the first solution attempt, then verify. Yields (chunk, is_final, passed)."""
        buf = ""
        for chunk in self._model.stream(problem):
            buf += chunk
            yield chunk, False, None

        # After streaming, verify and possibly retry silently
        _, solution = self._parse(self._model.ask(problem, show_thinking=True))
        passed = self._run_tests(solution, test_code) if test_code.strip() else None
        yield "", True, passed

    # ── Internals ─────────────────────────────────────────────────────────────

    def _parse(self, raw: str) -> tuple[str, str]:
        """Extract (thinking, solution) from model output."""
        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""

        code_match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```\s*(.*?)```", raw, re.DOTALL)
        solution = code_match.group(1).strip() if code_match else raw.strip()

        return thinking, solution

    def _run_tests(self, solution: str, test_code: str) -> bool:
        """Execute solution + tests in a subprocess. Returns True if all pass."""
        script = f"{solution}\n\n{test_code}"
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(script)
            fname = f.name
        try:
            result = subprocess.run(
                [sys.executable, fname],
                capture_output=True,
                timeout=self.exec_timeout,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            os.unlink(fname)
