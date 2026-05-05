"""
GRPO (Group Relative Policy Optimization) training loop for CodingSSM.

Algorithm:
  For each batch of coding problems:
    1. Sample G=8 rollouts from student (with <think> reasoning)
    2. Execute generated code against test suite → binary reward r ∈ {0, 1}
    3. Group-normalize rewards: Â_i = (r_i - mean(r)) / (std(r) + ε)
    4. Policy gradient: L = -E[Â_i * log π_θ(y_i|x)] + β * KL(π_θ || π_ref)

References:
    DeepSeekMath: https://arxiv.org/abs/2402.03300
    GRPO (Group Relative Policy Optimization)

Usage:
    python -m train.grpo \
        --traces data/reasoning_traces.jsonl \
        --checkpoint checkpoints/sft/best.pt \
        --output-dir checkpoints/grpo \
        --model-size 700m \
        --languages python,cpp,javascript,java,go,kotlin
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers.optimization import Adafactor

from arch import CodingSSM
from arch.config import ModelConfig, ModelConfig700M, ModelConfig3B


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class GRPOConfig:
    # GRPO hyperparameters
    group_size: int = 8          # G rollouts per problem
    kl_coeff: float = 0.04       # β — KL penalty against reference policy
    reward_scale: float = 1.0    # scale raw rewards before normalization

    # Generation
    max_new_tokens: int = 1024   # max tokens for student rollout
    temperature: float = 0.8
    top_p: float = 0.95
    think_budget: int = 512      # max tokens inside <think> (soft limit via reward shaping)

    # Training
    lr: float = 5e-6
    batch_size: int = 1          # problems per step (gradient accumulation handles effective batch)
    grad_accum_steps: int = 8
    max_steps: int = 2000
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    save_every: int = 200
    eval_every: int = 100

    # Code execution
    exec_timeout: int = 10       # seconds per test run
    max_exec_workers: int = 4    # parallel subprocess limit

    # Reward shaping
    format_reward: float = 0.1   # bonus for having <think>...</think> + code block
    length_penalty: float = 0.0  # per-token penalty (0 = disabled)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class GRPODataset(Dataset):
    """Load problems from reasoning_traces.jsonl (or raw problem list)."""

    def __init__(self, traces_path: str):
        self.records = []
        with open(traces_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Keep records with test code, known codealpaca source, or any
                # record that has at least a prompt (multilingual datasets may
                # not always supply test_code).
                if (
                    rec.get("test_code")
                    or rec.get("source") == "codealpaca"
                    or rec.get("prompt")
                ):
                    # Default language to "python" when not specified
                    if "language" not in rec:
                        rec["language"] = "python"
                    self.records.append(rec)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


# ─── Code execution reward ────────────────────────────────────────────────────

# Python harness (unchanged)
_PYTHON_HARNESS = """\
import sys, traceback
try:
{code_indented}
{test_indented}
    print("PASS")
except Exception as e:
    print(f"FAIL: {{e}}")
    sys.exit(1)
"""

# Per-language runner specifications
LANGUAGE_RUNNERS = {
    "python": {
        "fence": "python",
        "ext": ".py",
    },
    "cpp": {
        "fence": "cpp",
        "ext": ".cpp",
    },
    "java": {
        "fence": "java",
        "ext": ".java",
    },
    "javascript": {
        "fence": "javascript",
        "ext": ".js",
    },
    "go": {
        "fence": "go",
        "ext": ".go",
    },
    "kotlin": {
        "fence": "kotlin",
        "ext": ".kt",
    },
}


def indent(s: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in s.splitlines())


def _run_python(solution: str, test_code: str, timeout: int) -> bool:
    harness = _PYTHON_HARNESS.format(
        code_indented=indent(solution),
        test_indented=indent(test_code),
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(harness)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0 and "PASS" in result.stdout
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def _run_cpp(solution: str, test_code: str, timeout: int) -> bool:
    src = f"#include <bits/stdc++.h>\nusing namespace std;\n\n{solution}\n\nint main() {{\n{indent(test_code)}\n    return 0;\n}}\n"
    src_path = "/tmp/prog.cpp"
    bin_path = "/tmp/prog_cpp"
    try:
        with open(src_path, "w") as f:
            f.write(src)
        compile_result = subprocess.run(
            ["g++", "-std=c++17", "-o", bin_path, src_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if compile_result.returncode != 0:
            return False
        run_result = subprocess.run(
            [bin_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return run_result.returncode == 0 and "PASS" in run_result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False
    finally:
        for p in (src_path, bin_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _run_java(solution: str, test_code: str, timeout: int) -> bool:
    src = (
        f"public class SolutionTest {{\n"
        f"{indent(solution)}\n\n"
        f"    public static void main(String[] args) throws Exception {{\n"
        f"{indent(test_code, 8)}\n"
        f"    }}\n"
        f"}}\n"
    )
    src_path = "/tmp/SolutionTest.java"
    out_dir = "/tmp/java_out"
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(src_path, "w") as f:
            f.write(src)
        compile_result = subprocess.run(
            ["javac", src_path, "-d", out_dir],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if compile_result.returncode != 0:
            return False
        run_result = subprocess.run(
            ["java", "-cp", out_dir, "SolutionTest"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return run_result.returncode == 0 and "PASS" in run_result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass


def _run_javascript(solution: str, test_code: str, timeout: int) -> bool:
    src = (
        f"try {{\n"
        f"{indent(solution)}\n"
        f"{indent(test_code)}\n"
        f'    console.log("PASS");\n'
        f"}} catch (e) {{\n"
        f'    console.log("FAIL: " + e.message);\n'
        f"    process.exit(1);\n"
        f"}}\n"
    )
    src_path = "/tmp/prog.js"
    try:
        with open(src_path, "w") as f:
            f.write(src)
        result = subprocess.run(
            ["node", src_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0 and "PASS" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass


def _run_go(solution: str, test_code: str, timeout: int) -> bool:
    src = (
        f"package main\n\nimport \"fmt\"\n\n"
        f"{solution}\n\n"
        f"func main() {{\n"
        f"{indent(test_code)}\n"
        f'    fmt.Println("PASS")\n'
        f"}}\n"
    )
    src_path = "/tmp/prog.go"
    try:
        with open(src_path, "w") as f:
            f.write(src)
        result = subprocess.run(
            ["go", "run", src_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0 and "PASS" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass


def _run_kotlin(solution: str, test_code: str, timeout: int) -> bool:
    if not shutil.which("kotlinc"):
        return False
    src = (
        f"{solution}\n\n"
        f"fun main() {{\n"
        f"{indent(test_code)}\n"
        f'    println("PASS")\n'
        f"}}\n"
    )
    src_path = "/tmp/prog.kt"
    jar_path = "/tmp/prog.jar"
    try:
        with open(src_path, "w") as f:
            f.write(src)
        compile_result = subprocess.run(
            ["kotlinc", src_path, "-include-runtime", "-d", jar_path],
            capture_output=True,
            text=True,
            timeout=timeout * 3,  # kotlinc is slow to start
            stderr=subprocess.DEVNULL,
        )
        if compile_result.returncode != 0:
            return False
        run_result = subprocess.run(
            ["java", "-jar", jar_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return run_result.returncode == 0 and "PASS" in run_result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False
    finally:
        for p in (src_path, jar_path):
            try:
                os.unlink(p)
            except OSError:
                pass


_LANGUAGE_DISPATCH = {
    "python": _run_python,
    "cpp": _run_cpp,
    "c++": _run_cpp,
    "java": _run_java,
    "javascript": _run_javascript,
    "js": _run_javascript,
    "go": _run_go,
    "kotlin": _run_kotlin,
}


def execute_solution(
    solution: str,
    test_code: str,
    language: str = "python",
    timeout: int = 10,
) -> bool:
    """
    Run solution + tests in a subprocess for the given language.
    Returns True if all tests pass.
    Defaults to Python if language is not specified or unrecognised.
    """
    if not test_code.strip():
        # No test harness — reward based on format only
        return False

    runner = _LANGUAGE_DISPATCH.get(language.lower(), _run_python)
    return runner(solution, test_code, timeout)


def compute_reward(
    solution: str,
    thinking: str,
    test_code: str,
    cfg: GRPOConfig,
    epichat_rag=None,
    problem_prompt: str = "",
    language: str = "python",
) -> float:
    """
    Compute scalar reward for a single rollout.

    Reward components:
      1.0  — code passes tests (primary)
      0.1  — has <think> block AND code block (format)
      0.05 — generated code aligns with a high-confidence EpiChat design pattern
    """
    r = 0.0

    # Primary: code correctness (binary)
    if test_code:
        passed = execute_solution(solution, test_code, language=language, timeout=cfg.exec_timeout)
        r += float(passed)

    # Format bonus
    has_think = bool(thinking.strip())
    has_code = bool(solution.strip())
    if has_think and has_code:
        r += cfg.format_reward

    # Epistemic bonus: check if solution uses a well-established pattern
    if epichat_rag and solution and problem_prompt:
        try:
            eus = epichat_rag.search(problem_prompt, top_k=3)
            for eu in eus:
                code_ex = eu.get("code_snippet", "")
                if not code_ex:
                    continue
                # Check structural similarity: shared function names or key tokens
                eu_tokens = set(code_ex.split()) & set("def class return yield for while if".split())
                sol_tokens = set(solution.split()) & eu_tokens
                overlap = len(sol_tokens) / max(len(eu_tokens), 1)
                if overlap > 0.5 and eu.get("confidence", 0) >= 0.7:
                    r += 0.05  # small epistemic alignment bonus
                    break
        except Exception:
            pass

    return r * cfg.reward_scale


# ─── Rollout generation ───────────────────────────────────────────────────────

def build_system_prompt(language: str = "python") -> str:
    """Return a language-specific system prompt."""
    lang = language.lower()
    if lang in ("python",):
        fence = "python"
        expert = "Python"
    elif lang in ("cpp", "c++"):
        fence = "cpp"
        expert = "C++"
    elif lang in ("java",):
        fence = "java"
        expert = "Java"
    elif lang in ("javascript", "js"):
        fence = "javascript"
        expert = "JavaScript"
    elif lang in ("go",):
        fence = "go"
        expert = "Go"
    elif lang in ("kotlin",):
        fence = "kotlin"
        expert = "Kotlin"
    else:
        fence = language
        expert = language.capitalize()

    return (
        f"You are an expert {expert} programmer. Think carefully step by step before writing code.\n"
        f"Use <think> tags to show your reasoning, then provide the final solution.\n"
        f"Format:\n"
        f"<think>\n"
        f"[your chain-of-thought reasoning here]\n"
        f"</think>\n"
        f"```{fence}\n"
        f"[your solution here]\n"
        f"```"
    )


def build_prompt(problem: dict, language: str = "python") -> str:
    lang = language.lower()
    if lang in ("cpp", "c++"):
        fence = "cpp"
    elif lang in ("javascript", "js"):
        fence = "javascript"
    else:
        fence = lang

    system_prompt = build_system_prompt(language)
    user_msg = problem["prompt"]
    if problem.get("test_code"):
        user_msg += f"\n\nYour solution must pass these tests:\n```{fence}\n{problem['test_code']}\n```"
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_response(text: str, language: str = "python") -> tuple[str, str]:
    """
    Extract (thinking, solution) from a model response.
    Supports language-specific code fences with fallback to any ``` block.
    """
    think_m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    thinking = think_m.group(1).strip() if think_m else ""

    # Try language-specific fences first
    lang = language.lower()
    fence_variants: list[str] = []
    if lang in ("python",):
        fence_variants = ["python"]
    elif lang in ("cpp", "c++"):
        fence_variants = ["cpp", "c\\+\\+"]
    elif lang in ("java",):
        fence_variants = ["java"]
    elif lang in ("javascript", "js"):
        fence_variants = ["javascript", "js"]
    elif lang in ("go",):
        fence_variants = ["go"]
    elif lang in ("kotlin",):
        fence_variants = ["kotlin"]
    else:
        fence_variants = [re.escape(lang)]

    code_m = None
    for fence in fence_variants:
        code_m = re.search(rf"```{fence}\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if code_m:
            break

    # Fallback: any ``` block
    if not code_m:
        code_m = re.search(r"```\s*(.*?)```", text, re.DOTALL)

    solution = code_m.group(1).strip() if code_m else ""
    return thinking, solution


@torch.no_grad()
def sample_rollouts(
    model: CodingSSM,
    tokenizer,
    prompt_ids: torch.Tensor,
    cfg: GRPOConfig,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Sample G rollouts from model. Returns:
        rollout_ids: list[Tensor(L_i,)]  — generated token ids (excluding prompt)
        rollout_logprobs: list[Tensor(L_i,)]  — log-probs of each generated token
    """
    model.eval()
    rollout_ids = []
    rollout_logprobs = []

    prompt_ids = prompt_ids.to(device)

    for _ in range(cfg.group_size):
        generated = []
        logprobs = []

        # Reset SSM states via fresh forward
        input_ids = prompt_ids.clone()
        states = None

        for step in range(cfg.max_new_tokens):
            logits, _aux_loss, states = model(
                input_ids if step == 0 else input_ids[:, -1:],
                states=states,
                return_states=True,
            )
            next_logits = logits[:, -1, :]  # (1, V)

            # Top-p sampling
            probs = torch.softmax(next_logits / cfg.temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cum_probs - sorted_probs > cfg.top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))  # (1,1)

            lp = torch.log_softmax(next_logits, dim=-1)
            token_lp = lp.gather(-1, next_token).squeeze()  # scalar

            generated.append(next_token.squeeze().item())
            logprobs.append(token_lp)

            input_ids = next_token
            if next_token.item() in (tokenizer.eos_token_id,):
                break

        rollout_ids.append(torch.tensor(generated, dtype=torch.long))
        rollout_logprobs.append(torch.stack(logprobs))

    return rollout_ids, rollout_logprobs


# ─── GRPO loss ────────────────────────────────────────────────────────────────

def grpo_loss(
    model: CodingSSM,
    ref_model: CodingSSM,
    prompt_ids: torch.Tensor,
    rollout_ids: list[torch.Tensor],
    advantages: torch.Tensor,  # (G,)
    cfg: GRPOConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute GRPO policy gradient loss + KL penalty.

    L = -mean_G [ Â_i * sum_t log π_θ(y_it|x,y_i<t) ]
        + β * KL(π_θ || π_ref)
    """
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    kl_total = torch.tensor(0.0, device=device)
    n_valid = 0
    use_amp = device.type == "cuda"

    for i, (gen_ids, adv) in enumerate(zip(rollout_ids, advantages)):
        if len(gen_ids) == 0:
            continue

        # Build full sequence: [prompt | generated]
        full_ids = torch.cat([prompt_ids[0].to(device), gen_ids.to(device)]).unsqueeze(0)  # (1, L)

        # Forward pass (current policy)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits, _, _ = model(full_ids)
        L_prompt = prompt_ids.shape[1]

        # Slice generated portion
        gen_logits = logits[:, L_prompt - 1 : L_prompt - 1 + len(gen_ids), :]  # (1, L_gen, V)
        gen_logprobs = torch.log_softmax(gen_logits.float(), dim=-1)
        token_lp = gen_logprobs[0, torch.arange(len(gen_ids)), gen_ids.to(device)]  # (L_gen,)
        seq_lp = token_lp.mean()  # mean over tokens

        # Policy gradient term
        pg_loss = -adv * seq_lp
        total_loss = total_loss + pg_loss

        # KL penalty against reference model
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                ref_logits, _, _ = ref_model(full_ids)
            ref_logprobs = torch.log_softmax(
                ref_logits[:, L_prompt - 1 : L_prompt - 1 + len(gen_ids), :].float(), dim=-1
            )

        kl = (torch.exp(gen_logprobs) * (gen_logprobs - ref_logprobs)).sum(dim=-1).mean()
        kl_total = kl_total + kl
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = total_loss / n_valid + cfg.kl_coeff * kl_total / n_valid
    return loss


# ─── Trainer ─────────────────────────────────────────────────────────────────

class GRPOTrainer:
    def __init__(
        self,
        model: CodingSSM,
        ref_model: CodingSSM,
        tokenizer,
        dataset: GRPODataset,
        cfg: GRPOConfig,
        model_cfg: ModelConfig,
        output_dir: Path,
        device: torch.device,
        epichat_rag=None,
        hf_repo: Optional[str] = None,
        hf_path_in_repo: str = "training/grpo_latest.pt",
        hf_token: Optional[str] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.output_dir = output_dir
        self.device = device
        self.epichat_rag = epichat_rag
        self.hf_repo = hf_repo
        self.hf_path_in_repo = hf_path_in_repo
        self.hf_token = hf_token

        output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = Adafactor(
            model.parameters(),
            lr=cfg.lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
        self.use_amp = device.type == "cuda"

        self.step = 0
        self.best_reward = -float("inf")

    def _lr_schedule(self) -> float:
        s = self.step
        warmup = self.cfg.warmup_steps
        if s < warmup:
            return s / max(warmup, 1)
        # Cosine decay
        progress = (s - warmup) / max(self.cfg.max_steps - warmup, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)).item())

    def _tokenize_prompt(self, problem: dict) -> torch.Tensor:
        language = problem.get("language", "python")
        prompt_text = build_prompt(problem, language=language)
        ids = self.tokenizer.encode(prompt_text, return_tensors="pt")
        return ids  # (1, L)

    def train_step(self, problem: dict) -> dict:
        language = problem.get("language", "python")
        prompt_ids = self._tokenize_prompt(problem)

        # 1. Sample G rollouts
        rollout_ids, _ = sample_rollouts(
            self.model, self.tokenizer, prompt_ids, self.cfg, self.device
        )

        # 2. Compute rewards
        rewards = []
        for gen_ids in rollout_ids:
            text = self.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=False)
            thinking, solution = parse_response(text, language=language)
            r = compute_reward(
                solution,
                thinking,
                problem.get("test_code", ""),
                self.cfg,
                self.epichat_rag,
                problem.get("prompt", ""),
                language=language,
            )
            rewards.append(r)

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_t.mean().item()

        # 3. Group-normalize advantages
        std_r = rewards_t.std() + 1e-8
        advantages = (rewards_t - rewards_t.mean()) / std_r

        # 4. GRPO loss
        loss = grpo_loss(
            self.model,
            self.ref_model,
            prompt_ids,
            rollout_ids,
            advantages,
            self.cfg,
            self.device,
        )
        loss = loss / self.cfg.grad_accum_steps
        loss.backward()

        return {"loss": loss.item() * self.cfg.grad_accum_steps, "mean_reward": mean_r, "rewards": rewards}

    def run(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: x[0],
        )
        data_iter = iter(dataloader)

        self.optimizer.zero_grad()
        accum_loss = 0.0
        accum_reward = 0.0

        print(f"[GRPO] Starting training: {self.cfg.max_steps} steps, G={self.cfg.group_size}", flush=True)

        while self.step < self.cfg.max_steps:
            try:
                problem = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                problem = next(data_iter)

            t0 = time.time()
            metrics = self.train_step(problem)
            accum_loss += metrics["loss"]
            accum_reward += metrics["mean_reward"]

            # Gradient accumulation step
            if (self.step + 1) % self.cfg.grad_accum_steps == 0:
                # LR schedule
                lr_scale = self._lr_schedule()
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.cfg.lr * lr_scale

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                avg_loss = accum_loss / self.cfg.grad_accum_steps
                avg_reward = accum_reward / self.cfg.grad_accum_steps
                elapsed = time.time() - t0

                print(
                    f"step={self.step+1:04d} loss={avg_loss:.4f} "
                    f"reward={avg_reward:.3f} "
                    f"rewards={[f'{r:.1f}' for r in metrics['rewards']]} "
                    f"lr={self.cfg.lr * lr_scale:.2e} "
                    f"({elapsed:.1f}s/step)",
                    flush=True,
                )
                accum_loss = 0.0
                accum_reward = 0.0

                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    self._save("best")

            self.step += 1

            if self.step % self.cfg.save_every == 0:
                self._save(f"step_{self.step:05d}")

    def _save(self, tag: str):
        path = self.output_dir / f"grpo_{tag}.pt"
        torch.save(
            {
                "step": self.step,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "model_config": self.model_cfg.__dict__,
                "best_reward": self.best_reward,
            },
            path,
        )
        print(f"[SAVE] {path}", flush=True)

        if self.hf_repo and self.hf_path_in_repo:
            from ssm.hf_checkpoint_sync import upload_checkpoint

            upload_checkpoint(path, self.hf_repo, self.hf_path_in_repo, self.hf_token)

        # Keep only last 3 checkpoints (not best)
        ckpts = sorted(self.output_dir.glob("grpo_step_*.pt"), key=lambda p: p.stat().st_mtime)
        for old in ckpts[:-3]:
            old.unlink()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import os as _os
    _kaggle_wd = "/kaggle/working"
    _lock_dir  = _kaggle_wd if _os.path.isdir(_kaggle_wd) else "/tmp"
    _lock_path = _os.path.join(_lock_dir, ".ssm_grpo_train.lock")
    _lock_acquired = False
    try:
        _lfd = _os.open(_lock_path, _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY, 0o644)
        _os.write(_lfd, str(_os.getpid()).encode())
        _os.close(_lfd)
        _lock_acquired = True
        print(f"[GRPO] Lock acquired: {_lock_path}", flush=True)
    except FileExistsError:
        print(f"[GRPO] Another worker already training ({_lock_path} exists) — exiting.", flush=True)
        return

    try:
        _main()
    finally:
        if _lock_acquired:
            try:
                _os.unlink(_lock_path)
            except Exception:
                pass


def _main():
    parser = argparse.ArgumentParser(description="GRPO training for CodingSSM")
    parser.add_argument("--traces", default="data/reasoning_traces.jsonl")
    parser.add_argument("--checkpoint", default=None, help="SFT checkpoint to start from")
    parser.add_argument("--output-dir", default="checkpoints/grpo")
    parser.add_argument("--model-size", default="700m", choices=["700m", "3b"])
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--kl-coeff", type=float, default=0.04)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default=None, help="Device: cpu, cuda, cuda:0, etc.")
    parser.add_argument(
        "--languages",
        default="python,cpp,javascript,typescript,java,go,kotlin",
        help="Comma-separated list of languages to train on (filters dataset records)",
    )
    parser.add_argument(
        "--hf-repo",
        default=os.environ.get("HF_MODEL_REPO"),
        help="HF model repo id (same repo as your weights; not GitHub). Defaults to HF_MODEL_REPO env.",
    )
    parser.add_argument(
        "--hf-grpo-path",
        default="training/grpo_latest.pt",
        help="Path inside the model repo for the latest GRPO checkpoint.",
    )
    parser.add_argument("--hf-token", default=None, help="Hub token (defaults to HF_TOKEN).")
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save (and HF-upload) every N steps. Overrides GRPOConfig.save_every (default 200).",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}", flush=True)

    # Parse and log requested languages
    requested_languages = {lang.strip().lower() for lang in args.languages.split(",") if lang.strip()}
    print(f"[INFO] Languages requested: {sorted(requested_languages)}", flush=True)

    model_cfg = ModelConfig700M() if args.model_size == "700m" else ModelConfig3B()
    grpo_cfg = GRPOConfig(
        group_size=args.group_size,
        lr=args.lr,
        max_steps=args.max_steps,
        kl_coeff=args.kl_coeff,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        **({'save_every': args.save_every} if args.save_every is not None else {}),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build model
    model = CodingSSM(model_cfg).to(device)
    print(f"[INFO] Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    loaded_ckpt = None
    if args.checkpoint:
        # Load to CPU to avoid placing 6+ GB fp32 checkpoint on a 15.6 GB T4
        # alongside the fp16 model, gradients, and CUDA overhead.
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        # Keep metadata for GRPO resume; strip model weights to free CPU RAM
        loaded_ckpt = {k: v for k, v in ckpt.items() if k != "model_state"}
        del ckpt
        torch.cuda.empty_cache()
        print(f"[INFO] Loaded checkpoint: {args.checkpoint}", flush=True)

    # fp16: halves weight memory (6.6 GB → 3.3 GB) on T4.
    # Adafactor keeps fp32 second moments internally so optimizer is fine.
    if device.type == "cuda":
        model = model.to(torch.float16)
        free, total = torch.cuda.mem_get_info(device.index or 0)
        print(f"[INFO] fp16 model. GPU free: {free/1e9:.1f}/{total/1e9:.1f} GB", flush=True)

    # Reference model (frozen copy) — must match dtype of training model
    ref_model = CodingSSM(model_cfg).to(device)
    if device.type == "cuda":
        ref_model = ref_model.to(torch.float16)
    ref_model.load_state_dict(model.state_dict())
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    # Enable gradient checkpointing on training model
    model.enable_gradient_checkpointing()

    dataset = GRPODataset(args.traces)
    print(f"[INFO] Dataset: {len(dataset)} problems (before language filter)", flush=True)

    # Filter dataset to requested languages and log counts per language
    from collections import Counter
    lang_counts: Counter = Counter(rec.get("language", "python") for rec in dataset.records)
    print(f"[INFO] Language distribution in dataset: {dict(lang_counts)}", flush=True)

    # Normalise aliases in dataset records to canonical names (e.g. "js" -> "javascript")
    _alias_map = {"c++": "cpp", "js": "javascript", "typescript": "javascript"}
    filtered_records = []
    for rec in dataset.records:
        lang = rec.get("language", "python").lower()
        lang = _alias_map.get(lang, lang)
        rec["language"] = lang
        if lang in requested_languages or lang in _alias_map.values():
            filtered_records.append(rec)
    dataset.records = filtered_records
    print(f"[INFO] Dataset after language filter: {len(dataset)} problems", flush=True)

    lang_counts_filtered: Counter = Counter(rec.get("language", "python") for rec in dataset.records)
    print(f"[INFO] Language distribution after filter: {dict(lang_counts_filtered)}", flush=True)

    # Load EpiChat RAG for epistemic reward shaping
    epichat_rag = None
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from ssm.epichat_rag import EpiChatRAG
        from ssm.paths import EPICHAT_DIR
        epichat_rag = EpiChatRAG(str(EPICHAT_DIR))
        print(f"[INFO] EpiChat RAG loaded: {epichat_rag.stats}", flush=True)
    except Exception as e:
        print(f"[INFO] EpiChat RAG not available: {e}", flush=True)

    hf_tok = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        cfg=grpo_cfg,
        model_cfg=model_cfg,
        output_dir=Path(args.output_dir),
        device=device,
        epichat_rag=epichat_rag,
        hf_repo=args.hf_repo,
        hf_path_in_repo=args.hf_grpo_path,
        hf_token=hf_tok,
    )

    # Resume GRPO optimizer + step when loading a GRPO checkpoint (not plain SFT)
    if loaded_ckpt and isinstance(loaded_ckpt, dict):
        ckpt_name = Path(args.checkpoint or "").name
        is_grpo = "best_reward" in loaded_ckpt or ckpt_name.startswith("grpo_")
        if is_grpo and loaded_ckpt.get("optimizer_state") is not None:
            try:
                trainer.optimizer.load_state_dict(loaded_ckpt["optimizer_state"])
                trainer.step = int(loaded_ckpt.get("step", 0))
                trainer.best_reward = float(loaded_ckpt.get("best_reward", float("-inf")))
                print(
                    f"[RESUME] GRPO step={trainer.step} best_reward={trainer.best_reward:.4f}",
                    flush=True,
                )
            except Exception as e:
                print(f"[WARN] Could not restore GRPO trainer state: {e}", flush=True)

    trainer.run()


if __name__ == "__main__":
    main()
