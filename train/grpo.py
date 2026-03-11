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
        --model-size 700m
"""

import argparse
import json
import os
import re
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
                # Only keep records with test code (needed for verifiable reward)
                if rec.get("test_code") or rec.get("source") == "codealpaca":
                    self.records.append(rec)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


# ─── Code execution reward ────────────────────────────────────────────────────

EXEC_HARNESS = """\
import sys, traceback
try:
{code_indented}
{test_indented}
    print("PASS")
except Exception as e:
    print(f"FAIL: {{e}}")
    sys.exit(1)
"""


def indent(s: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in s.splitlines())


def execute_solution(solution: str, test_code: str, timeout: int = 10) -> bool:
    """Run solution + tests in a subprocess. Returns True if all tests pass."""
    if not test_code.strip():
        # No test harness — reward based on format only
        return False

    harness = EXEC_HARNESS.format(
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
        passed = result.returncode == 0 and "PASS" in result.stdout
        return passed
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def compute_reward(
    solution: str,
    thinking: str,
    test_code: str,
    cfg: GRPOConfig,
    epichat_rag=None,
    problem_prompt: str = "",
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
        passed = execute_solution(solution, test_code, timeout=cfg.exec_timeout)
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


def build_prompt(problem: dict) -> str:
    user_msg = problem["prompt"]
    if problem.get("test_code"):
        user_msg += f"\n\nYour solution must pass these tests:\n```python\n{problem['test_code']}\n```"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_response(text: str) -> tuple[str, str]:
    think_m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    thinking = think_m.group(1).strip() if think_m else ""
    code_m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
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
    B, L_prompt = prompt_ids.shape

    for _ in range(cfg.group_size):
        generated = []
        logprobs = []

        # Reset KV/SSM cache via fresh forward
        input_ids = prompt_ids.clone()
        ssm_states = None
        kv_cache = None

        for step in range(cfg.max_new_tokens):
            logits, ssm_states, kv_cache = model(
                input_ids if step == 0 else input_ids[:, -1:],
                ssm_states=ssm_states,
                kv_cache=kv_cache,
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

    for i, (gen_ids, adv) in enumerate(zip(rollout_ids, advantages)):
        if len(gen_ids) == 0:
            continue

        # Build full sequence: [prompt | generated]
        full_ids = torch.cat([prompt_ids[0], gen_ids.to(device)]).unsqueeze(0)  # (1, L)

        # Forward pass (current policy)
        logits, _, _ = model(full_ids)
        L_prompt = prompt_ids.shape[1]

        # Slice generated portion
        gen_logits = logits[:, L_prompt - 1 : L_prompt - 1 + len(gen_ids), :]  # (1, L_gen, V)
        gen_logprobs = torch.log_softmax(gen_logits, dim=-1)
        token_lp = gen_logprobs[0, torch.arange(len(gen_ids)), gen_ids.to(device)]  # (L_gen,)
        seq_lp = token_lp.mean()  # mean over tokens

        # Policy gradient term
        pg_loss = -adv * seq_lp
        total_loss = total_loss + pg_loss

        # KL penalty against reference model
        with torch.no_grad():
            ref_logits, _, _ = ref_model(full_ids)
            ref_logprobs = torch.log_softmax(
                ref_logits[:, L_prompt - 1 : L_prompt - 1 + len(gen_ids), :], dim=-1
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

        output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = Adafactor(
            model.parameters(),
            lr=cfg.lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

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
        prompt_text = build_prompt(problem)
        ids = self.tokenizer.encode(prompt_text, return_tensors="pt")
        return ids  # (1, L)

    def train_step(self, problem: dict) -> dict:
        prompt_ids = self._tokenize_prompt(problem)

        # 1. Sample G rollouts
        rollout_ids, _ = sample_rollouts(
            self.model, self.tokenizer, prompt_ids, self.cfg, self.device
        )

        # 2. Compute rewards
        rewards = []
        for gen_ids in rollout_ids:
            text = self.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=False)
            thinking, solution = parse_response(text)
            r = compute_reward(
                solution, thinking, problem.get("test_code", ""),
                self.cfg, self.epichat_rag, problem.get("prompt", "")
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

        # Keep only last 3 checkpoints (not best)
        ckpts = sorted(self.output_dir.glob("grpo_step_*.pt"), key=lambda p: p.stat().st_mtime)
        for old in ckpts[:-3]:
            old.unlink()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
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
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}", flush=True)

    model_cfg = ModelConfig700M() if args.model_size == "700m" else ModelConfig3B()
    grpo_cfg = GRPOConfig(
        group_size=args.group_size,
        lr=args.lr,
        max_steps=args.max_steps,
        kl_coeff=args.kl_coeff,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build model
    model = CodingSSM(model_cfg).to(device)
    print(f"[INFO] Model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded checkpoint: {args.checkpoint}", flush=True)

    # Reference model (frozen copy)
    ref_model = CodingSSM(model_cfg).to(device)
    ref_model.load_state_dict(model.state_dict())
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    # Enable gradient checkpointing on training model
    model.enable_gradient_checkpointing()

    dataset = GRPODataset(args.traces)
    print(f"[INFO] Dataset: {len(dataset)} problems", flush=True)

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
    )
    trainer.run()


if __name__ == "__main__":
    main()
