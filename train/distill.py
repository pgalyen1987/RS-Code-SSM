"""
CodingSSM distillation training loop.

Loss function:
  L_total = α * L_rkl + β * L_ce + γ * L_aux

Where:
  L_rkl = Reverse KLD (MiniLLM): KL(student || teacher)
          = sum_t E_{x~p_student}[log p_student(x) - log p_teacher(x)]
          Minimized via the sequence-level REINFORCE / token-level approximation.
          For efficiency we use the token-level approximation:
          L_rkl ≈ -sum_t sum_v p_student(v|ctx) * log(p_teacher(v|ctx) / p_student(v|ctx))

  L_ce  = Cross-entropy against ground-truth labels (standard LM loss)

  L_aux = MoE load-balancing auxiliary loss (sum across all MoE layers)

Training stages:
  Stage 1: CodeAlpaca-20k, α=0.7, β=0.3, γ=0.01, lr=3e-4, 3 epochs
  Stage 2: Evol-Instruct-80k, α=0.7, β=0.3, γ=0.01, lr=1e-4, 2 epochs
  Stage 3: OPSDC self-distillation, α=0.5, β=0.5, γ=0.01, lr=5e-5, 1 epoch

Optimizer: Adafactor (memory-efficient; no stored second moment for large matrices).
  On CPU with 3B params, AdamW requires ~48GB just for optimizer state.
  Adafactor uses factored approximation: ~3-4GB total.
Gradient checkpointing: enabled by default, recomputes activations during backward.
Gradient clipping: 1.0.
"""

import json
import math
import os
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from transformers.optimization import Adafactor
    HAS_ADAFACTOR = True
except ImportError:
    HAS_ADAFACTOR = False

# Optional rich logging
try:
    from rich.console import Console
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None


@dataclass
class TrainConfig:
    # Paths
    output_dir: str = "checkpoints/stage1"
    logit_cache_dir: Optional[str] = None   # offline teacher logits

    # Dataset
    dataset: str = "codealpaca"             # "codealpaca" | "evolinstruct" | "opsdc"
    max_seq_len: int = 2048
    max_examples: Optional[int] = None      # cap for debugging

    # Loss weights
    alpha: float = 0.7    # reverse KLD weight
    beta: float = 0.3     # CE weight
    gamma: float = 0.01   # MoE aux loss weight

    # Training
    epochs: int = 3
    batch_size: int = 1   # 1 on CPU to minimize activation memory
    grad_accum_steps: int = 16  # effective batch = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Memory
    use_adafactor: bool = True          # Adafactor vs AdamW (saves ~36GB optimizer state)
    gradient_checkpointing: bool = True  # recompute activations (saves ~60% activation RAM)

    # KD temperature
    kd_temperature: float = 1.0

    # Checkpointing
    save_every_n_steps: int = 500
    eval_every_n_steps: int = 200
    keep_last_n_checkpoints: int = 3

    # Misc
    seed: int = 42
    log_every_n_steps: int = 10
    tokenizer_name: str = "Qwen/Qwen2.5-Coder-7B"


def reverse_kld_loss(
    student_logits: torch.Tensor,    # (B, L, V)
    teacher_logits: torch.Tensor,    # (B, L, V)
    labels: torch.Tensor,            # (B, L)  -100 for masked positions
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Token-level approximation of reverse KLD: KL(q || p) where q=student, p=teacher.

    KL(q||p) = sum_v q(v) * (log q(v) - log p(v))
             = sum_v q(v) * log q(v) - sum_v q(v) * log p(v)
             = -H(q) + CE(q, p)    (cross-entropy of teacher under student distribution)

    We compute this per-token at positions where labels != -100.
    """
    B, L, V = student_logits.shape
    mask = labels != -100   # (B, L)

    if not mask.any():
        return torch.tensor(0.0, device=student_logits.device)

    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)   # (B, L, V)
    t_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)   # (B, L, V)
    s_probs = s_log_probs.exp()

    # KL(q||p) = sum_v q(v) * (log q(v) - log p(v))
    kl = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)   # (B, L)
    kl = kl * mask.float()
    return kl.sum() / mask.float().sum().clamp(min=1)


def ce_loss(
    student_logits: torch.Tensor,   # (B, L, V)
    labels: torch.Tensor,           # (B, L)
) -> torch.Tensor:
    """Standard cross-entropy language modeling loss."""
    B, L, V = student_logits.shape
    return F.cross_entropy(
        student_logits.view(B * L, V),
        labels.view(B * L),
        ignore_index=-100,
    )


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        device: torch.device = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")

        self.model.to(self.device)

        # Enable gradient checkpointing to halve activation memory
        if config.gradient_checkpointing and hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
            print("Gradient checkpointing enabled")

        # Adafactor: no stored second moment for large matrices → ~3GB vs ~48GB for AdamW
        if config.use_adafactor and HAS_ADAFACTOR:
            self.optimizer = Adafactor(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.learning_rate,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                weight_decay=config.weight_decay,
            )
            print("Optimizer: Adafactor (memory-efficient)")
        else:
            self.optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.95),
            )
            print("Optimizer: AdamW")

        self.global_step = 0
        self.epoch = 0
        self._checkpoint_paths = []

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {n_params:,} ({n_params/1e9:.2f}B)")

    def _get_lr(self, total_steps: int) -> float:
        """Cosine decay with linear warmup."""
        step = self.global_step
        warmup = self.config.warmup_steps
        if step < warmup:
            return self.config.learning_rate * step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return self.config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def train(self, dataloader, total_steps: int = None):
        if total_steps is None:
            total_steps = len(dataloader) * self.config.epochs // self.config.grad_accum_steps

        self.model.train()
        self.optimizer.zero_grad()

        running_loss = 0.0
        running_rkl = 0.0
        running_ce = 0.0
        running_aux = 0.0
        log_steps = 0

        step_in_accum = 0

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)
                has_teacher = "teacher_logits" in batch

                # Forward pass
                logits, aux_loss = self.model(input_ids)  # (B, L, V), scalar

                # CE loss (always)
                l_ce = ce_loss(logits, labels)

                # Reverse KLD (only if teacher logits available)
                if has_teacher:
                    teacher_logits = batch["teacher_logits"].to(self.device)
                    # Align lengths (teacher might be longer/shorter due to generation)
                    min_L = min(logits.shape[1], teacher_logits.shape[1])
                    l_rkl = reverse_kld_loss(
                        logits[:, :min_L],
                        teacher_logits[:, :min_L],
                        labels[:, :min_L],
                        temperature=self.config.kd_temperature,
                    )
                    loss = self.config.alpha * l_rkl + self.config.beta * l_ce
                else:
                    l_rkl = torch.tensor(0.0)
                    loss = l_ce

                loss = loss + self.config.gamma * aux_loss

                # Gradient accumulation
                loss = loss / self.config.grad_accum_steps
                loss.backward()

                running_loss += loss.item() * self.config.grad_accum_steps
                running_rkl  += l_rkl.item()
                running_ce   += l_ce.item()
                running_aux  += aux_loss.item()
                log_steps += 1
                step_in_accum += 1

                if step_in_accum >= self.config.grad_accum_steps:
                    # Gradient clip + optimizer step
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    lr = self._get_lr(total_steps)
                    self._set_lr(lr)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step_in_accum = 0
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.log_every_n_steps == 0:
                        avg_loss = running_loss / log_steps
                        avg_rkl  = running_rkl / log_steps
                        avg_ce   = running_ce / log_steps
                        avg_aux  = running_aux / log_steps
                        self._log(epoch, self.global_step, lr, avg_loss, avg_rkl, avg_ce, avg_aux)
                        running_loss = running_rkl = running_ce = running_aux = log_steps = 0

                    # Checkpoint
                    if self.global_step % self.config.save_every_n_steps == 0:
                        self._save_checkpoint()

        # Final checkpoint
        self._save_checkpoint(tag="final")

    def _log(self, epoch, step, lr, loss, rkl, ce_l, aux):
        msg = (
            f"epoch={epoch+1} step={step} lr={lr:.2e} "
            f"loss={loss:.4f} rkl={rkl:.4f} ce={ce_l:.4f} aux={aux:.4f}"
        )
        if HAS_RICH and console:
            console.print(f"[dim]{msg}[/dim]")
        else:
            print(msg)

    def _save_checkpoint(self, tag: str = None):
        tag = tag or f"step_{self.global_step}"
        ckpt_dir = Path(self.config.output_dir) / f"ckpt_{tag}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), ckpt_dir / "model.pt")

        # Save optimizer state
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        # Save training metadata
        meta = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": asdict(self.config),
        }
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Checkpoint saved: {ckpt_dir}")

        # Rotation: keep last N checkpoints
        self._checkpoint_paths.append(ckpt_dir)
        if len(self._checkpoint_paths) > self.config.keep_last_n_checkpoints:
            old = self._checkpoint_paths.pop(0)
            if "final" not in str(old):
                import shutil
                shutil.rmtree(old, ignore_errors=True)

    def load_checkpoint(self, checkpoint_dir: str):
        ckpt_dir = Path(checkpoint_dir)
        self.model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=self.device))
        if (ckpt_dir / "optimizer.pt").exists():
            self.optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt"))
        if (ckpt_dir / "meta.json").exists():
            with open(ckpt_dir / "meta.json") as f:
                meta = json.load(f)
            self.global_step = meta.get("global_step", 0)
            self.epoch = meta.get("epoch", 0)
        print(f"Loaded checkpoint from {ckpt_dir} (step={self.global_step})")


# ---------------------------------------------------------------------------
# Stage launchers
# ---------------------------------------------------------------------------

def run_stage(
    stage: int,
    model_config_path: str = None,
    checkpoint_dir: str = None,
    teacher_model_path: str = None,
    **kwargs,
):
    """
    Launch a training stage. Handles dataset selection and hyperparameter defaults.

    stage 1: CodeAlpaca-20k — architecture bringup
    stage 2: Evol-Instruct-80k — extended distillation
    stage 3: OPSDC — reasoning compression
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from arch.config import ModelConfig, ModelConfig700M, ModelConfig3B
    from arch.model import CodingSSM
    from train.dataset import CodeAlpacaDataset, EvolInstructDataset, OPSDCDataset, DistillDataset, make_dataloader, get_tokenizer

    # Stage 0 = 700M bringup; stages 1-3 default to 700M until explicitly set to 3B
    stage_defaults = {
        0: TrainConfig(
            output_dir="checkpoints/stage0",
            dataset="codealpaca",
            epochs=3,
            learning_rate=3e-4,
            batch_size=1,
            grad_accum_steps=16,
            max_seq_len=512,     # short seqs for bringup — fast iteration
            alpha=0.0, beta=1.0, gamma=0.01,  # CE only (no teacher yet)
            save_every_n_steps=200,
            log_every_n_steps=5,
        ),
        1: TrainConfig(
            output_dir="checkpoints/stage1",
            dataset="codealpaca",
            epochs=3,
            learning_rate=3e-4,
            batch_size=1,
            grad_accum_steps=16,
            max_seq_len=512,
            alpha=0.7, beta=0.3, gamma=0.01,
        ),
        2: TrainConfig(
            output_dir="checkpoints/stage2",
            dataset="evolinstruct",
            epochs=2,
            learning_rate=1e-4,
            batch_size=1,
            grad_accum_steps=16,
            max_seq_len=1024,
            alpha=0.7, beta=0.3, gamma=0.01,
        ),
        3: TrainConfig(
            output_dir="checkpoints/stage3",
            dataset="opsdc",
            epochs=1,
            learning_rate=5e-5,
            alpha=0.5, beta=0.5, gamma=0.01,
        ),
    }

    config = stage_defaults[stage]
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)

    print(f"=== Stage {stage} ===")
    print(f"Dataset: {config.dataset}, seq_len={config.max_seq_len}")
    print(f"Epochs: {config.epochs}, LR: {config.learning_rate}")
    print(f"Loss weights: α={config.alpha} β={config.beta} γ={config.gamma}")

    # Build model — use 700M for stages 0-1, 3B for stages 2-3 (after bringup proven)
    if model_config_path:
        with open(model_config_path) as f:
            model_cfg = ModelConfig(**json.load(f))
    elif kwargs.get("model_size") == "3b":
        model_cfg = ModelConfig3B()
    else:
        model_cfg = ModelConfig700M()   # default: memory-safe bringup size

    model = CodingSSM(model_cfg)
    n_params = model.num_parameters()
    print(f"Model: {n_params:,} parameters ({n_params/1e9:.2f}B)")

    if checkpoint_dir:
        trainer = Trainer(model, config)
        trainer.load_checkpoint(checkpoint_dir)
    else:
        trainer = Trainer(model, config)

    # Tokenizer
    tokenizer = get_tokenizer(config.tokenizer_name)

    # Dataset
    if config.dataset == "codealpaca":
        base_ds = CodeAlpacaDataset(tokenizer, max_length=config.max_seq_len)
    elif config.dataset == "evolinstruct":
        base_ds = EvolInstructDataset(tokenizer, max_length=config.max_seq_len,
                                      max_examples=config.max_examples)
    elif config.dataset == "opsdc":
        opsdc_path = kwargs.get("opsdc_jsonl", "data/opsdc_pairs.jsonl")
        base_ds = OPSDCDataset(tokenizer, opsdc_path, max_length=config.max_seq_len)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    if config.logit_cache_dir:
        ds = DistillDataset(base_ds, logit_cache_path=None)
        ds.base = base_ds
    else:
        ds = base_ds

    loader = make_dataloader(ds, batch_size=config.batch_size, shuffle=True)
    print(f"Dataset size: {len(base_ds) if hasattr(base_ds, '__len__') else 'streaming'}")

    trainer.train(loader)
    print(f"Stage {stage} complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CodingSSM distillation training")
    parser.add_argument("--stage", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--teacher-model", type=str, default=None)
    parser.add_argument("--logit-cache", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-size", type=str, default="700m", choices=["700m", "3b"])
    args = parser.parse_args()

    kwargs = {}
    if args.logit_cache:
        kwargs["logit_cache_dir"] = args.logit_cache
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    if args.max_examples:
        kwargs["max_examples"] = args.max_examples
    if args.epochs:
        kwargs["epochs"] = args.epochs
    if args.output_dir:
        kwargs["output_dir"] = args.output_dir
    if args.model_size:
        kwargs["model_size"] = args.model_size

    run_stage(
        stage=args.stage,
        checkpoint_dir=args.checkpoint,
        teacher_model_path=args.teacher_model,
        **kwargs,
    )
