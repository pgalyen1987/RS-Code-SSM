from dataclasses import dataclass, field
from typing import Optional


def ModelConfig700M() -> "ModelConfig":
    """
    700M parameter config for architecture bringup (Stage 0).
    Fits comfortably in ~14GB RAM (weights + Adafactor + activations).
    Same architecture as 3B — just smaller dims.

    Memory estimate:
      weights:    700M * 4B = 2.8GB
      grads:      2.8GB
      adafactor:  ~0.7GB (factored)
      activations (seq512, grad_ckpt): ~2GB
      total: ~9GB  ← fits with VS Code + browser on 64GB
    """
    return ModelConfig(
        d_model=1024,
        d_inner=2048,
        n_layers=24,
        n_heads=32,
        d_head=64,
        d_state=64,
        n_groups=4,
        chunk_size=256,
        n_attn_heads=16,
        d_attn_head=64,
        attn_every_n=4,
        attn_window=512,
        n_shared_attn=2,
        lora_rank=32,
        n_experts=8,
        max_active_experts=4,
        min_active_experts=1,
        d_ffn=4096,
        vocab_size=152064,
        max_seq_len=32768,
    )


def ModelConfig3B() -> "ModelConfig":
    """Full 3B parameter config. Use after Stage 0 validates the architecture."""
    return ModelConfig()  # all defaults are 3B


@dataclass
class ModelConfig:
    # Core dimensions
    d_model: int = 2048
    d_inner: int = 4096       # expand = 2 * d_model
    n_layers: int = 24

    # Mamba-2 / SSD
    n_heads: int = 64         # d_inner / d_head
    d_head: int = 64
    d_state: int = 128        # SSM state size N
    n_groups: int = 8         # number of SSM groups
    chunk_size: int = 256     # SSD chunk size

    # Sparse attention
    attn_every_n: int = 6     # 1 attention layer every N layers
    attn_window: int = 512    # sliding window size (tokens)
    n_attn_heads: int = 32
    d_attn_head: int = 64     # d_model / n_attn_heads

    # Shared attention + LoRA (Zamba2-style)
    n_shared_attn: int = 2    # number of shared attention weight sets (ABAB...)
    lora_rank: int = 64       # per-layer LoRA adapter rank for attention

    # MoE FFN
    n_experts: int = 8
    max_active_experts: int = 4  # dynamic top-k cap
    min_active_experts: int = 1  # dynamic top-k floor
    d_ffn: int = 8192         # FFN hidden dim (4 * d_model)
    moe_on_even_layers: bool = True  # MoE on even layers, dense FFN on odd

    # Vocabulary & sequence
    vocab_size: int = 152064  # Qwen2.5 tokenizer
    max_seq_len: int = 32768
    pad_token_id: int = 0

    # Training
    dropout: float = 0.0
    tie_embeddings: bool = True

    # Distillation
    kd_alpha: float = 0.7     # weight on reverse KLD
    kd_beta: float = 0.3      # weight on CE loss
    kd_temperature: float = 1.0

    def attn_layer_indices(self):
        """Return which layer indices (0-based) use attention instead of Mamba-2."""
        return [i for i in range(self.n_layers) if (i + 1) % self.attn_every_n == 0]

    def is_moe_layer(self, layer_idx: int) -> bool:
        return self.moe_on_even_layers and (layer_idx % 2 == 0)

    def expert_budget(self, layer_idx: int) -> int:
        """Descending capacity schedule: earlier layers get more active experts."""
        # Linearly decreasing from max to min across layers
        frac = layer_idx / max(self.n_layers - 1, 1)
        budget = self.max_active_experts - frac * (self.max_active_experts - self.min_active_experts)
        return max(self.min_active_experts, round(budget))
