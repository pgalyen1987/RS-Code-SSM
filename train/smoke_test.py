"""
End-to-end smoke test: tiny model + synthetic data through one training step.
No tokenizer download required — uses fake token IDs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from arch.config import ModelConfig
from arch.model import CodingSSM
from train.dataset import collate_fn
from train.distill import Trainer, TrainConfig, ce_loss, reverse_kld_loss

print("=== Smoke Test: CodingSSM Training Pipeline ===\n")

# 1. Tiny model
cfg = ModelConfig(
    d_model=128, d_inner=256, n_layers=4,
    n_heads=4, d_head=64, d_state=16, n_groups=2, chunk_size=32,
    n_attn_heads=2, d_attn_head=64, attn_every_n=2, attn_window=16,
    n_shared_attn=2, lora_rank=4,
    n_experts=4, max_active_experts=2, min_active_experts=1, d_ffn=256,
    vocab_size=512, max_seq_len=64,
)
model = CodingSSM(cfg)
n_params = model.num_parameters()
print(f"Model: {n_params:,} parameters")

# 2. Synthetic batch
B, L, V = 2, 32, 512
input_ids = torch.randint(0, V, (B, L))
labels = input_ids.clone()
labels[:, :8] = -100   # mask prompt

# With teacher logits
teacher_logits = torch.randn(B, L, V)

# 3. Forward pass
print("\nForward pass...")
logits, aux = model(input_ids)
print(f"  logits: {logits.shape}, aux_loss: {aux.item():.4f}")

# 4. Losses
l_ce = ce_loss(logits, labels)
l_rkl = reverse_kld_loss(logits, teacher_logits, labels)
loss = 0.7 * l_rkl + 0.3 * l_ce + 0.01 * aux
print(f"\nLoss breakdown:")
print(f"  CE:       {l_ce.item():.4f}")
print(f"  Rev-KLD:  {l_rkl.item():.4f}")
print(f"  Aux:      {aux.item():.4f}")
print(f"  Total:    {loss.item():.4f}")

# 5. Backward
print("\nBackward pass...")
loss.backward()
grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
print(f"  Layers with grads: {len(grad_norms)}")
print(f"  Max grad norm:     {max(grad_norms):.4f}")
print(f"  Mean grad norm:    {sum(grad_norms)/len(grad_norms):.4f}")

# 6. Trainer step
print("\nTrainer one-step test...")
train_cfg = TrainConfig(
    output_dir="/tmp/ssm_smoke_test",
    batch_size=2, grad_accum_steps=1,
    learning_rate=1e-3, epochs=1,
    save_every_n_steps=999999,
    log_every_n_steps=1,
)

# Re-init fresh model (no accumulated grads)
model2 = CodingSSM(cfg)
trainer = Trainer(model2, train_cfg, device=torch.device("cpu"))

# Fake dataset
class FakeDataset(torch.utils.data.Dataset):
    def __len__(self): return 4
    def __getitem__(self, i):
        ids = torch.randint(0, 512, (32,))
        lbl = ids.clone(); lbl[:8] = -100
        return {"input_ids": ids, "labels": lbl,
                "teacher_logits": torch.randn(32, 512)}

from torch.utils.data import DataLoader
loader = DataLoader(FakeDataset(), batch_size=2, collate_fn=collate_fn)
trainer.train(loader, total_steps=2)

print("\n=== All tests passed ===")
