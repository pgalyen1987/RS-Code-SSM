#!/usr/bin/env python3
"""Diagnostic: check top-10 token predictions from SFT model."""
import torch
import sys
sys.path.insert(0, '/workspace/RS-Code-SSM')
from arch.config import ModelConfig700M
from arch.model import CodingSSM
from transformers import AutoTokenizer

device = torch.device('cuda')
cfg = ModelConfig700M()
model = CodingSSM(cfg).to(device).to(torch.bfloat16)
ckpt = torch.load('checkpoints/sft/sft_latest.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
print(f"Loaded step={ckpt['step']}, best_loss={ckpt['best_loss']:.4f}", flush=True)

SYSTEM = 'You are an expert Python programmer. Use <think> tags to reason, then provide code.'
prefix = (
    f'<|im_start|>system\n{SYSTEM}<|im_end|>\n'
    f'<|im_start|>user\nWrite a Python function add(a, b) that returns the sum.<|im_end|>\n'
    f'<|im_start|>assistant\n'
)
ids = tok.encode(prefix, add_special_tokens=False)
ids_t = torch.tensor([ids], dtype=torch.long, device=device)

states = [None] * cfg.n_layers
with torch.no_grad():
    logits, _, states = model(ids_t, states=states, return_states=True)
nxt = logits[0, -1].float()
probs = torch.softmax(nxt, dim=-1)
topk = torch.topk(probs, 10)
print('\nTop-10 first tokens after <|im_start|>assistant\\n:')
for v, i in zip(topk.values.tolist(), topk.indices.tolist()):
    print(f'  {v:.4f}  id={i:6d}  {repr(tok.decode([i]))}')

# step forward through ```python\ndef add(
print('\nTop-10 after feeding ```python\\ndef add( token by token:')
forced_str = '```python\ndef add('
forced_ids = tok.encode(forced_str, add_special_tokens=False)
print(f'  forced tokens: {[tok.decode([x]) for x in forced_ids]}')
for fid in forced_ids:
    with torch.no_grad():
        logits, _, states = model(
            torch.tensor([[fid]], dtype=torch.long, device=device),
            states=states, return_states=True
        )
nxt2 = logits[0, 0].float()
probs2 = torch.softmax(nxt2, dim=-1)
topk2 = torch.topk(probs2, 10)
for v, i in zip(topk2.values.tolist(), topk2.indices.tolist()):
    print(f'  {v:.4f}  id={i:6d}  {repr(tok.decode([i]))}')

# Check entropy - is the distribution flat (broken) or peaked (learned)?
entropy = -(probs * (probs + 1e-10).log()).sum().item()
entropy2 = -(probs2 * (probs2 + 1e-10).log()).sum().item()
print(f'\nEntropy at assistant start: {entropy:.2f} (max={torch.log(torch.tensor(float(len(tok)))).item():.2f})')
print(f'Entropy after forced prefix: {entropy2:.2f}')
print('(low entropy = peaked/confident, high entropy = flat/confused)')
