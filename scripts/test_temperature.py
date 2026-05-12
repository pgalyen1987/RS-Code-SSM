#!/usr/bin/env python3
"""Test model with temperature sampling + check what follows the opening fence."""
import torch
import sys
sys.path.insert(0, '.')
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

SYSTEM = (
    "You are an expert Python programmer. Think carefully step by step before writing code.\n"
    "Use <think> tags to show your reasoning, then provide the final solution.\n"
    "Format:\n<think>\n[your chain-of-thought reasoning here]\n</think>\n"
    "```python\n[your solution here]\n```"
)

prefix = (
    f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    "<|im_start|>user\nWrite a Python function add(a, b) that returns the sum of two numbers.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
ids = tok.encode(prefix, add_special_tokens=False)
ids_t = torch.tensor([ids], dtype=torch.long, device=device)
stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')}

# Check top-10 after just '```'
backtick_id = tok.encode('```', add_special_tokens=False)[0]
states = [None] * cfg.n_layers
with torch.no_grad():
    logits, _, states = model(ids_t, states=states, return_states=True)
# feed the backtick
with torch.no_grad():
    logits2, _, states2 = model(
        torch.tensor([[backtick_id]], dtype=torch.long, device=device),
        states=states, return_states=True
    )
nxt_after_bt = logits2[0, 0].float()
probs = torch.softmax(nxt_after_bt, dim=-1)
topk = torch.topk(probs, 10)
print("\nTop-10 tokens predicted AFTER just '```':")
for v, i in zip(topk.values.tolist(), topk.indices.tolist()):
    print(f"  {v:.4f}  id={i:6d}  {repr(tok.decode([i]))}")

# Try temperature=0.8 sampling, run 3 times
print("\n=== Temperature=0.8 sampling (3 runs) ===")
for run in range(3):
    states_r = [None] * cfg.n_layers
    with torch.no_grad():
        logits_r, _, states_r = model(ids_t, states=states_r, return_states=True)
    nxt = logits_r[0, -1].float()
    generated = []
    with torch.no_grad():
        for _ in range(250):
            logits_scaled = nxt / 0.8
            probs_s = torch.softmax(logits_scaled, dim=-1)
            nid = int(torch.multinomial(probs_s, 1).item())
            if nid in stop_ids:
                break
            generated.append(nid)
            logits_r, _, states_r = model(
                torch.tensor([[nid]], dtype=torch.long, device=device),
                states=states_r, return_states=True
            )
            nxt = logits_r[0, 0].float()
    decoded = tok.decode(generated, skip_special_tokens=True)
    has_def = 'def ' in decoded
    has_return = 'return' in decoded
    print(f"\n--- run {run+1} ({len(generated)} tokens, def={has_def} return={has_return}) ---")
    print(decoded[:400])
