#!/usr/bin/env python3
"""Test natural generation from SFT model using exact training system prompt."""
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

for problem in [
    "Write a Python function add(a, b) that returns the sum of two numbers.",
    "Write a Python function is_even(n) that returns True if n is even.",
]:
    prefix = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    ids = tok.encode(prefix, add_special_tokens=False)
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')}

    states = [None] * cfg.n_layers
    with torch.no_grad():
        logits, _, states = model(ids_t, states=states, return_states=True)
    nxt = logits[0, -1].float()

    generated = []
    with torch.no_grad():
        for _ in range(300):
            if generated:
                w = torch.tensor(generated[-64:], device=device)
                c = torch.bincount(w, minlength=nxt.shape[0]).float()
                nxt = nxt - c * 2.0
            nid = int(torch.argmax(nxt).item())
            if nid in stop_ids:
                break
            generated.append(nid)
            logits, _, states = model(
                torch.tensor([[nid]], dtype=torch.long, device=device),
                states=states, return_states=True
            )
            nxt = logits[0, 0].float()

    decoded = tok.decode(generated, skip_special_tokens=False)
    print(f"\n=== {problem[:40]} ===")
    print(decoded[:500])
    print(f"(total tokens: {len(generated)})")
