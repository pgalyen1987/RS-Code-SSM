"""
Initialize CodingSSM 3B from Qwen2.5-3B pretrained weights.

Compatible weights copied:
  - embed_tokens   (vocab_size=152064, d_model=2048 — exact match)
  - final RMSNorm
  - Shared attention Q/O projections (same shape: [2048, 2048])
  - Shared attention K/V projections (tiled: Qwen GQA 8→32 heads)

Mamba-2 SSM, MoE FFN, Dense FFN, and per-layer LoRA adapters are NOT
touched — they initialize randomly and train from scratch during SFT.

Usage:
    python -m train.init_from_qwen \
        --output checkpoints/qwen_init.pt \
        --qwen-model Qwen/Qwen2.5-3B
"""

import torch
import argparse
from pathlib import Path


def init_from_qwen(model, qwen_model_id: str = "Qwen/Qwen2.5-3B") -> dict:
    """
    Copy compatible Qwen2.5-3B weights into a CodingSSM 3B model in-place.
    Returns a report dict mapping weight name → "copied" | "tiled" | "skipped".
    """
    from transformers import AutoModel

    print(f"[INIT] Loading {qwen_model_id} (this downloads ~6 GB) ...", flush=True)
    qwen = AutoModel.from_pretrained(
        qwen_model_id,
        trust_remote_code=True,
    )
    q = qwen.state_dict()
    report = {}

    def _copy(dst: torch.Tensor, src: torch.Tensor, name: str):
        dst.data.copy_(src.to(dst.dtype))
        report[name] = "copied"

    def _tile(dst: torch.Tensor, src: torch.Tensor, name: str):
        # Tile src along dim 0 until it reaches dst's size, then slice exactly.
        repeats = (dst.shape[0] + src.shape[0] - 1) // src.shape[0]
        tiled = src.repeat(repeats, 1)[:dst.shape[0]]
        dst.data.copy_(tiled.to(dst.dtype))
        report[name] = "tiled"

    # 1. Token embeddings — copy overlapping vocab rows (handles vocab size mismatch)
    embed_key = "embed_tokens.weight"
    if embed_key in q:
        src = q[embed_key]   # (src_vocab, d_model)
        dst = model.embed.weight
        if src.shape == dst.shape:
            _copy(dst, src, "embed")
        elif src.shape[1] == dst.shape[1]:
            # Partial copy: copy min(src_vocab, dst_vocab) rows
            n = min(src.shape[0], dst.shape[0])
            dst.data[:n].copy_(src[:n].to(dst.dtype))
            report["embed"] = f"partial({n}/{dst.shape[0]})"
            print(f"[INIT] Embed: copied {n}/{dst.shape[0]} vocab rows", flush=True)
        else:
            report["embed"] = "skipped"
            print(f"[INIT] WARNING: embed d_model mismatch {src.shape} vs {dst.shape} — skipping", flush=True)
    else:
        report["embed"] = "skipped"

    # 2. Final RMSNorm: [2048] — exact match
    norm_key = "norm.weight"
    if norm_key in q and q[norm_key].shape == model.norm_f.weight.shape:
        _copy(model.norm_f.weight, q[norm_key], "norm_f")
    else:
        report["norm_f"] = "skipped"

    # 3. Shared attention weights
    # Qwen2.5-3B: 36 layers, 16 Q heads × 128 = 2048, 8 KV heads × 128 = 1024
    # Our model: n_shared_attn=2 sets, 32 heads × 64 = 2048 per projection
    # Q/O: [2048, 2048] — direct copy
    # K/V: [1024, 2048] → tile to [2048, 2048]
    n_qwen_layers = sum(1 for k in q if k.startswith("layers.") and k.endswith(".self_attn.q_proj.weight"))
    n_shared = len(model.shared_attn)

    for si, shared in enumerate(model.shared_attn):
        # Divide Qwen layers evenly across shared sets
        start = si * n_qwen_layers // n_shared
        end = (si + 1) * n_qwen_layers // n_shared
        layer_indices = list(range(start, end))

        def avg(proj_key):
            tensors = [q[f"layers.{i}.self_attn.{proj_key}"] for i in layer_indices
                       if f"layers.{i}.self_attn.{proj_key}" in q]
            if not tensors:
                return None
            return torch.stack(tensors).mean(0)

        tag = f"shared_attn[{si}]"

        q_avg = avg("q_proj.weight")
        if q_avg is not None:
            if q_avg.shape == shared.q_proj.weight.shape:
                _copy(shared.q_proj.weight, q_avg, f"{tag}.q_proj")
            else:
                report[f"{tag}.q_proj"] = "skipped"

        o_avg = avg("o_proj.weight")
        if o_avg is not None:
            if o_avg.shape == shared.o_proj.weight.shape:
                _copy(shared.o_proj.weight, o_avg, f"{tag}.o_proj")
            else:
                report[f"{tag}.o_proj"] = "skipped"

        k_avg = avg("k_proj.weight")
        if k_avg is not None:
            if k_avg.shape == shared.k_proj.weight.shape:
                _copy(shared.k_proj.weight, k_avg, f"{tag}.k_proj")
            elif k_avg.shape[1] == shared.k_proj.weight.shape[1]:
                _tile(shared.k_proj.weight, k_avg, f"{tag}.k_proj")
            else:
                report[f"{tag}.k_proj"] = "skipped"

        v_avg = avg("v_proj.weight")
        if v_avg is not None:
            if v_avg.shape == shared.v_proj.weight.shape:
                _copy(shared.v_proj.weight, v_avg, f"{tag}.v_proj")
            elif v_avg.shape[1] == shared.v_proj.weight.shape[1]:
                _tile(shared.v_proj.weight, v_avg, f"{tag}.v_proj")
            else:
                report[f"{tag}.v_proj"] = "skipped"

    del qwen, q
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    copied = sum(1 for v in report.values() if v == "copied")
    tiled  = sum(1 for v in report.values() if v == "tiled")
    skipped = sum(1 for v in report.values() if v == "skipped")
    print(f"[INIT] Weights: {copied} copied, {tiled} tiled, {skipped} skipped", flush=True)
    return report


def freeze_base_weights(model) -> int:
    """
    Freeze embed, norm_f, and shared attention BASE weights.
    Per-layer LoRA adapters (q_lora_A/B etc.), Mamba-2 SSM, and FFN remain trainable.
    Returns the number of frozen parameters.
    """
    frozen = 0

    # Embeddings (also freezes lm_head via tied weights)
    model.embed.weight.requires_grad_(False)
    frozen += model.embed.weight.numel()

    # Final norm
    model.norm_f.weight.requires_grad_(False)
    frozen += model.norm_f.weight.numel()

    # Shared attention base weights only (not the LoRA A/B params on each layer)
    for shared in model.shared_attn:
        for proj in (shared.q_proj, shared.k_proj, shared.v_proj, shared.o_proj):
            proj.weight.requires_grad_(False)
            frozen += proj.weight.numel()
            if proj.bias_param is not None:
                proj.bias_param.requires_grad_(False)
                frozen += proj.bias_param.numel()

    return frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="checkpoints/qwen_init.pt")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-Coder-3B")
    args = parser.parse_args()

    from arch.config import ModelConfig3B
    from arch.model import CodingSSM

    print("[INIT] Building CodingSSM 3B ...", flush=True)
    cfg = ModelConfig3B()
    model = CodingSSM(cfg)
    n_params = model.num_parameters()
    print(f"[INIT] Model: {n_params:,} params ({n_params/1e9:.2f}B)", flush=True)

    init_from_qwen(model, qwen_model_id=args.qwen_model)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_config": cfg.__dict__,
        "step": 0,
        "best_loss": float("inf"),
        "init_source": args.qwen_model,
    }, out)
    size_gb = out.stat().st_size / 1e9
    print(f"[INIT] Saved to {out}  ({size_gb:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
