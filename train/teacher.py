"""
Teacher logit generation using a GGUF model via llama.cpp.

The teacher (Qwen3-Next-80B-A3B-Instruct or Qwen2.5-Coder-32B) generates:
  1. Per-token logit distributions over the vocabulary (for KD)
  2. Full response text (for dataset augmentation / Stage 3 OPSDC)

Two modes:
  A) OFFLINE: Pre-generate all teacher logits and cache as .pt files.
     Use for Stage 1 / Stage 2 distillation where the dataset is fixed.
  B) ONLINE:  Generate teacher logits on-the-fly during training.
     Slower but needed for Stage 3 OPSDC (teacher = self).

For offline mode we only need the top-K logits (K=64) to save disk space.
The student sees only these K positions; remaining positions get -inf.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Teacher wrapper
# ---------------------------------------------------------------------------

class Teacher:
    """
    Wraps a llama.cpp GGUF model and exposes:
      - generate_with_logits(prompt_ids) -> (text, logits)
      - logits shape: (L, vocab_size)  where L = number of generated tokens

    Logits are extracted via the llama_cpp eval_logits hook.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = -1,
        n_batch: int = 512,
        top_k_logits: int = 64,       # only save top-k logit positions
        temperature: float = 0.0,     # 0 = greedy (deterministic for caching)
        max_new_tokens: int = 1024,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.top_k_logits = top_k_logits
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

        if n_threads == -1:
            import multiprocessing
            n_threads = max(1, multiprocessing.cpu_count() - 1)

        print(f"Loading teacher: {Path(model_path).name}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            logits_all=True,   # <-- critical: return logits for every token
            verbose=verbose,
        )
        self.vocab_size = self.llm.n_vocab()
        print(f"Teacher loaded. vocab_size={self.vocab_size}")

    def _build_prompt(self, instruction: str, system: str) -> str:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate_with_logits(
        self,
        instruction: str,
        system: str = "You are CodingSSM, an expert AI coding assistant.",
        max_new_tokens: int = None,
    ) -> dict:
        """
        Generate a response and return token logits.

        Returns:
          {
            "text":       generated response string,
            "token_ids":  list of generated token IDs (length L),
            "logits":     tensor (L, vocab_size) — full or sparse (top_k_logits),
          }
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens
        prompt = self._build_prompt(instruction, system)

        output = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=self.temperature,
            echo=False,
            logprobs=None,   # we use eval_logits directly below
        )

        # Extract generated token IDs and raw logits
        # After calling llm(), the eval state holds logits for all evaluated tokens
        token_ids = [c["id"] for c in output.get("choices", [{}])[0].get("logprobs", {}).get("tokens", [])]
        text = output["choices"][0]["text"]

        # Get logits array from llama_cpp internal state
        # llama_cpp stores logits as a flat array: n_tokens * vocab_size
        import ctypes
        raw_logits = self.llm._ctx.get_logits()
        n_tokens = len(raw_logits) // self.vocab_size
        logits_tensor = torch.tensor(
            raw_logits, dtype=torch.float32
        ).view(n_tokens, self.vocab_size)

        # Only keep logits for the generated (non-prompt) portion
        prompt_tokens = self.llm.tokenize(prompt.encode())
        n_prompt = len(prompt_tokens)
        if logits_tensor.shape[0] > n_prompt:
            gen_logits = logits_tensor[n_prompt:]   # (L, vocab)
        else:
            gen_logits = logits_tensor              # fallback

        # Sparsify: keep only top-k logit positions to reduce memory
        if self.top_k_logits and self.top_k_logits < self.vocab_size:
            gen_logits = _sparsify_logits(gen_logits, self.top_k_logits)

        return {
            "text": text,
            "token_ids": token_ids,
            "logits": gen_logits,
        }

    def generate_text(self, instruction: str, system: str = None, max_new_tokens: int = None) -> str:
        """Lightweight text-only generation (no logit extraction)."""
        max_new_tokens = max_new_tokens or self.max_new_tokens
        system = system or "You are CodingSSM, an expert AI coding assistant."
        prompt = self._build_prompt(instruction, system)
        output = self.llm(prompt, max_tokens=max_new_tokens, temperature=self.temperature, echo=False)
        return output["choices"][0]["text"]


# ---------------------------------------------------------------------------
# Offline logit cache builder
# ---------------------------------------------------------------------------

def build_logit_cache(
    teacher: Teacher,
    instructions: list[str],
    output_dir: str,
    shard_size: int = 1000,
    resume: bool = True,
) -> None:
    """
    Generate teacher logits for all instructions and save as sharded .pt files.

    Output structure:
      output_dir/
        shard_0000.pt   -- list of dicts: {text, logits (L, V)}
        shard_0001.pt
        ...
        index.json      -- {total, shard_size, shards: [...]}

    Args:
        teacher:       Teacher instance
        instructions:  list of instruction strings
        output_dir:    directory to save cache
        shard_size:    number of examples per shard file
        resume:        if True, skip already-generated shards
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    index_path = out_path / "index.json"
    if resume and index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        start_shard = len(index["shards"])
        print(f"Resuming from shard {start_shard}")
    else:
        index = {"total": len(instructions), "shard_size": shard_size, "shards": []}
        start_shard = 0

    n_shards = (len(instructions) + shard_size - 1) // shard_size

    for shard_idx in range(start_shard, n_shards):
        shard_path = out_path / f"shard_{shard_idx:04d}.pt"
        start = shard_idx * shard_size
        end = min(start + shard_size, len(instructions))
        shard_instructions = instructions[start:end]

        print(f"\nShard {shard_idx}/{n_shards-1} — examples {start}-{end-1}")
        shard_data = []

        for i, instr in enumerate(shard_instructions):
            t0 = time.time()
            result = teacher.generate_with_logits(instr)
            elapsed = time.time() - t0
            shard_data.append({
                "text": result["text"],
                "logits": result["logits"],
            })
            print(f"  [{start+i+1}/{len(instructions)}] {elapsed:.1f}s | "
                  f"{len(result['text'])} chars | logits {result['logits'].shape}")

        torch.save(shard_data, shard_path)
        index["shards"].append(str(shard_path))
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"Saved shard {shard_idx} -> {shard_path}")

    print(f"\nDone. {len(instructions)} examples cached in {output_dir}")


def load_logit_cache(cache_dir: str) -> list[dict]:
    """Load all shards from a logit cache directory into memory."""
    index_path = Path(cache_dir) / "index.json"
    with open(index_path) as f:
        index = json.load(f)

    all_data = []
    for shard_path in index["shards"]:
        shard = torch.load(shard_path, map_location="cpu")
        all_data.extend(shard)
    print(f"Loaded {len(all_data)} cached examples from {cache_dir}")
    return all_data


# ---------------------------------------------------------------------------
# OPSDC response generator
# ---------------------------------------------------------------------------

def generate_opsdc_pairs(
    teacher: Teacher,
    instructions: list[str],
    output_path: str,
    resume: bool = True,
) -> None:
    """
    Generate (verbose, concise) response pairs for OPSDC Stage 3.

    For each instruction:
      1. Generate verbose response with standard system prompt
      2. Generate concise response with brevity-focused system prompt

    Output: JSONL with {instruction, verbose, concise}
    """
    VERBOSE_SYS = "You are CodingSSM, an expert AI coding assistant. Explain your reasoning thoroughly."
    CONCISE_SYS = "You are CodingSSM. Be as concise as possible while remaining accurate and complete."

    out_path = Path(output_path)
    done = set()

    if resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                ex = json.loads(line)
                done.add(ex["instruction"])
        print(f"Resuming OPSDC: {len(done)} already done")

    with open(out_path, "a") as fout:
        for i, instr in enumerate(instructions):
            if instr in done:
                continue

            t0 = time.time()
            verbose  = teacher.generate_text(instr, system=VERBOSE_SYS)
            concise  = teacher.generate_text(instr, system=CONCISE_SYS)
            elapsed  = time.time() - t0

            record = {"instruction": instr, "verbose": verbose, "concise": concise}
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            compression = len(concise) / max(len(verbose), 1)
            print(f"[{i+1}/{len(instructions)}] {elapsed:.1f}s | compression {compression:.2f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sparsify_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Keep only top_k logit values per token; set rest to -1e9 (effectively -inf after softmax).
    Shape preserved: (L, vocab_size). Stored as sparse for memory efficiency.
    For training we reconstruct full tensor from top-k positions.
    """
    vals, idx = logits.topk(top_k, dim=-1)
    sparse = torch.full_like(logits, -1e9)
    sparse.scatter_(-1, idx, vals)
    return sparse


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Generate teacher logit cache")
    parser.add_argument("--model", required=True, help="Path to teacher GGUF model")
    parser.add_argument("--dataset", default="codealpaca", choices=["codealpaca", "evolinstruct"])
    parser.add_argument("--output", required=True, help="Output cache directory")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=-1)
    parser.add_argument("--top-k-logits", type=int, default=64)
    parser.add_argument("--opsdc", action="store_true", help="Generate OPSDC pairs instead of logits")
    args = parser.parse_args()

    teacher = Teacher(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        top_k_logits=args.top_k_logits,
    )

    from datasets import load_dataset
    if args.dataset == "codealpaca":
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        instructions = []
        for ex in ds:
            instr = ex["instruction"]
            if ex.get("input", "").strip():
                instr = f"{instr}\n\n{ex['input']}"
            instructions.append(instr)
    else:
        ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
        instructions = [ex["instruction"] for ex in ds]

    if args.max_examples:
        instructions = instructions[:args.max_examples]

    print(f"Dataset: {args.dataset}, {len(instructions)} examples")

    if args.opsdc:
        generate_opsdc_pairs(teacher, instructions, args.output)
    else:
        build_logit_cache(teacher, instructions, args.output, shard_size=args.shard_size)
