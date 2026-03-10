"""
Dataset loading and tokenization for CodingSSM distillation training.

Stages:
  Stage 1 — CodeAlpaca-20k (sahil2801/CodeAlpaca-20k): instruction/response pairs
  Stage 2 — Evol-Instruct code (nickrosh/Evol-Instruct-Code-80k-v1): harder coding tasks
  Stage 3 — OPSDC self-distillation data (generated at runtime, no separate dataset)

All datasets are formatted as a standard instruction template:
  <|im_start|>system\n{system}<|im_end|>\n
  <|im_start|>user\n{instruction}<|im_end|>\n
  <|im_start|>assistant\n{response}<|im_end|>

We use the Qwen2.5 tokenizer (vocab_size=152064) throughout.
The teacher's tokenizer and the student's tokenizer must match.
"""

import json
import os
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

# Optional: rich progress for dataset preparation
try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

SYSTEM_PROMPT = (
    "You are CodingSSM, an expert AI coding assistant. "
    "You write clean, efficient, well-documented code and explain your reasoning clearly."
)

CONCISE_SYSTEM_PROMPT = (
    "You are CodingSSM. Be as concise as possible while remaining accurate and complete."
)

# Chat template tokens (Qwen2.5 / ChatML format)
IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"
EOT      = "<|endoftext|>"


def format_chat(instruction: str, response: str, system: str = SYSTEM_PROMPT) -> str:
    return (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{instruction}{IM_END}\n"
        f"{IM_START}assistant\n{response}{IM_END}"
    )


def get_tokenizer(model_name: str = "Qwen/Qwen2.5-Coder-7B"):
    """Load tokenizer (tries local cache first, then HuggingFace Hub)."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return tok
    except Exception as e:
        raise RuntimeError(
            f"Could not load tokenizer '{model_name}'. "
            f"Run: pip install transformers\n{e}"
        )


class CodeAlpacaDataset(Dataset):
    """
    sahil2801/CodeAlpaca-20k
    Fields: instruction, input, output
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        split: str = "train",
        cache_dir: str = None,
        local_path: str = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        if local_path and Path(local_path).exists():
            self._load_local(local_path)
        else:
            self._load_hub(split, cache_dir)

    def _load_hub(self, split, cache_dir):
        try:
            from datasets import load_dataset
            ds = load_dataset("sahil2801/CodeAlpaca-20k", split=split, cache_dir=cache_dir)
            for ex in ds:
                instruction = ex["instruction"]
                if ex.get("input", "").strip():
                    instruction = f"{instruction}\n\n{ex['input']}"
                self.examples.append({
                    "instruction": instruction,
                    "response": ex["output"],
                })
        except Exception as e:
            raise RuntimeError(f"Failed to load CodeAlpaca-20k from Hub: {e}")

    def _load_local(self, path):
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = format_chat(ex["instruction"], ex["response"])
        return self._tokenize(text)

    def _tokenize(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)     # (L,)
        # Labels: same as input_ids but mask the prompt (only train on response tokens)
        labels = input_ids.clone()

        # Find assistant token start and mask everything before it
        # We look for the last occurrence of <|im_start|>assistant
        asst_token_ids = self.tokenizer.encode(
            f"{IM_START}assistant\n", add_special_tokens=False
        )
        asst_start = _find_subseq(input_ids.tolist(), asst_token_ids)
        if asst_start >= 0:
            labels[:asst_start + len(asst_token_ids)] = -100

        return {"input_ids": input_ids, "labels": labels}


class EvolInstructDataset(Dataset):
    """
    nickrosh/Evol-Instruct-Code-80k-v1
    Fields: instruction, output
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        split: str = "train",
        cache_dir: str = None,
        max_examples: int = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        try:
            from datasets import load_dataset
            ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split=split, cache_dir=cache_dir)
            for i, ex in enumerate(ds):
                if max_examples and i >= max_examples:
                    break
                self.examples.append({
                    "instruction": ex["instruction"],
                    "response": ex["output"],
                })
        except Exception as e:
            raise RuntimeError(f"Failed to load Evol-Instruct dataset: {e}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = format_chat(ex["instruction"], ex["response"])
        return _tokenize_with_labels(self.tokenizer, text, self.max_length)


class DistillDataset(Dataset):
    """
    Wraps another dataset and pairs each example with pre-computed teacher
    logits from a .pt cache file. If no cache exists, teacher logits are None
    (useful for pure CE training or online distillation).

    Cache format: list of tensors, one per example, shape (L, vocab_size).
    """

    def __init__(self, base_dataset: Dataset, logit_cache_path: str = None):
        self.base = base_dataset
        self.logits = None
        if logit_cache_path and Path(logit_cache_path).exists():
            print(f"Loading teacher logits from {logit_cache_path}...")
            self.logits = torch.load(logit_cache_path, map_location="cpu")
            assert len(self.logits) == len(self.base), \
                f"Cache size {len(self.logits)} != dataset size {len(self.base)}"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if self.logits is not None:
            item["teacher_logits"] = self.logits[idx]
        return item


class OPSDCDataset(IterableDataset):
    """
    OPSDC (One-Pass Self-Distillation Compression) dataset.
    Generates on-the-fly by pairing:
      - original instruction
      - concise teacher response (generated with CONCISE_SYSTEM_PROMPT)

    Used in Stage 3. The teacher is the model itself (or Qwen teacher).
    Data is streamed from a JSONL file of format:
      {"instruction": "...", "verbose": "...", "concise": "..."}
    """

    def __init__(self, tokenizer, jsonl_path: str, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.jsonl_path = jsonl_path
        self.max_length = max_length

    def __iter__(self) -> Iterator[dict]:
        with open(self.jsonl_path) as f:
            for line in f:
                ex = json.loads(line.strip())
                # Verbose version (student input) trained with concise target
                verbose_text = format_chat(
                    ex["instruction"], ex["verbose"], system=SYSTEM_PROMPT
                )
                concise_text = format_chat(
                    ex["instruction"], ex["concise"], system=CONCISE_SYSTEM_PROMPT
                )
                item_v = _tokenize_with_labels(self.tokenizer, verbose_text, self.max_length)
                item_c = _tokenize_with_labels(self.tokenizer, concise_text, self.max_length)
                item_c["opsdc_pair"] = item_v["input_ids"]
                yield item_c


def collate_fn(batch: list[dict]) -> dict:
    """Pad a batch of variable-length examples to the same length."""
    keys = batch[0].keys()
    out = {}
    for key in keys:
        tensors = [b[key] for b in batch if key in b]
        if not tensors:
            continue
        if tensors[0].dim() == 1:
            # Pad 1D tensors
            max_len = max(t.shape[0] for t in tensors)
            pad_val = -100 if key == "labels" else 0
            padded = torch.stack([
                torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_val)
                for t in tensors
            ])
            out[key] = padded
        else:
            # 2D (teacher logits): pad along sequence dimension
            max_len = max(t.shape[0] for t in tensors)
            vocab = tensors[0].shape[1]
            padded = torch.zeros(len(tensors), max_len, vocab)
            for i, t in enumerate(tensors):
                padded[i, :t.shape[0]] = t
            out[key] = padded
    return out


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if not isinstance(dataset, IterableDataset) else False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_subseq(seq: list, subseq: list) -> int:
    """Return the last start index of subseq in seq, or -1."""
    result = -1
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i:i + m] == subseq:
            result = i
    return result


def _tokenize_with_labels(tokenizer, text: str, max_length: int) -> dict:
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].squeeze(0)
    labels = input_ids.clone()

    asst_token_ids = tokenizer.encode(
        f"{IM_START}assistant\n", add_special_tokens=False
    )
    asst_start = _find_subseq(input_ids.tolist(), asst_token_ids)
    if asst_start >= 0:
        labels[:asst_start + len(asst_token_ids)] = -100

    return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("Testing dataset utilities (no tokenizer needed for format test)...")

    text = format_chat("Write a hello world in Python.", "print('Hello, world!')")
    print("Chat format sample:")
    print(text)
    print()

    # Test collate with dummy tensors
    batch = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([4, 5, 6, 7]), "labels": torch.tensor([4, 5, 6, 7])},
    ]
    out = collate_fn(batch)
    print(f"Collated input_ids: {out['input_ids'].shape}")
    print(f"Collated labels:    {out['labels'].shape}")
    assert out["input_ids"].shape == (2, 4)
    print("OK")
