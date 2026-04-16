"""
Pick the best local pretrain checkpoint under checkpoints/pretrain/.

`pretrain_step_*.pt` in the output root is preferred (highest step in filename).
Also considers pretrain_final.pt and training/pretrain_latest.pt (HF cache layout).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import torch


def find_best_pretrain_checkpoint(
    pretrain_dir: Path,
) -> Tuple[Optional[Path], int, int]:
    """
    Returns (path, tokens_seen, step) for the best checkpoint, or (None, 0, 0).

    Selection order:
      1. Highest pretrain_step_XXXXXX.pt by numeric suffix (no full-file scan).
      2. Else pretrain_final.pt if present.
      3. Else training/pretrain_latest.pt (hf_hub_download layout).
    Then loads metadata once from the chosen file.
    """
    pretrain_dir = Path(pretrain_dir)
    if not pretrain_dir.is_dir():
        return None, 0, 0

    best_step = -1
    best_path: Optional[Path] = None

    for p in pretrain_dir.glob("pretrain_step_*.pt"):
        m = re.search(r"pretrain_step_(\d+)\.pt$", p.name)
        if not m:
            continue
        st = int(m.group(1))
        if st > best_step:
            best_step = st
            best_path = p

    if best_path is None:
        fin = pretrain_dir / "pretrain_final.pt"
        if fin.is_file():
            best_path = fin

    if best_path is None:
        hub_mirror = pretrain_dir / "training" / "pretrain_latest.pt"
        if hub_mirror.is_file():
            best_path = hub_mirror

    if best_path is None:
        return None, 0, 0

    try:
        meta = torch.load(best_path, map_location="cpu", weights_only=False)
        tokens_seen = int(meta.get("tokens_seen", 0))
        step = int(meta.get("step", 0))
        del meta
        return best_path, tokens_seen, step
    except Exception:
        return None, 0, 0
