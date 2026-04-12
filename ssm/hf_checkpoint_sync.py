"""
Upload / download training checkpoints on the Hugging Face *model* repo (HF_MODEL_REPO), not the GitHub code repo.

Uses HF_TOKEN or HUGGINGFACE_HUB_TOKEN from the environment.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


def get_hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def upload_checkpoint(
    local_path: Path,
    repo_id: str,
    path_in_repo: str,
    token: Optional[str] = None,
) -> bool:
    """Upload a single file to a model repo. Overwrites remote path on each call."""
    token = token or get_hf_token()
    if not token or not repo_id:
        return False
    local_path = Path(local_path)
    if not local_path.is_file():
        return False
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"[HF] {local_path.name} → https://huggingface.co/{repo_id}/blob/main/{path_in_repo}", flush=True)
        return True
    except Exception as e:
        print(f"[HF] Upload failed: {e}", flush=True)
        return False


def download_checkpoint(
    repo_id: str,
    path_in_repo: str,
    dest_path: Path,
    token: Optional[str] = None,
) -> bool:
    """
    Download a checkpoint from the Hub into dest_path (parent dirs created).
    Returns True if the file was written.
    """
    token = token or get_hf_token()
    if not token:
        return False
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="model",
            token=token,
        )
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, dest_path)
        print(f"[HF] Pulled {repo_id}/{path_in_repo} → {dest_path}", flush=True)
        return True
    except Exception as e:
        print(f"[HF] Pull skipped ({path_in_repo}): {e}", flush=True)
        return False
