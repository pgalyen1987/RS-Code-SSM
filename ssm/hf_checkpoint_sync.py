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
    Streams directly to disk with periodic progress so large files don't look frozen.
    """
    token = token or get_hf_token()
    if not token:
        return False
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import requests

        url = f"https://huggingface.co/{repo_id}/resolve/main/{path_in_repo}"
        headers = {"Authorization": f"Bearer {token}"}
        print(f"[HF] Downloading {repo_id}/{path_in_repo} ...", flush=True)
        with requests.get(url, headers=headers, stream=True, timeout=30) as r:
            if r.status_code == 404:
                print(f"[HF] Pull skipped ({path_in_repo}): not found (first run?)", flush=True)
                return False
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            last_pct = -1
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):  # 8 MB chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = int(downloaded / total * 100)
                        if pct >= last_pct + 10:
                            print(f"[HF]   {pct}% ({downloaded // 1024 // 1024} / {total // 1024 // 1024} MB)", flush=True)
                            last_pct = pct
        print(f"[HF] Pulled → {dest_path} ({downloaded // 1024 // 1024} MB)", flush=True)
        return True
    except Exception as e:
        print(f"[HF] Pull skipped ({path_in_repo}): {e}", flush=True)
        if dest_path.exists() and dest_path.stat().st_size == 0:
            dest_path.unlink()
        return False
