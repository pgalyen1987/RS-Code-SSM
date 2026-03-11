"""Centralized path configuration. Use env vars for Kaggle/portability."""
from pathlib import Path
import os

REPO_ROOT = Path(os.environ.get("REPO_ROOT") or Path(__file__).resolve().parents[1])
DATA_DIR = Path(os.environ.get("DATA_DIR") or REPO_ROOT / "data")
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR") or REPO_ROOT / "checkpoints")
WORK_DIR = Path(os.environ.get("WORK_DIR") or (Path("/kaggle/working") if Path("/kaggle").exists() else REPO_ROOT))

# EpiChat knowledge graph (EPICHAT_DIR env, or ../EpiChat, ../epichat, repo/epichat)
_ep = os.environ.get("EPICHAT_DIR")
if _ep:
    EPICHAT_DIR = Path(_ep)
else:
    for cand in (REPO_ROOT.parent / "EpiChat", REPO_ROOT.parent / "epichat", REPO_ROOT / "epichat"):
        if cand.exists():
            EPICHAT_DIR = cand
            break
    else:
        EPICHAT_DIR = REPO_ROOT / "epichat"

# Model storage (~/.ssm/models or WORK_DIR/models on Kaggle)
MODEL_DIR = Path(
    os.environ.get("MODEL_DIR")
    or (WORK_DIR / "models" if Path("/kaggle").exists() else Path.home() / ".ssm" / "models")
)

# Config dir (~/.ssm or WORK_DIR/.ssm on Kaggle)
CONFIG_DIR = Path(
    os.environ.get("CONFIG_DIR")
    or (WORK_DIR / ".ssm" if Path("/kaggle").exists() else Path.home() / ".ssm")
)

# Temp dir for logs, etc.
TMP_DIR = Path(os.environ.get("TMP_DIR", "/tmp"))


def strpath(p: Path) -> str:
    return str(p)
