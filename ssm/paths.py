"""Centralized path configuration. Use env vars for Kaggle/portability."""
from pathlib import Path
import os

REPO_ROOT = Path(os.environ.get("REPO_ROOT") or Path(__file__).resolve().parents[1])
DATA_DIR = Path(os.environ.get("DATA_DIR") or REPO_ROOT / "data")
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR") or REPO_ROOT / "checkpoints")
WORK_DIR = Path(os.environ.get("WORK_DIR") or (Path("/kaggle/working") if Path("/kaggle").exists() else REPO_ROOT))

# EpiChat knowledge graph — in-repo at REPO_ROOT/epichat (override with EPICHAT_DIR env)
EPICHAT_DIR = Path(os.environ.get("EPICHAT_DIR") or str(REPO_ROOT / "epichat"))

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
