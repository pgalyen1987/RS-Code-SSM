#!/usr/bin/env bash
# Download Qwen3.5-35B-A3B Q4_K_M teacher model (~20GB)
# Repo: unsloth/Qwen3.5-35B-A3B-GGUF
# Saves to ~/.ssm/models/

set -e
MODEL_DIR="$HOME/.ssm/models"
mkdir -p "$MODEL_DIR"

source "$(dirname "$0")/../.venv/bin/activate"

echo "=== Downloading Qwen3.5-35B-A3B Q4_K_M (teacher) ==="
echo "Target: $MODEL_DIR"
echo "Repo:   unsloth/Qwen3.5-35B-A3B-GGUF"
echo ""

# Try hf CLI first (faster, supports resume)
if command -v hf &>/dev/null; then
  hf download \
    unsloth/Qwen3.5-35B-A3B-GGUF \
    Qwen3.5-35B-A3B-Q4_K_M.gguf \
    --local-dir "$MODEL_DIR"
else
  # Fallback: huggingface_hub Python
  python - <<'EOF'
import os
from huggingface_hub import hf_hub_download
dest = os.path.expanduser("~/.ssm/models")
os.makedirs(dest, exist_ok=True)
path = hf_hub_download(
    repo_id="unsloth/Qwen3.5-35B-A3B-GGUF",
    filename="Qwen3.5-35B-A3B-Q4_K_M.gguf",
    local_dir=dest,
    resume_download=True,
)
print(f"Downloaded to: {path}")
EOF
fi

echo ""
echo "=== Download complete ==="
ls -lh "$MODEL_DIR/"*.gguf 2>/dev/null || true
