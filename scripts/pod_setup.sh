#!/usr/bin/env bash
# RunPod H100 setup: clone, install, download data, run SFT → GRPO in tmux.
#
# Usage (fresh RunPod terminal):
#   bash <(curl -fsSL https://raw.githubusercontent.com/pgalyen1987/RS-Code-SSM/main/scripts/pod_setup.sh)
# OR after cloning:
#   bash scripts/pod_setup.sh

set -e
REPO_DIR="/workspace/RS-Code-SSM"
REPO_URL="https://github.com/pgalyen1987/RS-Code-SSM.git"

echo "=== RS-Code-SSM RunPod Setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo ""

# ── Clone / pull repo ─────────────────────────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/5] Updating existing repo..."
    cd "$REPO_DIR" && git pull --ff-only
else
    echo "[1/5] Cloning RS-Code-SSM..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# ── Python env ────────────────────────────────────────────────────────────────
echo "[2/5] Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet

# PyTorch with CUDA (H100 = CUDA 12.1 / 12.4; cu121 wheel works for both)
echo "  Installing PyTorch+CUDA..."
pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet

# Project deps (no llama-cpp — not needed for training)
pip install transformers tokenizers huggingface_hub safetensors \
            faiss-cpu sentence-transformers \
            datasets --quiet

# Mamba-2 fast kernel (optional — falls back to pure-Python if unavailable)
pip install causal-conv1d mamba-ssm --quiet 2>/dev/null || \
    echo "  [warn] mamba-ssm CUDA kernel unavailable — pure-Python fallback"

# Install package in editable mode
pip install -e . --quiet --no-deps

echo "[2/5] Python environment ready."

# ── Download training data ────────────────────────────────────────────────────
echo "[3/5] Preparing training data..."
mkdir -p data logs checkpoints/sft checkpoints/grpo

# prepare_sft_data.py auto-downloads from HF if files not found
python -u scripts/prepare_sft_data.py \
    --hf-repo pgalyen1987/rs-code-ssm-traces \
    --output data/sft_clean.jsonl \
    --max-per-source 10000

NLINES=$(wc -l < data/sft_clean.jsonl)
echo "[3/5] SFT dataset: $NLINES records → data/sft_clean.jsonl"

# ── Smoke test ────────────────────────────────────────────────────────────────
echo "[4/5] Running smoke test..."
python -u -m train.smoke_test_sft
echo "[4/5] Smoke test passed."

# ── Launch training in tmux ───────────────────────────────────────────────────
echo "[5/5] Launching SFT → GRPO pipeline in tmux session 'train'..."

# Kill any stale session
tmux kill-session -t train 2>/dev/null || true

tmux new-session -d -s train -x 220 -y 50

# SFT then GRPO, chained
tmux send-keys -t train "
source .venv/bin/activate && \
bash scripts/train_sft_reasoning.sh && \
bash scripts/train_grpo.sh
" Enter

echo ""
echo "=== DONE ==="
echo "Training running in tmux session 'train'."
echo "Attach with:  tmux attach -t train"
echo "Detach with:  Ctrl-b d"
echo ""
echo "Logs in:      logs/"
echo "Checkpoints:  checkpoints/sft/  checkpoints/grpo/"
