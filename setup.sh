#!/usr/bin/env bash
set -e

# SSM Coder setup script
# Run this once: bash setup.sh

PYTHON=${PYTHON:-python3}

echo "==> Checking Python version..."
$PYTHON -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'" || {
    echo "ERROR: Python 3.10 or newer is required."
    exit 1
}

echo "==> Creating virtual environment (.venv)..."
$PYTHON -m venv .venv

echo "==> Activating .venv..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip --quiet

echo "==> Installing dependencies..."
pip install -e . --quiet

echo ""
echo "============================================"
echo " SSM Coder installed successfully!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Download a RWKV-7 World model from HuggingFace:"
echo "   https://huggingface.co/BlinkDL"
echo ""
echo "   Recommended: RWKV-7-World-7B.pth  (~14 GB in fp16)"
echo "   Faster/smaller: RWKV-7-World-1.6B.pth  (~3 GB)"
echo ""
echo "   Example with huggingface-cli:"
echo "   pip install huggingface-hub"
echo "   huggingface-cli download BlinkDL/rwkv-7-world RWKV-x070-World-7B-v2.8-20241210-ctx4096.pth"
echo ""
echo "3. Configure SSM Coder:"
echo "   ssm init --model /path/to/RWKV-7-World-7B.pth"
echo ""
echo "4. (Optional) Index your codebase:"
echo "   ssm index /path/to/your/project"
echo ""
echo "5. Start chatting:"
echo "   ssm chat"
echo ""
echo "Other commands:"
echo "   ssm ask 'How do I implement a binary search in Python?'"
echo "   ssm complete myfile.py"
echo "   ssm status"
echo ""
echo "Fine-tuning (optional, takes hours on CPU):"
echo "   python finetune/prepare_data.py --source local --dir . --out data/train.jsonl"
echo "   python finetune/state_tune.py --model /path/to/model.pth --data data/train.jsonl"
echo ""
