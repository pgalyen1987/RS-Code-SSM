#!/usr/bin/env bash
# GRPO training on code execution reward.
# Run AFTER train_sft_reasoning.sh finishes.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p logs checkpoints/grpo

# Find best SFT checkpoint
SFT_CKPT=$(ls -t checkpoints/sft/*.pt 2>/dev/null | head -1)
if [ -z "$SFT_CKPT" ]; then
  echo "ERROR: No SFT checkpoint found in checkpoints/sft/. Run train_sft_reasoning.sh first."
  exit 1
fi

LOG="logs/grpo_$(date +%Y%m%d_%H%M%S).log"
echo "=== CodingSSM GRPO Training ===" | tee "$LOG"
echo "SFT checkpoint: $SFT_CKPT" | tee -a "$LOG"
echo "Dataset:        data/reasoning_traces.jsonl" | tee -a "$LOG"
echo "Log:            $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

python -u -m train.grpo \
  --traces data/reasoning_traces.jsonl \
  --checkpoint "$SFT_CKPT" \
  --output-dir checkpoints/grpo \
  --model-size 700m \
  --group-size 8 \
  --lr 5e-6 \
  --max-steps 2000 \
  --kl-coeff 0.04 \
  --max-new-tokens 1024 \
  --temperature 0.8 \
  2>&1 | tee -a "$LOG"
