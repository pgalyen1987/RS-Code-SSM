#!/usr/bin/env bash
# Stage 1: CodingSSM architecture bringup on CodeAlpaca-20k
# Runs pure CE loss initially (no teacher needed) — alpha=0, beta=1
# Once teacher finishes downloading, re-run with --logit-cache to add KD.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOG="logs/stage1_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs checkpoints/stage1

echo "=== CodingSSM Stage 1 Training ===" | tee "$LOG"
echo "Dataset:    CodeAlpaca-20k" | tee -a "$LOG"
echo "Loss:       CE only (teacher not yet downloaded)" | tee -a "$LOG"
echo "Log:        $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

python -m train.distill \
  --stage 1 \
  --batch-size 2 \
  --epochs 3 \
  --output-dir checkpoints/stage1 \
  2>&1 | tee -a "$LOG"
