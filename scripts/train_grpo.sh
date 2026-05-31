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
echo "Dataset:        data/grpo_problems.jsonl" | tee -a "$LOG"
echo "Log:            $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Build the GRPO problem set (MBPP, with executable tests) if missing.
if [ ! -s data/grpo_problems.jsonl ]; then
  echo "[STEP 0] Generating GRPO problems (MBPP)..." | tee -a "$LOG"
  python -u scripts/gen_grpo_problems.py --output data/grpo_problems.jsonl 2>&1 | tee -a "$LOG"
fi

# Quality check: verify the SFT model can generate plausible code before GRPO
echo "[PRE-CHECK] Checking SFT quality..." | tee -a "$LOG"
python -u scripts/check_sft_quality.py --checkpoint "$SFT_CKPT" --model-size 700m 2>&1 | tee -a "$LOG"
echo "[PRE-CHECK] Done." | tee -a "$LOG"

python -u -m train.grpo \
  --traces data/grpo_problems.jsonl \
  --checkpoint "$SFT_CKPT" \
  --output-dir checkpoints/grpo \
  --model-size 700m \
  --languages python \
  --group-size 8 \
  --lr 5e-6 \
  --max-steps 2000 \
  --kl-coeff 0.04 \
  --max-new-tokens 1024 \
  --temperature 0.8 \
  2>&1 | tee -a "$LOG"

# Run benchmarks after GRPO completes
GRPO_CKPT=$(ls -t checkpoints/grpo/*.pt 2>/dev/null | head -1)
if [ -n "$GRPO_CKPT" ]; then
  echo "" | tee -a "$LOG"
  echo "=== Running Benchmarks ===" | tee -a "$LOG"
  python -u scripts/run_benchmarks.py \
    --checkpoint "$GRPO_CKPT" \
    --model-size 700m \
    --benchmarks humaneval mbpp bigcodebench \
    --output "runs/bench_grpo_$(date +%Y%m%d_%H%M%S).json" \
    2>&1 | tee -a "$LOG"
else
  echo "WARNING: No GRPO checkpoint found — skipping benchmarks." | tee -a "$LOG"
fi

echo "[COLLECT] Saving run logs..." | tee -a "$LOG"
python -u scripts/collect_logs.py 2>&1 | tee -a "$LOG"
