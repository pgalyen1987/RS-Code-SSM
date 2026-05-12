#!/usr/bin/env bash
# SFT on clean Python coding data.
# Sources: Magicoder-OSS-Instruct-75K + CodeFeedback-Filtered-Instruction
# Downloads from HF on first run, filters to valid Python functions only.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Optional: --init-checkpoint <path>  passed through to sft_reasoning
INIT_CKPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --init-checkpoint) INIT_CKPT="$2"; shift 2 ;;
    *) shift ;;
  esac
done

mkdir -p logs checkpoints/sft

LOG="logs/sft_reasoning_$(date +%Y%m%d_%H%M%S).log"
echo "=== CodingSSM SFT ===" | tee "$LOG"
echo "Log: $LOG" | tee -a "$LOG"

# Step 1: Prepare clean SFT data (downloads from HF if not cached)
echo "[STEP 1] Preparing clean SFT data..." | tee -a "$LOG"
python -u scripts/prepare_sft_data.py \
  --output data/sft_clean.jsonl \
  2>&1 | tee -a "$LOG"

NLINES=$(wc -l < data/sft_clean.jsonl)
echo "[INFO] SFT dataset: $NLINES records" | tee -a "$LOG"

# Step 2: Smoke test (fast sanity check before full training run)
echo "[STEP 2] Running smoke test..." | tee -a "$LOG"
python -u -m train.smoke_test_sft 2>&1 | tee -a "$LOG"

echo "[INFO] Smoke test passed. Starting SFT training..." | tee -a "$LOG"

# Step 3: Full SFT
# H100 80GB: batch 4 × accum 8 = effective 32; seq_len 2048 with grad checkpoint
INIT_ARG=""
if [ -n "$INIT_CKPT" ]; then
  echo "[INFO] Using init checkpoint: $INIT_CKPT" | tee -a "$LOG"
  INIT_ARG="--init-checkpoint $INIT_CKPT"
fi
python -u -m train.sft_reasoning \
  --traces data/sft_clean.jsonl \
  --output-dir checkpoints/sft \
  --model-size 700m \
  --epochs 3 \
  --lr 1e-4 \
  --batch-size 4 \
  --grad-accum 8 \
  --max-seq-len 2048 \
  --save-every 200 \
  $INIT_ARG \
  2>&1 | tee -a "$LOG"

echo "[COLLECT] Saving SFT logs..." | tee -a "$LOG"
python -u scripts/collect_logs.py --sft-log "$LOG" 2>&1 | tee -a "$LOG"
