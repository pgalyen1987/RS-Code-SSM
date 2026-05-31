#!/usr/bin/env bash
# End-to-end local test: SFT (tiny model, synthetic data) → GRPO (3 steps).
# Completes in ~5 minutes on CPU. Validates the full pipeline before RunPod.
#
# Usage:
#   cd RS-Code-SSM
#   bash scripts/local_test.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

LOG="logs/local_test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs checkpoints/local_sft checkpoints/local_grpo data
echo "=== Local End-to-End Test ===" | tee "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Step 0: unit smoke test ───────────────────────────────────────────────────
echo "[STEP 0] Smoke test (unit)..." | tee -a "$LOG"
python -u -m train.smoke_test_sft 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Step 1: generate synthetic coding dataset ─────────────────────────────────
echo "[STEP 1] Generating synthetic coding dataset..." | tee -a "$LOG"
python -u scripts/gen_local_test_data.py 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Step 2: SFT (tiny model, 3 epochs) ───────────────────────────────────────
echo "[STEP 2] SFT training (tiny model, 3 epochs)..." | tee -a "$LOG"
python -u -m train.sft_reasoning \
  --traces data/local_test.jsonl \
  --output-dir checkpoints/local_sft \
  --model-size tiny \
  --epochs 3 \
  --lr 5e-4 \
  --batch-size 2 \
  --grad-accum 4 \
  --max-seq-len 256 \
  --save-every 5 \
  2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Verify checkpoint was saved
SFT_CKPT="checkpoints/local_sft/sft_latest.pt"
if [ ! -f "$SFT_CKPT" ]; then
  echo "[FAIL] No SFT checkpoint at $SFT_CKPT" | tee -a "$LOG"
  exit 1
fi
echo "[PASS] SFT checkpoint: $SFT_CKPT" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Step 2b: SFT quality check (same prompting GRPO uses) ─────────────────────
echo "[STEP 2b] SFT quality check..." | tee -a "$LOG"
python -u scripts/check_sft_quality.py \
  --checkpoint "$SFT_CKPT" \
  --model-size tiny \
  --device cpu \
  2>&1 | tee -a "$LOG" || echo "[WARN] quality check non-zero exit (expected for tiny)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Step 3: GRPO (tiny model, 3 steps) ───────────────────────────────────────
echo "[STEP 3] GRPO training (tiny model, 3 steps)..." | tee -a "$LOG"
python -u -m train.grpo \
  --traces data/local_test.jsonl \
  --checkpoint "$SFT_CKPT" \
  --output-dir checkpoints/local_grpo \
  --model-size tiny \
  --group-size 4 \
  --grad-accum 1 \
  --lr 5e-6 \
  --max-steps 3 \
  --kl-coeff 0.04 \
  --max-new-tokens 64 \
  --temperature 0.9 \
  --save-every 3 \
  2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "=== LOCAL TEST PASSED ===" | tee -a "$LOG"
echo "SFT checkpoint: $SFT_CKPT" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
