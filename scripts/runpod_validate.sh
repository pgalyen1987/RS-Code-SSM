#!/usr/bin/env bash
# Budget-capped RunPod validation: prove SFT + GRPO both work at 700M scale on
# GPU before committing to a full multi-hour run. Designed to finish in
# ~30-45 min (a few dollars), NOT to produce a good model.
#
# What it does:
#   1. Build a SAMPLE of clean SFT data (fast, no full 230K download)
#   2. SFT (700m, bf16) with a hard time limit
#   3. SFT quality check (does it emit runnable Python?)
#   4. Build a small MBPP problem set
#   5. GRPO (700m, bf16) for a handful of steps — confirms rollouts, reward
#      execution, and policy updates all run on GPU
#
# It deliberately SKIPS the full HumanEval/MBPP/BigCodeBench benchmark sweep.
#
# Tunables (env vars):
#   SFT_SAMPLE        per-source SFT records       (default 3000 -> ~6k total)
#   SFT_MINUTES       SFT wall-clock cap           (default 25)
#   GRPO_STEPS        GRPO steps                    (default 40)
#   GRPO_PROBLEMS     MBPP problems for GRPO        (default 64)
#
# Usage (on the pod):
#   cd RS-Code-SSM && bash scripts/runpod_validate.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

SFT_SAMPLE="${SFT_SAMPLE:-3000}"
SFT_MINUTES="${SFT_MINUTES:-25}"
GRPO_STEPS="${GRPO_STEPS:-40}"
GRPO_PROBLEMS="${GRPO_PROBLEMS:-64}"

mkdir -p logs checkpoints/sft checkpoints/grpo data
rm -f /tmp/.ssm_sft_train.lock /tmp/.ssm_grpo_train.lock

LOG="logs/runpod_validate_$(date +%Y%m%d_%H%M%S).log"
echo "=== RunPod Validation (SFT + GRPO, capped) ===" | tee "$LOG"
echo "SFT_SAMPLE=$SFT_SAMPLE SFT_MINUTES=$SFT_MINUTES GRPO_STEPS=$GRPO_STEPS GRPO_PROBLEMS=$GRPO_PROBLEMS" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | tee -a "$LOG" || true
echo "" | tee -a "$LOG"

# ── 1. Sample of clean SFT data ───────────────────────────────────────────────
echo "[1/5] Preparing sampled SFT data..." | tee -a "$LOG"
python -u scripts/prepare_sft_data.py \
  --sample "$SFT_SAMPLE" \
  --output data/sft_validate.jsonl \
  2>&1 | tee -a "$LOG"

# ── 2. SFT (700m) with a hard time cap ───────────────────────────────────────
echo "[2/5] SFT (700m, bf16, ${SFT_MINUTES}min cap)..." | tee -a "$LOG"
python -u -m train.sft_reasoning \
  --traces data/sft_validate.jsonl \
  --output-dir checkpoints/sft \
  --model-size 700m \
  --epochs 1 \
  --lr 1e-4 \
  --batch-size 4 \
  --grad-accum 8 \
  --max-seq-len 1024 \
  --save-every 100 \
  --time-limit-minutes "$SFT_MINUTES" \
  --min-steps 50 \
  2>&1 | tee -a "$LOG"

SFT_CKPT="checkpoints/sft/sft_latest.pt"
[ -f "$SFT_CKPT" ] || { echo "[FAIL] no SFT checkpoint produced" | tee -a "$LOG"; exit 1; }

# ── 3. SFT quality check ─────────────────────────────────────────────────────
echo "[3/5] SFT quality check..." | tee -a "$LOG"
python -u scripts/check_sft_quality.py \
  --checkpoint "$SFT_CKPT" --model-size 700m \
  2>&1 | tee -a "$LOG" || echo "[WARN] quality check non-zero (early SFT may not pass tests yet)" | tee -a "$LOG"

# ── 4. Small MBPP problem set ────────────────────────────────────────────────
echo "[4/5] Building MBPP problem set ($GRPO_PROBLEMS)..." | tee -a "$LOG"
python -u scripts/gen_grpo_problems.py \
  --sample "$GRPO_PROBLEMS" \
  --output data/grpo_problems.jsonl \
  2>&1 | tee -a "$LOG"

# ── 5. GRPO (700m), a few steps ──────────────────────────────────────────────
echo "[5/5] GRPO (700m, bf16, ${GRPO_STEPS} steps)..." | tee -a "$LOG"
python -u -m train.grpo \
  --traces data/grpo_problems.jsonl \
  --checkpoint "$SFT_CKPT" \
  --output-dir checkpoints/grpo \
  --model-size 700m \
  --languages python \
  --group-size 8 \
  --lr 5e-6 \
  --max-steps "$GRPO_STEPS" \
  --kl-coeff 0.04 \
  --max-new-tokens 512 \
  --temperature 0.8 \
  2>&1 | tee -a "$LOG"

[ -n "$(ls -t checkpoints/grpo/*.pt 2>/dev/null | head -1)" ] \
  || { echo "[FAIL] no GRPO checkpoint produced" | tee -a "$LOG"; exit 1; }

echo "" | tee -a "$LOG"
echo "=== VALIDATION PASSED — SFT + GRPO both run on GPU at 700M scale ===" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
echo "If this looks healthy, run the full pipeline:" | tee -a "$LOG"
echo "  bash scripts/train_sft_reasoning.sh && bash scripts/train_grpo.sh" | tee -a "$LOG"
