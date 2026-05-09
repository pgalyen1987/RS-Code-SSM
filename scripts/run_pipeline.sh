#!/usr/bin/env bash
# Pipeline watchdog: wait for SFT, then run paper-config GRPO,
# then benchmark on HumanEval + MBPP + BigCodeBench.
# Polls every 60s; idempotent because checkpoint presence gates each phase.

set -u
cd "$(dirname "$0")/.."

mkdir -p logs runs checkpoints/grpo_clean

PIPELINE_LOG="logs/pipeline.log"
log() { echo "[$(date "+%Y-%m-%d %H:%M:%S")] $*" | tee -a "$PIPELINE_LOG"; }

SFT_CKPT="checkpoints/sft_clean/sft_latest.pt"
GRPO_DIR="checkpoints/grpo_clean"
GRPO_CKPT="$GRPO_DIR/grpo_latest.pt"
GRPO_TRACES="data/grpo_problems_clean.jsonl"

log "=== run_pipeline.sh starting (pid $$) ==="

# Wait until the SFT lockfile disappears (sft_reasoning unlinks it on exit).
# If the lock is missing AND a sft_latest.pt exists with size > 1GB, treat as done.
while true; do
  if [ ! -e /tmp/.ssm_sft_train.lock ]; then
    if [ -f "$SFT_CKPT" ] && [ "$(stat -c %s "$SFT_CKPT")" -gt 1073741824 ]; then
      break
    fi
    log "SFT lock gone but checkpoint missing/too small. Waiting another 60s."
  fi
  sleep 60
done

if [ ! -f "$SFT_CKPT" ]; then
  log "ERROR: SFT process exited but $SFT_CKPT does not exist. Aborting."
  exit 1
fi

log "Phase 1 done: SFT finished, checkpoint present ($(du -h "$SFT_CKPT" | cut -f1))."
tail -n 5 logs/sft_clean.log 2>/dev/null | tee -a "$PIPELINE_LOG"

log ""
log "=== Phase 2: GRPO (paper config kl-coeff=0.02 max-new-tokens=1024) ==="
if [ -f "$GRPO_CKPT" ]; then
  log "$GRPO_CKPT already exists, skipping GRPO."
else
  rm -f /tmp/.ssm_sft_train.lock 2>/dev/null
  python -u -m train.grpo \
    --traces "$GRPO_TRACES" \
    --checkpoint "$SFT_CKPT" \
    --output-dir "$GRPO_DIR" \
    --model-size 3b \
    --group-size 8 \
    --lr 5e-6 \
    --max-steps 2000 \
    --kl-coeff 0.02 \
    --max-new-tokens 1024 \
    --temperature 0.8 \
    2>&1 | tee -a logs/grpo_clean.log
  GRPO_RC=${PIPESTATUS[0]}
  log "GRPO exit code: $GRPO_RC"
fi

EVAL_CKPT=""
if [ -f "$GRPO_CKPT" ]; then
  EVAL_CKPT="$GRPO_CKPT"
elif [ -f "$SFT_CKPT" ]; then
  EVAL_CKPT="$SFT_CKPT"
  log "WARNING: GRPO checkpoint missing, falling back to SFT checkpoint for eval."
fi

if [ -z "$EVAL_CKPT" ]; then
  log "ERROR: no checkpoint to evaluate. Exiting."
  exit 1
fi

log ""
log "=== Phase 3: Benchmarks (humaneval + mbpp + bigcodebench) ==="
log "Evaluating $EVAL_CKPT"

OUT_JSON="runs/bench_$(basename "$EVAL_CKPT" .pt)_$(date +%Y%m%d_%H%M%S).json"
python -u scripts/run_benchmarks.py \
  --checkpoint "$EVAL_CKPT" \
  --model-size 3b \
  --benchmarks humaneval mbpp bigcodebench \
  --max-new-tokens 768 \
  --temperature 0.0 \
  --output "$OUT_JSON" \
  2>&1 | tee -a logs/bench.log
log "Wrote $OUT_JSON"

log "=== Pipeline complete ==="
log "Final checkpoint: $EVAL_CKPT"
log "Results: $OUT_JSON"
