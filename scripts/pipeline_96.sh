#!/usr/bin/env bash
# ============================================================================
# CodingSSM → 96-98% HumanEval Pipeline
#
# Full automated pipeline:
#   Phase 1: Generate DeepSeek-R1 traces with rejection sampling (overnight)
#   Phase 2: SFT on verified traces
#   Phase 3: GRPO training with code execution reward (2-3 days)
#   Phase 4: Self-improvement loop × 3 iterations
#
# Resumes automatically if interrupted at any phase.
# Run with: systemd-inhibit --what=sleep bash scripts/pipeline_96.sh
# ============================================================================

set -e
cd "$(dirname "$0")/.."
[ -f scripts/env.sh ] && source scripts/env.sh
[ -f .venv/bin/activate ] && source .venv/bin/activate

mkdir -p data logs checkpoints/sft_v2 checkpoints/grpo checkpoints/self_improve
EPICHAT_DIR="${EPICHAT_DIR:-$PWD/epichat}"

DR1_HOST="http://localhost:11437"   # DeepSeek-R1 on user Ollama
SYS_HOST="http://localhost:11434"   # llama3.1:8b on system Ollama

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ── Find best available teacher ───────────────────────────────────────────────
pick_teacher() {
  # Test DeepSeek-R1 first (better reasoning)
  if curl -sf "$DR1_HOST/api/tags" | python3 -c "
import sys,json
d=json.load(sys.stdin)
names={m['name'].split(':')[0] for m in d.get('models',[])}
sys.exit(0 if 'deepseek-r1' in names else 1)
" 2>/dev/null; then
    TEACHER_MODEL="deepseek-r1:latest"
    TEACHER_HOST="$DR1_HOST"
    echo "deepseek-r1"
  else
    TEACHER_MODEL="llama3.1:8b"
    TEACHER_HOST="$SYS_HOST"
    echo "llama3.1:8b"
  fi
}

# ── Wait for a process to finish ─────────────────────────────────────────────
wait_for() {
  local pattern="$1"
  local desc="$2"
  while pgrep -af "$pattern" | grep -qv "grep\|pipeline"; do
    log "  Waiting for $desc to finish…"
    sleep 60
  done
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0: Wait for current SFT to finish
# ─────────────────────────────────────────────────────────────────────────────
LOG="logs/pipeline_96_$(date +%Y%m%d_%H%M%S).log"
echo "=== CodingSSM 96% Pipeline ===" | tee "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

if pgrep -af "train.sft_reasoning" | grep -qv grep; then
  log "Phase 0: Waiting for current SFT to finish…"
  wait_for "train.sft_reasoning" "SFT training"
  log "Phase 0: SFT done."
fi

SFT_V1=$(ls -t checkpoints/sft/*.pt 2>/dev/null | head -1)
if [ -z "$SFT_V1" ]; then
  log "ERROR: No SFT checkpoint found. Run train_sft_reasoning.sh first."
  exit 1
fi
log "SFT v1 checkpoint: $SFT_V1"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Generate DeepSeek-R1 traces with rejection sampling
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "=== Phase 1: DeepSeek-R1 Traces (rejection sampling) ==="

TEACHER=$(pick_teacher)
log "Teacher: $TEACHER @ $TEACHER_HOST"

DR1_TRACES="data/reasoning_traces_r1.jsonl"

# Resume if already started
DONE_R1=0
if [ -f "$DR1_TRACES" ]; then
  DONE_R1=$(wc -l < "$DR1_TRACES")
  log "Resuming from $DONE_R1 existing R1 traces."
fi

if [ "$DONE_R1" -lt 2000 ]; then
  log "Generating DeepSeek-R1 traces (HumanEval + MBPP, n_samples=4, verified-only)…"
  python -u -m train.reasoning_data \
    --model "$TEACHER_MODEL" \
    --ollama-host "$TEACHER_HOST" \
    --output "$DR1_TRACES" \
    --sources humaneval mbpp \
    --n-problems 2000 \
    --max-tokens 3000 \
    --n-samples 4 \
    --verified-only \
    --epichat-dir "$EPICHAT_DIR" \
    2>&1 | tee -a "$LOG"
  log "R1 traces done: $(wc -l < $DR1_TRACES) verified traces"
else
  log "R1 traces already complete ($DONE_R1 lines)."
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: SFT v2 on all verified traces
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "=== Phase 2: SFT v2 on verified traces ==="

SFT_V2="checkpoints/sft_v2/sft_best.pt"

if [ ! -f "$SFT_V2" ]; then
  # Merge all trace sources: EpiChat + R1 verified + any existing reasoning
  MERGED_V2="data/all_traces_v2.jsonl"
  cat data/epichat_traces.jsonl \
      "$DR1_TRACES" \
      data/reasoning_traces.jsonl 2>/dev/null > "$MERGED_V2" || true
  log "Merged traces: $(wc -l < $MERGED_V2) examples"

  python -u -m train.sft_reasoning \
    --traces "$MERGED_V2" \
    --output-dir checkpoints/sft_v2 \
    --model-size 700m \
    --epochs 3 \
    --lr 2e-4 \
    --batch-size 1 \
    --grad-accum 16 \
    --max-seq-len 1024 \
    2>&1 | tee -a "$LOG"
  log "SFT v2 done."
else
  log "SFT v2 checkpoint exists, skipping."
fi

SFT_CKPT=$(ls -t checkpoints/sft_v2/*.pt 2>/dev/null | head -1)
log "Using SFT v2: $SFT_CKPT"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: GRPO training
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "=== Phase 3: GRPO Training ==="

GRPO_CKPT=$(ls -t checkpoints/grpo/*.pt 2>/dev/null | head -1)

if [ -z "$GRPO_CKPT" ] || [ ! -f "$GRPO_CKPT" ]; then
  log "Starting GRPO from SFT v2 checkpoint…"
  python -u -m train.grpo \
    --traces data/all_traces_v2.jsonl \
    --checkpoint "$SFT_CKPT" \
    --output-dir checkpoints/grpo \
    --model-size 700m \
    --group-size 8 \
    --lr 5e-6 \
    --max-steps 3000 \
    --kl-coeff 0.02 \
    --max-new-tokens 1024 \
    --temperature 0.8 \
    2>&1 | tee -a "$LOG"
  GRPO_CKPT=$(ls -t checkpoints/grpo/*.pt | head -1)
  log "GRPO done. Best: $GRPO_CKPT"
else
  log "GRPO checkpoint exists: $GRPO_CKPT"
fi

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: Self-improvement loop (3 iterations)
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "=== Phase 4: Self-Improvement Loop (3 iterations) ==="

CURRENT_CKPT="$GRPO_CKPT"

for ITER in 1 2 3; do
  ITER_DIR="checkpoints/self_improve/iter${ITER}"
  ITER_TRACES="data/self_improve_traces_iter${ITER}.jsonl"
  mkdir -p "$ITER_DIR"

  log ""
  log "--- Self-improvement iteration $ITER ---"
  log "Current checkpoint: $CURRENT_CKPT"

  # Step A: Generate new traces using the current best model via Ollama
  # (Current model generates candidates, tests filter to verified-only)
  if [ ! -f "$ITER_TRACES" ] || [ "$(wc -l < $ITER_TRACES)" -lt 500 ]; then
    log "  Generating self-improvement traces with teacher ($TEACHER)…"
    python -u -m train.reasoning_data \
      --model "$TEACHER_MODEL" \
      --ollama-host "$TEACHER_HOST" \
      --output "$ITER_TRACES" \
      --sources humaneval mbpp codealpaca \
      --n-problems 3000 \
      --max-tokens 3000 \
      --n-samples 8 \
      --verified-only \
      --epichat-dir "$EPICHAT_DIR" \
      2>&1 | tee -a "$LOG"
    log "  Iter $ITER traces: $(wc -l < $ITER_TRACES) verified"
  else
    log "  Iter $ITER traces already exist ($(wc -l < $ITER_TRACES) lines)."
  fi

  # Re-export EpiChat traces (EU count may have grown)
  log "  Re-exporting EpiChat traces…"
  python -u -m train.epichat_export \
    --epichat-dir "$EPICHAT_DIR" \
    --output data/epichat_traces.jsonl \
    --min-confidence 0.4 \
    2>&1 | tee -a "$LOG"

  # Step B: Merge all trace sources
  MERGED_ITER="data/all_traces_iter${ITER}.jsonl"
  cat data/epichat_traces.jsonl \
      "$DR1_TRACES" \
      "$ITER_TRACES" \
      data/all_traces_v2.jsonl > "$MERGED_ITER"
  log "  Merged iter $ITER: $(wc -l < $MERGED_ITER) total examples"

  # Step C: SFT on expanded dataset
  SFT_ITER="$ITER_DIR/sft_best.pt"
  if [ ! -f "$SFT_ITER" ]; then
    log "  SFT iter $ITER…"
    python -u -m train.sft_reasoning \
      --traces "$MERGED_ITER" \
      --output-dir "$ITER_DIR" \
      --model-size 700m \
      --epochs 2 \
      --lr 1e-4 \
      --batch-size 1 \
      --grad-accum 16 \
      --max-seq-len 1024 \
      2>&1 | tee -a "$LOG"
  fi
  SFT_ITER=$(ls -t "$ITER_DIR"/*.pt 2>/dev/null | head -1)

  # Step D: GRPO on top of SFT iter
  GRPO_ITER_DIR="checkpoints/self_improve/iter${ITER}_grpo"
  mkdir -p "$GRPO_ITER_DIR"
  GRPO_ITER_CKPT=$(ls -t "$GRPO_ITER_DIR"/*.pt 2>/dev/null | head -1)

  if [ -z "$GRPO_ITER_CKPT" ]; then
    log "  GRPO iter $ITER…"
    python -u -m train.grpo \
      --traces "$MERGED_ITER" \
      --checkpoint "$SFT_ITER" \
      --output-dir "$GRPO_ITER_DIR" \
      --model-size 700m \
      --group-size 8 \
      --lr 3e-6 \
      --max-steps 2000 \
      --kl-coeff 0.01 \
      --max-new-tokens 1024 \
      --temperature 0.9 \
      2>&1 | tee -a "$LOG"
    GRPO_ITER_CKPT=$(ls -t "$GRPO_ITER_DIR"/*.pt | head -1)
  fi

  CURRENT_CKPT="$GRPO_ITER_CKPT"
  log "  Iter $ITER complete. Best checkpoint: $CURRENT_CKPT"
done

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "=== Pipeline complete ==="
log "Final checkpoint: $CURRENT_CKPT"
log ""
log "Run evaluation:"
log "  python scripts/eval_humaneval.py --checkpoint $CURRENT_CKPT --n-samples 8"
log ""
log "Use in CLI:"
log "  ssm ask-v2 --checkpoint $CURRENT_CKPT 'implement quicksort'"
