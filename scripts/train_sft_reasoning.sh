#!/usr/bin/env bash
# SFT on reasoning traces (after gen_traces.sh finishes).
# Trains student to produce <think>...</think> + code.
# Run AFTER data/reasoning_traces.jsonl exists.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p logs checkpoints/sft

LOG="logs/sft_reasoning_$(date +%Y%m%d_%H%M%S).log"
echo "=== CodingSSM SFT on Reasoning Traces ===" | tee "$LOG"
echo "Datasets:   data/epichat_traces.jsonl + data/reasoning_traces.jsonl" | tee -a "$LOG"
echo "Log:        $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Merge both trace files into one for training
MERGED="data/all_traces.jsonl"
cat data/epichat_traces.jsonl data/reasoning_traces.jsonl > "$MERGED" 2>/dev/null || \
  cp data/epichat_traces.jsonl "$MERGED"  # fallback if teacher traces don't exist yet
echo "[INFO] Merged traces: $(wc -l < $MERGED) examples" | tee -a "$LOG"

python -u -m train.sft_reasoning \
  --traces "$MERGED" \
  --output-dir checkpoints/sft \
  --model-size 700m \
  --epochs 3 \
  --lr 3e-4 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-seq-len 2048 \
  2>&1 | tee -a "$LOG"
