#!/usr/bin/env bash
# Generate reasoning traces for CodingSSM training.
# Step 1: Export EpiChat knowledge graph → data/epichat_traces.jsonl (fast, no teacher)
# Step 2: Generate teacher reasoning traces → data/reasoning_traces.jsonl (slow, overnight)
#
# Resumes automatically if interrupted.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p data logs

LOG="logs/traces_$(date +%Y%m%d_%H%M%S).log"
echo "=== CodingSSM Trace Generation ===" | tee "$LOG"
echo "EpiChat:  /home/me/EpiChat" | tee -a "$LOG"
echo "Teacher:  Ollama HTTP API (auto-selects best available model)" | tee -a "$LOG"
echo "Log:      $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Step 1: EpiChat export (fast, ~30s, no teacher needed) ──────────────────
echo "=== Step 1: Exporting EpiChat knowledge ===" | tee -a "$LOG"
python -u -m train.epichat_export \
  --epichat-dir /home/me/EpiChat \
  --output data/epichat_traces.jsonl \
  --min-confidence 0.4 \
  2>&1 | tee -a "$LOG"

EPICHAT_COUNT=$(wc -l < data/epichat_traces.jsonl 2>/dev/null || echo 0)
echo "[INFO] EpiChat traces: $EPICHAT_COUNT" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Step 2: Teacher reasoning traces (via Ollama HTTP API) ───────────────────
echo "=== Step 2: Teacher reasoning traces ===" | tee -a "$LOG"
echo "[INFO] Teacher: Ollama (auto-selects qwen35-teacher or llama3.1:8b)" | tee -a "$LOG"

python -u -m train.reasoning_data \
  --output data/reasoning_traces.jsonl \
  --n-problems 5000 \
  --sources humaneval mbpp codealpaca \
  --max-tokens 2048 \
  --epichat-dir /home/me/EpiChat \
  2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== All traces complete ===" | tee -a "$LOG"
TOTAL=$(( $(wc -l < data/epichat_traces.jsonl 2>/dev/null || echo 0) + \
          $(wc -l < data/reasoning_traces.jsonl 2>/dev/null || echo 0) ))
echo "[DONE] Total training examples: $TOTAL" | tee -a "$LOG"
