# CodingSSM Roadmap — State-of-the-Art Efficiency Target

**Goal:** Best HumanEval pass@1 per parameter at 1.65B–3B scale.
**Target:** 82–88% pass@1 at 3B params (state-of-the-art for this size class).

---

## Phase 1: Code Pretraining ← START HERE

**Notebook:** `notebooks/kaggle_pretrain.ipynb`
**Script:** `train/pretrain.py`
**Target:** 2B tokens of Python code
**Sessions:** ~8–10 Kaggle T4 sessions (resumes automatically)

Train the model on raw Python code before any fine-tuning. This is the single highest-impact step — without it, the model has no knowledge of Python syntax, algorithms, or standard library. The "Phi" papers showed that 1–2B tokens of high-quality code can match models trained on 100B+ tokens of general text.

**Data source:** `bigcode/the-stack` (Python split, streamed — no full download needed)

**Expected impact:** ~50% → ~60% HumanEval after this phase alone.

**Status:** 🔲 Not started

---

## Phase 2: Expanded Reasoning Traces

**Script:** `scripts/fetch_ocr2.py`
**Target:** 50K+ verified reasoning traces in `data/all_traces.jsonl`

Current traces (26,242):
- 21,105 EpiChat knowledge traces
- 137 HumanEval/MBPP reasoning traces
- 5,000 OpenCodeReasoning-2 traces (pass_rate ≥ 0.8)

To add:
- 25,000 more OpenCodeReasoning-2 traces (`--source ocr2 --n 25000`)
- 25,000 Codeforces-CoTs traces (`--source codeforces --n 25000`)
- Complete the HumanEval/MBPP traces notebook (664 problems, `notebooks/kaggle_traces.ipynb`)

```bash
source .venv/bin/activate
python scripts/fetch_ocr2.py --n 50000 --source both
# Then upload data/all_traces.jsonl to HF dataset repo
```

**Expected impact:** Reasoning quality improves significantly with more diverse verified problems.

**Status:** 🔲 Not started (5K done, need 50K+)

---

## Phase 3: SFT on Full Trace Dataset

**Notebook:** `notebooks/kaggle_train.ipynb` (Cells 1–4)
**Checkpoint:** `checkpoints/sft_v2/sft_best.pt` → uploaded to `pgalyen1987/RS-Code-SSM-1.6B`

Run after Phase 1 (pretrain checkpoint) + Phase 2 (full traces):
- Initialize from pretrain checkpoint
- SFT on all_traces.jsonl (50K+ traces)
- 3 epochs, lr=2e-4, batch=1, grad_accum=16
- ~9 hours on 2x T4

**Expected impact:** ~60% (post-pretrain) → ~68–73% HumanEval

**Status:** 🔲 Not started (blocked on Phase 1 + 2)

---

## Phase 4: GRPO on Diverse Coding Problems

**Notebook:** `notebooks/kaggle_train.ipynb` (Cell 5)
**Script:** `train/grpo.py`
**Target:** 3,000+ training steps on verifiable coding problems

Expand GRPO problem set beyond HumanEval/MBPP (664 problems):
- APPS dataset: 5,000 problems with test cases
- CodeContests: 10,000+ competitive programming problems
- Each step: generate 8 solutions → execute → reward passing solutions

Run from SFT checkpoint. Multiple Kaggle sessions, resumes automatically.

**Expected impact:** ~68–73% → ~75–80% HumanEval

**Status:** 🔲 Not started (blocked on Phase 3)

---

## Phase 5: Scale to 3B + Test-Time Compute

### 5a: Scale to 3B params
Use `ModelConfig3B()` in training (same architecture, larger dims).
Fits on 64GB RAM at Q4 quantization (~2GB inference).
Kaggle training requires ~9 hours per SFT session.

### 5b: Test-Time Compute at Inference
At inference: generate N solutions, run lightweight test verification, return first passing solution.
Even a 50% pass@1 model achieves ~94% pass@8.

```python
# In ssm/epichat_rag.py or inference script:
for _ in range(n_samples):
    code = model.generate(prompt)
    if run_tests(code, test_cases):
        return code
```

**Expected impact:** ~75–80% → **82–88% effective pass@1** at 3B

**Status:** 🔲 Not started (blocked on Phase 4)

---

## Phase 6: Verifier Model + 98% Effective Pass@8

**Files:**
- `arch/verifier.py` — VerifierConfig, VerifierConfig100M, VerifierModel (~100M params)
- `train/train_verifier.py` — training script with hard negatives + binary cross-entropy
- `ssm/inference.py` — CodingSSMInference with test-time compute + verifier scoring

**Strategy:**
Generate N candidate solutions, rank them with the verifier, return the highest-scoring one.
No test execution needed at inference time — the verifier replaces the test runner.

**The math (pass@k formula):**
```
pass@k = 1 - (1 - pass@1)^k

pass@1=40%  →  pass@4=87.0%  pass@8=98.3%
pass@1=50%  →  pass@4=93.8%  pass@8=99.6%
pass@1=60%  →  pass@4=97.4%  pass@8=99.9%
```

With Phase 5 targeting 75–80% pass@1, pass@4 already exceeds 99%.
**Target: 98%+ effective pass@8 on HumanEval-X (all 6 languages)**

**Verifier architecture:**
- 6-layer transformer encoder, d_model=512, 8 heads → ~100M params
- Input: `[problem] <sep> [solution]` concatenated, max 1024 tokens
- Output: sigmoid score ∈ (0, 1) — probability solution is correct
- Training: positive = verified solutions from all_traces.jsonl; hard negatives = corrupted/truncated solutions
- Loss: binary cross-entropy

**Train the verifier:**
```bash
source .venv/bin/activate
python -m train.train_verifier \
    --traces data/all_traces.jsonl \
    --output-dir checkpoints/verifier \
    --epochs 5 \
    --device cuda
```

**Use at inference:**
```python
from ssm.inference import CodingSSMInference

engine = CodingSSMInference.from_checkpoint(
    'checkpoints/grpo/best.pt',
    verifier_path='checkpoints/verifier/verifier_best.pt',  # optional
)
result = engine.solve(
    prompt="Write a function that reverses a string",
    language='python',
    n_samples=8,
    use_verifier=True,   # rank by verifier score instead of test execution
)
print(f"Solution: {result.solution}")
print(f"Verifier score: {result.verifier_score:.3f}")
print(f"Pass rate: {result.pass_rate:.1%}")
```

**Kaggle notebook:** Add Cell 7b to `notebooks/kaggle_train.ipynb` after GRPO:
```python
subprocess.run([sys.executable, '-m', 'train.train_verifier',
    '--traces', traces, '--output-dir', 'checkpoints/verifier',
    '--device', DEVICE], check=True)
```

**Expected impact:** Enables **98%+ effective pass@8** across all 6 languages (Python, C++, Java, JS, Go, Kotlin)
at only 75–80% base pass@1. The verifier also allows laptop deployment without a test runner.

**Status:** 🔲 Not started (blocked on Phase 5) — code complete, needs training

---

## Current State (as of 2026-03-17)

| Component | Status |
|---|---|
| Architecture (Mamba-2 + MoE + LoRA) | ✅ Complete |
| EpiChat traces (21K) | ✅ On HF at `pgalyen1987/RS-Code-SSM-1.6B` |
| OCR2 reasoning traces (5K) | ✅ In `data/all_traces.jsonl` |
| all_traces.jsonl (26K) | ⚠️ Needs upload to HF dataset repo |
| kaggle_eus.ipynb | ✅ Working |
| kaggle_traces.ipynb | ✅ Working (multilingual, resumes, merges, uploads) |
| kaggle_train.ipynb | ✅ Working (OOM fixed, auth fixed) |
| kaggle_pretrain.ipynb | ✅ Ready to run |
| train/pretrain.py | ✅ Multi-language (8 langs, round-robin, resume) |
| train/grpo.py | ✅ Multilingual code execution (6 languages) |
| ssm/inference.py | ✅ Test-time compute + verifier scoring |
| arch/verifier.py | ✅ 100M verifier model architecture |
| train/train_verifier.py | ✅ Verifier training script |
| Phase 1 pretraining | 🔲 Not started |
| Phase 2 expanded traces | 🔲 Not started |

## Key Files

```
arch/                — Model architecture (Mamba-2, MoE, LoRA, SparseAttn)
arch/verifier.py     — 100M verifier for best-of-N ranking (Phase 6)
train/               — Training scripts
  pretrain.py        — Phase 1: multi-language code pretraining
  sft_reasoning.py   — Phase 3: SFT on reasoning traces
  grpo.py            — Phase 4: GRPO with multilingual code execution
  train_verifier.py  — Phase 6: verifier model training
ssm/inference.py     — Inference with test-time compute + verifier
scripts/             — Data prep (fetch_ocr2.py, merge_traces.py)
notebooks/           — Kaggle notebooks (pretrain, eus, traces, train)
data/                — Local trace files
```

## HuggingFace Repos

- Model: `pgalyen1987/RS-Code-SSM-1.6B` (weights, epichat_traces.jsonl)
- Dataset: `pgalyen1987/rs-code-ssm-traces` (all_traces.jsonl, reasoning_traces_r1.jsonl)
