# RS-Code-SSM (CodingSSM)

**RS-Code-SSM** (Reasoning State-Space Model for Code) is a ~1.65B-parameter hybrid **Mamba-2 + sparse attention** decoder for **multilingual code generation**, designed to run **fully offline** on consumer hardware (CPU or a single GPU). This README tracks the **paper** (`PAPER.md`); treat that document as the source of truth for methods and claims.

---

## Targets (evaluation)

| Setting | Target (paper) |
|--------|----------------|
| **HumanEval-X** pass@1 | ~75% |
| **HumanEval-X** pass@8 (test-time compute) | ~98% |

*pass@k: generate up to k candidates; success if any passes the official tests. See `PAPER.md` §5.2 and `ssm/test_time_compute.py`.*

---

## Architecture (summary — see `PAPER.md` §3)

| Component | Detail |
|-----------|--------|
| Core | Mamba-2 SSD blocks |
| Attention | Sparse sliding window (every **6th** layer in the full config, window **512**) |
| FFN | Mixture-of-Experts + dense layers (MoE on even layers in default config) |
| Shared weights | Zamba2-style shared attention + per-layer LoRA |
| Parameters | ~**1.65B** total, ~**800M** active per token (MoE sparsity) |
| Tokenizer | Qwen2.5 (**152,064** vocab) |

**Note:** Smaller bring-up configs (e.g. `ModelConfig700M()` in `arch/config.py`) use reduced dims and may differ on `attn_every_n` / expert budgets. The **paper** describes the intended 1.65B layout; **code** is authoritative for exact hyperparameters.

---

## Training pipeline (paper §4)

Stages build on each other; see `PAPER.md` for full detail.

1. **EpiChat SFT** — supervised fine-tuning on epistemically grounded traces from the EpiChat knowledge graph (`train/epichat_export.py`, merged traces in `data/`).
2. **RFT (rejection-sampling fine-tuning)** — verified DeepSeek-R1–style chain-of-thought traces on coding tasks (**7 languages** in the full pipeline: Python, C++, Java, JavaScript, TypeScript, Go, Rust); only **executing** solutions are kept.
3. **GRPO** — Group Relative Policy Optimization with **binary code-execution** reward (`train/grpo.py`).
4. **Test-time compute + verifier** — best-of-N with execution; optional **~100M-parameter verifier** to rank candidates when tests are unavailable (`arch/verifier.py`, `train/train_verifier.py`, `PAPER.md` §5.2).

**Self-improvement (§4.4):** after GRPO, the paper describes **3 iterations** of generate → filter → merge traces → SFT → GRPO with tuned KL. Automating this end-to-end is optional tooling on top of the core stages.

**Kaggle GPU path:** `notebooks/kaggle_rs_code_ssm_full.ipynb` — single notebook aligned with **PAPER §4**: optional pretrain, trace download, **SFT** (with optional `--init-checkpoint` from pretrain), **GRPO**, optional self-improve & verifier, export/upload. Legacy split notebooks: `kaggle_train.ipynb`, `kaggle_pretrain.ipynb`.

---

## Inference & benchmarks

| Piece | Role |
|-------|------|
| `ssm/inference_sft.py` | Load SFT/GRPO `.pt` checkpoint; `ask` / `stream` / `complete` |
| `ssm/test_time_compute.py` | **TTCInference** — sample N solutions, run tests, return first pass |
| `scripts/eval_humaneval.py` | HumanEval-style eval with pass@k |
| `train/eval.py` | HumanEval / MBPP loaders + sandboxed execution helpers |

There is **no** guarantee that a single training run matches the paper’s headline numbers; always **measure** with the scripts above on your checkpoint.

**CLI (optional):** after `pip install -e .`, see `ssm` subcommands in `cli.py` (e.g. `ask-v2`, `chat-v2` where implemented).

---

## Why RS-Code-SSM vs “README marketing”

Older one-page READMEs listed a **Python-only** HumanEval/MBPP table and simplified architecture (e.g. attention every 4th layer, top-4 MoE only). **This file defers to `PAPER.md`** for multilingual scope, **HumanEval-X** targets, verifier-based deployment, and exact training narrative.

---

## Hardware (paper / typical)

| Config | RAM | Speed (indicative) |
|--------|-----|---------------------|
| CPU inference | ~4–8 GB | ~1–2 tok/s |
| Kaggle T4 training | fits notebook settings | SFT ~tens of minutes; GRPO hours |

---

## License

Apache 2.0 — free to use, modify, and distribute.
