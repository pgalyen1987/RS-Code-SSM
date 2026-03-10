# CodingSSM: Architecture Design

A novel hybrid SSM architecture for coding assistance, runnable on a single CPU machine.
Based on a systematic literature review of Mamba-2, Jamba, Zamba2, MoE, and distillation research.

---

## 1. Motivation & Gap

Existing hybrid models leave a specific design space unexplored:

| Model     | SSM        | Attention         | MoE | Shared Attn | Notes                              |
|-----------|------------|-------------------|-----|-------------|------------------------------------|
| Jamba     | Mamba-1    | Full (1:7)        | Yes | No          | 52B total, 12B active              |
| Jamba-1.5 | Mamba-1    | Full (1:7)        | Yes | No          | 398B total, 94B active             |
| Zamba2    | Mamba-2    | Full shared (1:6) | No  | Yes + LoRA  | 1.2B / 2.7B / 7.4B                 |
| RWKV-7    | Pure SSM   | None              | No  | No          | Linear RNN, no attention at all    |
| Mixtral   | None       | Full quadratic    | Yes | No          | 47B total, 13B active              |

**Key insight from Jamba-1.5 paper**: "Mamba-1 + full attention outperforms Mamba-2 + full attention" because full attention eliminates the long-range retrieval problem that Mamba-2's larger state is designed to solve.

**Our insight**: Use Mamba-2 + *sparse local* attention instead. Sparse local attention (window=512) handles short-range structure but leaves the medium-to-long range work to Mamba-2's state, preserving the advantage of the larger state size. This combination has not been cleanly published.

---

## 2. Architecture: CodingSSM

### 2.1 Overview

```
CodingSSM-3B
  Total parameters:  ~3.0B
  Active per token:  ~800M
  Context length:    Unlimited (SSM = constant memory per step)
  Target hardware:   64GB RAM, CPU-only

  Embedding: vocab=65536, d_model=2048

  24 Blocks:
    Layers 0,1,2:   Mamba-2 block
    Layer 3:        SparseLocalAttention(window=512) [SHARED weights + LoRA]
    Layers 4,5,6:   Mamba-2 block
    Layer 7:        SparseLocalAttention(window=512) [SHARED weights + LoRA]
    ... (repeat pattern x6)

    All 24 layers:  MoE FFN on even layers, Dense FFN on odd layers

  LM Head (tied to embedding weights)
```

### 2.2 Mamba-2 Block (SSD)

Based on: Dao & Gu, "Transformers are SSMs" (ICML 2024). [arxiv:2405.21060]

The core computation is the Structured State Space Duality (SSD) layer:

```
Input X: (batch, length, n_heads, d_head)
Gates A: (batch, length, n_heads)          -- scalar decay, -exp parameterization
Keys  B: (batch, length, n_heads, d_state)
Vals  C: (batch, length, n_heads, d_state)
Output Y: (batch, length, n_heads, d_head)

Computation (chunk-based, chunk_size=256):
  For each chunk:
    L = exp(segsum(A))                     -- lower-triangular decay matrix
    Y_diag = einsum('bhcl,bhcls,bhcld->bhcd', L, B, X)   -- intra-chunk
    h = einsum('bhcl,bhcld->bhds', A_cumsum, B*X)        -- state update
    Y_state = einsum('bhds,bhcds->bhcd', h_prev, C)      -- inter-chunk
    Y = Y_diag + Y_state
```

**Hyperparameters** (our 3B model):
```python
d_model    = 2048
n_heads    = 32        # Mamba-2 heads (head_dim = d_model/n_heads = 64)
d_state    = 128       # SSM state size per head
d_conv     = 4         # local convolution width
expand     = 2         # inner dim = d_model * expand = 4096
chunk_size = 256
n_groups   = 8         # grouped B/C projections
```

**vs Mamba-1**: A is scalar-times-identity (vs diagonal), enables 2-8x faster hardware parallel scan via block matrix decomposition.

Full block structure:
```
x → Norm → [Linear proj to z, x, B, C, dt] → Conv1d → SSD → Gate(z) → Linear proj → residual
```

### 2.3 Sparse Local Attention Block

**Why sparse, not full**:
- Full attention gives O(n²) memory and eliminates Mamba-2's state advantage
- Window attention (size 512) handles local structure; Mamba-2 handles the rest
- KV cache only needed for attention layers: at 1:6 ratio, 6x smaller than pure transformer

**Shared weights + LoRA** (from Zamba2 [arxiv:2411.15242]):
```
Two shared attention blocks interleaved throughout the model (ABAB pattern).
Each occurrence gets its own LoRA adapters (rank=64) for query and value.

SharedAttn weights: 1 set of Q,K,V,O projections
LoRA per-layer:     dQ_i = lora_A_i @ lora_B_i  (rank 64, ~8M params each)
Effective Q_i = SharedQ + dQ_i
```

This gives the expressiveness of per-layer attention at a fraction of the parameter cost.

**Implementation**:
```python
class SparseLocalAttention(nn.Module):
    window_size: int = 512
    n_heads: int = 16
    head_dim: int = 128      # d_model=2048, n_heads=16
    lora_rank: int = 64

    def forward(self, x, shared_qkv, lora_params):
        # Sliding window attention — only attend to previous 512 tokens
        # No cross-chunk attention needed (Mamba handles that)
        q = shared_qkv.W_q(x) + lora_params.dq(x)
        k = shared_qkv.W_k(x)
        v = shared_qkv.W_v(x) + lora_params.dv(x)
        return sliding_window_attention(q, k, v, window=512)
```

### 2.4 Mixture-of-Experts FFN

Based on: Mixtral [arxiv:2401.04088], Jamba [arxiv:2403.19887].

Applied to even-numbered layers (12 out of 24). Odd layers use a standard dense FFN.

```python
n_experts    = 8      # total experts
top_k        = 2      # active per token
d_ff         = 5632   # SwiGLU inner dim (≈ 2.75 * d_model)
d_model      = 2048

# Router: linear(d_model → n_experts), softmax, top-2 selection
# Load balancing loss: auxiliary loss to prevent expert collapse
#   L_aux = n_experts * sum(f_i * P_i) where f_i = fraction of tokens, P_i = avg gate prob
```

**Expert specialization** (emergent, not forced): With coding data, experts naturally specialize:
- Likely clusters: syntax/structure, docstrings, logic/control-flow, APIs, math/algorithms, tests, error-handling, type annotations

Dense FFN (odd layers) uses SwiGLU:
```
FFN(x) = SiLU(W1(x)) * W3(x) @ W2  -- d_ff = 5632
```

### 2.5 Full Layer Stack

```
CodingSSM-3B complete:

  Embeddings: 65536 * 2048 = 134M params

  Per Mamba-2 block (18 blocks):
    Projections:     2048 * (2*2*4096 + 2*8*128 + 32) ≈ 34M
    Conv1d:          4 * 4096 = 16K
    Output proj:     4096 * 2048 = 8M
    Total per block: ~42M params

  Per attention block (2 shared sets, 6 occurrences each):
    Shared QKV:      3 * 2048 * 2048 = 12M (shared across 6 uses)
    LoRA per layer:  2 * (2048*64 + 64*2048) = 0.5M * 6 layers = 3M
    Total:           15M

  MoE FFN (12 layers):
    8 experts * 3 * 2048 * 5632 = 8 * 34M = 276M ... but only 2 active
    Router: 2048 * 8 = 16K
    Total params: 278M, active: ~70M per layer

  Dense FFN (12 layers):
    3 * 2048 * 5632 = 34M per layer = 412M total

  LM head: tied to embedding = 0 extra

  TOTAL: ~134M + 18*42M + 15M + 12*70M_active + 12*34M ≈ 3.0B total, ~800M active
```

---

## 3. Training Strategy

### 3.1 Why Distillation (Not Pretraining)

Training from scratch requires ~3T tokens and weeks of multi-GPU compute.
Distillation from a local 32B teacher model is feasible on a single CPU machine:
- Each distillation step: forward through student (fast) + teacher soft labels (precomputed)
- No teacher backprop needed — just store teacher logits
- Effective: DeepSeek-R1 distilled a 1.5B model to outperform non-distilled 32B models

### 3.2 Teacher Model

**Qwen2.5-Coder-32B-Instruct** (running locally via llama.cpp):
- ~1-2 tok/s on our 64GB CPU machine
- Apache 2.0 license — distillation is permitted
- Proven coding benchmark leader (HumanEval: 90.2%, MBPP: 82.3%)

For reasoning traces, we can also use:
- **DeepSeek-R1-Distill-Qwen-32B** for chain-of-thought distillation
  (already released distilled checkpoints, fine-tuned from Qwen2.5 base)

### 3.3 Data Pipeline

**Stage 1 — Basic coding SFT** (~20k samples, fast):
```
Sources:
  - CodeAlpaca-20k (sahil2801/CodeAlpaca-20k) — already downloaded
  - Evol-Instruct-Code-80k-v1 (nickrosh/Evol-Instruct-Code-80k-v1)
  - OSS-Instruct (magicoder dataset)

Format:
  system: "You are an expert coding assistant."
  user:   <question>
  assist: <answer>         ← real label
  teacher_logits: <float32 array>  ← precomputed from Qwen2.5-32B
```

**Stage 2 — Reasoning distillation** (~800k samples, slow):
```
Sources:
  - Generate coding problems, run through DeepSeek-R1-Distill-32B
  - Collect full chain-of-thought traces (<think>...</think>)
  - Filter: keep only answers that pass unit tests

Format: same as DeepSeek-R1 distillation
  <think>{chain of thought}</think>\n{final answer}
```

### 3.4 Loss Function

Based on MiniLLM [arxiv:2306.08543] — reverse KLD outperforms forward KLD for generative models:

```
Forward KLD (standard): KL(p_teacher || p_student)
  Problem: student tries to cover all teacher modes → blurry, overestimates low-prob regions

Reverse KLD (MiniLLM): KL(p_student || p_teacher)
  Advantage: student focuses on high-prob teacher regions → sharper, better calibrated

Combined loss (Stage 1):
  L = α * KL(p_student || p_teacher)   -- distillation signal
    + β * CrossEntropy(student, target) -- ground truth
    α=0.5, β=0.5

Stage 2 (reasoning):
  L = CrossEntropy(student, chain-of-thought trace)
  (teacher logits not needed — traces are pre-generated)
```

### 3.5 Training Schedule (CPU-feasible)

```
Stage 1 — Architecture bringup (1-3 days):
  Data:       20k CodeAlpaca samples
  Steps:      5k-10k
  Batch:      1 sequence (CPU memory constraint)
  Seq length: 512
  LR:         1e-4 with cosine decay
  Eval:       Every 500 steps on HumanEval subset (20 problems)
  Goal:       Loss decreasing, coherent code output

Stage 2 — Extended SFT (1-2 weeks):
  Data:       80k Evol-Instruct samples
  Steps:      50k
  Seq length: 1024
  Goal:       HumanEval pass@1 > 30% (competitive with small open models)

Stage 3 — Reasoning distillation (2-4 weeks):
  Data:       Synthetic chain-of-thought traces
  Steps:      100k+
  Seq length: 2048
  Goal:       Complex bug fixing and architecture reasoning
```

### 3.6 Checkpoint & Evaluation

```python
# Eval metrics (run every 500 steps):
- Loss on held-out validation set
- HumanEval pass@1 (164 problems, run generated code)
- MBPP pass@1 (subset of 100 problems)
- Custom: 50 coding problems sampled from real coding sessions

# Targets:
  5k steps:   coherent Python, pass@1 > 5%
  20k steps:  pass@1 > 20%
  50k steps:  pass@1 > 40%  (competitive with Qwen2.5-3B)
```

---

## 4. Inference & Deployment

### 4.1 CPU Inference Mode

SSMs have a key advantage over transformers at inference time:

```
Transformer:   O(n) compute per token, O(n) KV cache (grows with sequence length)
Mamba-2:       O(1) compute per token, O(d_state) hidden state (constant)

For our model:
  - Attention layers (4 of 24): small fixed KV cache (window=512 only)
  - Mamba-2 layers (20 of 24): fixed state = n_heads * d_state * d_model bytes
    = 32 * 128 * 2048 * 4 bytes = 33MB per layer * 20 = 660MB total state
  - No KV cache growth for long contexts
```

### 4.2 Quantization Plan

Export to INT4/INT8 for deployment:
```
MoE expert weights:  INT8 (following Jamba-1.5 ExpertsInt8 approach)
Mamba-2 projections: INT4 (via GGUF-style block quantization)
Activations:         BF16 / FP32
Embedding:           FP16

Target memory:
  3B INT4:  ~1.5GB (model weights)
  State:    ~660MB
  Total:    ~2.5GB at inference — runs on any modern PC
```

### 4.3 Integration with SSM CLI

After training, the model exports to GGUF or a custom format and integrates with the existing `ssm chat`, `ssm ask`, etc. commands via a new `CodingSSM` backend in `ssm/model.py`.

---

## 5. What Makes This Novel

Compared to all published work:

| Property                          | Published? | Our Model |
|-----------------------------------|------------|-----------|
| Mamba-2 + sparse local attention  | No         | Yes       |
| Shared attention + LoRA in SSM    | Zamba2 (full attn) | Sparse variant |
| MoE + Mamba-2 combination         | No (Jamba uses Mamba-1) | Yes |
| Code-domain distillation from 32B | No         | Yes       |
| Chain-of-thought trace distillation into SSM | No | Yes |
| CPU-first design target           | No         | Yes       |

The core novelty: **Mamba-2 + sparse window attention + MoE FFN + shared attention weights**, distilled from a large coding model, targeting CPU deployment. This is a clean combination that hasn't been published.

---

## 6. File Structure (to build)

```
SSM/
  arch/
    mamba2.py          -- Mamba-2 / SSD block implementation
    sparse_attn.py     -- Sliding window attention + shared weights + LoRA
    moe.py             -- Mixture-of-Experts FFN
    model.py           -- CodingSSM full model assembly
    config.py          -- ModelConfig dataclass
  train/
    dataset.py         -- CodeAlpaca + Evol-Instruct data loader
    teacher.py         -- Teacher logit generation (via llama.cpp)
    distill.py         -- Training loop: reverse KLD + CE
    eval.py            -- HumanEval / MBPP evaluation harness
  export/
    to_gguf.py         -- Export trained model to GGUF
  ssm/                 -- existing CLI (unchanged during arch development)
    model.py
    rag.py
    config.py
  cli.py
  DESIGN.md            -- this file
```

---

## 8. Literature Review: HuggingFace Daily Papers (Jan–Mar 2026)

Reviewed papers from Jan 9 – Mar 6, 2026. Below are the findings that directly affect our design.

### 8.1 Critical Architecture Papers

**2Mamba2Furious** (2602.17363, SMU AI)
- Improves Mamba-2 by simplifying to essential components (Mamba-2S), then enhancing the A-mask and increasing hidden state order
- Claims "nearly as accurate as softmax attention, yet much more memory efficient"
- **Impact**: We should adopt the enhanced A-mask from 2Mamba2Furious in our Mamba-2 blocks

**MiniCPM-SALA** (2602.11761, MiniCPM team)
- 9B hybrid: sparse attention (InfLLM-V2) + linear attention in **1:3 ratio** with layer selection algorithm
- 3.5x faster at 256K tokens, supports 1M token context
- Hybrid Positional Encoding (HyPE) for both mechanisms
- **Impact**: Our 1:6 Mamba-to-attention ratio is well-supported; the 1:3 sparse:dense ratio within attention layers is relevant

**DynaMoE** (2603.01697)
- Dynamic token-level expert activation: variable top-k per token based on input complexity
- Six capacity distribution strategies across layers (descending, ascending, pyramid, wave)
- "Descending" schedule (more experts in early layers) works best for language modeling small models
- **Impact**: Replace our fixed top-2 routing with dynamic routing; use descending schedule

**Memory Caching: RNNs with Growing Memory** (2602.24281)
- Caches hidden state checkpoints, enabling O(L) growing memory approaching transformer capability
- Four variants: gated aggregation, sparse selective mechanisms
- **Impact**: Consider as an alternative to pure fixed-size SSM state; optional enhancement

**ReHyAt: Recurrent Hybrid Attention** (2601.04342)
- Hybrid recurrent + attention for video diffusion transformers
- Validates our hybrid approach in a different domain

### 8.2 Teacher Model Update: Qwen3-Coder-Next

**Qwen3-Coder-Next** (2603.00729, Qwen team)
- **Architecture**: Hybrid attention + MoE, **80B total / 3B active** parameters
- **Training**: Agentic training on executable task synthesis + RL from environment feedback
- **Performance**: 71.3% SWE-Bench Verified — **comparable to Claude Sonnet**
- **GGUF availability**: `unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF`, Q4_K_M = **48.5 GB** (fits in 64GB RAM)
- **CPU status**: Runs via llama.cpp (slow ~5x below expected, but functional)

**This changes our teacher model**: Qwen3-Coder-Next is vastly superior to Qwen2.5-Coder-32B for coding tasks. The 48.5GB GGUF fits in 64GB RAM with room for the student model training.

```
Updated teacher pipeline:
  Teacher:  Qwen3-Next-80B-A3B-Instruct Q4_K_M (48.5GB)  ← 71.3% SWE-Bench
  Student:  CodingSSM-3B (our model, ~1.5GB at INT4)
  Method:   Offline logit generation + reverse KLD distillation
```

### 8.3 Training Enhancements

**On-Policy Self-Distillation for Reasoning Compression (OPSDC)** (2603.05433)
- Same model generates teacher logits with "be concise" instruction → teaches compression
- Reverse KLD loss (consistent with MiniLLM)
- **Results**: 57-59% token reduction on MATH-500 WITH 9-16 point accuracy improvement
- Automatically compresses easy problems more, hard problems less (no budget/difficulty estimator needed)
- **Impact**: After base distillation, run OPSDC to teach our model to reason efficiently

**Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation** (2602.12125)
- Student can outperform teacher by using reward signals beyond teacher's distribution
- **Impact**: Our student may exceed Qwen3-32B on coding subdomains with right reward shaping

**Reinforcement-aware Knowledge Distillation for LLM Reasoning** (2602.22495, AWS)
- Combines RL signals with KD for better reasoning transfer
- **Impact**: Stage 3 training option: RL on code execution + KD signal

**CHIMERA: Compact Synthetic Data for Generalizable LLM Reasoning** (2603.00889, Apple)
- Small, diverse synthetic datasets generalize better than large redundant ones
- **Impact**: Quality > quantity for our training data; curate 10K high-quality problems > 100K low-quality

### 8.4 MoE Insight: Illusion of Specialization

**The Illusion of Specialization** (2601.03425)
- MoE "standing committee": same subset of experts handles most tokens regardless of domain
- Expert specialization must be explicitly encouraged, not assumed to be emergent
- **Impact**: Our MoE needs explicit routing regularization or curriculum to force domain specialization

### 8.5 Alternative Paradigms Worth Monitoring

**dLLM: Simple Diffusion Language Modeling** (2602.22661, UC Berkeley — 2,130 upvotes)
- Framework unifying training/inference/evaluation of diffusion LMs (LLaDA, Dream)
- Converts any BERT-style encoder into a Diffusion LM
- Non-autoregressive: generates all tokens in parallel with masked diffusion
- **Not for our immediate use**: CPU inference is ~10x slower than AR models currently; parallel generation requires GPU

**Stable-DiffCoder** (2601.15892, ByteDance Seed — 75 upvotes)
- Diffusion-based code generation; "pushing the frontier of code diffusion"

**Focus-dLLM** (2602.02159)
- Confidence-guided context focusing for diffusion LLM inference acceleration
- Shows diffusion LMs are converging toward practical speed

**ThinkRouter** (2602.11683)
- Routing reasoning between latent and discrete spaces for efficiency
- Relevant for our test-time compute allocation

**The Flexibility Trap** (2601.15165, Tsinghua — 128 upvotes)
- Arbitrary token generation order limits reasoning in diffusion LMs
- Key weakness of dLLMs for reasoning tasks — validates our autoregressive approach

### 8.6 Key Benchmarks to Target

Based on papers reviewed, these are the benchmarks used to establish novelty:
- **SWE-Bench Verified**: Real GitHub issues (Qwen3-Coder-Next: 71.3%)
- **HumanEval pass@1**: Classic function completion
- **MBPP pass@1**: Python programming problems
- **Terminal-Bench** (2601.11868, 1.67k upvotes): Hard CLI agent tasks — emerging standard
- **LiveCodeBench**: Competition-style problems with time-stamped splits
- **AIME 2024**: Math reasoning

---

## 9. Revised Architecture (Post Literature Review)

### Architectural Changes

Based on the literature review, three changes to our original design:

**1. Enhanced A-mask (from 2Mamba2Furious)**
```python
# Original Mamba-2: scalar-times-identity A
# 2Mamba2Furious: enhanced A-mask, increased hidden state order
# Our implementation: adopt the 2Mamba variant for better accuracy
```

**2. Dynamic MoE routing (from DynaMoE)**
```python
# Original: fixed top-2 per token
# Revised: dynamic top-k (1-4 experts) based on token complexity
#          descending capacity schedule (more experts in early layers)
n_experts = 8
max_active = 4    # max experts per token
min_active = 1    # min experts per token
schedule = "descending"  # early layers use more experts
```

**3. Teacher model upgrade**
```
Old teacher: Qwen2.5-Coder-32B Q4_K_M (~20GB)
New teacher: Qwen3-Next-80B-A3B-Instruct Q4_K_M (~48.5GB)
Reason: 71.3% SWE-Bench vs ~50% for Qwen2.5-32B
Both fit in 64GB RAM (barely, with care)
```

### Updated Training Pipeline

```
Stage 0: Download teacher (48.5GB, ~6hrs on good connection)
Stage 1: Architecture bringup on CodeAlpaca-20k + teacher logits (3-5 days)
Stage 2: Extended distillation on Evol-Instruct + curated problems (1-2 weeks)
Stage 3: OPSDC self-distillation for reasoning compression (3-5 days)
Stage 4: Optional RL from code execution (ongoing)
```

---

## 7. References

- **Mamba-2 / SSD**: Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", ICML 2024. [arxiv:2405.21060](https://arxiv.org/abs/2405.21060)
- **Jamba**: Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model", AI21 Labs 2024. [arxiv:2403.19887](https://arxiv.org/abs/2403.19887)
- **Jamba-1.5**: Jamba at Scale — 94B active, ExpertsInt8, 256K context. [arxiv:2408.12570](https://arxiv.org/abs/2408.12570)
- **Zamba2**: Glorioso et al., "The Zamba2 Suite", Zyphra 2024. [arxiv:2411.15242](https://arxiv.org/abs/2411.15242)
- **Mixtral**: Mistral AI, "Mixtral of Experts", 2024. [arxiv:2401.04088](https://arxiv.org/abs/2401.04088)
- **MiniLLM**: Gu et al., "MiniLLM: Knowledge Distillation of Large Language Models", ICLR 2024. [arxiv:2306.08543](https://arxiv.org/abs/2306.08543)
- **DeepSeek-R1**: DeepSeek AI, "Incentivizing Reasoning Capability in LLMs via RL", 2025. [arxiv:2501.12948](https://arxiv.org/abs/2501.12948)
- **RWKV-7 Goose**: Peng et al., "RWKV-7 with Expressive Dynamic State Evolution", 2025. [arxiv:2503.14456](https://arxiv.org/abs/2503.14456)
- **minGRU/minLSTM**: "Were RNNs All We Needed?", 2024. [arxiv:2410.01201](https://arxiv.org/abs/2410.01201)
- **Mamba Survey**: "A Survey of Mamba", 2024. [arxiv:2408.01129](https://arxiv.org/abs/2408.01129)

### 2025–2026 Papers (from HuggingFace Daily Papers Review)
- **2Mamba2Furious**: Enhanced Mamba-2 A-mask, near-softmax accuracy. [arxiv:2602.17363](https://arxiv.org/abs/2602.17363)
- **MiniCPM-SALA**: Sparse+linear attention hybrid (1:3 ratio), 1M context, 3.5x speedup. [arxiv:2602.11761](https://arxiv.org/abs/2602.11761)
- **DynaMoE**: Dynamic token-level expert activation, layer-wise adaptive capacity. [arxiv:2603.01697](https://arxiv.org/abs/2603.01697)
- **OPSDC**: On-Policy Self-Distillation for Reasoning Compression; reverse KLD; 57-59% compression + accuracy gain. [arxiv:2603.05433](https://arxiv.org/abs/2603.05433)
- **Qwen3-Coder-Next**: 80B/3B active MoE, agentic training, 71.3% SWE-Bench. [arxiv:2603.00729](https://arxiv.org/abs/2603.00729)
- **Memory Caching (RNNs Growing Memory)**: Growing SSM state via checkpoint caching. [arxiv:2602.24281](https://arxiv.org/abs/2602.24281)
- **dLLM**: Diffusion Language Modeling framework (LLaDA/Dream). [arxiv:2602.22661](https://arxiv.org/abs/2602.22661)
- **The Flexibility Trap**: Arbitrary order limits reasoning in diffusion LMs. [arxiv:2601.15165](https://arxiv.org/abs/2601.15165)
- **MoE Illusion of Specialization**: Standing committee problem in MoE. [arxiv:2601.03425](https://arxiv.org/abs/2601.03425)
- **CHIMERA**: Compact synthetic data generalizes better for reasoning. [arxiv:2603.00889](https://arxiv.org/abs/2603.00889)
- **Terminal-Bench**: CLI agent benchmark, emerging standard. [arxiv:2601.11868](https://arxiv.org/abs/2601.11868)
- **RWKV-7 "Goose"**: Dynamic State Evolution, exceeds TC0. [arxiv:2503.14456](https://arxiv.org/abs/2503.14456)
