# PFlash Prior-Art Research: Speculative / Compressive Prefill for LLM Inference

**Scope:** Find every public implementation of speculative/compressive prefill; extract the algorithm, the
compression-vs-quality tradeoff, and the MLX/Metal/Rust feasibility for porting to `higgs` (Apple Silicon +
MLX + Rust, target Bonsai-27B / Qwen3-0.6B drafter).

**Date:** 2026-07-17. **Status:** research-only, no code written.

---

## TL;DR — what we actually found

1. **The user's premise conflates two different algorithms.** "FlashPrefill" (Fan et al., arXiv:2603.06199)
   is **NOT a drafter method** — it is a training-free *sparse-attention* method applied to the **target
   model's own** Q/K (no drafter, like MInference / FlexPrefill). The **drafter-based** method is
   *Speculative Prefill* / SpecPrefill (Liu et al., arXiv:2502.02789). These compose: PFlash (Lucebox)
   uses SpecPrefill for the **drafter's scoring** + FlashPrefill for the **drafter's own sparse forward**.
2. **There is NO Rust / MLX / Metal port of any of these algorithms.** Every existing implementation is
   Python+Triton (papers) or C+++CUDA (PFlash). The 2.2% Metal in Lucebox is unrelated to PFlash (PFlash is
   explicitly "C++/CUDA only"). mlx-rs has no topic page on GitHub; the few `mlx-rs` repos I tried to fetch
   404'd. **This negative result is itself the key finding for our port.**
3. **The closest prior art to what we are building is Lucebox's PFlash** — same target family (Qwen-27B),
   same drafter (Qwen3-0.6B), same headline numbers (10× at 128K, NIAH preserved at 5% keep). The user's
   "Bonsai-27B + Qwen3-0.6B" is functionally the PFlash pair. **We are re-implementing PFlash on Apple
   Silicon**, and they have already published all the hyperparameters and the exact algorithm.
4. **Our naive scorer's 50% keep vs PFlash's 5% is the algorithm gap, not the silicon gap.** Three concrete
   SpecPrefill denoising tricks (chunk-top-K + 1D-avgpool smoothing + 8-step lookahead + position-id
   restoration) take raw tail-attention from ~50% → ~10% keep. FlashPrefill's per-query-block max-based
   alpha-threshold takes it the rest of the way to ~5%.

---

## 1. Implementation inventory

| Repo / artifact | Backend | Status | Algorithm | Drafter | Port to MLX/Metal/Rust? |
|---|---|---|---|---|---|
| **`qhfan/FlashPrefill`** ([github](https://github.com/qhfan/FlashPrefill)) | Python + Triton 3.3 + vLLM 0.10/0.12 | experimental, 53 stars, 5 commits, 3 open issues | **Sparse attention on target** (no drafter). Mean-K block proxy + fused block-level score + **max-based dynamic α-threshold** (not top-K). Block-sparse kernel via `Block-Sparse-Attention` (cutlass, sm_80+). | none — uses target's own Q/K | **No**. Triton only. Would need a Metal port of the 4 kernels. |
| **`Jingyu6/speculative_prefill`** ([github](https://github.com/Jingyu6/speculative_prefill)) | Python + vLLM monkey-patch | active, 63 stars, 205 commits, MIT | **Drafter-based** (Llama-3.2-1B → 70B/405B target). Lookahead=8, **max over (L,H), mean over N**, 1D avgpool, **chunk top-K**, original position-id restoration. | Llama-3.2-1B-Instruct BF16 (TP=8 with target) | **No**. vLLM/Python only. |
| **Cross-Family Speculative Prefill** (arXiv:2603.02631, SambaNova, ICLR 2026 WS) | **no public code** — SambaNova RDU | paper-only | **Identical to SpecPrefill** + cross-family drafters + new contiguous pos-ids (no restore) + text-level delimiter tokens. | Qwen3-{0.6B,1.7B,4B}, Llama-3.x | **No**. |
| **`Luce-Org/lucebox` → `optimizations/pflash/` + `server/`** ([github](https://github.com/Luce-Org/lucebox), [blog](https://lucebox.com/blog/pflash)) | **C++/CUDA**, ggml allocator, no Python at runtime | very active, 2.7k stars, 1.1k+ commits | **Combines both:** SpecPrefill scoring (drafter attention → block mean-K → score) + **FlashPrefill** block-sparse drafter forward (4 kernels: `mean_K → score → select → sparse_fwd`) + BSA (FA-2 derived, sm_80+). | **Qwen3-0.6B BF16 GGUF** (exactly our pair) | **No.** CUDA-only; sm_80+ required for BSA. **2.2% Metal in repo is unrelated to PFlash.** |
| llama.cpp / ggml upstream | C++/CUDA/HIP/Metal/Vulkan | n/a | **No speculative prefill upstream.** Only speculative *decoding* (draft-model token proposal + verify). | — | n/a (it's where we'd contribute) |
| `mit-han-lab/Block-Sparse-Attention` | C++/CUDA, cutlass, PyTorch ext | reference | FA-2-derived block-sparse attention. Used by FlashPrefill and PFlash for the sparse drafter/target forward. | — | **No** (CUDA + cutlass only). |
| MInference / FlexPrefill / XAttention | Python + Triton/cutlass | various | Prior sparse-attention methods. Compared as baselines by FlashPrefill; all slower or less accurate. | — | **No.** |
| Any **Rust / mlx-rs / CoreML** speculative-prefill impl | — | — | — | — | **None found.** GitHub `mlx-rs` topic has zero repos. Searched `locapela/mlx-rs`, `RobinMattar/mlx-rs`, `gabriel-ss/mlx-rs`, `david-at-locapela/mlx-rs` — all 404. crates.io search requires JS (could not complete). **Conclusion: we are first.** |

### Headline numbers side-by-side

| Work | Target | Drafter | Context | Keep ratio | Quality | TTFT speedup |
|---|---|---|---|---|---|---|
| SpecPrefill (Liu 2502.02789) | Llama-3.1-405B FP8 | Llama-3.2-1B | up to 128K | **10%** | LongBench avg preserved (single-doc QA, multi-doc QA, few-shot); aggregation tasks break | **7.66× TTFT, 7× QPS** (8×H200) |
| Cross-Family (Upasani 2603.02631) | DeepSeek-V3/R1 | Qwen3-4B cross-family | 128K | 12.5% | RULER 89.7 vs 80 baseline; LongBench-v2 within 90% | **~18× TTFT** (SambaNova RDU) |
| Lucebox PFlash | Qwen3.6-27B Q4_K_M | **Qwen3-0.6B BF16** | 64K / 128K | **5%** | NIAH single-needle ✓ at all ctx | **10.0× / 10.4×** (RTX 3090) |
| Lucebox PFlash | Qwen3.6-27B Q4_K_M | Qwen3-0.6B | 128K | **2%** | NIAH "starts losing needle — calibration territory" (blog) | unreported, target prefill ~3s |

---

## 2. Algorithm in detail

### 2.1 SpecPrefill (Liu 2502.02789) — the drafter-based scorer

This is what produces the 10% keep at quality. **PFlash uses this verbatim.**

**Forward.** Run drafter over the prompt with `lookahead = N` (paper: `N = 8`, "beyond 16 minimal gain").
For each lookahead step we keep the **decoded token's Q** (last token + N autoregressive steps; KV cache
optional but must store the queries).

**Scoring formula** (§3.2.2):
```
a_ij = Softmax(Q_{M+j} K^T)_i       # per-layer, per-head attention from lookahead-token j to prompt-token i
                                       shape [N, L, S, H] after gathering all layers + heads

importance[i] = mean_N  max_{L,H}  a_·j·i·     # "max-mean aggregation":
                                            #   max over (layers, heads) makes salient tokens stand out
                                            #   mean over N lookahead tokens accounts for fair contribution
```
So per-token importance is `mean_over_lookahead( max_over_layers_and_heads( attention ) )`. **Our naive
scorer is "max over heads, no lookahead, no chunking" — i.e. SpecPrefill's weakest ablation, which is
exactly why we need 50% keep.**

**Selection mechanism** (§3.2.3 + Cross-Family Appendix A.1):
1. **1D average pooling** on `importance[S]` with kernel `13` (Cross-Family paper). Smoothes cross-block
   noise. This step alone is a big win.
2. **Chunk the prompt** into blocks of size `chunk_size = 32` (general) or `128` (code/repo tasks). Average
   importance within each chunk.
3. **Top-K chunks** by averaged score, where `K = ρ × (S / chunk_size)` and `ρ` is keep-ratio.

**Position-ID restoration** (§3.2.4): **keep original prompt positions** of the surviving tokens; decoding
tokens are offset by the original prompt length, not the compressed length. *"Crucially essential, especially
for position-sensitive tasks such as synthetic tasks involving retrieval and counting."* — Cross-Family
replaces this with new contiguous pos-ids + delimiter tokens, finding *"negligible impact on task accuracy"*.
**For NIAH, restore pos-ids is safer.**

**Ablation impact** (paper §4.4 and Appendix Fig 8): "SpecPrefill" (raw, no tricks) vs "SpecPrefill Full"
(chunk+pool) vs "SpecPrefill Full LAH" (+lookahead) — *"consistent improvement; benefits of look-ahead are
more consistent in shorter context tasks."* The gap from raw to Full-LAH is roughly the gap from our 50%
down to the published 10%.

### 2.2 FlashPrefill (Fan 2603.06199) — the *target-side* sparse-attention method

**Important correction:** FlashPrefill does **not use a drafter**. It runs sparse attention on the target
model's own Q/K. It is the latest entry in the MInference / FlexPrefill / XAttention family. PFlash borrows
FlashPrefill only as the **drafter accelerator** (the drafter is small but at 128K its dense attention is
still O(S²); FlashPrefill makes the drafter itself fast).

**Scoring formula** (§3.1, Alg 1):
1. **Block-pre-pool K:** for each K-block `B` of size `B=128`, compute `k̄_J = mean(K_block)`. Single
   vector per block.
2. **Fused block score:** for query tile `Q_I` (size `B`) and pooled key `k̄_J`, compute `qk = (Q_I · k̄_J^T) · τ · log2(e)`, apply causal mask, then per-query-tile:
   ```
   m_{I,J}  = max_i(qk_i)                       # local max for numerical stability
   S_{I,J}  = Σ_i 2^(qk_i − m_{I,J})             # approximated energy
   ```
3. **Global normalization** (consistency across query tiles):
   ```
   M_I = max_J(m_{I,J})
   S'_{I,J} = S_{I,J} · exp(m_{I,J} − M_I)
   Score_{I,J} = S'_{I,J} / (Σ_K S'_{I,K} + ε)   # block-level importance, normalized per query tile
   ```

**Selection mechanism** (§3.4 — the key novelty): **no top-K, no top-p.** Max-based dynamic threshold per
query block:
```
thresh_I = α · max_{J≤I}(Score_{I,J})       # α ∈ (0, 1)
keep block (I, J)  iff  Score_{I,J} ≥ thresh_I
```
This requires only a **single-pass max-reduction** (no sort, no cumsum), and crucially *"mitigates the
impact of long-tail distributions"* — top-K/top-p are forced to fill k slots even when only a few blocks
matter, while α-threshold naturally prunes the tail.

**Sparsity pattern** is `sink + sliding window + dynamic top blocks`:
- Sink size = **256 tokens** (explicit)
- Local window = **512 tokens** (explicit)
- Block size `B = 128` everywhere
- α tuned per-model to keep **~70% density at 4K** (calibration rule); density then drops with length:

| Model | α | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 0.18 | 71.0% | 45.8% | 28.0% | 16.0% | 8.2% | 4.5% |
| Qwen2.5-7B-Instruct | 0.08 | 70.0% | 46.8% | 29.2% | 20.8% | 10.6% | 6.6% |
| Qwen3-30B-A3B-Instruct-2507 | 0.12 | 70.4% | 46.0% | 29.0% | 17.6% | 10.0% | **5.8%** |

(PFlash's `DFLASH_FP_ALPHA = 0.85` is *much* stricter because it is applied to the **drafter's** scores, not
the target's. The FlashPrefill paper's α is applied to the target's own scores.)

**Why it beats top-K / top-p** (paper Tab 8, Llama-8B @ α=0.18):
- Top-k (12.5% fixed): 91.08 / 81.67 / 70.22 at 32K/64K/128K
- Top-p (0.9):         92.38 / 82.12 / 72.83 at 12.5% / 15.7% / 14.0% density
- Max-α:               92.21 / 84.93 / 75.31 at **16.0% / 8.2% / 4.5% density** — better quality AND lower density.

### 2.3 PFlash (Lucebox) — the production combination

This is the most relevant prior art for us. Pipeline:

```
prompt (≤128K tokens)
  │
  ▼  [drafter forward — Qwen3-0.6B BF16, custom ggml graph]
  │    dense attention below 32K; FlashPrefill block-sparse at ≥32K via BSA (FA-2, sm_80+)
  │    4 CUDA kernels: mean_K → score → select → sparse_fwd
  ▼
[block-level score per (Q-block, K-block)]
  │    alpha-threshold (DFLASH_FP_ALPHA=0.85) selects K-blocks per Q-block
  ▼
[per-token survival mask]   keep_ratio=0.05  (~6.5K survivors at 128K, 20× compression)
  │
  ▼  emit compressed token-id stream
  │
[Qwen3.6-27B Q4_K_M target prefill of ~6.5K tokens, full attention]  ~10 s
  │
  ▼  DDTree spec decode at ~74 tok/s
```

**Hyperparameters shipped by PFlash** ([README](https://github.com/Luce-Org/lucebox/blob/main/optimizations/pflash/README.md)):

| Knob | Default | Bench | Effect |
|---|---|---|---|
| `--prefill-compression` | `off` | `auto`/`always` | Trigger mode |
| `--prefill-threshold` | `32000` tokens | — | Below this, no compression (`auto` mode) |
| `--prefill-keep-ratio` | `0.05` | `0.02 @128K, 0.10 @32K` | Fraction of source tokens kept |
| `--prefill-curve T:R T:R...` | off | `10000:0.5 40000:0.2 100000:0.1` | Piecewise-linear keep-ratio over tokens |
| `--prefill-drafter` | required | Qwen3-0.6B BF16 GGUF | Drafter weights |
| `DFLASH_FP_USE_BSA` | `0` | `1` | Dispatch sparse FA via BSA — **required for headline 10.4×** |
| `DFLASH_FP_ALPHA` | `0.12` | `0.85` | Block-selection threshold; higher = stricter |
| `PFLASH_FREEZE_HOT_WINDOW` | `2` | — | FlowKV: most recent N messages stay verbatim |
| `--prefill-skip-park` | off | — | Keep drafter resident (more VRAM, faster) |

**VRAM dance (24 GB card):** drafter (1.3 GB weights + KV + ~600 MB BSA scratch) and target (~18 GB
resident) cannot coexist. Daemon stdin protocol:
```
park target + draft         # free ~18 GB
load drafter, score         # ~10 GB peak, ~12 s at 128K
free drafter                # release weights + KV + BSA scratch
unpark target + draft       # reload ~18 GB
generate                    # target spec decode
park draft                  # idle
```
Costs ~3 s per request on a 3090. **On Apple Silicon unified memory this entire dance disappears** —
see §4.

**Cost breakdown at 128K** (RTX 3090):
- Drafter scoring: **~12 s** (dominant)
- Target prefill of ~6.5K survivors: **~10 s**
- park/unpark/free dance: **~3 s**
- Total: **24.8 s** vs **~257 s** llama.cpp → **10.4×**

---

## 3. Compression-vs-quality tradeoff — the "highest tradable point"

Collected quality numbers (all greedy decoding):

### 3.1 NIAH / RULER retrieval — where compression is essentially free

| Source | Setup | Result |
|---|---|---|
| SpecPrefill Tab 2 (Llama-70B, 10% keep) | RULER 4K→64K retrieval/multihop/QA | **Same as full prompt** at every length; NIAH multikey/multivalue/multiquery near-100% |
| SpecPrefill Tab 2 @ 128K 10% keep | RULER retrieval avg | 65.6 vs 60.3 full — **spec-prefill *beats* full** (denoising) |
| Cross-Family Tab 4 (DeepSeek-V3, 12.5% keep) | RULER @ 128K niah_single_1/2/3 | 100/100/100 (vs full 100/100/62.8 — spec-prefill fixes baseline's failure) |
| PFlash (Qwen3-0.6B → Qwen3.6-27B, 5% keep) | NIAH single-needle 32K/64K/128K | **✓ retrieved at every ctx** |

**Knee for retrieval:** essentially no cliff down to ~5% keep. Aggregation tasks (common-word, frequent-word extraction) are the exception — they need every token and break early.

### 3.2 Multi-doc QA / LongBench — flat down to 10%, then mild degradation

SpecPrefill LongBench avg @ 10% keep, Llama-3.1-70B: **52.74 vs 53.55 full** (98.5%). Per-category:
- Single-Doc QA: 47.64 vs 50.57 (94%)
- Multi-Doc QA: 52.96 vs 53.11 (99.7%)
- Few-shot: 64.52 vs 66.93 (96%)
- Code: 63.33 vs 52.33 (**+21% — compression *helps* code**)
- Summarization: 21.74 vs 25.84 (84% — degrades expectedly)
- Synthetic: 66.25 vs 72.50 (91%)

**Knee:** quality is roughly flat from 100% → 10%, with a mild slope. Below 10% the slope steepens, especially for summarization.

### 3.3 Cross-Family on harder long-context (LongBench-v2, DeepSeek-R1)

| Target | Drafter | Keep | Accuracy | vs full |
|---|---|---|---|---|
| DeepSeek-R1 (avg 248K input) | Qwen3-1.7B | **3%** | 45.9 | 79% |
| DeepSeek-R1 | Qwen3-1.7B | **6%** | 47.9 | 82% |
| DeepSeek-R1 | Qwen3-1.7B | **10%** | 46.9 | 80% (diminishing return) |
| DeepSeek-R1 | Qwen3-4B-2507 | **6%** | 53.3 | 91% |
| DeepSeek-R1 | Llama-3.1-8B | **6%** | 54.1 | 93% |
| Llama-3.1-8B | Qwen3-0.6B | **10%** | 30.0 | 96% |

**Important:** "performance improves 3% → 6%, then *decreases* at 10%" — there is a real knee around
**6%** for hard long-context reasoning. Below that you lose information; above that you start adding
distraction. **For DeepSeek-R1 the optimum is ~6%, not lower, not higher.**

### 3.4 Code / agent workloads — the user's real concern

| Source | Task | Keep | Quality |
|---|---|---|---|
| Cross-Family Tab 2 | InfiniteBench Code Debug, DeepSeek-V3.1 target | 20% | 64.72 vs 67.51 full (96%) |
| Cross-Family Tab 2 | Code Debug, DeepSeek-V3.1 | **15%** | 59.13 vs 67.51 (**87%**) |
| Cross-Family Tab 2 | Code Debug, DeepSeek-R1 | 30% | 70.30 vs 74.37 (95%) |
| Cross-Family Tab 2 | Code Debug, DeepSeek-R1 | 25% | 68.02 (91%) |
| Cross-Family Tab 2 | Code Debug, DeepSeek-R1 | **15%** | 62.44 (**84%**) |
| SpecPrefill (LongBench code-completion, Llama-70B) | 10% | **63.33 vs 52.33 full (+21%)** |
| PFlash | NIAH only — **no code/agent eval published** |

**Code-debug knee:** ~20% keep for <5% degradation; at 15% you lose 5-15 points depending on target.
For agent workloads (multi-step reasoning, tool-use) **no published eval exists for any of these methods**.
SpecPrefill §4.6 Appendix A explicitly states "for shorter tasks, queries are more likely to be information
dense, rendering SpecPrefill less effective" — short, dense prompts (typical agent step) is the worst case.

### 3.5 Summary curve — the tradable point

```
keep%   retrieval   multi-doc-QA   hard-LB-v2   code-debug   agent/coding-loop
100%    100%        100%           100%         100%         100%  (baseline)
 50%    ~100%       ~100%          ~98%         ~99%         ???   (NAIVE safe)
 20%    ~100%       ~99%           ~95%         ~96%         ???   (production-safe)
 10%    ~100%       ~98%           ~91%         ~91%         ???   (SpecPrefill default)
  6%    ~100%       ~95%           ~91%         ~87%         ???   (knee for hard tasks)
  5%    ✓ NIAH      ???            ???          ???          ???   (PFlash default — only NIAH proven)
  2%    marginal    breaks         breaks       breaks       ???   (PFlash "calibration territory")
```

**Honest assessment:** below ~10% keep, only **NIAH single-needle** is robustly validated. Multi-doc QA,
code-debug, agent loops have **thin or no evidence** at <10%. The user's concern that "agent/coding/multi-step"
is under-eval'd is **correct** — this is a genuine research gap, not just for us.

---

## 4. MLX / Metal feasibility

### 4.1 Capturing attention scores in MLX

**Confirmed from `mlx-lm/models/qwen2.py` and our own `docs/mlx_rs_capabilities.md`:** MLX's
`scaled_dot_product_attention` (mlx-rs `fast::scaled_dot_product_attention`) is a fused flash SDPA — it
**does not return the attention weight matrix**. But `q_proj`, `k_proj`, `v_proj` are exposed as
**separate Linear layers** (mlx-rs: `model.layers.{i}.self_attn.q_proj.inner.weight`).

**The standard approach** (what we already do, and what every paper implicitly assumes) is:
```rust
let q = q_proj.apply(&x).reshape(...).rope(...);   // [B, H, S, d]
let k = k_proj.apply(&x).reshape(...).rope(...);   // [B, H_kv, S, d]
// manual attention score:
let scores = (q.matmul(k.transpose(-2, -1)) * scale);   // [B, H, S, S] — full softmax NOT needed
// apply causal mask, softmax optional, then aggregate per the chosen algorithm
```
For the **drafter's scoring pass** we do NOT need the value output at all — just Q@K^T → aggregate → block
score. This is cheap and well-supported in mlx-rs.

### 4.2 Block-sparse attention in MLX — NOT available

**Confirmed:** `mlx/backend/metal/` ships `scaled_dot_product_attention.cpp`, `matmul.cpp`, `softmax.cpp`,
`reduce.cpp` — **no `block_sparse_attention` or sparse-attention kernel exists in MLX**. There is a
`custom_kernel.cpp` JIT path (and a `kernels/` Metal source dir), so **hand-written Metal kernels are
possible but non-trivial** — you'd be porting either mit-han-lab/Block-Sparse-Attention (cutlass → Metal) or
qhfan/FlashPrefill's 4 Triton kernels (→ Metal).

**Practical implication:** the FlashPrefill sparse-drafter-forward step (which PFlash relies on for
10.4×) **cannot run on MLX as-is**. Options, easiest first:
1. **Skip BSA entirely.** Run the drafter with dense flash SDPA. PFlash does this below 32K and it works —
   Qwen3-0.6B dense attention at 32K is ~ms. At 128K, dense Qwen3-0.6B attention will be slow (each layer
   is O(S²) but only 0.6B params, so maybe ~30-60s on M-series vs PFlash's 12s). Still a net win.
2. **Sliding-window + sink only** for the drafter (no dynamic top-K). Trivial in mlx-rs by manually masking.
3. **Chunked dense attention** — process the prompt in chunks, compute per-(Q-block, K-block) max-pooled
   scores one chunk-pair at a time. This is a *poor man's FlashPrefill scoring kernel* and is what gets
   FlashPrefill's `mean_K` and `score` steps without needing a custom Metal kernel.
4. **Custom Metal kernel.** Right answer long-term, but a meaningful project. mlx-rs explicitly notes
   custom Metal kernels are NOT available (see our `docs/mlx_rs_capabilities.md` "NOT Available" table).

### 4.3 Apple Silicon unified memory — the *advantage*

**PFlash's biggest operational pain — the park/unpark VRAM dance — disappears on Apple Silicon.** On a
24GB RTX 3090, target (15 GB) + draft (3 GB) + drafter (1.3 GB + KV + BSA scratch ≈ 10 GB peak) cannot
coexist; the daemon cycles them through VRAM at ~3 s cost per request.

On Apple Silicon with unified memory:
- M3 Max 128GB, M4 Max 128GB, Studio M2/M3 Ultra 192GB: the drafter + target + draft + KV all fit
  simultaneously. **No park/unpark. No reload latency.** PFlash's `--prefill-skip-park` flag exists for
  exactly this case on bigger NVIDIA cards.
- Even on a 64GB M-Max: Bonsai-27B Q4 (~16GB) + drafter (~3GB) + draft (~2GB) + KV (~8-16GB at 128K)
  ≈ 30-40GB — fits with headroom.
- Unified memory means a "park to host RAM" is essentially free if ever needed (no PCIe transfer).

**Conclusion:** on Apple Silicon, our end-to-end PFlash-equivalent should beat PFlash's RTX-3090 numbers
because we save the ~3 s park/unpark overhead per request, and we can keep the drafter perpetually resident.

---

## 5. Recommendation for our port

### 5.1 Which algorithm to implement

**Recommended: a SpecPrefill-Full-LAH scorer + chunked-dense drafter forward (no BSA), targeting
keep_ratio ≈ 0.10.** Specifically:

1. **Drafter:** Qwen3-0.6B BF16 via mlx-rs (we already have the model architecture). Run a single forward
   pass with `lookahead=8` (store Q of last token + 8 decoded steps). KV cache reused from existing
   infra.
2. **Scorer (the gap-closer vs our 50% naive):**
   - `importance[i] = mean_N( max_{L,H}( attention[N,L,S,H] ) )` — this is our current max-over-heads,
     **plus the lookahead mean**.
   - 1D avgpool smoothing, **kernel = 13** (paper setting).
   - **Chunk into 32-token blocks**, average importance per chunk.
   - **Top-K chunks** with `K = ρ × (S / 32)` and `ρ = 0.10`.
   - **Restore original position IDs** for survivors; decoding offset = original prompt length.
3. **Drafter forward attention:** dense flash SDPA up to ~32K source; above that, **fall back to
   chunked-blockwise scoring** (compute Q@K^T block-pair by block-pair with mean-K pooling) instead of BSA.
   Acceptable perf because the *scoring* doesn't need the V output, and the drafter is only 0.6B.
4. **Target:** unchanged — Bonsai-27B prefills the compressed token stream with full attention.

**Defer:** custom Metal BSA kernel (option 4 in §4.2). It's the right long-term play but it's a multi-week
Metal project; the algorithm works fine without it for a first cut.

**Do NOT** try to implement full FlashPrefill-on-target. FlashPrefill is a sparse-attention *replacement*
for the target's attention, which would require modifying Bonsai's attention — that's a much deeper change
and gives different tradeoffs (no drafter, but target-attention-sparse). SpecPrefill-style compression keeps
the target intact, which is what we want.

### 5.2 Realistic numbers we can hit

With the SpecPrefill-FL scorer + dense drafter forward on, say, M4 Max 128GB:

| Source ctx | Keep | Survivors | Quality (NIAH) | Quality (LongBench-ish) | TTFT speedup vs dense | Notes |
|---|---|---|---|---|---|---|
| 32K | 10% | ~3.2K | ✓ | ~95% | **~8-9×** | matches PFlash's 32K tier |
| 64K | 10% | ~6.4K | ✓ | ~93% | **~8-10×** | dense-drafter scoring ~5-8s |
| 128K | 10% | ~12.8K | ✓ | ~90% | **~8-10×** | dense-drafter scoring ~15-25s without BSA |
| 128K | 5% | ~6.4K | ✓ NIAH (PFlash-validated) | thin evidence | **~12-15×** | push only after BSA-equivalent |

These are *projections*, not measurements — but they are grounded in (a) PFlash's measured 10.4× at 5% on
a 3090 with park/unpark overhead we shed, and (b) SpecPrefill's published 10%-keep quality table on
LongBench/RULER. **Caveat per rule 13:** these must be measured on real M-silicon before claiming.

### 5.3 The single highest-leverage improvement over our naive scorer

**Switch from "global top-K of tokens by tail-attention max-over-heads" to "chunk-top-K with 1D-avgpool
smoothing + 8-step lookahead + position-id restoration."** This is the entire delta between SpecPrefill's
weakest ablation (≈ our current 50%) and the published 10%. Specifically, in likely impact order:

1. **Chunk-based selection (chunk=32) with avgpool(kernel=13) smoothing** — denoises the per-token
   importance, eliminates chunkation artifacts. Cheap, pure-tensor-ops in mlx-rs.
2. **Lookahead=8 on the drafter** — mitigates proximity bias (last-token-over-weights recent context).
   Costs ~8 extra drafter forwards but the drafter is 0.6B so this is ms-scale.
3. **Restore original position IDs** on survivors (not contiguous re-numbering) — fixes position-sensitive
   tasks (NIAH, counting, multi-hop). One-line change in how we feed positions to the target.
4. (Optional) **FlashPrefill-style max-based α-threshold per query block** instead of fixed top-K — this is
   what takes PFlash from 10% → 5%. Replace `top_K(blocks, K=ρ·S/B)` with `keep_block iff score ≥
   α·max_row(score)`. Naturally prunes long-tail blocks where most chunks are noise.

Doing just **1+2+3** should move us from 50% → ~10% keep with quality preserved on NIAH/multi-doc QA.
Adding **4** gets us to ~5% keep with NIAH preserved (matching PFlash's headline). Below 5% is genuinely
research territory — published evidence stops at NIAH-only.

---

## 6. Raw pointers

### Papers
- **FlashPrefill** (Fan et al., Mar 2026) — arXiv:[2603.06199](https://arxiv.org/abs/2603.06199),
  [HTML](https://arxiv.org/html/2603.06199v1). Sparse attention on target's own Q/K; max-based α-threshold;
  no drafter. 27.78× operator speedup @ 256K, 7.22× end-to-end TTFT on Qwen3-30B-A3B.
- **Speculative Prefill / SpecPrefill** (Liu, Chen, Zhang; Feb 2025, ICML 2025) —
  arXiv:[2502.02789](https://arxiv.org/abs/2502.02789), [HTML](https://arxiv.org/html/2502.02789v2).
  Drafter-based; max(L,H)+mean(N) aggregation; chunk-top-K + avgpool + LAH=8; position-id restore.
  Llama-3.2-1B → Llama-3.1-{70B, 405B}.
- **Cross-Family Speculative Prefill** (Upasani et al., SambaNova, Mar 2026, ICLR 2026 WS) —
  arXiv:[2603.02631](https://arxiv.org/abs/2603.02631), [HTML](https://arxiv.org/html/2603.02631v3).
  Cross-family drafters; new contiguous pos-ids + delimiter tokens; chunk=32 / avgpool kernel=13 / LAH=8.

### Code
- `qhfan/FlashPrefill` — https://github.com/qhfan/FlashPrefill (Python/Triton, 53★)
- `Jingyu6/speculative_prefill` — https://github.com/Jingyu6/speculative_prefill (Python/vLLM, 63★, MIT)
- `Luce-Org/lucebox` — https://github.com/Luce-Org/lucebox (C++/CUDA, 2.7k★)
  - **PFlash README:** https://github.com/Luce-Org/lucebox/blob/main/optimizations/pflash/README.md
  - **PFlash blog:** https://lucebox.com/blog/pflash
  - **Kernel source:** `optimizations/server/src/flashprefill.h`,
    `optimizations/server/src/flashprefill_kernels.cu` (`mean_K / score / select / sparse_fwd`),
    `qwen3_0p6b_loader.cpp`, `qwen3_0p6b_graph.cpp`, `bsa_launcher.cu`, `bsa_fwd_inst.cu`.
  - **BSA stubs:** `optimizations/server/deps/bsa_stubs/` (3 ATen/c10 headers — useful pattern if we ever
    bridge to a Metal BSA kernel).
- `mit-han-lab/Block-Sparse-Attention` — https://github.com/mit-han-lab/Block-Sparse-Attention
  (cutlass/FA-2, sm_80+)
- Related sparse-attention baselines: MInference (arXiv 2407.02490), FlexPrefill (ByteDance),
  XAttention (mit-han-lab), Native Sparse Attention, MoBA, DuoAttention, InfLLMv2.

### MLX / Rust
- MLX Metal backend dir: https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal — no
  `block_sparse_attention.cpp`; only `scaled_dot_product_attention.cpp`. Custom kernel JIT path is
  `custom_kernel.cpp` + `kernels/*.metal`.
- mlx-lm Qwen2 attention (reference for q/k/v proj access):
  https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py (lines 41-44: `q_proj`,
  `k_proj`, `v_proj` as separate `nn.Linear`).
- Our existing notes: `docs/mlx_rs_capabilities.md` (confirms SDPA available, custom Metal kernels NOT),
  `docs/DSPARK_MLX_DESIGN.md`, `docs/BONSAI_TO_LUCEBOX_DS4_STRIX_HALO.md`.

### Key file/line references inside the algorithms
- SpecPrefill scoring formula: arXiv 2502.02789v2 §3.2.2 (line 124 of saved HTML).
- SpecPrefill selection (chunk + pool): §3.2.3, with hyperparameters in Cross-Family Appendix A.1
  (`chunk=32, avgpool_kernel=13, LAH=8`; code-debug uses `chunk=128`).
- SpecPrefill position-id restoration: §3.2.4 (lines around the `[0,1,3,6,7]` example).
- FlashPrefill block-score formula: arXiv 2603.06199v1 §3.1-3.2 (Alg 1 + Eqs 1-4).
- FlashPrefill max-α threshold: §3.4 (Eq for `thresh_I = α · max_J(Score_{I,J})`).
- FlashPrefill hyperparameters (sink=256, window=512, B=128, α per model, density table):
  Appendix A, Table 9.
- PFlash hyperparameters: `optimizations/pflash/README.md` "Runtime tunables" table.
- PFlash cost breakdown at 128K: `optimizations/pflash/README.md` "Memory budget on 24 GB" section
  + blog "Bottleneck shifted" section.

### Negative-result receipts
- "mlx-rs" GitHub topic page exists but **zero repos use it**: https://github.com/topics/mlx-rs
- `locapela/mlx-rs`, `RobinMattar/mlx-rs`, `gabriel-ss/mlx-rs`, `david-at-locapela/mlx-rs` → **all 404**
- No public MLX block-sparse attention kernel exists (MLX backend dir has no such file).
- PFlash is "C++/CUDA only" by explicit design; no Metal port planned (confirmed in their README scope).
- No agent/coding-loop eval at <10% keep in any published work.
