# Validation report: pr3-mla (MLA latent KV cache with weight absorption)

- chip: Apple M4 Max
- ram_gb: 128
- macos_version: 26.5.1
- branch: claude/ds4-p3-mla-latent-cache
- model: mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx (MoE, 27 layers, kv_lora_rank 512, rope 64)
- tools: bench_frontier (PR2) + quality_gate (PR1), built from a local merge of this branch with those branches

## Pre-registered acceptance criteria
1. >= 1.25x decode t/s at 16k+ context (MLA on vs off).
2. >= 5x measured KV bytes/token reduction.
3. PR1 quality gates green (token-exact or documented max-logprob-delta).
4. Prefix-cache store -> lookup -> continue test passes.

## Results

### Criterion 1: decode throughput — PASS
bench_frontier, greedy 64-token probes, medians of 3 sweeps (dense 32k: 1 sweep, see caveat):

| Context | Dense median t/s | MLA median t/s | Speedup |
| ---: | ---: | ---: | ---: |
| 2048 | 121.0 | 122.4 | 1.01x |
| 8192 | 77.5 | 105.5 | 1.36x |
| 16384 | 52.5 | 90.0 | 1.71x |
| 32768 | 32.6 | 71.3 | 2.19x |

Caveat honestly noted: the dense-mode 3-sweep run through 32k was killed by the OS (exit 137, memory
pressure: ~11.3 GB dense KV on top of 9.5 GB weights, repeated sweeps); the dense 32k number is from a
single sweep. The MLA-mode 3x32k run completed without incident — itself evidence of the memory win.

### Criterion 2: KV memory — PASS
Measured kv_bytes/token: dense 276,480-285,120 (capacity rounding); MLA 31,104 = exactly
27 layers x (512+64) x 2 bytes. Reduction: 8.9x.

### Criterion 3: quality gates — PASS with documented drift
quality_gate check vs the PR1 committed baseline (recorded via the decompressed path), 12 prompts x 64
greedy tokens, teacher-forced:
- MLA off (control): token-exact 12/12, max |delta logprob| = 0.0.
- MLA on: token-exact 11/12; per-prompt max |delta logprob| 0.010-0.062 for the 11 exact prompts;
  one prompt flips a near-tie argmax and compounds to 0.227.
The drift is the expected consequence of absorption: the absorbed path multiplies by the dequantized
fp16 kv_b_proj instead of running the fused 4-bit quantized matmul. The plan's a-priori estimate
(~1e-3) was optimistic for a 4-bit checkpoint; observed typical drift is ~3e-2 to 6e-2 logprob.
Documented here per the pre-registered "token-exact or documented max-logprob-delta" criterion.
Rollout remains opt-in (HIGGS_MLA_LATENT_CACHE, default off).

### Criterion 4: prefix cache — PASS
test_mla_cache_store_lookup_and_continue (paged_prefix_cache.rs) covers store -> multi-block lookup ->
from_latent_array materialization -> continued append. cargo test -p higgs-engine: 288 passed.

### Additional correctness
- SDPA smoke (examples/sdpa_mla_smoke.rs): fused SDPA accepts MQA 576/512 decode + causal prefill,
  max diff vs explicit softmax ~2e-7.
- Absorbed-vs-decompressed parity unit test (H=4 synthetic weights): fails on a scrambled weight
  layout (delta 210), passes on the per-head split (< 1e-2), for both L=6 prefill and 1-token decode.
- Review found and fixed a real per-head weight-layout bug before any hardware run; the parity test
  now guards it.

Verdict: PASS — recommend merge with default OFF; flip-on decision belongs to a follow-up after
longer-context soak and the config-surface (doctor/README/init) work.
