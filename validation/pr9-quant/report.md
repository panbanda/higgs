# Validation report: pr9-quant (per-tensor quantization loader; ds4 P2 loader half)

- chip: Apple M4 Max
- ram_gb: 128
- macos_version: 26.5.1
- branch: claude/ds4-p9-asymmetric-quant

## Scope decision (recorded honestly)
ds4 P2 has two halves. This PR ships the LOADER half (per-tensor quantization settings, dense-mode
QLinear/QEmbedding, load-time width validation). The CALIBRATION half (activation collector +
imatrix + asymmetric recipe) and the flagship quality claim (asymmetric >= uniform logprob at equal
bytes) are DEFERRED: validating that claim requires producing uniform- and mixed-quant variants of a
real MoE model from its bf16 checkpoint (~31 GB download + ~14 GB outputs), and the validation host
had 5.4 GB of free disk. Per program rules an unvalidated claim does not merge, so it was cut from
scope rather than shipped unproven. Recorded in docs/ds4-analysis-2026-08.md as deferred/blocked.

## Pre-registered criteria for the shipped half
1. Mixed-quant checkpoints load correctly (real-checkpoint proof, not just synthetic).
2. Per-layer/per-tensor configs that an architecture cannot honor fail loudly at load, never silent
   garbage or request-time MLX errors.
3. No dense-model load regressions (full suites).

## Results

### Criterion 1 — PASS (real mixed checkpoint)
Built a real mixed checkpoint from mlx-community/Qwen3-1.7B-4bit by dequantizing the tied embedding
to dense bf16 and marking "model.embed_tokens": false in config.quantization (script under
scripts in this report's raw notes; mlx 0.32). Baseline main build: loads, then EVERY request fails
with MLX "[dequantize] The matrix should be given as a uint32" (request-time failure, criterion-2
violation class). This branch: loads and generates coherent greedy text. Also covered by unit tests:
mixed dense+quant synthetic load/forward; per-tensor resolution fixture modeled on real
mlx-community config shapes ("quantization" map with scalar defaults + per-tensor entries + false;
duplicate "quantization_config" ignored).

### Criterion 2 — PASS
Load-time width validation (packed last dim == in*bits/32, scales last dim == in/group_size) with
errors naming the tensor path; negative test included. Scalar-only architectures (phi3, starcoder2,
gemma2, llava) now REJECT configs carrying per-tensor entries with a clear load error instead of
producing request-time MLX failures. Fused MoE expert groups reject differing per-expert settings
within one layer loudly (documented limitation; per-layer asymmetry supported).

### Criterion 3 — PASS
Full suites green: higgs-models 435, higgs 474+99, higgs-engine 287. Existing qwen3_next per-layer
gate override and GDN per-layer scan behavior preserved (their tests unchanged and green).

## Per-architecture per-tensor capability (documented)
- Qwen3-Next, DeepSeek-V2: embed/lm_head + MoE expert projections (per-layer granularity within
  fused groups).
- Plain transformer (qwen3/llama/mistral/qwen2): embed_tokens + lm_head.
- phi3/starcoder2/gemma2/llava: scalar-only; per-tensor entries rejected loudly.

Verdict: PASS for the shipped scope; calibration half deferred (disk-blocked), not merged unproven.
