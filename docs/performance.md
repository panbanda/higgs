# Performance and Apple Silicon notes

Current head-to-head numbers against oMLX are in [omlx-comparison.md](omlx-comparison.md). This page keeps the earlier cross-engine measurements (mlx_lm, vllm-mlx, llama.cpp, Ollama) and the Apple Silicon deployment notes.

## Apple Silicon notes

- Release artifacts bundle `mlx.metallib`.
- Source builds also require `mlx.metallib` next to the executable. Higgs now restores it automatically from Cargo build output when possible, then fails loudly if it still cannot be found.
- `[memory].raise_wired_limit` defaults to `false`. Enable it only when you explicitly want MLX to raise the process wired-memory limit.
- With several large models, `load = "on_demand"` plus `max_loaded_models`, `guard`, or `idle_ttl` keeps unified memory from filling up; unloading shuts the engine down and clears the MLX buffer cache.
- Every model is served by one engine that decodes concurrent requests together in batched forward passes whenever the architecture allows it. A model that cannot (see [docs/models.md](models.md)), or one using TurboQuant KV, queues requests and decodes them one at a time. `batch = false` (or `higgs serve --no-batch`) limits any model to one request at a time; `true` and `"auto"` are accepted for old configuration files and change nothing. See [docs/configuration.md](configuration.md#concurrent-requests).
- `prefill.yield_tokens` can interleave long prompt prefills with decode for models that decode concurrent requests; use `0` or omit it for the synchronous default.

## Earlier measurements

These measurements were run on an M4 Max with 128 GB and are kept for the comparison with engines other than oMLX. For current Higgs numbers, use the oMLX comparison. Methodology, harness details, and benchmark-driven defaults are documented in [docs/benchmarking.md](benchmarking.md).

A head-to-head comparison with oMLX (where Higgs is ahead, level, and behind, with reproduction steps) is in [docs/omlx-comparison.md](omlx-comparison.md).

### Decode Throughput (tok/s)

Single request, 500 generated tokens, median of 3 runs.

| Model | higgs | mlx_lm | vllm-mlx | llama.cpp | Ollama |
|---|---|---|---|---|---|
| Llama-3.2-1B-4bit | 448 | 421 | 433 | 314 | 305 |
| Mistral-7B-v0.3-4bit | 103 | 103 | -- | 87 | 85 |
| Qwen3-1.7B-4bit | 305 | 293 | 300 | 216 | 183 |
| Qwen3-30B-A3B-8bit | 75 | 86 | 87 | 83 | 73 |
| Gemma-2-2B-4bit | 163 | 185 | 91 | -- | -- |
| Phi-3-mini-4bit | 171 | 170 | 95 | -- | -- |
| Starcoder2-3B-4bit | 107 | 176 | 165 | -- | -- |
| DeepSeek-V2-Lite-4bit | 140 | 174 | 99 | -- | -- |

MLX models use 4-bit, or 8-bit for MoE. `llama.cpp` and Ollama use `Q4_K_M`, or `Q8_0` for MoE.

### MoE Prefill (time to first token)

Measured on DeepSeek-V2-Lite-4bit with global batch sorting before `gather_qmm`.

| Prompt tokens | Before | After | Speedup |
|---|---|---|---|
| 59 | 472ms | 227ms | 2.1x |
| 481 | 3,734ms | 863ms | 4.3x |
| 1,831 | 14,390ms | 3,123ms | 4.6x |
| 4,532 | 37,489ms | 8,860ms | 4.2x |

### Continuous Batching (Llama-1B)

| Concurrent requests | higgs tok/s | vllm-mlx tok/s |
|---|---|---|
| 1 | 280 | 250 |
| 2 | 585 | 459 |
| 4 | 698 | 510 |
| 8 | 755 | 646 |

### Memory (RSS in MB)

| Model | higgs | mlx_lm | vllm-mlx |
|---|---|---|---|
| Llama-3.2-1B-4bit | 974 | 1,356 | 1,380 |
| Mistral-7B-v0.3-4bit | 3,965 | 4,384 | -- |
| Qwen3-1.7B-4bit | 1,127 | 1,609 | 1,641 |
| Qwen3-30B-A3B-8bit | 31,139 | 31,640 | 31,658 |
| Gemma-2-2B-4bit | 1,645 | 2,329 | 2,350 |
| Phi-3-mini-4bit | 2,126 | 2,548 | 2,573 |
| DeepSeek-V2-Lite-4bit | 8,528 | 8,972 | 8,998 |

### Feature Comparison

| | higgs | vllm-mlx |
|---|---|---|
| Structured output (10 prompts, JSON schema) | 100% | 0% |
| Reasoning extraction (5 questions, Qwen3) | 5/5 | 4/5 |
| All architectures produce coherent output | Yes | Yes |

