### Single-request decode throughput

Prompt: ~100 tokens, generation: 500 tokens, temperature=0, median of 3 runs.  
M4 Max 128GB. One server at a time, warmup pass excluded.

| Model | mlx-server (tok/s) | mlx-server RSS (MB) | vllm-mlx (tok/s) | vllm-mlx RSS (MB) |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct-4bit | 448.4 | 974 | 432.8 | 1380 |
| Qwen3-1.7B-4bit | 305.2 | 1126 | 299.6 | 1641 |
| Qwen3-30B-A3B-8bit | 74.5 | 31139 | 86.8 | 31658 |

### Continuous batching throughput

Model: Llama-3.2-1B-Instruct-4bit. Concurrent requests, 200 tokens each.

| Concurrency | mlx-server tok/s | mlx-server TTFT p50 | vllm-mlx tok/s | vllm-mlx TTFT p50 |
|---|---|---|---|---|
| 1 | 279.6 | 134ms | 249.7 | 41ms |
| 2 | 585.1 | 43ms | 458.9 | 16ms |
| 4 | 698.2 | 81ms | 509.6 | 24ms |
| 8 | 754.8 | 126ms | 645.8 | 49ms |

### Structured output (JSON schema compliance)

Model: Llama-3.2-1B-Instruct-4bit. 10 prompts, `json_schema` response format with 4 required fields.

| | mlx-server | vllm-mlx |
|---|---|---|
| Schema compliance | 100.0% | 0.0% |

### Log probability accuracy

Model: Llama-3.2-1B-Instruct-4bit. 5 prompts, `top_logprobs=5`.

| Metric | Value |
|---|---|
| Top-1 token agreement (mlx-server vs vllm-mlx) | 0% |
| Mean absolute error on log probabilities | None |

*Both servers run the same MLX weights; any gap reflects implementation divergence.*

### Reasoning model (think-tag extraction)

Model: Qwen3-1.7B-4bit. 5 math/logic questions.

| | mlx-server | vllm-mlx |
|---|---|---|
| `reasoning_content` populated | 5/5 | 4/5 |
| Correct answers | 5/5 | 5/5 |

### Embeddings

Model: all-MiniLM-L6-v2-4bit. 8 STS-style sentence pairs.

| Metric | mlx-server | vllm-mlx |
|---|---|---|
| Spearman correlation (vs reference scores) | 0.357 | 0.833 |
| Mean L2 norm (should be ~1.0) | 1.0 | 1.0 |
| Embedding dimension | 2048 | 384 |

### Vision (multimodal)

Model: nanoLLaVA-1.5-4bit. 5 solid-color test images, single-word color identification.

| | mlx-server | vllm-mlx |
|---|---|---|
| Color identification accuracy | 0.0% | 0.0% |

### Architecture smoke tests

Single 100-token generation, temperature=0. Coherent = non-empty, non-garbled output.

| Architecture | mlx-server tok/s | mlx-server RSS (MB) | mlx-server coherent | vllm-mlx tok/s | vllm-mlx RSS (MB) | vllm-mlx coherent |
|---|---|---|---|---|---|---|
| gemma2 | 121.7 | 1633 | True | 91.0 | 2350 | True |
| phi3 | 133.9 | 2124 | True | 95.0 | 2573 | True |
| starcoder2 | 104.5 | 1907 | True | 164.7 | 2342 | True |
| deepseek_v2 | 100.3 | 8524 | True | 99.0 | 8998 | True |
