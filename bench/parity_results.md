## Parity Benchmark: higgs (Rust) vs mlx_lm (Python)

Max tokens: 200, temperature: 0, single request.

| Architecture | Model | Python tok/s | Rust tok/s | Ratio | Python RSS (MB) | Rust RSS (MB) | Word Overlap |
|---|---|---|---|---|---|---|---|
| llama | Llama-3.2-1B-Instruct-4bit | 420.9 | 454.6 | 1.08x | 1356 | 974 | 30% |
| mistral | Mistral-7B-Instruct-v0.3-4bit | 102.9 | 103.3 | 1.00x | 4384 | 3965 | 30% |
| qwen2 | Qwen3-1.7B-4bit | 292.8 | 307.1 | 1.05x | 1609 | 1127 | 0% |
| qwen3_moe | Qwen3-30B-A3B-8bit | 85.7 | 75.1 | 0.88x | 31640 | 31138 | 0% |
| gemma2 | gemma-2-2b-it-4bit | 184.8 | 157.9 | 0.85x | 2307 | 1639 | 19% |
| phi3 | Phi-3-mini-4k-instruct-4bit | 170.0 | 170.9 | 1.01x | 2548 | 2126 | 66% |
| starcoder2 | starcoder2-3b-4bit | 176.2 | 107.3 | 0.61x | 2318 | 1908 | 100% |
| deepseek_v2 | DeepSeek-V2-Lite-Chat-4bit-mlx | 173.7 | 137.3 | 0.79x | 8972 | 8528 | 37% |
