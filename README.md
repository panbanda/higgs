# higgs

[![CI](https://github.com/panbanda/higgs/actions/workflows/ci.yml/badge.svg)](https://github.com/panbanda/higgs/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/panbanda/higgs)](https://github.com/panbanda/higgs/releases)
[![Crates.io](https://img.shields.io/crates/v/higgs)](https://crates.io/crates/higgs)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)

OpenAI and Anthropic-compatible inference server for Apple Silicon. Single static Rust binary, no Python runtime. Built on [mlx-rs](https://github.com/oxideai/mlx-rs).

## Install

```bash
brew install panbanda/brews/higgs
```

Or build from source (Rust 1.87.0+, Xcode CLI Tools):

```bash
cargo build --release
```

## Usage

```bash
higgs --model mlx-community/Llama-3.2-1B-Instruct-4bit
higgs --model mlx-community/Llama-3.2-1B-Instruct-4bit --model mlx-community/Qwen3-1.7B-4bit
```

Accepts HuggingFace model IDs (resolved from `~/.cache/huggingface/hub/`) or local paths. Prompts to download if not cached.

Models must be **MLX safetensors format** from [mlx-community](https://huggingface.co/mlx-community).

## Features

- **OpenAI + Anthropic APIs** -- chat completions, text completions, embeddings, messages
- **Structured output** -- `json_schema` response format (100% schema compliance)
- **Reasoning models** -- `<think>` tag extraction to `reasoning_content`
- **Continuous batching** -- 755 tok/s aggregate at 8 concurrent requests
- **Radix tree prefix cache** -- shared prefix reuse across requests
- **Vision** -- multimodal image+text (LLaVA-Qwen2)
- **11 architectures** -- LLaMA, Mistral, Qwen2/3, Qwen3-MoE, Qwen3-Next, Gemma 2, Phi-3, Starcoder2, DeepSeek-V2, LLaVA-Qwen2

## Performance

All benchmarks on M4 Max 128GB. Temperature=0, warmup pass excluded.

### Decode throughput (tok/s)

Single request, 500 generated tokens, median of 3 runs.

| Model | higgs | mlx_lm | vllm-mlx | llama.cpp | Ollama |
|---|---|---|---|---|---|
| Llama-3.2-1B-4bit | 448 | 421 | 433 | 314 | 305 |
| Mistral-7B-v0.3-4bit | 103 | 103 | -- | 87 | 85 |
| Qwen3-1.7B-4bit | 305 | 293 | 300 | 216 | 183 |
| Qwen3-30B-A3B-8bit | 75 | 86 | 87 | 83 | 73 |
| Gemma-2-2B-4bit | 158 | 185 | 91 | -- | -- |
| Phi-3-mini-4bit | 171 | 170 | 95 | -- | -- |
| Starcoder2-3B-4bit | 107 | 176 | 165 | -- | -- |
| DeepSeek-V2-Lite-4bit | 137 | 174 | 99 | -- | -- |

MLX models use 4-bit (8-bit for MoE). llama.cpp/Ollama use Q4_K_M (Q8_0 for MoE).

### Continuous batching (Llama-1B)

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
| Gemma-2-2B-4bit | 1,639 | 2,307 | 2,350 |
| Phi-3-mini-4bit | 2,126 | 2,548 | 2,573 |
| DeepSeek-V2-Lite-4bit | 8,528 | 8,972 | 8,998 |

### Feature comparison

| | higgs | vllm-mlx |
|---|---|---|
| Structured output (10 prompts, JSON schema) | 100% | 0% |
| Reasoning extraction (5 questions, Qwen3) | 5/5 | 4/5 |
| All architectures produce coherent output | Yes | Yes |

## Configuration

| CLI Flag | Env Variable | Default | Description |
|---|---|---|---|
| `--model` | `HIGGS_MODELS` | *(required)* | Model path or HF ID (repeatable) |
| `--host` | `HIGGS_HOST` | `0.0.0.0` | Bind address |
| `--port` | `HIGGS_PORT` | `8000` | Bind port |
| `--max-tokens` | `HIGGS_MAX_TOKENS` | `32768` | Max generation tokens |
| `--api-key` | `HIGGS_API_KEY` | *(none)* | Bearer token for auth |
| `--rate-limit` | `HIGGS_RATE_LIMIT` | `0` | Requests/min per client |
| `--timeout` | `HIGGS_TIMEOUT` | `300` | Request timeout (seconds) |
| `--batch` | -- | `false` | Enable continuous batching |

## API

**OpenAI**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`
**Anthropic**: `/v1/messages`, `/v1/messages/count_tokens`
**Health**: `/health`

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
       "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Supported Architectures

| Architecture | `model_type` | Examples |
|---|---|---|
| LLaMA | `llama` | Llama 3/3.1, CodeLlama |
| Mistral | `mistral` | Mistral 7B |
| Qwen2 | `qwen2` | Qwen2, Qwen2.5 |
| Qwen3 | `qwen3` | Qwen3 |
| Qwen3-Next | `qwen3_next` | Qwen3-Coder (SSM hybrid) |
| Qwen3-MoE | `qwen3_moe` | Qwen3-30B-A3B (sparse MoE) |
| Gemma 2 | `gemma2` | Gemma 2 2B/9B/27B |
| Phi-3 | `phi3` | Phi-3 Mini/Small/Medium |
| Starcoder2 | `starcoder2` | Starcoder2 3B/7B/15B |
| DeepSeek-V2 | `deepseek_v2` | DeepSeek-V2-Lite (MLA + MoE) |
| LLaVA-Qwen2 | `llava-qwen2` | nanoLLaVA-1.5 (vision) |

## Development

```bash
cargo test -- --test-threads=1
cargo clippy
cargo fmt --check
```

## License

MIT
