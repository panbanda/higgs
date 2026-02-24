# higgs

[![CI](https://github.com/panbanda/higgs/actions/workflows/ci.yml/badge.svg)](https://github.com/panbanda/higgs/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/panbanda/higgs)](https://github.com/panbanda/higgs/releases)
[![Crates.io](https://img.shields.io/crates/v/higgs)](https://crates.io/crates/higgs)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)

OpenAI and Anthropic-compatible inference server for Apple Silicon, built in Rust on top of [mlx-rs](https://github.com/oxideai/mlx-rs).

Runs quantized LLMs locally using the Metal GPU with no Python runtime.

## Features

- **OpenAI API** -- chat completions, text completions, embeddings
- **Anthropic API** -- messages
- **Advanced sampling** -- top-k, min-p, repetition/frequency/presence penalties, logprobs
- **Structured output** -- `response_format` with JSON mode and JSON schema constraints
- **Reasoning models** -- `<think>` tag parsing with `reasoning_content` in responses
- **Continuous batching** -- concurrent request processing with shared decode loop
- **Radix tree prefix cache** -- shared prefix reuse across requests
- **Vision** -- multimodal image+text via LLaVA-Qwen2 (nanoLLaVA)
- **11 architectures** -- LLaMA, Mistral, Qwen2/3, Qwen3-MoE, Qwen3-Next, Gemma 2, Phi-3, Starcoder2, DeepSeek-V2, LLaVA-Qwen2
- **Zero dependencies** -- single static Rust binary, no Python runtime

### Comparison

| | higgs (Rust) | [vllm-mlx](https://github.com/waybarrios/vllm-mlx) (Python) | mlx_lm (Python) |
|---|---|---|---|
| **Install** | `brew install higgs` | `pip install` + Python + mlx ecosystem | `pip install mlx-lm` + Python |
| **Run** | `higgs --model org/name` | `vllm-mlx --model org/name` | Write a script |
| **Deploy** | Single static binary | Ship a Python environment | Ship a Python environment |
| **Modalities** | Text + image | Text + image + video + audio | Text |
| **Batching** | Continuous batching | Continuous batching | Single request |
| **Structured output** | JSON mode + JSON schema | JSON mode + JSON schema | No |
| **Text perf** | ~450 tok/s (1B) | Built on mlx_lm | ~453 tok/s (1B) |

Text inference performance is near-identical to Python `mlx_lm` (within 1-2%) since both use the same MLX Metal kernels. vllm-mlx supports more modalities (video, audio). higgs ships as a zero-dependency binary with structured output and vision support.

## Requirements

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)

## Install

```bash
brew tap panbanda/brews
brew install higgs
```

**Build from source** (requires Rust 1.87.0+, Xcode Command Line Tools, and `huggingface-cli`):

```bash
cargo build --release
```

## Usage

```bash
# HuggingFace model ID (resolved from local HF cache)
higgs --model mlx-community/Llama-3.1-8B-Instruct-4bit

# Local MLX model directory
higgs --model ~/dev/models/My-Custom-Model

# Multiple models
higgs --model mlx-community/Llama-3.1-8B-Instruct-4bit --model mlx-community/Qwen3-Coder-Next-4bit
```

The `--model` flag accepts HuggingFace model IDs (`org/name`) or local directory paths. HuggingFace IDs are resolved from the local cache at `~/.cache/huggingface/hub/` (or `$HF_HUB_CACHE` if set, else `$HF_HOME/hub`).

If a model ID is not found in the cache and stdin is a terminal, the server will prompt to download it:

```
Model 'mlx-community/Llama-3.1-8B-Instruct-4bit' not found in HuggingFace cache. Download now? [y/N]
```

In non-interactive mode (piped or scripted), startup fails immediately with a hint to run `huggingface-cli download org/name` manually.

Models must be in **MLX safetensors format**. Pre-quantized weights are available from [mlx-community](https://huggingface.co/mlx-community) on HuggingFace. To convert your own:

```bash
pip install mlx-lm
mlx_lm.convert --hf-path meta-llama/Llama-3.1-8B-Instruct -q --upload-repo your-org/Llama-3.1-8B-4bit-mlx
```

## Configuration

Settings are resolved in order (later wins): defaults, environment variables, CLI flags.

| CLI Flag | Env Variable | Default | Description |
|---|---|---|---|
| `--model` | `HIGGS_MODELS` | *(required)* | Model path or HF model ID (repeatable; env uses JSON array `'["a","b"]'`) |
| `--host` | `HIGGS_HOST` | `0.0.0.0` | Bind address |
| `--port` | `HIGGS_PORT` | `8000` | Bind port |
| `--max-tokens` | `HIGGS_MAX_TOKENS` | `32768` | Default max generation tokens |
| `--api-key` | `HIGGS_API_KEY` | *(none)* | Bearer token for auth (disabled if unset) |
| `--rate-limit` | `HIGGS_RATE_LIMIT` | `0` | Requests per minute per client (0 = disabled) |
| `--timeout` | `HIGGS_TIMEOUT` | `300` | Request timeout in seconds |

Log level is controlled via `RUST_LOG` (e.g., `RUST_LOG=debug`).

## API

### OpenAI-compatible

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completions (streaming and non-streaming) |
| `/v1/completions` | POST | Text completions |
| `/v1/embeddings` | POST | Token embeddings |
| `/v1/models` | GET | List loaded models |

### Anthropic-compatible

| Endpoint | Method | Description |
|---|---|---|
| `/v1/messages` | POST | Messages API |
| `/v1/messages/count_tokens` | POST | Token counting |

### Health

| Endpoint | Method |
|---|---|
| `/health` | GET |

### Example

```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'

# Structured output (JSON schema)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "messages": [{"role": "user", "content": "List 3 colors"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "colors",
        "schema": {"type": "object", "properties": {"colors": {"type": "array", "items": {"type": "string"}}}}
      }
    }
  }'

# With logprobs
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "logprobs": true,
    "top_logprobs": 5
  }'

# List all loaded models
curl http://localhost:8000/v1/models
```

## Supported Models

| Architecture | `model_type` | Examples |
|---|---|---|
| LLaMA | `llama` | Llama 3, Llama 3.1, CodeLlama |
| Mistral | `mistral` | Mistral 7B, Mixtral (dense only) |
| Qwen2 | `qwen2` | Qwen2, Qwen2.5 |
| Qwen3 | `qwen3` | Qwen3 |
| Qwen3-Next | `qwen3_next` | Qwen3-Coder (hybrid SSM/attention + MoE) |
| Qwen3-MoE | `qwen3_moe` | Qwen3-30B-A3B, Qwen3-235B-A22B (sparse MoE) |
| Gemma 2 | `gemma2` | Gemma 2 2B/9B/27B (logit soft-capping, sliding window) |
| Phi-3 | `phi3` | Phi-3 Mini/Small/Medium (SuRoPE) |
| Starcoder2 | `starcoder2` | Starcoder2 3B/7B/15B (sliding window, MQA) |
| DeepSeek-V2 | `deepseek_v2` | DeepSeek-V2-Lite, DeepSeek-V2 (MLA + MoE) |
| LLaVA-Qwen2 | `llava-qwen2` | nanoLLaVA-1.5 (vision-language, SigLIP + Qwen2) |

## Performance

Decode throughput on M4 Max 128GB. Prompt: long-form technical design document (~100 input tokens), 500 generated tokens, temperature=0. Each engine runs alone (no concurrent processes), with a warmup pass before measurement.

| Model | Rust | Python mlx_lm | llama.cpp | Ollama |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct-4bit | 449.8 | 453.3 | 313.5 | 305.2 |
| Mistral-7B-Instruct-v0.3-4bit | 101.1 | 101.6 | 86.7 | 84.7 |
| Qwen3-1.7B-4bit | 303.4 | 305.4 | 215.7 | 183.1 |
| Qwen3-30B-A3B-8bit (MoE) | 73.2 | 88.0 | 82.7 | 72.8 |

Quantization: MLX models use 4-bit (8-bit for MoE). llama.cpp/Ollama use Q4_K_M (Q8_0 for MoE).

Peak RSS memory (MB) after 500-token generation:

| Model | Rust | Python mlx_lm |
|---|---|---|
| Llama-3.2-1B-Instruct-4bit | 883 | 1,094 |
| Mistral-7B-Instruct-v0.3-4bit | 3,953 | 4,062 |
| Qwen3-1.7B-4bit | 1,089 | 1,316 |
| Qwen3-30B-A3B-8bit (MoE) | 31,095 | 31,348 |

Model weights (in unified GPU memory) dominate RSS. Rust saves ~200 MB on small models by eliminating the Python runtime.

## Development

```bash
cargo check
cargo test -- --test-threads=1   # single-threaded to avoid Metal GPU teardown issues
cargo clippy
cargo fmt --check
```

## License

MIT
