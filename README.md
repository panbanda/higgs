# higgs

[![CI](https://github.com/panbanda/higgs/actions/workflows/ci.yml/badge.svg)](https://github.com/panbanda/higgs/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/panbanda/higgs)](https://github.com/panbanda/higgs/releases)
[![Crates.io](https://img.shields.io/crates/v/higgs)](https://crates.io/crates/higgs)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)

Unified AI gateway for Apple Silicon. Serve local MLX models and proxy to remote providers (OpenAI, Anthropic, Ollama, etc.) through a single endpoint with automatic format translation. Single static Rust binary, no Python runtime. Built on [mlx-rs](https://github.com/oxideai/mlx-rs).

## Install

```bash
brew install panbanda/brews/higgs
```

Or build from source (Rust 1.87.0+, Xcode CLI Tools):

```bash
cargo build --release
```

## Quick Start

### Simple mode (no config file)

```bash
higgs serve --model mlx-community/Llama-3.2-1B-Instruct-4bit
higgs serve --model mlx-community/Llama-3.2-1B-Instruct-4bit --model mlx-community/Qwen3-1.7B-4bit
```

Accepts HuggingFace model IDs (resolved from `~/.cache/huggingface/hub/`) or local paths. Prompts to download if not cached. Models must be **MLX safetensors format** from [mlx-community](https://huggingface.co/mlx-community).

### Gateway mode (config file)

```bash
higgs init        # create ~/.config/higgs/config.toml
higgs serve       # start with config
higgs start       # start as background daemon
higgs attach      # attach TUI dashboard to running daemon
higgs stop        # stop daemon
```

## Features

### Local inference
- **OpenAI + Anthropic APIs** -- chat completions, text completions, embeddings, messages
- **Structured output** -- `json_schema` response format (100% schema compliance)
- **Reasoning models** -- `<think>` tag extraction to `reasoning_content`
- **Continuous batching** -- 755 tok/s aggregate at 8 concurrent requests
- **Radix tree prefix cache** -- shared prefix reuse across requests
- **Vision** -- multimodal image+text (LLaVA-Qwen2)
- **11 architectures** -- LLaMA, Mistral, Qwen2/3, Qwen3-MoE, Qwen3-Next, Gemma 2, Phi-3, Starcoder2, DeepSeek-V2, LLaVA-Qwen2

### Gateway
- **Remote providers** -- proxy requests to OpenAI, Anthropic, Ollama, or any OpenAI-compatible API
- **Format translation** -- send OpenAI requests to Anthropic providers (and vice versa) with automatic conversion of request/response formats, including streaming
- **Pattern routing** -- regex-based model name matching to route requests to the right provider
- **Model rewriting** -- map model aliases to upstream model names
- **Auto-router** -- classify requests using a local LLM to pick the best provider
- **Metrics dashboard** -- TUI with live request rates, latency, token throughput, and error tracking
- **Daemon mode** -- `higgs start`/`stop`/`attach` for background operation
- **Config management** -- `higgs config get/set`, `higgs doctor` for validation

## Configuration

### Simple mode (CLI flags)

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

### Gateway mode (config file)

Run `higgs init` to create `~/.config/higgs/config.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8000
# max_tokens = 32768
# timeout = 300.0
# api_key = "sk-..."

# --- Local models ---
[[models]]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
# batch = false

# --- Remote providers ---
[provider.anthropic]
url = "https://api.anthropic.com"
format = "anthropic"

[provider.openai]
url = "https://api.openai.com"
format = "openai"

[provider.ollama]
url = "http://localhost:11434"
strip_auth = true

# --- Routes ---
# First regex match wins. Requests matching a local model name are served locally.

[[routes]]
pattern = "claude-.*"
provider = "anthropic"

[[routes]]
pattern = "gpt-.*"
provider = "openai"

# Model rewriting: requests for "my-alias" are sent to the provider as "actual-model-name"
# [[routes]]
# pattern = "my-alias"
# provider = "openai"
# model = "gpt-4o"

# --- Default route ---
[default]
provider = "higgs"   # "higgs" = local models only; set to a provider name to proxy unmatched requests

# --- Auto router (optional) ---
# Classify requests with a local LLM to pick the best provider automatically.
# [auto_router]
# enabled = true
# model = "mlx-community/Arch-Router-1.5B-4bit"
# timeout_ms = 2000

# --- Metrics & dashboard ---
[retention]
enabled = true
minutes = 60

[logging.metrics]
enabled = true
# path = "~/.config/higgs/logs/metrics.jsonl"
# max_size_mb = 50
# max_files = 5
```

#### Provider options

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | string | *(required)* | Base URL of the upstream API |
| `format` | `"openai"` or `"anthropic"` | `"openai"` | API format the provider speaks |
| `api_key` | string | *(none)* | API key to inject into proxied requests |
| `strip_auth` | bool | `false` | Remove the client's Authorization header before proxying |
| `stub_count_tokens` | bool | `false` | Return a stub for `/v1/messages/count_tokens` |

#### Route options

| Field | Type | Description |
|---|---|---|
| `pattern` | regex | Match against the `model` field in requests |
| `provider` | string | Provider name to forward to |
| `model` | string | Rewrite the model field before forwarding |
| `name` | string | Human label (used by auto-router) |
| `description` | string | Route description (used by auto-router for classification) |

## API

**OpenAI**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`
**Anthropic**: `/v1/messages`, `/v1/messages/count_tokens`
**Metrics**: `/metrics` (JSON)
**Health**: `/health`

Format translation works transparently: send an OpenAI-format request to higgs and it will translate to Anthropic format if the matched route points to an Anthropic provider (and vice versa), including streaming responses.

```bash
# Local model
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
       "messages": [{"role": "user", "content": "Hello!"}]}'

# Proxied to Anthropic (translated from OpenAI format automatically)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
  -d '{"model": "claude-sonnet-4-6",
       "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Shell integration

Add to your shell profile to point AI tools at higgs:

```bash
eval "$(higgs shellenv)"
# Exports ANTHROPIC_BASE_URL and OPENAI_BASE_URL when the server is reachable
```

## CLI Commands

| Command | Description |
|---|---|
| `higgs serve` | Start the server in the foreground |
| `higgs start` | Start as a background daemon |
| `higgs stop` | Stop a running daemon |
| `higgs attach` | Attach TUI dashboard to a running daemon |
| `higgs init` | Create default config at `~/.config/higgs/config.toml` |
| `higgs shellenv` | Print `export` lines for `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` |
| `higgs config get <key>` | Read a config value (dot-separated key) |
| `higgs config set <key> <value>` | Write a config value |
| `higgs config path` | Print the resolved config file path |
| `higgs doctor` | Validate config, check model paths, probe providers |

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
| Gemma-2-2B-4bit | 163 | 185 | 91 | -- | -- |
| Phi-3-mini-4bit | 171 | 170 | 95 | -- | -- |
| Starcoder2-3B-4bit | 107 | 176 | 165 | -- | -- |
| DeepSeek-V2-Lite-4bit | 140 | 174 | 99 | -- | -- |

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
| Gemma-2-2B-4bit | 1,645 | 2,329 | 2,350 |
| Phi-3-mini-4bit | 2,126 | 2,548 | 2,573 |
| DeepSeek-V2-Lite-4bit | 8,528 | 8,972 | 8,998 |

### Feature comparison

| | higgs | vllm-mlx |
|---|---|---|
| Structured output (10 prompts, JSON schema) | 100% | 0% |
| Reasoning extraction (5 questions, Qwen3) | 5/5 | 4/5 |
| All architectures produce coherent output | Yes | Yes |

## Development

```bash
cargo test -- --test-threads=1
cargo clippy
cargo fmt --check
```

## License

MIT
