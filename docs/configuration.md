# Configuration

This document collects the full CLI, environment, and config-file reference for Higgs.

## Modes

- **Simple mode**: pass one or more `--model` flags to serve local MLX models without a config file.
- **Gateway mode**: run `higgs init` and edit `~/.config/higgs/config.toml` to combine local models, providers, routes, metrics, and daemon behavior.

## Simple Mode

### CLI flags and environment variables

| CLI Flag | Env Variable | Default | Description |
|---|---|---|---|
| `--model` | `HIGGS_MODELS` | *(required)* | Model path or HF ID (repeatable) |
| `--host` | `HIGGS_HOST` | `0.0.0.0` | Bind address |
| `--port` | `HIGGS_PORT` | `8000` | Bind port |
| `--max-tokens` | `HIGGS_MAX_TOKENS` | `32768` | Max generation tokens |
| `--api-key` | `HIGGS_API_KEY` | *(none)* | Bearer token for auth |
| `--rate-limit` | `HIGGS_RATE_LIMIT` | `0` | Requests/min per client |
| `--timeout` | `HIGGS_TIMEOUT` | `300` | Request timeout in seconds |
| `--mlx-profile` | `HIGGS_MLX_PROFILE` | `auto` | MLX tuning profile: `auto`, `latency`, `balanced`, or `throughput` |
| `--batch` | -- | `false` | Enable continuous batching |
| `--kv-cache` | -- | `off` | KV cache mode: `off` or `turboquant` |
| `--kv-bits` | -- | `3` | Default TurboQuant KV bit width |
| `--kv-key-bits` | -- | `kv-bits - 1` | Override TurboQuant key bit width |
| `--kv-value-bits` | -- | `kv-bits` | Override TurboQuant value bit width |
| `--kv-no-norm-correction` | -- | `false` | Disable TurboQuant norm correction |
| `--kv-adaptive-dense-layers` | -- | `0` | Keep the last N KV cache layers dense |
| `--kv-seed` | -- | `0` | TurboQuant seed |

`auto` resolves to `balanced` for small and medium models, and `throughput` for large and huge models.

### Additional environment toggles

- `HIGGS_ENABLE_THINKING=0|1` forces Qwen thinking on or off.
- `HIGGS_CHUNKED_PREFILL_THRESHOLD` enables chunked prefill above a token threshold.
- `HIGGS_CHUNKED_PREFILL_CHUNK_SIZE` controls chunk size during chunked prefill.
- `HIGGS_MTP=0|1` overrides the tuning profile's speculative decode choice when conditions allow.
- `HIGGS_MTP_DRAFT_N_MAX` controls the maximum MTP draft tokens per speculative cycle. The default is `3` when MTP is enabled, clamped to `1..=8`.
- `HIGGS_CLEAR_CACHE_AFTER_PREFILL` overrides the selected MLX profile behavior for cache clearing.
- `HIGGS_TURBOQUANT_MIN_TOKENS` overrides the TurboQuant activation threshold. The default is `2048`.
- `HIGGS_EXPERIMENTAL_PAGED_KV=1` enables the experimental paged-KV path.
- Qwen thinking budget is currently fixed at `256` tokens and is not currently configurable.

## Gateway Mode

Run `higgs init` to create `~/.config/higgs/config.toml`:

```toml
[server]
host = "0.0.0.0"
port = 8000
# max_tokens = 32768
# timeout = 300.0
# max_body_size = 10485760
# api_key = "sk-..."
# rate_limit = 0

# --- Local defaults ---
[local]
mlx_profile = "auto"
raise_wired_limit = false

# --- Local models ---
[[models]]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
# name = "llama"
# mlx_profile = "throughput"
# batch = false
# kv_cache = "turboquant"
# kv_bits = 3
# kv_key_bits = 2
# kv_value_bits = 3
# kv_norm_correction = true
# kv_adaptive_dense_layers = 0
# kv_seed = 0

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
[[routes]]
pattern = "claude-.*"
provider = "anthropic"

[[routes]]
pattern = "gpt-.*"
provider = "openai"

# [[routes]]
# pattern = "my-alias"
# provider = "openai"
# model = "gpt-4o"

# --- Default route ---
[default]
provider = "higgs"

# --- Auto router ---
# [auto_router]
# enabled = true
# model = "llama"
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

### Profile precedence for local models

Order of precedence:

1. `[[models]].mlx_profile`
2. `--mlx-profile`
3. `HIGGS_MLX_PROFILE`
4. `[local].mlx_profile`
5. built-in default `auto`

## Provider Options

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | string | *(required)* | Base URL of the upstream API |
| `format` | `"openai"` or `"anthropic"` | `"openai"` | API format the provider speaks |
| `api_key` | string | *(none)* | API key to inject into proxied requests |
| `strip_auth` | bool | `false` | Remove the client's `Authorization` header before proxying |
| `stub_count_tokens` | bool | `false` | Return a stub for `/v1/messages/count_tokens` |

## Route Options

| Field | Type | Description |
|---|---|---|
| `pattern` | regex | Match against the `model` field in requests |
| `provider` | string | Provider name to forward to |
| `model` | string | Rewrite the model field before forwarding |
| `name` | string | Human label used by the auto-router |
| `description` | string | Route description used for auto-router classification |

## Routing Behavior

Higgs resolves requests in this order:

1. Auto-router when `model == "auto"` or force mode is enabled
2. Direct local engine lookup by model name
3. Regex pattern routing, first match wins
4. Default route fallback

That means Higgs supports:

- direct local model selection
- pattern routing to local or remote targets
- model alias rewriting before forwarding
- auto-routing with a local classifier model
- a default target when nothing else matches

## Local Model Notes

- `batch=true` is only supported for transformer families with true batched decode support: `llama`, `mistral`, `qwen2`, and `qwen3`.
- `higgs doctor` and server startup now reject unsupported `batch=true` combinations instead of silently degrading.
- `[local].raise_wired_limit` defaults to `false`. Turn it on only when you explicitly want MLX to raise the process wired-memory limit.
- Source builds on macOS require `mlx.metallib`. Higgs restores it from Cargo build output when possible and fails startup if it still cannot be resolved.

## Shell Integration

Export Higgs as the local OpenAI and Anthropic base URL:

```bash
eval "$(higgs shellenv)"
```

Run one command with those variables set:

```bash
higgs exec -- claude
higgs exec -- aider --model openai/gpt-4o
```

`higgs exec` verifies that the server is reachable, sets `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL`, then execs the command.
`higgs shellenv` uses the same strict config loading and reachability checks.

## CLI Overview

| Command | Description |
|---|---|
| `higgs serve` | Start the server in the foreground |
| `higgs start` | Start a background daemon from config or profile |
| `higgs stop` | Stop a running daemon (`--force` escalates to `SIGKILL`) |
| `higgs attach` | Open the daemon metrics dashboard |
| `higgs init` | Create the default config file |
| `higgs shellenv` | Print `export` lines for `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL` |
| `higgs exec -- <cmd>` | Set env vars and exec a command |
| `higgs config get <key>` | Read a config value |
| `higgs config set <key> <value>` | Write a config value |
| `higgs config path` | Print the resolved config file path |
| `higgs doctor` | Validate config, model paths, and providers |

### Global flags

| Flag | Description |
|---|---|
| `--config <FILE>` | Path to config file, conflicts with `--profile` |
| `--profile <NAME>` | Named profile, resolves to `config.<NAME>.toml`, conflicts with `--config` |
| `--verbose` | Enable debug logging |

## Migration Notes

- `higgs start` no longer accepts serve-style flags like `--model`, `--port`, or `--batch`.
- `higgs attach` now fails fast unless the daemon is alive, `/health` passes, and metrics logging is enabled.
- `/metrics` is available and `server.max_body_size` is enforced on API routes.
