# Higgs

[![CI](https://github.com/panbanda/higgs/actions/workflows/ci.yml/badge.svg)](https://github.com/panbanda/higgs/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/panbanda/higgs)](https://github.com/panbanda/higgs/releases)
[![Crates.io](https://img.shields.io/crates/v/higgs)](https://crates.io/crates/higgs)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)

Run open-weight MLX models locally on Apple Silicon, route requests across local and remote providers, and expose everything through one endpoint.

Higgs is a single static Rust binary that serves local models, proxies to providers like OpenAI, Anthropic, and Ollama, and translates between OpenAI and Anthropic-style APIs so your existing tools and apps do not need a new integration.

**Why care**
- Run open-weight models locally on your Mac, including supported Qwen, Llama, Mistral, Gemma (2, 3, 4), Phi, DeepSeek, and vision-capable MLX families.
- Send requests to local models or remote providers through one endpoint.
- Plug tools into Higgs with `higgs shellenv` or `higgs exec` instead of reconfiguring each client separately.

**Use Higgs if**
- you want local open-weight model serving on Apple Silicon
- you switch between local and hosted models
- you want one API surface for apps, agents, and terminal tools

## Breaking-Change Highlights

- `higgs serve` remains the ad hoc foreground entrypoint for `--model`, `--port`, `--batch`, and related flags.
- `higgs start` is now config/profile-only. Use `higgs init`, then `higgs start`.
- `higgs attach` is a daemon metrics dashboard. It now requires a live daemon, a passing `/health` probe, and metrics logging.
- Exact local model names now beat regex routes.
- `/metrics` is a real endpoint, and `server.max_body_size` is enforced on API requests.
- `higgs shellenv` and `higgs exec` now fail fast on bad config or an unreachable server.
- The server now binds `127.0.0.1` by default (was `0.0.0.0`). Set `server.host = "0.0.0.0"` (and an `api_key`) to expose it on the network.
- CORS headers are no longer sent unless `server.cors_origins` is set (`["*"]` restores the old permissive behavior).

## Quick Links

- [Quick Start](#quick-start)
- [Configuration](docs/configuration.md)
- [Supported Models](docs/models.md)
- [Benchmarking](docs/benchmarking.md)
- [Contributing](CONTRIBUTING.md)

## Quick Start

Install:

```bash
brew install panbanda/brews/higgs
```

The Homebrew release currently targets Apple Silicon (`aarch64-apple-darwin`).

Or build from source (Rust 1.88.0+, Xcode CLI Tools):

```bash
cargo build --release
```

Run a local open-weight model:

```bash
higgs serve --model mlx-community/Qwen3.6-35B-A3B-4bit
```

Send a request to the local endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
    "messages": [{"role": "user", "content": "Write one sentence about Cape Town."}]
  }'
```

Point an existing tool at Higgs:

```bash
higgs exec -- claude
```

Requests can also target routed remote models through the same endpoint. For example, an OpenAI-format request can be translated and proxied to Anthropic based on your route configuration:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## What Higgs Does

### Run open-weight models locally on Apple Silicon

- Serve MLX models from Hugging Face IDs or local paths.
- Support current model families including Qwen 3.6, Qwen 3.x, Llama, Mistral, Gemma 2, Phi-3, Starcoder2, DeepSeek-V2, and LLaVA-Qwen2.
- Expose local serving through OpenAI and Anthropic-compatible endpoints.

### Use one endpoint for local and remote models

- Serve local MLX models and proxy remote providers from the same server.
- Keep client integrations stable while you switch between local and hosted backends.
- Route unmatched requests to a configured default target.

### Route and translate requests across providers

- Resolve requests by direct local model selection, regex pattern routing, model alias rewriting, or the optional auto-router.
- Translate OpenAI-format requests to Anthropic providers and Anthropic-format requests back to OpenAI-style clients, including streaming where supported.
- Proxy to OpenAI, Anthropic, Ollama, and other OpenAI-compatible APIs.

### Plug Higgs into existing tools

- Use `higgs shellenv` to export `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL`.
- Use `higgs exec -- <cmd>` to launch a command with those variables set.
- Point tools such as Claude Code, Aider, and other OpenAI/Anthropic-compatible clients at a single local endpoint.

### Monitor usage with daemon mode and dashboard

- Run in the foreground with `higgs serve` or as a background daemon with config-driven `higgs start`.
- Open the daemon metrics dashboard with `higgs attach` for routing, latency, throughput, and error visibility.
- Validate config and model/provider setup with `higgs doctor`.

## Apple Silicon Notes

- Release artifacts bundle `mlx.metallib`.
- Source builds also require `mlx.metallib` next to the executable. Higgs now restores it automatically from Cargo build output when possible, then fails loudly if it still cannot be found.
- `[local].raise_wired_limit` defaults to `false`. Enable it only when you explicitly want MLX to raise the process wired-memory limit.
- `[local].allow_runtime_model_load` defaults to `false`. Enable it to load/unload models at runtime via `POST`/`DELETE /v1/models`; protect it with `server.api_key`.
- `batch=true` is only supported for transformer families with true batched decode support.

## Performance

Benchmarks below were run on M4 Max 128GB. Methodology, harness details, and benchmark-driven defaults are documented in [docs/benchmarking.md](docs/benchmarking.md).

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

## API and CLI Overview

**API endpoints**

- OpenAI: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`
  - Streaming requests may set `"return_progress": true` to receive
    llama.cpp-compatible `prompt_progress` chunks (`{total, cache, processed,
    time_ms}`) during chunked prefill.
- Anthropic: `/v1/messages`, `/v1/messages/count_tokens`
- OpenAI and Anthropic requests accept an optional `"speculation"` field
  (`auto` | `dflash` | `mtp` | `none`) to pick the speculative-decode method per
  request. `auto` (default) uses the DFlash drafter when one is loaded — including
  while streaming — otherwise the MTP head; `none` disables speculation.
- The dSpark (DFlash) verifier defaults to the block (`BatchedTape`) schedule for
  the 2-bit Bonsai target (validated end-to-end) and to canonical S=1 for the
  1-bit Bonsai target. Override per-process with `HIGGS_DFLASH_VERIFY_MODE`
  (`block` | `canonical`).
- Compressive prefill (PFlash, experimental): an optional `prefill_drafter`
  (e.g. `mlx-community/Qwen3-0.6B-4bit`) scores the prompt so the target
  prefills only a fraction of it (`prefill_compression`, `prefill_keep_ratio`).
  Off by default; see `docs/configuration.md` and
  `.planning/DESIGN-pflash-higgs.md`.
- Runtime model management (opt-in): `POST /v1/models`, `DELETE /v1/models/{name}`
- Metrics: `/metrics`
- Health: `/health`

### Reasoning / thinking budget

For thinking-capable models, `/v1/chat/completions` accepts two extra body fields
(additive, OpenAI clients ignore them):

- **`reasoning_budget`** (int) — max `<think>` tokens before `</think>` is
  force-closed. Omitted → engine default (256). Sent by clients like nanobot's
  `/thinking N`.
- **`chat_template_kwargs.enable_thinking`** (bool) — per-request override of the
  model's reasoning mode; wins over `reasoning.effort`. `false` forces reasoning
  off; `true` enables it on a thinking-capable model. A top-level
  **`enable_thinking`** (bool) is accepted as an alias for clients that send it
  there; `chat_template_kwargs.enable_thinking` takes precedence when both are set.

```bash
curl http://localhost:8000/v1/chat/completions -d '{
  "model": "active", "messages": [{"role":"user","content":"hi"}],
  "reasoning_budget": 1024,
  "chat_template_kwargs": {"enable_thinking": true}
}'
```

### Runtime model management

By default the set of loaded models is fixed at startup. To add or remove models
while the server runs, set `allow_runtime_model_load = true` under `[local]`
(protect it with `server.api_key` -- the load endpoint can read local model
directories and trigger downloads). Changes are in-memory only and do not persist
to the config file.

```bash
# Load a model (body mirrors a [[models]] entry; path required)
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer $HIGGS_API_KEY" \
  -d '{"path": "mlx-community/Llama-3.2-1B-Instruct-4bit", "name": "llama"}'

# Unload it and free its GPU memory (204 once freed, 202 if a request is still in flight)
curl -X DELETE http://localhost:8000/v1/models/llama \
  -H "Authorization: Bearer $HIGGS_API_KEY"
```

An in-flight request keeps the model resident until it completes; `DELETE` removes
it from routing immediately and frees memory once the last request drains. A model
bound to the auto-router cannot be unloaded.

**Capability discovery.** `GET /v1/models` adds two OpenAI-compatible *additive*
fields (standard clients ignore unknown keys) so a client can check before acting:
the top-level **`runtime_model_load`** (bool) mirrors `allow_runtime_model_load`
-- if `false`, switch/load/unload will `403`; and per-model **`vision`** (bool)
reports whether the model accepts image input (true only for VLMs).

```jsonc
{
  "object": "list",
  "runtime_model_load": true,
  "data": [
    { "id": "qwen36-35b", "object": "model", "created": 0, "owned_by": "local", "vision": false }
  ]
}
```

**Core commands**

- `higgs serve`: start in the foreground
- `higgs start`: start a background daemon from config or profile
- `higgs stop`: stop a running daemon, or use `higgs stop --force`
- `higgs attach`: open the daemon metrics dashboard
- `higgs init`: create `~/.config/higgs/config.toml`
- `higgs doctor`: validate config, model paths, and providers
- `higgs shellenv`: print `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL` after verifying the server is reachable
- `higgs exec -- <cmd>`: run a tool with those variables set after the same reachability check

## Migration Notes

- Replace old `higgs start --model ...` usage with `higgs serve --model ...`.
- If you previously treated `higgs attach` as a best-effort log viewer, expect it to fail fast when the daemon is down or metrics logging is disabled.
- If you relied on regex routes to override a local model with the same exact name, rename the local model or route; local exact matches now win.
- If you run local source builds, make sure `cargo build` completes before first serve so Higgs can restore `mlx.metallib` from Cargo output when needed.

## Release Validation

- Run `scripts/release_smoke_cached_models.sh` to validate the cached MLX models already present on the machine without downloading anything.
- Set `HIGGS_SMOKE_INCLUDE_OPTIONAL_MODELS=1` to include optional large/private cached models like `mlx-community/Qwen3.6-35B-A3B-4bit`.
- The harness covers single-model serve, streaming and non-streaming requests, multi-model startup, routing precedence, daemon start/attach/stop, and the batch-support guardrails.
- The current smoke matrix exercised these cached models on this machine: `mlx-community/Llama-3.2-1B-Instruct-4bit`, `mlx-community/Qwen2.5-3B-Instruct-4bit`, `mlx-community/Qwen3-1.7B-4bit`, `mlx-community/Qwen3-Coder-Next-4bit`, and `mlx-community/Qwen3.6-35B-A3B-4bit`.

For full configuration reference, routing options, supported model families, and benchmark details, see:

- [docs/configuration.md](docs/configuration.md)
- [docs/models.md](docs/models.md)
- [docs/benchmarking.md](docs/benchmarking.md)

## Development

```bash
cargo test -- --test-threads=1
cargo clippy
cargo fmt --check
```

Contributor workflow, project structure, and doc update expectations live in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
