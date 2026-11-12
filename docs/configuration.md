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
| `--host` | `HIGGS_HOST` | `127.0.0.1` | Bind address (set `0.0.0.0` to expose on the network; pair with `--api-key`) |
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
| `--load` | -- | `startup` | `startup` loads every model before serving; `on-demand` loads each on its first request |
| `--idle-ttl` | -- | *(none)* | Unload a model after this many idle seconds (`0` never unloads) |
| `--memory-guard` | -- | `off` | Memory ceiling tier: `off`, `safe`, `balanced`, or `aggressive` (see [Model lifecycle](#model-lifecycle)) |
| `--max-loaded-models` | -- | *(none)* | Keep at most N models loaded, evicting the least recently used |

`auto` resolves to `balanced` for small and medium models, and `throughput` for large and huge models.

### Additional environment toggles

- `HIGGS_ENABLE_THINKING=0|1` forces Qwen thinking on or off.
- `HIGGS_CHUNKED_PREFILL_THRESHOLD` enables chunked prefill above a token threshold.
- `HIGGS_CHUNKED_PREFILL_CHUNK_SIZE` controls chunk size during chunked prefill.
- `HIGGS_MTP=0|1` overrides the tuning profile's speculative decode choice when conditions allow.
- `HIGGS_MTP_DRAFT_N_MAX` controls the maximum MTP draft tokens per speculative cycle. The default is `2` for huge checkpoints and `1` otherwise, clamped to `1..=8`.
- `HIGGS_MTP_ADAPTIVE_DRAFT=1` lets the decode loop increase or decrease the MTP draft window based on recent verifier acceptance.
- `HIGGS_MTP_PROMPT_LOOKUP=1` enables a hybrid MTP loop that tries verified prompt-lookup drafts when the prompt/history has a repeated suffix, then keeps the MTP cache synchronized for later MTP-head cycles.
- `HIGGS_CLEAR_CACHE_AFTER_PREFILL` overrides the selected MLX profile behavior for cache clearing.
- `HIGGS_TURBOQUANT_MIN_TOKENS` overrides the TurboQuant activation threshold. The default is `2048`.
- `HIGGS_EXPERIMENTAL_PAGED_KV=1` enables the experimental paged-KV path.
- Qwen thinking budget is currently fixed at `256` tokens and is not currently configurable.

## Gateway Mode

Run `higgs init` to create `~/.config/higgs/config.toml`:

```toml
[server]
# Bind to loopback by default. To expose on the network, set host = "0.0.0.0"
# and set an api_key.
host = "127.0.0.1"
port = 8000
# max_tokens = 32768
# timeout = 300.0
# max_body_size = 10485760
# api_key = "sk-..."
# rate_limit = 0
# CORS origin allow-list for browser clients. Unset = no CORS headers;
# ["*"] allows any origin.
# cors_origins = ["https://app.example.com"]

# --- Local defaults ---
[local]
mlx_profile = "auto"
raise_wired_limit = false
# thinking_budget = 256 # reasoning tokens before </think> is forced; minimum 1
# load = "on_demand"
# idle_ttl_seconds = 900
# max_loaded_models = 2
# memory_guard = "balanced"
# memory_limit_gb = 48

# --- Local models ---
[[models]]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
# name = "llama"
# load = "startup"
# idle_ttl_seconds = 600
# pinned = true
# mlx_profile = "throughput"
# batch = false
# prefill_yield_tokens = 512 # 0 or omitted keeps synchronous prefill
# thinking_budget = 256 # overrides [local].thinking_budget for this model
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

## Model lifecycle

By default every `[[models]]` entry loads at startup and stays resident until the server stops. Set `load = "on_demand"` to load a model on the first request that routes to it, and add an idle timeout, a model-count cap, or a memory ceiling to free memory for models that are not in use.

### Fields

`[local]` sets defaults; the same-named `[[models]]` field overrides it for one model.

| Field | Where | Type | Default | Description |
|---|---|---|---|---|
| `load` | `[local]`, `[[models]]` | `"startup"` or `"on_demand"` | `"startup"` | When the model is loaded |
| `idle_ttl_seconds` | `[local]`, `[[models]]` | integer | *(none)* | Unload after this many seconds with no in-flight request. `0` or unset never unloads on idle; a model-level `0` opts one model out of a `[local]` default |
| `pinned` | `[[models]]` | bool | `false` | Never evicted or unloaded on idle. An explicit `/unload` still works |
| `max_loaded_models` | `[local]` | integer >= 1 | *(none)* | Most models allowed in memory at once |
| `memory_limit_gb` | `[local]` | number > 0 | *(none)* | Ceiling on MLX active memory, in GiB. Wins over `memory_guard` |
| `memory_guard` | `[local]` | `"off"`, `"safe"`, `"balanced"`, `"aggressive"` | `"off"` | Derive the ceiling from total unified memory by reserving 20%, 8%, or 2% for the rest of the system |

`higgs doctor` rejects out-of-range values and warns when pinned models alone exceed the ceiling (by weight size on disk), when more models load at startup than `max_loaded_models` allows, and when `on_demand` is used with a single model, which only delays the first request.

### Rules

- **Lazy loading.** A request for an unloaded model loads it first. Concurrent first requests share one load. If it fails, every waiting request gets a `503` and the next request tries again; a failed load is never cached.
- **Startup models.** Models that resolve to `startup` load before the server binds, as before. The `auto_router` classifier always loads at startup and is never evicted or expired, whatever its `load`.
- **Eviction.** Before a load that would exceed `max_loaded_models` or the memory ceiling, Higgs unloads the least recently used model that is idle and not pinned, repeating until the model fits. Models with requests in flight and pinned models are never evicted.
- **Memory accounting.** The size of a model before loading is estimated from its weight files on disk. After loading, the growth in MLX active memory is recorded as the measured size, which includes any other model generating at that moment, so treat it as approximate. The ceiling is compared with MLX active memory (weights plus KV caches), not process RSS or MLX's reusable buffer cache.
- **Refusal.** If the model still does not fit once nothing more can be evicted, or the model alone exceeds the ceiling, the load is refused with `503` and `code = "insufficient_memory"`. The message names the model, its estimated size, the ceiling, and each loaded model with the reason it could not be evicted (pinned or in use). A model that turns out larger than estimated is checked again after loading and unloaded if it breaks the ceiling.
- **Idle expiry.** A model with `idle_ttl_seconds` is unloaded once it has had no in-flight request for that long. A request that is still streaming keeps its model alive; the idle clock starts when the last request finishes.
- **Unloading frees memory.** The engine is shut down on the thread that owns its MLX state, Higgs waits for that to finish, and then clears the MLX buffer cache, so the memory is back by the time the unload returns.
- **Reloading.** An unloaded model, however it was unloaded, loads again on its next request.

### Endpoints

All use the same authentication as the rest of the API. None of them count as traffic in `/metrics`, and none of the read-only ones ever trigger a load. A model id containing a slash (`org/name`) can be written as is or percent-encoded (`org%2Fname`).

| Endpoint | Description |
|---|---|
| `GET /v1/models` | Every configured model, loaded or not, each with `"loaded": true\|false` |
| `GET /v1/models/status` | Lifecycle state of every configured local model (fields below) |
| `POST /v1/models/{id}/load` | Load the model if needed and return its status. `503` if the load fails or is refused, `404` for an unknown model |
| `POST /v1/models/{id}/unload` | Unload the model and return its status. `409` while requests are in flight |
| `POST /v1/models/{id}/unload?force=true` | Stop handing out the model, wait for in-flight requests to finish (streams are not cut), then unload |
| `GET /v1/system` | Process and memory facts plus the same per-model fields for every configured model |

Unloading a model that is already unloaded succeeds. Unloading a model that is loading or unloading returns `409`.

`GET /v1/models/status` returns `{"object": "list", "data": [...]}`; each entry has:

| Field | Description |
|---|---|
| `name` | Model name |
| `state` | `unloaded`, `loading`, `loaded`, `unloading`, or `failed` |
| `last_error` | Why the last load failed (only in state `failed`), otherwise `null` |
| `pinned` | Whether the model is exempt from eviction and idle expiry |
| `load` | Effective `startup` or `on_demand` |
| `in_flight` | Requests currently using the model |
| `last_used_at` | When a request last started or finished with the model, or `null` |
| `load_duration_ms` | How long the current load took, while loaded |
| `estimated_bytes` | Size predicted from weight files before loading |
| `measured_bytes` | Growth in MLX active memory while loading, while loaded |
| `engine` | `simple` or `batch` |
| `idle_ttl_seconds` | Effective idle timeout, or `null` |

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

- `batch=true` is only supported for transformer families with true batched decode support: `llama`, `mistral`, `qwen2`, `qwen3`, `qwen3_moe`, `qwen3_5`, `qwen3_5_moe`, `phi3`, and `starcoder2`.
- `higgs doctor` and server startup now reject unsupported `batch=true` combinations instead of silently degrading.
- `[local].raise_wired_limit` defaults to `false`. Turn it on only when you explicitly want MLX to raise the process wired-memory limit.
- With several resident models, consider `max_loaded_models`, `memory_guard`, or `load = "on_demand"`; see [Model lifecycle](#model-lifecycle).
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

## API request options

These apply to locally served models. Requests routed to a remote provider are
forwarded unchanged.

### Usage fields

Responses from `/v1/chat/completions` and `/v1/completions` (and the terminal chunk
requested with `stream_options.include_usage`) carry a `usage` object. Beyond the OpenAI
token counts it includes the fields below. A field is omitted when it was not measured for
the request; the server never fills in zeros as placeholders.

| Field | Meaning |
|---|---|
| `prompt_tokens_details.cached_tokens` | Prompt tokens served from the prefix cache (memory or disk) instead of being prefilled. Reported as `0` when the engine measured a miss. |
| `completion_tokens_details.reasoning_tokens` | Tokens in the reasoning block, chat only, present when the response had reasoning content. Counted by re-encoding the extracted reasoning text with the model's tokenizer, so it can differ by a token or two from the tokens actually sampled. |
| `time_to_first_token` | Seconds from receiving the request to the first completion token. Includes queueing, template rendering and prefill. |
| `prompt_eval_duration` | Seconds from the engine starting on the prompt to the first completion token. Excludes queueing. |
| `generation_duration` | Seconds from the first completion token to the last. Omitted for single-token completions. |
| `total_time` | Seconds from receiving the request to the last completion token. |
| `prompt_tokens_per_second` | Prompt tokens that were actually prefilled (`prompt_tokens - cached_tokens`) divided by `prompt_eval_duration`. Omitted when the whole prompt was cached. |
| `generation_tokens_per_second` | `(completion_tokens - 1) / generation_duration`; the first token is produced by prefill and is not part of the decode rate. |

For streaming requests the engine events are timestamped as the server receives them, so
the figures include the channel hop between engine and HTTP handler. The batch engine
reports a prefix-cache hit only for requests prefilled individually; requests prefilled
together in a group are reported with `cached_tokens: 0`.

`/v1/messages` adds `usage.cache_read_input_tokens` when the cache hit is known.
Unlike Anthropic's API, `input_tokens` still includes those cached tokens.

### Guided output

`/v1/chat/completions` can constrain generation to a grammar. Exactly one of these may be
set per request; setting more than one returns `400` naming the conflicting fields.

| Field | Constraint |
|---|---|
| `response_format: {"type": "json_object"}` | Any JSON object |
| `response_format: {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}` | The JSON schema (a bare schema is also accepted) |
| `response_format: {"type": "regex", "regex": "..."}` | Output must match the regular expression |
| `guided_regex: "..."` | Same as the regex response format |
| `guided_json: {...}` | Same as `json_schema`, with the schema given directly |

Regexes use the Rust `regex` syntax and are matched against the whole output; look-around
and backreferences are not supported. An invalid pattern returns `400` with the compile
error. `response_format: {"type": "text"}` means no constraint. The constraint is applied
by masking token logits, so it requires the model's tokenizer to be representable by the
grammar compiler; patterns whose automaton is very large take time and memory to compile.

### Tool choice and seed

`tool_choice` on `/v1/chat/completions` accepts `"auto"` (the default, also used when the
field is absent or `null`) and `"none"`. With `"none"` the `tools` array is dropped before
the prompt is rendered and the output is never parsed for tool calls, streaming or not.
`"required"`, a named function (`{"type": "function", ...}`) and any other value return `400`;
they are not silently treated as `"auto"`. Requests routed to a remote provider are forwarded
untouched, so the provider interprets `tool_choice` itself.

`seed` must be an integer when present but has no effect: MLX samples from a process-wide
random key, and sampling happens in several decode paths (single, pipelined, batched), so a
per-request seed cannot be applied without affecting concurrent requests. Treat sampling as
non-reproducible for now; use `temperature: 0` for repeatable output.

### Thinking budget

Models that emit a `<think>` block get a cap on reasoning tokens: once the budget is spent
the engine forces `</think>` so the model moves on to its answer. The budget resolves in this
order, first match wins:

1. Per request: `reasoning.max_tokens` on `/v1/chat/completions`, or
   `thinking.budget_tokens` on `/v1/messages`.
2. `thinking_budget` in the model's `[[models]]` entry.
3. `thinking_budget` under `[local]`.
4. The built-in default, 256.

A budget must be at least 1, in config (`higgs doctor` fails on `0`) and in requests (`400`).
To turn thinking off use reasoning effort `"none"`; a zero budget is not a way to do that.
`higgs doctor` warns when a budget is not below `server.max_tokens`, since reasoning could
then consume the whole response. The budget only applies to the simple engine; models with
`batch = true` do not reason, and `thinking_budget` has no effect on them. On `/v1/messages`
only `budget_tokens` is read from `thinking`; `type` is ignored.

### Tokenizer endpoints

`POST /v1/tokenize` and `POST /v1/detokenize` (also served as `/tokenize` and
`/detokenize`) expose the loaded model's tokenizer. Requests for models routed to a remote
provider fail with `400`.

```json
{"model": "qwen", "text": "Hello"}
{"model": "qwen", "messages": [{"role": "user", "content": "Hello"}]}
```

return `{"tokens": [ids], "count": n}`. `messages` are rendered through the chat template
with the generation prompt, exactly as `/v1/chat/completions` would feed them to the
model; `tools` and `reasoning` are honored so the count matches. `text` is encoded without
special tokens unless `"add_special_tokens": true`, matching `/v1/completions`.
`text` and `messages` are mutually exclusive.

```json
{"model": "qwen", "tokens": [9707]}
```

to `/v1/detokenize` returns `{"text": "Hello"}`. Special tokens are kept unless
`"skip_special_tokens": true`. Token ids missing from the vocabulary are rejected with `400`.

## Migration Notes

- `higgs start` no longer accepts serve-style flags like `--model`, `--port`, or `--batch`.
- `higgs attach` now fails fast unless the daemon is alive, `/health` passes, and metrics logging is enabled.
- `/metrics` is available and `server.max_body_size` is enforced on API routes.

## Local model options

For a local model, set `kv_disk_dir` in its `[[models]]` entry to retain block-aligned
prefix KV state across restarts. `kv_disk_space_mb` controls its on-disk LRU budget
(default `4096`, minimum `64`). The disk tier is model, quantization, configuration,
tokenizer, and chat-template bound; incompatible entries are ignored.

For DeepSeek-V2 models, set `mla_latent_cache = true` in its `[[models]]` entry to store
the MLA KV cache as compressed latent rows instead of dense per-head tensors (default
off). It cannot be combined with `kv_cache = "turboquant"`, and is a no-op for
non-DeepSeek-V2 architectures. The `HIGGS_MLA_LATENT_CACHE` env var, when set to a
recognized value (`1`/`0`, `true`/`false`, `on`/`off`, `yes`/`no`), overrides the config
value either way.

## Migration notes

- Replace old `higgs start --model ...` usage with `higgs serve --model ...`.
- `higgs attach` fails fast when the daemon is down or metrics logging is disabled; it is a
  metrics dashboard, not a best-effort log viewer.
- Exact local model names now beat regex routes; rename a local model or route if you relied
  on a regex overriding a local model of the same name.
- The server binds `127.0.0.1` by default. Set `server.host = "0.0.0.0"` and an `api_key` to
  expose it on the network.
- CORS headers are only sent when `server.cors_origins` is set (`["*"]` restores the old
  permissive behavior).
- Source builds need `mlx.metallib` next to the executable; `cargo build` restores it from
  Cargo output when possible.
