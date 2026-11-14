# Configuration

This document collects the full CLI, environment, and config-file reference for Higgs.

## Modes

- **Simple mode**: pass one or more `--model` flags to serve local MLX models without a config file.
- **Gateway mode**: run `higgs init` and edit `~/.config/higgs/config.toml` to combine local models, providers, routes, metrics, and daemon behavior.

## Simple Mode

### CLI flags and environment variables

| CLI Flag | Env Variable | Default | Description |
|---|---|---|---|
| `--model` | -- | *(required)* | Model path or HF ID (repeatable) |
| `--host` | `HIGGS_HOST` | `127.0.0.1` | Bind address (set `0.0.0.0` to expose on the network; pair with `--api-key`) |
| `--port` | `HIGGS_PORT` | `8000` | Bind port |
| `--max-tokens` | `HIGGS_MAX_TOKENS` | `32768` | Max generation tokens |
| `--api-key` | `HIGGS_API_KEY` | *(none)* | Bearer token for auth |
| `--rate-limit` | `HIGGS_RATE_LIMIT` | `0` | Requests/min per client |
| `--timeout` | `HIGGS_TIMEOUT` | `300` | Request timeout in seconds |
| `--mlx-profile` | `HIGGS_MLX_PROFILE` | `auto` | MLX tuning profile: `auto`, `latency`, `balanced`, or `throughput` |
| `--batch` | -- | -- | Deprecated; has no effect |
| `--no-batch` | -- | *(off)* | Limit every `--model` to one request at a time |
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
| `--mcp-config` | -- | *(none)* | JSON file of MCP servers in the common `{"mcpServers": {...}}` format |

`auto` resolves to `balanced` for small and medium models, and `throughput` for large and huge models.

The flags apply to a config file too: `higgs serve --config ...` accepts every flag above. The `--kv-*`,
`--batch` and `--no-batch` flags describe the models given with `--model`; the server and lifecycle flags
override the matching config keys.

## Gateway Mode

Run `higgs init` to create `~/.config/higgs/config.toml`. Every section of the file is optional and
documented in the generated template. A minimal file:

```toml
[server]
host = "127.0.0.1"
port = 8000
api_key = "${HIGGS_API_KEY}"        # environment reference; a missing variable is a startup error
request_timeout = "5m"
max_body_size = "10MB"

[memory]
guard = "balanced"                  # off | safe | balanced | aggressive
max_loaded_models = 3

[defaults]                          # applies to every model; a [[models]] entry overrides it
load = "on_demand"
idle_ttl = "15m"
profile = "auto"
thinking_budget = 256

[[models]]
path = "mlx-community/Qwen3.6-35B-A3B-4bit"
name = "qwen"
aliases = ["default"]               # extra request names that resolve to this model
pinned = true

[models.kv_cache]
mode = "turboquant"
bits = 3

[providers.anthropic]
url = "https://api.anthropic.com"
format = "anthropic"
api_key = "${ANTHROPIC_API_KEY}"

[[routes]]
match = "claude-.*"
provider = "anthropic"

[routing]
default = "higgs"

[observability]
retention = "60m"

[observability.metrics_log]
path = "~/.config/higgs/logs/metrics.jsonl"
max_size = "50MB"
max_files = 5
```

### Conventions

- **Durations** are strings: `"500ms"`, `"90s"`, `"5m"`, `"1h30m"`. Units are `ns`, `us`, `ms`, `s`, `m`, `h`, `d`.
  A bare `0` is allowed where zero is meaningful (`idle_ttl = "0"` never unloads); any other bare number
  is an error that asks for a unit.
- **Sizes** are strings: `"512MB"`, `"4GB"`, `"64KB"`. The multiples are 1024-based (`MB` is a mebibyte; `MiB` is
  accepted as a synonym). A bare integer is a number of bytes.
- **Environment references.** String values that can hold a secret or a path (`server.host`, `server.api_key`,
  `server.cors_origins`, `models[].path`, `prefix_cache.disk_dir`, `providers.*.url`, `providers.*.api_key`,
  `observability.metrics_log.path`, `web.searxng_url`, `include` entries, and the `env` and `headers` of MCP
  servers) may contain `${VAR}` or `${VAR:-default}`. A variable that is not set and has no default is an error
  that names the key and its line. Write `$${` for a literal `${`. Resolved values are never logged;
  `higgs doctor` and `higgs config explain` print references as written and mask literal secrets.
- **`~/`** at the start of a path key expands to the home directory.
- **Strict keys.** A key Higgs does not know is an error that names the key, its section and the closest valid key,
  with the file and line: `config.toml:12: unknown key `idel_ttl` in [defaults]; did you mean `idle_ttl`?`.
- **Deprecated keys** from earlier versions keep working. Each one prints a single warning at startup (and in
  `higgs doctor`) naming the key that replaces it. Setting an old and a new spelling of the same thing in one file
  is an error that names both. See [Migrating an existing file](#migrating-an-existing-file).

### Includes

```toml
include = ["providers.toml", "models.toml"]
```

Each path is relative to the file that lists it (and may use `${VAR}` and `~/`). Included files are merged first,
in order, and the including file on top: tables merge key by key, later files override scalars and plain arrays, and
`[[models]]` and `[[routes]]` entries are appended. Included files may include others. A missing file or a cycle is an
error.

### Inheritance and precedence

`[defaults]` accepts every per-model key (`load`, `idle_ttl`, `batch`, `profile`, `thinking_budget`, `pinned`,
`kv_cache.*`, `prefix_cache.*`, `prefill.*`). A `[[models]]` entry overrides it key by key, including inside the
`kv_cache`, `prefix_cache` and `prefill` tables. For a per-model key the order is:

1. `higgs serve` flag (for `--model` entries: `--batch`, `--no-batch`, `--kv-*`)
2. the model entry
3. `--load`, `--idle-ttl`, `--mlx-profile`
4. `HIGGS_MLX_PROFILE` (for `profile`)
5. `[defaults]`
6. the built-in default

Server keys follow flag > environment variable > file > built-in. The `[experimental]` keys follow environment
variable > file > built-in (they have no flag and no per-model level). Any key can also be set with a
`HIGGS_<SECTION>__<KEY>` variable such as `HIGGS_SERVER__PORT=9000`, which overrides the file. Use
`higgs config explain [model]` to see the winning source of every value.

### Commands

| Command | Description |
|---|---|
| `higgs config check` | Parse and validate the file: strict keys, units, references, includes, names. No model or network probes. Exits non-zero on errors; suitable for CI. |
| `higgs config explain [model]` | Print every resolved setting for the server, or for one model (name, alias or path), and where each value comes from: built-in default, `[defaults]`, the model entry, an environment variable, or a flag. Accepts the `serve` flags. |
| `higgs config migrate [--write]` | Print the file rewritten in the current schema. `--write` replaces the file and keeps the original as `<file>.bak`. Comments and key order are preserved where the structure allows; `${VAR}` references are never expanded; an unknown key is an error, not dropped. |
| `higgs config get <key>` / `set <key> <value>` / `path` | Read or write one value (current or deprecated key names). |
| `higgs doctor` | Everything `check` does plus model, provider and port probes, deprecation and environment-override reports. |

## Configuration reference

### `[server]`

| Key | Type | Default | Description |
|---|---|---|---|
| `host` | string | `"127.0.0.1"` | Bind address. A non-loopback host without `api_key` is flagged by `higgs doctor` |
| `port` | integer | `8000` | Bind port |
| `api_key` | string | *(none)* | Bearer token required on API routes. Use `"${HIGGS_API_KEY}"` |
| `request_timeout` | duration | `"5m"` | Per-request timeout; must be longer than zero |
| `max_body_size` | size | `"10MB"` | Largest request body |
| `max_tokens` | integer | `32768` | Default maximum generation tokens |
| `rate_limit` | integer | `0` | Requests per minute per client; `0` turns it off |
| `cors_origins` | array of strings | *(unset)* | CORS allow-list; unset sends no CORS headers, `["*"]` allows any origin |

### `[memory]`

| Key | Type | Default | Description |
|---|---|---|---|
| `guard` | `"off"`, `"safe"`, `"balanced"`, `"aggressive"` | `"off"` | Derive the ceiling from total unified memory by reserving 20%, 8%, or 2% for the system |
| `limit` | size | *(none)* | Explicit ceiling on MLX active memory; wins over `guard`; must be above zero |
| `max_loaded_models` | integer >= 1 | *(none)* | Most models allowed in memory at once |
| `raise_wired_limit` | bool | `false` | Raise MLX's wired memory limit to the device-recommended maximum |

### `[defaults]` and `[[models]]`

Keys valid in both. A `[[models]]` entry additionally needs `path` and may set `name` and `aliases`.

| Key | Type | Default | Description |
|---|---|---|---|
| `path` | string | *(required, models only)* | Filesystem path or Hugging Face reference |
| `name` | string | *(derived)* | Name exposed to API clients (models only) |
| `aliases` | array of strings | `[]` | Extra request names that resolve to this model; must not collide with another model's name or alias (models only) |
| `load` | `"startup"`, `"on_demand"` | `"startup"` | When the model is loaded |
| `idle_ttl` | duration | *(none)* | Unload after this long with no in-flight request. `"0"` or unset never unloads; a model-level `"0"` opts one model out of a `[defaults]` value |
| `pinned` | bool | `false` | Never evicted or unloaded on idle. An explicit `/unload` still works |
| `batch` | `true`, `false`, `"auto"` | `"auto"` | `false` limits the model to one request at a time; `true` and `"auto"` mean no limit. See [Concurrent requests](#concurrent-requests) |
| `profile` | `"auto"`, `"latency"`, `"balanced"`, `"throughput"` | `"auto"` | MLX tuning preset |
| `thinking_budget` | integer >= 1 | `256` | Reasoning tokens the model may spend before `</think>` is forced |

`[models.kv_cache]` (and `[defaults.kv_cache]`):

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` | `"off"`, `"turboquant"` | `"off"` | KV-cache storage mode |
| `bits` | integer | `3` | Default TurboQuant bit width |
| `key_bits` | integer | `bits - 1` | Key-code bit width |
| `value_bits` | integer | `bits` | Value-code bit width |
| `norm_correction` | bool | `true` | Norm correction after TurboQuant reconstruction |
| `adaptive_dense_layers` | integer | `0` | Final layers that stay dense when TurboQuant is on |
| `seed` | integer | `0` | Seed for the TurboQuant rotation and QJL matrices |
| `mla_latent` | bool | `false` | Store DeepSeek-V2 MLA caches as compressed latent rows; cannot be combined with `mode = "turboquant"`; a no-op for other architectures. The `HIGGS_MLA_LATENT_CACHE` variable overrides it either way |

`[models.prefix_cache]`:

| Key | Type | Default | Description |
|---|---|---|---|
| `disk_dir` | path | *(none)* | Directory for durable, disk-backed prefix KV entries, retained across restarts |
| `disk_budget` | size | `"4GB"` | LRU budget of the disk tier; at least `64MB` |
| `max_entries` | integer 0-1024 | `8` | Prompts the in-memory prefix cache keeps, least recently used first out; `0` turns the cache off |
| `checkpoint_interval_tokens` | integer 1-1048576 | `512` | Hybrid models: tokens between recurrent-state checkpoints. The engine rounds it up to a multiple of the prefill chunk (512 tokens for the large hybrids) and of the 32-token cache block, so a smaller value than the chunk behaves like the chunk |
| `max_checkpoints_per_entry` | integer 0-64 | `8` | Hybrid models: checkpoints one cached prompt keeps (the newest half of the allowance goes to the latest boundaries, the rest is spread over earlier ones); `0` turns checkpoints off |
| `checkpoint_budget` | size | `"2GB"` | Hybrid models: checkpoint state kept across all entries, least recently used entries losing theirs first; `0` turns checkpoints off. The `HIGGS_PREFIX_CHECKPOINT_BUDGET_MB` variable (whole mebibytes) replaces it for every model |

A hybrid model (one with recurrent linear-attention layers, such as Qwen3-Next) can only resume a prompt
that shares part of a cached prompt from a recurrent-state checkpoint, because that state cannot be
cut back to an arbitrary token. A checkpoint of a 27B hybrid is about 150 MB, so a smaller interval
buys finer reuse at the price of memory. `checkpoint_interval_tokens`, `max_checkpoints_per_entry`
and `checkpoint_budget` have no effect on other architectures. `higgs config explain <model>` shows
each value and where it came from.

`[models.prefill]`:

| Key | Type | Default | Description |
|---|---|---|---|
| `yield_tokens` | integer | *(none)* | Maximum prompt tokens to prefill per batch-loop iteration; `0` or unset keeps synchronous prefill. Values from 1 to 127 are rejected by `higgs doctor` |

### `[providers.<name>]`

| Key | Type | Default | Description |
|---|---|---|---|
| `url` | string | *(required)* | Base URL of the upstream API |
| `format` | `"openai"`, `"anthropic"` | `"openai"` | API format the provider speaks |
| `api_key` | string | *(none)* | API key to inject into proxied requests. Use `"${NAME_API_KEY}"` |
| `strip_auth` | bool | `false` | Remove the client's `Authorization` header before proxying |
| `stub_count_tokens` | bool | `false` | Return a stub for `/v1/messages/count_tokens` |

### `[[routes]]`

| Key | Type | Description |
|---|---|---|
| `match` | regex | Match against the `model` field in requests |
| `provider` | string | Provider name to forward to (`higgs` for the local models) |
| `model` | string | Rewrite the model field before forwarding |
| `name` | string | Human label used by the auto-router |
| `description` | string | Route description used for auto-router classification |

A route needs `match` or `description`; a description needs a `name`.

### `[routing]`

| Key | Type | Default | Description |
|---|---|---|---|
| `default` | string | `"higgs"` | Where a request goes when no model or route matches |
| `auto.enabled` | bool | `false` | Classify `model = "auto"` requests with a local model |
| `auto.model` | string | `"katanemo/Arch-Router-1.5B"` | The classifier: a `[[models]]` name or path (added when missing) |
| `auto.timeout` | duration | `"2s"` | Bound on one classification |
| `auto.force` | bool | `false` | Classify every request |

### `[observability]`

| Key | Type | Default | Description |
|---|---|---|---|
| `retention` | duration or `false` | `"60m"` | How long metrics stay in memory for the dashboard. `"0"` or `false` turns the window off; `true` is the default window |
| `metrics_log.enabled` | bool | `true` | Append request metrics to a JSONL file for `attach` history |
| `metrics_log.path` | path | `~/.config/higgs/logs/metrics.jsonl` | Log file (a named profile gets its own default) |
| `metrics_log.max_size` | size | `"50MB"` | Rotate the file past this size |
| `metrics_log.max_files` | integer | `5` | Rotated files to keep; `0` never rotates |

### `[experimental]`

Config equivalents of engine environment variables. The variable, when set, wins over the file;
`higgs doctor` reports every one that is set.

| Key | Type | Default | Environment variable | Description |
|---|---|---|---|---|
| `mtp` | `"auto"`, `"on"`, `"off"` | `"auto"` | `HIGGS_MTP` | Use the checkpoint's multi-token-prediction head. `auto` lets the MLX profile decide |
| `mtp_draft_tokens` | integer 1-8 | `2` for huge checkpoints, else `1` | `HIGGS_MTP_DRAFT_N_MAX` | Draft tokens per speculative cycle |
| `chunked_prefill.threshold` | integer | from the profile | `HIGGS_CHUNKED_PREFILL_THRESHOLD` | Prompts longer than this prefill in chunks |
| `chunked_prefill.chunk` | integer | from the profile | `HIGGS_CHUNKED_PREFILL_CHUNK_SIZE` | Tokens per prefill chunk |
| `clear_cache_after_prefill` | bool | from the profile | `HIGGS_CLEAR_CACHE_AFTER_PREFILL` | Clear the MLX allocator cache after prefill |
| `prompt_lookup` | bool | `false` | `HIGGS_PROMPT_LOOKUP` | Prompt-lookup decoding while a request runs alone (see [Speculative decode](#speculative-decode)) |
| `mtp_adaptive_draft` | bool | `false` | `HIGGS_MTP_ADAPTIVE_DRAFT` | Grow or shrink the MTP draft window from recent acceptance |
| `mtp_prompt_lookup` | bool | `false` | `HIGGS_MTP_PROMPT_LOOKUP` | Hybrid MTP loop that also tries verified prompt-lookup drafts |

Other `HIGGS_*` variables that change behavior and have no config key: `HIGGS_ENABLE_THINKING`
(`0|1` forces Qwen thinking off or on), `HIGGS_TURBOQUANT_MIN_TOKENS` (TurboQuant activation threshold, default
`2048`). The rest are debugging or
profiling switches for the model kernels.

### `[mcp]` and `[web]`

Keep their previous shape; see [docs/mcp.md](mcp.md). Their durations and sizes follow the conventions above.

| Key | Type | Default | Description |
|---|---|---|---|
| `mcp.max_result_size` | size | `"256KB"` | Larger tool results are truncated |
| `mcp.servers.<name>.timeout` | duration | `"30s"` | Bound on connecting, listing and each tool call |
| `web.timeout` | duration | `"15s"` | Bound on one search or fetch, redirects included |

## Migrating an existing file

Every key from earlier versions still loads; use `higgs config migrate` to rewrite them. The table lists each
old key and what replaced it.

| Old key | New key | Notes |
|---|---|---|
| `[server] timeout` | `[server] request_timeout` | seconds (number) becomes a duration string |
| `[server] max_body_size` | `[server] max_body_size` | same key; a bare integer is still bytes, strings like `"10MB"` also work |
| `[local] mlx_profile` | `[defaults] profile` | |
| `[local] raise_wired_limit` | `[memory] raise_wired_limit` | |
| `[local] thinking_budget` | `[defaults] thinking_budget` | |
| `[local] load` | `[defaults] load` | |
| `[local] idle_ttl_seconds` | `[defaults] idle_ttl` | seconds become a duration |
| `[local] max_loaded_models` | `[memory] max_loaded_models` | |
| `[local] memory_limit_gb` | `[memory] limit` | GiB become a size, e.g. `48` becomes `"48GB"` |
| `[local] memory_guard` | `[memory] guard` | |
| `[[models]] mlx_profile` | `[[models]] profile` | |
| `[[models]] idle_ttl_seconds` | `[[models]] idle_ttl` | |
| `[[models]] kv_cache = "turboquant"` | `[models.kv_cache] mode` | |
| `[[models]] kv_bits` | `[models.kv_cache] bits` | |
| `[[models]] kv_key_bits` | `[models.kv_cache] key_bits` | |
| `[[models]] kv_value_bits` | `[models.kv_cache] value_bits` | |
| `[[models]] kv_norm_correction` | `[models.kv_cache] norm_correction` | |
| `[[models]] kv_adaptive_dense_layers` | `[models.kv_cache] adaptive_dense_layers` | |
| `[[models]] kv_seed` | `[models.kv_cache] seed` | |
| `[[models]] mla_latent_cache` | `[models.kv_cache] mla_latent` | |
| `[[models]] kv_disk_dir` | `[models.prefix_cache] disk_dir` | |
| `[[models]] kv_disk_space_mb` | `[models.prefix_cache] disk_budget` | MiB become a size |
| `[[models]] prefill_yield_tokens` | `[models.prefill] yield_tokens` | |
| `[provider.<name>]` | `[providers.<name>]` | every key keeps its name |
| `[[routes]] pattern` | `[[routes]] match` | |
| `[default] provider` | `[routing] default` | |
| `[auto_router] enabled`, `force`, `model` | `[routing.auto] enabled`, `force`, `model` | |
| `[auto_router] timeout_ms` | `[routing.auto] timeout` | milliseconds become a duration |
| `[logging.metrics] enabled`, `path`, `max_files` | `[observability.metrics_log] enabled`, `path`, `max_files` | |
| `[logging.metrics] max_size_mb` | `[observability.metrics_log] max_size` | MiB become a size |
| `[retention] enabled`, `minutes` | `[observability] retention` | `minutes = 30` becomes `"30m"`; `enabled = false` becomes `false` |
| `[mcp] max_result_bytes` | `[mcp] max_result_size` | |
| `[mcp.servers.<name>] timeout_seconds` | `[mcp.servers.<name>] timeout` | |
| `[web] timeout_seconds` | `[web] timeout` | |

Behavior notes:

- Settings that used to apply only to `[local]` now also apply through `[defaults]` to the `kv_cache`, `prefix_cache`
  and `prefill` groups and to `batch` and `pinned`.
- `~/` in `models[].path`, `prefix_cache.disk_dir`, `metrics_log.path` and `include` entries now expands to the home
  directory (the old template suggested `~/...` for the log path, which was previously used literally).
- `retention = "0"` (or `false`) turns the in-memory window off. The old `[retention] enabled = true, minutes = 0`
  kept a zero-length window.
- The old server-side environment overlay `HIGGS_SERVER__PORT` (and the other `HIGGS_<SECTION>__<KEY>` names, in
  either the old or the new spelling) keeps working.

## Model lifecycle

By default every `[[models]]` entry loads at startup and stays resident until the server stops. Set `load = "on_demand"` to load a model on the first request that routes to it, and add an idle timeout, a model-count cap, or a memory ceiling to free memory for models that are not in use.

### Fields

`[defaults]` sets values for every model; the same-named `[[models]]` key overrides it for one model.

| Field | Where | Type | Default | Description |
|---|---|---|---|---|
| `load` | `[defaults]`, `[[models]]` | `"startup"` or `"on_demand"` | `"startup"` | When the model is loaded |
| `idle_ttl` | `[defaults]`, `[[models]]` | duration | *(none)* | Unload after this long with no in-flight request. `"0"` or unset never unloads on idle; a model-level `"0"` opts one model out of a `[defaults]` value |
| `pinned` | `[defaults]`, `[[models]]` | bool | `false` | Never evicted or unloaded on idle. An explicit `/unload` still works |
| `max_loaded_models` | `[memory]` | integer >= 1 | *(none)* | Most models allowed in memory at once |
| `limit` | `[memory]` | size > 0 | *(none)* | Ceiling on MLX active memory. Wins over `guard` |
| `guard` | `[memory]` | `"off"`, `"safe"`, `"balanced"`, `"aggressive"` | `"off"` | Derive the ceiling from total unified memory by reserving 20%, 8%, or 2% for the rest of the system |

`higgs doctor` rejects out-of-range values and warns when pinned models alone exceed the ceiling (by weight size on disk), when more models load at startup than `max_loaded_models` allows, and when `on_demand` is used with a single model, which only delays the first request.

### Rules

- **Lazy loading.** A request for an unloaded model loads it first. Concurrent first requests share one load. If it fails, every waiting request gets a `503` and the next request tries again; a failed load is never cached.
- **Startup models.** Models that resolve to `startup` load before the server binds, as before. The `routing.auto` classifier always loads at startup and is never evicted or expired, whatever its `load`.
- **Eviction.** Before a load that would exceed `max_loaded_models` or the memory ceiling, Higgs unloads the least recently used model that is idle and not pinned, repeating until the model fits. Models with requests in flight and pinned models are never evicted.
- **Memory accounting.** The size of a model before loading is estimated from its weight files on disk. After loading, the growth in MLX active memory is recorded as the measured size, which includes any other model generating at that moment, so treat it as approximate. The ceiling is compared with MLX active memory (weights plus KV caches), not process RSS or MLX's reusable buffer cache.
- **Refusal.** If the model still does not fit once nothing more can be evicted, or the model alone exceeds the ceiling, the load is refused with `503` and `code = "insufficient_memory"`. The message names the model, its estimated size, the ceiling, and each loaded model with the reason it could not be evicted (pinned or in use). A model that turns out larger than estimated is checked again after loading and unloaded if it breaks the ceiling.
- **Idle expiry.** A model with `idle_ttl` is unloaded once it has had no in-flight request for that long. A request that is still streaming keeps its model alive; the idle clock starts when the last request finishes.
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
| `GET /v1/system` | Process and memory facts plus the same per-model fields for every configured model, with `path`, `mlx_profile`, `kv_cache` and `prefix_cache_entries` (the prefix cache's `max_entries`; `0` means off) |

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
| `engine` | `batched` when concurrent requests decode together, `sequential` when requests queue and decode one at a time, or `unknown` while the checkpoint cannot be inspected yet (not downloaded) |
| `idle_ttl_seconds` | Effective idle timeout, or `null` |

## Routing Behavior

Higgs resolves requests in this order:

1. Auto-router when `model == "auto"` or force mode is enabled
2. Direct local engine lookup by model name or alias
3. Regex route (`match`), first match wins
4. `routing.default` fallback

That means Higgs supports:

- direct local model selection, under the model's name or any of its `aliases`
- pattern routing to local or remote targets
- model alias rewriting before forwarding
- auto-routing with a local classifier model
- a default target when nothing else matches

A local model name or alias always wins over a route, so a route whose `match` also matches a local name
is never used for that name; `higgs doctor` warns about it.

## Local Model Notes

- `batch = false` limits a model to one request at a time; `true` and `"auto"` (the default) mean no limit. See [Concurrent requests](#concurrent-requests).
- `[memory] raise_wired_limit` defaults to `false`. Turn it on only when you explicitly want MLX to raise the process wired-memory limit.
- With several resident models, consider `max_loaded_models`, `guard`, or `load = "on_demand"`; see [Model lifecycle](#model-lifecycle).
- For a local model, set `prefix_cache.disk_dir` to retain block-aligned prefix KV state across restarts.
  `prefix_cache.disk_budget` controls its on-disk LRU budget (default `4GB`, minimum `64MB`). The disk tier is
  model, quantization, configuration, tokenizer, and chat-template bound; incompatible entries are ignored.
- `prefix_cache.max_entries` bounds the in-memory prefix cache of every model; hybrid models also keep
  recurrent-state checkpoints per entry, sized by `prefix_cache.checkpoint_interval_tokens`,
  `max_checkpoints_per_entry` and `checkpoint_budget` (see the `[models.prefix_cache]` table).
- For DeepSeek-V2 models, set `kv_cache.mla_latent = true` to store the MLA KV cache as compressed latent rows
  instead of dense per-head tensors (default off). It cannot be combined with `kv_cache.mode = "turboquant"`, and
  is a no-op for non-DeepSeek-V2 architectures. The `HIGGS_MLA_LATENT_CACHE` env var, when set to a recognized
  value (`1`/`0`, `true`/`false`, `on`/`off`, `yes`/`no`), overrides the config value either way.
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
| `higgs config check` | Parse and validate the config file |
| `higgs config explain [model]` | Show every resolved setting and its source |
| `higgs config migrate [--write]` | Rewrite deprecated keys in the current schema |
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
the figures include the channel hop between engine and HTTP handler. The engine
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
3. `thinking_budget` under `[defaults]`.
4. The built-in default, 256.

A budget must be at least 1, in config (`higgs doctor` fails on `0`) and in requests (`400`).
To turn thinking off use reasoning effort `"none"`; a zero budget is not a way to do that.
`higgs doctor` warns when a budget is not below `server.max_tokens`, since reasoning could
then consume the whole response. The engine forces `</think>` in place of the sampled token, so
its KV cache always matches the text the client receives, including during speculative decode. On `/v1/messages` only
`budget_tokens` is read from `thinking`; `type` is ignored.

### Concurrent requests

One engine serves every model. It runs on a worker thread that owns the model, admits requests
as they arrive, prefills them (through the prefix cache, in chunks for long prompts) and then
decodes every active request together. How a model decodes depends on the model, not on a
setting:

| Model | Decoding |
|---|---|
| An architecture with batched decode (see [docs/models.md](models.md)) | All active requests share one batched forward pass per token, so concurrent requests share the GPU instead of queueing (measured on an M4 Max: Phi-3 at 8 clients 164 to 461 tok/s, a 27B Qwen3.5 model at 4 clients 27 to 69 tok/s) |
| Any other architecture, a checkpoint with an MTP head, or `kv_cache.mode = "turboquant"` | Requests queue in arrival order and decode one at a time, the lone request running a token ahead of the one it reports |

A request that needs per-request handling does not slow the others' setup: a request with
`logprobs` or a structured-output constraint makes the round step every active request on its own
forward pass until it finishes, and a request with images prefills alone and never reads or
writes the prefix cache.

The per-model `batch` setting is a concurrency cap:

| `batch` | Effect |
|---|---|
| `"auto"` (default), `true` | No limit. `true` is the same as `"auto"`; `higgs doctor` warns that it has no effect |
| `false` | One request at a time, whatever the model could batch (for example to bound KV memory) |

`higgs serve --no-batch` sets `false` for every `--model`; `--batch` is accepted and ignored.
`higgs doctor` and the server's load log report which of the two a model gets and why, and
`/v1/models/status` shows it as `engine`.

The prefix cache is block-paged and keyed without the chat template's generation prompt, so
repeated prompts and multi-turn conversations reuse everything but that suffix, and a prompt that
diverges mid-way reuses the shared leading blocks. The thinking budget, logprobs, stop sequences,
structured output, `return_progress` and embeddings work the same for every model.

### Speculative decode

Speculative decode drafts several tokens, verifies them with the model in one pass, and keeps the
accepted prefix. It advances one sequence at a time, so it only runs while a request is alone and
only for greedy requests (`temperature = 0`, no penalties) without `logprobs` or a constraint:

- Prompt-lookup decoding (`experimental.prompt_lookup`) copies drafts from text the request
  already produced. It works on any architecture.
- MTP decode uses the checkpoint's multi-token-prediction head (`experimental.mtp`, subject to
  the MLX profile). A checkpoint with an MTP head decodes one request at a time, so its requests
  queue and each runs speculatively.

When a second request arrives while a prompt-lookup request is running, the first request is
handed over to ordinary batched decode at the next token boundary: the token the model had already
settled on is reported, the draft state is dropped, and the request continues in the batch. When it is
alone again it returns to speculative decode. An MTP request never needs a handover because
its model queues requests; if its reasoning budget forces `</think>` it finishes on ordinary decode.

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

## Other migration notes

- `higgs start` no longer accepts serve-style flags like `--model`, `--port`, or `--batch`.
- `/metrics` is available and `server.max_body_size` is enforced on API routes.
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
