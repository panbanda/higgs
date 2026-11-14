<div align="center">

<img src="docs/images/higgs-header.png" alt="Higgs: model router for Apple hardware" width="920">

**Run open-weight LLMs on your Mac and point your existing tools at them. One Rust binary, OpenAI and Anthropic compatible APIs, built on MLX for Apple Silicon.**

[![CI](https://github.com/panbanda/higgs/actions/workflows/ci.yml/badge.svg)](https://github.com/panbanda/higgs/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/panbanda/higgs)](https://github.com/panbanda/higgs/releases)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-black)](#install)

<img src="docs/images/higgs-chat.gif" alt="Higgs Desktop: a Qwen3.8-27B conversation with live tokens per second and the request trace" width="920">

</div>

Higgs serves MLX models from Hugging Face on your Mac behind the same endpoints that OpenAI and Anthropic clients already speak: chat completions, completions, Responses, embeddings and messages. It can also route requests to remote providers, run MCP tools for the model, and show how fast every request ran. There is no Python environment to manage.

It is for people who want local models behind a real API: coding agents, scripts, apps, and several models on one machine.

## Quick start

Apple Silicon Mac required.

```bash
brew install panbanda/brews/higgs
higgs serve --model mlx-community/Llama-3.2-1B-Instruct-4bit
```

If the model is not in your Hugging Face cache, Higgs asks before downloading it with the `hf` CLI; answer `y`. In another terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Llama-3.2-1B-Instruct-4bit","messages":[{"role":"user","content":"Write one sentence about Cape Town."}]}'
```

That is a 1B model chosen so the first run is fast. Swap in a larger one when it works; see [which model to start with](#which-model-should-i-start-with). To use Higgs from Claude Code, the OpenAI SDK or an app, see [Recipes](#recipes).

## Contents

- [Why Higgs](#why-higgs)
- [Install](#install)
- [Recipes](#recipes)
- [Supported models](#supported-models)
- [Configuration](#configuration)
- [API surface](#api-surface)
- [Commands](#commands)
- [Desktop app](#desktop-app)
- [FAQ and troubleshooting](#faq-and-troubleshooting)
- [Development](#development)
- [License](#license)

## Why Higgs

- **Fast under load.** Concurrent requests decode together in batched forward passes, for most architectures including the large Qwen3.5 and Qwen3.6 hybrids. On four simultaneous clients Higgs delivered 1.1x to 1.6x the throughput of oMLX on every model tested.
- **Fast follow-up turns.** Prefix KV state stays in memory, so a chat or agent loop that repeats its history skips the prefill. On hybrid Qwen3.5 and Qwen3.6 models, which cannot simply truncate a cache, Higgs checkpoints the recurrent state during prefill: a new question over a 3,850-token shared context took 3.7 s to first token instead of 20.1 s.
- **Memory you control.** Load models on first request, unload them when idle, cap how many stay resident, pin the ones that must stay, and set a memory guard. Peak memory was lower than oMLX on every model measured.
- **One endpoint for local and remote.** Local models and remote providers (OpenAI, Anthropic, Ollama and others) behind one URL, with regex routes, aliases, an optional auto-router, and OpenAI/Anthropic format translation.
- **Tools that work.** Tool calls are parsed for the Qwen, Mistral, Llama, Gemma, GLM and MiniMax formats, streaming included, and `tool_choice` can force a valid call. An MCP client and web search and fetch let the server run tools for the model.
- **Strict, inspectable config.** Unknown keys are errors with a suggestion. `higgs config explain` shows where every value came from, and `higgs doctor` checks models and providers before you start.
- **You can see what is happening.** Time to first token, decode speed and prefix-cache hits per request, in `/metrics`, a terminal dashboard and a desktop app.
- **A single binary.** No Python, no virtual environment. Release builds bundle the Metal shader library.

### Numbers against oMLX

Measured on an Apple M4 Max with 128 GB, same client harness against both servers, medians of 3 trials, one server at a time, temperature 0. Source, method and reproduction steps: [docs/omlx-comparison.md](docs/omlx-comparison.md). That document calls itself a first pass and notes that differences under about 5% are noise. A fresh benchmark run is pending; when it lands, update this table and the comparison document together.

<!-- BENCHMARK TABLE: every figure below is copied from docs/omlx-comparison.md. Update both together. -->

| Measure | Model | Higgs | oMLX |
|---|---|---|---|
| Time to first token, follow-up turn (about 3,000 tokens of history) | Llama-3.2-1B-4bit | 32 ms | 117 ms |
| | Qwen3-1.7B-4bit | 57 ms | 219 ms |
| | Thinklite-27B-4bit | 389 ms | 457 ms |
| | Qwen3.6-35B-A3B-4bit | 84 ms | 156 ms |
| Aggregate tokens/s, 4 concurrent clients | Llama-3.2-1B-4bit | 1105 | 931 |
| | Qwen3-1.7B-4bit | 787 | 708 |
| | Thinklite-27B-4bit | 69 | 43 |
| | Qwen3.6-35B-A3B-4bit | 255 | 165 to 175 |
| Peak server memory | Llama-3.2-1B-4bit | 1.2 GB | 1.9 GB |
| | Qwen3-1.7B-4bit | 1.3 GB | 5.5 GB |
| | Thinklite-27B-4bit | 14.9 GB | 16.2 GB |
| Single-stream decode, tokens/s | Llama-3.2-1B-4bit | 480 | 472 |
| | Qwen3-1.7B-4bit | 329 | 325 |
| | Thinklite-27B-4bit | 29.0 | 29.6 |
| | Qwen3.6-35B-A3B-4bit | 134 | 130 to 185 |

<!-- END BENCHMARK TABLE -->

Where Higgs does not win:

- **Qwen3.6-35B-A3B single-stream decode.** oMLX reaches about 184 tok/s on most requests (86 to 130 on others) against 134 for Higgs, because it ships fused Metal kernels for that architecture's MoE block. Closing this gap is open work.
- **An 8-client burst on 1B to 2B models.** Level to slightly behind: 1350 against 1385 tok/s on Llama-1B and 952 against 1016 on Qwen3-1.7B.
- **Features.** oMLX is broader: audio, rerank, EBNF grammars, broad vision support, an admin web UI and clustering. The full table is in the comparison document.

Single-stream decode of dense models and cold prefill are level, because both stacks run the same MLX kernels and hit the memory-bandwidth limit.

## Install

Homebrew:

```bash
brew install panbanda/brews/higgs                  # CLI and server
brew install --cask panbanda/brews/higgs-desktop   # desktop app; bundles the CLI
```

Release downloads, from the [latest release](https://github.com/panbanda/higgs/releases/latest):

- `higgs_<version>_aarch64-apple-darwin.tar.gz`: the `higgs` binary and `mlx.metallib`, with a `.sha256` file. Keep the two files in the same directory.
- `Higgs_<version>_aarch64.dmg` and `Higgs_<version>_aarch64.app.zip`: the desktop app. The bundle is not notarized, so on first launch right-click and choose Open, or run `xattr -dr com.apple.quarantine /Applications/Higgs.app`.

The Homebrew formula also installs the `hf` CLI, which Higgs uses to download models. With any other install method, run `brew install hf`; in scripts and other non-interactive runs, fetch a model first with `hf download <model-id>`.

From source, with Rust 1.88 or newer and the Xcode Command Line Tools:

```bash
git clone https://github.com/panbanda/higgs && cd higgs
cargo build --release
./target/release/higgs --help
```

Higgs is not published on crates.io, because it depends on a pinned git revision of `mlx-rs`. Only Apple Silicon macOS builds are produced.

## Recipes

Each recipe is self-contained. The ones that use `higgs exec`, `higgs shellenv` or `higgs doctor` read a config file, so they start from `~/.config/higgs/config.toml`; create it with `higgs init` and edit it, or write your own.

### Use it from Claude Code

Claude Code needs a model name Higgs knows. This config serves one local model and sends every `claude-*` model name to it:

```toml
[[models]]
path = "mlx-community/Qwen3.6-35B-A3B-4bit"
name = "qwen"

[[routes]]
match = "claude-.*"
provider = "higgs"      # the local models
model = "qwen"          # rewrite the requested name
```

Start the server in one terminal with `higgs serve`, then in another:

```bash
ANTHROPIC_AUTH_TOKEN=higgs higgs exec -- claude
```

`higgs exec` checks that the server is reachable, sets `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL`, and runs the command. `eval "$(higgs shellenv)"` exports the same variables into your shell instead. Both work with Aider and any other tool that reads these variables. Use a model that is good at tool calling; small models struggle with agent prompts.

### Use it from the OpenAI SDK or any OpenAI client

Point the client at the server. The API key can be any string unless you set `server.api_key`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    messages=[{"role": "user", "content": "Write one sentence about Cape Town."}],
)
print(reply.choices[0].message.content)
```

`GET /v1/models` lists the names the server accepts. The Anthropic SDK works the same way against `http://localhost:8000` and `/v1/messages`.

### Serve several models and load them on demand

```toml
[memory]
max_loaded_models = 2       # evict the least recently used beyond two
guard = "balanced"          # keep about 8% of unified memory for the system

[defaults]                  # applies to every model; a [[models]] entry overrides it
load = "on_demand"          # load on the first request
idle_ttl = "15m"            # unload after 15 idle minutes

[[models]]
path = "mlx-community/Qwen3.6-35B-A3B-4bit"
name = "qwen"
pinned = true               # never evicted or expired

[[models]]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
name = "llama"
```

`POST /v1/models/{id}/load` and `/unload` control models by hand and `GET /v1/models/status` shows state and memory per model. Details: [Model lifecycle](docs/configuration.md#model-lifecycle).

### Tool calling

Send `tools` as you would to OpenAI. Higgs parses the call out of the model's own format and returns structured `tool_calls`. To force a call, set `tool_choice` to `"required"` or to a named function:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "tool_choice": "required",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{"type": "function", "function": {
      "name": "get_weather",
      "description": "Get the weather for a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
  }'
```

Forcing works for the Qwen, Mistral and Llama formats and returns a 400 for the others. See [docs/tool-calling.md](docs/tool-calling.md) for each family's format.

### MCP tools

Higgs connects to MCP servers (stdio or Streamable HTTP) and can run their tools for the model:

```toml
[mcp.servers.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Documents"]
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen",
       "messages": [{"role": "user", "content": "List the files in my Documents folder."}],
       "mcp": {"enabled": true, "servers": ["filesystem"], "max_iterations": 5}}'
```

The request uses a model named `qwen`, as in the recipes above. Tools are namespaced `server__tool`. An existing `{"mcpServers": {...}}` file loads with `higgs serve --mcp-config mcp.json`. Optional web search and fetch endpoints are off by default. See [docs/mcp.md](docs/mcp.md).

### Structured output

Constrain the response to a JSON schema:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Give me a person with a name and an age."}],
    "response_format": {"type": "json_schema", "json_schema": {"name": "person", "schema": {
      "type": "object",
      "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
      "required": ["name", "age"]}}}
  }'
```

`response_format` also accepts `json_object` and `{"type": "regex", "regex": "..."}`; `guided_json` and `guided_regex` are accepted too. Details: [Guided output](docs/configuration.md#guided-output).

### Route to a remote provider

Requests for local model names run locally. Names that match a route go to the provider, and anything else goes to `routing.default`:

```toml
[[models]]
path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
name = "llama"

[providers.anthropic]
url = "https://api.anthropic.com"
format = "anthropic"
api_key = "${ANTHROPIC_API_KEY}"

[providers.openai]
url = "https://api.openai.com"
api_key = "${OPENAI_API_KEY}"

[[routes]]
match = "claude-.*"
provider = "anthropic"

[[routes]]
match = "gpt-.*"
provider = "openai"

[routing]
default = "openai"      # unmatched names go here instead of the local models
```

Higgs translates between the OpenAI and Anthropic wire formats, so an OpenAI client can use a Claude model through the same URL. This is routing, not failover: a request is not retried on another target when one is down. Resolution order, the auto-router and Ollama as a provider are in [docs/configuration.md](docs/configuration.md#routing-behavior).

## Supported models

MLX `safetensors` checkpoints, from a Hugging Face ID or a local path.

| Family | Batched decode | Vision |
|---|---|---|
| Llama, Mistral | yes | |
| Qwen2, Qwen3, Qwen3-MoE | yes | |
| Qwen3.5, 3.6 and 3.8 (dense and MoE) | yes | |
| Qwen3-Next | no | |
| Gemma 2, Gemma 3 | yes | |
| Gemma 4 | yes, except the MoE variant | |
| Phi-3, Starcoder2 | yes | |
| DeepSeek-V2 | no | |
| LLaVA-Qwen2 | no | yes |

Models that cannot batch are still served; their requests queue and decode one at a time. Unknown newer versions of a supported family load with the nearest adapter and a warning, and unknown families are rejected with the supported list. Architecture identifiers, notes on Gemma and Qwen checkpoints, and example model IDs: [docs/models.md](docs/models.md).

## Configuration

`higgs serve --model <id>` needs no config file. For anything more, run `higgs init` to write `~/.config/higgs/config.toml`, or write one:

```toml
[server]
host = "127.0.0.1"
port = 8000
api_key = "${HIGGS_API_KEY}"    # environment reference; a missing variable is a startup error
request_timeout = "5m"          # durations and sizes are strings: "90s", "5m", "512MB", "4GB"

[[models]]
path = "mlx-community/Qwen3.6-35B-A3B-4bit"
name = "qwen"
aliases = ["default"]           # extra request names that resolve to this model

[providers.anthropic]
url = "https://api.anthropic.com"
format = "anthropic"
api_key = "${ANTHROPIC_API_KEY}"

[[routes]]
match = "claude-.*"
provider = "anthropic"

[routing]
default = "higgs"
```

Unknown keys are errors with a suggestion (`idel_ttl` is reported as `idle_ttl`). `higgs config check` validates a file without loading models, `higgs config explain` shows where every resolved value comes from, `higgs config migrate` rewrites a file written for an earlier version, and `higgs doctor` checks models and providers. Everything else is in [docs/configuration.md](docs/configuration.md): every flag and environment variable, routing precedence, the auto-router, KV cache and prefill options, speculative decoding, profiles, includes and metrics.

## API surface

| Endpoint | Compatibility |
|---|---|
| `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models` | OpenAI |
| `/v1/responses` (streaming, stored responses) | OpenAI Responses, local models and OpenAI-format providers; [docs/responses.md](docs/responses.md) |
| `/v1/messages`, `/v1/messages/count_tokens` | Anthropic |
| `/v1/tokenize`, `/v1/detokenize` | llama.cpp style, local models |
| `/v1/models/{id}/load`, `/v1/models/{id}/unload`, `/v1/models/status` | Higgs model lifecycle |
| `/v1/mcp/servers`, `/v1/mcp/tools`, `/v1/mcp/execute` | Higgs MCP client; [docs/mcp.md](docs/mcp.md) |
| `/v1/web/search`, `/v1/web/fetch` | Higgs; off by default |
| `/metrics`, `/v1/system`, `/health` | Higgs dashboard and control |

Streaming, `stream_options.include_usage`, cached-token and timing fields in `usage`, `return_progress`, a reasoning-token budget, `seed` (accepted, not yet reproducible) and the rest of the request options are documented in [docs/api.md](docs/api.md).

## Commands

| Command | What it does |
|---|---|
| `higgs serve` | Start the server in the foreground (`--model` repeats; every flag also works with a config) |
| `higgs start`, `higgs stop` | Run as a background daemon from config or profile; stop it (`--force` escalates to `SIGKILL`) |
| `higgs attach` | Terminal dashboard for the daemon |
| `higgs ui` | Open the desktop app |
| `higgs init` | Write `~/.config/higgs/config.toml` |
| `higgs doctor` | Validate config, model paths and providers |
| `higgs config check\|explain\|migrate` | Validate a file, show where each value comes from, rewrite an old file |
| `higgs config get\|set\|path` | Read or write a value, print the config path |
| `higgs shellenv` | Print `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL` after checking the server is reachable |
| `higgs exec -- <cmd>` | Run a command with those variables set |
| `higgs bench <model>` | Measure decode, prefill, prefix-cache and concurrency speed of one model on a private server |
| `higgs selftest <model>` | Check one model for correct output: chat, streaming, prefix cache, JSON schema, tool calls and concurrency. Exits non-zero when a check fails |

`higgs bench` accepts `--scenarios decode,prefill,prefix_cache,concurrency`, `--trials`, `--warmup`, `--max-tokens` and `--json <path>`. Global flags are `--config <FILE>`, `--profile <NAME>` and `--verbose`. Methodology and the benchmark harnesses are in [docs/benchmarking.md](docs/benchmarking.md).

## Desktop app

<img src="docs/images/desktop-overview.png" alt="Higgs Desktop overview" width="920">

A native macOS dashboard for a running server, and the graphical counterpart of `higgs attach`. It shows requests per minute, error rate, memory, prefix-cache hit rate, time to first token and decode speed, and has a Hub for browsing and downloading `mlx-community` checkpoints, a config editor, and a chat whose inspector traces every span of a request (prefill and cache hits, thinking, generation, tool calls) with Replay and Copy as curl.

Install with `brew install --cask panbanda/brews/higgs-desktop` or from the release page; `higgs ui` opens it. Metrics need a config-file start (`higgs start`, or `higgs serve --config <file>`). Details: [docs/desktop.md](docs/desktop.md).

## FAQ and troubleshooting

### Which model should I start with?

Start with the 1B model from the quick start to check that everything works. For daily use, pick the largest 4-bit model that fits comfortably in unified memory next to your other apps. The weight files on disk are a floor on what a loaded model uses; the KV cache comes on top. The models used in the [benchmarks](docs/omlx-comparison.md) are `Qwen3-1.7B-4bit`, `Thinklite-27B-4bit` and `Qwen3.6-35B-A3B-4bit`, and [docs/models.md](docs/models.md) lists IDs that are known to load. Browse more in the desktop app's Hub.

### How much memory will it use?

`GET /v1/models/status` reports `estimated_bytes` (from the weight files, before loading) and `measured_bytes` (after loading) per model. To stay within a budget, set `[memory] guard` (`safe`, `balanced` or `aggressive` reserve 20%, 8% or 2% of RAM for the system) or `[memory] limit`. A load that cannot fit is refused with a 503 and `insufficient_memory`, and the message names what could not be evicted. See [Model lifecycle](docs/configuration.md#model-lifecycle).

### Which Macs and systems does it run on?

Apple Silicon Macs only. Release builds target `aarch64-apple-darwin`, built on macOS 15. The minimum macOS version is not documented. Intel Macs, Linux and Windows are not supported.

### The download prompt does not appear, or the model is not found.

Downloads need the `hf` CLI on your `PATH` and an interactive terminal. Otherwise run `hf download <model-id>` yourself, then start Higgs again. Gated repositories need a Hugging Face token.

### It fails with an `mlx.metallib` error.

The server needs `mlx.metallib` next to the executable. Release archives and Homebrew installs include it. With a source build, Higgs restores it from the Cargo build output; if you copy `target/release/higgs` elsewhere, copy the library with it.

### My request returns `model_not_found`.

The `model` field must be a name the server knows: the configured `name`, one of its `aliases`, or for a model started with `--model` the name shown by `GET /v1/models`. A name that matches no model or route goes to `routing.default`. To make other clients' names work, add `aliases` or a route with a `model` rewrite, as in the Claude Code recipe.

### `higgs shellenv` or `higgs exec` says there is no model or provider.

These commands read a config file, not `serve` flags. Run `higgs init` and add a `[[models]]` entry, or pass `--config <file>`, then start the server with `higgs serve` (or `higgs start`) so the command can reach it.

### A model with thinking returns an empty answer.

Reasoning models such as Qwen3 spend tokens thinking first, and a small `max_tokens` can run out before the answer. Raise `max_tokens`, lower the reasoning budget (`reasoning.max_tokens` per request, `thinking_budget` per model), or turn thinking off for the request with `"reasoning": {"effort": "none"}`. See [Thinking budget](docs/configuration.md#thinking-budget).

### Something is wrong with my config.

Run `higgs config check`, then `higgs doctor`. The first validates the file without touching models; the second also checks model paths, providers and memory settings.

### How does it compare with Ollama, LM Studio and oMLX?

- **oMLX**, another MLX server, written in Python: measured head to head in [docs/omlx-comparison.md](docs/omlx-comparison.md), including where it is ahead.
- **Ollama**: in the earlier single-stream decode measurements, Higgs reached 448 tok/s on Llama-3.2-1B-4bit and 305 on Qwen3-1.7B-4bit against Ollama's 305 and 183 on `Q4_K_M` quantizations; on the Qwen3-30B-A3B MoE at 8-bit the two were close (75 and 73). See [docs/performance.md](docs/performance.md).
- **LM Studio**: not benchmarked. It is a closed-source desktop application, while Higgs is an open-source server and CLI with a companion desktop app.

## Development

```bash
cargo clippy -p higgs                         # must be clean
cargo fmt -p higgs -- --check
cargo test -p higgs -- --test-threads=1       # the thread limit is required: tests share ports
cd apps/desktop && pnpm install && pnpm typecheck && pnpm build
```

Project layout, the doc expectations for config and CLI changes, and release validation are in [CONTRIBUTING.md](CONTRIBUTING.md). Notes for older `higgs start` flags are in [docs/configuration.md](docs/configuration.md#other-migration-notes).

More reference: [API reference](docs/api.md), [performance and Apple Silicon notes](docs/performance.md), [benchmarking](docs/benchmarking.md), [MCP and web tools](docs/mcp.md), [Responses API](docs/responses.md).

## License

MIT
