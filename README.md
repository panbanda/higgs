<div align="center">

<img src="docs/images/higgs-header.png" alt="Higgs: model router for Apple hardware" width="920">

**Run open-weight LLMs on your Mac and point your existing tools at them. One Rust binary, OpenAI and Anthropic compatible APIs, built on MLX for Apple Silicon.**

[![CI](https://github.com/panbanda/higgs/actions/workflows/ci.yml/badge.svg)](https://github.com/panbanda/higgs/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/panbanda/higgs)](https://github.com/panbanda/higgs/releases)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)
[![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-black)](#install)

<img src="docs/images/higgs-chat.gif" alt="Higgs Desktop: a Qwen3.8-27B conversation with live tokens per second and the request trace" width="920">

</div>

Higgs serves MLX models from Hugging Face on your Mac. It uses the same endpoints that OpenAI and Anthropic clients already use: chat completions, completions, Responses, embeddings and messages. It can also route requests to remote providers and run MCP tools for the model. It shows how fast each request ran. There is no Python environment to manage.

Use Higgs when you want local models behind a real API. Typical uses are coding agents, scripts, apps, and several models on one machine.

## Quick start

You need an Apple Silicon Mac.

1. Install Higgs and start a model:

```bash
brew install panbanda/brews/higgs
higgs serve --model mlx-community/Llama-3.2-1B-Instruct-4bit
```

If the model is not in your Hugging Face cache, Higgs asks before it downloads the model with the `hf` CLI. Answer `y`.

2. In a second terminal, send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Llama-3.2-1B-Instruct-4bit","messages":[{"role":"user","content":"Write one sentence about Cape Town."}]}'
```

This 1B model is small, so the first run is fast. When it works, use a larger model. See [which model to start with](#which-model-should-i-start-with). To use Higgs from Claude Code, the OpenAI SDK or an app, see [Recipes](#recipes).

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

- **Fast under load.** Concurrent requests decode together in batched forward passes. This includes the large Qwen3.5 and Qwen3.6 hybrid models. With four simultaneous clients, Higgs gave 1.1x to 1.6x the throughput of oMLX on every model tested.
- **Fast follow-up turns.** Higgs keeps prefix KV state in memory. A chat or agent loop that repeats its history skips the prefill.
- **Hybrid models reuse their cache.** The recurrent layers of Qwen3.5 and Qwen3.6 cannot cut a cache back to a shorter prefix. Higgs saves checkpoints of that state during prefill. A new question over a 3,850-token shared context took 3.7 s to the first token, not 20.1 s.
- **Memory you control.** Load a model on its first request. Unload it when it is idle. Limit how many models stay loaded, pin the important ones, and set a memory guard. Peak memory was lower than oMLX on every model measured.
- **One endpoint for local and remote models.** Local models and remote providers (OpenAI, Anthropic, Ollama and others) use one URL. Higgs has regex routes, aliases, an optional auto-router, and OpenAI and Anthropic format translation.
- **Tool calling.** Higgs parses tool calls in the Qwen, Mistral, Llama, Gemma, GLM and MiniMax formats, also when streaming. `tool_choice` can force a valid call. An MCP client and web search and fetch let the server run tools for the model.
- **Strict config.** Unknown keys are errors, and the error suggests the correct key. `higgs config explain` shows where each value comes from. `higgs doctor` checks models and providers before you start.
- **Visible performance.** Higgs records time to first token, decode speed and prefix-cache hits for each request. You can see them in `/metrics`, in a terminal dashboard and in a desktop app.
- **One binary.** There is no Python and no virtual environment. Release builds include the Metal shader library.

### Numbers against oMLX

The numbers come from an Apple M4 Max with 128 GB. The same client harness ran against both servers, one server at a time. Each value is the median of 3 trials at temperature 0. The method and the steps to reproduce are in [docs/omlx-comparison.md](docs/omlx-comparison.md). That document is a first pass. Differences under about 5% are noise. A new benchmark run is pending. When it finishes, update this table and the comparison document together.

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

- **Qwen3.6-35B-A3B single-stream decode.** oMLX reaches about 184 tok/s on most requests and 86 to 130 tok/s on others. Higgs reaches 134 tok/s. oMLX has fused Metal kernels for the MoE block of this architecture. Higgs does not have them yet.
- **An 8-client burst on 1B to 2B models.** Higgs is level or slightly behind. On Llama-1B the result is 1350 against 1385 tok/s. On Qwen3-1.7B it is 952 against 1016 tok/s.
- **Features.** oMLX has more features: audio, rerank, EBNF grammars, broad vision support, an admin web UI and clustering. The comparison document has the full table.

Single-stream decode of dense models and cold prefill are level. Both servers use the same MLX kernels and reach the memory-bandwidth limit.

## Install

With Homebrew:

```bash
brew install panbanda/brews/higgs                  # CLI and server
brew install --cask panbanda/brews/higgs-desktop   # desktop app; bundles the CLI
```

From the [latest release](https://github.com/panbanda/higgs/releases/latest):

- `higgs_<version>_aarch64-apple-darwin.tar.gz` contains the `higgs` binary and `mlx.metallib`. A `.sha256` file is included. Keep the binary and the library in the same directory.
- `Higgs_<version>_aarch64.dmg` and `Higgs_<version>_aarch64.app.zip` contain the desktop app. The bundle is not notarized. On first launch, right-click the app and choose Open. You can also run `xattr -dr com.apple.quarantine /Applications/Higgs.app`.

The Homebrew formula also installs the `hf` CLI. Higgs uses it to download models. With any other install method, run `brew install hf`. In scripts and other non-interactive runs, download the model first with `hf download <model-id>`.

To build from source, use Rust 1.88 or newer and the Xcode Command Line Tools:

```bash
git clone https://github.com/panbanda/higgs && cd higgs
cargo build --release
./target/release/higgs --help
```

Higgs is not on crates.io, because it depends on a pinned git revision of `mlx-rs`. The release process produces Apple Silicon macOS builds only.

## Recipes

Each recipe is complete on its own. The commands `higgs exec`, `higgs shellenv` and `higgs doctor` read a config file. Create `~/.config/higgs/config.toml` with `higgs init` and edit it, or write your own file.

### Use it from Claude Code

Claude Code requests model names that Higgs does not know. This config serves one local model and sends each `claude-*` name to it:

```toml
[[models]]
path = "mlx-community/Qwen3.6-35B-A3B-4bit"
name = "qwen"

[[routes]]
match = "claude-.*"
provider = "higgs"      # the local models
model = "qwen"          # rewrite the requested name
```

1. Start the server in one terminal with `higgs serve`.
2. In a second terminal, run Claude Code through Higgs:

```bash
ANTHROPIC_AUTH_TOKEN=higgs higgs exec -- claude
```

`higgs exec` checks that the server is reachable. It sets `ANTHROPIC_BASE_URL` and `OPENAI_BASE_URL`, then runs the command. `eval "$(higgs shellenv)"` exports the same variables into your shell. Both methods work with Aider and with any other tool that reads these variables. Choose a model that calls tools well. Small models often fail with agent prompts.

### Use it from the OpenAI SDK or any OpenAI client

Set the base URL of the client to the server. The API key can be any string, unless you set `server.api_key`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    messages=[{"role": "user", "content": "Write one sentence about Cape Town."}],
)
print(reply.choices[0].message.content)
```

`GET /v1/models` lists the model names that the server accepts. The Anthropic SDK works the same way. Use `http://localhost:8000` as the base URL.

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

`POST /v1/models/{id}/load` and `POST /v1/models/{id}/unload` load and unload a model by hand. `GET /v1/models/status` shows the state and memory of each model. For details, see [Model lifecycle](docs/configuration.md#model-lifecycle).

### Tool calling

Send `tools` as you would send them to OpenAI. Higgs reads the call from the output format of the model and returns structured `tool_calls`. To force a call, set `tool_choice` to `"required"` or to the name of a function:

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

A forced call works for the Qwen, Mistral and Llama formats. For the other formats, Higgs returns a 400 error. [docs/tool-calling.md](docs/tool-calling.md) describes the format of each model family.

### MCP tools

Higgs connects to MCP servers (stdio or Streamable HTTP). It can run their tools for the model. Add a server to the config:

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

The request uses a model named `qwen`, as in the recipes above. Higgs names each tool `server__tool`. To load an existing `{"mcpServers": {...}}` file, use `higgs serve --mcp-config mcp.json`. The web search and fetch endpoints are optional and off by default. See [docs/mcp.md](docs/mcp.md).

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

`response_format` also accepts `json_object` and `{"type": "regex", "regex": "..."}`. Higgs also accepts `guided_json` and `guided_regex`. For details, see [Guided output](docs/configuration.md#guided-output).

### Route to a remote provider

A request for a local model name runs locally. A name that matches a route goes to the provider of that route. All other names go to `routing.default`:

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

Higgs translates between the OpenAI and Anthropic formats. An OpenAI client can use a Claude model through the same URL. This is routing, not failover. Higgs does not retry a request on another target when a provider is down. [docs/configuration.md](docs/configuration.md#routing-behavior) describes the resolution order and the auto-router.

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

Higgs also serves models that cannot batch. Their requests wait in a queue and decode one at a time. A newer version of a supported family loads with the nearest adapter and a warning. Higgs rejects an unknown family and shows the list of supported families. [docs/models.md](docs/models.md) has the architecture identifiers, notes on Gemma and Qwen models, and example model IDs.

## Configuration

`higgs serve --model <id>` needs no config file. For more options, run `higgs init` to write `~/.config/higgs/config.toml`, or write your own file:

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

An unknown key is an error, and the error suggests the correct key. For example, `idel_ttl` is reported as `idle_ttl`. These commands help with config files:

- `higgs config check` validates a file without loading models.
- `higgs config explain` shows where each resolved value comes from.
- `higgs config migrate` rewrites a file that was written for an earlier version.
- `higgs doctor` checks models and providers.

[docs/configuration.md](docs/configuration.md) has the complete reference. It covers every flag and environment variable, routing precedence, the auto-router, KV cache and prefill options, speculative decoding, profiles, includes and metrics.

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

[docs/api.md](docs/api.md) describes the request options. These include streaming, `stream_options.include_usage`, the cached-token and timing fields in `usage`, `return_progress` and the reasoning-token budget. Higgs accepts `seed`, but the output is not yet reproducible.

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

`higgs bench` accepts `--scenarios decode,prefill,prefix_cache,concurrency`, `--trials`, `--warmup`, `--max-tokens` and `--json <path>`. The global flags are `--config <FILE>`, `--profile <NAME>` and `--verbose`. [docs/benchmarking.md](docs/benchmarking.md) describes the method and the benchmark harnesses.

## Desktop app

<img src="docs/images/desktop-overview.png" alt="Higgs Desktop overview" width="920">

The desktop app is a native macOS dashboard for a running server. It is the graphical version of `higgs attach`.

- The dashboard shows requests per minute, error rate, memory, prefix-cache hit rate, time to first token and decode speed.
- The Hub lets you browse and download `mlx-community` models.
- The config editor changes the config file and checks it.
- The chat has an inspector. It traces each span of a request: prefill and cache hits, thinking, generation and tool calls. It also offers Replay and Copy as curl.

Install the app with `brew install --cask panbanda/brews/higgs-desktop` or from the release page. `higgs ui` opens it. Metrics need a start from a config file: `higgs start`, or `higgs serve --config <file>`. For details, see [docs/desktop.md](docs/desktop.md).

## FAQ and troubleshooting

### Which model should I start with?

Start with the 1B model from the quick start to check that everything works. For daily use, choose the largest 4-bit model that fits in unified memory with your other apps. A loaded model uses at least the size of its weight files on disk. The KV cache uses more memory. The [benchmarks](docs/omlx-comparison.md) used `Qwen3-1.7B-4bit`, `Thinklite-27B-4bit` and `Qwen3.6-35B-A3B-4bit`. [docs/models.md](docs/models.md) lists model IDs that are known to load. The Hub in the desktop app shows more models.

### How much memory will it use?

`GET /v1/models/status` reports `estimated_bytes` for each model. Higgs computes it from the weight files before loading. After loading, the endpoint also reports `measured_bytes`. To stay within a budget, set `[memory] guard` or `[memory] limit`. The values `safe`, `balanced` and `aggressive` reserve 20%, 8% and 2% of RAM for the system. If a model cannot fit, Higgs refuses the load with a 503 error and the code `insufficient_memory`. The message names the models that Higgs could not evict. See [Model lifecycle](docs/configuration.md#model-lifecycle).

### Which Macs and systems does it run on?

Higgs runs on Apple Silicon Macs only. Release builds target `aarch64-apple-darwin` and are built on macOS 15. The minimum macOS version is not documented. Higgs does not support Intel Macs, Linux or Windows.

### The download prompt does not appear, or the model is not found.

A download needs the `hf` CLI on your `PATH` and an interactive terminal. If you do not have both, run `hf download <model-id>`, then start Higgs again. A gated repository needs a Hugging Face token.

### It fails with an `mlx.metallib` error.

The server needs `mlx.metallib` next to the executable. Release archives and Homebrew installs include it. For a source build, Higgs restores the library from the Cargo build output. If you copy `target/release/higgs` to another directory, copy the library with it.

### My request returns `model_not_found`.

The `model` field must contain a name that the server knows. Use the configured `name`, one of its `aliases`, or the name that `GET /v1/models` shows. A name that matches no model and no route goes to `routing.default`. To support the names that other clients send, add `aliases` or a route with a `model` rewrite. The Claude Code recipe shows a route.

### `higgs shellenv` or `higgs exec` says there is no model or provider.

These commands read a config file. They do not read `serve` flags. Run `higgs init` and add a `[[models]]` entry, or pass `--config <file>`. Then start the server with `higgs serve` or `higgs start`, so that the command can reach it.

### A model with thinking returns an empty answer.

A reasoning model such as Qwen3 thinks before it answers. A small `max_tokens` value can run out before the answer starts. Use one of these fixes:

- Increase `max_tokens`.
- Decrease the reasoning budget. Use `reasoning.max_tokens` for one request or `thinking_budget` for one model.
- Turn thinking off for the request with `"reasoning": {"effort": "none"}`.

See [Thinking budget](docs/configuration.md#thinking-budget).

### Something is wrong with my config.

Run `higgs config check`. It validates the file and does not touch models. Then run `higgs doctor`. It also checks model paths, providers and memory settings.

### How does it compare with Ollama, LM Studio and oMLX?

- **oMLX** is another MLX server. It is written in Python. [docs/omlx-comparison.md](docs/omlx-comparison.md) compares it with Higgs directly, including the areas where oMLX is ahead.
- **Ollama.** In the earlier single-stream decode measurements, Higgs reached 448 tok/s on Llama-3.2-1B-4bit and 305 tok/s on Qwen3-1.7B-4bit. Ollama reached 305 and 183 tok/s with `Q4_K_M` quantization. On the Qwen3-30B-A3B MoE model at 8-bit, the results were close: 75 and 73 tok/s. See [docs/performance.md](docs/performance.md).
- **LM Studio.** Higgs has not been benchmarked against it. LM Studio is a closed-source desktop application. Higgs is an open-source server and CLI with a companion desktop app.

## Development

```bash
cargo clippy -p higgs                         # must be clean
cargo fmt -p higgs -- --check
cargo test -p higgs -- --test-threads=1       # the thread limit is required: tests share ports
cd apps/desktop && pnpm install && pnpm typecheck && pnpm build
```

[CONTRIBUTING.md](CONTRIBUTING.md) describes the project layout, the documentation rules for config and CLI changes, and release validation. [docs/configuration.md](docs/configuration.md#other-migration-notes) has notes for older `higgs start` flags.

More reference: [API reference](docs/api.md), [performance and Apple Silicon notes](docs/performance.md), [benchmarking](docs/benchmarking.md), [MCP and web tools](docs/mcp.md), [Responses API](docs/responses.md).

## License

MIT
