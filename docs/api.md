# API reference

The endpoints Higgs serves, with the request fields and response details that are specific to Higgs. Requests for models routed to a remote provider are forwarded unchanged. Options that apply to locally served models are also covered in [configuration.md](configuration.md#api-request-options).

| Endpoint | Compatibility | Notes |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | Streaming, tools, structured output, vision (LLaVA) |
| `POST /v1/completions` | OpenAI | Also serves models without a chat template |
| `POST /v1/responses`, `GET`/`DELETE /v1/responses/{id}` | OpenAI Responses | [responses.md](responses.md) |
| `POST /v1/embeddings` | OpenAI | |
| `GET /v1/models` | OpenAI | Every configured model with a `loaded` flag |
| `POST /v1/messages`, `POST /v1/messages/count_tokens` | Anthropic | |
| `POST /v1/tokenize`, `POST /v1/detokenize` | llama.cpp style | Local models only |
| `POST /v1/models/{id}/load`, `POST /v1/models/{id}/unload`, `GET /v1/models/status` | Higgs | [Model lifecycle](configuration.md#model-lifecycle) |
| `GET /v1/mcp/servers`, `GET /v1/mcp/tools`, `POST /v1/mcp/execute` | Higgs | [mcp.md](mcp.md) |
| `POST /v1/web/search`, `POST /v1/web/fetch` | Higgs | Off by default; [mcp.md](mcp.md#web-search-and-fetch) |
| `GET /metrics`, `GET /v1/system`, `GET /health` | Higgs | Dashboard and control endpoints |

## Details


- OpenAI: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`
  - Streaming requests may set `"return_progress": true` to receive
    llama.cpp-compatible `prompt_progress` chunks (`{total, cache, processed,
    time_ms}`) during chunked prefill.
  - Streaming requests on both chat and completions may set
    `"stream_options": {"include_usage": true}` to receive a final chunk with
    an empty `choices` array and a `usage` object.
  - `usage` carries `prompt_tokens`, `completion_tokens`, `total_tokens`, plus
    only the fields that were measured for that request:
    `prompt_tokens_details.cached_tokens` (prompt tokens served from the prefix
    cache), `completion_tokens_details.reasoning_tokens` (chat only, when the
    response had reasoning content), and oMLX-compatible timings in seconds:
    `time_to_first_token`, `prompt_eval_duration`, `generation_duration`,
    `total_time`, `prompt_tokens_per_second`, `generation_tokens_per_second`.
    See [Usage fields](configuration.md#usage-fields) for exact definitions.
  - Constrained output on `/v1/chat/completions`: `response_format`
    (`json_object`, `json_schema`, or `{"type": "regex", "regex": "..."}`), or
    the vLLM-style `guided_regex` (string) and `guided_json` (schema object).
    At most one may be set; more than one, an invalid regex, or a schema the
    grammar compiler rejects returns 400 with the reason.
  - `reasoning.max_tokens` sets the reasoning-token budget for a request;
    Anthropic `thinking.budget_tokens` does the same on `/v1/messages`.
    Defaults come from `[[models]].thinking_budget`, then `[defaults].thinking_budget`,
    then 256. See [Thinking budget](configuration.md#thinking-budget).
  - `tool_choice`: `"auto"` (default), `"none"` (tools are not shown to the
    model and tool calls are not parsed), `"required"`, and
    `{"type": "function", "function": {"name": "..."}}`. The last two force a
    call with constrained decoding for models whose tool-call format allows
    it, and return 400 for the others. `parallel_tool_calls: false` limits a
    response to one call. The same fields work on `/v1/responses` and, as
    `tool_choice` `any`/`tool`/`none` and `disable_parallel_tool_use`, on
    `/v1/messages`. See [Tool calling](tool-calling.md) for the format
    each model family uses and which ones support forcing.
  - `seed` is accepted for client compatibility but does not make sampling
    reproducible yet.
  - Unrecognized request fields are ignored.
- Tokenizer: `POST /v1/tokenize` and `POST /v1/detokenize` (also at `/tokenize`
  and `/detokenize`), local models only.
  - `{"model": "m", "text": "hello"}` or `{"model": "m", "messages": [...]}`
    (rendered through the chat template, optional `tools`) returns
    `{"tokens": [...], "count": n}`. Text input does not add special tokens
    unless `"add_special_tokens": true`.
  - `{"model": "m", "tokens": [...]}` to `/v1/detokenize` returns
    `{"text": "..."}`; optional `"skip_special_tokens": true`. Unknown token
    ids are rejected with 400, as are models routed to a remote provider.
- Anthropic: `/v1/messages`, `/v1/messages/count_tokens`
  - `usage.cache_read_input_tokens` reports prompt tokens served from the
    prefix cache when known. `usage.input_tokens` still counts the whole
    prompt, cached tokens included.
- Model lifecycle: `POST /v1/models/{id}/load`, `POST /v1/models/{id}/unload[?force=true]`,
  `GET /v1/models/status` (state, pinned, in-flight requests, last use, load time,
  estimated and measured memory per configured model); see
  [docs/configuration.md](configuration.md#model-lifecycle). `/v1/models` lists every
  configured model with a `loaded` flag.
- OpenAI Responses: `POST /v1/responses` (streaming and non-streaming),
  `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}`. Local models and
  OpenAI-format providers only; supported fields, unsupported features, and
  store limits are in [docs/responses.md](responses.md).
- MCP: `GET /v1/mcp/servers` (state, last error, tool count, protocol version),
  `GET /v1/mcp/tools` (every ready server's tools as OpenAI function tools),
  `POST /v1/mcp/execute` (run `server__tool` with JSON arguments). Servers come from
  `[mcp.servers.<name>]` in the config or `higgs serve --mcp-config mcp.json`
  (the common `{"mcpServers": {...}}` file format). Config reference, security
  notes, and verification steps: [docs/mcp.md](mcp.md). Chat completion and
  Responses requests can set `"mcp": {"enabled": true, "servers": [...], "max_iterations": 5}`
  to have Higgs run those tools itself and continue the generation, streaming included
  ([tool loop](mcp.md#server-side-tool-loop)).
- Web: `POST /v1/web/search` (`query`, `max_results`) and `POST /v1/web/fetch`
  (`url`, `max_chars`: readable text, title, final URL). Disabled and answering 404
  unless `[web] enabled = true`; fetch blocks non-http(s) schemes and private,
  loopback and link-local addresses unless `allow_private_addresses = true`.
  See [docs/mcp.md](mcp.md#web-search-and-fetch).
- Metrics: `/metrics` (window totals, latency and time-to-first-token
  percentiles, aggregate tokens/s, per-model and per-provider groups with
  TTFT, tokens/s, and prefix-cache hits). Local streaming requests record
  `ttft_ms` and `cached_tokens` in the metrics JSONL log.
- System: `/v1/system` (version, pid, uptime, unified memory, process RSS,
  MLX active/peak/cache memory, every configured model with its lifecycle state,
  engine kind and profile)
- Health: `/health`

`/health`, `/metrics`, `/v1/models`, `/v1/models/status`, `/v1/system`, the
model load and unload endpoints, and `/v1/mcp/*`, and `/v1/web/*` are dashboard and control endpoints and never count as
traffic in metrics. The read-only ones never load a model.
