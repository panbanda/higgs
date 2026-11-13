# Responses API

Higgs serves the OpenAI Responses API alongside Chat Completions and the
Anthropic Messages API:

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/responses` | Create a response, streaming or not |
| `GET /v1/responses/{id}` | Fetch a stored response |
| `DELETE /v1/responses/{id}` | Delete a stored response |

Local models run through the same generation path as `/v1/chat/completions`
(chat template, prefix cache, tool-call parsing, reasoning extraction,
structured output) and are recorded in `/metrics` like any other request.

A model that routes to a provider with `format = "openai"` has the request
proxied unchanged to that provider's `/v1/responses` (the `model` field is
rewritten if the route sets one). Responses for those requests are stored and
served by the provider, not by Higgs. A model that routes to an
`format = "anthropic"` provider is rejected with a 400.

## Supported request fields

| Field | Notes |
| --- | --- |
| `model` | Resolved through the normal routing table |
| `input` | A string, or an array of items (below) |
| `instructions` | Becomes the leading system message. Not carried across `previous_response_id`, as in the OpenAI API |
| `max_output_tokens` | Falls back to `server.max_tokens` |
| `temperature`, `top_p` | |
| `tools` | `{"type": "function", "name", "description", "parameters"}` only |
| `tool_choice` | `auto` (default), `none` (removes the tools from the prompt), `required`, or `{"type": "function", "name"}`; the last two force a call as described in [Tool calling](tool-calling.md) |
| `parallel_tool_calls` | Echoed; `false` limits the response to one call |
| `reasoning` | `{"effort": ...}` with the same semantics as on chat completions |
| `text.format` | `text`, `json_object`, or `json_schema` (`name`, `schema`), mapped to constrained decoding |
| `stream` | Server-sent events (below) |
| `store` | Defaults to `true` |
| `previous_response_id` | Prepends the earlier turn's input and output |
| `metadata` | Echoed back |

Input items:

- Messages with role `user`, `assistant`, `system`, or `developer` (`developer`
  is sent to the model as `system`). Content is a string or an array of
  `input_text` / `output_text` parts, plus `input_image` with a base64 data URI
  when the model is a vision model.
- `function_call` and `function_call_output` items. Consecutive calls become
  one assistant message with several tool calls.
- `reasoning` items are accepted so clients can replay a previous output
  verbatim, but they are not shown to the model.

Fields that are not listed (`user`, `truncation`, `include`, `service_tier`,
and similar) are ignored.

## Rejected with a 400

Higgs returns an `invalid_request_error` naming the offending thing instead of
dropping it:

- Tools other than `function` (`web_search`, `file_search`, `code_interpreter`,
  `computer_use`, `mcp`, ...).
- Input item types other than the ones above (`item_reference`,
  `computer_call`, `web_search_call`, ...).
- Content part types other than `input_text`, `output_text`, `input_image`
  (`input_file`, `refusal`, ...); `input_image` on a model that is not a
  vision model; images that are not base64 data URIs (`file_id` or `http(s)`
  URLs).
- `tool_choice` of any other type (`allowed_tools`, hosted tools, ...), and
  forcing a call for a model whose tool-call format cannot be constrained
  (see [Tool calling](tool-calling.md)).
- `background: true` and `conversation`.
- `text.format` types other than the three above, or `json_schema` without a
  schema.
- Missing, empty, or malformed `input`.

An unknown `previous_response_id` returns 404 with code
`previous_response_not_found`.

## Response object

`output` holds, in order, a `reasoning` item (when the model produced
reasoning text), `message` items with `output_text` parts, and `function_call`
items with `call_id`, `name` and `arguments`. `output_text` concatenates the
message text. Reasoning is returned as a `reasoning_text` content part, not as
a summary.

`usage` reports `input_tokens`, `output_tokens` and `total_tokens`.
`input_tokens_details.cached_tokens` is the measured prefix-cache hit, streamed
or not, and is omitted when the engine did not report one.
`output_tokens_details.reasoning_tokens` appears when the response had a
reasoning item; it is a re-encoding of the reasoning text, so it can differ by
a token or two from the tokens sampled.

A response stopped by the token limit has `status: "incomplete"` and
`incomplete_details: {"reason": "max_output_tokens"}`.

## Streaming

Each frame carries an `event:` line equal to the `type` field and a
monotonically increasing `sequence_number` starting at 0:

1. `response.created`, `response.in_progress`
2. Per output item: `response.output_item.added`, then
   `response.content_part.added`, `response.output_text.delta` (or
   `response.reasoning_text.delta`) repeated, `response.output_text.done` (or
   `response.reasoning_text.done`), `response.content_part.done`, and
   `response.output_item.done`. A function call is emitted whole:
   `response.output_item.added`, `response.function_call_arguments.delta`,
   `response.function_call_arguments.done`, `response.output_item.done`.
3. One of `response.completed`, `response.incomplete`, or `response.failed`,
   carrying the final response object. The stream ends there; there is no
   `[DONE]` sentinel.

Tool-call arguments arrive as a single delta because the tool-call parser
needs the complete call before it can emit one.

## Stored responses

With `store` true (the default) a finished response and the conversation that
produced it are kept in memory. Failed responses are not stored.

- At most 256 responses are kept; the least recently used is evicted first.
- A response expires one hour after it was stored.
- The store is per server process and is lost on restart.

These limits are constants in `crates/higgs/src/responses/store.rs`, not
configuration. A response can only be named in `previous_response_id` if it
was stored, so leave `store` true on turns you want to continue from.

## Server-side MCP tools

Higgs extension: `"mcp": {"enabled": true, "servers": [...], "max_iterations": 5}`
makes Higgs run tools from configured MCP servers (and the built-in web tools)
itself and continue the generation with the results, as on chat completions.
Only the final answer appears in `output`; `function_call` items are produced
only for tools you supplied in `tools`. Details: [docs/mcp.md](mcp.md#server-side-tool-loop).
