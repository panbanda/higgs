# MCP tools

Higgs can act as a [Model Context Protocol](https://modelcontextprotocol.io) client. It connects to the servers you configure, lists their tools, and runs them on request, so a local model can use tools without every client implementing MCP itself.

Nothing happens unless you configure it: with no `[mcp.servers.*]` tables there are no child processes, no outbound connections, and the endpoints report empty lists.

## Configuring servers

Servers live under `[mcp.servers.<name>]` in the config file:

```toml
[mcp]
max_result_size = "256KB"          # larger tool results are truncated (default 256KB)

[mcp.servers.filesystem]           # stdio: Higgs starts the process
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Documents"]
env = { API_TOKEN = "${API_TOKEN}" }
cwd = "/Users/me"
timeout = "30s"
enabled = true

[mcp.servers.remote]               # Streamable HTTP: Higgs connects to a URL
url = "https://mcp.example.com/mcp"
headers = { Authorization = "Bearer ${MCP_TOKEN}" }
```

| Field | Applies to | Meaning |
|---|---|---|
| `transport` | both | `"stdio"` or `"http"` (Streamable HTTP). Inferred from `command` or `url` when omitted. `"sse"` (the legacy HTTP+SSE transport) is rejected. |
| `command`, `args` | stdio | Executable and arguments. Run directly, never through a shell. |
| `env` | stdio | Extra environment for the child, on top of the Higgs environment. |
| `cwd` | stdio | Working directory of the child. |
| `url` | http | Streamable HTTP endpoint (`http` or `https`). |
| `headers` | http | Extra request headers. |
| `timeout` | both | Bound on connecting, on listing tools, and on each tool call, as a duration string (`"30s"`). Default `"30s"`. |
| `enabled` | both | `false` keeps the entry but never starts it. Default `true`. |

Server names may contain letters, digits, `-` and `_`, but not `__` (the tool namespace separator), and `higgs` is reserved for built-in tools.

### Environment variables

`${VAR}` in an `env` or `headers` value is replaced with the Higgs process environment. A reference to an unset variable is an error at startup (and a failure in `higgs doctor`), never an empty string, so a typo cannot silently drop a credential. Write `$${` for a literal `${`. Disabled servers are not checked.

### Reusing a desktop-client config

`higgs serve --mcp-config mcp.json` loads servers from the common JSON format and merges them with the config file:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Documents"],
      "env": {"API_TOKEN": "${API_TOKEN}"}
    },
    "remote": {"type": "http", "url": "https://mcp.example.com/mcp"}
  }
}
```

`type` is accepted as an alias for `transport`, `disabled: true` and `enabled: false` both turn a server off, and keys other clients add (`autoApprove`, ...) are ignored. A server name defined in both the file and the config is an error. The flag also works with `higgs doctor`. `higgs start` (the background daemon) takes no flags, so put servers in the config file for daemon use.

## Lifecycle

- Servers connect in the background at startup. A slow or failing server never delays Higgs from serving; its state is `connecting`, then `ready` or `failed`.
- The handshake negotiates the protocol version (2025-06-18, 2025-03-26 and 2024-11-05 are supported) and `tools/list` follows pagination.
- `notifications/tools/list_changed` refreshes the tool list without reconnecting.
- A failed server is retried with exponential backoff (1 s doubling to 60 s). A tool call against a failed or crashed server cuts the wait short and reconnects immediately, so a crashed stdio server is restarted on its next use.
- On shutdown Higgs closes every server's stdin, waits briefly, then kills the process group, so launchers like `npx` leave no orphaned children.

## Endpoints

They use the same authentication as the other API endpoints (`api_key`) and never count as traffic in `/metrics`.

### `GET /v1/mcp/servers`

```json
{"servers": [
  {"name": "filesystem", "transport": "stdio", "state": "ready", "error": null,
   "tool_count": 14, "protocol_version": "2025-06-18",
   "server_info": {"name": "secure-filesystem-server", "version": "0.2.0"}},
  {"name": "remote", "transport": "http", "state": "failed",
   "error": "handshake failed: HTTP 401 Unauthorized: ...", "tool_count": 0,
   "protocol_version": null, "server_info": null}
]}
```

`state` is `connecting`, `ready`, `failed`, or `disabled`.

### `GET /v1/mcp/tools`

Tools of every ready server, namespaced `server__tool`, in the OpenAI function-tool shape plus the originating `server`:

```json
{"count": 1, "tools": [
  {"type": "function", "server": "filesystem",
   "function": {"name": "filesystem__read_file", "description": "...", "parameters": {"type": "object", "...": "..."}}}
]}
```

The list can be passed as `tools` of a chat request. Strict clients that reject unknown keys should drop `server` first.

### `POST /v1/mcp/execute`

```json
{"name": "filesystem__read_file", "arguments": {"path": "/Users/me/Documents/notes.txt"}}
```

(`tool_name` and `tool` are accepted for `name`.) The response:

```json
{"name": "filesystem__read_file", "server": "filesystem", "is_error": false,
 "text": "file contents", "content": [{"type": "text", "text": "file contents"}],
 "truncated": false}
```

`text` joins the text blocks; `content` carries every block as the server sent it, so images and resources come back structurally; `structured_content` is present when the server returned it. A tool that reports failure answers HTTP 200 with `"is_error": true`.

| Status | Meaning |
|---|---|
| 400 | Malformed body, or `arguments` is not an object (or the server rejected them as invalid) |
| 404 | Unknown tool |
| 502 | The server failed or answered with a protocol error |
| 503 | The server is `failed` or still `connecting` (the message carries its last error) |
| 504 | The tool call exceeded `timeout` |

## Security notes

- Stdio servers run local commands taken from your own config. That is by design; treat the config file (and `--mcp-config` files) as code.
- Request data never reaches a command line or the environment: tool names and arguments travel only as JSON-RPC parameters. The executable and arguments are fixed by the config and started without a shell.
- Anyone who can call `/v1/mcp/execute` can run every enabled tool. Set `api_key` before binding to anything but loopback, and give MCP servers only the access they need (for example, point the filesystem server at one directory).
- Results are capped at `max_result_size`; text is cut on a character boundary and `truncated` is set. Messages from a server are capped at 16 MiB.
- `timeout` bounds every call. A timed-out call is cancelled with `notifications/cancelled`.
- Credentials in `headers` and `env` come from `${VAR}`; they are not logged, and `higgs doctor` prints URLs without credentials or query strings.
- HTTP servers are contacted at the configured URL only; redirects are not followed.

## Server-side tool loop

Calling `/v1/mcp/execute` yourself works with any client. For clients that just want the model to use tools, add an `mcp` field to a chat completion (or Responses) request and Higgs runs the loop:

```json
{"model": "qwen", "messages": [{"role": "user", "content": "How big is /tmp/mcp-demo/hello.txt?"}],
 "mcp": {"enabled": true, "servers": ["fs"], "max_iterations": 5}}
```

| Field | Meaning |
|---|---|
| `enabled` | Default `true`; sending the object is the opt-in. `false` behaves as if the field were absent. |
| `servers` | Only offer these servers' tools (`higgs` selects the built-in web tools). Default: every available tool. An unknown name is a 400. |
| `max_iterations` | How many times tool calls may be executed, 1 to 25, default 5. |

Higgs adds the selected tools to the request's `tools`, generates, and when the model calls them runs the calls, appends the assistant message and one `tool` message per result to the conversation, and generates again. Calls within one round run in order. This repeats until the model answers without calling a server tool, or `max_iterations` rounds have run; if the model still wants a tool then, the pending call is returned as an ordinary `tool_calls` response with `"truncated": true` in the `mcp` block.

- Tools you supply yourself in `tools` are never run by Higgs. If the model calls one, the loop stops and the call is returned to you as usual. If one response mixes your tools and server tools, only your calls are returned. A client tool with the same name as a server tool is treated as yours.
- `tool_choice: "none"` turns the loop off.
- Tool failures (an error result, a timeout, an unreachable server, arguments that are not JSON) are given to the model as the tool's output, prefixed `Error:`, so it can recover. Images and other non-text results are summarized for the model as `[image content omitted: image/png]`.
- `usage` sums the tokens of every round, including re-reading the conversation each time.
- Requests without `mcp` take exactly the path they always did.

A non-streaming response gains an `mcp` object:

```json
"mcp": {"iterations": 1, "truncated": false,
        "tool_calls": [{"round": 1, "id": "call_0_...", "name": "fs__get_file_info", "arguments": {"path": "..."}, "is_error": false}]}
```

### Streaming

With `"stream": true`, assistant text and reasoning stream as they are produced in every round, under one response id and one initial `role` chunk. Calls to server tools never appear as `tool_calls` deltas. Their progress arrives as extra chunks whose `choices` is an empty array (the same shape as the usage chunk) and which carry a `higgs_mcp` field, so standard clients ignore them:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[],"higgs_mcp":{"event":"tool_call","round":1,"id":"call_0_...","name":"fs__get_file_info","arguments":{"path":"..."}}}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[],"higgs_mcp":{"event":"tool_result","round":1,"id":"call_0_...","name":"fs__get_file_info","is_error":false,"content":"size: 15"}}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[],"higgs_mcp":{"event":"summary","iterations":1,"truncated":false,"tool_calls":[...]}}
```

`tool_result.content` is cut at 1000 characters. The `summary` event comes last, before `[DONE]`, and only if a tool ran. With `stream_options.include_usage` a single usage chunk with the summed counts is sent. A client tool call (or, once `max_iterations` is exhausted, the pending server call) streams as a normal `tool_calls` delta with finish reason `tool_calls`. On the Responses route the same loop runs; intermediate calls are not represented in the output items, only the final answer is.

## Web search and fetch

Two endpoints for pages the model cannot reach itself. They are off by default and answer 404 (`web_disabled`) until enabled:

```toml
[web]
enabled = true
search_provider = "duckduckgo"           # or "searxng"
searxng_url = "http://localhost:8888"    # required for searxng
allow_private_addresses = false
expose_as_tools = false
timeout = "15s"
```

| Field | Meaning |
|---|---|
| `enabled` | Serve the endpoints. Default `false`. |
| `search_provider` | `duckduckgo` (keyless; reads the HTML endpoint, so it can break or be rate limited) or `searxng` (self-hosted, more reliable). |
| `searxng_url` | Base URL of a SearXNG instance with the `json` format enabled under `search.formats` in its `settings.yml`. |
| `allow_private_addresses` | Let `/v1/web/fetch` reach loopback, private and link-local addresses. Default `false`. |
| `expose_as_tools` | List `web_search` and `web_fetch` in `/v1/mcp/tools` under the built-in server `higgs` (as `higgs__web_search` and `higgs__web_fetch`) so the tool loop can use them. |
| `timeout` | Bound on one search or one fetch, redirects included, as a duration string. Default `"15s"`. |

The search provider URL comes from your config and is trusted; only URLs in `/v1/web/fetch` requests are subject to the address checks. Higgs does not use Brave or other keyed providers.

### `POST /v1/web/search`

```json
{"query": "rust async runtime", "max_results": 5}
```

returns `{"query": "...", "count": 5, "results": [{"title": "...", "url": "...", "snippet": "..."}]}`. `max_results` defaults to 5 and is capped at 20.

### `POST /v1/web/fetch`

```json
{"url": "https://example.com/article", "max_chars": 20000}
```

returns `{"url": "<final url>", "title": "...", "text": "...", "truncated": false, "status": 200, "content_type": "text/html"}`. HTML is reduced to readable text (scripts, styles, navigation and footers dropped, links kept as `text (url)`); plain text and JSON are returned as is. `max_chars` defaults to 20000 and is capped at 200000. Bodies larger than 2 MiB are cut off; pages are decoded as UTF-8.

| Status | Meaning |
|---|---|
| 400 | Missing or invalid `query`, malformed `url` |
| 403 | The scheme is not `http`/`https`, the URL embeds credentials, or an address (including after a redirect or DNS lookup) is not public |
| 404 | `[web]` is disabled |
| 415 | The response is not text (images, archives, ...) |
| 502 | The search provider or the page failed, or an upstream status was not 2xx |
| 504 | Timed out |

### Fetch protections

- Only `http` and `https`; URLs with embedded credentials are refused.
- The host is resolved by Higgs and **every** resolved address must be public. Loopback, private (RFC 1918), link-local (including the 169.254.169.254 metadata address), carrier-grade NAT, multicast, documentation and reserved ranges are refused, as are IPv6 forms that embed them (`::ffff:127.0.0.1`, NAT64, 6to4, unique-local). Numeric tricks such as `http://2130706433/` or `http://0x7f.1/` are normalized before the check.
- The connection is pinned to the addresses that were checked, so a name cannot resolve to something public for the check and something internal for the connection (DNS rebinding).
- Redirects are followed by Higgs, at most 5, and every hop gets the same checks. System proxy settings are ignored for fetches.
- Requests carry a generic `Mozilla/5.0 (compatible; higgs-fetch/<version>)` user agent and no cookies, credentials, or host information.

## `higgs doctor`

`higgs doctor` validates every server without starting it: unknown transport, missing `command` or `url`, a malformed URL, `timeout` of zero, unresolved `${VAR}` references (failures), and a `command` that is not on `PATH` or a `cwd` that is missing (warnings). Each valid server is reported with its transport. For `[web]` it fails on an unknown `search_provider`, a missing or invalid `searxng_url`, and an out-of-range timeout, and warns when `allow_private_addresses` is on, when `expose_as_tools` has no effect because web is disabled, or when `searxng_url` is set for another provider.

## Verifying against a real server

```bash
mkdir -p /tmp/mcp-demo && echo "hello from mcp" > /tmp/mcp-demo/hello.txt
cat > /tmp/mcp.json <<'JSON'
{"mcpServers": {"fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/mcp-demo"]}}}
JSON
higgs doctor --mcp-config /tmp/mcp.json -c ~/.config/higgs/config.toml
higgs serve --config ~/.config/higgs/config.toml --mcp-config /tmp/mcp.json

curl -s localhost:8000/v1/mcp/servers | jq
curl -s localhost:8000/v1/mcp/tools | jq '.tools[].function.name'
curl -s localhost:8000/v1/mcp/execute -H 'content-type: application/json' \
  -d '{"name": "fs__read_text_file", "arguments": {"path": "/tmp/mcp-demo/hello.txt"}}' | jq
```

Kill the `node` process while Higgs runs and call `execute` again to watch it restart.

For web search, run a SearXNG instance with `json` enabled (or use `duckduckgo`), set `[web] enabled = true`, then:

```bash
curl -s localhost:8000/v1/web/search -H 'content-type: application/json' -d '{"query": "mlx apple silicon"}' | jq
curl -s localhost:8000/v1/web/fetch -H 'content-type: application/json' -d '{"url": "https://example.com"}' | jq
curl -s localhost:8000/v1/web/fetch -H 'content-type: application/json' -d '{"url": "http://169.254.169.254/"}'   # 403
```
