# Tool calling

Higgs turns the tool calls a local model writes into structured `tool_calls`
(Chat Completions), `function_call` items (Responses) and `tool_use` blocks
(Anthropic Messages), streaming or not. Each model family writes calls its own
way, so the parser is chosen per model.

## How the format is chosen

When a model loads, Higgs looks at its chat template, then its `model_type`
(from `config.json`), and picks a format. The template wins because it is what
the model was taught to answer in.

| The template mentions | Format |
| --- | --- |
| `<\|tool_call>` | Gemma 4 |
| `<start_function_call>` | FunctionGemma |
| `[TOOL_CALLS]` (and `[ARGS]`) | Mistral (name and arguments) |
| `[TOOL_CALLS]` only | Mistral (JSON array) |
| `minimax:tool_call` | MiniMax |
| `<arg_key>` | GLM |
| `<tool_call>` | Qwen |
| `<\|python_tag\|>`, or Llama 3 headers with a `"parameters"` convention | Llama |
| `` ```tool_code `` | Gemma (pythonic) |

When the template says nothing, `model_type` decides: `qwen*` and `minicpm*`
use Qwen, `glm*` GLM, `minimax*` MiniMax, `gemma4*` Gemma 4. Mistral, Llama and
Gemma 3 follow their `model_type` (`mistral*`/`mixtral*`/`ministral*`,
`llama*`, `gemma3*`) only when the model has no chat template at all, because
their output is only a call when the template taught it. Anything else is
**auto**: calls are recognized from the output text
when they start with a marker that belongs to one format only (`<tool_call>`,
`[TOOL_CALLS]`, `<|python_tag|>`, `<function=`, `<function `, a `tool_code`
fence, `<|tool_call>`, `<start_function_call>`, `<minimax:tool_call>`). Auto
never treats bare JSON or a bare `[call(...)]` list as a call, because without
knowing the model that would misread ordinary answers.

## Formats

| Model family | Format the model writes | Streaming | Forced `tool_choice` |
| --- | --- | --- | --- |
| Qwen 2.5 / Qwen3 / Hermes-style | `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` | yes | yes |
| Qwen3.5 / Qwen3.6, Qwen3-Coder | `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` | yes | yes (JSON envelope, see below) |
| MiniCPM5 | `<function name="..."><param name="...">...</param></function>` | yes | yes (JSON envelope, as for Qwen3.5) |
| Mistral, Mixtral, Ministral | `[TOOL_CALLS] [{"name": ..., "arguments": {...}}, ...]` | yes | yes |
| Mistral (newer templates) | `[TOOL_CALLS]name[ARGS]{...}` or `[TOOL_CALLS]name{...}` | yes | yes |
| Llama 3.1 / 3.2 | `{"name": ..., "parameters": {...}}`, optionally after `<\|python_tag\|>`, calls separated by `;` | yes | yes (one call) |
| Llama 3.1 | `<function=name>{...}</function>` | yes | yes (as the JSON form above) |
| Gemma 3 (prompted) | a ```` ```tool_code ```` fence, or a bare `[name(arg=value), ...]` list | yes | no |
| FunctionGemma | `<start_function_call>call:name{key:<escape>value<escape>}<end_function_call>` | yes | no |
| Gemma 4 | `<\|tool_call>call:name{key:<\|"\|>value<\|"\|>}<tool_call\|>` | yes | no |
| GLM 4.x | `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` | yes | no |
| MiniMax M2 | `<minimax:tool_call><invoke name="..."><parameter name="...">...</parameter></invoke></minimax:tool_call>` | yes | no |

Not supported: Kimi K2, Phi (no tool-call convention in its chat template),
DeepSeek, Gemma 4's parenthesized `call:name(key=value)` variant, and
positional arguments in pythonic calls.

### Behavior shared by every format

- Arguments are returned as a JSON string, as the OpenAI API requires.
- A call that does not parse stays in the message text; content is never
  dropped and a malformed call is never an error.
- Several calls in one response are returned where the format can express them.
- Streaming holds back anything that could be the start of a call until it is
  known not to be one, so half a call never appears in `content`.
- Formats whose values are plain text (Qwen XML, GLM, MiniMax) coerce them to
  the JSON types the tool's `parameters` declare.
- Bare formats (Llama JSON, pythonic lists) are only calls when they open the
  response (or follow another call) and name a declared tool. While such a
  response is being written, streaming withholds it until the call closes.
- Gemma 4's string delimiter `<|"|>` is a control token that some engines
  strip from decoded text; string values then arrive bare and parse as long as
  they contain no `,`, `}` or `]`.

## `tool_choice` and `parallel_tool_calls`

| Value | Effect |
| --- | --- |
| `"auto"` or absent | The model decides |
| `"none"` | Tools are not shown to the model and calls are not parsed |
| `"required"` (Anthropic `{"type": "any"}`) | The model must call at least one provided tool |
| `{"type": "function", "function": {"name": "X"}}` (Responses `{"type": "function", "name": "X"}`, Anthropic `{"type": "tool", "name": "X"}`) | The model must call `X`, with arguments matching its `parameters` schema |
| `parallel_tool_calls: false` (Anthropic `disable_parallel_tool_use: true`) | At most one call is returned; with a forced choice, exactly one |

Forcing uses constrained decoding. Higgs builds a regular expression for a
complete call in the model's own format, with the arguments constrained by each
tool's JSON schema (a union over the tools for `required`), so the output is
exactly what the parser extracts and the response finishes with `tool_calls`
(`tool_use` on Anthropic). A forced call replaces the model's reasoning, so
thinking is turned off for that request.

Forcing needs a format whose arguments are JSON: the Qwen format (any model
detected as Qwen, MiniCPM included), Mistral and Llama. For the others Higgs
returns a 400 naming the format rather than guessing. The forced text is always
the JSON form of the format: a Qwen3.5/3.6 or MiniCPM model, which is prompted
for an XML form, is forced into the `<tool_call>{"name": ..., "arguments":
...}</tool_call>` envelope, which the parser accepts but is not what the model
was trained to write; a Llama model is forced into the bare JSON object. Forcing cannot be combined with
`response_format`, `guided_regex` or `guided_json`; a request that tries gets a
400, as does naming a function that is not in `tools` or forcing a call with no
tools.

Notes on the constrained machinery:

- Models whose call marker is a special token (for example Mistral's
  `[TOOL_CALLS]`) cannot emit that token under a constraint, which only admits
  ordinary vocabulary tokens; they write the marker as text, which parses the
  same.
- The constraint engine can only continue a text that already matches the
  pattern through tokens of two or more bytes, so a second *consecutive* call
  after a complete one is reachable only when the vocabulary has such a token.
  Mistral's JSON array form, where the calls sit inside one array, is not
  affected. Treat parallel forced calls as best effort except there.
