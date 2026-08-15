use minijinja::value::Kwargs;
use minijinja::{Environment, Value};
use serde::Serialize;

use crate::error::EngineError;

/// A chat message for template rendering.
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Exact model-emitted `<tool_call>…</tool_call>` text, when it is safe
    /// to replay instead of the template's normalized serialization.
    ///
    /// This is intentionally not exposed to the Jinja template. After normal
    /// rendering, [`ChatTemplateRenderer`] replaces only a matching tool-call
    /// wrapper region. Thus replay is an optimization: a template without the
    /// expected wrapper remains byte-for-byte on its normal rendering path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_tool_call_text: Option<Vec<Option<String>>>,
}

/// Upper bound on template-engine instructions per render. Generous enough
/// for complex HF chat templates over long conversations, but stops a
/// malicious template from looping forever.
const TEMPLATE_FUEL: u64 = 5_000_000;

/// Renders chat messages using a Jinja2 template (`HuggingFace` format).
pub struct ChatTemplateRenderer {
    env: Environment<'static>,
    /// Special tokens loaded from `tokenizer_config.json` for template rendering.
    bos_token: String,
    eos_token: String,
}

impl ChatTemplateRenderer {
    /// Create a renderer from a Jinja2 template string.
    pub fn new<S: Into<String>>(template_source: S) -> Result<Self, EngineError> {
        let mut env = Environment::new();
        // Templates come from model directories (tokenizer_config.json /
        // chat_template.jinja), which are third-party content; bound execution
        // so a hostile template cannot loop forever.
        env.set_fuel(Some(TEMPLATE_FUEL));
        env.add_filter("tojson", tojson_filter);
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("chat".to_owned(), template_source.into())
            .map_err(|e| EngineError::Template(e.to_string()))?;
        Ok(Self {
            env,
            bos_token: String::new(),
            eos_token: String::new(),
        })
    }

    /// Load template from a model directory (`chat_template.jinja` or `tokenizer_config.json`).
    pub fn from_model_dir(model_dir: &std::path::Path) -> Result<Self, EngineError> {
        Self::try_from_model_dir(model_dir)?.ok_or_else(|| {
            EngineError::Template("No chat template found in model directory".to_owned())
        })
    }

    /// Like [`Self::from_model_dir`] but returns `Ok(None)` when no template is present,
    /// rather than an error. Parse/IO failures still propagate as `Err`.
    pub fn try_from_model_dir(model_dir: &std::path::Path) -> Result<Option<Self>, EngineError> {
        // Load tokenizer_config.json for special tokens (needed by both paths)
        let config_path = model_dir.join("tokenizer_config.json");
        let config: Option<serde_json::Value> = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| EngineError::Template(format!("Failed to read config: {e}")))?;
            Some(
                serde_json::from_str(&config_str)
                    .map_err(|e| EngineError::Template(format!("Invalid JSON: {e}")))?,
            )
        } else {
            None
        };

        let extract_token = |cfg: &serde_json::Value, key: &str| -> String {
            cfg.get(key)
                .and_then(|v| {
                    // Token can be a string or {"content": "..."} object
                    v.as_str().map(ToOwned::to_owned).or_else(|| {
                        v.get("content")
                            .and_then(|c| c.as_str())
                            .map(ToOwned::to_owned)
                    })
                })
                .unwrap_or_default()
        };

        let set_tokens = |renderer: &mut Self| {
            if let Some(ref cfg) = config {
                renderer.bos_token = extract_token(cfg, "bos_token");
                renderer.eos_token = extract_token(cfg, "eos_token");
            }
        };

        // Prefer standalone chat_template.jinja
        let jinja_path = model_dir.join("chat_template.jinja");
        if jinja_path.exists() {
            let template = std::fs::read_to_string(&jinja_path)
                .map_err(|e| EngineError::Template(format!("Failed to read template: {e}")))?;
            let mut renderer = Self::new(&template)?;
            set_tokens(&mut renderer);
            return Ok(Some(renderer));
        }

        // Fall back to tokenizer_config.json
        if let Some(ref cfg) = config {
            if let Some(ct) = cfg.get("chat_template") {
                // String template
                if let Some(template) = ct.as_str() {
                    let mut renderer = Self::new(template)?;
                    set_tokens(&mut renderer);
                    return Ok(Some(renderer));
                }
                // Array of {name, template} objects -- use "default" or first entry
                if let Some(arr) = ct.as_array() {
                    let found = arr
                        .iter()
                        .find(|v| v.get("name").and_then(|n| n.as_str()) == Some("default"))
                        .or_else(|| arr.first())
                        .and_then(|v| v.get("template"))
                        .and_then(|v| v.as_str());
                    if let Some(template) = found {
                        let mut renderer = Self::new(template)?;
                        set_tokens(&mut renderer);
                        return Ok(Some(renderer));
                    }
                }
                tracing::warn!("chat_template field present but not a string or valid array");
            }
        }

        Ok(None)
    }

    /// Apply the chat template to messages, returning the formatted prompt string.
    pub fn apply(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
    ) -> Result<String, EngineError> {
        self.apply_with_thinking(messages, tools, add_generation_prompt, false)
    }

    /// Apply the chat template with explicit `enable_thinking` control.
    pub fn apply_with_thinking(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String, EngineError> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| EngineError::Template(e.to_string()))?;

        let bos = &self.bos_token;
        let eos = &self.eos_token;
        let context = tools.map_or_else(
            || {
                minijinja::context! {
                    messages => messages,
                    add_generation_prompt => add_generation_prompt,
                    enable_thinking => enable_thinking,
                    bos_token => bos,
                    eos_token => eos,
                }
            },
            |tool_list| {
                minijinja::context! {
                    messages => messages,
                    tools => tool_list,
                    add_generation_prompt => add_generation_prompt,
                    enable_thinking => enable_thinking,
                    bos_token => bos,
                    eos_token => eos,
                }
            },
        );

        let rendered = tmpl
            .render(context)
            .map_err(|e| EngineError::Template(e.to_string()))?;
        Ok(replay_tool_call_text(rendered, messages))
    }
}

/// Replace matching normalized Qwen-style tool-call regions with the exact
/// bytes the model previously emitted. The replacement is deliberately narrow:
/// a region is replaced only when its parsed function name and arguments match
/// the stored raw call. This ensures literal wrapper text in user or tool
/// content remains untouched.
fn replay_tool_call_text(mut rendered: String, messages: &[ChatMessage]) -> String {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut search_from = 0;

    for message in messages {
        let call_count = message.tool_calls.as_ref().map_or(0, Vec::len);
        for call_index in 0..call_count {
            let replay_text = message
                .raw_tool_call_text
                .as_ref()
                .and_then(|calls| calls.get(call_index))
                .and_then(Option::as_deref);
            let Some(raw) = replay_text else {
                continue;
            };
            let Some(raw_identity) = tool_call_identity(raw) else {
                continue;
            };

            let mut candidate_from = search_from;
            while let Some((start, end)) =
                find_tool_call_region(&rendered, candidate_from, OPEN, CLOSE)
            {
                if tool_call_identity(&rendered[start..end]) == Some(raw_identity.clone()) {
                    rendered.replace_range(start..end, raw);
                    search_from = start.saturating_add(raw.len());
                    break;
                }
                candidate_from = end;
            }
        }
    }
    rendered
}

fn find_tool_call_region(
    rendered: &str,
    search_from: usize,
    open: &str,
    close: &str,
) -> Option<(usize, usize)> {
    let start_relative = rendered.get(search_from..)?.find(open)?;
    let start = search_from + start_relative;
    let end_relative = rendered.get(start..)?.find(close)?;
    Some((start, start + end_relative + close.len()))
}

fn tool_call_identity(wrapper: &str) -> Option<(String, serde_json::Value)> {
    let json = wrapper
        .strip_prefix("<tool_call>")?
        .strip_suffix("</tool_call>")?
        .trim();
    let call: serde_json::Value = serde_json::from_str(json).ok()?;
    let object = call.as_object()?;
    let function_object = object
        .get("function")
        .and_then(serde_json::Value::as_object);
    let name = object
        .get("name")
        .or_else(|| function_object.and_then(|function| function.get("name")))
        .and_then(serde_json::Value::as_str)?;
    let argument_value = object
        .get("arguments")
        .or_else(|| function_object.and_then(|function| function.get("arguments")))?;
    let parsed_arguments = match argument_value.as_str() {
        Some(argument_text) => serde_json::from_str(argument_text).ok()?,
        None => argument_value.clone(),
    };
    Some((name.to_owned(), parsed_arguments))
}

/// Normalise a tool-call JSON object so Qwen-Hermes-style chat templates
/// can render it without crashing on `tool_call.arguments|items`.
///
/// Two transformations are applied in place:
///
/// 1. **Flatten `function.{name,arguments}` to top level.** The `OpenAI`
///    request shape nests them under `function`; Qwen's
///    `chat_template.jinja` references `tool_call.name` and
///    `tool_call.arguments` directly. After this call, both shapes are
///    accessible.
/// 2. **Coerce `arguments` to a mapping.** `OpenAI` sends
///    `function.arguments` as a JSON-encoded string, but Qwen's template
///    iterates it via `|items`. A string that parses to a JSON object is
///    replaced by that object; anything that does not resolve to an object
///    (unparseable strings, or JSON that isn't an object) is coerced to an
///    empty object `{}` by [`normalize_arguments_value`] so the template
///    can't raise `cannot convert value into pairs`. The original string
///    does NOT survive when it isn't object-shaped.
///
/// Other fields (`id`, `type`, …) are preserved unchanged. Callers that
/// already supply the flat shape pay only the cost of a `serde_json::Value`
/// match.
pub fn normalize_tool_call_for_template(tc: &mut serde_json::Value) {
    let Some(obj) = tc.as_object_mut() else {
        return;
    };

    // Promote `function.name` / `function.arguments` to the top level.
    if let Some(function) = obj.get("function").cloned() {
        if let Some(func_obj) = function.as_object() {
            if !obj.contains_key("name") {
                if let Some(name) = func_obj.get("name") {
                    obj.insert("name".to_owned(), name.clone());
                }
            }
            if !obj.contains_key("arguments") {
                if let Some(arguments) = func_obj.get("arguments") {
                    obj.insert("arguments".to_owned(), arguments.clone());
                }
            }
        }
    }

    // Normalize the top-level `arguments` (used by Qwen-flat templates).
    if let Some(args) = obj.get_mut("arguments") {
        normalize_arguments_value(args);
    }

    // Normalize the nested `function.arguments` too. Qwen's
    // `chat_template.jinja` lines 107-108 rebind `tool_call` to
    // `tool_call.function` when present, so if we only normalised the
    // top-level copy the template still walks into a string and crashes
    // at `|items`. Templates that don't rebind are unaffected.
    if let Some(function) = obj.get_mut("function") {
        if let Some(func_obj) = function.as_object_mut() {
            if let Some(nested_args) = func_obj.get_mut("arguments") {
                normalize_arguments_value(nested_args);
            }
        }
    }
}

/// Coerce a `tool_call.arguments` (or `function.arguments`) value into
/// the mapping shape that `chat_template.jinja:120` requires.
///
/// 1. If it's a JSON-string, try to parse it back to a `Value`.
/// 2. If the result still isn't an object (null, bool, number, array,
///    or unparseable string), coerce to an empty object so the
///    template's `|items` doesn't raise. A warn is logged so the
///    pathological shape is visible.
fn normalize_arguments_value(args: &mut serde_json::Value) {
    if let Some(s) = args.as_str() {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
            *args = parsed;
        }
    }
    if args.is_object() {
        return;
    }
    let shape = match args {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        // `is_object()` already returned for this case above.
        serde_json::Value::Object(_) => "object",
    };
    tracing::warn!(
        shape,
        "tool_call arguments not a mapping after normalization; coercing to empty object so the chat template can render"
    );
    *args = serde_json::Value::Object(serde_json::Map::new());
}

/// `tojson` filter. `_kwargs` absorbs keyword arguments HF chat templates pass
/// — notably `tojson(ensure_ascii=false)` (e.g. `MiniCPM5`). `serde_json` already
/// emits UTF-8, which matches `ensure_ascii=false`, so the kwarg is accepted and
/// ignored rather than aborting the render with "too many arguments".
#[allow(clippy::needless_pass_by_value)]
fn tojson_filter(value: Value, _kwargs: Kwargs) -> Result<String, minijinja::Error> {
    let serialized = serde_json::to_string(&value).map_err(|e| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "JSON serialization failed",
        )
        .with_source(e)
    })?;
    Ok(serialized)
}

#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::shadow_unrelated,
    clippy::shadow_reuse
)]
#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_owned(),
            content: content.to_owned(),
            tool_calls: None,
            raw_tool_call_text: None,
        }
    }

    /// Create a minijinja environment with the tojson filter and return the
    /// compiled template for `{{ value | tojson }}`.
    fn tojson_env(template_source: &str) -> minijinja::Environment<'static> {
        let mut env = Environment::new();
        env.add_filter("tojson", tojson_filter);
        env.add_template_owned("test".to_owned(), template_source.to_owned())
            .unwrap();
        env
    }

    const CHATML_TEMPLATE: &str = r"{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}";

    const TOJSON_TEMPLATE: &str = r"{{ value | tojson }}";

    #[test]
    fn test_simple_chatml_template() {
        let renderer = ChatTemplateRenderer::new(CHATML_TEMPLATE).unwrap();
        let messages = vec![msg("system", "You are helpful."), msg("user", "Hello!")];

        let result = renderer.apply(&messages, None, true).unwrap();
        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains("You are helpful."));
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello!"));
        assert!(result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_tojson_filter() {
        let env = tojson_env(TOJSON_TEMPLATE);
        let tmpl = env.get_template("test").unwrap();
        let result = tmpl
            .render(minijinja::context! { value => "hello" })
            .unwrap();
        assert_eq!(result, r#""hello""#);
    }

    /// HF templates (e.g. `MiniCPM5` at `chat:6`) call `tojson(ensure_ascii=…)`.
    /// The filter must accept the kwarg instead of aborting with "too many
    /// arguments"; the value is ignored since `serde_json` emits UTF-8.
    #[test]
    fn test_tojson_filter_accepts_ensure_ascii_kwarg() {
        let env = tojson_env(r"{{ value | tojson(ensure_ascii=false) }}");
        let tmpl = env.get_template("test").unwrap();
        let result = tmpl
            .render(minijinja::context! { value => "café" })
            .unwrap();
        // UTF-8 preserved (not \u-escaped), and the call did not error.
        assert_eq!(result, "\"café\"");
    }

    #[test]
    fn test_invalid_template_syntax_returns_error() {
        assert!(ChatTemplateRenderer::new("{%- invalid syntax %}}}").is_err());
    }

    #[test]
    fn test_apply_without_generation_prompt() {
        let renderer = ChatTemplateRenderer::new(CHATML_TEMPLATE).unwrap();
        let result = renderer
            .apply(&[msg("user", "Hello!")], None, false)
            .unwrap();
        assert!(!result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_apply_empty_messages() {
        let template = r"{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor %}";
        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let result = renderer.apply(&[], None, false).unwrap();
        assert!(!result.contains("<|im_start|>"));
    }

    #[test]
    fn test_apply_with_tools() {
        let template = r"{%- for message in messages %}
{{ message.content }}
{%- endfor %}
{%- if tools %}
TOOLS:{{ tools | length }}
{%- endif %}";

        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let tools = vec![serde_json::json!({"type": "function", "function": {"name": "test"}})];
        let result = renderer
            .apply(&[msg("user", "Hi")], Some(&tools), false)
            .unwrap();
        assert!(result.contains("TOOLS:1"));
    }

    #[test]
    fn test_from_model_dir_no_template_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(ChatTemplateRenderer::from_model_dir(dir.path()).is_err());
    }

    #[test]
    fn test_from_model_dir_tokenizer_config_no_chat_template_field() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"model_type": "qwen2"}"#,
        )
        .unwrap();
        assert!(ChatTemplateRenderer::from_model_dir(dir.path()).is_err());
    }

    #[test]
    fn test_from_model_dir_standalone_jinja_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            r"{%- for message in messages %}{{ message.content }}{%- endfor %}",
        )
        .unwrap();
        let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
        let result = renderer
            .apply(&[msg("user", "hello")], None, false)
            .unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_from_model_dir_jinja_takes_priority_over_tokenizer_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "JINJA:{{ messages[0].content }}",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "CONFIG:{{ messages[0].content }}"}"#,
        )
        .unwrap();
        let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
        let result = renderer.apply(&[msg("user", "test")], None, false).unwrap();
        assert!(result.starts_with("JINJA:"));
    }

    #[test]
    fn test_from_model_dir_fallback_to_tokenizer_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "{%- for message in messages %}{{ message.content }}{%- endfor %}"}"#,
        )
        .unwrap();
        let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
        let result = renderer
            .apply(&[msg("user", "from_config")], None, false)
            .unwrap();
        assert_eq!(result, "from_config");
    }

    #[test]
    fn test_from_model_dir_malformed_tokenizer_config_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            "this is not valid json {{{",
        )
        .unwrap();
        match ChatTemplateRenderer::from_model_dir(dir.path()) {
            Err(e) => assert!(e.to_string().contains("Invalid JSON")),
            Ok(_) => panic!("Expected error for malformed JSON"),
        }
    }

    #[test]
    fn test_apply_with_assistant_role() {
        let template = r"{%- for message in messages %}<|{{ message.role }}|>{{ message.content }}{%- endfor %}";
        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let messages = vec![msg("user", "What is 2+2?"), msg("assistant", "4")];
        let result = renderer.apply(&messages, None, false).unwrap();
        assert!(result.contains("<|assistant|>4"));
    }

    #[test]
    fn test_apply_with_tool_calls_field() {
        let template = r"{%- for message in messages %}{{ message.role }}:{{ message.content }}{%- if message.tool_calls %} [tools]{%- endif %}{%- endfor %}";
        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let messages = vec![ChatMessage {
            role: "assistant".to_owned(),
            content: "calling tool".to_owned(),
            tool_calls: Some(vec![serde_json::json!({
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
            })]),
            raw_tool_call_text: None,
        }];
        let result = renderer.apply(&messages, None, false).unwrap();
        assert!(result.contains("[tools]"));
    }

    #[test]
    fn replayed_tool_call_is_rendered_verbatim() {
        let template = "{% for message in messages %}{{ message.role }}: {% for tool_call in message.tool_calls %}<tool_call>\n{{ tool_call | tojson }}\n</tool_call>{% endfor %}{% endfor %}";
        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let messages = vec![ChatMessage {
            role: "assistant".to_owned(),
            content: String::new(),
            tool_calls: Some(vec![serde_json::json!({
                "name": "weather",
                "arguments": {"city": "Denver"}
            })]),
            raw_tool_call_text: Some(vec![Some(
                "<tool_call>\n{\"arguments\":{\"city\":\"Denver\"},\"name\":\"weather\"}\n</tool_call>"
                    .to_owned(),
            )]),
        }];

        assert_eq!(
            renderer.apply(&messages, None, false).unwrap(),
            "assistant: <tool_call>\n{\"arguments\":{\"city\":\"Denver\"},\"name\":\"weather\"}\n</tool_call>"
        );
    }

    #[test]
    fn replay_skips_user_tool_call_text_until_it_finds_the_matching_call() {
        let template = "{% for message in messages %}{{ message.role }}: {{ message.content }}{% for tool_call in message.tool_calls %}<tool_call>{{ tool_call | tojson }}</tool_call>{% endfor %}\n{% endfor %}";
        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let decoy = r#"<tool_call>{"name": "decoy", "arguments": {"value": 1}}</tool_call>"#;
        let raw =
            r#"<tool_call>{"name": "weather", "arguments": { "city": "Denver" }}</tool_call>"#;
        let messages = vec![
            msg("user", &format!("Please do not run this example: {decoy}")),
            ChatMessage {
                role: "assistant".to_owned(),
                content: String::new(),
                tool_calls: Some(vec![serde_json::json!({
                    "name": "weather",
                    "arguments": {"city": "Denver"}
                })]),
                raw_tool_call_text: Some(vec![Some(raw.to_owned())]),
            },
        ];

        let result = renderer.apply(&messages, None, false).unwrap();

        assert!(result.contains(decoy));
        assert!(result.contains(raw));
        assert!(!result.contains(
            r#"<tool_call>{"arguments":{"city":"Denver"},"name":"weather"}</tool_call>"#
        ));
    }

    #[test]
    fn replay_keeps_turn_one_prefix_for_qwen_style_template() {
        let template = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}{% for tool_call in message.tool_calls %}<tool_call>\n{{ tool_call | tojson }}\n</tool_call>{% endfor %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let turn_one = vec![ChatMessage {
            role: "user".to_owned(),
            content: "weather?".to_owned(),
            tool_calls: None,
            raw_tool_call_text: None,
        }];
        let generated = "<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Denver\"}}\n</tool_call>";
        let turn_one_prefix = format!(
            "{}{}",
            renderer.apply(&turn_one, None, true).unwrap(),
            generated
        );
        let turn_two = vec![
            turn_one.first().cloned().unwrap(),
            ChatMessage {
                role: "assistant".to_owned(),
                content: String::new(),
                tool_calls: Some(vec![serde_json::json!({
                    "name": "weather",
                    "arguments": {"city": "Denver"}
                })]),
                raw_tool_call_text: Some(vec![Some(generated.to_owned())]),
            },
            ChatMessage {
                role: "tool".to_owned(),
                content: "sunny".to_owned(),
                tool_calls: None,
                raw_tool_call_text: None,
            },
        ];
        let turn_two_prompt = renderer.apply(&turn_two, None, true).unwrap();
        assert!(
            turn_two_prompt
                .as_bytes()
                .starts_with(turn_one_prefix.as_bytes())
        );
    }

    #[test]
    fn test_tojson_filter_with_nested_objects() {
        let env = tojson_env(TOJSON_TEMPLATE);
        let tmpl = env.get_template("test").unwrap();
        let nested = serde_json::json!({"a": {"b": [1, 2, 3]}});
        let result = tmpl
            .render(minijinja::context! { value => nested })
            .unwrap();
        let reparsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(
            reparsed.get("a").unwrap().get("b").unwrap(),
            &serde_json::json!([1, 2, 3])
        );
    }

    #[test]
    fn test_tojson_filter_with_arrays() {
        let env = tojson_env(TOJSON_TEMPLATE);
        let tmpl = env.get_template("test").unwrap();
        let result = tmpl
            .render(minijinja::context! { value => vec![1, 2, 3] })
            .unwrap();
        let reparsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(reparsed, serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_tojson_filter_with_special_characters() {
        let env = tojson_env(TOJSON_TEMPLATE);
        let tmpl = env.get_template("test").unwrap();
        let result = tmpl
            .render(minijinja::context! { value => "quotes: \"hello\" and backslash: \\" })
            .unwrap();
        let reparsed: String = serde_json::from_str(&result).unwrap();
        assert!(reparsed.contains("quotes: \"hello\""));
        assert!(reparsed.contains("backslash: \\"));
    }

    #[test]
    fn test_from_model_dir_array_of_templates_uses_default() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": [
                {"name": "rag", "template": "RAG:{{ messages[0].content }}"},
                {"name": "default", "template": "DEFAULT:{{ messages[0].content }}"}
            ]}"#,
        )
        .unwrap();
        let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
        let result = renderer.apply(&[msg("user", "hi")], None, false).unwrap();
        assert!(
            result.starts_with("DEFAULT:"),
            "Expected default template, got: {result}"
        );
    }

    #[test]
    fn test_from_model_dir_array_of_templates_falls_back_to_first() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": [
                {"name": "rag", "template": "RAG:{{ messages[0].content }}"},
                {"name": "tool_use", "template": "TOOL:{{ messages[0].content }}"}
            ]}"#,
        )
        .unwrap();
        let renderer = ChatTemplateRenderer::from_model_dir(dir.path()).unwrap();
        let result = renderer.apply(&[msg("user", "hi")], None, false).unwrap();
        assert!(
            result.starts_with("RAG:"),
            "Expected first template, got: {result}"
        );
    }

    #[test]
    fn test_from_model_dir_array_template_empty_array_errors() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": []}"#,
        )
        .unwrap();
        assert!(ChatTemplateRenderer::from_model_dir(dir.path()).is_err());
    }

    #[test]
    fn test_template_rendering_error_undefined_variable() {
        let renderer = ChatTemplateRenderer::new(r"{{ undefined_variable.nested_field }}").unwrap();
        assert!(renderer.apply(&[msg("user", "hi")], None, false).is_err());
    }

    // -----------------------------------------------------------------------
    // try_from_model_dir
    // -----------------------------------------------------------------------

    #[test]
    fn try_from_model_dir_empty_directory_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let result = ChatTemplateRenderer::try_from_model_dir(dir.path()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn try_from_model_dir_config_without_template_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"model_type": "starcoder2"}"#,
        )
        .unwrap();
        let result = ChatTemplateRenderer::try_from_model_dir(dir.path()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn try_from_model_dir_with_jinja_returns_some() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            r"{{ messages[0].content }}",
        )
        .unwrap();
        let result = ChatTemplateRenderer::try_from_model_dir(dir.path()).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn try_from_model_dir_with_config_template_returns_some() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "{{ messages[0].content }}"}"#,
        )
        .unwrap();
        let result = ChatTemplateRenderer::try_from_model_dir(dir.path()).unwrap();
        assert!(result.is_some());
    }

    // -----------------------------------------------------------------------
    // enable_thinking context passing
    // -----------------------------------------------------------------------

    /// Template that uses the `enable_thinking` variable.
    const THINKING_TEMPLATE: &str = r"{%- for message in messages %}{{ message.content }}{%- endfor %}{%- if enable_thinking %}<think>{%- endif %}";

    #[test]
    fn apply_with_thinking_false_omits_think_tag() {
        let renderer = ChatTemplateRenderer::new(THINKING_TEMPLATE).unwrap();
        let result = renderer
            .apply_with_thinking(&[msg("user", "hello")], None, false, false)
            .unwrap();
        assert!(
            !result.contains("<think>"),
            "should not contain <think> when disabled"
        );
    }

    #[test]
    fn apply_with_thinking_true_emits_think_tag() {
        let renderer = ChatTemplateRenderer::new(THINKING_TEMPLATE).unwrap();
        let result = renderer
            .apply_with_thinking(&[msg("user", "hello")], None, false, true)
            .unwrap();
        assert!(
            result.contains("<think>"),
            "should contain <think> when enabled"
        );
    }

    #[test]
    fn apply_delegates_to_apply_with_thinking_false() {
        let renderer = ChatTemplateRenderer::new(THINKING_TEMPLATE).unwrap();
        let via_apply = renderer.apply(&[msg("user", "hi")], None, false).unwrap();
        let via_explicit = renderer
            .apply_with_thinking(&[msg("user", "hi")], None, false, false)
            .unwrap();
        assert_eq!(
            via_apply, via_explicit,
            "apply() should delegate with enable_thinking=false"
        );
    }

    #[test]
    fn apply_with_thinking_and_tools() {
        let template = r"{%- for message in messages %}{{ message.content }}{%- endfor %}{%- if tools %}[TOOLS]{%- endif %}{%- if enable_thinking %}<think>{%- endif %}";
        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let tools = vec![serde_json::json!({"type": "function"})];
        let result = renderer
            .apply_with_thinking(&[msg("user", "hi")], Some(&tools), false, true)
            .unwrap();
        assert!(result.contains("[TOOLS]"), "tools should be rendered");
        assert!(result.contains("<think>"), "thinking tag should be present");
    }

    #[test]
    fn try_from_model_dir_malformed_json_returns_err() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            "not valid json {{{",
        )
        .unwrap();
        assert!(ChatTemplateRenderer::try_from_model_dir(dir.path()).is_err());
    }

    // -----------------------------------------------------------------
    // normalize_tool_call_for_template
    // -----------------------------------------------------------------
    //
    // Invariants asserted, one test per shape we observed in production:
    //
    // 1. `OpenAI` shape (name/arguments nested under `function`,
    //    arguments as JSON-encoded STRING) → after normalize, top-level
    //    name and arguments-as-mapping. This is the case that crashed
    //    Qwen's `chat_template.jinja:120` with "cannot convert value
    //    into pairs".
    // 2. Qwen-flat shape (top-level name/arguments, arguments already
    //    an object) → no-op, identity.
    // 3. Non-JSON string in `function.arguments` → flattened but kept
    //    as string (template can decide what to do).
    // 4. Non-object input (string, null, array) → no-op, can't panic.

    fn parsed(s: &str) -> serde_json::Value {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn normalize_openai_shape_to_qwen_flat() {
        let mut tc = parsed(
            r#"{
                "id": "call_0",
                "type": "function",
                "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" }
            }"#,
        );
        normalize_tool_call_for_template(&mut tc);

        // Top-level name and arguments are present.
        assert_eq!(tc.get("name").and_then(|v| v.as_str()), Some("get_weather"));
        // arguments is now an OBJECT, not a string.
        let args = tc.get("arguments").unwrap();
        assert!(
            args.is_object(),
            "expected arguments to be an object, got {args:?}"
        );
        assert_eq!(args.get("city").and_then(|v| v.as_str()), Some("Paris"));
        // id and type preserved.
        assert_eq!(tc.get("id").and_then(|v| v.as_str()), Some("call_0"));
        assert_eq!(tc.get("type").and_then(|v| v.as_str()), Some("function"));
    }

    #[test]
    fn normalize_qwen_flat_shape_is_noop() {
        let original = parsed(r#"{ "name": "search", "arguments": { "q": "rust" } }"#);
        let mut tc = original.clone();
        normalize_tool_call_for_template(&mut tc);
        assert_eq!(tc, original, "already-flat shape must be a no-op");
    }

    #[test]
    fn normalize_unparseable_string_arguments_coerced_to_empty_object() {
        // Unparseable string arguments are coerced to `{}` so the chat
        // template's `|items` doesn't blow up. The model loses the
        // pathological arguments, which is strictly better than the
        // entire conversation 500-ing.
        let mut tc = parsed(
            r#"{
                "function": { "name": "f", "arguments": "this is not json" }
            }"#,
        );
        normalize_tool_call_for_template(&mut tc);
        assert_eq!(tc.get("name").and_then(|v| v.as_str()), Some("f"));
        assert_eq!(tc.get("arguments"), Some(&parsed("{}")));
    }

    #[test]
    fn normalize_non_object_is_noop() {
        let mut s = parsed(r#""not a tool call""#);
        normalize_tool_call_for_template(&mut s);
        assert_eq!(s, parsed(r#""not a tool call""#));

        let mut n = parsed("null");
        normalize_tool_call_for_template(&mut n);
        assert_eq!(n, parsed("null"));

        let mut a = parsed("[1, 2, 3]");
        normalize_tool_call_for_template(&mut a);
        assert_eq!(a, parsed("[1, 2, 3]"));
    }

    /// Qwen's `chat_template.jinja:107-108` rebinds `tool_call` to
    /// `tool_call.function` when the latter is defined. If we only
    /// normalised the hoisted top-level `arguments` and left
    /// `function.arguments` as the original JSON-encoded string, the
    /// rebinding would walk straight into a string and the template
    /// would crash at `|items`. This test pins both paths.
    #[test]
    fn normalize_handles_qwen_rebind_to_function() {
        let mut tc = parsed(
            r#"{
                "id": "call_0",
                "type": "function",
                "function": { "name": "f", "arguments": "{\"city\":\"London\"}" }
            }"#,
        );
        normalize_tool_call_for_template(&mut tc);

        // Top-level arguments — Qwen-flat templates see this.
        let top_args = tc.get("arguments").unwrap();
        assert!(
            top_args.is_object(),
            "top-level arguments must be a mapping"
        );
        assert_eq!(
            top_args.get("city").and_then(|v| v.as_str()),
            Some("London")
        );

        // Nested function.arguments — Qwen's standard template walks this
        // after rebinding via `set tool_call = tool_call.function`.
        let func_args = tc
            .get("function")
            .and_then(|f| f.get("arguments"))
            .expect("function.arguments must still be present");
        assert!(
            func_args.is_object(),
            "nested function.arguments must ALSO be a mapping, got {func_args:?}"
        );
        assert_eq!(
            func_args.get("city").and_then(|v| v.as_str()),
            Some("London")
        );
    }

    /// Arguments shaped as something other than an object after normalization
    /// must be coerced to an empty object so the chat template's
    /// `tool_call.arguments|items` can render. Without this, Qwen's
    /// `chat_template.jinja:120` raises `cannot convert value into pairs`
    /// when prior conversation turns carried weird tool-call shapes.
    #[test]
    fn arguments_coerced_to_empty_object_when_not_mapping() {
        // Null arguments → empty object.
        let mut tc = parsed(r#"{ "name": "f", "arguments": null }"#);
        normalize_tool_call_for_template(&mut tc);
        assert_eq!(tc.get("arguments"), Some(&parsed("{}")));

        // Array arguments → empty object.
        let mut tc = parsed(r#"{ "name": "f", "arguments": [1, 2, 3] }"#);
        normalize_tool_call_for_template(&mut tc);
        assert_eq!(tc.get("arguments"), Some(&parsed("{}")));

        // Number arguments → empty object.
        let mut tc = parsed(r#"{ "name": "f", "arguments": 42 }"#);
        normalize_tool_call_for_template(&mut tc);
        assert_eq!(tc.get("arguments"), Some(&parsed("{}")));

        // Unparseable string arguments → empty object (the model can't
        // express what it wanted; better than a 500).
        let mut tc = parsed(r#"{ "name": "f", "arguments": "this is not json" }"#);
        normalize_tool_call_for_template(&mut tc);
        assert_eq!(tc.get("arguments"), Some(&parsed("{}")));

        // Valid-JSON-string-that-parses-to-array → coerced via the
        // second pass (parse succeeds, result is still not an object).
        let mut tc = parsed(r#"{ "name": "f", "arguments": "[1,2,3]" }"#);
        normalize_tool_call_for_template(&mut tc);
        assert_eq!(tc.get("arguments"), Some(&parsed("{}")));
    }
}
