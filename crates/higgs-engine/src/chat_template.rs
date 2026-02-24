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
}

/// Renders chat messages using a Jinja2 template (`HuggingFace` format).
pub struct ChatTemplateRenderer {
    env: Environment<'static>,
}

impl ChatTemplateRenderer {
    /// Create a renderer from a Jinja2 template string.
    pub fn new<S: Into<String>>(template_source: S) -> Result<Self, EngineError> {
        let mut env = Environment::new();
        env.add_filter("tojson", tojson_filter);
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("chat".to_owned(), template_source.into())
            .map_err(|e| EngineError::Template(e.to_string()))?;
        Ok(Self { env })
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
        // Prefer standalone chat_template.jinja
        let jinja_path = model_dir.join("chat_template.jinja");
        if jinja_path.exists() {
            let template = std::fs::read_to_string(&jinja_path)
                .map_err(|e| EngineError::Template(format!("Failed to read template: {e}")))?;
            return Self::new(&template).map(Some);
        }

        // Fall back to tokenizer_config.json
        let config_path = model_dir.join("tokenizer_config.json");
        if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| EngineError::Template(format!("Failed to read config: {e}")))?;
            let config: serde_json::Value = serde_json::from_str(&config_str)
                .map_err(|e| EngineError::Template(format!("Invalid JSON: {e}")))?;
            if let Some(ct) = config.get("chat_template") {
                // String template
                if let Some(template) = ct.as_str() {
                    return Self::new(template).map(Some);
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
                        return Self::new(template).map(Some);
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
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| EngineError::Template(e.to_string()))?;

        let context = tools.map_or_else(
            || {
                minijinja::context! {
                    messages => messages,
                    add_generation_prompt => add_generation_prompt,
                }
            },
            |tool_list| {
                minijinja::context! {
                    messages => messages,
                    tools => tool_list,
                    add_generation_prompt => add_generation_prompt,
                }
            },
        );

        tmpl.render(context)
            .map_err(|e| EngineError::Template(e.to_string()))
    }
}

/// Custom tojson filter for minijinja (used by HF chat templates).
#[allow(clippy::needless_pass_by_value)]
fn tojson_filter(value: Value) -> Result<String, minijinja::Error> {
    let serialized = serde_json::to_string(&value).map_err(|e| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "JSON serialization failed",
        )
        .with_source(e)
    })?;
    Ok(serialized)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_owned(),
            content: content.to_owned(),
            tool_calls: None,
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
        }];
        let result = renderer.apply(&messages, None, false).unwrap();
        assert!(result.contains("[tools]"));
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
}
