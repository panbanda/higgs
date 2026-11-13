//! The `tool_choice` request field, in the `OpenAI` Chat Completions,
//! `OpenAI` Responses and Anthropic Messages spellings.

use serde_json::Value;

use crate::error::ServerError;

/// How the model may use the tools offered in a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model decides whether to call a tool. The default.
    Auto,
    /// Tools are neither shown to the model nor parsed from its output.
    Disabled,
    /// The model must call at least one of the provided tools.
    Required,
    /// The model must call this tool.
    Function(String),
}

fn invalid(detail: &str) -> ServerError {
    ServerError::BadRequest(format!("Invalid tool_choice: {detail}"))
}

fn function_name(value: &Value, nested_under_function: bool) -> Result<String, ServerError> {
    let name = if nested_under_function {
        value
            .get("function")
            .and_then(|function| function.get("name"))
    } else {
        value.get("name")
    };
    name.and_then(Value::as_str)
        .filter(|candidate_name| !candidate_name.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| invalid("a function choice needs a non-empty function name"))
}

impl ToolChoice {
    /// Interpret a Chat Completions `tool_choice`: `"auto"`, `"none"`,
    /// `"required"` or `{"type": "function", "function": {"name": …}}`.
    pub fn from_request(value: Option<&Value>) -> Result<Self, ServerError> {
        match value {
            None | Some(Value::Null) => Ok(Self::Auto),
            Some(Value::String(choice)) => Self::from_keyword(choice),
            Some(object @ Value::Object(_)) => match object.get("type").and_then(Value::as_str) {
                Some("function") => Ok(Self::Function(function_name(object, true)?)),
                Some(other) => Err(invalid(&format!("unsupported type \"{other}\""))),
                None => Err(invalid("an object choice needs a \"type\"")),
            },
            Some(_) => Err(invalid(
                "expected \"auto\", \"none\", \"required\" or a function object",
            )),
        }
    }

    /// Interpret a Responses API `tool_choice`, where a named function is
    /// `{"type": "function", "name": …}`.
    pub fn from_responses(value: Option<&Value>) -> Result<Self, ServerError> {
        match value {
            Some(object @ Value::Object(_)) => match object.get("type").and_then(Value::as_str) {
                Some("function") => Ok(Self::Function(function_name(object, false)?)),
                Some(other) => Err(invalid(&format!(
                    "type \"{other}\" is not supported; use \"auto\", \"none\", \"required\" or a function"
                ))),
                None => Err(invalid("an object choice needs a \"type\"")),
            },
            other => Self::from_request(other),
        }
    }

    /// Interpret an Anthropic Messages `tool_choice`: `{"type": "auto"}`,
    /// `{"type": "any"}`, `{"type": "tool", "name": …}` or `{"type": "none"}`.
    pub fn from_anthropic(value: Option<&Value>) -> Result<Self, ServerError> {
        match value {
            None | Some(Value::Null) => Ok(Self::Auto),
            Some(object @ Value::Object(_)) => match object.get("type").and_then(Value::as_str) {
                Some("auto") => Ok(Self::Auto),
                Some("any") => Ok(Self::Required),
                Some("none") => Ok(Self::Disabled),
                Some("tool") => Ok(Self::Function(function_name(object, false)?)),
                Some(other) => Err(invalid(&format!("unsupported type \"{other}\""))),
                None => Err(invalid("an object choice needs a \"type\"")),
            },
            Some(_) => Err(invalid("expected an object with a \"type\"")),
        }
    }

    fn from_keyword(keyword: &str) -> Result<Self, ServerError> {
        match keyword {
            "auto" => Ok(Self::Auto),
            "none" => Ok(Self::Disabled),
            "required" => Ok(Self::Required),
            other => Err(invalid(&format!(
                "\"{other}\"; supported values are \"auto\", \"none\" and \"required\""
            ))),
        }
    }

    /// The Chat Completions spelling of this choice.
    pub fn to_chat_value(&self) -> Value {
        match self {
            Self::Auto => Value::String("auto".to_owned()),
            Self::Disabled => Value::String("none".to_owned()),
            Self::Required => Value::String("required".to_owned()),
            Self::Function(name) => {
                serde_json::json!({"type": "function", "function": {"name": name}})
            }
        }
    }
}

#[allow(clippy::unwrap_used, clippy::panic)]
#[cfg(test)]
mod tests {
    use super::*;

    fn value(json: &str) -> Value {
        serde_json::from_str(json).unwrap()
    }

    fn chat(json: &str) -> Result<ToolChoice, ServerError> {
        ToolChoice::from_request(Some(&value(json)))
    }

    fn message(result: Result<ToolChoice, ServerError>) -> String {
        match result {
            Err(ServerError::BadRequest(text)) => text,
            _ => panic!("expected a BadRequest error"),
        }
    }

    #[test]
    fn absent_and_null_mean_auto() {
        assert_eq!(ToolChoice::from_request(None).unwrap(), ToolChoice::Auto);
        assert_eq!(chat("null").unwrap(), ToolChoice::Auto);
        assert_eq!(ToolChoice::from_anthropic(None).unwrap(), ToolChoice::Auto);
        assert_eq!(ToolChoice::from_responses(None).unwrap(), ToolChoice::Auto);
    }

    #[test]
    fn chat_keywords_and_named_function() {
        assert_eq!(chat(r#""auto""#).unwrap(), ToolChoice::Auto);
        assert_eq!(chat(r#""none""#).unwrap(), ToolChoice::Disabled);
        assert_eq!(chat(r#""required""#).unwrap(), ToolChoice::Required);
        assert_eq!(
            chat(r#"{"type":"function","function":{"name":"get_weather"}}"#).unwrap(),
            ToolChoice::Function("get_weather".to_owned())
        );
    }

    #[test]
    fn chat_rejects_malformed_choices() {
        assert!(message(chat(r#""sometimes""#)).contains("\"sometimes\""));
        assert!(message(chat("3")).contains("Invalid tool_choice"));
        assert!(message(chat("[]")).contains("Invalid tool_choice"));
        assert!(message(chat(r#"{"function":{"name":"f"}}"#)).contains("\"type\""));
        assert!(message(chat(r#"{"type":"function"}"#)).contains("function name"));
        assert!(
            message(chat(r#"{"type":"function","function":{"name":""}}"#))
                .contains("function name")
        );
        assert!(message(chat(r#"{"type":"allowed_tools"}"#)).contains("allowed_tools"));
    }

    #[test]
    fn responses_uses_the_flat_function_shape() {
        let parse = |json: &str| ToolChoice::from_responses(Some(&value(json)));
        assert_eq!(parse(r#""required""#).unwrap(), ToolChoice::Required);
        assert_eq!(
            parse(r#"{"type":"function","name":"f"}"#).unwrap(),
            ToolChoice::Function("f".to_owned())
        );
        assert!(message(parse(r#"{"type":"function"}"#)).contains("function name"));
        assert!(message(parse(r#"{"type":"web_search"}"#)).contains("web_search"));
        assert!(message(parse(r#""bogus""#)).contains("bogus"));
    }

    #[test]
    fn anthropic_spellings() {
        let parse = |json: &str| ToolChoice::from_anthropic(Some(&value(json)));
        assert_eq!(parse(r#"{"type":"auto"}"#).unwrap(), ToolChoice::Auto);
        assert_eq!(parse(r#"{"type":"any"}"#).unwrap(), ToolChoice::Required);
        assert_eq!(parse(r#"{"type":"none"}"#).unwrap(), ToolChoice::Disabled);
        assert_eq!(
            parse(r#"{"type":"tool","name":"f"}"#).unwrap(),
            ToolChoice::Function("f".to_owned())
        );
        assert!(message(parse(r#"{"type":"tool"}"#)).contains("function name"));
        assert!(message(parse(r#"{"type":"mystery"}"#)).contains("mystery"));
        assert!(message(parse(r#""any""#)).contains("Invalid tool_choice"));
    }

    #[test]
    fn chat_value_round_trips() {
        for choice in [
            ToolChoice::Auto,
            ToolChoice::Disabled,
            ToolChoice::Required,
            ToolChoice::Function("f".to_owned()),
        ] {
            assert_eq!(
                ToolChoice::from_request(Some(&choice.to_chat_value())).unwrap(),
                choice
            );
        }
    }
}
