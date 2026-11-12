//! The `OpenAI` `tool_choice` request field.

use crate::error::ServerError;

/// How the model may use the tools offered in a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model decides whether to call a tool. The default.
    Auto,
    /// Tools are neither shown to the model nor parsed from its output.
    Disabled,
}

impl ToolChoice {
    /// Interpret a request's `tool_choice`.
    ///
    /// `"required"` and a named function are valid `OpenAI` values that Higgs
    /// cannot enforce yet, so they are rejected rather than silently treated
    /// as `"auto"`.
    pub fn from_request(value: Option<&serde_json::Value>) -> Result<Self, ServerError> {
        match value {
            None | Some(serde_json::Value::Null) => Ok(Self::Auto),
            Some(serde_json::Value::String(choice)) => match choice.as_str() {
                "auto" => Ok(Self::Auto),
                "none" => Ok(Self::Disabled),
                "required" => Err(ServerError::BadRequest(
                    "tool_choice \"required\" is not supported yet; use \"auto\" or \"none\""
                        .to_owned(),
                )),
                other => Err(ServerError::BadRequest(format!(
                    "Invalid tool_choice \"{other}\"; supported values are \"auto\" and \"none\""
                ))),
            },
            Some(serde_json::Value::Object(_)) => Err(ServerError::BadRequest(
                "tool_choice naming a specific function is not supported yet; use \"auto\" or \"none\""
                    .to_owned(),
            )),
            Some(_) => Err(ServerError::BadRequest(
                "Invalid tool_choice; supported values are \"auto\" and \"none\"".to_owned(),
            )),
        }
    }
}

#[allow(clippy::unwrap_used, clippy::panic)]
#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> Result<ToolChoice, ServerError> {
        let value: serde_json::Value = serde_json::from_str(json).unwrap();
        ToolChoice::from_request(Some(&value))
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
        assert_eq!(parse("null").unwrap(), ToolChoice::Auto);
    }

    #[test]
    fn auto_and_none_are_supported() {
        assert_eq!(parse(r#""auto""#).unwrap(), ToolChoice::Auto);
        assert_eq!(parse(r#""none""#).unwrap(), ToolChoice::Disabled);
    }

    #[test]
    fn required_is_rejected_as_unsupported() {
        assert!(message(parse(r#""required""#)).contains("\"required\" is not supported yet"));
    }

    #[test]
    fn named_function_is_rejected_as_unsupported() {
        let text = message(parse(
            r#"{"type":"function","function":{"name":"get_weather"}}"#,
        ));
        assert!(
            text.contains("specific function is not supported yet"),
            "{text}"
        );
    }

    #[test]
    fn unknown_strings_and_other_types_are_invalid() {
        assert!(message(parse(r#""sometimes""#)).contains("Invalid tool_choice \"sometimes\""));
        assert!(message(parse("3")).contains("Invalid tool_choice"));
        assert!(message(parse("[]")).contains("Invalid tool_choice"));
    }
}
