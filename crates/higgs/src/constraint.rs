//! Output constraints requested through the OpenAI-compatible chat API.

use crate::error::ServerError;
use crate::types::openai::ResponseFormat;

/// A grammar the generated text must follow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintSpec {
    JsonObject,
    JsonSchema(serde_json::Value),
    Regex(String),
}

impl ConstraintSpec {
    /// Resolve the constraint requested by `response_format`, `guided_regex`
    /// and `guided_json`. At most one of them may ask for a constraint.
    pub fn from_request(
        response_format: Option<&ResponseFormat>,
        guided_regex: Option<&str>,
        guided_json: Option<&serde_json::Value>,
    ) -> Result<Option<Self>, ServerError> {
        let mut requested: Vec<(&str, Self)> = Vec::new();

        if let Some(format) = response_format
            && let Some(spec) = Self::from_response_format(format)?
        {
            requested.push(("response_format", spec));
        }
        if let Some(pattern) = guided_regex {
            requested.push(("guided_regex", Self::regex(pattern)?));
        }
        if let Some(schema) = guided_json {
            if !schema.is_object() {
                return Err(ServerError::BadRequest(
                    "guided_json must be a JSON schema object".to_owned(),
                ));
            }
            requested.push(("guided_json", Self::JsonSchema(schema.clone())));
        }

        if requested.len() > 1 {
            let names: Vec<&str> = requested.iter().map(|(name, _)| *name).collect();
            return Err(ServerError::BadRequest(format!(
                "Only one output constraint can be set per request, got: {}",
                names.join(", ")
            )));
        }
        Ok(requested.into_iter().next().map(|(_, spec)| spec))
    }

    fn from_response_format(format: &ResponseFormat) -> Result<Option<Self>, ServerError> {
        match format.r#type.as_str() {
            "text" => Ok(None),
            "json_object" => Ok(Some(Self::JsonObject)),
            "json_schema" => Ok(Some(format.json_schema.as_ref().map_or(
                Self::JsonObject,
                |schema| {
                    // The OpenAI shape nests the schema under a `schema` key:
                    // {"name": "...", "schema": {...}}. Bare schemas are
                    // accepted as well.
                    Self::JsonSchema(schema.get("schema").unwrap_or(schema).clone())
                },
            ))),
            "regex" => {
                let pattern = format.regex.as_deref().ok_or_else(|| {
                    ServerError::BadRequest(
                        "response_format type \"regex\" requires a \"regex\" string".to_owned(),
                    )
                })?;
                Ok(Some(Self::regex(pattern)?))
            }
            other => Err(ServerError::BadRequest(format!(
                "Unsupported response_format type: {other}"
            ))),
        }
    }

    fn regex(pattern: &str) -> Result<Self, ServerError> {
        if pattern.is_empty() {
            return Err(ServerError::BadRequest(
                "regex constraint must not be empty".to_owned(),
            ));
        }
        regex::Regex::new(pattern).map_err(|error| {
            ServerError::BadRequest(format!("Invalid regex constraint: {error}"))
        })?;
        Ok(Self::Regex(pattern.to_owned()))
    }
}

#[allow(clippy::unwrap_used, clippy::panic, clippy::needless_pass_by_value)]
#[cfg(test)]
mod tests {
    use super::*;

    fn format(json: &str) -> ResponseFormat {
        serde_json::from_str(json).unwrap()
    }

    fn resolve(
        response_format: Option<&str>,
        guided_regex: Option<&str>,
        guided_json: Option<serde_json::Value>,
    ) -> Result<Option<ConstraintSpec>, ServerError> {
        let parsed = response_format.map(format);
        ConstraintSpec::from_request(parsed.as_ref(), guided_regex, guided_json.as_ref())
    }

    fn message(error: ServerError) -> String {
        if let ServerError::BadRequest(text) = error {
            text
        } else {
            panic!("expected BadRequest")
        }
    }

    #[test]
    fn no_constraint_by_default() {
        assert_eq!(resolve(None, None, None).unwrap(), None);
        assert_eq!(
            resolve(Some(r#"{"type":"text"}"#), None, None).unwrap(),
            None
        );
    }

    #[test]
    fn json_object_format() {
        assert_eq!(
            resolve(Some(r#"{"type":"json_object"}"#), None, None).unwrap(),
            Some(ConstraintSpec::JsonObject)
        );
    }

    #[test]
    fn json_schema_format_unwraps_openai_envelope() {
        let spec = resolve(
            Some(r#"{"type":"json_schema","json_schema":{"name":"n","schema":{"type":"object"}}}"#),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            spec,
            Some(ConstraintSpec::JsonSchema(
                serde_json::json!({"type": "object"})
            ))
        );
    }

    #[test]
    fn json_schema_format_accepts_bare_schema() {
        let spec = resolve(
            Some(r#"{"type":"json_schema","json_schema":{"type":"object"}}"#),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            spec,
            Some(ConstraintSpec::JsonSchema(
                serde_json::json!({"type": "object"})
            ))
        );
    }

    #[test]
    fn json_schema_format_without_schema_falls_back_to_json_object() {
        assert_eq!(
            resolve(Some(r#"{"type":"json_schema"}"#), None, None).unwrap(),
            Some(ConstraintSpec::JsonObject)
        );
    }

    #[test]
    fn regex_format() {
        assert_eq!(
            resolve(Some(r#"{"type":"regex","regex":"[0-9]+"}"#), None, None).unwrap(),
            Some(ConstraintSpec::Regex("[0-9]+".to_owned()))
        );
    }

    #[test]
    fn regex_format_requires_pattern() {
        let error = resolve(Some(r#"{"type":"regex"}"#), None, None).unwrap_err();
        assert!(message(error).contains("requires a \"regex\""));
    }

    #[test]
    fn guided_regex_is_a_regex_constraint() {
        assert_eq!(
            resolve(None, Some("a|b"), None).unwrap(),
            Some(ConstraintSpec::Regex("a|b".to_owned()))
        );
    }

    #[test]
    fn guided_json_is_a_schema_constraint() {
        let schema = serde_json::json!({"type": "object", "properties": {"schema": {}}});
        assert_eq!(
            resolve(None, None, Some(schema.clone())).unwrap(),
            Some(ConstraintSpec::JsonSchema(schema))
        );
    }

    #[test]
    fn guided_json_must_be_an_object() {
        let error = resolve(None, None, Some(serde_json::json!("{}"))).unwrap_err();
        assert!(message(error).contains("JSON schema object"));
    }

    #[test]
    fn invalid_regex_reports_the_compile_error() {
        let error = resolve(None, Some("(unclosed"), None).unwrap_err();
        let text = message(error);
        assert!(text.contains("Invalid regex constraint"), "{text}");
        assert!(text.contains("unclosed"), "{text}");
    }

    #[test]
    fn empty_regex_is_rejected() {
        let error = resolve(None, Some(""), None).unwrap_err();
        assert!(message(error).contains("must not be empty"));
    }

    #[test]
    fn unsupported_response_format_type_is_rejected() {
        let error = resolve(Some(r#"{"type":"xml"}"#), None, None).unwrap_err();
        assert!(message(error).contains("Unsupported response_format type: xml"));
    }

    #[test]
    fn multiple_constraints_are_rejected_and_named() {
        let error = resolve(
            Some(r#"{"type":"json_object"}"#),
            Some("a"),
            Some(serde_json::json!({"type": "object"})),
        )
        .unwrap_err();
        let text = message(error);
        assert!(text.contains("Only one output constraint"), "{text}");
        assert!(
            text.contains("response_format, guided_regex, guided_json"),
            "{text}"
        );
    }

    #[test]
    fn text_format_does_not_conflict_with_guided_fields() {
        assert_eq!(
            resolve(Some(r#"{"type":"text"}"#), Some("a"), None).unwrap(),
            Some(ConstraintSpec::Regex("a".to_owned()))
        );
    }

    #[test]
    fn two_guided_fields_conflict() {
        let error = resolve(None, Some("a"), Some(serde_json::json!({}))).unwrap_err();
        assert!(message(error).contains("guided_regex, guided_json"));
    }
}
