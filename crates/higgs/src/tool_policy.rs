//! What a request's `tool_choice` and `parallel_tool_calls` mean for
//! generation, shared by the Chat Completions, Responses and Anthropic routes.

use higgs_engine::tool_format::forced::{ForcedCallError, ForcedChoice, forced_call_pattern};
use higgs_engine::tool_format::{ToolCallFormat, ToolCallParser};
use serde_json::Value;

use crate::constraint::ConstraintSpec;
use crate::error::ServerError;
use crate::tool_choice::ToolChoice;

/// How the model may use the request's tools, and how many calls it may make.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolPolicy {
    choice: ToolChoice,
    parallel_tool_calls: bool,
}

impl ToolPolicy {
    pub const fn new(choice: ToolChoice, parallel_tool_calls: bool) -> Self {
        Self {
            choice,
            parallel_tool_calls,
        }
    }

    /// From Chat Completions `tool_choice` and `parallel_tool_calls`.
    pub fn from_chat(
        tool_choice: Option<&Value>,
        parallel_tool_calls: Option<bool>,
    ) -> Result<Self, ServerError> {
        Ok(Self::new(
            ToolChoice::from_request(tool_choice)?,
            parallel_tool_calls.unwrap_or(true),
        ))
    }

    /// From an Anthropic `tool_choice`, whose `disable_parallel_tool_use`
    /// flag is the inverse of `parallel_tool_calls`.
    pub fn from_anthropic(tool_choice: Option<&Value>) -> Result<Self, ServerError> {
        let disable_parallel = tool_choice
            .and_then(|choice| choice.get("disable_parallel_tool_use"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        Ok(Self::new(
            ToolChoice::from_anthropic(tool_choice)?,
            !disable_parallel,
        ))
    }

    /// Whether the tools are shown to the model and parsed from its output.
    pub fn exposes_tools(&self) -> bool {
        self.choice != ToolChoice::Disabled
    }

    /// Whether the model must produce a tool call.
    pub const fn forces_call(&self) -> bool {
        matches!(self.choice, ToolChoice::Required | ToolChoice::Function(_))
    }

    /// The most calls a response may carry.
    pub const fn max_calls(&self) -> Option<usize> {
        if self.parallel_tool_calls {
            None
        } else {
            Some(1)
        }
    }

    /// The parser for a model that writes calls in `format`, limited to this
    /// policy's call count.
    pub fn parser(&self, format: ToolCallFormat, tools: Option<&[Value]>) -> ToolCallParser {
        ToolCallParser::new(format, tools).with_max_calls(self.max_calls())
    }

    /// The constraint that makes the model call a tool, or `None` when the
    /// choice does not force one.
    ///
    /// Fails with a client error when the model's tool-call format cannot be
    /// constrained, or the named tool does not exist.
    pub fn forced_constraint(
        &self,
        tools: &[Value],
        format: ToolCallFormat,
    ) -> Result<Option<ConstraintSpec>, ServerError> {
        let choice = match &self.choice {
            ToolChoice::Auto | ToolChoice::Disabled => return Ok(None),
            ToolChoice::Required => ForcedChoice::AnyTool,
            ToolChoice::Function(name) => ForcedChoice::Function(name),
        };
        forced_call_pattern(format, tools, choice, self.parallel_tool_calls)
            .map(|pattern| Some(ConstraintSpec::Regex(pattern)))
            .map_err(|error| match error {
                ForcedCallError::UnsupportedFormat(_)
                | ForcedCallError::NoTools
                | ForcedCallError::UnknownFunction(_)
                | ForcedCallError::InvalidSchema { .. } => {
                    ServerError::BadRequest(error.to_string())
                }
            })
    }
}

/// Combine the constraint a client asked for with the one a forced tool call
/// needs. Both would constrain the same output, so asking for both is an error.
pub fn combine_constraints(
    requested: Option<ConstraintSpec>,
    forced: Option<ConstraintSpec>,
) -> Result<Option<ConstraintSpec>, ServerError> {
    match (requested, forced) {
        (Some(_), Some(_)) => Err(ServerError::BadRequest(
            "tool_choice that forces a tool call cannot be combined with response_format, \
             guided_regex or guided_json"
                .to_owned(),
        )),
        (only_requested, None) => Ok(only_requested),
        (None, only_forced) => Ok(only_forced),
    }
}

#[allow(clippy::unwrap_used, clippy::panic)]
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Vec<Value> {
        vec![json!({"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }})]
    }

    fn message(error: ServerError) -> String {
        if let ServerError::BadRequest(text) = error {
            text
        } else {
            panic!("expected BadRequest")
        }
    }

    #[test]
    fn defaults_expose_tools_without_forcing() {
        let policy = ToolPolicy::from_chat(None, None).unwrap();
        assert!(policy.exposes_tools());
        assert!(!policy.forces_call());
        assert_eq!(policy.max_calls(), None);
        assert_eq!(
            policy
                .forced_constraint(&tools(), ToolCallFormat::Qwen)
                .unwrap(),
            None
        );
    }

    #[test]
    fn none_hides_tools() {
        let policy = ToolPolicy::from_chat(Some(&json!("none")), None).unwrap();
        assert!(!policy.exposes_tools());
        assert!(!policy.forces_call());
    }

    #[test]
    fn parallel_tool_calls_false_limits_to_one_call() {
        let policy = ToolPolicy::from_chat(None, Some(false)).unwrap();
        assert_eq!(policy.max_calls(), Some(1));
        let anthropic = ToolPolicy::from_anthropic(Some(
            &json!({"type": "auto", "disable_parallel_tool_use": true}),
        ))
        .unwrap();
        assert_eq!(anthropic.max_calls(), Some(1));
    }

    #[test]
    fn required_and_named_choices_build_regex_constraints() {
        for choice in [
            json!("required"),
            json!({"type": "function", "function": {"name": "get_weather"}}),
        ] {
            let policy = ToolPolicy::from_chat(Some(&choice), None).unwrap();
            assert!(policy.forces_call());
            let constraint = policy
                .forced_constraint(&tools(), ToolCallFormat::Qwen)
                .unwrap();
            assert!(
                matches!(constraint, Some(ConstraintSpec::Regex(_))),
                "{choice}"
            );
        }
    }

    #[test]
    fn forcing_failures_are_client_errors() {
        let required = ToolPolicy::from_chat(Some(&json!("required")), None).unwrap();
        let unsupported = required
            .forced_constraint(&tools(), ToolCallFormat::Gemma4)
            .unwrap_err();
        assert!(message(unsupported).contains("gemma 4"));
        let no_tools = required
            .forced_constraint(&[], ToolCallFormat::Qwen)
            .unwrap_err();
        assert!(message(no_tools).contains("no tools"));

        let named = ToolPolicy::from_chat(
            Some(&json!({"type": "function", "function": {"name": "missing"}})),
            None,
        )
        .unwrap();
        let unknown = named
            .forced_constraint(&tools(), ToolCallFormat::Qwen)
            .unwrap_err();
        assert!(message(unknown).contains("\"missing\""));
    }

    #[test]
    fn requested_and_forced_constraints_cannot_be_combined() {
        let regex = Some(ConstraintSpec::Regex("a".to_owned()));
        assert_eq!(combine_constraints(None, None).unwrap(), None);
        assert_eq!(combine_constraints(regex.clone(), None).unwrap(), regex);
        assert_eq!(combine_constraints(None, regex.clone()).unwrap(), regex);
        let error = combine_constraints(regex.clone(), regex).unwrap_err();
        assert!(message(error).contains("cannot be combined"));
    }
}
