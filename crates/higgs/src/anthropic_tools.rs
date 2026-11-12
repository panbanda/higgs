//! Tool use for the Anthropic Messages route: tool conversion, `tool_choice`
//! handling and `tool_use` content blocks.

use higgs_engine::tool_format::{ToolCallFormat, ToolCallParser};
use higgs_engine::tool_parser::ParsedToolCall;
use serde_json::{Value, json};

use crate::constraint::ConstraintSpec;
use crate::error::ServerError;
use crate::tool_policy::ToolPolicy;
use crate::types::anthropic::ToolUseBlock;

/// Anthropic tool definitions in the `OpenAI` function shape that chat
/// templates and the tool-call parsers expect.
pub fn openai_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.get("name").cloned().unwrap_or_else(|| Value::String(String::new())),
                    "description": tool
                        .get("description")
                        .cloned()
                        .unwrap_or_else(|| Value::String(String::new())),
                    "parameters": tool
                        .get("input_schema")
                        .cloned()
                        .unwrap_or_else(|| json!({"type": "object"})),
                }
            })
        })
        .collect()
}

/// Everything a Messages request's tool settings decide before generation.
pub struct ToolPlan {
    /// The tools to render into the prompt; `None` when there are none or
    /// `tool_choice` is `none`.
    pub tools: Option<Vec<Value>>,
    pub policy: ToolPolicy,
    /// The constraint that forces a call, when `tool_choice` demands one.
    pub constraint: Option<ConstraintSpec>,
    pub parser: ToolCallParser,
}

impl ToolPlan {
    pub fn for_request(
        tools: Option<&[Value]>,
        tool_choice: Option<&Value>,
        format: ToolCallFormat,
    ) -> Result<Self, ServerError> {
        let policy = ToolPolicy::from_anthropic(tool_choice)?;
        let converted = tools
            .filter(|declared| !declared.is_empty() && policy.exposes_tools())
            .map(openai_tools);
        let constraint =
            policy.forced_constraint(converted.as_deref().unwrap_or_default(), format)?;
        let parser = policy.parser(format, converted.as_deref());
        Ok(Self {
            tools: converted,
            policy,
            constraint,
            parser,
        })
    }
}

/// A fresh `tool_use` block id.
pub fn new_tool_use_id() -> String {
    format!("toolu_{}", uuid::Uuid::new_v4().simple())
}

/// The non-streaming content block for a parsed call.
pub fn tool_use_block(call: &ParsedToolCall) -> ToolUseBlock {
    ToolUseBlock {
        block_type: "tool_use",
        id: new_tool_use_id(),
        name: call.name.clone(),
        input: call.arguments.clone(),
    }
}

/// The three server-sent events (`(event name, data)`) that stream one
/// complete call as content block `index`.
pub fn tool_use_stream_events(index: u32, call: &ParsedToolCall) -> [(&'static str, String); 3] {
    let start = json!({
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "tool_use",
            "id": new_tool_use_id(),
            "name": call.name,
            "input": {},
        },
    });
    let delta = json!({
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": call.arguments.to_string()},
    });
    let stop = json!({"type": "content_block_stop", "index": index});
    [
        ("content_block_start", start.to_string()),
        ("content_block_delta", delta.to_string()),
        ("content_block_stop", stop.to_string()),
    ]
}

#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_tools_become_function_tools() {
        let converted = openai_tools(&[json!({
            "name": "get_weather",
            "description": "Weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
        })]);
        assert_eq!(converted[0]["type"], "function");
        assert_eq!(converted[0]["function"]["name"], "get_weather");
        assert_eq!(
            converted[0]["function"]["parameters"]["properties"]["city"]["type"],
            "string"
        );
    }

    #[test]
    fn streamed_call_is_three_events_with_the_arguments_as_one_json_delta() {
        let call = ParsedToolCall {
            name: "f".to_owned(),
            arguments: json!({"a": 1}),
        };
        let events = tool_use_stream_events(2, &call);
        assert_eq!(events[0].0, "content_block_start");
        let start: Value = serde_json::from_str(&events[0].1).unwrap();
        assert_eq!(start["index"], 2);
        assert_eq!(start["content_block"]["name"], "f");
        let delta: Value = serde_json::from_str(&events[1].1).unwrap();
        assert_eq!(delta["delta"]["partial_json"], "{\"a\":1}");
        assert_eq!(events[2].0, "content_block_stop");
    }
}
