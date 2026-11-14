//! Tool use for the Anthropic Messages route: tool conversion, `tool_choice`
//! handling and `tool_use` content blocks.

use serde_json::{Value, json};

use crate::error::ServerError;
use crate::generation::GeneratedToolCall;
use crate::tool_policy::ToolPolicy;
use crate::types::anthropic::{
    AnthropicContent, AnthropicMessage, ContentBlock, SystemPrompt, ToolUseBlock,
};
use higgs_engine::chat_template::ChatMessage;

fn text_of(content: &AnthropicContent) -> String {
    match content {
        AnthropicContent::Text(text) => text.clone(),
        AnthropicContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                ContentBlock::ToolUse { .. }
                | ContentBlock::ToolResult { .. }
                | ContentBlock::Thinking { .. }
                | ContentBlock::RedactedThinking { .. }
                | ContentBlock::Image { .. }
                | ContentBlock::Document { .. }
                | ContentBlock::ServerToolUse { .. }
                | ContentBlock::WebSearchToolResult { .. }
                | ContentBlock::CodeExecutionToolResult { .. }
                | ContentBlock::Other => None,
            })
            .collect(),
    }
}

/// Anthropic history as template messages, keeping tool use.
///
/// An assistant `tool_use` block becomes a `tool_calls` entry, and each user `tool_result`
/// block becomes a `tool` message ahead of the rest of that user turn.
pub fn messages_with_tool_history(
    messages: &[AnthropicMessage],
    system: Option<&SystemPrompt>,
) -> Vec<ChatMessage> {
    let mut result = Vec::new();
    if let Some(system_prompt) = system {
        result.push(ChatMessage {
            role: "system".to_owned(),
            content: system_prompt.to_text(),
            tool_calls: None,
        });
    }
    for message in messages {
        let AnthropicContent::Blocks(blocks) = &message.content else {
            result.push(ChatMessage {
                role: message.role.clone(),
                content: text_of(&message.content),
                tool_calls: None,
            });
            continue;
        };
        let mut text = String::new();
        let mut calls = Vec::new();
        for block in blocks {
            match block {
                ContentBlock::Text { text: part } => text.push_str(part),
                ContentBlock::ToolUse { id, name, input } => {
                    let mut call = json!({
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": input.to_string()},
                    });
                    higgs_engine::chat_template::normalize_tool_call_for_template(&mut call);
                    calls.push(call);
                }
                ContentBlock::ToolResult { content, .. } => result.push(ChatMessage {
                    role: "tool".to_owned(),
                    content: text_of(content),
                    tool_calls: None,
                }),
                ContentBlock::Thinking { .. }
                | ContentBlock::RedactedThinking { .. }
                | ContentBlock::Image { .. }
                | ContentBlock::Document { .. }
                | ContentBlock::ServerToolUse { .. }
                | ContentBlock::WebSearchToolResult { .. }
                | ContentBlock::CodeExecutionToolResult { .. }
                | ContentBlock::Other => {}
            }
        }
        let has_calls = !calls.is_empty();
        if has_calls
            || !text.is_empty()
            || !blocks
                .iter()
                .any(|block| matches!(block, ContentBlock::ToolResult { .. }))
        {
            result.push(ChatMessage {
                role: message.role.clone(),
                content: text,
                tool_calls: has_calls.then_some(calls),
            });
        }
    }
    result
}

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

/// The tools to offer and the policy for using them, from a Messages
/// request's `tools` and `tool_choice`. No tools are offered when there are
/// none or `tool_choice` is `none`.
pub fn tool_settings(
    tools: Option<&[Value]>,
    tool_choice: Option<&Value>,
) -> Result<(Vec<Value>, ToolPolicy), ServerError> {
    let policy = ToolPolicy::from_anthropic(tool_choice)?;
    let offered = tools
        .filter(|declared| !declared.is_empty() && policy.exposes_tools())
        .map(openai_tools)
        .unwrap_or_default();
    Ok((offered, policy))
}

/// A fresh `tool_use` block id.
pub fn new_tool_use_id() -> String {
    format!("toolu_{}", uuid::Uuid::new_v4().simple())
}

/// The non-streaming content block for a parsed call.
pub fn tool_use_block(call: &GeneratedToolCall) -> ToolUseBlock {
    ToolUseBlock {
        block_type: "tool_use",
        id: new_tool_use_id(),
        name: call.name.clone(),
        input: call.arguments.clone(),
    }
}

/// The three server-sent events (`(event name, data)`) that stream one
/// complete call as content block `index`.
pub fn tool_use_stream_events(index: u32, call: &GeneratedToolCall) -> [(&'static str, String); 3] {
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

    fn history() -> Vec<AnthropicMessage> {
        serde_json::from_value(json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
                {"type": "text", "text": "thanks"}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": [{"type": "text", "text": "ok"}]}
            ]}
        ]))
        .unwrap()
    }

    #[test]
    fn tool_history_maps_to_tool_calls_and_tool_messages() {
        let messages = messages_with_tool_history(&history(), None);
        let roles: Vec<&str> = messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["user", "assistant", "tool", "user", "tool"]);
        assert_eq!(messages[1].content, "Checking.");
        let calls = messages[1].tool_calls.as_ref().unwrap();
        assert_eq!(calls[0]["name"], "get_weather");
        assert_eq!(calls[0]["arguments"]["city"], "Paris");
        assert_eq!(messages[2].content, "sunny");
        assert_eq!(messages[3].content, "thanks");
        assert_eq!(messages[4].content, "ok");
    }

    #[test]
    fn system_prompt_becomes_leading_system_message() {
        let messages: Vec<AnthropicMessage> =
            serde_json::from_value(json!([{"role": "user", "content": "Hello"}])).unwrap();
        let system = SystemPrompt::Text("Be helpful".to_owned());
        let converted = messages_with_tool_history(&messages, Some(&system));
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].role, "system");
        assert_eq!(converted[0].content, "Be helpful");
        assert_eq!(converted[1].role, "user");
    }

    #[test]
    fn text_blocks_are_concatenated_and_empty_history_is_empty() {
        let messages: Vec<AnthropicMessage> = serde_json::from_value(json!([
            {"role": "user", "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "World"}
            ]}
        ]))
        .unwrap();
        let converted = messages_with_tool_history(&messages, None);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].content, "Hello World");
        assert!(messages_with_tool_history(&[], None).is_empty());
    }

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
        let call = GeneratedToolCall::new(0, "f".to_owned(), json!({"a": 1}));
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
