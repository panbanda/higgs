//! The chat messages of an `OpenAI`-shaped request, as the engine's chat
//! template takes them.

use higgs_engine::chat_template::{ChatMessage, normalize_tool_call_for_template};

use super::GeneratedToolCall;
use crate::types::openai::{ChatCompletionMessage, MessageContent, ToolCall, ToolCallFunction};

/// The messages the chat template renders and the images they carry, with
/// `<image>` placeholders standing in for the images.
pub fn prompt_messages(messages: &[ChatCompletionMessage]) -> (Vec<ChatMessage>, Vec<Vec<u8>>) {
    let images = extract_images(messages);
    if images.is_empty() {
        (convert_messages(messages), images)
    } else {
        (
            convert_messages(&inject_image_placeholders(messages)),
            images,
        )
    }
}

pub fn convert_messages(messages: &[ChatCompletionMessage]) -> Vec<ChatMessage> {
    messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m
                .content
                .as_ref()
                .map_or_else(String::new, MessageContent::text),
            tool_calls: m
                .tool_calls
                .as_ref()
                .map(|calls| calls.iter().filter_map(template_tool_call).collect()),
        })
        .collect()
}

/// A tool call in the shape chat templates expect: `function.{name,arguments}`
/// hoisted to the top level and string-encoded arguments parsed to a JSON
/// value. Without this, Qwen's chat template crashes on
/// `tool_call.arguments|items`.
pub fn template_tool_call(call: &ToolCall) -> Option<serde_json::Value> {
    let mut value = serde_json::to_value(call).ok()?;
    normalize_tool_call_for_template(&mut value);
    Some(value)
}

/// The `OpenAI` wire form of a generated call.
pub fn wire_tool_call(call: &GeneratedToolCall) -> ToolCall {
    ToolCall {
        id: call.id.clone(),
        r#type: "function".to_owned(),
        function: ToolCallFunction {
            name: call.name.clone(),
            arguments: call.arguments.to_string(),
        },
    }
}

/// Extract image bytes from base64 data URIs in message content parts.
/// Returns decoded image bytes for each image found across all messages.
fn extract_images(messages: &[ChatCompletionMessage]) -> Vec<Vec<u8>> {
    use base64::Engine as _;
    let mut images = Vec::new();
    for msg in messages {
        let Some(content) = &msg.content else {
            continue;
        };
        for url in content.image_urls() {
            if let Some(data) = url.strip_prefix("data:") {
                // data:[<mediatype>];base64,<data>
                if let Some(base64_start) = data.find(";base64,") {
                    let encoded = &data[base64_start + 8..];
                    match base64::engine::general_purpose::STANDARD.decode(encoded) {
                        Ok(bytes) => images.push(bytes),
                        Err(e) => tracing::warn!(error = %e, "Failed to decode base64 image"),
                    }
                }
            }
            // HTTP/HTTPS URLs are not supported yet; could be fetched in the future
        }
    }
    images
}

/// Build text content with `<image>` placeholders injected for each image.
/// For VLMs, each image in a message gets a `<image>\n` prefix before the text.
fn inject_image_placeholders(messages: &[ChatCompletionMessage]) -> Vec<ChatCompletionMessage> {
    messages
        .iter()
        .map(|m| {
            let Some(content) = &m.content else {
                return m.clone();
            };
            if !content.has_images() {
                return m.clone();
            }

            let image_count = content.image_urls().len();
            let text = content.text();
            let prefix = "<image>\n".repeat(image_count);
            let combined = format!("{prefix}{text}");

            ChatCompletionMessage {
                role: m.role.clone(),
                content: Some(MessageContent::Text(combined)),
                reasoning_content: m.reasoning_content.clone(),
                tool_calls: m.tool_calls.clone(),
                tool_call_id: m.tool_call_id.clone(),
            }
        })
        .collect()
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::openai::{ToolCall, ToolCallFunction};

    fn simple_message(role: &str, content: Option<&str>) -> ChatCompletionMessage {
        ChatCompletionMessage {
            role: role.to_owned(),
            content: content.map(|s| MessageContent::Text(s.to_owned())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_owned(),
            r#type: "function".to_owned(),
            function: ToolCallFunction {
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            },
        }
    }

    fn tool_message(role: &str, calls: Vec<ToolCall>) -> ChatCompletionMessage {
        ChatCompletionMessage {
            role: role.to_owned(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(calls),
            tool_call_id: None,
        }
    }

    #[test]
    fn test_convert_messages() {
        let msgs = vec![
            simple_message("user", Some("Hello")),
            simple_message("assistant", None),
        ];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted.first().map(|m| m.role.as_str()), Some("user"));
        assert_eq!(converted.first().map(|m| m.content.as_str()), Some("Hello"));
        assert_eq!(converted.get(1).map(|m| m.content.as_str()), Some(""));
    }

    #[test]
    fn test_convert_messages_with_tool_calls() {
        let msgs = vec![tool_message(
            "assistant",
            vec![tool_call("call_1", "get_weather", r#"{"city":"NYC"}"#)],
        )];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        let calls = converted
            .first()
            .and_then(|m| m.tool_calls.as_ref())
            .unwrap();
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn test_convert_messages_empty_list() {
        let result = convert_messages(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_convert_messages_with_null_content() {
        let msgs = vec![simple_message("assistant", None)];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted.first().map(|m| m.content.as_str()), Some(""));
    }

    #[test]
    fn test_convert_messages_with_tool_calls_complex_arguments() {
        let msgs = vec![tool_message(
            "assistant",
            vec![
                tool_call(
                    "call_1",
                    "search",
                    r#"{"query":"rust programming","filters":{"language":"en","year":2024}}"#,
                ),
                tool_call("call_2", "calculate", r#"{"expression":"2+2"}"#),
            ],
        )];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        let calls = converted
            .first()
            .and_then(|m| m.tool_calls.as_ref())
            .unwrap();
        assert_eq!(calls.len(), 2);
    }
}
