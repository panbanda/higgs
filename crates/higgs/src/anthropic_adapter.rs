use crate::types::anthropic::{AnthropicContent, AnthropicMessage};

/// Map an `OpenAI` `finish_reason` to an Anthropic `stop_reason`.
pub fn openai_finish_to_anthropic_stop(finish_reason: &str) -> String {
    match finish_reason {
        "stop" => "end_turn".to_owned(),
        "length" => "max_tokens".to_owned(),
        "tool_calls" => "tool_use".to_owned(),
        other => other.to_owned(),
    }
}

/// Convert Anthropic messages to the engine's `ChatMessage` format.
pub fn anthropic_messages_to_engine(
    messages: &[AnthropicMessage],
    system: Option<&str>,
) -> Vec<higgs_engine::chat_template::ChatMessage> {
    let mut result = Vec::new();

    if let Some(sys) = system {
        result.push(higgs_engine::chat_template::ChatMessage {
            role: "system".to_owned(),
            content: sys.to_owned(),
            tool_calls: None,
        });
    }

    for msg in messages {
        let content = match &msg.content {
            AnthropicContent::Text(s) => s.clone(),
            AnthropicContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    crate::types::anthropic::ContentBlock::Text { text } => Some(text.as_str()),
                    crate::types::anthropic::ContentBlock::ToolUse { .. }
                    | crate::types::anthropic::ContentBlock::ToolResult { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        };

        result.push(higgs_engine::chat_template::ChatMessage {
            role: msg.role.clone(),
            content,
            tool_calls: None,
        });
    }

    result
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::types::anthropic::ContentBlock;

    fn text_message(role: &str, content: &str) -> AnthropicMessage {
        AnthropicMessage {
            role: role.to_owned(),
            content: AnthropicContent::Text(content.to_owned()),
        }
    }

    fn blocks_message(role: &str, blocks: Vec<ContentBlock>) -> AnthropicMessage {
        AnthropicMessage {
            role: role.to_owned(),
            content: AnthropicContent::Blocks(blocks),
        }
    }

    #[test]
    fn test_finish_reason_mapping() {
        let cases = [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("other", "other"),
            ("", ""),
            ("content_filter", "content_filter"),
            ("something_new", "something_new"),
        ];
        for (input, expected) in cases {
            assert_eq!(openai_finish_to_anthropic_stop(input), expected);
        }
    }

    #[test]
    fn test_anthropic_messages_to_engine_with_system() {
        let messages = vec![text_message("user", "Hello")];

        let result = anthropic_messages_to_engine(&messages, Some("Be helpful"));
        assert_eq!(result.len(), 2);
        assert_eq!(result.first().map(|m| m.role.as_str()), Some("system"));
        assert_eq!(
            result.first().map(|m| m.content.as_str()),
            Some("Be helpful")
        );
    }

    #[test]
    fn test_anthropic_messages_to_engine_without_system() {
        let messages = vec![text_message("user", "Hello")];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result.first().map(|m| m.role.as_str()), Some("user"));
    }

    #[test]
    fn test_anthropic_messages_to_engine_content_blocks() {
        let messages = vec![blocks_message(
            "user",
            vec![
                ContentBlock::Text {
                    text: "Hello ".to_owned(),
                },
                ContentBlock::Text {
                    text: "World".to_owned(),
                },
            ],
        )];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert_eq!(
            result.first().map(|m| m.content.as_str()),
            Some("Hello World")
        );
    }

    #[test]
    fn test_anthropic_messages_to_engine_mixed_blocks_filters_non_text() {
        let messages = vec![blocks_message(
            "user",
            vec![
                ContentBlock::Text {
                    text: "Hello".to_owned(),
                },
                ContentBlock::ToolUse {
                    id: "tu_1".to_owned(),
                    name: "get_weather".to_owned(),
                    input: serde_json::json!({}),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "tu_1".to_owned(),
                    content: "72 degrees".to_owned(),
                },
            ],
        )];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result.first().map(|m| m.content.as_str()), Some("Hello"));
    }

    #[test]
    fn test_anthropic_messages_to_engine_empty_messages() {
        let result = anthropic_messages_to_engine(&[], None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_anthropic_messages_to_engine_multiple_messages() {
        let messages = vec![
            text_message("user", "First"),
            text_message("assistant", "Second"),
            text_message("user", "Third"),
        ];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 3);
        assert_eq!(result.first().map(|m| m.content.as_str()), Some("First"));
        assert_eq!(result.get(1).map(|m| m.content.as_str()), Some("Second"));
        assert_eq!(result.get(2).map(|m| m.content.as_str()), Some("Third"));
    }

    #[test]
    fn test_system_as_empty_string() {
        let messages = vec![text_message("user", "Hello")];

        let result = anthropic_messages_to_engine(&messages, Some(""));
        assert_eq!(result.len(), 2);
        assert_eq!(result.first().map(|m| m.role.as_str()), Some("system"));
        assert_eq!(result.first().map(|m| m.content.as_str()), Some(""));
    }

    #[test]
    fn test_only_non_text_content_blocks_results_in_empty_content() {
        let messages = vec![blocks_message(
            "assistant",
            vec![
                ContentBlock::ToolUse {
                    id: "tu_1".to_owned(),
                    name: "get_weather".to_owned(),
                    input: serde_json::json!({"city": "NYC"}),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "tu_1".to_owned(),
                    content: "72 degrees".to_owned(),
                },
            ],
        )];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result.first().map(|m| m.content.as_str()), Some(""));
    }

    #[test]
    fn test_unicode_content() {
        let messages = vec![text_message(
            "user",
            "Hej! Jag talar svenska. \u{1F600} \u{4F60}\u{597D}",
        )];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert!(
            result
                .first()
                .is_some_and(|m| m.content.contains("svenska"))
        );
    }

    #[test]
    fn test_very_long_content() {
        let long_content = "a".repeat(100_000);
        let messages = vec![text_message("user", &long_content)];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result.first().map(|m| m.content.len()), Some(100_000));
    }
}
