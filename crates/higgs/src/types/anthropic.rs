use serde::{Deserialize, Serialize};

/// The Anthropic API `system` field: either a plain string or an array of
/// text blocks (with optional `cache_control` / `citations` we ignore).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct SystemBlock {
    pub text: String,
}

impl SystemPrompt {
    pub fn to_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n"),
        }
    }
}

/// POST /v1/messages request body (Anthropic Messages API).
#[derive(Debug, Clone, Deserialize)]
pub struct CreateMessageRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
}

/// A message in the Anthropic format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

/// Content can be a plain string or an array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A content block in the Anthropic format.
///
/// Unknown future block types are captured as `Other`
/// and silently skipped during text extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: AnthropicContent,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "image")]
    Image { source: serde_json::Value },
    #[serde(rename = "document")]
    Document { source: serde_json::Value },
    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "web_search_tool_result")]
    WebSearchToolResult {
        tool_use_id: String,
        content: serde_json::Value,
    },
    #[serde(rename = "code_execution_tool_result")]
    CodeExecutionToolResult {
        tool_use_id: String,
        content: serde_json::Value,
    },
    #[serde(other)]
    Other,
}

impl AnthropicContent {
    /// Flatten to a single string, joining text blocks with newlines.
    pub fn to_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
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
                .collect::<Vec<_>>()
                .join("\n\n"),
        }
    }
}

/// POST /v1/messages response (non-streaming).
#[derive(Debug, Clone, Serialize)]
pub struct CreateMessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: &'static str,
    pub role: &'static str,
    pub content: Vec<ContentBlockResponse>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: AnthropicUsage,
}

/// A content block in the response.
#[derive(Debug, Clone, Serialize)]
pub struct ContentBlockResponse {
    #[serde(rename = "type")]
    pub block_type: &'static str,
    pub text: String,
}

/// Anthropic usage stats.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// POST `/v1/messages/count_tokens` request.
#[derive(Debug, Clone, Deserialize)]
pub struct CountTokensRequest {
    #[allow(dead_code)] // Required for API deserialization compatibility
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
}

/// POST `/v1/messages/count_tokens` response.
#[derive(Debug, Clone, Serialize)]
pub struct CountTokensResponse {
    pub input_tokens: u32,
}

// --- Streaming event types ---

/// Server-sent event wrapper for Anthropic streaming.
#[derive(Debug, Clone, Serialize)]
pub struct MessageStartEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub message: MessageStartPayload,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageStartPayload {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: &'static str,
    pub role: &'static str,
    pub content: Vec<serde_json::Value>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlockStartEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub index: u32,
    pub content_block: ContentBlockStartPayload,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlockStartPayload {
    #[serde(rename = "type")]
    pub block_type: &'static str,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlockDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub index: u32,
    pub delta: TextDelta,
}

#[derive(Debug, Clone, Serialize)]
pub struct TextDelta {
    #[serde(rename = "type")]
    pub delta_type: &'static str,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlockStopEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub index: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub delta: MessageDelta,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageStopEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn make_usage(input: u32, output: u32) -> AnthropicUsage {
        AnthropicUsage {
            input_tokens: input,
            output_tokens: output,
        }
    }

    fn make_message_start_event(id: &str, usage: AnthropicUsage) -> MessageStartEvent {
        MessageStartEvent {
            event_type: "message_start",
            message: MessageStartPayload {
                id: id.to_owned(),
                message_type: "message",
                role: "assistant",
                content: vec![],
                model: "test".to_owned(),
                stop_reason: None,
                usage,
            },
        }
    }

    /// Deserialize an `AnthropicMessage` from a JSON string.
    fn parse_message(json: &str) -> AnthropicMessage {
        serde_json::from_str(json).unwrap()
    }

    /// Unwrap the Blocks variant or panic.
    fn expect_blocks(msg: &AnthropicMessage) -> &[ContentBlock] {
        match &msg.content {
            AnthropicContent::Blocks(blocks) => blocks,
            AnthropicContent::Text(_) => panic!("expected Blocks variant"),
        }
    }

    /// Deserialize a `CreateMessageRequest` with an extra field merged in.
    fn anthropic_request_with(extra_field: &str) -> CreateMessageRequest {
        let json = format!(
            r#"{{"model": "test", "messages": [{{"role": "user", "content": "Hello"}}], "max_tokens": 100, {extra_field}}}"#,
        );
        serde_json::from_str(&json).unwrap()
    }

    /// Serialize a value to JSON, assert `json["type"]` equals the expected event type.
    fn assert_event_type(event: &impl serde::Serialize, expected_type: &str) -> serde_json::Value {
        let json: serde_json::Value = serde_json::to_value(event).unwrap();
        assert_eq!(json["type"], expected_type);
        json
    }

    #[test]
    fn test_anthropic_request_deserialization() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }"#;
        let req: CreateMessageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test");
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_anthropic_content_text() {
        let msg = parse_message(r#"{"role": "user", "content": "Hello"}"#);
        assert!(matches!(msg.content, AnthropicContent::Text(ref s) if s == "Hello"));
    }

    #[test]
    fn test_anthropic_content_blocks() {
        let msg = parse_message(
            r#"{"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}"#,
        );
        assert!(matches!(msg.content, AnthropicContent::Blocks(ref blocks) if blocks.len() == 2));
    }

    #[test]
    fn test_anthropic_response_serialization() {
        let resp = CreateMessageResponse {
            id: "msg_123".to_owned(),
            message_type: "message",
            role: "assistant",
            content: vec![ContentBlockResponse {
                block_type: "text",
                text: "Hello!".to_owned(),
            }],
            model: "test".to_owned(),
            stop_reason: Some("end_turn".to_owned()),
            usage: make_usage(5, 1),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"type\":\"message\""));
        assert!(json.contains("end_turn"));
    }

    #[test]
    fn test_count_tokens_request() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are helpful."
        }"#;
        let req: CountTokensRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.system.unwrap().to_text(), "You are helpful.");
    }

    #[test]
    fn test_message_start_event_serialization() {
        let event = make_message_start_event("msg_123", make_usage(5, 0));
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("message_start"));
    }

    #[test]
    fn test_anthropic_content_tool_use_block() {
        let msg = parse_message(
            r#"{"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "London"}}]}"#,
        );
        let blocks = expect_blocks(&msg);
        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], ContentBlock::ToolUse { name, .. } if name == "get_weather"));
    }

    #[test]
    fn test_anthropic_content_tool_result_block() {
        let msg = parse_message(
            r#"{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "72 degrees"}]}"#,
        );
        let blocks = expect_blocks(&msg);
        assert_eq!(blocks.len(), 1);
        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
        } = &blocks[0]
        {
            assert_eq!(tool_use_id, "tu_1");
            assert_eq!(content.to_text(), "72 degrees");
        } else {
            panic!("expected ToolResult block");
        }
    }

    #[test]
    fn test_anthropic_request_with_stop_sequences() {
        let req = anthropic_request_with(r#""stop_sequences": ["END", "STOP"]"#);
        let stops = req.stop_sequences.unwrap();
        assert_eq!(stops.len(), 2);
        assert_eq!(stops[0], "END");
        assert_eq!(stops[1], "STOP");
    }

    #[test]
    fn test_create_message_request_max_tokens_zero() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 0
        }"#;
        let req: CreateMessageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 0);
    }

    #[test]
    fn test_create_message_request_temperature_zero() {
        let req = anthropic_request_with(r#""temperature": 0.0"#);
        assert!((req.temperature.unwrap()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_anthropic_content_text_deserialization() {
        let msg = parse_message(r#"{"role": "user", "content": "simple text"}"#);
        match &msg.content {
            AnthropicContent::Text(s) => assert_eq!(s, "simple text"),
            AnthropicContent::Blocks(_) => panic!("expected Text variant"),
        }
    }

    #[test]
    fn test_anthropic_content_blocks_mixed_types() {
        let msg = parse_message(
            r#"{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check"},
                {"type": "tool_use", "id": "tu_1", "name": "calculator", "input": {"expr": "2+2"}},
                {"type": "text", "text": "The answer is 4"}
            ]
        }"#,
        );
        let blocks = expect_blocks(&msg);
        assert_eq!(blocks.len(), 3);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Let me check"));
        assert!(matches!(&blocks[1], ContentBlock::ToolUse { name, .. } if name == "calculator"));
        assert!(matches!(&blocks[2], ContentBlock::Text { text } if text == "The answer is 4"));
    }

    #[test]
    fn test_tool_use_with_complex_json_input() {
        let msg = parse_message(
            r#"{
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "tu_complex",
                "name": "database_query",
                "input": {
                    "query": "SELECT * FROM users",
                    "params": [1, "hello", null, true],
                    "nested": {"key": {"deep": [1,2,3]}}
                }
            }]
        }"#,
        );
        let blocks = expect_blocks(&msg);
        if let ContentBlock::ToolUse { input, .. } = &blocks[0] {
            assert_eq!(input["query"], "SELECT * FROM users");
            assert!(input["params"].is_array());
            assert!(input["nested"]["key"]["deep"].is_array());
        } else {
            panic!("expected ToolUse block");
        }
    }

    #[test]
    fn test_tool_result_with_very_long_content() {
        let long_content = "x".repeat(10_000);
        let json = format!(
            r#"{{"role": "user", "content": [{{"type": "tool_result", "tool_use_id": "tu_1", "content": "{long_content}"}}]}}"#,
        );
        let msg: AnthropicMessage = serde_json::from_str(&json).unwrap();
        let blocks = expect_blocks(&msg);
        if let ContentBlock::ToolResult { content, .. } = &blocks[0] {
            assert_eq!(content.to_text().len(), 10_000);
        } else {
            panic!("expected ToolResult block");
        }
    }

    #[test]
    fn test_count_tokens_request_empty_messages() {
        let json = r#"{
            "model": "test",
            "messages": []
        }"#;
        let req: CountTokensRequest = serde_json::from_str(json).unwrap();
        assert!(req.messages.is_empty());
        assert!(req.system.is_none());
    }

    #[test]
    fn test_count_tokens_request_with_system_prompt() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "system": "You are a helpful assistant."
        }"#;
        let req: CountTokensRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            req.system.unwrap().to_text(),
            "You are a helpful assistant."
        );
    }

    #[test]
    fn test_message_start_event_type_field() {
        let event = make_message_start_event("msg_1", make_usage(0, 0));
        assert_event_type(&event, "message_start");
    }

    #[test]
    fn test_content_block_start_event_type_field() {
        let event = ContentBlockStartEvent {
            event_type: "content_block_start",
            index: 0,
            content_block: ContentBlockStartPayload {
                block_type: "text",
                text: String::new(),
            },
        };
        assert_event_type(&event, "content_block_start");
    }

    #[test]
    fn test_content_block_delta_event_type_field() {
        let event = ContentBlockDeltaEvent {
            event_type: "content_block_delta",
            index: 0,
            delta: TextDelta {
                delta_type: "text_delta",
                text: "Hello".to_owned(),
            },
        };
        let json = assert_event_type(&event, "content_block_delta");
        assert_eq!(json["delta"]["type"], "text_delta");
    }

    #[test]
    fn test_content_block_stop_event_type_field() {
        let event = ContentBlockStopEvent {
            event_type: "content_block_stop",
            index: 0,
        };
        assert_event_type(&event, "content_block_stop");
    }

    #[test]
    fn test_message_delta_event_type_field() {
        let event = MessageDeltaEvent {
            event_type: "message_delta",
            delta: MessageDelta {
                stop_reason: Some("end_turn".to_owned()),
            },
            usage: make_usage(10, 5),
        };
        assert_event_type(&event, "message_delta");
    }

    #[test]
    fn test_message_stop_event_type_field() {
        let event = MessageStopEvent {
            event_type: "message_stop",
        };
        assert_event_type(&event, "message_stop");
    }

    #[test]
    fn test_anthropic_request_with_tools() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "max_tokens": 100,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}}
                    }
                }
            ]
        }"#;
        let req: CreateMessageRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_some());
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
    }

    #[test]
    fn test_system_as_string() {
        let req = anthropic_request_with(r#""system": "Be concise.""#);
        assert_eq!(req.system.unwrap().to_text(), "Be concise.");
    }

    #[test]
    fn test_system_as_array_of_blocks() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "You are a helpful assistant."},
                {"type": "text", "text": "Be concise.", "cache_control": {"type": "ephemeral"}}
            ]
        }"#;
        let req: CreateMessageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(
            req.system.unwrap().to_text(),
            "You are a helpful assistant.\n\nBe concise."
        );
    }

    #[test]
    fn test_tool_result_content_as_array_of_blocks() {
        let msg = parse_message(
            r#"{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]}]}"#,
        );
        let blocks = expect_blocks(&msg);
        if let ContentBlock::ToolResult { content, .. } = &blocks[0] {
            assert_eq!(content.to_text(), "first\n\nsecond");
        } else {
            panic!("expected ToolResult block");
        }
    }

    #[test]
    fn test_thinking_block() {
        let msg = parse_message(
            r#"{"role": "assistant", "content": [{"type": "thinking", "thinking": "Let me think...", "signature": "sig123"}]}"#,
        );
        let blocks = expect_blocks(&msg);
        assert!(matches!(
            &blocks[0],
            ContentBlock::Thinking { thinking, signature }
                if thinking == "Let me think..." && signature == "sig123"
        ));
    }

    #[test]
    fn test_redacted_thinking_block() {
        let msg = parse_message(
            r#"{"role": "assistant", "content": [{"type": "redacted_thinking", "data": "abc123"}]}"#,
        );
        let blocks = expect_blocks(&msg);
        assert!(matches!(
            &blocks[0],
            ContentBlock::RedactedThinking { data } if data == "abc123"
        ));
    }

    #[test]
    fn test_image_block() {
        let msg = parse_message(
            r#"{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR..."}}]}"#,
        );
        let blocks = expect_blocks(&msg);
        assert!(matches!(&blocks[0], ContentBlock::Image { source } if source["type"] == "base64"));
    }

    #[test]
    fn test_document_block() {
        let msg = parse_message(
            r#"{"role": "user", "content": [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBER..."}}]}"#,
        );
        let blocks = expect_blocks(&msg);
        assert!(
            matches!(&blocks[0], ContentBlock::Document { source } if source["media_type"] == "application/pdf")
        );
    }

    #[test]
    fn test_server_tool_use_block() {
        let msg = parse_message(
            r#"{"role": "assistant", "content": [{"type": "server_tool_use", "id": "stu_1", "name": "web_search", "input": {"query": "rust lang"}}]}"#,
        );
        let blocks = expect_blocks(&msg);
        assert!(matches!(
            &blocks[0],
            ContentBlock::ServerToolUse { id, name, .. }
                if id == "stu_1" && name == "web_search"
        ));
    }

    #[test]
    fn test_web_search_tool_result_block() {
        let msg = parse_message(
            r#"{"role": "user", "content": [{"type": "web_search_tool_result", "tool_use_id": "stu_1", "content": [{"type": "web_search_result", "url": "https://example.com", "title": "Example", "encrypted_content": "abc"}]}]}"#,
        );
        let blocks = expect_blocks(&msg);
        assert!(matches!(
            &blocks[0],
            ContentBlock::WebSearchToolResult { tool_use_id, .. }
                if tool_use_id == "stu_1"
        ));
    }

    #[test]
    fn test_code_execution_tool_result_block() {
        let msg = parse_message(
            r#"{"role": "user", "content": [{"type": "code_execution_tool_result", "tool_use_id": "stu_2", "content": [{"type": "code_execution_output", "output": "hello world"}]}]}"#,
        );
        let blocks = expect_blocks(&msg);
        assert!(matches!(
            &blocks[0],
            ContentBlock::CodeExecutionToolResult { tool_use_id, .. }
                if tool_use_id == "stu_2"
        ));
    }

    #[test]
    fn test_thinking_block_excluded_from_to_text() {
        let content = AnthropicContent::Blocks(vec![
            ContentBlock::Thinking {
                thinking: "internal thought".to_owned(),
                signature: "sig".to_owned(),
            },
            ContentBlock::Text {
                text: "visible".to_owned(),
            },
        ]);
        assert_eq!(content.to_text(), "visible");
    }

    #[test]
    fn test_unknown_block_type_becomes_other() {
        let msg = parse_message(
            r#"{"role": "user", "content": [{"type": "some_future_type", "data": "whatever"}]}"#,
        );
        let blocks = expect_blocks(&msg);
        assert!(matches!(&blocks[0], ContentBlock::Other));
    }
}
