//! Tests for request body deserialization and validation.
//!
//! These tests verify that the type system correctly accepts valid requests
//! and rejects malformed ones at the serde level, before any engine interaction.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::tests_outside_test_module
)]

use higgs::types::anthropic::{
    AnthropicContent, AnthropicMessage, ContentBlock, CountTokensRequest, CreateMessageRequest,
};
use higgs::types::openai::{
    ChatCompletionMessage, ChatCompletionRequest, CompletionRequest, EmbeddingInput,
    EmbeddingRequest, StopSequence,
};

// ---------------------------------------------------------------------------
// OpenAI Chat Completions
// ---------------------------------------------------------------------------

#[test]
fn chat_request_minimal_valid() {
    let json = r#"{"model": "test-model", "messages": [{"role": "user", "content": "hello"}]}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, "test-model");
    assert_eq!(req.messages.len(), 1);
    assert!(req.max_tokens.is_none());
    assert!(req.temperature.is_none());
    assert!(req.top_p.is_none());
    assert!(req.stream.is_none());
    assert!(req.stop.is_none());
    assert!(req.tools.is_none());
    assert!(req.response_format.is_none());
}

#[test]
fn chat_request_all_optional_fields() {
    let json = r#"{
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": true,
        "stop": ["END", "\n"],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "response_format": {"type": "json_object"}
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.max_tokens, Some(256));
    assert!((req.temperature.unwrap() - 0.7).abs() < f32::EPSILON);
    assert_eq!(req.stream, Some(true));
    assert!(req.tools.is_some());
    assert!(req.response_format.is_some());
}

#[test]
fn chat_request_missing_model_fails() {
    let json = r#"{"messages": [{"role": "user", "content": "hi"}]}"#;
    let result = serde_json::from_str::<ChatCompletionRequest>(json);
    assert!(result.is_err());
}

#[test]
fn chat_request_missing_messages_fails() {
    let json = r#"{"model": "m"}"#;
    let result = serde_json::from_str::<ChatCompletionRequest>(json);
    assert!(result.is_err());
}

#[test]
fn chat_request_empty_messages_deserializes() {
    // Empty messages array deserializes fine; the handler validates emptiness
    let json = r#"{"model": "m", "messages": []}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert!(req.messages.is_empty());
}

#[test]
fn chat_request_stop_single_string() {
    let json = r#"{"model": "m", "messages": [], "stop": "STOP"}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    match req.stop.unwrap() {
        StopSequence::Single(s) => assert_eq!(s, "STOP"),
        StopSequence::Multiple(_) => panic!("expected Single variant"),
    }
}

#[test]
fn chat_request_stop_array() {
    let json = r#"{"model": "m", "messages": [], "stop": ["a", "b", "c"]}"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    match req.stop.unwrap() {
        StopSequence::Multiple(v) => assert_eq!(v.len(), 3),
        StopSequence::Single(_) => panic!("expected Multiple variant"),
    }
}

#[test]
fn stop_sequence_extract_none() {
    let result = StopSequence::extract(None);
    assert!(result.is_empty());
}

#[test]
fn stop_sequence_extract_single() {
    let result = StopSequence::extract(Some(StopSequence::Single("end".to_owned())));
    assert_eq!(result, vec!["end"]);
}

#[test]
fn stop_sequence_extract_multiple() {
    let result = StopSequence::extract(Some(StopSequence::Multiple(vec![
        "a".to_owned(),
        "b".to_owned(),
    ])));
    assert_eq!(result, vec!["a", "b"]);
}

#[test]
fn chat_message_with_tool_calls() {
    let json = r#"{
        "role": "assistant",
        "tool_calls": [{
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
        }]
    }"#;
    let msg: ChatCompletionMessage = serde_json::from_str(json).unwrap();
    assert_eq!(msg.role, "assistant");
    assert!(msg.content.is_none());
    let calls = msg.tool_calls.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call_abc");
    assert_eq!(calls[0].function.name, "get_weather");
}

#[test]
fn chat_message_tool_result() {
    let json = r#"{
        "role": "tool",
        "content": "72 degrees Fahrenheit",
        "tool_call_id": "call_abc"
    }"#;
    let msg: ChatCompletionMessage = serde_json::from_str(json).unwrap();
    assert_eq!(msg.role, "tool");
    assert_eq!(msg.tool_call_id, Some("call_abc".to_owned()));
}

#[test]
fn chat_request_wrong_type_for_max_tokens_fails() {
    let json = r#"{"model": "m", "messages": [], "max_tokens": "not_a_number"}"#;
    let result = serde_json::from_str::<ChatCompletionRequest>(json);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// OpenAI Completions
// ---------------------------------------------------------------------------

#[test]
fn completion_request_minimal() {
    let json = r#"{"model": "m", "prompt": "Once upon"}"#;
    let req: CompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.prompt, "Once upon");
    assert!(req.stream.is_none());
}

#[test]
fn completion_request_missing_prompt_fails() {
    let json = r#"{"model": "m"}"#;
    let result = serde_json::from_str::<CompletionRequest>(json);
    assert!(result.is_err());
}

#[test]
fn completion_request_missing_model_fails() {
    let json = r#"{"prompt": "test"}"#;
    let result = serde_json::from_str::<CompletionRequest>(json);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// OpenAI Embeddings
// ---------------------------------------------------------------------------

#[test]
fn embedding_request_single_input() {
    let json = r#"{"model": "m", "input": "hello world"}"#;
    let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
    assert!(matches!(req.input, EmbeddingInput::Single(ref s) if s == "hello world"));
}

#[test]
fn embedding_request_multiple_inputs() {
    let json = r#"{"model": "m", "input": ["hello", "world"]}"#;
    let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
    match &req.input {
        EmbeddingInput::Multiple(v) => assert_eq!(v.len(), 2),
        EmbeddingInput::Single(_) => panic!("expected Multiple variant"),
    }
}

#[test]
fn embedding_request_with_encoding_format() {
    let json = r#"{"model": "m", "input": "hi", "encoding_format": "float"}"#;
    let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.encoding_format, Some("float".to_owned()));
}

#[test]
fn embedding_request_missing_input_fails() {
    let json = r#"{"model": "m"}"#;
    let result = serde_json::from_str::<EmbeddingRequest>(json);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Anthropic Messages
// ---------------------------------------------------------------------------

#[test]
fn anthropic_request_minimal() {
    let json = r#"{
        "model": "claude-3",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1024
    }"#;
    let req: CreateMessageRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, "claude-3");
    assert_eq!(req.max_tokens, 1024);
    assert!(req.system.is_none());
    assert!(req.stream.is_none());
}

#[test]
fn anthropic_request_with_system_prompt() {
    let json = r#"{
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "system": "You are a pirate."
    }"#;
    let req: CreateMessageRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.system, Some("You are a pirate.".to_owned()));
}

#[test]
fn anthropic_request_missing_max_tokens_fails() {
    let json = r#"{
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}]
    }"#;
    let result = serde_json::from_str::<CreateMessageRequest>(json);
    assert!(result.is_err(), "max_tokens is required for Anthropic API");
}

#[test]
fn anthropic_content_plain_text() {
    let json = r#"{"role": "user", "content": "hello"}"#;
    let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
    assert!(matches!(msg.content, AnthropicContent::Text(ref s) if s == "hello"));
}

#[test]
fn anthropic_content_blocks() {
    let json = r#"{
        "role": "user",
        "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"}
        ]
    }"#;
    let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
    match &msg.content {
        AnthropicContent::Blocks(blocks) => {
            assert_eq!(blocks.len(), 2);
            match &blocks[0] {
                ContentBlock::Text { text } => assert_eq!(text, "first"),
                ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. } => {
                    panic!("expected Text block")
                }
            }
        }
        AnthropicContent::Text(_) => panic!("expected Blocks variant"),
    }
}

#[test]
fn anthropic_tool_use_block() {
    let json = r#"{
        "role": "assistant",
        "content": [{
            "type": "tool_use",
            "id": "toolu_123",
            "name": "get_weather",
            "input": {"city": "SF"}
        }]
    }"#;
    let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
    match &msg.content {
        AnthropicContent::Blocks(blocks) => match &blocks[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_123");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "SF");
            }
            ContentBlock::Text { .. } | ContentBlock::ToolResult { .. } => {
                panic!("expected ToolUse block")
            }
        },
        AnthropicContent::Text(_) => panic!("expected Blocks variant"),
    }
}

#[test]
fn anthropic_tool_result_block() {
    let json = r#"{
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "toolu_123",
            "content": "72 degrees"
        }]
    }"#;
    let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
    match &msg.content {
        AnthropicContent::Blocks(blocks) => match &blocks[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
            } => {
                assert_eq!(tool_use_id, "toolu_123");
                assert_eq!(content, "72 degrees");
            }
            ContentBlock::Text { .. } | ContentBlock::ToolUse { .. } => {
                panic!("expected ToolResult block")
            }
        },
        AnthropicContent::Text(_) => panic!("expected Blocks variant"),
    }
}

#[test]
fn anthropic_count_tokens_request() {
    let json = r#"{
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "system": "be helpful"
    }"#;
    let req: CountTokensRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.system, Some("be helpful".to_owned()));
    assert_eq!(req.messages.len(), 1);
}

#[test]
fn anthropic_request_with_stop_sequences() {
    let json = r#"{
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "stop_sequences": ["END", "STOP"]
    }"#;
    let req: CreateMessageRequest = serde_json::from_str(json).unwrap();
    let stops = req.stop_sequences.unwrap();
    assert_eq!(stops.len(), 2);
    assert_eq!(stops[0], "END");
}

// ---------------------------------------------------------------------------
// Anthropic adapter
// ---------------------------------------------------------------------------

#[test]
fn anthropic_finish_reason_mapping() {
    use higgs::anthropic_adapter::openai_finish_to_anthropic_stop;

    assert_eq!(openai_finish_to_anthropic_stop("stop"), "end_turn");
    assert_eq!(openai_finish_to_anthropic_stop("length"), "max_tokens");
    assert_eq!(openai_finish_to_anthropic_stop("tool_calls"), "tool_use");
    assert_eq!(openai_finish_to_anthropic_stop("unknown"), "unknown");
}

fn make_single_user_message() -> Vec<AnthropicMessage> {
    vec![AnthropicMessage {
        role: "user".to_owned(),
        content: AnthropicContent::Text("hello".to_owned()),
    }]
}

#[test]
fn anthropic_messages_to_engine_without_system() {
    use higgs::anthropic_adapter::anthropic_messages_to_engine;

    let messages = make_single_user_message();
    let result = anthropic_messages_to_engine(&messages, None);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].role, "user");
    assert_eq!(result[0].content, "hello");
}

#[test]
fn anthropic_messages_to_engine_with_system() {
    use higgs::anthropic_adapter::anthropic_messages_to_engine;

    let messages = make_single_user_message();
    let result = anthropic_messages_to_engine(&messages, Some("be helpful"));
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].role, "system");
    assert_eq!(result[0].content, "be helpful");
    assert_eq!(result[1].role, "user");
}

#[test]
fn anthropic_messages_to_engine_concatenates_text_blocks() {
    use higgs::anthropic_adapter::anthropic_messages_to_engine;

    let messages = vec![AnthropicMessage {
        role: "user".to_owned(),
        content: AnthropicContent::Blocks(vec![
            ContentBlock::Text {
                text: "part one ".to_owned(),
            },
            ContentBlock::Text {
                text: "part two".to_owned(),
            },
        ]),
    }];
    let result = anthropic_messages_to_engine(&messages, None);
    assert_eq!(result[0].content, "part one part two");
}

#[test]
fn anthropic_messages_to_engine_ignores_tool_blocks() {
    use higgs::anthropic_adapter::anthropic_messages_to_engine;

    let messages = vec![AnthropicMessage {
        role: "assistant".to_owned(),
        content: AnthropicContent::Blocks(vec![
            ContentBlock::Text {
                text: "thinking...".to_owned(),
            },
            ContentBlock::ToolUse {
                id: "t1".to_owned(),
                name: "calc".to_owned(),
                input: serde_json::json!({}),
            },
        ]),
    }];
    let result = anthropic_messages_to_engine(&messages, None);
    assert_eq!(result[0].content, "thinking...");
}

// ---------------------------------------------------------------------------
// Extra unknown fields
// ---------------------------------------------------------------------------

#[test]
fn chat_request_with_extra_unknown_fields_accepted() {
    let json = r#"{
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_field": 42,
        "another_unknown": {"nested": true},
        "vendor_specific_param": "value"
    }"#;
    let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, "m");
    assert_eq!(req.messages.len(), 1);
}

#[test]
fn anthropic_request_with_tools_array() {
    let json = r#"{
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "tools": [
            {
                "name": "get_weather",
                "description": "Gets current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        ]
    }"#;
    let req: CreateMessageRequest = serde_json::from_str(json).unwrap();
    let tools = req.tools.unwrap();
    assert_eq!(tools.len(), 1);
}

#[test]
fn completion_request_with_stop_as_array() {
    let json = r#"{
        "model": "m",
        "prompt": "Once upon",
        "stop": ["\n\n", "END", "---"]
    }"#;
    let req: CompletionRequest = serde_json::from_str(json).unwrap();
    match req.stop.unwrap() {
        StopSequence::Multiple(v) => assert_eq!(v.len(), 3),
        StopSequence::Single(_) => panic!("expected Multiple variant"),
    }
}
