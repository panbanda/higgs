//! Tests for response serialization contracts.
//!
//! Verifies that response types serialize to JSON matching the `OpenAI` and
//! Anthropic API specifications (correct field names, types, and structure).

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::tests_outside_test_module
)]

use higgs::types::anthropic::{
    AnthropicUsage, ContentBlockDeltaEvent, ContentBlockResponse, ContentBlockStartEvent,
    ContentBlockStartPayload, ContentBlockStopEvent, CountTokensResponse, CreateMessageResponse,
    MessageDelta, MessageDeltaEvent, MessageStartEvent, MessageStartPayload, MessageStopEvent,
    TextDelta,
};
use higgs::types::openai::{
    ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionDelta,
    ChatCompletionMessage, ChatCompletionResponse, CompletionChoice, CompletionChunk,
    CompletionChunkChoice, CompletionResponse, CompletionUsage, EmbeddingObject, EmbeddingResponse,
    EmbeddingUsage, MessageContent, ModelList, ModelObject, ToolCall, ToolCallFunction,
};

const fn make_usage(prompt: u32, completion: u32) -> CompletionUsage {
    CompletionUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    }
}

const fn make_anthropic_usage(input: u32, output: u32) -> AnthropicUsage {
    AnthropicUsage {
        input_tokens: input,
        output_tokens: output,
    }
}

fn make_chat_chunk(
    id: &str,
    delta: ChatCompletionDelta,
    finish_reason: Option<String>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_owned(),
        object: "chat.completion.chunk",
        created: 1_700_000_000,
        model: "test".to_owned(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta,
            finish_reason,
            logprobs: None,
        }],
    }
}

fn make_empty_delta_chunk(id: &str, finish_reason: Option<String>) -> ChatCompletionChunk {
    make_chat_chunk(
        id,
        ChatCompletionDelta {
            role: None,
            content: None,
            reasoning_content: None,
            tool_calls: None,
        },
        finish_reason,
    )
}

fn make_tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        id: id.to_owned(),
        r#type: "function".to_owned(),
        function: ToolCallFunction {
            name: name.to_owned(),
            arguments: arguments.to_owned(),
        },
    }
}

fn assert_event_type(event: &impl serde::Serialize, expected_type: &str) -> serde_json::Value {
    let json: serde_json::Value = serde_json::to_value(event).unwrap();
    assert_eq!(json["type"], expected_type);
    json
}

// ---------------------------------------------------------------------------
// OpenAI Chat Completion response
// ---------------------------------------------------------------------------

#[test]
fn chat_completion_response_has_required_fields() {
    let resp = ChatCompletionResponse {
        id: "chatcmpl-abc123".to_owned(),
        object: "chat.completion",
        created: 1_700_000_000,
        model: "test-model".to_owned(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_owned(),
                content: Some(MessageContent::Text("Hello!".to_owned())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: "stop".to_owned(),
            logprobs: None,
        }],
        usage: make_usage(10, 5),
    };

    let json: serde_json::Value = serde_json::to_value(&resp).unwrap();

    assert_eq!(json["id"], "chatcmpl-abc123");
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["created"], 1_700_000_000);
    assert_eq!(json["model"], "test-model");
    assert_eq!(json["choices"][0]["index"], 0);
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["message"]["content"], "Hello!");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["prompt_tokens"], 10);
    assert_eq!(json["usage"]["completion_tokens"], 5);
    assert_eq!(json["usage"]["total_tokens"], 15);
}

#[test]
fn chat_completion_response_omits_null_optional_fields() {
    let msg = ChatCompletionMessage {
        role: "assistant".to_owned(),
        content: Some(MessageContent::Text("hi".to_owned())),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
    };
    let json: serde_json::Value = serde_json::to_value(&msg).unwrap();

    // tool_calls and tool_call_id should be omitted (skip_serializing_if)
    assert!(json.get("tool_calls").is_none());
    assert!(json.get("tool_call_id").is_none());
}

#[test]
fn chat_completion_response_includes_tool_calls() {
    let msg = ChatCompletionMessage {
        role: "assistant".to_owned(),
        content: None,
        reasoning_content: None,
        tool_calls: Some(vec![make_tool_call(
            "call_001",
            "get_weather",
            r#"{"city":"London"}"#,
        )]),
        tool_call_id: None,
    };
    let json: serde_json::Value = serde_json::to_value(&msg).unwrap();

    // content should be omitted when None
    assert!(json.get("content").is_none());
    assert_eq!(json["tool_calls"][0]["id"], "call_001");
    assert_eq!(json["tool_calls"][0]["type"], "function");
    assert_eq!(json["tool_calls"][0]["function"]["name"], "get_weather");
}

// ---------------------------------------------------------------------------
// OpenAI Chat streaming chunk
// ---------------------------------------------------------------------------

#[test]
fn chat_chunk_has_correct_object_type() {
    let chunk = make_chat_chunk(
        "chatcmpl-abc",
        ChatCompletionDelta {
            role: Some("assistant".to_owned()),
            content: None,
            reasoning_content: None,
            tool_calls: None,
        },
        None,
    );
    let json: serde_json::Value = serde_json::to_value(&chunk).unwrap();

    assert_eq!(json["object"], "chat.completion.chunk");
    assert_eq!(json["choices"][0]["delta"]["role"], "assistant");
    assert!(json["choices"][0]["finish_reason"].is_null());
}

#[test]
fn chat_chunk_content_delta() {
    let chunk = make_chat_chunk(
        "chatcmpl-abc",
        ChatCompletionDelta {
            role: None,
            content: Some("Hello".to_owned()),
            reasoning_content: None,
            tool_calls: None,
        },
        None,
    );
    let json: serde_json::Value = serde_json::to_value(&chunk).unwrap();

    assert_eq!(json["choices"][0]["delta"]["content"], "Hello");
    // role should be omitted
    assert!(json["choices"][0]["delta"].get("role").is_none());
}

#[test]
fn chat_chunk_finish_reason() {
    let chunk = make_empty_delta_chunk("chatcmpl-abc", Some("stop".to_owned()));
    let json: serde_json::Value = serde_json::to_value(&chunk).unwrap();
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
}

// ---------------------------------------------------------------------------
// OpenAI Completions response
// ---------------------------------------------------------------------------

#[test]
fn completion_response_has_required_fields() {
    let resp = CompletionResponse {
        id: "cmpl-abc".to_owned(),
        object: "text_completion",
        created: 1_700_000_000,
        model: "test".to_owned(),
        choices: vec![CompletionChoice {
            index: 0,
            text: "once upon a time".to_owned(),
            finish_reason: "stop".to_owned(),
            logprobs: None,
        }],
        usage: make_usage(3, 4),
    };
    let json: serde_json::Value = serde_json::to_value(&resp).unwrap();

    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["choices"][0]["text"], "once upon a time");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["total_tokens"], 7);
}

#[test]
fn completion_chunk_has_correct_object_type() {
    let chunk = CompletionChunk {
        id: "cmpl-abc".to_owned(),
        object: "text_completion",
        created: 1_700_000_000,
        model: "test".to_owned(),
        choices: vec![CompletionChunkChoice {
            index: 0,
            text: "token".to_owned(),
            finish_reason: None,
        }],
    };
    let json: serde_json::Value = serde_json::to_value(&chunk).unwrap();

    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["choices"][0]["text"], "token");
    assert!(json["choices"][0]["finish_reason"].is_null());
}

// ---------------------------------------------------------------------------
// OpenAI Models response
// ---------------------------------------------------------------------------

#[test]
fn model_list_serialization() {
    let list = ModelList {
        object: "list",
        data: vec![ModelObject {
            id: "my-model".to_owned(),
            object: "model",
            created: 1_700_000_000,
            owned_by: "local".to_owned(),
        }],
    };
    let json: serde_json::Value = serde_json::to_value(&list).unwrap();

    assert_eq!(json["object"], "list");
    assert_eq!(json["data"][0]["id"], "my-model");
    assert_eq!(json["data"][0]["object"], "model");
    assert_eq!(json["data"][0]["owned_by"], "local");
}

// ---------------------------------------------------------------------------
// OpenAI Embeddings response
// ---------------------------------------------------------------------------

#[test]
fn embedding_response_serialization() {
    let resp = EmbeddingResponse {
        object: "list",
        data: vec![EmbeddingObject {
            object: "embedding",
            embedding: vec![0.1, 0.2, 0.3],
            index: 0,
        }],
        model: "embed-model".to_owned(),
        usage: EmbeddingUsage {
            prompt_tokens: 5,
            total_tokens: 5,
        },
    };
    let json: serde_json::Value = serde_json::to_value(&resp).unwrap();

    assert_eq!(json["object"], "list");
    assert_eq!(json["data"][0]["object"], "embedding");
    assert_eq!(json["data"][0]["index"], 0);
    assert_eq!(json["model"], "embed-model");
    assert_eq!(json["usage"]["prompt_tokens"], 5);
    let emb = json["data"][0]["embedding"].as_array().unwrap();
    assert_eq!(emb.len(), 3);
}

// ---------------------------------------------------------------------------
// Anthropic Messages response
// ---------------------------------------------------------------------------

#[test]
fn anthropic_response_has_type_field() {
    let resp = CreateMessageResponse {
        id: "msg_abc123".to_owned(),
        message_type: "message",
        role: "assistant",
        content: vec![ContentBlockResponse {
            block_type: "text",
            text: "Hello!".to_owned(),
        }],
        model: "claude-3".to_owned(),
        stop_reason: Some("end_turn".to_owned()),
        usage: make_anthropic_usage(10, 5),
    };
    let json: serde_json::Value = serde_json::to_value(&resp).unwrap();

    // Anthropic uses "type" instead of "object"
    assert_eq!(json["type"], "message");
    assert_eq!(json["role"], "assistant");
    assert_eq!(json["content"][0]["type"], "text");
    assert_eq!(json["content"][0]["text"], "Hello!");
    assert_eq!(json["stop_reason"], "end_turn");
    assert_eq!(json["usage"]["input_tokens"], 10);
    assert_eq!(json["usage"]["output_tokens"], 5);
}

#[test]
fn anthropic_count_tokens_response() {
    let resp = CountTokensResponse { input_tokens: 42 };
    let json: serde_json::Value = serde_json::to_value(&resp).unwrap();
    assert_eq!(json["input_tokens"], 42);
}

// ---------------------------------------------------------------------------
// Anthropic streaming event types
// ---------------------------------------------------------------------------

#[test]
fn anthropic_message_start_event() {
    let event = MessageStartEvent {
        event_type: "message_start",
        message: MessageStartPayload {
            id: "msg_001".to_owned(),
            message_type: "message",
            role: "assistant",
            content: vec![],
            model: "claude-3".to_owned(),
            stop_reason: None,
            usage: make_anthropic_usage(10, 0),
        },
    };
    let json = assert_event_type(&event, "message_start");
    assert_eq!(json["message"]["type"], "message");
    assert_eq!(json["message"]["role"], "assistant");
    assert!(json["message"]["content"].as_array().unwrap().is_empty());
    assert!(json["message"]["stop_reason"].is_null());
}

#[test]
fn anthropic_content_block_start_event() {
    let event = ContentBlockStartEvent {
        event_type: "content_block_start",
        index: 0,
        content_block: ContentBlockStartPayload {
            block_type: "text",
            text: String::new(),
        },
    };
    let json = assert_event_type(&event, "content_block_start");
    assert_eq!(json["index"], 0);
    assert_eq!(json["content_block"]["type"], "text");
    assert_eq!(json["content_block"]["text"], "");
}

#[test]
fn anthropic_content_block_delta_event() {
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
    assert_eq!(json["delta"]["text"], "Hello");
}

#[test]
fn anthropic_content_block_stop_event() {
    let event = ContentBlockStopEvent {
        event_type: "content_block_stop",
        index: 0,
    };
    let json = assert_event_type(&event, "content_block_stop");
    assert_eq!(json["index"], 0);
}

#[test]
fn anthropic_message_delta_event() {
    let event = MessageDeltaEvent {
        event_type: "message_delta",
        delta: MessageDelta {
            stop_reason: Some("end_turn".to_owned()),
        },
        usage: make_anthropic_usage(10, 25),
    };
    let json = assert_event_type(&event, "message_delta");
    assert_eq!(json["delta"]["stop_reason"], "end_turn");
    assert_eq!(json["usage"]["output_tokens"], 25);
}

#[test]
fn anthropic_message_stop_event() {
    let event = MessageStopEvent {
        event_type: "message_stop",
    };
    assert_event_type(&event, "message_stop");
}

// ---------------------------------------------------------------------------
// SSE data format
// ---------------------------------------------------------------------------

#[test]
fn chat_chunk_serializes_as_valid_sse_data() {
    let chunk = make_chat_chunk(
        "chatcmpl-test",
        ChatCompletionDelta {
            role: None,
            content: Some("word".to_owned()),
            reasoning_content: None,
            tool_calls: None,
        },
        None,
    );

    let json_str = serde_json::to_string(&chunk).unwrap();
    // Verify it's valid JSON (can be parsed back)
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(parsed["choices"][0]["delta"]["content"], "word");

    // Verify no newlines in the JSON (SSE data lines must not contain bare newlines)
    assert!(
        !json_str.contains('\n'),
        "SSE data must not contain raw newlines"
    );
}

#[test]
fn completion_chunk_serializes_as_valid_sse_data() {
    let chunk = CompletionChunk {
        id: "cmpl-test".to_owned(),
        object: "text_completion",
        created: 1_700_000_000,
        model: "test".to_owned(),
        choices: vec![CompletionChunkChoice {
            index: 0,
            text: "token".to_owned(),
            finish_reason: None,
        }],
    };

    let json_str = serde_json::to_string(&chunk).unwrap();
    assert!(!json_str.contains('\n'));

    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(parsed["choices"][0]["text"], "token");
}

// ---------------------------------------------------------------------------
// Response with tool_calls in choices
// ---------------------------------------------------------------------------

#[test]
fn chat_completion_response_with_tool_calls_in_choices() {
    let resp = ChatCompletionResponse {
        id: "chatcmpl-tools".to_owned(),
        object: "chat.completion",
        created: 1_700_000_000,
        model: "test".to_owned(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_owned(),
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![
                    make_tool_call("call_1", "get_weather", r#"{"city":"London"}"#),
                    make_tool_call("call_2", "get_time", r#"{"timezone":"UTC"}"#),
                ]),
                tool_call_id: None,
            },
            finish_reason: "tool_calls".to_owned(),
            logprobs: None,
        }],
        usage: make_usage(10, 20),
    };
    let json: serde_json::Value = serde_json::to_value(&resp).unwrap();

    assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");
    let tool_calls = json["choices"][0]["message"]["tool_calls"]
        .as_array()
        .unwrap();
    assert_eq!(tool_calls.len(), 2);
    assert_eq!(tool_calls[0]["id"], "call_1");
    assert_eq!(tool_calls[1]["function"]["name"], "get_time");
    // content should be omitted when None
    assert!(json["choices"][0]["message"].get("content").is_none());
}

// ---------------------------------------------------------------------------
// Completion response with multiple choices
// ---------------------------------------------------------------------------

#[test]
fn completion_response_with_multiple_choices() {
    let resp = CompletionResponse {
        id: "cmpl-multi".to_owned(),
        object: "text_completion",
        created: 1_700_000_000,
        model: "test".to_owned(),
        choices: vec![
            CompletionChoice {
                index: 0,
                text: "first completion".to_owned(),
                finish_reason: "stop".to_owned(),
                logprobs: None,
            },
            CompletionChoice {
                index: 1,
                text: "second completion".to_owned(),
                finish_reason: "length".to_owned(),
                logprobs: None,
            },
        ],
        usage: make_usage(5, 10),
    };
    let json: serde_json::Value = serde_json::to_value(&resp).unwrap();

    let choices = json["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0]["index"], 0);
    assert_eq!(choices[0]["text"], "first completion");
    assert_eq!(choices[1]["index"], 1);
    assert_eq!(choices[1]["finish_reason"], "length");
}

// ---------------------------------------------------------------------------
// Streaming chunk with finish_reason set
// ---------------------------------------------------------------------------

#[test]
fn streaming_chunk_with_finish_reason_set() {
    let chunk = make_empty_delta_chunk("chatcmpl-final", Some("stop".to_owned()));
    let json: serde_json::Value = serde_json::to_value(&chunk).unwrap();

    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    let delta = &json["choices"][0]["delta"];
    assert!(delta.get("role").is_none());
    assert!(delta.get("content").is_none());
}
