//! A prompt that fills the model's context is refused before generation on every API, and a
//! large `max_tokens` is capped at the end of the context.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use http::StatusCode;
use serde_json::{Value, json};

use crate::test_support::{
    StubEngine, app_with_stub, generation_output, post_json, progress_event, token_event,
};

const PROMPT_TOKENS: u32 = 5;
const PROMPT_LENGTH: usize = 5;

fn stub(context_length: Option<u32>) -> StubEngine {
    let mut stub = StubEngine::named("stub");
    stub.script.prompt_tokens = vec![1; PROMPT_LENGTH];
    stub.script.context_length = context_length;
    stub.script.generation = Some(generation_output("hello", PROMPT_TOKENS, 1));
    stub.script.stream = vec![
        progress_event(0, PROMPT_TOKENS),
        token_event("hello", 1, Some("stop")),
    ];
    stub
}

async fn post(stub: StubEngine, path: &str, request: &Value) -> (StatusCode, Value, StubEngine) {
    let (app, engine) = app_with_stub(stub);
    let (status, text) = post_json(&app, path, request).await;
    let body = serde_json::from_str(&text).unwrap_or(Value::String(text));
    (status, body, engine)
}

fn chat(stream: bool) -> Value {
    json!({"model": "stub", "stream": stream, "messages": [{"role": "user", "content": "hello"}]})
}

fn assert_message_names_both_lengths(body: &Value, message: &Value) {
    let text = message.as_str().unwrap();
    assert!(text.contains('4') && text.contains('5'), "{body}");
}

#[tokio::test]
async fn chat_rejects_a_prompt_that_fills_the_context_in_the_openai_shape() {
    for stream in [false, true] {
        let (status, body, engine) =
            post(stub(Some(4)), "/v1/chat/completions", &chat(stream)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "stream={stream} {body}");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "context_length_exceeded");
        assert_message_names_both_lengths(&body, &body["error"]["message"]);
        assert!(engine.recorded_calls().is_empty(), "no generation ran");
    }
}

#[tokio::test]
async fn a_prompt_equal_to_the_context_is_rejected_and_one_token_less_is_not() {
    let (equal, ..) = post(
        stub(Some(PROMPT_TOKENS)),
        "/v1/chat/completions",
        &chat(false),
    )
    .await;
    assert_eq!(equal, StatusCode::BAD_REQUEST);
    let (one_more, ..) = post(
        stub(Some(PROMPT_TOKENS + 1)),
        "/v1/chat/completions",
        &chat(false),
    )
    .await;
    assert_eq!(one_more, StatusCode::OK);
}

#[tokio::test]
async fn completions_reject_a_prompt_that_fills_the_context() {
    let request = json!({"model": "stub", "prompt": "a a a a a", "stream": true});
    let (status, body, _) = post(stub(Some(4)), "/v1/completions", &request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "context_length_exceeded");
}

#[tokio::test]
async fn anthropic_rejects_with_an_invalid_request_error() {
    for stream in [false, true] {
        let request = json!({
            "model": "stub",
            "max_tokens": 16,
            "stream": stream,
            "messages": [{"role": "user", "content": "hi"}],
        });
        let (status, body, _) = post(stub(Some(4)), "/v1/messages", &request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "stream={stream} {body}");
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_message_names_both_lengths(&body, &body["error"]["message"]);
    }
}

#[tokio::test]
async fn responses_reject_with_the_openai_error_shape() {
    for stream in [false, true] {
        let request = json!({"model": "stub", "input": "hi", "stream": stream});
        let (status, body, _) = post(stub(Some(4)), "/v1/responses", &request).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "stream={stream} {body}");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "context_length_exceeded");
    }
}

#[tokio::test]
async fn max_tokens_is_capped_at_the_end_of_the_context() {
    let mut request = chat(false);
    request["max_tokens"] = json!(100_000);
    let (status, body, engine) = post(stub(Some(8)), "/v1/chat/completions", &request).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(engine.recorded_calls()[0].max_tokens, 8 - PROMPT_TOKENS);

    let mut streamed = chat(true);
    streamed["max_tokens"] = json!(100_000);
    let (_, _, streaming_engine) = post(stub(Some(8)), "/v1/chat/completions", &streamed).await;
    assert_eq!(
        streaming_engine.recorded_calls()[0].max_tokens,
        8 - PROMPT_TOKENS
    );
}

#[tokio::test]
async fn a_request_that_fits_keeps_its_max_tokens_and_no_limit_changes_nothing() {
    let mut small = chat(false);
    small["max_tokens"] = json!(2);
    let (_, _, fitting) = post(stub(Some(100)), "/v1/chat/completions", &small).await;
    assert_eq!(fitting.recorded_calls()[0].max_tokens, 2);

    let mut large = chat(false);
    large["max_tokens"] = json!(100_000);
    let (status, _, unlimited) = post(stub(None), "/v1/chat/completions", &large).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(unlimited.recorded_calls()[0].max_tokens, 100_000);
}
