//! End-to-end proxy integration tests using wiremock as an upstream provider.
//!
//! These tests build a full `AppState` with no local engines, one remote
//! provider pointing at a wiremock mock server, and a catch-all route.
//! Requests go through the real axum router via `tower::ServiceExt::oneshot`.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::tests_outside_test_module,
    clippy::needless_pass_by_value,
    clippy::unreadable_literal,
    clippy::needless_borrows_for_generic_args
)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use http::Request;
use http_body_util::BodyExt;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use higgs::config::ApiFormat;
use higgs::metrics::MetricsStore;
use higgs::router::Router;
use higgs::state::AppState;

fn build_test_state(mock_url: &str, format: ApiFormat) -> Arc<AppState> {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    let config_toml = format!(
        r#"
        [provider.mock]
        url = "{mock_url}"
        format = "{fmt}"

        [[routes]]
        pattern = ".*"
        provider = "mock"

        [default]
        provider = "mock"
    "#,
        fmt = match format {
            ApiFormat::OpenAi => "openai",
            ApiFormat::Anthropic => "anthropic",
        }
    );
    std::fs::write(&config_path, &config_toml).unwrap();
    let config = higgs::config::load_config_file(&config_path, None).unwrap();

    let router = Router::from_config(&config, HashMap::new()).unwrap();
    let metrics = Arc::new(MetricsStore::new(Duration::from_secs(60)));

    Arc::new(AppState {
        router,
        config,
        http_client: reqwest::Client::new(),
        metrics: Some(metrics),
    })
}

fn openai_chat_response() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    })
}

fn openai_chat_request_body() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hi"}]
    })
}

fn build_app(state: Arc<AppState>) -> axum::Router {
    higgs::build_router(state, 300.0, None, 0)
}

fn post_json(uri: &str, body: &serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap()
}

// ---------------------------------------------------------------------------
// 1. OpenAI passthrough
// ---------------------------------------------------------------------------

#[tokio::test]
async fn proxy_openai_passthrough() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&openai_chat_response()))
        .mount(&mock_server)
        .await;

    let state = build_test_state(&mock_server.uri(), ApiFormat::OpenAi);
    let app = build_app(state);

    let response = app
        .oneshot(post_json(
            "/v1/chat/completions",
            &openai_chat_request_body(),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["id"], "chatcmpl-test");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello!");
}

// ---------------------------------------------------------------------------
// 2. Upstream error status preserved
// ---------------------------------------------------------------------------

#[tokio::test]
async fn proxy_upstream_error_preserved() {
    let mock_server = MockServer::start().await;

    let error_body = serde_json::json!({
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error"
        }
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_json(&error_body))
        .mount(&mock_server)
        .await;

    let state = build_test_state(&mock_server.uri(), ApiFormat::OpenAi);
    let app = build_app(state);

    let response = app
        .oneshot(post_json(
            "/v1/chat/completions",
            &openai_chat_request_body(),
        ))
        .await
        .unwrap();

    // proxy_request passes through the upstream status code directly
    assert_eq!(response.status(), 429);
}

// ---------------------------------------------------------------------------
// 3. Model rewrite
// ---------------------------------------------------------------------------

#[tokio::test]
async fn proxy_model_rewrite() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&openai_chat_response()))
        .expect(1)
        .mount(&mock_server)
        .await;

    // Build config with a model rewrite rule
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    let config_toml = format!(
        r#"
        [provider.mock]
        url = "{}"
        format = "openai"

        [[routes]]
        pattern = "my-alias"
        provider = "mock"
        model = "actual-upstream-model"

        [default]
        provider = "mock"
    "#,
        mock_server.uri()
    );
    std::fs::write(&config_path, &config_toml).unwrap();
    let config = higgs::config::load_config_file(&config_path, None).unwrap();
    let router = Router::from_config(&config, HashMap::new()).unwrap();
    let metrics = Arc::new(MetricsStore::new(Duration::from_secs(60)));
    let state = Arc::new(AppState {
        router,
        config,
        http_client: reqwest::Client::new(),
        metrics: Some(metrics),
    });
    let app = build_app(state);

    let request_body = serde_json::json!({
        "model": "my-alias",
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = app
        .oneshot(post_json("/v1/chat/completions", &request_body))
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    // Verify the mock received exactly one request and the model field was rewritten
    let received = mock_server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    let upstream_body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(
        upstream_body["model"].as_str().unwrap(),
        "actual-upstream-model",
        "model field should be rewritten before sending to upstream"
    );
}

// ---------------------------------------------------------------------------
// 4. Metrics recorded for proxy requests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn metrics_recorded_for_proxy() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&openai_chat_response()))
        .mount(&mock_server)
        .await;

    let state = build_test_state(&mock_server.uri(), ApiFormat::OpenAi);
    let metrics = Arc::clone(state.metrics.as_ref().unwrap());
    let app = build_app(state);

    let response = app
        .oneshot(post_json(
            "/v1/chat/completions",
            &openai_chat_request_body(),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let records = metrics.snapshot();
    assert_eq!(records.len(), 1, "expected exactly one metrics record");
    assert_eq!(records[0].provider, "mock");
    assert_eq!(records[0].status, 200);
    assert_eq!(records[0].model, "gpt-4");
}

// ---------------------------------------------------------------------------
// 5. Cross-format: OpenAI request -> Anthropic provider -> translated response
// ---------------------------------------------------------------------------

#[tokio::test]
async fn proxy_openai_to_anthropic_translation() {
    let mock_server = MockServer::start().await;

    // Upstream returns an Anthropic-format response
    let anthropic_response = serde_json::json!({
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "Hello from Anthropic!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 8}
    });

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&anthropic_response))
        .expect(1)
        .mount(&mock_server)
        .await;

    // Provider is Anthropic format -- the gateway must translate OpenAI -> Anthropic
    let state = build_test_state(&mock_server.uri(), ApiFormat::Anthropic);
    let app = build_app(state);

    // Send an OpenAI-format request
    let response = app
        .oneshot(post_json(
            "/v1/chat/completions",
            &openai_chat_request_body(),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

    // Response should be translated back to OpenAI format
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Hello from Anthropic!"
    );
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(body["choices"][0]["finish_reason"].is_string());

    // Verify the upstream received an Anthropic-format request
    let received = mock_server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    let upstream_body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    // Anthropic requests have "messages" array and no "model" at top level is rewritten
    assert!(
        upstream_body.get("messages").is_some(),
        "upstream should receive Anthropic-format request with messages"
    );
    assert!(
        upstream_body.get("max_tokens").is_some(),
        "Anthropic requests require max_tokens"
    );
}

// ---------------------------------------------------------------------------
// 6. Cross-format: Anthropic request -> OpenAI provider -> translated response
// ---------------------------------------------------------------------------

#[tokio::test]
async fn proxy_anthropic_to_openai_translation() {
    let mock_server = MockServer::start().await;

    // Upstream returns an OpenAI-format response
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&openai_chat_response()))
        .expect(1)
        .mount(&mock_server)
        .await;

    // Provider is OpenAI format, request comes in as Anthropic
    let state = build_test_state(&mock_server.uri(), ApiFormat::OpenAi);
    let app = build_app(state);

    // Send an Anthropic-format request to the Anthropic endpoint
    let anthropic_request = serde_json::json!({
        "model": "gpt-4",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hi"}]
    });

    let response = app
        .oneshot(post_json("/v1/messages", &anthropic_request))
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

    // Response should be translated to Anthropic format
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert!(body["content"].is_array());
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["content"][0]["text"], "Hello!");

    // Verify the upstream received an OpenAI-format request
    let received = mock_server.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    let upstream_body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert!(
        upstream_body.get("messages").is_some(),
        "upstream should receive OpenAI-format request"
    );
}
