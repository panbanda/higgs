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

use higgs::config::{ApiFormat, MetricsLogConfig};
use higgs::metrics::MetricsStore;
use higgs::metrics_log::MetricsLogger;
use higgs::router::Router;
use higgs::state::AppState;

const METRICS_WINDOW_SECS: u64 = 60;

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
    let metrics = Arc::new(MetricsStore::new(Duration::from_secs(METRICS_WINDOW_SECS)));

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
    higgs::build_router(state, 300.0, None, 0, 10 * 1024 * 1024, None)
}

fn build_selective_test_state(mock_url: &str, metrics: Arc<MetricsStore>) -> Arc<AppState> {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    let config_toml = format!(
        r#"
        [provider.mock]
        url = "{mock_url}"
        format = "anthropic"

        [[routes]]
        pattern = "^known-model$"
        provider = "mock"

        [default]
        provider = "higgs"
    "#
    );
    std::fs::write(&config_path, &config_toml).unwrap();
    let config = higgs::config::load_config_file(&config_path, None).unwrap();
    let router = Router::from_config(&config, HashMap::new()).unwrap();

    Arc::new(AppState {
        router,
        config,
        http_client: reqwest::Client::new(),
        metrics: Some(metrics),
    })
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
    let metrics = Arc::new(MetricsStore::new(Duration::from_secs(METRICS_WINDOW_SECS)));
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
    assert_eq!(records[0].provider.as_deref(), Some("mock"));
    assert_eq!(records[0].status, 200);
    assert_eq!(records[0].model.as_deref(), Some("gpt-4"));
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn metrics_record_success_and_all_http_failures_once() {
    let mock_server = MockServer::start().await;
    let anthropic_response = serde_json::json!({
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "model": "known-model",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 8}
    });
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&anthropic_response))
        .expect(1)
        .mount(&mock_server)
        .await;

    let log_dir = tempfile::tempdir().unwrap();
    let log_path = log_dir.path().join("metrics.jsonl");
    let logger = MetricsLogger::new(&MetricsLogConfig {
        enabled: true,
        path: log_path.to_string_lossy().into_owned(),
        max_size_mb: 1,
        max_files: 1,
    })
    .unwrap();
    let metrics = Arc::new(MetricsStore::with_logger(
        Duration::from_secs(METRICS_WINDOW_SECS),
        logger,
    ));
    let state = build_selective_test_state(&mock_server.uri(), Arc::clone(&metrics));
    let app = build_app(state);

    for _ in 0..20 {
        let health_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health_response.status(), 200);
    }
    assert!(
        metrics.snapshot().is_empty(),
        "infrastructure health polls must not create metrics records"
    );

    let good = serde_json::json!({
        "model": "known-model",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let good_response = app
        .clone()
        .oneshot(post_json("/v1/chat/completions", &good))
        .await
        .unwrap();
    assert_eq!(good_response.status(), 200);

    let unknown = serde_json::json!({
        "model": "nonexistent-model",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let unknown_response = app
        .clone()
        .oneshot(post_json("/v1/chat/completions", &unknown))
        .await
        .unwrap();
    assert_eq!(unknown_response.status(), 404);
    let unknown_body = unknown_response
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&unknown_body).unwrap();
    assert_eq!(
        json["error"]["message"],
        "model 'nonexistent-model' not found among loaded local models"
    );

    let malformed_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from("{"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(malformed_response.status(), 400);

    let fallback_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/unknown-route")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(fallback_response.status(), 404);

    let metrics_response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(metrics_response.status(), 200);
    let metrics_body = metrics_response
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    let snapshot: serde_json::Value = serde_json::from_slice(&metrics_body).unwrap();
    assert_eq!(snapshot["totals"]["requests"], 4);
    assert_eq!(snapshot["totals"]["errors"], 3);
    assert_eq!(snapshot["totals"]["input_tokens"], 12);
    assert_eq!(snapshot["totals"]["output_tokens"], 8);
    assert_eq!(snapshot["status_counts"]["200"], 1);
    assert_eq!(snapshot["status_counts"]["400"], 1);
    assert_eq!(snapshot["status_counts"]["404"], 2);
    let models = snapshot["models"].as_array().unwrap();
    let model_names: Vec<&str> = models
        .iter()
        .map(|model| model["name"].as_str().unwrap())
        .collect();
    assert_eq!(model_names, vec!["known-model", "nonexistent-model"]);
    assert!(model_names.iter().all(|name| !name.starts_with('/')));
    let unknown_model = models
        .iter()
        .find(|model| model["name"] == "nonexistent-model")
        .unwrap();
    assert_eq!(unknown_model["errors"], 1);
    assert_eq!(unknown_model["input_tokens"], 0);
    assert_eq!(unknown_model["output_tokens"], 0);
    assert_eq!(unknown_model["p50_ms"], 0);
    assert_eq!(unknown_model["p95_ms"], 0);
    let provider_names: Vec<&str> = snapshot["providers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|provider| provider["name"].as_str().unwrap())
        .collect();
    assert_eq!(provider_names, vec!["mock"]);
    assert!(provider_names.iter().all(|name| !name.starts_with('/')));

    let records = metrics.snapshot();
    assert_eq!(records.len(), 4);
    assert_eq!(MetricsStore::status_counts(&records).get(&200), Some(&1));
    assert_eq!(MetricsStore::status_counts(&records).get(&400), Some(&1));
    assert_eq!(MetricsStore::status_counts(&records).get(&404), Some(&2));
    assert_eq!(
        records
            .iter()
            .map(|record| record.input_tokens)
            .sum::<u64>(),
        12
    );
    assert_eq!(
        records
            .iter()
            .map(|record| record.output_tokens)
            .sum::<u64>(),
        8
    );
    assert!(
        records
            .iter()
            .filter(|record| record.status >= 400)
            .all(|record| { record.input_tokens == 0 && record.output_tokens == 0 })
    );

    let logged = std::fs::read_to_string(log_path).unwrap();
    let entries: Vec<serde_json::Value> = logged
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(entries.len(), 4);
    let statuses: Vec<u64> = entries
        .iter()
        .map(|entry| entry["status"].as_u64().unwrap())
        .collect();
    assert_eq!(statuses.iter().filter(|&&status| status == 200).count(), 1);
    assert_eq!(statuses.iter().filter(|&&status| status == 400).count(), 1);
    assert_eq!(statuses.iter().filter(|&&status| status == 404).count(), 2);
    assert!(entries.iter().all(|entry| {
        entry["model"]
            .as_str()
            .is_none_or(|model| !model.starts_with('/'))
            && entry["provider"]
                .as_str()
                .is_none_or(|provider| !provider.starts_with('/'))
    }));
    assert_eq!(
        entries
            .iter()
            .filter(|entry| entry["model"] == "nonexistent-model")
            .count(),
        1
    );
    assert_eq!(
        entries
            .iter()
            .filter(|entry| entry["model"].is_null())
            .count(),
        2
    );
    assert_eq!(
        entries
            .iter()
            .filter(|entry| entry["provider"].is_null())
            .count(),
        3
    );
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
