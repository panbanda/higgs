//! `/v1/responses` routing through the real axum router: remote proxying,
//! provider-format rejection, and the stored-response endpoints.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::tests_outside_test_module,
    clippy::needless_pass_by_value
)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use higgs::config::ApiFormat;
use higgs::metrics::MetricsStore;
use higgs::router::Router;
use higgs::state::AppState;
use http::Request;
use http_body_util::BodyExt;
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn build_state(mock_url: &str, format: ApiFormat, metrics: Arc<MetricsStore>) -> Arc<AppState> {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    let format_name = match format {
        ApiFormat::OpenAi => "openai",
        ApiFormat::Anthropic => "anthropic",
    };
    std::fs::write(
        &config_path,
        format!(
            r#"
            [provider.mock]
            url = "{mock_url}"
            format = "{format_name}"

            [[routes]]
            pattern = "^alias$"
            provider = "mock"
            model = "upstream-model"

            [[routes]]
            pattern = ".*"
            provider = "mock"

            [default]
            provider = "mock"
            "#
        ),
    )
    .unwrap();
    let config = higgs::config::load_config_file(&config_path, None).unwrap();
    let router = Router::from_config(&config, HashMap::new()).unwrap();
    Arc::new(AppState {
        router,
        config,
        http_client: reqwest::Client::new(),
        metrics: Some(metrics),
        tools: Arc::new(higgs::tools::ToolHub::disabled()),
    })
}

fn build_app(state: Arc<AppState>) -> axum::Router {
    higgs::build_router(state, 300.0, None, 0, 10 * 1024 * 1024, None)
}

fn metrics_store() -> Arc<MetricsStore> {
    Arc::new(MetricsStore::new(Duration::from_secs(60)))
}

fn post_json(uri: &str, body: &serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap()
}

fn empty_request(http_method: &str, uri: &str) -> Request<Body> {
    Request::builder()
        .method(http_method)
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

async fn json_body(response: axum::response::Response) -> serde_json::Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

fn upstream_response() -> serde_json::Value {
    serde_json::json!({
        "id": "resp_upstream",
        "object": "response",
        "status": "completed",
        "output": [],
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    })
}

#[tokio::test]
async fn openai_provider_receives_the_request_unchanged() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
        .expect(1)
        .mount(&mock_server)
        .await;

    let metrics = metrics_store();
    let app = build_app(build_state(
        &mock_server.uri(),
        ApiFormat::OpenAi,
        Arc::clone(&metrics),
    ));
    let request_body = serde_json::json!({
        "model": "gpt-5",
        "input": [{"type": "web_search_call", "id": "ws_1"}],
        "tools": [{"type": "web_search"}],
        "previous_response_id": "resp_remote_prev"
    });
    let response = app
        .oneshot(post_json("/v1/responses", &request_body))
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    assert_eq!(json_body(response).await["id"], "resp_upstream");

    let received = mock_server.received_requests().await.unwrap();
    let upstream_body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(upstream_body, request_body);

    let records = metrics.snapshot();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].provider.as_deref(), Some("mock"));
    assert_eq!(records[0].status, 200);
    assert_eq!(records[0].model.as_deref(), Some("gpt-5"));
}

#[tokio::test]
async fn openai_provider_model_alias_is_rewritten() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
        .mount(&mock_server)
        .await;

    let app = build_app(build_state(
        &mock_server.uri(),
        ApiFormat::OpenAi,
        metrics_store(),
    ));
    let response = app
        .oneshot(post_json(
            "/v1/responses",
            &serde_json::json!({"model": "alias", "input": "hi"}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);

    let received = mock_server.received_requests().await.unwrap();
    let upstream_body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(upstream_body["model"], "upstream-model");
    assert_eq!(upstream_body["input"], "hi");
}

#[tokio::test]
async fn openai_provider_stream_is_passed_through_verbatim() {
    let mock_server = MockServer::start().await;
    let sse = "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0}\n\n\
               event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":1}\n\n";
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse, "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    let app = build_app(build_state(
        &mock_server.uri(),
        ApiFormat::OpenAi,
        metrics_store(),
    ));
    let response = app
        .oneshot(post_json(
            "/v1/responses",
            &serde_json::json!({"model": "gpt-5", "input": "hi", "stream": true}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(std::str::from_utf8(&bytes).unwrap(), sse);
}

#[tokio::test]
async fn upstream_errors_are_forwarded_with_their_status() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
            "error": {"message": "slow down", "type": "rate_limit_error"}
        })))
        .mount(&mock_server)
        .await;

    let metrics = metrics_store();
    let app = build_app(build_state(
        &mock_server.uri(),
        ApiFormat::OpenAi,
        Arc::clone(&metrics),
    ));
    let response = app
        .oneshot(post_json(
            "/v1/responses",
            &serde_json::json!({"model": "gpt-5", "input": "hi"}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), 429);
    assert_eq!(metrics.snapshot()[0].status, 429);
}

#[tokio::test]
async fn anthropic_provider_is_rejected_with_an_explanation() {
    let mock_server = MockServer::start().await;
    let metrics = metrics_store();
    let app = build_app(build_state(
        &mock_server.uri(),
        ApiFormat::Anthropic,
        Arc::clone(&metrics),
    ));
    let response = app
        .oneshot(post_json(
            "/v1/responses",
            &serde_json::json!({"model": "claude-sonnet", "input": "hi"}),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
    let body = json_body(response).await;
    assert_eq!(body["error"]["type"], "invalid_request_error");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("only available for local models and OpenAI-format"));
    assert!(message.contains("claude-sonnet"));
    assert!(mock_server.received_requests().await.unwrap().is_empty());

    let records = metrics.snapshot();
    assert_eq!(records.len(), 1, "rejected requests still show in metrics");
    assert_eq!(records[0].status, 400);
    assert_eq!(records[0].model.as_deref(), Some("claude-sonnet"));
}

#[tokio::test]
async fn unknown_response_ids_return_the_openai_404_shape() {
    let mock_server = MockServer::start().await;
    let app = build_app(build_state(
        &mock_server.uri(),
        ApiFormat::OpenAi,
        metrics_store(),
    ));

    for http_method in ["GET", "DELETE"] {
        let response = app
            .clone()
            .oneshot(empty_request(http_method, "/v1/responses/resp_missing"))
            .await
            .unwrap();
        assert_eq!(response.status(), 404, "{http_method}");
        let body = json_body(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "response_not_found");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("resp_missing")
        );
    }
}
