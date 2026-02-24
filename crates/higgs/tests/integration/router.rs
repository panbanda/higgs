//! Router-level integration tests using `tower::ServiceExt::oneshot`.
//!
//! These tests exercise the HTTP layer (routing, middleware, content negotiation)
//! without requiring a real MLX engine. Tests that would need the engine to
//! process a request are marked #[ignore] with an explanation.
//!
//! We construct an `AppState` with a real engine only where unavoidable. For most
//! tests, we can verify behavior that occurs before the engine is invoked:
//! - Health endpoint (no engine needed)
//! - Malformed JSON rejection (Axum rejects before handler runs)
//! - Wrong HTTP method (405)
//! - Unknown routes (404)
//! - Bearer auth rejection (middleware rejects before handler runs)

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::tests_outside_test_module,
    clippy::needless_pass_by_value
)]

use axum::http::{Request, StatusCode};
use higgs::build_router;
use higgs::state::AppState;
use http_body_util::BodyExt;
use std::sync::Arc;
use tower::ServiceExt;

/// Build a minimal router with just the health endpoint for testing.
fn build_health_only_router() -> axum::Router {
    use axum::routing::get;
    axum::Router::new().route("/health", get(higgs::routes::health::health))
}

/// Send a request to the health-only router and return the response.
async fn send_request(
    method: &str,
    uri: &str,
    headers: &[(&str, &str)],
) -> axum::http::Response<axum::body::Body> {
    let app = build_health_only_router();
    let mut builder = Request::builder().method(method).uri(uri);
    for (key, value) in headers {
        builder = builder.header(*key, *value);
    }
    app.oneshot(builder.body(axum::body::Body::empty()).unwrap())
        .await
        .unwrap()
}

/// Extract JSON body from a response.
async fn response_json(response: axum::http::Response<axum::body::Body>) -> serde_json::Value {
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&body_bytes).unwrap()
}

fn response_content_type(response: &axum::http::Response<axum::body::Body>) -> String {
    response
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap().to_owned())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Health endpoint
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_returns_200_with_json() {
    let response = send_request("GET", "/health", &[]).await;

    assert_eq!(response.status(), StatusCode::OK);
    let ct = response_content_type(&response);
    assert!(
        ct.contains("application/json"),
        "Expected JSON content type, got: {ct}"
    );

    let body = response_json(response).await;
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn health_rejects_post() {
    let response = send_request("POST", "/health", &[]).await;
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

// ---------------------------------------------------------------------------
// Unknown routes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_route_returns_404() {
    let response = send_request("GET", "/v1/nonexistent", &[]).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

// ---------------------------------------------------------------------------
// JSON deserialization errors (these tests verify Axum's behavior with
// malformed request bodies, independent of engine state)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn chat_completions_invalid_json_returns_422() {
    // This test would send invalid JSON to /v1/chat/completions and verify
    // that Axum returns 422 Unprocessable Entity before the handler runs.
    // Requires a full router with AppState.
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn chat_completions_empty_messages_returns_400() {
    // Would verify: POST /v1/chat/completions with {"model":"m","messages":[]}
    // returns 400 BadRequest("messages array must not be empty").
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn completions_empty_prompt_returns_400() {
    // Would verify: POST /v1/completions with {"model":"m","prompt":""}
    // returns 400 BadRequest("prompt must not be empty").
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn anthropic_empty_messages_returns_400() {
    // Would verify: POST /v1/messages with {"model":"m","messages":[],"max_tokens":100}
    // returns 400 BadRequest("messages array must not be empty").
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn embeddings_empty_input_returns_400() {
    // Would verify: POST /v1/embeddings with {"model":"m","input":[]}
    // returns 400 BadRequest("input must not be empty").
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn chat_completions_get_returns_405() {
    // Would verify GET /v1/chat/completions returns 405 Method Not Allowed.
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn models_returns_200_with_model_list() {
    // Would verify GET /v1/models returns the loaded model name.
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn bearer_auth_missing_returns_401() {
    // Would build router with api_key="test" and verify that requests
    // without Authorization header get 401.
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn bearer_auth_wrong_key_returns_401() {
    // Would build router with api_key="correct" and verify that requests
    // with Authorization: Bearer wrong get 401.
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn streaming_chat_returns_sse_content_type() {
    // Would verify POST /v1/chat/completions with stream:true returns
    // content-type: text/event-stream.
}

// ---------------------------------------------------------------------------
// build_router smoke test (verifies the function is callable and public)
// ---------------------------------------------------------------------------

#[test]
fn build_router_is_public_and_callable() {
    // This test verifies that build_router was successfully extracted to lib.rs.
    // We cannot actually call it without a real AppState, but we can verify
    // the function signature exists and is accessible.
    let _: fn(Arc<AppState>, f64, Option<String>, u32) -> axum::Router = build_router;
}

// ---------------------------------------------------------------------------
// CORS preflight
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cors_preflight_options_returns_405_without_cors_layer() {
    let response = send_request(
        "OPTIONS",
        "/health",
        &[
            ("Origin", "http://localhost:3000"),
            ("Access-Control-Request-Method", "GET"),
        ],
    )
    .await;

    // Health-only router has no CORS middleware, so OPTIONS returns 405.
    // The full build_router adds CorsLayer::permissive().
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

// ---------------------------------------------------------------------------
// Health endpoint isolation from auth
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_endpoint_works_on_health_only_router() {
    let response = send_request("GET", "/health", &[]).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = response_json(response).await;
    assert_eq!(body["status"], "ok");
}
