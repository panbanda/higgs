//! HTTP integration tests for the current API contracts.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::tests_outside_test_module,
    clippy::needless_pass_by_value
)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use chrono::Utc;
use higgs::build_router;
use higgs::metrics::{MetricsStore, RequestRecord, RoutingMethod};
use higgs::router::Router;
use higgs::state::AppState;
use http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

fn build_test_state(metrics: Option<Arc<MetricsStore>>) -> Arc<AppState> {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        r#"
        [provider.mock]
        url = "http://127.0.0.1:1"

        [[routes]]
        pattern = ".*"
        provider = "mock"

        [default]
        provider = "mock"
        "#,
    )
    .unwrap();
    let config = higgs::config::load_config_file(&config_path, None).unwrap();
    let router = Router::from_config(&config, HashMap::new()).unwrap();

    Arc::new(AppState {
        router,
        config,
        http_client: reqwest::Client::new(),
        metrics,
    })
}

#[tokio::test]
#[allow(unknown_lints)]
#[allow(clippy::duration_suboptimal_units)]
async fn metrics_endpoint_returns_snapshot_json() {
    let metrics = Arc::new(MetricsStore::new(Duration::from_secs(60)));
    metrics.record(RequestRecord {
        id: 1,
        timestamp: Instant::now(),
        wallclock: Utc::now(),
        model: Some("llama".to_owned()),
        provider: Some("higgs".to_owned()),
        routing_method: RoutingMethod::Higgs,
        status: 200,
        duration: Duration::from_millis(42),
        input_tokens: 12,
        output_tokens: 34,
        error_body: None,
    });

    let app = build_router(build_test_state(Some(metrics)), 300.0, None, 0, 1024, None);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["totals"]["requests"], 1);
    assert_eq!(json["totals"]["input_tokens"], 12);
    assert_eq!(json["totals"]["output_tokens"], 34);
    assert_eq!(json["models"][0]["name"], "llama");
    assert_eq!(json["providers"][0]["name"], "higgs");
}

#[tokio::test]
async fn request_body_limit_is_enforced() {
    let app = build_router(build_test_state(None), 300.0, None, 0, 64, None);
    let body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "x".repeat(512)}]
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status().as_u16(), 413);
}
