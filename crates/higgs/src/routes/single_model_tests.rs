//! A server with exactly one local model, and no routes or providers, serves requests for
//! any model name with that model, on every API.

#![allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]

use serde_json::{Value, json};

use crate::test_support::{
    StubEngine, app_with_stub, app_with_stub_and_config, generation_output, post_json,
};

fn scripted_stub() -> StubEngine {
    let mut stub = StubEngine::named("stub");
    stub.script.generation = Some(generation_output("hello", 3, 2));
    stub
}

async fn post_json_value(
    app: &axum::Router,
    path: &str,
    body: &Value,
) -> (http::StatusCode, Value) {
    let (status, text) = post_json(app, path, body).await;
    (status, serde_json::from_str(&text).unwrap_or(Value::Null))
}

fn requests_for(model: &str) -> Vec<(&'static str, Value)> {
    vec![
        (
            "/v1/chat/completions",
            json!({"model": model, "messages": [{"role": "user", "content": "hi"}]}),
        ),
        (
            "/v1/completions",
            json!({"model": model, "prompt": "hi", "max_tokens": 4}),
        ),
        ("/v1/responses", json!({"model": model, "input": "hi"})),
        (
            "/v1/messages",
            json!({
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}]
            }),
        ),
    ]
}

#[tokio::test]
async fn unknown_names_are_served_by_the_only_model_and_report_it() {
    let (app, _engine) = app_with_stub(scripted_stub());
    for model in ["claude-sonnet-4-5", "gpt-4o"] {
        for (path, body) in requests_for(model) {
            let (status, json) = post_json_value(&app, path, &body).await;
            assert_eq!(status, http::StatusCode::OK, "{path} {model}: {json}");
            assert_eq!(json["model"], "stub", "{path} {model}: {json}");
        }
    }
}

#[tokio::test]
async fn tokenize_uses_the_only_model_for_unknown_names() {
    let (app, _engine) = app_with_stub(scripted_stub());
    let (status, json) = post_json_value(
        &app,
        "/v1/tokenize",
        &json!({"model": "claude-sonnet-4-5", "text": "hello world"}),
    )
    .await;
    assert_eq!(status, http::StatusCode::OK, "{json}");
}

#[tokio::test]
async fn exact_name_still_reports_itself() {
    let (app, _engine) = app_with_stub(scripted_stub());
    for (path, body) in requests_for("stub") {
        let (status, json) = post_json_value(&app, path, &body).await;
        assert_eq!(status, http::StatusCode::OK, "{path}: {json}");
        assert_eq!(json["model"], "stub", "{path}: {json}");
    }
}

#[tokio::test]
async fn configured_provider_keeps_unknown_names_not_found() {
    let (app, _engine) = app_with_stub_and_config(
        scripted_stub(),
        "\n[provider.remote]\nurl = \"http://127.0.0.1:1\"\n",
    );
    for (path, body) in requests_for("claude-sonnet-4-5") {
        let (status, json) = post_json_value(&app, path, &body).await;
        assert_eq!(status, http::StatusCode::NOT_FOUND, "{path}: {json}");
        assert_eq!(json["error"]["type"], "model_not_found", "{path}: {json}");
    }
}

#[tokio::test]
async fn configured_route_keeps_unknown_names_not_found() {
    let (app, _engine) = app_with_stub_and_config(
        scripted_stub(),
        "\n[provider.remote]\nurl = \"http://127.0.0.1:1\"\n\n[[routes]]\npattern = \"^remote-\"\nprovider = \"remote\"\n",
    );
    let (path, body) = &requests_for("gpt-4o")[0];
    let (status, json) = post_json_value(&app, path, body).await;
    assert_eq!(status, http::StatusCode::NOT_FOUND, "{json}");
}
