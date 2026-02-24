//! Tests that `ServerError` variants produce the correct HTTP status codes,
//! response shapes, and content types per the `OpenAI` error contract.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::tests_outside_test_module
)]

use axum::http::StatusCode;
use axum::response::IntoResponse;
use higgs::error::ServerError;
use http_body_util::BodyExt;

async fn extract_response(error: ServerError) -> (StatusCode, serde_json::Value, String) {
    let response = error.into_response();
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap().to_owned())
        .unwrap_or_default();
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    (status, body, content_type)
}

/// Asserts that the given error produces a masked 500 response
/// and the leaked detail does not appear in the client message.
async fn assert_masked_500(error: ServerError, leaked_detail: &str) {
    let (status, body, _) = extract_response(error).await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let message = body["error"]["message"].as_str().unwrap();
    assert_eq!(message, "Internal server error");
    assert!(
        !message.contains(leaked_detail),
        "Internal error detail leaked to client: {leaked_detail}"
    );
    assert_eq!(body["error"]["type"], "server_error");
}

#[tokio::test]
async fn bad_request_returns_400() {
    let error = ServerError::BadRequest("field 'model' is required".to_owned());
    let (status, body, content_type) = extract_response(error).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(content_type.contains("application/json"));
    assert_eq!(body["error"]["message"], "field 'model' is required");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["code"].is_null());
}

#[tokio::test]
async fn internal_error_returns_500_with_masked_message() {
    assert_masked_500(
        ServerError::InternalError("database unreachable".to_owned()),
        "database unreachable",
    )
    .await;
}

#[tokio::test]
async fn engine_error_returns_500_with_masked_message() {
    let engine_err =
        higgs_engine::error::EngineError::Generation("detailed internal stack trace".to_owned());
    assert_masked_500(
        ServerError::Engine(engine_err),
        "detailed internal stack trace",
    )
    .await;
}

#[tokio::test]
async fn engine_tokenization_error_masked() {
    let engine_err = higgs_engine::error::EngineError::Tokenization(
        "invalid byte 0xFF at position 42".to_owned(),
    );
    assert_masked_500(ServerError::Engine(engine_err), "0xFF").await;
}

#[tokio::test]
async fn engine_template_error_masked() {
    let engine_err =
        higgs_engine::error::EngineError::Template("missing variable 'content'".to_owned());
    assert_masked_500(ServerError::Engine(engine_err), "missing variable").await;
}

#[tokio::test]
async fn error_response_has_openai_shape() {
    let error = ServerError::BadRequest("test".to_owned());
    let (_, body, _) = extract_response(error).await;

    assert!(body.get("error").is_some(), "missing top-level 'error' key");
    let error_obj = &body["error"];
    assert!(
        error_obj.get("message").is_some(),
        "missing 'error.message'"
    );
    assert!(error_obj.get("type").is_some(), "missing 'error.type'");
    assert!(error_obj.get("code").is_some(), "missing 'error.code'");
}

#[tokio::test]
async fn error_response_content_type_is_json() {
    let error = ServerError::BadRequest("test".to_owned());
    let (_, _, content_type) = extract_response(error).await;

    assert!(
        content_type.contains("application/json"),
        "Expected application/json, got: {content_type}"
    );
}

#[tokio::test]
async fn error_response_with_very_long_message() {
    let long_msg = "a".repeat(5000);
    let error = ServerError::BadRequest(long_msg.clone());
    let (status, body, _) = extract_response(error).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let message = body["error"]["message"].as_str().unwrap();
    assert_eq!(message.len(), 5000);
    assert_eq!(message, long_msg);
}

#[tokio::test]
async fn error_response_with_special_characters() {
    let special_msg = r#"Invalid "field": <script>alert('xss')</script> & more \n\t"#;
    let error = ServerError::BadRequest(special_msg.to_owned());
    let (status, body, _) = extract_response(error).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let message = body["error"]["message"].as_str().unwrap();
    assert_eq!(message, special_msg);
}

#[tokio::test]
async fn internal_error_with_special_characters_still_masked() {
    let special_msg = r#"DB error: column "user's data" has <invalid> type & NULL"#;
    assert_masked_500(
        ServerError::InternalError(special_msg.to_owned()),
        "DB error",
    )
    .await;
}
