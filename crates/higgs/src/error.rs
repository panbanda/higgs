use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

/// JSON error response body (OpenAI-compatible).
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Individual error detail within an [`ErrorResponse`].
#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

/// Server error types.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("Engine error: {0}")]
    Engine(#[from] higgs_engine::error::EngineError),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match &self {
            Self::Engine(e) => {
                tracing::error!(error = %e, "Engine error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    "Internal server error".to_owned(),
                )
            }
            Self::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                msg.clone(),
            ),
            Self::ModelNotFound(model) => (
                StatusCode::NOT_FOUND,
                "model_not_found",
                format!("Model '{model}' is not loaded"),
            ),
            Self::InternalError(msg) => {
                tracing::error!(error = %msg, "Internal error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    "Internal server error".to_owned(),
                )
            }
        };

        let body = Json(ErrorResponse {
            error: ErrorDetail {
                message,
                r#type: error_type.to_owned(),
                code: None,
            },
        });

        (status, body).into_response()
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;

    async fn response_status_and_body(resp: Response) -> (StatusCode, serde_json::Value) {
        let status = resp.status();
        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        (status, body)
    }

    /// Asserts that the given error produces a 500 with a masked message
    /// that does not contain `leaked_detail`.
    async fn assert_masked_500(error: ServerError, leaked_detail: &str) {
        let resp = error.into_response();
        let (status, body) = response_status_and_body(resp).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        let message = body["error"]["message"].as_str().unwrap();
        assert_eq!(message, "Internal server error");
        assert!(
            !message.contains(leaked_detail),
            "Internal error detail leaked: {leaked_detail}"
        );
        assert_eq!(body["error"]["type"].as_str().unwrap(), "server_error");
    }

    #[tokio::test]
    async fn test_engine_error_returns_500_with_masked_message() {
        let engine_err =
            higgs_engine::error::EngineError::Generation("sensitive internal details".to_owned());
        assert_masked_500(
            ServerError::Engine(engine_err),
            "sensitive internal details",
        )
        .await;
    }

    async fn assert_bad_request(msg: &str) {
        let error = ServerError::BadRequest(msg.to_owned());
        let resp = error.into_response();
        let (status, body) = response_status_and_body(resp).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["message"].as_str().unwrap(), msg);
        assert_eq!(
            body["error"]["type"].as_str().unwrap(),
            "invalid_request_error"
        );
    }

    #[tokio::test]
    async fn test_bad_request_returns_400_with_actual_message() {
        assert_bad_request("missing field: model").await;
    }

    #[tokio::test]
    async fn test_internal_error_returns_500_with_masked_message() {
        assert_masked_500(
            ServerError::InternalError("disk full".to_owned()),
            "disk full",
        )
        .await;
    }

    #[tokio::test]
    async fn test_error_code_field_is_null() {
        let error = ServerError::BadRequest("test".to_owned());
        let resp = error.into_response();
        let (_, body) = response_status_and_body(resp).await;

        assert!(body["error"]["code"].is_null());
    }

    #[tokio::test]
    async fn test_engine_tokenization_error_masked() {
        let engine_err = higgs_engine::error::EngineError::Tokenization(
            "tokenizer failed on byte 0xFF".to_owned(),
        );
        assert_masked_500(ServerError::Engine(engine_err), "0xFF").await;
    }

    #[tokio::test]
    async fn test_engine_template_error_masked() {
        let engine_err =
            higgs_engine::error::EngineError::Template("template parse failed".to_owned());
        assert_masked_500(ServerError::Engine(engine_err), "template parse failed").await;
    }

    #[tokio::test]
    async fn test_bad_request_with_empty_message() {
        assert_bad_request("").await;
    }

    #[tokio::test]
    async fn test_bad_request_with_very_long_message() {
        let long_msg = "x".repeat(2000);
        assert_bad_request(&long_msg).await;
    }

    #[tokio::test]
    async fn test_internal_error_with_empty_message_still_masked() {
        let resp = ServerError::InternalError(String::new()).into_response();
        let (status, body) = response_status_and_body(resp).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            body["error"]["message"].as_str().unwrap(),
            "Internal server error"
        );
    }

    #[tokio::test]
    async fn test_error_response_json_structure() {
        let error = ServerError::BadRequest("test".to_owned());
        let resp = error.into_response();
        let (_, body) = response_status_and_body(resp).await;

        assert!(body.get("error").is_some());
        let error_obj = body.get("error").unwrap();
        assert!(error_obj.get("message").is_some());
        assert!(error_obj.get("type").is_some());
        assert!(error_obj.get("code").is_some());
    }

    #[tokio::test]
    async fn test_error_response_content_type_is_json() {
        let error = ServerError::BadRequest("test".to_owned());
        let resp = error.into_response();

        let content_type = resp
            .headers()
            .get("content-type")
            .map(|v| v.to_str().unwrap().to_owned())
            .unwrap_or_default();
        assert!(
            content_type.contains("application/json"),
            "Expected application/json, got: {content_type}"
        );
    }
}
