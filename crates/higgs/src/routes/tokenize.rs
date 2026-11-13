//! `POST /v1/tokenize` and `POST /v1/detokenize` (also served without the
//! `/v1` prefix). Both operate on the tokenizer of a locally loaded model.

use axum::{
    Json,
    extract::{Extension, State},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::{
    backend::EngineLease,
    error::ServerError,
    generation::messages::convert_messages,
    metrics::RequestMetricsContext,
    router::ResolvedRoute,
    state::SharedState,
    types::openai::{ChatCompletionMessage, ReasoningConfig},
};

/// Request body for `/v1/tokenize`: either raw `text` or chat `messages`.
#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    pub model: String,
    #[serde(default)]
    pub text: Option<String>,
    /// Rendered through the model's chat template, like `/v1/chat/completions`.
    #[serde(default)]
    pub messages: Option<Vec<ChatCompletionMessage>>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    /// Applies to `text` only; matches `/v1/completions` when left unset.
    #[serde(default)]
    pub add_special_tokens: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
    pub count: usize,
}

#[derive(Debug, Deserialize)]
pub struct DetokenizeRequest {
    pub model: String,
    pub tokens: Vec<u32>,
    #[serde(default)]
    pub skip_special_tokens: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct DetokenizeResponse {
    pub text: String,
}

pub async fn tokenize(
    State(state): State<SharedState>,
    Extension(request_metrics): Extension<RequestMetricsContext>,
    body: Bytes,
) -> Result<Response, ServerError> {
    let req: TokenizeRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid request body: {e}")))?;
    request_metrics.set_requested_model(&req.model);

    let input = match (&req.text, &req.messages) {
        (Some(_), Some(_)) => {
            return Err(ServerError::BadRequest(
                "provide either text or messages, not both".to_owned(),
            ));
        }
        (None, None) => {
            return Err(ServerError::BadRequest(
                "either text or messages is required".to_owned(),
            ));
        }
        (Some(_), None) => None,
        (None, Some(messages)) if messages.is_empty() => {
            return Err(ServerError::BadRequest(
                "messages array must not be empty".to_owned(),
            ));
        }
        (None, Some(messages)) => serde_json::to_value(messages)
            .ok()
            .and_then(|value| value.as_array().cloned()),
    };

    let (engine, model_name) = local_engine(&state, &req.model, input.as_deref()).await?;

    let tokens = match (&req.text, &req.messages) {
        (Some(text), _) => engine
            .tokenizer()
            .encode(text.as_str(), req.add_special_tokens.unwrap_or(false))
            .map_err(|e| ServerError::BadRequest(format!("Tokenization error: {e}")))?
            .get_ids()
            .to_vec(),
        (None, Some(messages)) => {
            let engine_messages = convert_messages(messages);
            let tools = req.tools.as_deref().filter(|tools| !tools.is_empty());
            let thinking_enabled = crate::reasoning::effective_thinking_enabled(
                engine.enable_thinking(),
                &[engine.model_name(), model_name.as_str()],
                req.reasoning.as_ref(),
            );
            engine
                .prepare_chat_prompt(&engine_messages, tools, thinking_enabled)
                .map_err(ServerError::Engine)?
        }
        (None, None) => Vec::new(),
    };

    Ok(Json(TokenizeResponse {
        count: tokens.len(),
        tokens,
    })
    .into_response())
}

pub async fn detokenize(
    State(state): State<SharedState>,
    Extension(request_metrics): Extension<RequestMetricsContext>,
    body: Bytes,
) -> Result<Response, ServerError> {
    let req: DetokenizeRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid request body: {e}")))?;
    request_metrics.set_requested_model(&req.model);

    let (engine, _model_name) = local_engine(&state, &req.model, None).await?;
    let tokenizer = engine.tokenizer();

    if let Some(unknown) = req
        .tokens
        .iter()
        .find(|token| tokenizer.id_to_token(**token).is_none())
    {
        return Err(ServerError::BadRequest(format!(
            "token id {unknown} is not in this model's vocabulary"
        )));
    }

    let text = tokenizer
        .decode(&req.tokens, req.skip_special_tokens.unwrap_or(false))
        .map_err(|e| ServerError::BadRequest(format!("Detokenization error: {e}")))?;

    Ok(Json(DetokenizeResponse { text }).into_response())
}

async fn local_engine(
    state: &SharedState,
    model: &str,
    messages: Option<&[serde_json::Value]>,
) -> Result<(EngineLease, String), ServerError> {
    let resolved = state.router.resolve(model, messages).await?;
    match resolved {
        ResolvedRoute::Higgs {
            engine, model_name, ..
        } => Ok((engine, model_name)),
        ResolvedRoute::Remote { provider_name, .. } => Err(ServerError::BadRequest(format!(
            "tokenization is only available for local models; '{model}' is routed to remote provider '{provider_name}'"
        ))),
    }
}

#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use crate::test_support::{StubEngine, app_with_stub, post_json};

    fn stub() -> StubEngine {
        let mut stub = StubEngine::named("stub");
        stub.script.prompt_tokens = vec![9, 8, 7, 6];
        stub
    }

    async fn call(path: &str, body: serde_json::Value) -> (http::StatusCode, serde_json::Value) {
        let (app, _engine) = app_with_stub(stub());
        let (status, text) = post_json(&app, path, &body).await;
        (
            status,
            serde_json::from_str(&text).unwrap_or(serde_json::Value::Null),
        )
    }

    #[tokio::test]
    async fn tokenizes_text() {
        let (status, json) = call(
            "/v1/tokenize",
            serde_json::json!({"model": "stub", "text": "hello world"}),
        )
        .await;
        assert_eq!(status, http::StatusCode::OK);
        assert_eq!(json, serde_json::json!({"tokens": [1, 2], "count": 2}));
    }

    #[tokio::test]
    async fn tokenizes_text_without_v1_prefix() {
        let (status, json) = call(
            "/tokenize",
            serde_json::json!({"model": "stub", "text": "world hello"}),
        )
        .await;
        assert_eq!(status, http::StatusCode::OK);
        assert_eq!(json["tokens"], serde_json::json!([2, 1]));
    }

    #[tokio::test]
    async fn empty_text_yields_no_tokens() {
        let (status, json) = call(
            "/v1/tokenize",
            serde_json::json!({"model": "stub", "text": ""}),
        )
        .await;
        assert_eq!(status, http::StatusCode::OK);
        assert_eq!(json, serde_json::json!({"tokens": [], "count": 0}));
    }

    #[tokio::test]
    async fn tokenizes_chat_messages_through_the_template() {
        let (status, json) = call(
            "/v1/tokenize",
            serde_json::json!({
                "model": "stub",
                "messages": [{"role": "user", "content": "hello"}],
            }),
        )
        .await;
        assert_eq!(status, http::StatusCode::OK);
        assert_eq!(
            json,
            serde_json::json!({"tokens": [9, 8, 7, 6], "count": 4})
        );
    }

    #[tokio::test]
    async fn rejects_text_and_messages_together() {
        let (status, json) = call(
            "/v1/tokenize",
            serde_json::json!({
                "model": "stub",
                "text": "hello",
                "messages": [{"role": "user", "content": "hello"}],
            }),
        )
        .await;
        assert_eq!(status, http::StatusCode::BAD_REQUEST);
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not both")
        );
    }

    #[tokio::test]
    async fn rejects_missing_input() {
        let (status, json) = call("/v1/tokenize", serde_json::json!({"model": "stub"})).await;
        assert_eq!(status, http::StatusCode::BAD_REQUEST);
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("required")
        );
    }

    #[tokio::test]
    async fn rejects_empty_messages() {
        let (status, _json) = call(
            "/v1/tokenize",
            serde_json::json!({"model": "stub", "messages": []}),
        )
        .await;
        assert_eq!(status, http::StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn unknown_model_is_not_found() {
        let (status, _json) = call(
            "/v1/tokenize",
            serde_json::json!({"model": "nonexistent", "text": "hi"}),
        )
        .await;
        assert_eq!(status, http::StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn remote_models_get_a_clear_client_error() {
        let (app, _engine) = crate::test_support::app_with_stub_and_config(
            stub(),
            "\n[provider.remote]\nurl = \"http://127.0.0.1:1\"\n\n[[routes]]\npattern = \"remote-.*\"\nprovider = \"remote\"\n",
        );
        for (path, body) in [
            (
                "/v1/tokenize",
                serde_json::json!({"model": "remote-gpt", "text": "hi"}),
            ),
            (
                "/v1/detokenize",
                serde_json::json!({"model": "remote-gpt", "tokens": [1]}),
            ),
        ] {
            let (status, text) = post_json(&app, path, &body).await;
            assert_eq!(status, http::StatusCode::BAD_REQUEST, "{path}: {text}");
            assert!(text.contains("only available for local models"), "{text}");
            assert!(text.contains("remote"), "{text}");
        }
    }

    #[tokio::test]
    async fn detokenizes_tokens() {
        let (status, json) = call(
            "/v1/detokenize",
            serde_json::json!({"model": "stub", "tokens": [1, 2]}),
        )
        .await;
        assert_eq!(status, http::StatusCode::OK);
        assert_eq!(json, serde_json::json!({"text": "hello world"}));
    }

    #[tokio::test]
    async fn detokenizes_without_v1_prefix() {
        let (status, json) = call(
            "/detokenize",
            serde_json::json!({"model": "stub", "tokens": [2]}),
        )
        .await;
        assert_eq!(status, http::StatusCode::OK);
        assert_eq!(json["text"], "world");
    }

    #[tokio::test]
    async fn detokenize_empty_tokens_yields_empty_text() {
        let (status, json) = call(
            "/v1/detokenize",
            serde_json::json!({"model": "stub", "tokens": []}),
        )
        .await;
        assert_eq!(status, http::StatusCode::OK);
        assert_eq!(json, serde_json::json!({"text": ""}));
    }

    #[tokio::test]
    async fn detokenize_rejects_unknown_token_id() {
        let (status, json) = call(
            "/v1/detokenize",
            serde_json::json!({"model": "stub", "tokens": [1, 4242]}),
        )
        .await;
        assert_eq!(status, http::StatusCode::BAD_REQUEST);
        assert!(json["error"]["message"].as_str().unwrap().contains("4242"));
    }

    #[tokio::test]
    async fn detokenize_rejects_non_integer_tokens() {
        let (status, _json) = call(
            "/v1/detokenize",
            serde_json::json!({"model": "stub", "tokens": ["hello"]}),
        )
        .await;
        assert_eq!(status, http::StatusCode::BAD_REQUEST);
    }
}
