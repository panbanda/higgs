use std::time::Instant;

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use bytes::Bytes;

use crate::{
    config::ApiFormat,
    error::ServerError,
    metrics::RequestRecord,
    router::ResolvedRoute,
    state::SharedState,
    types::openai::{
        EmbeddingInput, EmbeddingObject, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    },
};

#[allow(clippy::too_many_lines)]
pub async fn embeddings(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ServerError> {
    let req: EmbeddingRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid request body: {e}")))?;

    let resolved = state
        .router
        .resolve(&req.model, None)
        .await
        .map_err(ServerError::ModelNotFound)?;

    match resolved {
        ResolvedRoute::Higgs {
            engine,
            model_name,
            routing_method,
        } => {
            let inputs = match &req.input {
                EmbeddingInput::Single(s) => vec![s.clone()],
                EmbeddingInput::Multiple(v) => v.clone(),
            };

            if inputs.is_empty() {
                return Err(ServerError::BadRequest(
                    "input must not be empty".to_owned(),
                ));
            }

            let start = Instant::now();
            let mut data = Vec::new();
            let mut total_tokens: u32 = 0;

            for (idx, text) in inputs.iter().enumerate() {
                let encoding = engine
                    .tokenizer()
                    .encode(text.as_str(), false)
                    .map_err(|e| ServerError::BadRequest(format!("Tokenization error: {e}")))?;

                let token_ids = encoding.get_ids();
                let token_count: u32 = token_ids
                    .len()
                    .try_into()
                    .map_err(|_| ServerError::BadRequest("Input too long".to_owned()))?;
                total_tokens = total_tokens.saturating_add(token_count);

                let embedding = engine
                    .embed(token_ids)
                    .map_err(|e| ServerError::InternalError(format!("Embedding error: {e}")))?;

                let index: u32 = idx
                    .try_into()
                    .map_err(|_| ServerError::BadRequest("Too many inputs".to_owned()))?;

                data.push(EmbeddingObject {
                    object: "embedding",
                    embedding,
                    index,
                });
            }

            if let Some(ref metrics) = state.metrics {
                metrics.record(RequestRecord {
                    id: 0,
                    timestamp: Instant::now(),
                    wallclock: chrono::Utc::now(),
                    model: model_name,
                    provider: "higgs".to_owned(),
                    routing_method: routing_method.into(),
                    status: 200,
                    duration: start.elapsed(),
                    input_tokens: u64::from(total_tokens),
                    output_tokens: 0,
                    error_body: None,
                });
            }

            Ok(Json(EmbeddingResponse {
                object: "list",
                data,
                model: req.model,
                usage: EmbeddingUsage {
                    prompt_tokens: total_tokens,
                    total_tokens,
                },
            })
            .into_response())
        }
        ResolvedRoute::Remote {
            provider_name,
            provider_url,
            provider_format,
            strip_auth,
            api_key,
            model_rewrite,
            routing_method,
            ..
        } => {
            if provider_format != ApiFormat::OpenAi {
                return Err(ServerError::BadRequest(
                    "Embeddings proxy only supported for OpenAI-format providers".to_owned(),
                ));
            }
            let proxy_body = if let Some(ref rewrite) = model_rewrite {
                crate::proxy::rewrite_model_in_body(&body, rewrite)?
            } else {
                body
            };
            let start = Instant::now();
            let response = crate::proxy::proxy_request(
                &state.http_client,
                &provider_url,
                "/v1/embeddings",
                proxy_body,
                &headers,
                strip_auth,
                api_key.as_deref(),
            )
            .await;
            if let Some(ref metrics) = state.metrics {
                let status = response.as_ref().map_or(502, |resp| resp.status().as_u16());
                metrics.record(RequestRecord {
                    id: 0,
                    timestamp: Instant::now(),
                    wallclock: chrono::Utc::now(),
                    model: req.model,
                    provider: provider_name,
                    routing_method: routing_method.into(),
                    status,
                    duration: start.elapsed(),
                    input_tokens: 0,
                    output_tokens: 0,
                    error_body: None,
                });
            }
            response
        }
    }
}
