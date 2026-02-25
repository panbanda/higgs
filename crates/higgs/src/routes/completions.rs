use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive},
    },
};
use bytes::Bytes;
use tokio_stream::Stream;

use crate::{
    config::ApiFormat,
    error::ServerError,
    metrics::{MetricsStore, RequestRecord},
    router::ResolvedRoute,
    state::{Engine, SharedState},
    types::openai::{
        ChoiceLogprobs, CompletionChoice, CompletionChunk, CompletionChunkChoice,
        CompletionRequest, CompletionResponse, CompletionUsage, StopSequence, TokenLogprob,
        TopLogprob,
    },
};
use higgs_models::SamplingParams;

#[allow(clippy::too_many_lines)]
pub async fn completions(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ServerError> {
    let req: CompletionRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid request body: {e}")))?;

    if req.prompt.is_empty() {
        return Err(ServerError::BadRequest(
            "prompt must not be empty".to_owned(),
        ));
    }

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
            if req.stream == Some(true) {
                let metrics = state.metrics.clone();
                let stream = completions_stream(
                    Arc::clone(&state),
                    req,
                    engine,
                    model_name,
                    metrics,
                    routing_method,
                )?;
                let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                Ok(sse.into_response())
            } else {
                let start = Instant::now();
                let response = completions_non_streaming(Arc::clone(&state), req, engine).await?;
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
                        input_tokens: u64::from(response.usage.prompt_tokens),
                        output_tokens: u64::from(response.usage.completion_tokens),
                        error_body: None,
                    });
                }
                Ok(Json(response).into_response())
            }
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
                    "Cross-format proxying not yet supported for /v1/completions".to_owned(),
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
                "/v1/completions",
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

async fn completions_non_streaming(
    state: SharedState,
    req: CompletionRequest,
    engine: Arc<Engine>,
) -> Result<CompletionResponse, ServerError> {
    let max_tokens = req.max_tokens.unwrap_or(state.config.server.max_tokens);
    let sampling = build_sampling_params(&req);
    let stop_sequences = StopSequence::extract(req.stop);
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs = req.top_logprobs;

    let encoding = engine
        .tokenizer()
        .encode(req.prompt.as_str(), false)
        .map_err(|e| ServerError::BadRequest(format!("Tokenization error: {e}")))?;
    let prompt_tokens = encoding.get_ids().to_vec();

    let tokenizer = engine.tokenizer().clone();
    let output = tokio::task::spawn_blocking(move || {
        engine.generate(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            want_logprobs,
            top_logprobs,
            None,
            None,
        )
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
    .map_err(ServerError::Engine)?;

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());

    let logprobs_response = output
        .token_logprobs
        .as_ref()
        .map(|lps| logprobs_to_response(lps, &tokenizer));

    Ok(CompletionResponse {
        id: request_id,
        object: "text_completion",
        created: chrono::Utc::now().timestamp(),
        model: req.model,
        choices: vec![CompletionChoice {
            index: 0,
            text: output.text,
            finish_reason: output.finish_reason,
            logprobs: logprobs_response,
        }],
        usage: CompletionUsage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        },
    })
}

#[allow(clippy::needless_pass_by_value)]
fn completions_stream(
    state: SharedState,
    req: CompletionRequest,
    engine: Arc<Engine>,
    model_name: String,
    metrics: Option<Arc<MetricsStore>>,
    routing_method: crate::router::RoutingMethod,
) -> Result<impl Stream<Item = Result<Event, Infallible>>, ServerError> {
    let max_tokens = req.max_tokens.unwrap_or(state.config.server.max_tokens);
    let sampling = build_sampling_params(&req);
    let stop_sequences = StopSequence::extract(req.stop);
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs = req.top_logprobs;

    let encoding = engine
        .tokenizer()
        .encode(req.prompt.as_str(), false)
        .map_err(|e| ServerError::BadRequest(format!("Tokenization error: {e}")))?;
    let prompt_tokens = encoding.get_ids().to_vec();
    let input_token_count = u64::try_from(prompt_tokens.len()).unwrap_or(u64::MAX);

    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let model = req.model;

    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    tokio::task::spawn_blocking(move || {
        let result = engine.generate_streaming(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            want_logprobs,
            top_logprobs,
            &tx,
            None,
            None,
        );
        if let Err(e) = result {
            tracing::error!(error = %e, "Generation error during streaming");
        }
    });

    let start = Instant::now();
    let pending_id = metrics.as_ref().map(|m| {
        m.record_pending(RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: chrono::Utc::now(),
            model: model_name,
            provider: "higgs".to_owned(),
            routing_method: routing_method.into(),
            status: 200,
            duration: std::time::Duration::ZERO,
            input_tokens: input_token_count,
            output_tokens: 0,
            error_body: None,
        })
    });

    let stream = async_stream::stream! {
        let mut output_tokens: u64 = 0;
        while let Some(output) = rx.recv().await {
            output_tokens = u64::from(output.completion_tokens);
            let chunk = CompletionChunk {
                id: request_id.clone(),
                object: "text_completion",
                created,
                model: model.clone(),
                choices: vec![CompletionChunkChoice {
                    index: 0,
                    text: output.new_text,
                    finish_reason: output.finish_reason,
                }],
            };
            match serde_json::to_string(&chunk) {
                Ok(json) => yield Ok(Event::default().data(json)),
                Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
            }
        }

        if let Some(ref m) = metrics {
            if let Some(id) = pending_id {
                m.finalize_stream(id, output_tokens, start.elapsed());
            }
        }

        yield Ok(Event::default().data("[DONE]"));
    };

    Ok(stream)
}

fn build_sampling_params(req: &CompletionRequest) -> SamplingParams {
    SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k,
        min_p: req.min_p,
        repetition_penalty: req.repetition_penalty,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
    }
}

fn logprobs_to_response(
    infos: &[higgs_models::TokenLogprobInfo],
    tokenizer: &higgs_engine::tokenizers::Tokenizer,
) -> ChoiceLogprobs {
    let content = infos
        .iter()
        .map(|info| {
            let token_str = tokenizer
                .decode(&[info.token_id], false)
                .unwrap_or_default();
            let top = info
                .top_logprobs
                .iter()
                .map(|e| {
                    let t = tokenizer.decode(&[e.token_id], false).unwrap_or_default();
                    TopLogprob {
                        token: t,
                        logprob: e.logprob,
                    }
                })
                .collect();
            TokenLogprob {
                token: token_str,
                logprob: info.logprob,
                top_logprobs: top,
            }
        })
        .collect();
    ChoiceLogprobs { content }
}
