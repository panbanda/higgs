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
    anthropic_adapter::{anthropic_messages_to_engine, openai_finish_to_anthropic_stop},
    config::ApiFormat,
    error::ServerError,
    metrics::MetricsStore,
    router::ResolvedRoute,
    state::{Engine, SharedState},
    types::anthropic::{
        AnthropicUsage, ContentBlockDeltaEvent, ContentBlockResponse, ContentBlockStartEvent,
        ContentBlockStartPayload, ContentBlockStopEvent, CountTokensRequest, CountTokensResponse,
        CreateMessageRequest, CreateMessageResponse, MessageDelta, MessageDeltaEvent,
        MessageStartEvent, MessageStartPayload, MessageStopEvent, TextDelta,
    },
};
use higgs_models::SamplingParams;

#[allow(clippy::too_many_lines)]
pub async fn create_message(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ServerError> {
    let req: CreateMessageRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid request body: {e}")))?;

    if req.messages.is_empty() {
        return Err(ServerError::BadRequest(
            "messages array must not be empty".to_owned(),
        ));
    }

    let messages_json = serde_json::to_value(&req.messages).ok().and_then(|v| {
        if let serde_json::Value::Array(a) = v {
            Some(a)
        } else {
            None
        }
    });
    let resolved = state
        .router
        .resolve(&req.model, messages_json.as_deref())
        .await
        .map_err(ServerError::ModelNotFound)?;

    match resolved {
        ResolvedRoute::Higgs {
            engine,
            routing_method,
            ..
        } => {
            let start = Instant::now();
            if req.stream == Some(true) {
                let stream =
                    create_message_stream(req, engine, state.metrics.clone(), routing_method)?;
                let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                Ok(sse.into_response())
            } else {
                let response = create_message_non_streaming(req, engine).await?;
                if let Some(ref metrics) = state.metrics {
                    metrics.record(crate::metrics::RequestRecord {
                        id: 0,
                        timestamp: Instant::now(),
                        wallclock: chrono::Utc::now(),
                        model: response.model.clone(),
                        provider: "higgs".to_owned(),
                        routing_method: routing_method.into(),
                        status: 200,
                        duration: start.elapsed(),
                        input_tokens: u64::from(response.usage.input_tokens),
                        output_tokens: u64::from(response.usage.output_tokens),
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
            let start = Instant::now();
            let model_name = req.model.clone();
            let is_streaming = req.stream == Some(true);
            let result = match provider_format {
                ApiFormat::Anthropic => {
                    let proxy_body = if let Some(ref rewrite) = model_rewrite {
                        crate::proxy::rewrite_model_in_body(&body, rewrite)?
                    } else {
                        body
                    };
                    crate::proxy::proxy_request(
                        &state.http_client,
                        &provider_url,
                        "/v1/messages",
                        proxy_body,
                        &headers,
                        strip_auth,
                        api_key.as_deref(),
                    )
                    .await
                }
                ApiFormat::OpenAi => {
                    let translated = crate::translate::anthropic_to_openai_request(&body)?;
                    let proxy_body = if let Some(ref rewrite) = model_rewrite {
                        crate::proxy::rewrite_model_in_body(&translated, rewrite)?
                    } else {
                        translated
                    };

                    let upstream = crate::proxy::send_to_provider(
                        &state.http_client,
                        &provider_url,
                        "/v1/chat/completions",
                        proxy_body,
                        &headers,
                        strip_auth,
                        api_key.as_deref(),
                    )
                    .await?;
                    let upstream_status = upstream.status().as_u16();

                    if upstream_status >= 400 {
                        let status_code = axum::http::StatusCode::from_u16(upstream_status)
                            .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
                        let resp_bytes = upstream.bytes().await.map_err(|e| {
                            ServerError::ProxyError(format!("Failed to read response: {e}"))
                        })?;
                        Ok((
                            status_code,
                            [(axum::http::header::CONTENT_TYPE, "application/json")],
                            resp_bytes,
                        )
                            .into_response())
                    } else if is_streaming {
                        let stream =
                            crate::translate::openai_stream_to_anthropic(upstream, req.model);
                        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                        Ok(sse.into_response())
                    } else {
                        let resp_bytes = upstream.bytes().await.map_err(|e| {
                            ServerError::ProxyError(format!("Failed to read response: {e}"))
                        })?;
                        let translated_resp = crate::translate::openai_response_to_anthropic(
                            &resp_bytes,
                            &req.model,
                        )?;
                        Ok((
                            [(axum::http::header::CONTENT_TYPE, "application/json")],
                            translated_resp,
                        )
                            .into_response())
                    }
                }
            };
            if let Some(ref metrics) = state.metrics {
                let status = result.as_ref().map_or(502, |resp| resp.status().as_u16());
                metrics.record(crate::metrics::RequestRecord {
                    id: 0,
                    timestamp: Instant::now(),
                    wallclock: chrono::Utc::now(),
                    model: model_name,
                    provider: provider_name,
                    routing_method: routing_method.into(),
                    status,
                    duration: start.elapsed(),
                    input_tokens: 0,
                    output_tokens: 0,
                    error_body: None,
                });
            }
            result
        }
    }
}

async fn create_message_non_streaming(
    req: CreateMessageRequest,
    engine: Arc<Engine>,
) -> Result<CreateMessageResponse, ServerError> {
    let max_tokens = req.max_tokens;
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k,
        ..SamplingParams::default()
    };
    let stop_sequences = req.stop_sequences.unwrap_or_default();

    let engine_messages = anthropic_messages_to_engine(&req.messages, req.system.as_deref());
    let tools = req.tools.as_deref();

    let prompt_tokens = engine
        .prepare_chat_prompt(&engine_messages, tools)
        .map_err(ServerError::Engine)?;

    let output = tokio::task::spawn_blocking(move || {
        engine.generate(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            false,
            None,
            None,
            None,
        )
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
    .map_err(ServerError::Engine)?;

    let stop_reason = openai_finish_to_anthropic_stop(&output.finish_reason);
    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());

    Ok(CreateMessageResponse {
        id: msg_id,
        message_type: "message",
        role: "assistant",
        content: vec![ContentBlockResponse {
            block_type: "text",
            text: output.text,
        }],
        model: req.model,
        stop_reason: Some(stop_reason),
        usage: AnthropicUsage {
            input_tokens: output.prompt_tokens,
            output_tokens: output.completion_tokens,
        },
    })
}

#[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
fn create_message_stream(
    req: CreateMessageRequest,
    engine: Arc<Engine>,
    metrics: Option<Arc<MetricsStore>>,
    routing_method: crate::router::RoutingMethod,
) -> Result<impl Stream<Item = Result<Event, Infallible>>, ServerError> {
    let max_tokens = req.max_tokens;
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k,
        ..SamplingParams::default()
    };
    let stop_sequences = req.stop_sequences.unwrap_or_default();

    let engine_messages = anthropic_messages_to_engine(&req.messages, req.system.as_deref());
    let tools = req.tools.as_deref();

    let prompt_tokens = engine
        .prepare_chat_prompt(&engine_messages, tools)
        .map_err(ServerError::Engine)?;

    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let model = req.model;
    let prompt_token_count = u32::try_from(prompt_tokens.len())
        .map_err(|_| ServerError::BadRequest("Token count overflow".to_owned()))?;

    // Spawn generation before creating the stream so prefill starts immediately
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    tokio::task::spawn_blocking(move || {
        let result = engine.generate_streaming(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            false,
            None,
            &tx,
            None,
            None,
        );
        if let Err(e) = result {
            tracing::error!(error = %e, "Generation error during Anthropic streaming");
        }
    });

    let start = Instant::now();
    let metrics_id = metrics.as_ref().map(|m| {
        m.record_pending(crate::metrics::RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: chrono::Utc::now(),
            model: model.clone(),
            provider: "higgs".to_owned(),
            routing_method: routing_method.into(),
            status: 200,
            duration: std::time::Duration::ZERO,
            input_tokens: u64::from(prompt_token_count),
            output_tokens: 0,
            error_body: None,
        })
    });

    let stream = async_stream::stream! {
        // 1. message_start
        let start_event = MessageStartEvent {
            event_type: "message_start",
            message: MessageStartPayload {
                id: msg_id.clone(),
                message_type: "message",
                role: "assistant",
                content: vec![],
                model: model.clone(),
                stop_reason: None,
                usage: AnthropicUsage {
                    input_tokens: prompt_token_count,
                    output_tokens: 0,
                },
            },
        };
        match serde_json::to_string(&start_event) {
            Ok(json) => yield Ok(Event::default().event("message_start").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        // 2. content_block_start
        let block_start = ContentBlockStartEvent {
            event_type: "content_block_start",
            index: 0,
            content_block: ContentBlockStartPayload {
                block_type: "text",
                text: String::new(),
            },
        };
        match serde_json::to_string(&block_start) {
            Ok(json) => yield Ok(Event::default().event("content_block_start").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        // 3. content_block_delta events (one per token)
        let mut final_stop_reason = None;
        let mut total_output_tokens: u32 = 0;

        while let Some(output) = rx.recv().await {
            if !output.new_text.is_empty() {
                let delta_event = ContentBlockDeltaEvent {
                    event_type: "content_block_delta",
                    index: 0,
                    delta: TextDelta {
                        delta_type: "text_delta",
                        text: output.new_text,
                    },
                };
                match serde_json::to_string(&delta_event) {
                    Ok(json) => yield Ok(Event::default().event("content_block_delta").data(json)),
                    Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
                }
            }
            total_output_tokens = output.completion_tokens;
            if let Some(reason) = output.finish_reason {
                final_stop_reason = Some(openai_finish_to_anthropic_stop(&reason));
            }
        }

        if let Some(ref m) = metrics {
            if let Some(id) = metrics_id {
                m.finalize_stream(id, u64::from(total_output_tokens), start.elapsed());
            }
        }

        // 4. content_block_stop
        let block_stop = ContentBlockStopEvent {
            event_type: "content_block_stop",
            index: 0,
        };
        match serde_json::to_string(&block_stop) {
            Ok(json) => yield Ok(Event::default().event("content_block_stop").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        // 5. message_delta
        let msg_delta = MessageDeltaEvent {
            event_type: "message_delta",
            delta: MessageDelta {
                stop_reason: final_stop_reason,
            },
            usage: AnthropicUsage {
                input_tokens: prompt_token_count,
                output_tokens: total_output_tokens,
            },
        };
        match serde_json::to_string(&msg_delta) {
            Ok(json) => yield Ok(Event::default().event("message_delta").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        // 6. message_stop
        let msg_stop = MessageStopEvent {
            event_type: "message_stop",
        };
        match serde_json::to_string(&msg_stop) {
            Ok(json) => yield Ok(Event::default().event("message_stop").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }
    };

    Ok(stream)
}

pub async fn count_tokens(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ServerError> {
    let req: CountTokensRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid request body: {e}")))?;

    let messages_json = serde_json::to_value(&req.messages).ok().and_then(|v| {
        if let serde_json::Value::Array(a) = v {
            Some(a)
        } else {
            None
        }
    });
    let resolved = state
        .router
        .resolve(&req.model, messages_json.as_deref())
        .await
        .map_err(ServerError::ModelNotFound)?;

    match resolved {
        ResolvedRoute::Higgs { engine, .. } => {
            let engine_messages =
                anthropic_messages_to_engine(&req.messages, req.system.as_deref());
            let tools = req.tools.as_deref();

            let tokens = engine
                .prepare_chat_prompt(&engine_messages, tools)
                .map_err(ServerError::Engine)?;

            let count = u32::try_from(tokens.len())
                .map_err(|_| ServerError::BadRequest("Token count overflow".to_owned()))?;

            Ok(Json(CountTokensResponse {
                input_tokens: count,
            })
            .into_response())
        }
        ResolvedRoute::Remote {
            stub_count_tokens,
            provider_url,
            provider_format,
            strip_auth,
            api_key,
            model_rewrite,
            ..
        } => {
            if stub_count_tokens || provider_format != ApiFormat::Anthropic {
                // OpenAI providers have no count_tokens equivalent; return stub
                return Ok(crate::proxy::stub_count_tokens_response());
            }
            let proxy_body = if let Some(ref rewrite) = model_rewrite {
                crate::proxy::rewrite_model_in_body(&body, rewrite)?
            } else {
                body
            };
            crate::proxy::proxy_request(
                &state.http_client,
                &provider_url,
                "/v1/messages/count_tokens",
                proxy_body,
                &headers,
                strip_auth,
                api_key.as_deref(),
            )
            .await
        }
    }
}
