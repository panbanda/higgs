use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json,
    extract::{Extension, State},
    http::HeaderMap,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive},
    },
};
use bytes::Bytes;
use tokio_stream::{Stream, StreamExt};

use crate::{
    anthropic_adapter::openai_finish_to_anthropic_stop,
    anthropic_tools::{
        messages_with_tool_history, openai_tools, tool_settings, tool_use_block,
        tool_use_stream_events,
    },
    config::ApiFormat,
    error::ServerError,
    generation::{
        ChatPrompt, Completed, EventStream, GenerationEvent, GenerationRequest, Prompt,
        StreamMetrics,
    },
    metrics::RequestMetricsContext,
    router::ResolvedRoute,
    state::SharedState,
    types::anthropic::{
        AnthropicUsage, ContentBlockResponse, ContentBlockStartEvent, ContentBlockStartPayload,
        ContentBlockStopEvent, CountTokensRequest, CountTokensResponse, CreateMessageRequest,
        CreateMessageResponse, MessageDelta, MessageDeltaEvent, MessageStartEvent,
        MessageStartPayload, MessageStopEvent, ResponseContentBlock, TextDelta,
    },
};
use higgs_models::SamplingParams;

fn requested_thinking_budget(
    thinking: Option<&crate::types::anthropic::ThinkingConfig>,
) -> Result<Option<u32>, ServerError> {
    crate::reasoning::validate_thinking_budget(
        thinking.and_then(|config| config.budget_tokens),
        "thinking.budget_tokens",
    )
}

#[allow(clippy::too_many_lines)]
pub async fn create_message(
    State(state): State<SharedState>,
    Extension(request_metrics): Extension<RequestMetricsContext>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ServerError> {
    // Captured before parsing and prompt preparation so TTFT measures
    // the client-observed wait, not only generation.
    let received_at = Instant::now();
    let mut req: CreateMessageRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid request body: {e}")))?;
    request_metrics.set_requested_model(&req.model);

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
        .await?;

    match resolved {
        ResolvedRoute::Higgs {
            engine,
            model_name,
            routing_method,
        } => {
            req.model = model_name;
            let start = Instant::now();
            let request = generation_request(&req)?;
            if req.stream == Some(true) {
                let events = crate::generation::stream(
                    Arc::clone(&state),
                    request,
                    engine,
                    state.metrics.clone().map(|store| StreamMetrics {
                        store,
                        routing_method,
                    }),
                    received_at,
                )?;
                let sse = Sse::new(message_event_stream(events, req.model))
                    .keep_alive(KeepAlive::default());
                if state.metrics.is_some() {
                    request_metrics.mark_recorded();
                }
                Ok(sse.into_response())
            } else {
                let completed =
                    crate::generation::complete(Arc::clone(&state), request, engine, received_at)
                        .await?;
                if let Some(ref metrics) = state.metrics {
                    crate::generation::record_completion(
                        metrics,
                        &req.model,
                        routing_method,
                        start,
                        &completed.usage,
                    );
                    request_metrics.mark_recorded();
                }
                let response = message_response(completed, req.model);
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
            let metrics_model = model_rewrite.as_deref().unwrap_or(&req.model).to_owned();
            let is_streaming = req.stream == Some(true);
            let mut usage = (0u64, 0u64);

            let result = match provider_format {
                ApiFormat::Anthropic => {
                    let proxy_body = if let Some(ref rewrite) = model_rewrite {
                        crate::proxy::rewrite_model_in_body(&body, rewrite)?
                    } else {
                        body
                    };
                    if is_streaming {
                        crate::proxy::proxy_request(
                            &state.http_client,
                            &provider_url,
                            "/v1/messages",
                            proxy_body,
                            &headers,
                            strip_auth,
                            api_key.as_ref().map(crate::config::Secret::expose),
                        )
                        .await
                    } else {
                        let (status, resp_bytes) = crate::proxy::send_and_read(
                            &state.http_client,
                            &provider_url,
                            "/v1/messages",
                            proxy_body,
                            &headers,
                            strip_auth,
                            api_key.as_ref().map(crate::config::Secret::expose),
                        )
                        .await?;
                        usage = crate::proxy::extract_usage(&resp_bytes);
                        Ok((
                            status,
                            [(axum::http::header::CONTENT_TYPE, "application/json")],
                            resp_bytes,
                        )
                            .into_response())
                    }
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
                        api_key.as_ref().map(crate::config::Secret::expose),
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
                        usage = crate::proxy::extract_usage(&resp_bytes);
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
                if !(200..300).contains(&status) {
                    usage = (0, 0);
                }
                metrics.record(crate::metrics::RequestRecord {
                    id: 0,
                    timestamp: Instant::now(),
                    wallclock: chrono::Utc::now(),
                    model: Some(metrics_model),
                    provider: Some(provider_name),
                    routing_method: routing_method.into(),
                    status,
                    duration: start.elapsed(),
                    input_tokens: usage.0,
                    output_tokens: usage.1,
                    error_body: None,
                    timing: crate::metrics::RequestTiming::default(),
                });
                request_metrics.mark_recorded();
            }
            result
        }
    }
}

/// Translate a Messages request into the shared request.
fn generation_request(req: &CreateMessageRequest) -> Result<GenerationRequest, ServerError> {
    let messages = messages_with_tool_history(&req.messages, req.system.as_ref());
    let (tools, tool_policy) = tool_settings(req.tools.as_deref(), req.tool_choice.as_ref())?;
    let thinking_budget = requested_thinking_budget(req.thinking.as_ref())?;
    Ok(GenerationRequest {
        model: req.model.clone(),
        prompt: Prompt::Chat(ChatPrompt {
            messages,
            images: Vec::new(),
            tools,
            tool_policy,
            reasoning: None,
            thinking_budget,
            parse_reasoning_without_thinking: true,
        }),
        max_tokens: req.max_tokens,
        sampling: SamplingParams {
            temperature: req.temperature.unwrap_or(1.0),
            top_p: req.top_p.unwrap_or(1.0),
            top_k: req.top_k,
            ..SamplingParams::default()
        },
        stop_sequences: req.stop_sequences.clone().unwrap_or_default(),
        logprobs: false,
        top_logprobs: None,
        constraint: None,
        mcp: None,
    })
}

fn message_response(completed: Completed, model: String) -> CreateMessageResponse {
    let usage = &completed.usage;
    let anthropic_usage = AnthropicUsage {
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
        cache_read_input_tokens: usage
            .prompt_tokens_details
            .as_ref()
            .map(|details| details.cached_tokens),
    };
    let stop_reason = openai_finish_to_anthropic_stop(&completed.finish_reason);
    let content = if completed.tool_calls.is_empty() {
        vec![ResponseContentBlock::Text(ContentBlockResponse {
            block_type: "text",
            text: completed.text,
        })]
    } else {
        let text_block = (!completed.text.is_empty()).then(|| {
            ResponseContentBlock::Text(ContentBlockResponse {
                block_type: "text",
                text: completed.text.clone(),
            })
        });
        text_block
            .into_iter()
            .chain(
                completed
                    .tool_calls
                    .iter()
                    .map(|call| ResponseContentBlock::ToolUse(tool_use_block(call))),
            )
            .collect()
    };
    CreateMessageResponse {
        id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
        message_type: "message",
        role: "assistant",
        content,
        model,
        stop_reason: Some(stop_reason),
        usage: anthropic_usage,
    }
}

/// Render generation events as Messages server-sent events. Reasoning is not
/// part of the stream; tool calls follow the text as `tool_use` blocks.
#[allow(clippy::too_many_lines)]
fn message_event_stream(
    mut events: EventStream,
    model: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    async_stream::stream! {
        let mut prompt_tokens: u32 = 0;
        let mut output_tokens: u32 = 0;
        let mut cache_read_input_tokens: Option<u32> = None;
        let mut stop_reason: Option<String> = None;
        let mut delta_writer = crate::sse::AnthropicDeltaWriter::new();
        let mut tool_calls = Vec::new();

        while let Some(event) = events.next().await {
            match event {
                GenerationEvent::Started { prompt_tokens: count } => {
                    prompt_tokens = count;
                    let start_event = MessageStartEvent {
                        event_type: "message_start",
                        message: MessageStartPayload {
                            id: message_id.clone(),
                            message_type: "message",
                            role: "assistant",
                            content: vec![],
                            model: model.clone(),
                            stop_reason: None,
                            usage: AnthropicUsage {
                                input_tokens: prompt_tokens,
                                output_tokens: 0,
                                cache_read_input_tokens: None,
                            },
                        },
                    };
                    match serde_json::to_string(&start_event) {
                        Ok(json) => yield Ok(Event::default().event("message_start").data(json)),
                        Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
                    }
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
                }
                GenerationEvent::TextDelta { text, .. } => {
                    let delta = TextDelta {
                        delta_type: "text_delta",
                        text,
                    };
                    match delta_writer.write(&delta) {
                        Ok(json) => yield Ok(Event::default().event("content_block_delta").data(json)),
                        Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
                    }
                }
                GenerationEvent::ToolCall(call) => tool_calls.push(call),
                GenerationEvent::Finish { reason, .. } => {
                    stop_reason = Some(openai_finish_to_anthropic_stop(&reason));
                }
                GenerationEvent::Usage(usage) => {
                    output_tokens = usage.completion_tokens;
                    cache_read_input_tokens = usage
                        .prompt_tokens_details
                        .map(|details| details.cached_tokens);
                }
                GenerationEvent::PrefillProgress(_)
                | GenerationEvent::ReasoningDelta(_)
                | GenerationEvent::ToolLoop(_)
                | GenerationEvent::Error(_) => {}
            }
        }

        let block_stop = ContentBlockStopEvent {
            event_type: "content_block_stop",
            index: 0,
        };
        match serde_json::to_string(&block_stop) {
            Ok(json) => yield Ok(Event::default().event("content_block_stop").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        // Tool calls follow the text block.
        for (offset, call) in tool_calls.iter().enumerate() {
            let index = u32::try_from(offset + 1).unwrap_or(u32::MAX);
            for (event_name, data) in tool_use_stream_events(index, call) {
                yield Ok(Event::default().event(event_name).data(data));
            }
        }
        if !tool_calls.is_empty() {
            stop_reason = Some("tool_use".to_owned());
        }

        let message_delta = MessageDeltaEvent {
            event_type: "message_delta",
            delta: MessageDelta { stop_reason },
            usage: AnthropicUsage {
                input_tokens: prompt_tokens,
                output_tokens,
                cache_read_input_tokens,
            },
        };
        match serde_json::to_string(&message_delta) {
            Ok(json) => yield Ok(Event::default().event("message_delta").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        let message_stop = MessageStopEvent {
            event_type: "message_stop",
        };
        match serde_json::to_string(&message_stop) {
            Ok(json) => yield Ok(Event::default().event("message_stop").data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }
    }
}

pub async fn count_tokens(
    State(state): State<SharedState>,
    Extension(request_metrics): Extension<RequestMetricsContext>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ServerError> {
    let req: CountTokensRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid request body: {e}")))?;
    request_metrics.set_requested_model(&req.model);

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
        .await?;

    match resolved {
        ResolvedRoute::Higgs {
            engine, model_name, ..
        } => {
            let engine_messages = messages_with_tool_history(&req.messages, req.system.as_ref());
            let converted_tools = req.tools.as_deref().map(openai_tools);
            let tools = converted_tools.as_deref();
            let thinking_enabled = crate::reasoning::effective_thinking_enabled(
                engine.enable_thinking(),
                &[engine.model_name(), model_name.as_str()],
                None,
            );

            let tokens = engine
                .prepare_chat_prompt(&engine_messages, tools, thinking_enabled)
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
                api_key.as_ref().map(crate::config::Secret::expose),
            )
            .await
        }
    }
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use crate::test_support::{
        StubEngine, app_with_stub, generation_output, post_json, progress_event, token_event,
    };
    use higgs_engine::engine::GenerationStats;

    fn body(stream: bool) -> serde_json::Value {
        serde_json::json!({
            "model": "stub",
            "max_tokens": 16,
            "stream": stream,
            "messages": [{"role": "user", "content": "hi"}],
        })
    }

    #[tokio::test]
    async fn non_streaming_reports_cache_read_input_tokens() {
        let mut stub = StubEngine::named("stub");
        let mut output = generation_output("hello", 9, 2);
        output.stats = GenerationStats {
            cached_tokens: Some(6),
            ..GenerationStats::default()
        };
        stub.script.generation = Some(output);
        let (app, _engine) = app_with_stub(stub);

        let (status, text) = post_json(&app, "/v1/messages", &body(false)).await;

        assert_eq!(status, http::StatusCode::OK, "{text}");
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["usage"]["input_tokens"], 9);
        assert_eq!(json["usage"]["cache_read_input_tokens"], 6);
    }

    #[tokio::test]
    async fn non_streaming_omits_cache_read_when_unknown() {
        let mut stub = StubEngine::named("stub");
        stub.script.generation = Some(generation_output("hello", 9, 2));
        let (app, _engine) = app_with_stub(stub);

        let (_status, text) = post_json(&app, "/v1/messages", &body(false)).await;

        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(
            json["usage"].get("cache_read_input_tokens").is_none(),
            "{text}"
        );
    }

    #[tokio::test]
    async fn streaming_message_delta_reports_cache_read_input_tokens() {
        let mut stub = StubEngine::named("stub");
        stub.script.prompt_tokens = vec![1, 2, 3, 4];
        stub.script.stream = vec![
            progress_event(3, 4),
            token_event("hi", 1, None),
            token_event("!", 2, Some("stop")),
        ];
        let (app, _engine) = app_with_stub(stub);

        let (status, text) = post_json(&app, "/v1/messages", &body(true)).await;

        assert_eq!(status, http::StatusCode::OK, "{text}");
        let delta_line = text
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .find(|data| data.contains("\"message_delta\""))
            .unwrap();
        let delta: serde_json::Value = serde_json::from_str(delta_line.trim()).unwrap();
        assert_eq!(delta["usage"]["cache_read_input_tokens"], 3);
        assert_eq!(delta["usage"]["input_tokens"], 4);
        assert_eq!(delta["usage"]["output_tokens"], 2);
    }

    fn thinking_stub() -> StubEngine {
        let mut stub = StubEngine::named("stub");
        stub.script.enable_thinking = true;
        stub.script.generation = Some(generation_output("hello", 9, 2));
        stub
    }

    #[tokio::test]
    async fn thinking_budget_tokens_sets_the_budget() {
        let (app, engine) = app_with_stub(thinking_stub());
        let mut request = body(false);
        request["thinking"] = serde_json::json!({"type": "enabled", "budget_tokens": 2048});

        let (status, text) = post_json(&app, "/v1/messages", &request).await;

        assert_eq!(status, http::StatusCode::OK, "{text}");
        let calls = engine.recorded_calls();
        assert_eq!(calls[0].thinking_budget, Some(2048));
        assert!(calls[0].thinking_enabled);
    }

    #[tokio::test]
    async fn absent_thinking_leaves_the_budget_to_configuration() {
        let (app, engine) = app_with_stub(thinking_stub());

        let (status, _text) = post_json(&app, "/v1/messages", &body(false)).await;

        assert_eq!(status, http::StatusCode::OK);
        assert_eq!(engine.recorded_calls()[0].thinking_budget, None);
    }

    #[tokio::test]
    async fn zero_budget_tokens_is_rejected() {
        let (app, engine) = app_with_stub(thinking_stub());
        let mut request = body(false);
        request["thinking"] = serde_json::json!({"type": "enabled", "budget_tokens": 0});

        let (status, text) = post_json(&app, "/v1/messages", &request).await;

        assert_eq!(status, http::StatusCode::BAD_REQUEST, "{text}");
        assert!(
            text.contains("thinking.budget_tokens must be at least 1"),
            "{text}"
        );
        assert!(engine.recorded_calls().is_empty());
    }
}
