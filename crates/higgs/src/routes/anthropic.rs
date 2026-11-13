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
use tokio_stream::Stream;

use super::chat::build_constraint;
use crate::{
    anthropic_adapter::openai_finish_to_anthropic_stop,
    anthropic_tools::{
        ToolPlan, messages_with_tool_history, openai_tools, tool_use_block, tool_use_stream_events,
    },
    config::ApiFormat,
    error::ServerError,
    metrics::{MetricsStore, RequestMetricsContext, RequestTiming},
    router::ResolvedRoute,
    state::{EngineLease, SharedState},
    types::anthropic::{
        AnthropicUsage, ContentBlockResponse, ContentBlockStartEvent, ContentBlockStartPayload,
        ContentBlockStopEvent, CountTokensRequest, CountTokensResponse, CreateMessageRequest,
        CreateMessageResponse, MessageDelta, MessageDeltaEvent, MessageStartEvent,
        MessageStartPayload, MessageStopEvent, ResponseContentBlock, TextDelta,
    },
};
use higgs_engine::engine::ThinkingMode;
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
            if req.stream == Some(true) {
                let stream = create_message_stream(
                    req,
                    engine,
                    state.metrics.clone(),
                    routing_method,
                    received_at,
                )?;
                let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                if state.metrics.is_some() {
                    request_metrics.mark_recorded();
                }
                Ok(sse.into_response())
            } else {
                let response = create_message_non_streaming(req, engine).await?;
                if let Some(ref metrics) = state.metrics {
                    metrics.record(crate::metrics::RequestRecord {
                        id: 0,
                        timestamp: Instant::now(),
                        wallclock: chrono::Utc::now(),
                        model: Some(response.model.clone()),
                        provider: Some("higgs".to_owned()),
                        routing_method: routing_method.into(),
                        status: 200,
                        duration: start.elapsed(),
                        input_tokens: u64::from(response.usage.input_tokens),
                        output_tokens: u64::from(response.usage.output_tokens),
                        error_body: None,
                        timing: crate::metrics::RequestTiming::default(),
                    });
                    request_metrics.mark_recorded();
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

#[allow(clippy::too_many_lines)]
async fn create_message_non_streaming(
    req: CreateMessageRequest,
    engine: EngineLease,
) -> Result<CreateMessageResponse, ServerError> {
    let max_tokens = req.max_tokens;
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k,
        ..SamplingParams::default()
    };
    let stop_sequences = req.stop_sequences.unwrap_or_default();

    let engine_messages = messages_with_tool_history(&req.messages, req.system.as_ref());
    let plan = ToolPlan::for_request(
        req.tools.as_deref(),
        req.tool_choice.as_ref(),
        engine.tool_call_format(),
    )?;
    let tools = plan.tools.as_deref();
    let thinking_budget = requested_thinking_budget(req.thinking.as_ref())?;
    let thinking_enabled =
        crate::reasoning::thinking_enabled_for_request(&engine, &plan.policy, &req.model, None);

    let prompt_tokens = engine
        .prepare_chat_prompt_with_thinking(&engine_messages, tools, thinking_enabled)
        .map_err(ServerError::Engine)?;
    let constraint = build_constraint(plan.constraint.as_ref(), &engine)?;

    let output = tokio::task::spawn_blocking(move || {
        engine.generate_with_thinking(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            false,
            None,
            ThinkingMode::new(thinking_enabled, thinking_budget),
            constraint,
            None,
        )
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
    .map_err(ServerError::Engine)?;

    let stop_reason = openai_finish_to_anthropic_stop(&output.finish_reason);
    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());

    let output_text = output.text;
    // When thinking is enabled, the template already opened `<think>` so
    // the model output starts inside the thinking block. We wrap it in a
    // properly closed `<think>...</think>` before parsing so the parser
    // always sees balanced tags, even if the model was length-stopped.
    let parse_input = if thinking_enabled {
        if output_text.contains("</think>") {
            format!("<think>{output_text}")
        } else {
            format!("<think>{output_text}</think>")
        }
    } else {
        output_text.clone()
    };
    let reasoning_result = higgs_engine::reasoning_parser::parse_reasoning(&parse_input);
    let visible_text = if reasoning_result.reasoning.is_some() {
        reasoning_result.text
    } else {
        output_text
    };

    let with_calls = plan
        .tools
        .as_ref()
        .map(|_| plan.parser.parse(&visible_text))
        .filter(|parsed_output| !parsed_output.tool_calls.is_empty());
    let (content, response_stop_reason) = if let Some(parsed_output) = with_calls {
        let text_block = (!parsed_output.text.is_empty()).then(|| {
            ResponseContentBlock::Text(ContentBlockResponse {
                block_type: "text",
                text: parsed_output.text.clone(),
            })
        });
        let blocks = text_block
            .into_iter()
            .chain(
                parsed_output
                    .tool_calls
                    .iter()
                    .map(|call| ResponseContentBlock::ToolUse(tool_use_block(call))),
            )
            .collect();
        (blocks, "tool_use".to_owned())
    } else {
        (
            vec![ResponseContentBlock::Text(ContentBlockResponse {
                block_type: "text",
                text: visible_text,
            })],
            stop_reason,
        )
    };

    Ok(CreateMessageResponse {
        id: msg_id,
        message_type: "message",
        role: "assistant",
        content,
        model: req.model,
        stop_reason: Some(response_stop_reason),
        usage: AnthropicUsage {
            input_tokens: output.prompt_tokens,
            output_tokens: output.completion_tokens,
            cache_read_input_tokens: output.stats.cached_tokens,
        },
    })
}

#[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
fn create_message_stream(
    req: CreateMessageRequest,
    engine: EngineLease,
    metrics: Option<Arc<MetricsStore>>,
    routing_method: crate::router::RoutingMethod,
    received_at: Instant,
) -> Result<impl Stream<Item = Result<Event, Infallible>>, ServerError> {
    let max_tokens = req.max_tokens;
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k,
        ..SamplingParams::default()
    };
    let stop_sequences = req.stop_sequences.unwrap_or_default();

    let engine_messages = messages_with_tool_history(&req.messages, req.system.as_ref());
    let plan = ToolPlan::for_request(
        req.tools.as_deref(),
        req.tool_choice.as_ref(),
        engine.tool_call_format(),
    )?;
    let tools = plan.tools.as_deref();
    let thinking_budget = requested_thinking_budget(req.thinking.as_ref())?;
    let thinking_enabled =
        crate::reasoning::thinking_enabled_for_request(&engine, &plan.policy, &req.model, None);

    let prompt_tokens = engine
        .prepare_chat_prompt_with_thinking(&engine_messages, tools, thinking_enabled)
        .map_err(ServerError::Engine)?;
    let constraint = build_constraint(plan.constraint.as_ref(), &engine)?;
    let mut tool_tracker = plan.parser.tracker(plan.tools.is_some());

    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let model = req.model;
    let prompt_token_count = u32::try_from(prompt_tokens.len())
        .map_err(|_| ServerError::BadRequest("Token count overflow".to_owned()))?;

    // Spawn generation before creating the stream so prefill starts immediately
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    tokio::task::spawn_blocking(move || {
        let result = engine.generate_streaming_with_thinking(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            false,
            None,
            &tx,
            ThinkingMode::new(thinking_enabled, thinking_budget),
            // Progress events are consumed for metrics only; the Anthropic
            // stream never surfaces them.
            true,
            constraint,
            None,
        );
        if let Err(e) = result {
            tracing::error!(error = %e, "Generation error during Anthropic streaming");
        }
    });

    let start = received_at;
    let metrics_id = metrics.as_ref().map(|m| {
        m.record_pending(crate::metrics::RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: chrono::Utc::now(),
            model: Some(model.clone()),
            provider: Some("higgs".to_owned()),
            routing_method: routing_method.into(),
            status: 200,
            duration: std::time::Duration::ZERO,
            input_tokens: u64::from(prompt_token_count),
            output_tokens: 0,
            error_body: None,
            timing: crate::metrics::RequestTiming::default(),
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
                    cache_read_input_tokens: None,
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
        let mut reasoning_tracker = if thinking_enabled {
            higgs_engine::reasoning_parser::StreamingReasoningTracker::new_inside_think()
        } else {
            higgs_engine::reasoning_parser::StreamingReasoningTracker::new()
        };

        let mut delta_writer = crate::sse::AnthropicDeltaWriter::new();
        let mut tool_calls = Vec::new();
        let mut timing = RequestTiming::default();
        while let Some(output) = rx.recv().await {
            if let Some(p) = output.prefill_progress {
                timing.cached_tokens = Some(u64::from(p.cached));
                continue;
            }
            if timing.ttft_ms.is_none() {
                timing.ttft_ms = Some(u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX));
            }
            let (reasoning_visible, _reasoning) = reasoning_tracker.process(&output.new_text);
            let tool_out = tool_tracker.process(&reasoning_visible);
            tool_calls.extend(tool_out.new_tool_calls);

            if !tool_out.visible.is_empty() {
                let td = TextDelta {
                    delta_type: "text_delta",
                    text: tool_out.visible,
                };
                match delta_writer.write(&td) {
                    Ok(json) => yield Ok(Event::default().event("content_block_delta").data(json)),
                    Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
                }
            }
            total_output_tokens = output.completion_tokens;
            if let Some(reason) = output.finish_reason {
                final_stop_reason = Some(openai_finish_to_anthropic_stop(&reason));
            }
        }

        // Flush any remaining visible text buffered by the reasoning tracker
        let (flush_visible, _flush_reasoning) = reasoning_tracker.flush();
        let flush_tool_out = tool_tracker.process(&flush_visible);
        let final_tool_out = tool_tracker.flush();
        tool_calls.extend(flush_tool_out.new_tool_calls);
        tool_calls.extend(final_tool_out.new_tool_calls);
        for remaining_text in [flush_tool_out.visible, final_tool_out.visible] {
            if remaining_text.is_empty() {
                continue;
            }
            let td = TextDelta {
                delta_type: "text_delta",
                text: remaining_text,
            };
            match delta_writer.write(&td) {
                Ok(json) => yield Ok(Event::default().event("content_block_delta").data(json)),
                Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
            }
        }

        let cache_read_input_tokens =
            timing.cached_tokens.and_then(|cached| u32::try_from(cached).ok());
        if let Some(ref m) = metrics {
            if let Some(id) = metrics_id {
                m.finalize_stream(id, u64::from(total_output_tokens), start.elapsed(), timing);
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

        // 4b. tool_use blocks follow the text block.
        for (offset, call) in tool_calls.iter().enumerate() {
            let index = u32::try_from(offset + 1).unwrap_or(u32::MAX);
            for (event_name, data) in tool_use_stream_events(index, call) {
                yield Ok(Event::default().event(event_name).data(data));
            }
        }
        if !tool_calls.is_empty() {
            final_stop_reason = Some("tool_use".to_owned());
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
                cache_read_input_tokens,
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
                .prepare_chat_prompt_with_thinking(&engine_messages, tools, thinking_enabled)
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
        let calls = engine.stub_calls();
        assert_eq!(calls[0].thinking_budget, Some(2048));
        assert!(calls[0].thinking_enabled);
    }

    #[tokio::test]
    async fn absent_thinking_leaves_the_budget_to_configuration() {
        let (app, engine) = app_with_stub(thinking_stub());

        let (status, _text) = post_json(&app, "/v1/messages", &body(false)).await;

        assert_eq!(status, http::StatusCode::OK);
        assert_eq!(engine.stub_calls()[0].thinking_budget, None);
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
        assert!(engine.stub_calls().is_empty());
    }
}
