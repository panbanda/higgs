use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

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

use crate::{
    config::ApiFormat,
    constraint::ConstraintSpec,
    error::ServerError,
    metrics::{MetricsStore, RequestMetricsContext, RequestRecord, RequestTiming},
    router::ResolvedRoute,
    state::{EngineLease, SharedState},
    tool_choice::ToolChoice,
    types::openai::{
        ChatCompletionChoice, ChatCompletionDelta, ChatCompletionMessage, ChatCompletionRequest,
        ChatCompletionResponse, ChoiceLogprobs, MessageContent, StopSequence, TokenLogprob,
        ToolCall, ToolCallDelta, ToolCallFunction, ToolCallFunctionDelta, TopLogprob,
    },
    usage::{TokenCounts, build_usage, count_text_tokens},
};
use higgs_engine::engine::{GenerationStats, ThinkingMode};
use higgs_models::SamplingParams;

#[allow(clippy::too_many_lines)]
pub async fn chat_completions(
    State(state): State<SharedState>,
    Extension(request_metrics): Extension<RequestMetricsContext>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ServerError> {
    // Captured before parsing and prompt preparation so TTFT measures
    // the client-observed wait, not only generation.
    let received_at = Instant::now();
    let mut req: ChatCompletionRequest = serde_json::from_slice(&body)
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
            if ToolChoice::from_request(req.tool_choice.as_ref())? == ToolChoice::Disabled {
                req.tools = None;
            }
            let constraint_spec = ConstraintSpec::from_request(
                req.response_format.as_ref(),
                req.guided_regex.as_deref(),
                req.guided_json.as_ref(),
            )?;
            if req.stream == Some(true) {
                let stream = chat_completions_stream(
                    Arc::clone(&state),
                    req,
                    engine,
                    constraint_spec,
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
                let start = Instant::now();
                let response = crate::tool_loop::complete(
                    Arc::clone(&state),
                    req,
                    engine,
                    constraint_spec,
                    received_at,
                )
                .await?;
                if let Some(ref metrics) = state.metrics {
                    metrics.record(RequestRecord {
                        id: 0,
                        timestamp: Instant::now(),
                        wallclock: chrono::Utc::now(),
                        model: Some(response.model.clone()),
                        provider: Some("higgs".to_owned()),
                        routing_method: routing_method.into(),
                        status: 200,
                        duration: start.elapsed(),
                        input_tokens: u64::from(response.usage.prompt_tokens),
                        output_tokens: u64::from(response.usage.completion_tokens),
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
            let metrics_model = model_rewrite.as_deref().unwrap_or(&req.model).to_owned();
            let is_streaming = req.stream == Some(true);
            match provider_format {
                ApiFormat::OpenAi => {
                    let proxy_body = if let Some(ref rewrite) = model_rewrite {
                        crate::proxy::rewrite_model_in_body(&body, rewrite)?
                    } else {
                        body
                    };
                    let start = Instant::now();
                    let result = crate::proxy::proxy_request(
                        &state.http_client,
                        &provider_url,
                        "/v1/chat/completions",
                        proxy_body,
                        &headers,
                        strip_auth,
                        api_key.as_deref(),
                    )
                    .await;
                    if let Some(ref metrics) = state.metrics {
                        metrics.record(RequestRecord {
                            id: 0,
                            timestamp: Instant::now(),
                            wallclock: chrono::Utc::now(),
                            model: Some(metrics_model.clone()),
                            provider: Some(provider_name.clone()),
                            routing_method: routing_method.into(),
                            status: result.as_ref().map_or(502, |resp| resp.status().as_u16()),
                            duration: start.elapsed(),
                            input_tokens: 0,
                            output_tokens: 0,
                            error_body: None,
                            timing: crate::metrics::RequestTiming::default(),
                        });
                        request_metrics.mark_recorded();
                    }
                    result
                }
                ApiFormat::Anthropic => {
                    let translated = crate::translate::openai_to_anthropic_request(
                        &body,
                        state.config.server.max_tokens,
                    )?;
                    let proxy_body = if let Some(ref rewrite) = model_rewrite {
                        crate::proxy::rewrite_model_in_body(&translated, rewrite)?
                    } else {
                        translated
                    };

                    let start = Instant::now();
                    let upstream = crate::proxy::send_to_provider(
                        &state.http_client,
                        &provider_url,
                        "/v1/messages",
                        proxy_body,
                        &headers,
                        strip_auth,
                        api_key.as_deref(),
                    )
                    .await?;
                    let upstream_status = upstream.status().as_u16();

                    if is_streaming {
                        if let Some(ref metrics) = state.metrics {
                            metrics.record(RequestRecord {
                                id: 0,
                                timestamp: Instant::now(),
                                wallclock: chrono::Utc::now(),
                                model: Some(metrics_model.clone()),
                                provider: Some(provider_name.clone()),
                                routing_method: routing_method.into(),
                                status: upstream_status,
                                duration: start.elapsed(),
                                input_tokens: 0,
                                output_tokens: 0,
                                error_body: None,
                                timing: crate::metrics::RequestTiming::default(),
                            });
                            request_metrics.mark_recorded();
                        }
                        if upstream_status >= 400 {
                            let status_code = axum::http::StatusCode::from_u16(upstream_status)
                                .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
                            let resp_bytes = upstream.bytes().await.map_err(|e| {
                                ServerError::ProxyError(format!("Failed to read response: {e}"))
                            })?;
                            return Ok((
                                status_code,
                                [(axum::http::header::CONTENT_TYPE, "application/json")],
                                resp_bytes,
                            )
                                .into_response());
                        }
                        let stream =
                            crate::translate::anthropic_stream_to_openai(upstream, req.model);
                        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                        Ok(sse.into_response())
                    } else {
                        let resp_bytes = upstream.bytes().await.map_err(|e| {
                            ServerError::ProxyError(format!("Failed to read response: {e}"))
                        })?;
                        let usage = if (200..300).contains(&upstream_status) {
                            crate::proxy::extract_usage(&resp_bytes)
                        } else {
                            (0, 0)
                        };
                        if let Some(ref metrics) = state.metrics {
                            metrics.record(RequestRecord {
                                id: 0,
                                timestamp: Instant::now(),
                                wallclock: chrono::Utc::now(),
                                model: Some(metrics_model.clone()),
                                provider: Some(provider_name.clone()),
                                routing_method: routing_method.into(),
                                status: upstream_status,
                                duration: start.elapsed(),
                                input_tokens: usage.0,
                                output_tokens: usage.1,
                                error_body: None,
                                timing: crate::metrics::RequestTiming::default(),
                            });
                            request_metrics.mark_recorded();
                        }
                        let status_code = axum::http::StatusCode::from_u16(upstream_status)
                            .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
                        if upstream_status >= 400 {
                            Ok((
                                status_code,
                                [(axum::http::header::CONTENT_TYPE, "application/json")],
                                resp_bytes,
                            )
                                .into_response())
                        } else {
                            let translated_resp = crate::translate::anthropic_response_to_openai(
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
                }
            }
        }
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) async fn chat_completions_non_streaming(
    state: SharedState,
    req: ChatCompletionRequest,
    engine: EngineLease,
    constraint_spec: Option<ConstraintSpec>,
    received_at: Instant,
) -> Result<ChatCompletionResponse, ServerError> {
    let max_tokens = req.max_tokens.unwrap_or(state.config.server.max_tokens);
    let sampling = build_sampling_params(&req);
    let stop_sequences = StopSequence::extract(req.stop);
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs = req.top_logprobs;

    // Extract images and inject <image> placeholders for VLMs
    let images = extract_images(&req.messages);
    let effective_messages = if images.is_empty() {
        req.messages.clone()
    } else {
        inject_image_placeholders(&req.messages)
    };

    let messages = convert_messages(&effective_messages);
    // Treat an empty `tools: []` as absent (mirrors the streaming path) so it
    // doesn't define `tools` in the template context or trigger tool parsing.
    let tools = req.tools.as_deref().filter(|t| !t.is_empty());
    let thinking_budget = crate::reasoning::requested_thinking_budget(req.reasoning.as_ref())?;
    let thinking_enabled = crate::reasoning::effective_thinking_enabled(
        engine.enable_thinking(),
        &[engine.model_name(), req.model.as_str()],
        req.reasoning.as_ref(),
    );

    let mut prompt_tokens = engine
        .prepare_chat_prompt_with_thinking(&messages, tools, thinking_enabled)
        .map_err(ServerError::Engine)?;

    // Preprocess images for VLM
    let pixel_values = if !images.is_empty() && engine.is_vlm() {
        engine.replace_image_tokens(&mut prompt_tokens);
        let image_size = engine.vlm_image_size().unwrap_or(384);
        #[allow(clippy::as_conversions, clippy::cast_sign_loss)]
        let size = image_size as u32;
        let first_image = images
            .into_iter()
            .next()
            .ok_or_else(|| ServerError::BadRequest("Image data is empty".to_owned()))?;
        let pv = higgs_models::siglip::preprocess_image(&first_image, size)
            .map_err(|e| ServerError::InternalError(format!("Image preprocessing failed: {e}")))?;
        Some(pv)
    } else {
        None
    };

    let constraint = build_constraint(constraint_spec.as_ref(), &engine)?;

    let tokenizer = engine.tokenizer().clone();
    let output = tokio::task::spawn_blocking(move || {
        engine.generate_with_thinking(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            want_logprobs,
            top_logprobs,
            ThinkingMode::new(thinking_enabled, thinking_budget),
            constraint,
            pixel_values,
        )
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
    .map_err(ServerError::Engine)?;

    let request_id = generate_request_id();
    let has_tools = tools.is_some();

    let logprobs_response = output
        .token_logprobs
        .as_ref()
        .map(|lps| logprobs_to_response(lps, &tokenizer));

    let output_text = output.text;
    // Parse reasoning (think tags) from the output.
    // When thinking mode is enabled, the template already opened `<think>` in the prompt,
    // so the generated text starts inside the think block. Prepend `<think>` so the parser
    // can find the matching `</think>` and split reasoning from visible content.
    let (raw_text, reasoning_content) = if thinking_enabled {
        let parse_input = if output_text.contains("</think>") {
            format!("<think>{output_text}")
        } else {
            // Model was length-stopped mid-thinking — close the tag so the
            // parser can extract reasoning instead of leaking raw `<think>`.
            format!("<think>{output_text}</think>")
        };
        let reasoning_result = higgs_engine::reasoning_parser::parse_reasoning(&parse_input);
        let raw_text = if reasoning_result.reasoning.is_some() {
            reasoning_result.text
        } else {
            output_text
        };
        (raw_text, reasoning_result.reasoning)
    } else {
        (output_text, None)
    };

    let (content, tool_calls, finish_reason) = if has_tools {
        let schema = higgs_engine::tool_parser::ToolSchema::from_tools(tools);
        let parsed = higgs_engine::tool_parser::parse_tool_calls(&raw_text, schema.as_ref());
        if parsed.tool_calls.is_empty() {
            (
                Some(MessageContent::Text(raw_text)),
                None,
                output.finish_reason,
            )
        } else {
            let calls: Vec<ToolCall> = parsed
                .tool_calls
                .iter()
                .enumerate()
                .map(|(i, tc)| ToolCall {
                    id: format!("call_{i}_{}", uuid::Uuid::new_v4()),
                    r#type: "function".to_owned(),
                    function: ToolCallFunction {
                        name: tc.name.clone(),
                        arguments: tc.arguments.to_string(),
                    },
                })
                .collect();
            let text = if parsed.text.is_empty() {
                None
            } else {
                Some(MessageContent::Text(parsed.text))
            };
            (text, Some(calls), "tool_calls".to_owned())
        }
    } else {
        (
            Some(MessageContent::Text(raw_text)),
            None,
            output.finish_reason,
        )
    };

    let reasoning_tokens = reasoning_content
        .as_deref()
        .and_then(|text| count_text_tokens(&tokenizer, text));

    Ok(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created: current_unix_timestamp(),
        model: req.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_owned(),
                content,
                reasoning_content,
                tool_calls,
                tool_call_id: None,
            },
            finish_reason,
            logprobs: logprobs_response,
        }],
        usage: build_usage(
            TokenCounts {
                prompt: output.prompt_tokens,
                completion: output.completion_tokens,
                reasoning: reasoning_tokens,
            },
            &output.stats,
            received_at,
        ),
        mcp: None,
    })
}

fn chat_completions_stream(
    state: SharedState,
    req: ChatCompletionRequest,
    engine: EngineLease,
    constraint_spec: Option<ConstraintSpec>,
    metrics: Option<Arc<MetricsStore>>,
    routing_method: crate::router::RoutingMethod,
    received_at: Instant,
) -> Result<impl Stream<Item = Result<Event, Infallible>>, ServerError> {
    let chunks = crate::tool_loop::chunk_stream(
        state,
        req,
        engine,
        constraint_spec,
        metrics,
        routing_method,
        received_at,
    )?;
    Ok(tokio_stream::StreamExt::map(chunks, |data| {
        Ok(Event::default().data(data))
    }))
}

/// Streaming generation as raw SSE `data` payloads (JSON chunks, ending with
/// the `[DONE]` sentinel), shared by every route that re-frames chat chunks.
#[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
pub(crate) fn chat_completions_chunk_stream(
    state: SharedState,
    req: ChatCompletionRequest,
    engine: EngineLease,
    constraint_spec: Option<ConstraintSpec>,
    metrics: Option<Arc<MetricsStore>>,
    routing_method: crate::router::RoutingMethod,
    received_at: Instant,
) -> Result<impl Stream<Item = String>, ServerError> {
    let stream_includes_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());
    // Built here (before the `async_stream::stream!` block, which captures by
    // move) so the tracker can coerce XML-format tool-call values to their
    // declared JSON types.
    let tool_schema = higgs_engine::tool_parser::ToolSchema::from_tools(req.tools.as_deref());

    if stream_includes_tools {
        tracing::debug!(
            request_model = req.model,
            tool_count = req.tools.as_ref().map_or(0, Vec::len),
            "Streaming with tool-calls enabled; will emit tool_calls deltas via StreamingToolCallTracker",
        );
    }

    let max_tokens = req.max_tokens.unwrap_or(state.config.server.max_tokens);
    let sampling = build_sampling_params(&req);
    let stop_sequences = StopSequence::extract(req.stop);
    let want_logprobs = req.logprobs.unwrap_or(false);
    let top_logprobs = req.top_logprobs;

    // Extract images and inject <image> placeholders for VLMs
    let images = extract_images(&req.messages);
    let effective_messages = if images.is_empty() {
        req.messages.clone()
    } else {
        inject_image_placeholders(&req.messages)
    };

    let messages = convert_messages(&effective_messages);
    let thinking_budget = crate::reasoning::requested_thinking_budget(req.reasoning.as_ref())?;
    let thinking_enabled_stream = crate::reasoning::effective_thinking_enabled(
        engine.enable_thinking(),
        &[engine.model_name(), req.model.as_str()],
        req.reasoning.as_ref(),
    );

    // Pass tools into prompt rendering so the chat template emits the
    // tool spec the model recognises. The on-the-fly
    // [`StreamingToolCallTracker`] below intercepts `<tool_call>…
    // </tool_call>` blocks the model produces and turns them into
    // structured `ToolCallDelta` SSE events.
    let prompt_tools = req.tools.as_deref().filter(|t| !t.is_empty());
    let mut prompt_tokens = engine
        .prepare_chat_prompt_with_thinking(&messages, prompt_tools, thinking_enabled_stream)
        .map_err(ServerError::Engine)?;

    // Preprocess images for VLM
    let pixel_values = if !images.is_empty() && engine.is_vlm() {
        engine.replace_image_tokens(&mut prompt_tokens);
        let image_size = engine.vlm_image_size().unwrap_or(384);
        #[allow(clippy::as_conversions, clippy::cast_sign_loss)]
        let size = image_size as u32;
        let first_image = images
            .into_iter()
            .next()
            .ok_or_else(|| ServerError::BadRequest("Image data is empty".to_owned()))?;
        let pv = higgs_models::siglip::preprocess_image(&first_image, size)
            .map_err(|e| ServerError::InternalError(format!("Image preprocessing failed: {e}")))?;
        Some(pv)
    } else {
        None
    };

    let constraint = build_constraint(constraint_spec.as_ref(), &engine)?;

    let request_id = generate_request_id();
    let include_usage = req
        .stream_options
        .as_ref()
        .is_some_and(|opts| opts.include_usage.unwrap_or(false));
    let return_progress = req.return_progress.unwrap_or(false);
    let created = current_unix_timestamp();
    let model = req.model;
    let prompt_token_count = u32::try_from(prompt_tokens.len()).unwrap_or(0);

    let start = received_at;
    let metrics_id = metrics.as_ref().map(|m| {
        m.record_pending(RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: chrono::Utc::now(),
            model: Some(model.clone()),
            provider: Some("higgs".to_owned()),
            routing_method: routing_method.into(),
            status: 200,
            duration: Duration::ZERO,
            input_tokens: u64::from(prompt_token_count),
            output_tokens: 0,
            error_body: None,
            timing: crate::metrics::RequestTiming::default(),
        })
    });
    let tokenizer = engine.tokenizer().clone();
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    tokio::task::spawn_blocking(move || {
        let result = engine.generate_streaming_with_thinking(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            want_logprobs,
            top_logprobs,
            &tx,
            ThinkingMode::new(thinking_enabled_stream, thinking_budget),
            // Always requested so the metrics record sees prefix-cache hits;
            // progress chunks are only forwarded when the client opted in.
            true,
            constraint,
            pixel_values,
        );
        if let Err(e) = result {
            tracing::error!(error = %e, "Generation error during streaming");
        }
    });

    let stream = async_stream::stream! {
        let mut writer = crate::sse::ChatChunkWriter::new(&request_id, created, &model);

        // Helper to emit a chunk carrying a delta.
        macro_rules! emit_delta {
            ($delta:expr, $finish:expr, $logprobs:expr) => {
                match writer.write_delta($delta, $finish, $logprobs) {
                    Ok(json) => yield json.to_owned(),
                    Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
                }
            };
        }

        // Send initial role chunk
        let role_delta = ChatCompletionDelta {
            role: Some("assistant".to_owned()),
            content: None,
            reasoning_content: None,
            tool_calls: None,
        };
        emit_delta!(&role_delta, None, None);

        let mut reasoning_tracker = if thinking_enabled_stream {
            higgs_engine::reasoning_parser::StreamingReasoningTracker::new_inside_think()
        } else {
            higgs_engine::reasoning_parser::StreamingReasoningTracker::new()
        };
        // Streaming tool-call extractor — passthrough when no tools were
        // requested, otherwise watches for `<tool_call>…</tool_call>`
        // blocks and emits structured `ToolCallDelta` events.
        let mut tool_tracker = higgs_engine::tool_parser::StreamingToolCallTracker::new(
            stream_includes_tools,
            tool_schema,
        );

        // Closure that turns a `ParsedToolCall` into the OpenAI streaming
        // delta shape. Index is the running zero-based position of the
        // call in this response.
        let make_tool_delta = |index: u32, parsed: &higgs_engine::tool_parser::ParsedToolCall| {
            ToolCallDelta {
                index,
                id: Some(format!("call_{index}_{}", uuid::Uuid::new_v4())),
                r#type: Some("function".to_owned()),
                function: Some(ToolCallFunctionDelta {
                    name: Some(parsed.name.clone()),
                    arguments: Some(parsed.arguments.to_string()),
                }),
            }
        };

        let mut output_token_count: u32 = 0;
        let mut pending_finish_reason: Option<String> = None;
        let mut pending_finish_logprobs: Option<ChoiceLogprobs> = None;
        let mut timing = RequestTiming::default();
        let mut stats = GenerationStats::default();
        let mut reasoning_text = String::new();

        while let Some(output) = rx.recv().await {
            stats.observe(&output, Instant::now());
            // Prefill-progress events carry no tokens: forward as
            // `prompt_progress` chunks when the client opted in, and keep
            // them away from the delta/tool trackers either way.
            if let Some(p) = output.prefill_progress {
                timing.cached_tokens = Some(u64::from(p.cached));
                if return_progress {
                    let time_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
                    let json = writer.write_prompt_progress(p.total, p.cached, p.processed, time_ms);
                    yield json.to_owned();
                }
                continue;
            }
            if timing.ttft_ms.is_none() {
                timing.ttft_ms = Some(u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX));
            }
            output_token_count = output.completion_tokens;
            let chunk_logprobs = output
                .token_logprob
                .as_ref()
                .map(|lp| logprobs_to_response(std::slice::from_ref(lp), &tokenizer));

            let (visible, reasoning) = reasoning_tracker.process(&output.new_text);

            if !reasoning.is_empty() {
                if include_usage {
                    reasoning_text.push_str(&reasoning);
                }
                let d = ChatCompletionDelta {
                    role: None,
                    content: None,
                    reasoning_content: Some(reasoning),
                    tool_calls: None,
                };
                emit_delta!(&d, None, None);
            }

            // Run the visible-text portion through the tool-call tracker
            // so `<tool_call>…</tool_call>` blocks become structured
            // deltas rather than being spoken aloud as plain text.
            let tool_out = tool_tracker.process(&visible);
            let visible_is_empty = tool_out.visible.is_empty();

            // Tool-call indices count up across the whole response. Each
            // chunk that closes N tool calls covers indices
            // `[base_index .. base_index+N)` where `base_index` is the
            // total completed *before* this chunk.
            let base_index = tool_tracker
                .completed_count()
                .saturating_sub(tool_out.new_tool_calls.len());
            for (i, parsed) in tool_out.new_tool_calls.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                let idx = u32::try_from(base_index + i).unwrap_or(u32::MAX);
                let d = ChatCompletionDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![make_tool_delta(idx, parsed)]),
                };
                emit_delta!(&d, None, None);
            }

            if !tool_out.visible.is_empty() {
                let d = ChatCompletionDelta {
                    role: None,
                    content: Some(tool_out.visible),
                    reasoning_content: None,
                    tool_calls: None,
                };
                emit_delta!(&d, None, chunk_logprobs.as_ref());
            }

            if let Some(finish_reason) = output.finish_reason {
                pending_finish_reason = Some(finish_reason);
                pending_finish_logprobs = if visible_is_empty { chunk_logprobs } else { None };
            }
        }

        // Flush any remaining buffered content.
        let (flush_vis, flush_reas) = reasoning_tracker.flush();
        if !flush_reas.is_empty() {
            if include_usage {
                reasoning_text.push_str(&flush_reas);
            }
            let d = ChatCompletionDelta {
                role: None,
                content: None,
                reasoning_content: Some(flush_reas),
                tool_calls: None,
            };
            emit_delta!(&d, None, None);
        }
        // Drain the tool tracker (handles unclosed `<tool_call>` tags by
        // re-emitting their buffered prefix as visible content — never
        // silently drop tokens).
        let flush_tool_out = tool_tracker.process(&flush_vis);
        let flush_base_index = tool_tracker
            .completed_count()
            .saturating_sub(flush_tool_out.new_tool_calls.len());
        for (i, parsed) in flush_tool_out.new_tool_calls.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let idx = u32::try_from(flush_base_index + i).unwrap_or(u32::MAX);
            let d = ChatCompletionDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![make_tool_delta(idx, parsed)]),
            };
            emit_delta!(&d, None, None);
        }
        if !flush_tool_out.visible.is_empty() {
            let d = ChatCompletionDelta {
                role: None,
                content: Some(flush_tool_out.visible),
                reasoning_content: None,
                tool_calls: None,
            };
            emit_delta!(&d, None, None);
        }
        let final_tool_out = tool_tracker.flush();
        if !final_tool_out.visible.is_empty() {
            let d = ChatCompletionDelta {
                role: None,
                content: Some(final_tool_out.visible),
                reasoning_content: None,
                tool_calls: None,
            };
            emit_delta!(&d, None, None);
        }

        // Defer `finish_reason` until after the tracker has drained so we
        // know whether to report `"tool_calls"` or `"stop"`.
        if let Some(finish_reason) = pending_finish_reason {
            let effective_finish = if tool_tracker.has_tool_calls() {
                "tool_calls".to_owned()
            } else {
                finish_reason
            };
            let d = ChatCompletionDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: None,
            };
            emit_delta!(&d, Some(effective_finish.as_str()), pending_finish_logprobs.as_ref());
        }

        // Emit final chunk with usage only when explicitly requested.
        if include_usage {
            let usage = build_usage(
                TokenCounts {
                    prompt: prompt_token_count,
                    completion: output_token_count,
                    reasoning: count_text_tokens(&tokenizer, &reasoning_text),
                },
                &stats,
                received_at,
            );
            match writer.write_usage(&usage) {
                Ok(json) => yield json.to_owned(),
                Err(e) => tracing::error!(error = %e, "Failed to serialize usage chunk"),
            }
        }

        if let Some(ref m) = metrics {
            if let Some(id) = metrics_id {
                m.finalize_stream(id, u64::from(output_token_count), start.elapsed(), timing);
            }
        }

        // Send [DONE] sentinel
        yield "[DONE]".to_owned();
    };

    Ok(stream)
}

pub(crate) fn convert_messages(
    messages: &[ChatCompletionMessage],
) -> Vec<higgs_engine::chat_template::ChatMessage> {
    messages
        .iter()
        .map(|m| {
            let tool_calls_json = m.tool_calls.as_ref().map(|calls| {
                calls
                    .iter()
                    .filter_map(|tc| serde_json::to_value(tc).ok())
                    .map(|mut tc_value| {
                        // Make the tool call template-friendly: hoist
                        // `function.{name,arguments}` to the top level
                        // and parse string-encoded arguments to a JSON
                        // value. Without this, Qwen's chat template
                        // crashes on `tool_call.arguments|items`.
                        higgs_engine::chat_template::normalize_tool_call_for_template(
                            &mut tc_value,
                        );
                        tc_value
                    })
                    .collect()
            });
            let content = m
                .content
                .as_ref()
                .map_or_else(String::new, MessageContent::text);
            higgs_engine::chat_template::ChatMessage {
                role: m.role.clone(),
                content,
                tool_calls: tool_calls_json,
            }
        })
        .collect()
}

/// Extract image bytes from base64 data URIs in message content parts.
/// Returns decoded image bytes for each image found across all messages.
fn extract_images(messages: &[ChatCompletionMessage]) -> Vec<Vec<u8>> {
    use base64::Engine as _;
    let mut images = Vec::new();
    for msg in messages {
        let Some(content) = &msg.content else {
            continue;
        };
        for url in content.image_urls() {
            if let Some(data) = url.strip_prefix("data:") {
                // data:[<mediatype>];base64,<data>
                if let Some(base64_start) = data.find(";base64,") {
                    let encoded = &data[base64_start + 8..];
                    match base64::engine::general_purpose::STANDARD.decode(encoded) {
                        Ok(bytes) => images.push(bytes),
                        Err(e) => tracing::warn!(error = %e, "Failed to decode base64 image"),
                    }
                }
            }
            // HTTP/HTTPS URLs are not supported yet; could be fetched in the future
        }
    }
    images
}

/// Build text content with `<image>` placeholders injected for each image.
/// For VLMs, each image in a message gets a `<image>\n` prefix before the text.
fn inject_image_placeholders(messages: &[ChatCompletionMessage]) -> Vec<ChatCompletionMessage> {
    messages
        .iter()
        .map(|m| {
            let Some(content) = &m.content else {
                return m.clone();
            };
            if !content.has_images() {
                return m.clone();
            }

            let image_count = content.image_urls().len();
            let text = content.text();
            let prefix = "<image>\n".repeat(image_count);
            let combined = format!("{prefix}{text}");

            ChatCompletionMessage {
                role: m.role.clone(),
                content: Some(MessageContent::Text(combined)),
                reasoning_content: m.reasoning_content.clone(),
                tool_calls: m.tool_calls.clone(),
                tool_call_id: m.tool_call_id.clone(),
            }
        })
        .collect()
}

fn build_sampling_params(req: &ChatCompletionRequest) -> SamplingParams {
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

/// Compile the requested constraint against the model's vocabulary.
///
/// A pattern or schema the grammar compiler rejects is the client's fault, so
/// those failures are reported as `400`; failing to build the vocabulary is a
/// server error.
fn build_constraint(
    requested: Option<&ConstraintSpec>,
    engine: &crate::state::Engine,
) -> Result<Option<higgs_engine::constrained::ConstrainedGenerator>, ServerError> {
    use higgs_engine::constrained::ConstrainedGenerator;

    let Some(spec) = requested else {
        return Ok(None);
    };
    let eos_id = engine.eos_token_ids().first().copied().unwrap_or(0);
    let vocab = higgs_engine::constrained::build_vocabulary(engine.tokenizer(), eos_id)
        .map_err(ServerError::Engine)?;
    let constraint = match spec {
        ConstraintSpec::JsonObject => ConstrainedGenerator::for_json_object(&vocab),
        ConstraintSpec::JsonSchema(schema) => {
            ConstrainedGenerator::from_json_schema(&schema.to_string(), &vocab)
        }
        ConstraintSpec::Regex(pattern) => ConstrainedGenerator::from_regex(pattern, &vocab),
    }
    .map_err(|error| ServerError::BadRequest(error.to_string()))?;
    Ok(Some(constraint))
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

fn generate_request_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4())
}

fn current_unix_timestamp() -> i64 {
    chrono::Utc::now().timestamp()
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;

    fn simple_message(role: &str, content: Option<&str>) -> ChatCompletionMessage {
        ChatCompletionMessage {
            role: role.to_owned(),
            content: content.map(|s| MessageContent::Text(s.to_owned())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_owned(),
            r#type: "function".to_owned(),
            function: ToolCallFunction {
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            },
        }
    }

    fn tool_message(role: &str, calls: Vec<ToolCall>) -> ChatCompletionMessage {
        ChatCompletionMessage {
            role: role.to_owned(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(calls),
            tool_call_id: None,
        }
    }

    #[test]
    fn test_convert_messages() {
        let msgs = vec![
            simple_message("user", Some("Hello")),
            simple_message("assistant", None),
        ];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted.first().map(|m| m.role.as_str()), Some("user"));
        assert_eq!(converted.first().map(|m| m.content.as_str()), Some("Hello"));
        assert_eq!(converted.get(1).map(|m| m.content.as_str()), Some(""));
    }

    #[test]
    fn test_generate_request_id_format() {
        let id = generate_request_id();
        assert!(id.starts_with("chatcmpl-"));
        assert!(id.len() > "chatcmpl-".len());
    }

    #[test]
    fn test_convert_messages_with_tool_calls() {
        let msgs = vec![tool_message(
            "assistant",
            vec![tool_call("call_1", "get_weather", r#"{"city":"NYC"}"#)],
        )];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        let calls = converted
            .first()
            .and_then(|m| m.tool_calls.as_ref())
            .unwrap();
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn test_convert_messages_empty_list() {
        let result = convert_messages(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_convert_messages_with_null_content() {
        let msgs = vec![simple_message("assistant", None)];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted.first().map(|m| m.content.as_str()), Some(""));
    }

    #[test]
    fn test_convert_messages_with_tool_calls_complex_arguments() {
        let msgs = vec![tool_message(
            "assistant",
            vec![
                tool_call(
                    "call_1",
                    "search",
                    r#"{"query":"rust programming","filters":{"language":"en","year":2024}}"#,
                ),
                tool_call("call_2", "calculate", r#"{"expression":"2+2"}"#),
            ],
        )];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        let calls = converted
            .first()
            .and_then(|m| m.tool_calls.as_ref())
            .unwrap();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_generate_request_id_uniqueness() {
        let mut ids = std::collections::HashSet::new();
        for _ in 0..100 {
            let id = generate_request_id();
            assert!(ids.insert(id), "duplicate request ID generated");
        }
        assert_eq!(ids.len(), 100);
    }

    #[test]
    fn test_current_unix_timestamp_reasonable_value() {
        let ts = current_unix_timestamp();
        assert!(ts > 1_700_000_000, "timestamp too old: {ts}");
    }

    mod route {
        use crate::test_support::{
            StubEngine, app_with_stub, generation_output, post_json, progress_event,
            sse_json_events, token_event, word_tokenizer,
        };
        use higgs_engine::engine::GenerationStats;

        fn stub_with_tokenizer() -> StubEngine {
            let mut stub = StubEngine::named("stub");
            stub.tokenizer = Some(word_tokenizer());
            stub.script.prompt_tokens = vec![1, 2, 1, 2, 1];
            stub.script.stamp_times = true;
            stub
        }

        fn chat_body(extra: &serde_json::Value) -> serde_json::Value {
            let mut body = serde_json::json!({
                "model": "stub",
                "messages": [{"role": "user", "content": "hello"}],
            });
            if let (Some(target), Some(fields)) = (body.as_object_mut(), extra.as_object()) {
                target.extend(fields.clone());
            }
            body
        }

        #[tokio::test]
        async fn non_streaming_usage_reports_cache_hit_and_timings() {
            let mut stub = stub_with_tokenizer();
            let mut output = generation_output("hello world", 5, 4);
            output.stats = GenerationStats {
                cached_tokens: Some(3),
                ..GenerationStats::default()
            };
            stub.script.generation = Some(output);
            let (app, _engine) = app_with_stub(stub);

            let (status, body) = post_json(
                &app,
                "/v1/chat/completions",
                &chat_body(&serde_json::json!({})),
            )
            .await;

            assert_eq!(status, http::StatusCode::OK, "{body}");
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            let usage = &json["usage"];
            assert_eq!(usage["prompt_tokens"], 5);
            assert_eq!(usage["completion_tokens"], 4);
            assert_eq!(usage["total_tokens"], 9);
            assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 3);
            for field in [
                "time_to_first_token",
                "prompt_eval_duration",
                "generation_duration",
                "total_time",
            ] {
                assert!(usage[field].is_f64(), "{field} missing in {usage}");
            }
            assert!(usage.get("completion_tokens_details").is_none());
        }

        #[tokio::test]
        async fn non_streaming_usage_omits_unmeasured_fields() {
            let mut stub = stub_with_tokenizer();
            stub.script.stamp_times = false;
            stub.script.generation = Some(generation_output("hello", 5, 2));
            let (app, _engine) = app_with_stub(stub);

            let (status, body) = post_json(
                &app,
                "/v1/chat/completions",
                &chat_body(&serde_json::json!({})),
            )
            .await;

            assert_eq!(status, http::StatusCode::OK, "{body}");
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                json["usage"],
                serde_json::json!({"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
            );
        }

        #[tokio::test]
        async fn non_streaming_usage_counts_reasoning_tokens() {
            let mut stub = stub_with_tokenizer();
            stub.script.enable_thinking = true;
            stub.script.generation = Some(generation_output(
                "think again hello</think>hello world",
                5,
                8,
            ));
            let (app, _engine) = app_with_stub(stub);

            let (status, body) = post_json(
                &app,
                "/v1/chat/completions",
                &chat_body(&serde_json::json!({})),
            )
            .await;

            assert_eq!(status, http::StatusCode::OK, "{body}");
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(
                json["choices"][0]["message"]["reasoning_content"],
                "think again hello"
            );
            assert_eq!(
                json["usage"]["completion_tokens_details"]["reasoning_tokens"],
                3
            );
        }

        #[tokio::test]
        async fn streaming_usage_chunk_carries_details_and_timings() {
            let mut stub = stub_with_tokenizer();
            stub.script.stream = vec![
                progress_event(2, 5),
                token_event("hello", 1, None),
                token_event(" world", 2, Some("stop")),
            ];
            let (app, _engine) = app_with_stub(stub);
            let request = chat_body(&serde_json::json!({
                "stream": true,
                "stream_options": {"include_usage": true},
            }));

            let (status, body) = post_json(&app, "/v1/chat/completions", &request).await;

            assert_eq!(status, http::StatusCode::OK, "{body}");
            let events = sse_json_events(&body);
            let usage_chunk = events.last().unwrap();
            assert_eq!(usage_chunk["choices"], serde_json::json!([]));
            let usage = &usage_chunk["usage"];
            assert_eq!(usage["prompt_tokens"], 5);
            assert_eq!(usage["completion_tokens"], 2);
            assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 2);
            for field in [
                "time_to_first_token",
                "prompt_eval_duration",
                "generation_duration",
                "total_time",
            ] {
                assert!(usage[field].is_f64(), "{field} missing in {usage}");
            }
            assert!(
                !body.contains("prompt_progress"),
                "progress chunks must stay opt-in"
            );
        }

        #[tokio::test]
        async fn streaming_without_include_usage_has_no_usage_chunk() {
            let mut stub = stub_with_tokenizer();
            stub.script.stream = vec![progress_event(0, 5), token_event("hello", 1, Some("stop"))];
            let (app, _engine) = app_with_stub(stub);

            let (status, body) = post_json(
                &app,
                "/v1/chat/completions",
                &chat_body(&serde_json::json!({"stream": true})),
            )
            .await;

            assert_eq!(status, http::StatusCode::OK, "{body}");
            assert!(!body.contains("\"usage\""), "{body}");
        }

        #[tokio::test]
        async fn streaming_usage_counts_reasoning_tokens() {
            let mut stub = stub_with_tokenizer();
            stub.script.enable_thinking = true;
            stub.script.stream = vec![
                progress_event(0, 5),
                token_event("think again", 2, None),
                token_event("</think>", 3, None),
                token_event("hello", 4, Some("stop")),
            ];
            let (app, _engine) = app_with_stub(stub);
            let request = chat_body(&serde_json::json!({
                "stream": true,
                "stream_options": {"include_usage": true},
            }));

            let (_status, body) = post_json(&app, "/v1/chat/completions", &request).await;

            let events = sse_json_events(&body);
            let usage = &events.last().unwrap()["usage"];
            assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], 2);
            assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 0);
        }

        async fn post_chat(
            stub: StubEngine,
            extra: &serde_json::Value,
        ) -> (
            http::StatusCode,
            String,
            std::sync::Arc<crate::state::Engine>,
        ) {
            let (app, engine) = app_with_stub(stub);
            let (status, body) = post_json(&app, "/v1/chat/completions", &chat_body(extra)).await;
            (status, body, engine)
        }

        fn recorded_calls(engine: &crate::state::Engine) -> Vec<crate::test_support::StubCall> {
            engine.stub_calls()
        }

        fn generating_stub() -> StubEngine {
            let mut stub = stub_with_tokenizer();
            stub.script.generation = Some(generation_output("hello", 5, 1));
            stub
        }

        #[tokio::test]
        async fn guided_regex_constrains_generation() {
            let (status, body, engine) = post_chat(
                generating_stub(),
                &serde_json::json!({"guided_regex": "hello|world"}),
            )
            .await;
            assert_eq!(status, http::StatusCode::OK, "{body}");
            let calls = recorded_calls(&engine);
            assert_eq!(calls.len(), 1);
            assert!(calls[0].has_constraint);
        }

        #[tokio::test]
        async fn regex_response_format_constrains_generation() {
            let extra =
                serde_json::json!({"response_format": {"type": "regex", "regex": "hello|world"}});
            let (status, body, engine) = post_chat(generating_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::OK, "{body}");
            assert!(recorded_calls(&engine)[0].has_constraint);
        }

        #[tokio::test]
        async fn guided_regex_constrains_streaming_generation() {
            let mut stub = stub_with_tokenizer();
            stub.script.stream = vec![progress_event(0, 5), token_event("hello", 1, Some("stop"))];
            let extra = serde_json::json!({"stream": true, "guided_regex": "hello|world"});
            let (status, body, engine) = post_chat(stub, &extra).await;
            assert_eq!(status, http::StatusCode::OK, "{body}");
            assert!(recorded_calls(&engine)[0].has_constraint);
        }

        #[tokio::test]
        async fn unconstrained_requests_carry_no_constraint() {
            let (status, _body, engine) =
                post_chat(generating_stub(), &serde_json::json!({})).await;
            assert_eq!(status, http::StatusCode::OK);
            assert!(!recorded_calls(&engine)[0].has_constraint);
        }

        #[tokio::test]
        async fn invalid_regex_is_a_client_error_with_the_compile_error() {
            let (status, body, engine) = post_chat(
                generating_stub(),
                &serde_json::json!({"guided_regex": "(unclosed"}),
            )
            .await;
            assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
            assert!(body.contains("Invalid regex constraint"), "{body}");
            assert!(recorded_calls(&engine).is_empty());
        }

        #[tokio::test]
        async fn invalid_regex_via_response_format_is_a_client_error() {
            let extra = serde_json::json!({"response_format": {"type": "regex", "regex": "[a-"}});
            let (status, body, _engine) = post_chat(generating_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
        }

        #[tokio::test]
        async fn multiple_constraints_are_rejected_before_generation() {
            let extra = serde_json::json!({
                "guided_regex": "hello",
                "response_format": {"type": "json_object"},
            });
            let (status, body, engine) = post_chat(generating_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
            assert!(body.contains("Only one output constraint"), "{body}");
            assert!(recorded_calls(&engine).is_empty());
        }

        #[tokio::test]
        async fn unsatisfiable_schema_is_a_client_error() {
            let extra = serde_json::json!({"guided_json": {"type": "bogus"}});
            let (status, body, _engine) = post_chat(generating_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
        }

        #[tokio::test]
        async fn guided_json_must_be_an_object() {
            let extra = serde_json::json!({"guided_json": "not a schema"});
            let (status, body, _engine) = post_chat(generating_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
        }

        fn thinking_stub() -> StubEngine {
            let mut stub = generating_stub();
            stub.script.enable_thinking = true;
            stub.script.stream = vec![progress_event(0, 5), token_event("hello", 1, Some("stop"))];
            stub
        }

        #[tokio::test]
        async fn thinking_budget_defaults_to_engine_configuration() {
            let (status, _body, engine) = post_chat(thinking_stub(), &serde_json::json!({})).await;
            assert_eq!(status, http::StatusCode::OK);
            let call = &recorded_calls(&engine)[0];
            assert!(call.thinking_enabled);
            assert_eq!(call.thinking_budget, None);
        }

        #[tokio::test]
        async fn request_reasoning_max_tokens_sets_the_budget() {
            let extra = serde_json::json!({"reasoning": {"max_tokens": 64}});
            let (status, body, engine) = post_chat(thinking_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::OK, "{body}");
            assert_eq!(recorded_calls(&engine)[0].thinking_budget, Some(64));
        }

        #[tokio::test]
        async fn request_reasoning_max_tokens_applies_when_streaming() {
            let extra = serde_json::json!({"stream": true, "reasoning": {"max_tokens": 32}});
            let (status, body, engine) = post_chat(thinking_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::OK, "{body}");
            assert_eq!(recorded_calls(&engine)[0].thinking_budget, Some(32));
        }

        #[tokio::test]
        async fn zero_reasoning_max_tokens_is_rejected() {
            let extra = serde_json::json!({"reasoning": {"max_tokens": 0}});
            let (status, body, engine) = post_chat(thinking_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
            assert!(
                body.contains("reasoning.max_tokens must be at least 1"),
                "{body}"
            );
            assert!(recorded_calls(&engine).is_empty());
        }

        #[tokio::test]
        async fn budget_is_inert_when_thinking_is_off() {
            let extra = serde_json::json!({"reasoning": {"effort": "none", "max_tokens": 64}});
            let (status, _body, engine) = post_chat(thinking_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::OK);
            let call = &recorded_calls(&engine)[0];
            assert!(!call.thinking_enabled);
            assert_eq!(call.thinking_budget, None);
        }

        fn tools_json() -> serde_json::Value {
            serde_json::json!([{
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object"}},
            }])
        }

        const TOOL_CALL_TEXT: &str =
            r#"<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>"#;

        fn tool_stub() -> StubEngine {
            let mut stub = stub_with_tokenizer();
            stub.script.generation = Some(generation_output(TOOL_CALL_TEXT, 5, 12));
            stub
        }

        #[tokio::test]
        async fn tool_choice_auto_exposes_and_parses_tools() {
            let extra = serde_json::json!({"tools": tools_json(), "tool_choice": "auto"});
            let (status, body, engine) = post_chat(tool_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::OK, "{body}");
            assert_eq!(engine.stub_prompt_tools(), vec![Some(1)]);
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(json["choices"][0]["finish_reason"], "tool_calls");
            assert_eq!(
                json["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                "get_weather"
            );
        }

        #[tokio::test]
        async fn absent_tool_choice_behaves_like_auto() {
            let extra = serde_json::json!({"tools": tools_json()});
            let (status, _body, engine) = post_chat(tool_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::OK);
            assert_eq!(engine.stub_prompt_tools(), vec![Some(1)]);
        }

        #[tokio::test]
        async fn tool_choice_none_hides_and_does_not_parse_tools() {
            let extra = serde_json::json!({"tools": tools_json(), "tool_choice": "none"});
            let (status, body, engine) = post_chat(tool_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::OK, "{body}");
            assert_eq!(engine.stub_prompt_tools(), vec![None]);
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            let message = &json["choices"][0]["message"];
            assert!(message.get("tool_calls").is_none(), "{body}");
            assert_eq!(message["content"], TOOL_CALL_TEXT);
            assert_eq!(json["choices"][0]["finish_reason"], "stop");
        }

        #[tokio::test]
        async fn tool_choice_none_applies_when_streaming() {
            let mut stub = stub_with_tokenizer();
            stub.script.stream = vec![
                progress_event(0, 5),
                token_event(TOOL_CALL_TEXT, 12, Some("stop")),
            ];
            let extra = serde_json::json!({
                "stream": true,
                "tools": tools_json(),
                "tool_choice": "none",
            });
            let (status, body, engine) = post_chat(stub, &extra).await;
            assert_eq!(status, http::StatusCode::OK, "{body}");
            assert_eq!(engine.stub_prompt_tools(), vec![None]);
            assert!(!body.contains("tool_calls"), "{body}");
            assert!(body.contains("get_weather"), "text stays visible: {body}");
        }

        #[tokio::test]
        async fn tool_choice_required_is_rejected_before_generation() {
            let extra = serde_json::json!({"tools": tools_json(), "tool_choice": "required"});
            let (status, body, engine) = post_chat(tool_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
            let json: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert!(
                json["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("\"required\" is not supported yet"),
                "{body}"
            );
            assert!(engine.stub_calls().is_empty());
        }

        #[tokio::test]
        async fn named_tool_choice_is_rejected_before_generation() {
            let extra = serde_json::json!({
                "tools": tools_json(),
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            });
            let (status, body, engine) = post_chat(tool_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::BAD_REQUEST, "{body}");
            assert!(body.contains("not supported yet"), "{body}");
            assert!(engine.stub_calls().is_empty());
        }

        #[tokio::test]
        async fn seed_is_accepted_and_does_not_change_the_call() {
            let (status, body, engine) =
                post_chat(generating_stub(), &serde_json::json!({"seed": 42})).await;
            assert_eq!(status, http::StatusCode::OK, "{body}");
            assert_eq!(engine.stub_calls().len(), 1);
        }

        #[tokio::test]
        async fn non_integer_seed_is_rejected() {
            let (status, _body, _engine) =
                post_chat(generating_stub(), &serde_json::json!({"seed": "abc"})).await;
            assert_eq!(status, http::StatusCode::BAD_REQUEST);
        }

        #[tokio::test]
        async fn unknown_fields_are_still_ignored() {
            let (status, body, _engine) = post_chat(
                generating_stub(),
                &serde_json::json!({"frobnicate": true, "parallel_tool_calls": false}),
            )
            .await;
            assert_eq!(status, http::StatusCode::OK, "{body}");
        }
    }
}
