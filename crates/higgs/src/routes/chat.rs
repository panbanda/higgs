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
    config::ApiFormat,
    constraint::ConstraintSpec,
    error::ServerError,
    generation::{
        ChatPrompt, Completed, EventStream, GenerationEvent, GenerationRequest, Prompt,
        ToolLoopEvent,
        messages::{prompt_messages, wire_tool_call},
    },
    metrics::{RequestMetricsContext, RequestRecord},
    router::ResolvedRoute,
    state::SharedState,
    tool_policy::ToolPolicy,
    types::openai::{
        ChatCompletionChoice, ChatCompletionDelta, ChatCompletionMessage, ChatCompletionRequest,
        ChatCompletionResponse, MessageContent, StopSequence, ToolCallDelta, ToolCallFunctionDelta,
    },
};
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
            let request = generation_request(&req, state.config.server.max_tokens)?;
            if req.stream == Some(true) {
                let events = crate::generation::stream(
                    Arc::clone(&state),
                    request,
                    engine,
                    state
                        .metrics
                        .clone()
                        .map(|store| crate::generation::StreamMetrics {
                            store,
                            routing_method,
                        }),
                    received_at,
                )?;
                let chunks = chat_chunk_stream(
                    events,
                    req.model,
                    ChunkOptions {
                        include_usage: req
                            .stream_options
                            .as_ref()
                            .is_some_and(|opts| opts.include_usage.unwrap_or(false)),
                        return_progress: req.return_progress.unwrap_or(false),
                        received_at,
                    },
                );
                let sse = Sse::new(tokio_stream::StreamExt::map(chunks, |data| {
                    Ok::<_, Infallible>(Event::default().data(data))
                }))
                .keep_alive(KeepAlive::default());
                if state.metrics.is_some() {
                    request_metrics.mark_recorded();
                }
                Ok(sse.into_response())
            } else {
                let start = Instant::now();
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
                let response = chat_response(completed, req.model);
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
                        api_key.as_ref().map(crate::config::Secret::expose),
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
                        api_key.as_ref().map(crate::config::Secret::expose),
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

/// Translate a Chat Completions request into the shared request.
fn generation_request(
    req: &ChatCompletionRequest,
    default_max_tokens: u32,
) -> Result<GenerationRequest, ServerError> {
    let tool_policy = ToolPolicy::from_chat(req.tool_choice.as_ref(), req.parallel_tool_calls)?;
    let constraint = ConstraintSpec::from_request(
        req.response_format.as_ref(),
        req.guided_regex.as_deref(),
        req.guided_json.as_ref(),
    )?;
    let thinking_budget = crate::reasoning::requested_thinking_budget(req.reasoning.as_ref())?;
    let tools = if tool_policy.exposes_tools() {
        req.tools.clone().unwrap_or_default()
    } else {
        Vec::new()
    };
    let (messages, images) = prompt_messages(&req.messages);
    Ok(GenerationRequest {
        model: req.model.clone(),
        prompt: Prompt::Chat(ChatPrompt {
            messages,
            images,
            tools,
            tool_policy,
            reasoning: req.reasoning.clone(),
            thinking_budget,
            parse_reasoning_without_thinking: false,
        }),
        max_tokens: req.max_tokens.unwrap_or(default_max_tokens),
        sampling: build_sampling_params(req),
        stop_sequences: StopSequence::extract(req.stop.clone()),
        logprobs: req.logprobs.unwrap_or(false),
        top_logprobs: req.top_logprobs,
        constraint,
        mcp: req.mcp.clone(),
    })
}

fn chat_response(completed: Completed, model: String) -> ChatCompletionResponse {
    let content = (completed.tool_calls.is_empty() || !completed.text.is_empty())
        .then_some(MessageContent::Text(completed.text));
    let tool_calls = (!completed.tool_calls.is_empty())
        .then(|| completed.tool_calls.iter().map(wire_tool_call).collect());
    ChatCompletionResponse {
        id: generate_request_id(),
        object: "chat.completion",
        created: current_unix_timestamp(),
        model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_owned(),
                content,
                reasoning_content: completed.reasoning,
                tool_calls,
                tool_call_id: None,
            },
            finish_reason: completed.finish_reason,
            logprobs: completed.logprobs,
        }],
        usage: completed.usage,
        mcp: completed.tool_loop,
    }
}

/// What the client asked of the chunk stream.
struct ChunkOptions {
    include_usage: bool,
    return_progress: bool,
    received_at: Instant,
}

/// Render generation events as chat completion chunks: raw SSE `data`
/// payloads (JSON chunks, ending with the `[DONE]` sentinel).
#[allow(clippy::needless_pass_by_value)]
fn chat_chunk_stream(
    mut events: EventStream,
    model: String,
    options: ChunkOptions,
) -> impl Stream<Item = String> {
    let request_id = generate_request_id();
    let created = current_unix_timestamp();
    async_stream::stream! {
        let mut writer = crate::sse::ChatChunkWriter::new(&request_id, created, &model);

        macro_rules! emit_delta {
            ($delta:expr, $finish:expr, $logprobs:expr) => {
                match writer.write_delta($delta, $finish, $logprobs) {
                    Ok(json) => yield json.to_owned(),
                    Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
                }
            };
        }

        let role_delta = ChatCompletionDelta {
            role: Some("assistant".to_owned()),
            content: None,
            reasoning_content: None,
            tool_calls: None,
        };
        emit_delta!(&role_delta, None, None);

        while let Some(event) = events.next().await {
            match event {
                GenerationEvent::Started { .. } => {}
                GenerationEvent::PrefillProgress(progress) => {
                    if options.return_progress {
                        let time_ms =
                            u64::try_from(options.received_at.elapsed().as_millis()).unwrap_or(u64::MAX);
                        let json = writer.write_prompt_progress(
                            progress.total,
                            progress.cached,
                            progress.processed,
                            time_ms,
                        );
                        yield json.to_owned();
                    }
                }
                GenerationEvent::ReasoningDelta(text) => {
                    let delta = ChatCompletionDelta {
                        role: None,
                        content: None,
                        reasoning_content: Some(text),
                        tool_calls: None,
                    };
                    emit_delta!(&delta, None, None);
                }
                GenerationEvent::TextDelta { text, logprobs } => {
                    let delta = ChatCompletionDelta {
                        role: None,
                        content: Some(text),
                        reasoning_content: None,
                        tool_calls: None,
                    };
                    emit_delta!(&delta, None, logprobs.as_ref());
                }
                GenerationEvent::ToolCall(call) => {
                    let delta = ChatCompletionDelta {
                        role: None,
                        content: None,
                        reasoning_content: None,
                        tool_calls: Some(vec![ToolCallDelta {
                            index: call.index,
                            id: Some(call.id),
                            r#type: Some("function".to_owned()),
                            function: Some(ToolCallFunctionDelta {
                                name: Some(call.name),
                                arguments: Some(call.arguments.to_string()),
                            }),
                        }]),
                    };
                    emit_delta!(&delta, None, None);
                }
                GenerationEvent::Finish { reason, logprobs } => {
                    let delta = ChatCompletionDelta {
                        role: None,
                        content: None,
                        reasoning_content: None,
                        tool_calls: None,
                    };
                    emit_delta!(&delta, Some(reason.as_str()), logprobs.as_ref());
                }
                GenerationEvent::Usage(usage) => {
                    if options.include_usage {
                        match writer.write_usage(&usage) {
                            Ok(json) => yield json.to_owned(),
                            Err(e) => tracing::error!(error = %e, "Failed to serialize usage chunk"),
                        }
                    }
                }
                GenerationEvent::ToolLoop(progress) => {
                    if let Some(chunk) = vendor_chunk(&request_id, created, &model, &progress) {
                        yield chunk;
                    }
                }
                GenerationEvent::Error(message) => {
                    yield serde_json::json!({
                        "error": {"message": message, "type": "server_error"}
                    })
                    .to_string();
                }
            }
        }

        yield "[DONE]".to_owned();
    }
}

/// Key under which tool-loop progress is reported in stream chunks.
const TOOL_LOOP_KEY: &str = "higgs_mcp";

/// A chunk with no choices that announces tool-loop progress. Standard
/// clients ignore it, as they do usage chunks.
fn vendor_chunk(id: &str, created: i64, model: &str, progress: &ToolLoopEvent) -> Option<String> {
    let payload = match progress {
        ToolLoopEvent::Call {
            round,
            id: call_id,
            name,
            arguments,
        } => serde_json::json!({
            "event": "tool_call",
            "round": round,
            "id": call_id,
            "name": name,
            "arguments": arguments,
        }),
        ToolLoopEvent::Result {
            round,
            id: call_id,
            name,
            is_error,
            content,
        } => serde_json::json!({
            "event": "tool_result",
            "round": round,
            "id": call_id,
            "name": name,
            "is_error": is_error,
            "content": content,
        }),
        ToolLoopEvent::Messages(_) => return None,
        ToolLoopEvent::Summary(summary) => {
            let mut payload = summary.clone();
            if let Some(object) = payload.as_object_mut() {
                object.insert("event".to_owned(), serde_json::json!("summary"));
            }
            payload
        }
    };
    Some(
        serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            TOOL_LOOP_KEY: payload,
        })
        .to_string(),
    )
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

    #[test]
    fn test_generate_request_id_format() {
        let id = generate_request_id();
        assert!(id.starts_with("chatcmpl-"));
        assert!(id.len() > "chatcmpl-".len());
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
            sse_json_events, token_event,
        };
        use higgs_engine::engine::GenerationStats;

        fn stub_with_tokenizer() -> StubEngine {
            let mut stub = StubEngine::named("stub");
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
        ) -> (http::StatusCode, String, StubEngine) {
            let (app, engine) = app_with_stub(stub);
            let (status, body) = post_json(&app, "/v1/chat/completions", &chat_body(extra)).await;
            (status, body, engine)
        }

        fn recorded_calls(engine: &StubEngine) -> Vec<crate::test_support::StubCall> {
            engine.recorded_calls()
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

        #[tokio::test]
        async fn a_constrained_request_does_not_think() {
            let extra = serde_json::json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}},
                        "required": ["a"],
                    }},
                },
            });
            let mut stub = thinking_stub();
            stub.tokenizer = crate::test_support::char_tokenizer();
            stub.script.generation = Some(generation_output(r#"{"a":1}"#, 5, 5));
            let (status, body, engine) = post_chat(stub, &extra).await;
            assert_eq!(status, http::StatusCode::OK, "{body}");
            let call = &recorded_calls(&engine)[0];
            assert!(call.has_constraint);
            assert!(!call.thinking_enabled);
        }

        #[tokio::test]
        async fn an_unconstrained_request_still_thinks() {
            let (status, _body, engine) = post_chat(thinking_stub(), &serde_json::json!({})).await;
            assert_eq!(status, http::StatusCode::OK);
            assert!(recorded_calls(&engine)[0].thinking_enabled);
        }

        #[tokio::test]
        async fn a_failed_stream_ends_with_an_error_frame_then_done() {
            let mut stub = stub_with_tokenizer();
            stub.script.stream = vec![
                token_event("par", 1, None),
                crate::test_support::failure_event(),
            ];
            let (status, body, _engine) =
                post_chat(stub, &serde_json::json!({"stream": true})).await;

            assert_eq!(status, http::StatusCode::OK, "{body}");
            let frames: Vec<&str> = body
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .collect();
            assert_eq!(frames.last(), Some(&"[DONE]"), "{body}");
            let error: serde_json::Value = serde_json::from_str(frames[frames.len() - 2]).unwrap();
            assert_eq!(error["error"]["type"], "server_error", "{body}");
            assert!(!body.contains("\"finish_reason\":\"error\""), "{body}");
        }

        #[tokio::test]
        async fn a_failed_generation_is_an_http_error_when_not_streaming() {
            let (status, body, _engine) =
                post_chat(stub_with_tokenizer(), &serde_json::json!({})).await;

            assert_eq!(status, http::StatusCode::INTERNAL_SERVER_ERROR, "{body}");
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
            assert_eq!(engine.recorded_prompt_tools(), vec![Some(1)]);
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
            assert_eq!(engine.recorded_prompt_tools(), vec![Some(1)]);
        }

        #[tokio::test]
        async fn tool_choice_none_hides_and_does_not_parse_tools() {
            let extra = serde_json::json!({"tools": tools_json(), "tool_choice": "none"});
            let (status, body, engine) = post_chat(tool_stub(), &extra).await;
            assert_eq!(status, http::StatusCode::OK, "{body}");
            assert_eq!(engine.recorded_prompt_tools(), vec![None]);
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
            assert_eq!(engine.recorded_prompt_tools(), vec![None]);
            assert!(!body.contains("tool_calls"), "{body}");
            assert!(body.contains("get_weather"), "text stays visible: {body}");
        }

        #[tokio::test]
        async fn seed_is_accepted_and_does_not_change_the_call() {
            let (status, body, engine) =
                post_chat(generating_stub(), &serde_json::json!({"seed": 42})).await;
            assert_eq!(status, http::StatusCode::OK, "{body}");
            assert_eq!(engine.recorded_calls().len(), 1);
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
