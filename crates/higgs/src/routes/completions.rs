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
    error::ServerError,
    generation::{
        Completed, EventStream, GenerationEvent, GenerationRequest, Prompt, StreamMetrics,
    },
    metrics::{RequestMetricsContext, RequestRecord},
    router::ResolvedRoute,
    state::SharedState,
    types::openai::{
        CompletionChoice, CompletionChunkChoice, CompletionRequest, CompletionResponse,
        StopSequence,
    },
};
use higgs_models::SamplingParams;

#[allow(clippy::too_many_lines)]
pub async fn completions(
    State(state): State<SharedState>,
    Extension(request_metrics): Extension<RequestMetricsContext>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ServerError> {
    // Captured before parsing and prompt preparation so TTFT measures
    // the client-observed wait, not only generation.
    let received_at = Instant::now();
    let req: CompletionRequest = serde_json::from_slice(&body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid request body: {e}")))?;
    request_metrics.set_requested_model(&req.model);

    if req.prompt.is_empty() {
        return Err(ServerError::BadRequest(
            "prompt must not be empty".to_owned(),
        ));
    }

    let resolved = state.router.resolve(&req.model, None).await?;

    match resolved {
        ResolvedRoute::Higgs {
            engine,
            model_name,
            routing_method,
        } => {
            let request =
                generation_request(&req, model_name.clone(), state.config.server.max_tokens);
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
                let include_usage = req
                    .stream_options
                    .as_ref()
                    .is_some_and(|opts| opts.include_usage.unwrap_or(false));
                let sse = Sse::new(completion_chunk_stream(events, req.model, include_usage))
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
                        &model_name,
                        routing_method,
                        start,
                        &completed.usage,
                    );
                    request_metrics.mark_recorded();
                }
                let response = completion_response(completed, req.model);
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
                api_key.as_ref().map(crate::config::Secret::expose),
            )
            .await;
            let metrics_model = model_rewrite.as_deref().unwrap_or(&req.model).to_owned();
            if let Some(ref metrics) = state.metrics {
                let status = response.as_ref().map_or(502, |resp| resp.status().as_u16());
                metrics.record(RequestRecord {
                    id: 0,
                    timestamp: Instant::now(),
                    wallclock: chrono::Utc::now(),
                    model: Some(metrics_model),
                    provider: Some(provider_name),
                    routing_method: routing_method.into(),
                    status,
                    duration: start.elapsed(),
                    input_tokens: 0,
                    output_tokens: 0,
                    error_body: None,
                    timing: crate::metrics::RequestTiming::default(),
                });
                request_metrics.mark_recorded();
            }
            response
        }
    }
}

fn generation_request(
    req: &CompletionRequest,
    model_name: String,
    default_max_tokens: u32,
) -> GenerationRequest {
    GenerationRequest {
        model: model_name,
        prompt: Prompt::Text(req.prompt.clone()),
        max_tokens: req.max_tokens.unwrap_or(default_max_tokens),
        sampling: build_sampling_params(req),
        stop_sequences: StopSequence::extract(req.stop.clone()),
        logprobs: req.logprobs.unwrap_or(false),
        top_logprobs: req.top_logprobs,
        constraint: None,
        mcp: None,
    }
}

fn completion_response(completed: Completed, model: String) -> CompletionResponse {
    CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion",
        created: chrono::Utc::now().timestamp(),
        model,
        choices: vec![CompletionChoice {
            index: 0,
            text: completed.text,
            finish_reason: completed.finish_reason,
            logprobs: completed.logprobs,
        }],
        usage: completed.usage,
    }
}

/// Render generation events as completion chunks. The finish reason rides on
/// the chunk of the last piece of text, as it did when each piece of engine
/// output was its own chunk.
fn completion_chunk_stream(
    mut events: EventStream,
    model: String,
    include_usage: bool,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    async_stream::stream! {
        let mut writer = crate::sse::CompletionChunkWriter::new(&request_id, created, &model);
        let mut held_text: Option<String> = None;

        macro_rules! emit_chunk {
            ($text:expr, $finish_reason:expr) => {
                let choice = CompletionChunkChoice {
                    index: 0,
                    text: $text,
                    finish_reason: $finish_reason,
                };
                match writer.write(&choice) {
                    Ok(json) => yield Ok(Event::default().data(json)),
                    Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
                }
            };
        }

        while let Some(event) = events.next().await {
            match event {
                GenerationEvent::TextDelta { text, .. } => {
                    if let Some(previous) = held_text.replace(text) {
                        emit_chunk!(previous, None);
                    }
                }
                GenerationEvent::Finish { reason, .. } => {
                    emit_chunk!(held_text.take().unwrap_or_default(), Some(reason));
                }
                GenerationEvent::Usage(usage) => {
                    if let Some(text) = held_text.take() {
                        emit_chunk!(text, None);
                    }
                    if include_usage {
                        match writer.write_usage(&usage) {
                            Ok(json) => yield Ok(Event::default().data(json)),
                            Err(e) => tracing::error!(error = %e, "Failed to serialize usage chunk"),
                        }
                    }
                }
                GenerationEvent::Started { .. }
                | GenerationEvent::PrefillProgress(_)
                | GenerationEvent::ReasoningDelta(_)
                | GenerationEvent::ToolCall(_)
                | GenerationEvent::ToolLoop(_)
                | GenerationEvent::Error(_) => {}
            }
        }
        if let Some(text) = held_text.take() {
            emit_chunk!(text, None);
        }

        yield Ok(Event::default().data("[DONE]"));
    }
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

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use crate::test_support::{
        StubEngine, app_with_stub, generation_output, post_json, progress_event, sse_json_events,
        token_event,
    };
    use higgs_engine::engine::GenerationStats;

    fn stub() -> StubEngine {
        let mut stub = StubEngine::named("stub");
        stub.script.stamp_times = true;
        stub
    }

    fn body(extra: &serde_json::Value) -> serde_json::Value {
        let mut body = serde_json::json!({"model": "stub", "prompt": "hello world hello"});
        if let (Some(target), Some(fields)) = (body.as_object_mut(), extra.as_object()) {
            target.extend(fields.clone());
        }
        body
    }

    #[tokio::test]
    async fn non_streaming_usage_reports_cache_hit_and_timings() {
        let mut stub = stub();
        let mut output = generation_output("hello", 3, 4);
        output.stats = GenerationStats {
            cached_tokens: Some(2),
            ..GenerationStats::default()
        };
        stub.script.generation = Some(output);
        let (app, _engine) = app_with_stub(stub);

        let (status, text) =
            post_json(&app, "/v1/completions", &body(&serde_json::json!({}))).await;

        assert_eq!(status, http::StatusCode::OK, "{text}");
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        let usage = &json["usage"];
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 2);
        assert_eq!(usage["total_tokens"], 7);
        assert!(usage["time_to_first_token"].is_f64());
        assert!(usage["total_time"].is_f64());
        assert!(usage.get("completion_tokens_details").is_none());
    }

    #[tokio::test]
    async fn streaming_include_usage_appends_usage_chunk_without_progress_chunks() {
        let mut stub = stub();
        stub.script.stream = vec![
            progress_event(1, 3),
            token_event("hello", 1, None),
            token_event(" world", 2, Some("stop")),
        ];
        let (app, _engine) = app_with_stub(stub);
        let request = body(&serde_json::json!({
            "stream": true,
            "stream_options": {"include_usage": true},
        }));

        let (status, text) = post_json(&app, "/v1/completions", &request).await;

        assert_eq!(status, http::StatusCode::OK, "{text}");
        let events = sse_json_events(&text);
        assert_eq!(events.len(), 3, "two token chunks plus usage: {text}");
        let usage_chunk = events.last().unwrap();
        assert_eq!(usage_chunk["choices"], serde_json::json!([]));
        assert_eq!(usage_chunk["usage"]["prompt_tokens"], 3);
        assert_eq!(usage_chunk["usage"]["completion_tokens"], 2);
        assert_eq!(
            usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"],
            1
        );
        assert!(usage_chunk["usage"]["time_to_first_token"].is_f64());
        for chunk in &events[..2] {
            assert!(chunk.get("usage").is_none());
            assert_eq!(chunk["choices"].as_array().unwrap().len(), 1);
        }
    }

    #[tokio::test]
    async fn streaming_without_include_usage_has_no_usage_and_no_empty_chunks() {
        let mut stub = stub();
        stub.script.stream = vec![progress_event(0, 3), token_event("hello", 1, Some("stop"))];
        let (app, _engine) = app_with_stub(stub);

        let (status, text) = post_json(
            &app,
            "/v1/completions",
            &body(&serde_json::json!({"stream": true})),
        )
        .await;

        assert_eq!(status, http::StatusCode::OK, "{text}");
        let events = sse_json_events(&text);
        assert_eq!(events.len(), 1, "{text}");
        assert!(!text.contains("\"usage\""));
    }

    #[tokio::test]
    async fn include_usage_false_adds_nothing() {
        let mut stub = stub();
        stub.script.stream = vec![token_event("hello", 1, Some("stop"))];
        let (app, _engine) = app_with_stub(stub);
        let request = body(&serde_json::json!({
            "stream": true,
            "stream_options": {"include_usage": false},
        }));

        let (_status, text) = post_json(&app, "/v1/completions", &request).await;

        assert!(!text.contains("\"usage\""), "{text}");
    }
}
