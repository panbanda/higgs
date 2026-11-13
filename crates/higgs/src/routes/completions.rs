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

use crate::{
    config::ApiFormat,
    error::ServerError,
    metrics::{MetricsStore, RequestMetricsContext, RequestRecord, RequestTiming},
    router::ResolvedRoute,
    state::{EngineLease, SharedState},
    types::openai::{
        ChoiceLogprobs, CompletionChoice, CompletionChunkChoice, CompletionRequest,
        CompletionResponse, StopSequence, TokenLogprob, TopLogprob,
    },
    usage::{TokenCounts, build_usage},
};
use higgs_engine::engine::{GenerationStats, ThinkingMode};
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
            if req.stream == Some(true) {
                let metrics = state.metrics.clone();
                let stream = completions_stream(
                    Arc::clone(&state),
                    req,
                    engine,
                    model_name,
                    metrics,
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
                let response =
                    completions_non_streaming(Arc::clone(&state), req, engine, received_at).await?;
                if let Some(ref metrics) = state.metrics {
                    metrics.record(RequestRecord {
                        id: 0,
                        timestamp: Instant::now(),
                        wallclock: chrono::Utc::now(),
                        model: Some(model_name),
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

async fn completions_non_streaming(
    state: SharedState,
    req: CompletionRequest,
    engine: EngineLease,
    received_at: Instant,
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
        usage: build_usage(
            TokenCounts {
                prompt: output.prompt_tokens,
                completion: output.completion_tokens,
                reasoning: None,
            },
            &output.stats,
            received_at,
        ),
    })
}

#[allow(clippy::needless_pass_by_value, clippy::too_many_lines)]
fn completions_stream(
    state: SharedState,
    req: CompletionRequest,
    engine: EngineLease,
    model_name: String,
    metrics: Option<Arc<MetricsStore>>,
    routing_method: crate::router::RoutingMethod,
    received_at: Instant,
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
    let include_usage = req
        .stream_options
        .as_ref()
        .is_some_and(|opts| opts.include_usage.unwrap_or(false));
    let prompt_token_count = u32::try_from(prompt_tokens.len()).unwrap_or(u32::MAX);

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
            ThinkingMode::new(engine.enable_thinking(), None),
            // Progress events feed the usage block and metrics; they are
            // never written to the client as completion chunks.
            true,
            None,
            None,
        );
        if let Err(e) = result {
            tracing::error!(error = %e, "Generation error during streaming");
        }
    });

    let start = received_at;
    let pending_id = metrics.as_ref().map(|m| {
        m.record_pending(RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: chrono::Utc::now(),
            model: Some(model_name),
            provider: Some("higgs".to_owned()),
            routing_method: routing_method.into(),
            status: 200,
            duration: std::time::Duration::ZERO,
            input_tokens: input_token_count,
            output_tokens: 0,
            error_body: None,
            timing: crate::metrics::RequestTiming::default(),
        })
    });

    let stream = async_stream::stream! {
        let mut writer = crate::sse::CompletionChunkWriter::new(&request_id, created, &model);
        let mut output_tokens: u64 = 0;
        let mut timing = RequestTiming::default();
        let mut stats = GenerationStats::default();
        while let Some(output) = rx.recv().await {
            stats.observe(&output, Instant::now());
            if let Some(progress) = output.prefill_progress {
                timing.cached_tokens = Some(u64::from(progress.cached));
                continue;
            }
            if timing.ttft_ms.is_none() {
                timing.ttft_ms = Some(u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX));
            }
            output_tokens = u64::from(output.completion_tokens);
            let choice = CompletionChunkChoice {
                index: 0,
                text: output.new_text,
                finish_reason: output.finish_reason,
            };
            match writer.write(&choice) {
                Ok(json) => yield Ok(Event::default().data(json)),
                Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
            }
        }

        if include_usage {
            let usage = build_usage(
                TokenCounts {
                    prompt: prompt_token_count,
                    completion: u32::try_from(output_tokens).unwrap_or(u32::MAX),
                    reasoning: None,
                },
                &stats,
                start,
            );
            match writer.write_usage(&usage) {
                Ok(json) => yield Ok(Event::default().data(json)),
                Err(e) => tracing::error!(error = %e, "Failed to serialize usage chunk"),
            }
        }

        if let Some(ref m) = metrics {
            if let Some(id) = pending_id {
                m.finalize_stream(id, output_tokens, start.elapsed(), timing);
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

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use crate::test_support::{
        StubEngine, app_with_stub, generation_output, post_json, progress_event, sse_json_events,
        token_event, word_tokenizer,
    };
    use higgs_engine::engine::GenerationStats;

    fn stub() -> StubEngine {
        let mut stub = StubEngine::named("stub");
        stub.tokenizer = Some(word_tokenizer());
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
