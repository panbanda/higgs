use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
        ChatCompletionChoice, ChatCompletionDelta, ChatCompletionMessage, ChatCompletionRequest,
        ChatCompletionResponse, ChoiceLogprobs, CompletionUsage, MessageContent, StopSequence,
        TokenLogprob, ToolCall, ToolCallDelta, ToolCallFunction, ToolCallFunctionDelta, TopLogprob,
    },
};
use higgs_models::SamplingParams;

#[allow(clippy::too_many_lines)]
pub async fn chat_completions(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ServerError> {
    let mut req: ChatCompletionRequest = serde_json::from_slice(&body)
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
            model_name,
            routing_method,
        } => {
            req.model = model_name;
            if req.stream == Some(true) {
                let stream = chat_completions_stream(
                    Arc::clone(&state),
                    req,
                    engine,
                    state.metrics.clone(),
                    routing_method,
                )?;
                let sse = Sse::new(stream).keep_alive(KeepAlive::default());
                Ok(sse.into_response())
            } else {
                let start = Instant::now();
                let response =
                    chat_completions_non_streaming(Arc::clone(&state), req, engine).await?;
                if let Some(ref metrics) = state.metrics {
                    metrics.record(RequestRecord {
                        id: 0,
                        timestamp: Instant::now(),
                        wallclock: chrono::Utc::now(),
                        model: response.model.clone(),
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
                            model: metrics_model.clone(),
                            provider: provider_name.clone(),
                            routing_method: routing_method.into(),
                            status: result.as_ref().map_or(502, |resp| resp.status().as_u16()),
                            duration: start.elapsed(),
                            input_tokens: 0,
                            output_tokens: 0,
                            error_body: None,
                        });
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
                                model: metrics_model.clone(),
                                provider: provider_name.clone(),
                                routing_method: routing_method.into(),
                                status: upstream_status,
                                duration: start.elapsed(),
                                input_tokens: 0,
                                output_tokens: 0,
                                error_body: None,
                            });
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
                        let usage = crate::proxy::extract_usage(&resp_bytes);
                        if let Some(ref metrics) = state.metrics {
                            metrics.record(RequestRecord {
                                id: 0,
                                timestamp: Instant::now(),
                                wallclock: chrono::Utc::now(),
                                model: metrics_model.clone(),
                                provider: provider_name.clone(),
                                routing_method: routing_method.into(),
                                status: upstream_status,
                                duration: start.elapsed(),
                                input_tokens: usage.0,
                                output_tokens: usage.1,
                                error_body: None,
                            });
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
async fn chat_completions_non_streaming(
    state: SharedState,
    req: ChatCompletionRequest,
    engine: Arc<Engine>,
) -> Result<ChatCompletionResponse, ServerError> {
    let max_tokens = req.max_tokens.unwrap_or(state.config.server.max_tokens);
    let sampling = build_sampling_params(&req)?;
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
    let thinking_enabled = crate::reasoning::effective_thinking_enabled(
        engine.enable_thinking(),
        &[engine.model_name(), req.model.as_str()],
        req.reasoning.as_ref(),
        req.chat_template_kwargs
            .as_ref()
            .and_then(|k| k.enable_thinking)
            .or(req.enable_thinking),
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

    let constraint = build_constraint(req.response_format.as_ref(), &engine)?;

    // Opt-in multi-turn KV-cache reuse. Only honored on the non-streaming path
    // for the Simple engine; `generate_continued` errors for other variants and
    // we never reach this branch for them because `retained`/`generate_continued`
    // degrade cleanly. Images and constrained decoding are not supported by the
    // continued path, so fall back to the normal generation when present.
    //
    // BEST-EFFORT, not exact replay: the retained KV is TurboQuant-compressed
    // (lossy) and the prompt is reconciled in text space below, so a continued
    // turn may differ slightly from a stateless full prefill. Clients needing
    // bit-identical output should omit `session_id` — the radix prefix cache on
    // the normal path is exact. See `SimpleEngine::generate_continued`.
    let session_id = req
        .session_id
        .filter(|_| pixel_values.is_none() && constraint.is_none());

    let tokenizer = engine.tokenizer().clone();
    let checkpoint_id = req.checkpoint_id.clone();
    let request_id = generate_request_id();
    let has_tools = tools.is_some();

    if let Some(sid) = session_id {
        let continued_prompt = continued_prompt_tokens(
            &engine,
            sid,
            &prompt_tokens,
            &messages,
            tools,
            thinking_enabled,
        );

        let sampling_c = sampling.clone();
        let engine_c = Arc::clone(&engine);
        let session_output = tokio::task::spawn_blocking(move || {
            engine_c.generate_continued_with_thinking(
                sid,
                &continued_prompt,
                max_tokens,
                &sampling_c,
                thinking_enabled,
            )
        })
        .await
        .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
        .map_err(ServerError::Engine)?;

        return Ok(build_session_response(
            &req.model,
            &request_id,
            session_output,
            tools,
            has_tools,
            thinking_enabled,
        ));
    }

    let output = tokio::task::spawn_blocking(move || {
        engine.generate_with_thinking(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            want_logprobs,
            top_logprobs,
            thinking_enabled,
            constraint,
            pixel_values,
            checkpoint_id.as_deref(),
        )
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
    .map_err(ServerError::Engine)?;

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
        // Model-emitted reasoning (e.g. VibeThinker writes its own
        // `<think>...</think>`): parse it out without the prompt-injection
        // prepend used for template-opened thinking. No-op when absent, so it
        // matches the streaming path's `new()` tracker for every model.
        let reasoning_result = higgs_engine::reasoning_parser::parse_reasoning(&output_text);
        if reasoning_result.reasoning.is_some() {
            (reasoning_result.text, reasoning_result.reasoning)
        } else {
            (output_text, None)
        }
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
        usage: CompletionUsage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        },
    })
}

/// Build the token sequence fed to `generate_continued`. The engine guard
/// prefills only the suffix IFF the retained tokens are a strict token-prefix of
/// what we pass, so we hand it `retained ++ delta`.
///
/// `retained` carries the `<think>...</think>` block the chat template injected
/// for the prior assistant turn. The canonical multi-turn render strips
/// `<think>` from historical assistant turns, so match a think-stripped copy of
/// `retained` against the full render to find the new-turn delta, then append
/// that delta to the original retained tokens.
// `print_stderr`: env-gated (HIGGS_DIAG_SESSION_TIMING) diagnostics only.
#[allow(clippy::print_stderr)]
fn continued_prompt_tokens(
    engine: &Arc<Engine>,
    session_id: u64,
    prompt_tokens: &[u32],
    messages: &[higgs_engine::chat_template::ChatMessage],
    tools: Option<&[serde_json::Value]>,
    thinking_enabled: bool,
) -> Vec<u32> {
    let diag = std::env::var("HIGGS_DIAG_SESSION_TIMING").is_ok_and(|v| v == "1");
    let Some(retained) = engine.retained_session_tokens(session_id) else {
        if diag {
            eprintln!(
                "DIAG session-continue-fallback: reason=no_retained_cache session_id={session_id} prompt_tokens={}",
                prompt_tokens.len()
            );
        }
        return prompt_tokens.to_vec();
    };
    let retained_text = match engine.tokenizer().decode(&retained, false) {
        Ok(text) => text,
        Err(error) => {
            if diag {
                eprintln!(
                    "DIAG session-continue-fallback: reason=decode_retained_failed session_id={session_id} retained_tokens={} error={error}",
                    retained.len()
                );
            }
            return prompt_tokens.to_vec();
        }
    };
    let full_text = match engine.render_chat_prompt_with_thinking(messages, tools, thinking_enabled)
    {
        Ok(text) => text,
        Err(error) => {
            if diag {
                eprintln!(
                    "DIAG session-continue-fallback: reason=render_full_failed session_id={session_id} retained_tokens={} prompt_tokens={} error={error}",
                    retained.len(),
                    prompt_tokens.len()
                );
            }
            return prompt_tokens.to_vec();
        }
    };
    let Some(delta_text) = message_boundary_delta(&retained_text, &full_text) else {
        if diag {
            let common = common_prefix_bytes(&retained_text, &full_text);
            eprintln!(
                "DIAG session-continue-fallback: reason=boundary_splice_failed session_id={session_id} retained_tokens={} prompt_tokens={} retained_bytes={} full_bytes={} retained_msgs={} full_msgs={} common_bytes={} retained_tail={:?} full_at_mismatch={:?}",
                retained.len(),
                prompt_tokens.len(),
                retained_text.len(),
                full_text.len(),
                retained_text.matches(IM_START).count(),
                full_text.matches(IM_START).count(),
                common,
                preview_around(&retained_text, common),
                preview_around(&full_text, common)
            );
        }
        return prompt_tokens.to_vec();
    };
    let Ok(delta_enc) = engine.tokenizer().encode(delta_text, false) else {
        if diag {
            eprintln!(
                "DIAG session-continue-fallback: reason=encode_delta_failed session_id={session_id} delta_bytes={}",
                delta_text.len()
            );
        }
        return prompt_tokens.to_vec();
    };
    if diag {
        eprintln!(
            "DIAG session-continue: matched session_id={session_id} retained_tokens={} delta_tokens={} full_prompt_tokens={}",
            retained.len(),
            delta_enc.get_ids().len(),
            prompt_tokens.len()
        );
    }
    let mut combined = retained;
    combined.extend_from_slice(delta_enc.get_ids());
    combined
}

fn common_prefix_bytes(left: &str, right: &str) -> usize {
    left.as_bytes()
        .iter()
        .zip(right.as_bytes())
        .take_while(|(l, r)| l == r)
        .count()
}

fn preview_around(text: &str, byte_pos: usize) -> String {
    let start = char_boundary_at_or_before(text, byte_pos.saturating_sub(80));
    let end = char_boundary_at_or_after(text, byte_pos.saturating_add(80));
    text.get(start..end).unwrap_or_default().to_owned()
}

fn char_boundary_at_or_before(text: &str, byte_pos: usize) -> usize {
    let pos = byte_pos.min(text.len());
    (0..=pos)
        .rev()
        .find(|idx| text.is_char_boundary(*idx))
        .unwrap_or(0)
}

fn char_boundary_at_or_after(text: &str, byte_pos: usize) -> usize {
    let pos = byte_pos.min(text.len());
    (pos..=text.len())
        .find(|idx| text.is_char_boundary(*idx))
        .unwrap_or(text.len())
}

/// Remove `<think>...</think>` blocks — and the single `\n\n` the chat template
/// emits after each — from a rendered prompt string. Used by the cache-resident
/// continuation path to reconcile a retained KV's detokenization (which carries
/// the per-turn think injection) with the canonical multi-turn render (which
/// strips think from historical turns), so the retained prefix can be matched.
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

/// Splice point for a continued turn: the suffix of the canonical render that
/// covers the messages the retained KV does NOT yet cover.
///
/// The retained detokenization can never byte-match the canonical re-render of
/// the same messages — think blocks are stripped or replaced by a placeholder
/// depending on the turn's position relative to the conversation's last user
/// message, and content/tool-call join whitespace differs from what the model
/// generated. Instead of canonicalizing (template lore, position-dependent),
/// splice at message boundaries: the retained text covers as many messages as
/// it has `<|im_start|>` markers; everything from that boundary onward in the
/// fresh render is the delta to prefill. The retained prefix keeps the model's
/// original per-turn text (real thinking, original whitespace) — a known,
/// accepted divergence source of the best-effort continuation path.
///
/// The last retained message usually ends without `<|im_end|>` (the engine
/// pops the EOS token before stashing), so the delta starts AT the covered
/// messages' final `<|im_end|>`; when the retained text does end with it, the
/// delta starts right after instead. Returns `None` (caller falls back to a
/// full prefill) when the render has fewer messages than the retained text or
/// the shared first message diverges (client mutated history).
fn message_boundary_delta<'a>(retained_text: &str, full_text: &'a str) -> Option<&'a str> {
    let covered = retained_text.matches(IM_START).count();
    if covered == 0 {
        return None;
    }
    // Mutation guard: the first message (system/user opener) must agree
    // byte-for-byte between the retained text and the fresh render.
    let first_end = retained_text
        .find(IM_END)
        .map_or_else(|| retained_text.len().min(256), |p| p.min(256));
    if full_text.get(..first_end) != retained_text.get(..first_end) {
        return None;
    }
    // Offset of the `covered`-th message's closing <|im_end|> in the render:
    // each covered message renders exactly one.
    let mut pos = 0usize;
    for _ in 0..covered {
        let found = full_text.get(pos..)?.find(IM_END)?;
        pos = pos + found + IM_END.len();
    }
    // pos is just past the covered messages' final <|im_end|>. The retained
    // text usually lacks that closer (EOS popped at stash time) — include it
    // in the delta; skip it when the retained text already ends with it.
    let trimmed = retained_text.trim_end_matches('\n');
    if trimmed.ends_with(IM_END) {
        full_text.get(pos..)
    } else {
        full_text.get(pos - IM_END.len()..)
    }
}

/// Map a [`SessionGeneration`] (cache-resident continued turn) onto the same
/// `ChatCompletionResponse` shape as the normal path, preserving reasoning
/// extraction, tool-call parsing, and the `finish_reason: "tool_calls"`
/// override. The continued path uses greedy decode without logprobs/streaming,
/// so logprobs are absent and `finish_reason` defaults to `"stop"`.
fn build_session_response(
    model: &str,
    request_id: &str,
    output: higgs_engine::simple::SessionGeneration,
    tools: Option<&[serde_json::Value]>,
    has_tools: bool,
    thinking_enabled: bool,
) -> ChatCompletionResponse {
    let output_text = output.text;
    // Same reasoning-tag handling as the normal path: the template opens
    // `<think>` in the prompt, so generated text starts inside the think block.
    let (raw_text, reasoning_content) = if thinking_enabled {
        let parse_input = if output_text.contains("</think>") {
            format!("<think>{output_text}")
        } else {
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
                "stop".to_owned(),
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
            "stop".to_owned(),
        )
    };

    ChatCompletionResponse {
        id: request_id.to_owned(),
        object: "chat.completion",
        created: current_unix_timestamp(),
        model: model.to_owned(),
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
            logprobs: None,
        }],
        usage: CompletionUsage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        },
    }
}

#[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
fn chat_completions_stream(
    state: SharedState,
    req: ChatCompletionRequest,
    engine: Arc<Engine>,
    metrics: Option<Arc<MetricsStore>>,
    routing_method: crate::router::RoutingMethod,
) -> Result<Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>, ServerError> {
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
    let sampling = build_sampling_params(&req)?;
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
    let thinking_enabled_stream = crate::reasoning::effective_thinking_enabled(
        engine.enable_thinking(),
        &[engine.model_name(), req.model.as_str()],
        req.reasoning.as_ref(),
        req.chat_template_kwargs
            .as_ref()
            .and_then(|k| k.enable_thinking)
            .or(req.enable_thinking),
    );

    // Pass tools into prompt rendering so the chat template emits the
    // tool spec the model recognises. The on-the-fly
    // [`StreamingToolCallTracker`] below intercepts `<tool_call>…
    // </tool_call>` blocks the model produces and turns them into
    // structured `ToolCallDelta` SSE events.
    let prompt_tools = req
        .tools
        .as_deref()
        .and_then(|t| if t.is_empty() { None } else { Some(t) });
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

    let constraint = build_constraint(req.response_format.as_ref(), &engine)?;

    let request_id = generate_request_id();
    let include_usage = req
        .stream_options
        .as_ref()
        .is_some_and(|opts| opts.include_usage.unwrap_or(false));
    let return_progress = req.return_progress.unwrap_or(false);
    let created = current_unix_timestamp();
    let tools_for_session = req.tools.clone();
    let request_session_id = req.session_id;
    let model = req.model;
    let checkpoint_id = req.checkpoint_id;
    let prompt_token_count = u32::try_from(prompt_tokens.len()).unwrap_or(0);

    let start = Instant::now();
    let metrics_id = metrics.as_ref().map(|m| {
        m.record_pending(RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: chrono::Utc::now(),
            model: model.clone(),
            provider: "higgs".to_owned(),
            routing_method: routing_method.into(),
            status: 200,
            duration: Duration::ZERO,
            input_tokens: u64::from(prompt_token_count),
            output_tokens: 0,
            error_body: None,
        })
    });

    let stream_session_id = request_session_id
        .filter(|_| pixel_values.is_none() && constraint.is_none() && !want_logprobs)
        .filter(|_| checkpoint_id.is_none());
    if let Some(sid) = stream_session_id {
        let continued_prompt = continued_prompt_tokens(
            &engine,
            sid,
            &prompt_tokens,
            &messages,
            prompt_tools,
            thinking_enabled_stream,
        );
        let engine_c = Arc::clone(&engine);
        let sampling_c = sampling.clone();
        let request_id_c = request_id.clone();
        let model_c = model.clone();
        let stream = async_stream::stream! {
            let mut writer = crate::sse::ChatChunkWriter::new(&request_id_c, created, &model_c);

            macro_rules! emit_delta {
                ($delta:expr, $finish:expr, $logprobs:expr) => {
                    match writer.write_delta($delta, $finish, $logprobs) {
                        Ok(json) => yield Ok(Event::default().data(json)),
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

            let session_result = tokio::task::spawn_blocking(move || {
                engine_c.generate_continued_with_thinking(
                    sid,
                    &continued_prompt,
                    max_tokens,
                    &sampling_c,
                    thinking_enabled_stream,
                )
            })
            .await;

            match session_result {
                Ok(Ok(output)) => {
                    let tools = tools_for_session.as_deref().filter(|t| !t.is_empty());
                    let response = build_session_response(
                        &model_c,
                        &request_id_c,
                        output,
                        tools,
                        stream_includes_tools,
                        thinking_enabled_stream,
                    );
                    if let Some(choice) = response.choices.first() {
                        if let Some(reasoning) = choice.message.reasoning_content.clone() {
                            let d = ChatCompletionDelta {
                                role: None,
                                content: None,
                                reasoning_content: Some(reasoning),
                                tool_calls: None,
                            };
                            emit_delta!(&d, None, None);
                        }
                        if let Some(content) = choice.message.content.as_ref() {
                            let text = content.text();
                            if !text.is_empty() {
                                let d = ChatCompletionDelta {
                                    role: None,
                                    content: Some(text),
                                    reasoning_content: None,
                                    tool_calls: None,
                                };
                                emit_delta!(&d, None, None);
                            }
                        }
                        if let Some(tool_calls) = choice.message.tool_calls.as_ref() {
                            for (index, tool_call) in tool_calls.iter().enumerate() {
                                let d = ChatCompletionDelta {
                                    role: None,
                                    content: None,
                                    reasoning_content: None,
                                    tool_calls: Some(vec![ToolCallDelta {
                                        index: u32::try_from(index).unwrap_or(u32::MAX),
                                        id: Some(tool_call.id.clone()),
                                        r#type: Some(tool_call.r#type.clone()),
                                        function: Some(ToolCallFunctionDelta {
                                            name: Some(tool_call.function.name.clone()),
                                            arguments: Some(tool_call.function.arguments.clone()),
                                        }),
                                    }]),
                                };
                                emit_delta!(&d, None, None);
                            }
                        }
                        let final_delta = ChatCompletionDelta {
                            role: None,
                            content: None,
                            reasoning_content: None,
                            tool_calls: None,
                        };
                        emit_delta!(&final_delta, Some(choice.finish_reason.as_str()), None);
                    }

                    if include_usage {
                        match writer.write_usage(&response.usage) {
                            Ok(json) => yield Ok(Event::default().data(json)),
                            Err(e) => tracing::error!(error = %e, "Failed to serialize usage chunk"),
                        }
                    }
                    if let Some(ref m) = metrics {
                        if let Some(id) = metrics_id {
                            m.finalize_stream(id, u64::from(response.usage.completion_tokens), start.elapsed());
                        }
                    }
                }
                Ok(Err(e)) => tracing::error!(error = %e, "Session generation error during streaming"),
                Err(e) => tracing::error!(error = %e, "Session generation task join error during streaming"),
            }

            yield Ok(Event::default().data("[DONE]"));
        };

        return Ok(Box::pin(stream));
    }

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
            thinking_enabled_stream,
            return_progress,
            constraint,
            pixel_values,
            checkpoint_id.as_deref(),
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
                    Ok(json) => yield Ok(Event::default().data(json)),
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

        while let Some(output) = rx.recv().await {
            // Prefill-progress events carry no tokens: forward as
            // `prompt_progress` chunks when the client opted in, and keep
            // them away from the delta/tool trackers either way.
            if let Some(p) = output.prefill_progress {
                if return_progress {
                    let time_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
                    let json = writer.write_prompt_progress(p.total, p.cached, p.processed, time_ms);
                    yield Ok(Event::default().data(json));
                }
                continue;
            }
            output_token_count = output.completion_tokens;
            let chunk_logprobs = output
                .token_logprob
                .as_ref()
                .map(|lp| logprobs_to_response(std::slice::from_ref(lp), &tokenizer));

            let (visible, reasoning) = reasoning_tracker.process(&output.new_text);

            if !reasoning.is_empty() {
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
            let usage = CompletionUsage {
                prompt_tokens: prompt_token_count,
                completion_tokens: output_token_count,
                total_tokens: prompt_token_count + output_token_count,
            };
            match writer.write_usage(&usage) {
                Ok(json) => yield Ok(Event::default().data(json)),
                Err(e) => tracing::error!(error = %e, "Failed to serialize usage chunk"),
            }
        }

        if let Some(ref m) = metrics {
            if let Some(id) = metrics_id {
                m.finalize_stream(id, u64::from(output_token_count), start.elapsed());
            }
        }

        // Send [DONE] sentinel
        yield Ok(Event::default().data("[DONE]"));
    };

    Ok(Box::pin(stream))
}

fn convert_messages(
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

fn build_sampling_params(req: &ChatCompletionRequest) -> Result<SamplingParams, ServerError> {
    let speculation =
        higgs_models::Speculation::parse(req.speculation.as_deref()).map_err(|v| {
            ServerError::BadRequest(format!(
                "invalid 'speculation' value '{v}' (expected auto|dflash|mtp|none)"
            ))
        })?;
    Ok(SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k,
        min_p: req.min_p,
        repetition_penalty: req.repetition_penalty,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
        speculation,
        thinking_budget: req.reasoning_budget,
    })
}

/// Build a constrained generator from the request's `response_format`.
///
/// Returns `None` if no constraint is needed (text mode or absent).
fn build_constraint(
    response_format: Option<&crate::types::openai::ResponseFormat>,
    engine: &std::sync::Arc<crate::state::Engine>,
) -> Result<Option<higgs_engine::constrained::ConstrainedGenerator>, ServerError> {
    let Some(fmt) = response_format else {
        return Ok(None);
    };

    match fmt.r#type.as_str() {
        "text" => Ok(None),
        "json_object" | "json_schema" => {
            let eos_id = engine.eos_token_ids().first().copied().unwrap_or(0);
            let vocab = higgs_engine::constrained::build_vocabulary(engine.tokenizer(), eos_id)
                .map_err(ServerError::Engine)?;
            let constraint = if fmt.r#type == "json_schema" {
                if let Some(ref schema) = fmt.json_schema {
                    // OpenAI spec wraps the actual schema under a `schema` key:
                    // {"name": "...", "schema": {<actual schema>}}
                    // Fall back to the whole value for bare schemas.
                    let inner = schema
                        .get("schema")
                        .cloned()
                        .unwrap_or_else(|| schema.clone());
                    let schema_str = inner.to_string();
                    higgs_engine::constrained::ConstrainedGenerator::from_json_schema(
                        &schema_str,
                        &vocab,
                    )
                    .map_err(ServerError::Engine)?
                } else {
                    higgs_engine::constrained::ConstrainedGenerator::for_json_object(&vocab)
                        .map_err(ServerError::Engine)?
                }
            } else {
                higgs_engine::constrained::ConstrainedGenerator::for_json_object(&vocab)
                    .map_err(ServerError::Engine)?
            };

            Ok(Some(constraint))
        }
        other => Err(ServerError::BadRequest(format!(
            "Unsupported response_format type: {other}"
        ))),
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

fn generate_request_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4())
}

fn current_unix_timestamp() -> i64 {
    chrono::Utc::now().timestamp()
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn boundary_delta_splices_after_covered_messages() {
        // Retained: system + user + generated assistant reply (EOS popped, so
        // no trailing <|im_end|>), with real thinking the render won't have.
        let retained = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nq1<|im_end|>\n<|im_start|>assistant\n<think>\nreasoning\n</think>\n\nanswer1";
        // Fresh render: same three messages canonically rendered (thinking
        // stripped, whitespace differs) plus a new user turn + gen suffix.
        let full = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nq1<|im_end|>\n<|im_start|>assistant\nanswer1<|im_end|>\n<|im_start|>user\nq2<|im_end|>\n<|im_start|>assistant\n<think>";
        let delta = message_boundary_delta(retained, full).unwrap();
        // Delta starts at the covered messages' closing <|im_end|> (retained
        // lacks it) and carries everything new.
        assert_eq!(
            delta,
            "<|im_end|>\n<|im_start|>user\nq2<|im_end|>\n<|im_start|>assistant\n<think>"
        );
    }

    #[test]
    fn boundary_delta_skips_closer_when_retained_has_it() {
        let retained = "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\nans<|im_end|>";
        let full = "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\nans<|im_end|>\n<|im_start|>user\nq2<|im_end|>";
        let delta = message_boundary_delta(retained, full).unwrap();
        assert_eq!(delta, "\n<|im_start|>user\nq2<|im_end|>");
    }

    #[test]
    fn boundary_delta_rejects_mutated_first_message() {
        // Client rewrote the system prompt between turns: no splice.
        let retained = "<|im_start|>system\noriginal<|im_end|>\n<|im_start|>user\nq<|im_end|>";
        let full = "<|im_start|>system\nMUTATED!<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>user\nq2<|im_end|>";
        assert!(message_boundary_delta(retained, full).is_none());
    }

    #[test]
    fn boundary_delta_rejects_render_with_fewer_messages() {
        // Render has fewer message closers than retained covers (history
        // shrank): no splice.
        let retained = "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\nans";
        let full = "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nq<|im_end|>";
        assert!(message_boundary_delta(retained, full).is_none());
    }

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
}
