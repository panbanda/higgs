use std::convert::Infallible;

use axum::{
    Json,
    extract::State,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive},
    },
};
use tokio_stream::Stream;

use crate::{
    error::ServerError,
    state::SharedState,
    types::openai::{
        ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionDelta,
        ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse, ChoiceLogprobs,
        CompletionUsage, MessageContent, StopSequence, TokenLogprob, ToolCall, ToolCallFunction,
        TopLogprob,
    },
};
use higgs_models::SamplingParams;

pub async fn chat_completions(
    State(state): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    if req.messages.is_empty() {
        return Err(ServerError::BadRequest(
            "messages array must not be empty".to_owned(),
        ));
    }

    if req.stream == Some(true) {
        let stream = chat_completions_stream(state, req)?;
        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
        Ok(sse.into_response())
    } else {
        let response = chat_completions_non_streaming(state, req).await?;
        Ok(Json(response).into_response())
    }
}

#[allow(clippy::too_many_lines)]
async fn chat_completions_non_streaming(
    state: SharedState,
    req: ChatCompletionRequest,
) -> Result<ChatCompletionResponse, ServerError> {
    let engine = state
        .engine_for(&req.model)
        .ok_or_else(|| ServerError::ModelNotFound(req.model.clone()))?;

    let max_tokens = req.max_tokens.unwrap_or(state.config.max_tokens);
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
    let tools = req.tools.as_deref();

    let mut prompt_tokens = engine
        .prepare_chat_prompt(&messages, tools)
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

    let tokenizer = engine.tokenizer().clone();
    let output = tokio::task::spawn_blocking(move || {
        engine.generate(
            &prompt_tokens,
            max_tokens,
            &sampling,
            &stop_sequences,
            want_logprobs,
            top_logprobs,
            constraint,
            pixel_values,
        )
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
    .map_err(ServerError::Engine)?;

    let request_id = generate_request_id();
    let has_tools = req.tools.is_some();

    let logprobs_response = output
        .token_logprobs
        .as_ref()
        .map(|lps| logprobs_to_response(lps, &tokenizer));

    // Parse reasoning (think tags) from the output
    let reasoning_result = higgs_engine::reasoning_parser::parse_reasoning(&output.text);
    let raw_text = if reasoning_result.reasoning.is_some() {
        reasoning_result.text
    } else {
        output.text
    };
    let reasoning_content = reasoning_result.reasoning;

    let (content, tool_calls, finish_reason) = if has_tools {
        let parsed = higgs_engine::tool_parser::parse_tool_calls(&raw_text);
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

#[allow(clippy::too_many_lines, clippy::needless_pass_by_value)]
fn chat_completions_stream(
    state: SharedState,
    req: ChatCompletionRequest,
) -> Result<impl Stream<Item = Result<Event, Infallible>>, ServerError> {
    if req.tools.is_some() {
        return Err(ServerError::BadRequest(
            "Streaming with tool_calls is not yet supported".to_owned(),
        ));
    }

    let engine = state
        .engine_for(&req.model)
        .ok_or_else(|| ServerError::ModelNotFound(req.model.clone()))?;

    let max_tokens = req.max_tokens.unwrap_or(state.config.max_tokens);
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
    let tools = req.tools.as_deref();

    let mut prompt_tokens = engine
        .prepare_chat_prompt(&messages, tools)
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
    let created = current_unix_timestamp();
    let model = req.model;

    let tokenizer = engine.tokenizer().clone();
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
            constraint,
            pixel_values,
        );
        if let Err(e) = result {
            tracing::error!(error = %e, "Generation error during streaming");
        }
    });

    let stream = async_stream::stream! {
        // Send initial role chunk
        let role_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    role: Some("assistant".to_owned()),
                    content: None,
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
        };
        match serde_json::to_string(&role_chunk) {
            Ok(json) => yield Ok(Event::default().data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        let mut reasoning_tracker = higgs_engine::reasoning_parser::StreamingReasoningTracker::new();

        while let Some(output) = rx.recv().await {
            let chunk_logprobs = output
                .token_logprob
                .as_ref()
                .map(|lp| logprobs_to_response(std::slice::from_ref(lp), &tokenizer));

            let (visible, reasoning) = reasoning_tracker.process(&output.new_text);

            // Emit reasoning chunk if there's reasoning content
            if !reasoning.is_empty() {
                let reas_chunk = ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionDelta {
                            role: None,
                            content: None,
                            reasoning_content: Some(reasoning),
                            tool_calls: None,
                        },
                        finish_reason: None,
                        logprobs: None,
                    }],
                };
                match serde_json::to_string(&reas_chunk) {
                    Ok(json) => yield Ok(Event::default().data(json)),
                    Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
                }
            }

            // Emit visible content chunk
            if !visible.is_empty() {
                let chunk = ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionDelta {
                            role: None,
                            content: Some(visible),
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        finish_reason: output.finish_reason.clone(),
                        logprobs: chunk_logprobs,
                    }],
                };
                match serde_json::to_string(&chunk) {
                    Ok(json) => yield Ok(Event::default().data(json)),
                    Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
                }
            } else if output.finish_reason.is_some() {
                // Even if no visible text, still emit the finish_reason
                let chunk = ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionDelta {
                            role: None,
                            content: None,
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        finish_reason: output.finish_reason,
                        logprobs: chunk_logprobs,
                    }],
                };
                match serde_json::to_string(&chunk) {
                    Ok(json) => yield Ok(Event::default().data(json)),
                    Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
                }
            }
        }

        // Flush any remaining buffered content from the reasoning tracker
        let (flush_vis, flush_reas) = reasoning_tracker.flush();
        if !flush_reas.is_empty() {
            let reas_chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: ChatCompletionDelta {
                        role: None,
                        content: None,
                        reasoning_content: Some(flush_reas),
                        tool_calls: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
            };
            match serde_json::to_string(&reas_chunk) {
                Ok(json) => yield Ok(Event::default().data(json)),
                Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
            }
        }
        if !flush_vis.is_empty() {
            let vis_chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: ChatCompletionDelta {
                        role: None,
                        content: Some(flush_vis),
                        reasoning_content: None,
                        tool_calls: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
            };
            match serde_json::to_string(&vis_chunk) {
                Ok(json) => yield Ok(Event::default().data(json)),
                Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
            }
        }

        // Send [DONE] sentinel
        yield Ok(Event::default().data("[DONE]"));
    };

    Ok(stream)
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
