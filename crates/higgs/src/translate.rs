//! Cross-format translation between `OpenAI` and Anthropic API formats.
//!
//! Used when a request arrives in one format (e.g., `OpenAI` `/v1/chat/completions`)
//! but routes to a provider that speaks the other format (e.g., Anthropic).

use std::convert::Infallible;

use axum::response::sse::Event;
use bytes::Bytes;

use crate::error::ServerError;

// ---------------------------------------------------------------------------
// Finish-reason mapping
// ---------------------------------------------------------------------------

fn anthropic_stop_to_openai_finish(stop_reason: &str) -> String {
    match stop_reason {
        "end_turn" => "stop".to_owned(),
        "max_tokens" => "length".to_owned(),
        "tool_use" => "tool_calls".to_owned(),
        other => other.to_owned(),
    }
}

fn openai_finish_to_anthropic_stop(finish_reason: &str) -> String {
    crate::anthropic_adapter::openai_finish_to_anthropic_stop(finish_reason)
}

// ---------------------------------------------------------------------------
// Request translation
// ---------------------------------------------------------------------------

/// Translate an `OpenAI` `/v1/chat/completions` request body to Anthropic `/v1/messages`.
#[allow(clippy::too_many_lines)]
pub fn openai_to_anthropic_request(
    body: &[u8],
    default_max_tokens: u32,
) -> Result<Bytes, ServerError> {
    let openai: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid JSON: {e}")))?;

    let messages = openai
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or_else(|| ServerError::BadRequest("missing messages array".to_owned()))?;

    // Extract system messages into the `system` field
    let mut system_parts = Vec::new();
    let mut anthropic_messages = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        match role {
            "system" => {
                if let Some(text) = extract_text_content(msg) {
                    system_parts.push(text);
                }
            }
            "tool" => {
                // OpenAI tool role -> Anthropic tool_result content block
                let tool_call_id = msg
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned();
                let content = extract_text_content(msg).unwrap_or_default();
                anthropic_messages.push(serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content,
                    }]
                }));
            }
            "assistant" => {
                let mut blocks = Vec::new();
                if let Some(text) = extract_text_content(msg) {
                    if !text.is_empty() {
                        blocks.push(serde_json::json!({"type": "text", "text": text}));
                    }
                }
                // Convert tool_calls to tool_use blocks
                if let Some(calls) = msg.get("tool_calls").and_then(|c| c.as_array()) {
                    for call in calls {
                        let id = call
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_owned();
                        let name = call
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            .unwrap_or("")
                            .to_owned();
                        let input = call
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(|a| a.as_str())
                            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                        blocks.push(serde_json::json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": input,
                        }));
                    }
                }
                if blocks.is_empty() {
                    blocks.push(serde_json::json!({"type": "text", "text": ""}));
                }
                anthropic_messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": blocks,
                }));
            }
            _ => {
                // user or other roles
                let text = extract_text_content(msg).unwrap_or_default();
                anthropic_messages.push(serde_json::json!({
                    "role": role,
                    "content": text,
                }));
            }
        }
    }

    let max_tokens = openai
        .get("max_tokens")
        .and_then(serde_json::Value::as_u64)
        .map_or(default_max_tokens, |v| {
            u32::try_from(v.min(u64::from(u32::MAX))).unwrap_or(default_max_tokens)
        });

    let mut result = serde_json::json!({
        "model": openai.get("model").cloned().unwrap_or(serde_json::Value::String(String::new())),
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
    });

    if !system_parts.is_empty() {
        let system_text = system_parts.join("\n");
        if let Some(obj) = result.as_object_mut() {
            obj.insert("system".to_owned(), serde_json::Value::String(system_text));
        }
    }

    // Forward compatible fields
    copy_optional_field(&openai, &mut result, "temperature");
    copy_optional_field(&openai, &mut result, "top_p");
    copy_optional_field(&openai, &mut result, "top_k");
    copy_optional_field(&openai, &mut result, "stream");

    // stop -> stop_sequences
    if let Some(stop) = openai.get("stop") {
        let stop_sequences = match stop {
            serde_json::Value::String(s) => {
                serde_json::Value::Array(vec![serde_json::Value::String(s.clone())])
            }
            serde_json::Value::Array(_) => stop.clone(),
            serde_json::Value::Null
            | serde_json::Value::Bool(_)
            | serde_json::Value::Number(_)
            | serde_json::Value::Object(_) => serde_json::Value::Null,
        };
        if !stop_sequences.is_null() {
            if let Some(obj) = result.as_object_mut() {
                obj.insert("stop_sequences".to_owned(), stop_sequences);
            }
        }
    }

    // tools translation (OpenAI -> Anthropic format)
    if let Some(tools) = openai.get("tools").and_then(|t| t.as_array()) {
        let anthropic_tools: Vec<serde_json::Value> = tools
            .iter()
            .filter_map(|tool| {
                let func = tool.get("function")?;
                Some(serde_json::json!({
                    "name": func.get("name")?,
                    "description": func.get("description").cloned().unwrap_or(serde_json::Value::String(String::new())),
                    "input_schema": func.get("parameters").cloned().unwrap_or(serde_json::json!({"type": "object"})),
                }))
            })
            .collect();
        if !anthropic_tools.is_empty() {
            if let Some(obj) = result.as_object_mut() {
                obj.insert(
                    "tools".to_owned(),
                    serde_json::Value::Array(anthropic_tools),
                );
            }
        }
    }

    serde_json::to_vec(&result)
        .map(Bytes::from)
        .map_err(|e| ServerError::InternalError(format!("Serialization error: {e}")))
}

/// Translate an Anthropic `/v1/messages` request body to `OpenAI` `/v1/chat/completions`.
#[allow(clippy::too_many_lines)]
pub fn anthropic_to_openai_request(body: &[u8]) -> Result<Bytes, ServerError> {
    let anthropic: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| ServerError::BadRequest(format!("Invalid JSON: {e}")))?;

    let mut openai_messages = Vec::new();

    // system field -> system message
    if let Some(system) = anthropic.get("system").and_then(|s| s.as_str()) {
        if !system.is_empty() {
            openai_messages.push(serde_json::json!({
                "role": "system",
                "content": system,
            }));
        }
    }

    if let Some(messages) = anthropic.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");

            match msg.get("content") {
                Some(serde_json::Value::String(text)) => {
                    openai_messages.push(serde_json::json!({
                        "role": role,
                        "content": text,
                    }));
                }
                Some(serde_json::Value::Array(blocks)) => {
                    let text = extract_text_from_blocks(blocks);
                    let tool_uses = extract_tool_uses(blocks);
                    let tool_results = extract_tool_results(blocks);

                    if !tool_results.is_empty() {
                        // Each tool result becomes a separate "tool" role message
                        for tr in &tool_results {
                            openai_messages.push(serde_json::json!({
                                "role": "tool",
                                "tool_call_id": tr.0,
                                "content": tr.1,
                            }));
                        }
                        // Preserve any text alongside tool results as a user message
                        if !text.is_empty() {
                            openai_messages.push(serde_json::json!({
                                "role": "user",
                                "content": text,
                            }));
                        }
                    } else if !tool_uses.is_empty() {
                        let mut msg_obj = serde_json::json!({
                            "role": "assistant",
                            "tool_calls": tool_uses,
                        });
                        if !text.is_empty() {
                            if let Some(obj) = msg_obj.as_object_mut() {
                                obj.insert("content".to_owned(), serde_json::Value::String(text));
                            }
                        }
                        openai_messages.push(msg_obj);
                    } else {
                        openai_messages.push(serde_json::json!({
                            "role": role,
                            "content": text,
                        }));
                    }
                }
                _ => {
                    openai_messages.push(serde_json::json!({
                        "role": role,
                        "content": "",
                    }));
                }
            }
        }
    }

    let mut result = serde_json::json!({
        "model": anthropic.get("model").cloned().unwrap_or(serde_json::Value::String(String::new())),
        "messages": openai_messages,
    });

    copy_optional_field(&anthropic, &mut result, "max_tokens");
    copy_optional_field(&anthropic, &mut result, "temperature");
    copy_optional_field(&anthropic, &mut result, "top_p");
    copy_optional_field(&anthropic, &mut result, "top_k");
    copy_optional_field(&anthropic, &mut result, "stream");

    // stop_sequences -> stop
    if let Some(stop_seqs) = anthropic.get("stop_sequences") {
        if let Some(obj) = result.as_object_mut() {
            obj.insert("stop".to_owned(), stop_seqs.clone());
        }
    }

    // tools translation (Anthropic -> OpenAI format)
    if let Some(tools) = anthropic.get("tools").and_then(|t| t.as_array()) {
        let openai_tools: Vec<serde_json::Value> = tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.get("name").cloned().unwrap_or(serde_json::Value::String(String::new())),
                        "description": tool.get("description").cloned().unwrap_or(serde_json::Value::String(String::new())),
                        "parameters": tool.get("input_schema").cloned().unwrap_or(serde_json::json!({"type": "object"})),
                    }
                })
            })
            .collect();
        if !openai_tools.is_empty() {
            if let Some(obj) = result.as_object_mut() {
                obj.insert("tools".to_owned(), serde_json::Value::Array(openai_tools));
            }
        }
    }

    serde_json::to_vec(&result)
        .map(Bytes::from)
        .map_err(|e| ServerError::InternalError(format!("Serialization error: {e}")))
}

// ---------------------------------------------------------------------------
// Non-streaming response translation
// ---------------------------------------------------------------------------

/// Translate an Anthropic response body to `OpenAI` format.
pub fn anthropic_response_to_openai(body: &[u8], model: &str) -> Result<Bytes, ServerError> {
    let resp: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| ServerError::ProxyError(format!("Failed to parse upstream response: {e}")))?;

    let id = resp
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("chatcmpl-proxy");

    let content_blocks = resp
        .get("content")
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();

    let text = content_blocks
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
        .collect::<Vec<_>>()
        .join("");

    let stop_reason = resp
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("stop");
    let finish_reason = anthropic_stop_to_openai_finish(stop_reason);

    let usage = resp.get("usage");
    let input_tokens = usage
        .and_then(|u| u.get("input_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .and_then(|u| u.get("output_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);

    // Extract tool_use blocks into tool_calls
    let tool_uses: Vec<serde_json::Value> = content_blocks
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
        .map(|b| {
            serde_json::json!({
                "id": b.get("id").cloned().unwrap_or(serde_json::Value::String(String::new())),
                "type": "function",
                "function": {
                    "name": b.get("name").cloned().unwrap_or(serde_json::Value::String(String::new())),
                    "arguments": b.get("input").map_or(String::new(), std::string::ToString::to_string),
                }
            })
        })
        .collect();

    let mut message = serde_json::json!({
        "role": "assistant",
    });
    if let Some(obj) = message.as_object_mut() {
        if !text.is_empty() || tool_uses.is_empty() {
            obj.insert("content".to_owned(), serde_json::Value::String(text));
        }
        if !tool_uses.is_empty() {
            obj.insert("tool_calls".to_owned(), serde_json::Value::Array(tool_uses));
        }
    }

    let final_finish = if message.get("tool_calls").is_some() && finish_reason == "stop" {
        "tool_calls".to_owned()
    } else {
        finish_reason
    };

    let result = serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": final_finish,
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    });

    serde_json::to_vec(&result)
        .map(Bytes::from)
        .map_err(|e| ServerError::InternalError(format!("Serialization error: {e}")))
}

/// Translate an `OpenAI` response body to Anthropic format.
pub fn openai_response_to_anthropic(body: &[u8], model: &str) -> Result<Bytes, ServerError> {
    let resp: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| ServerError::ProxyError(format!("Failed to parse upstream response: {e}")))?;

    let choice = resp
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first());

    let message = choice.and_then(|c| c.get("message"));
    let text = message
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    let finish_reason = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|f| f.as_str())
        .unwrap_or("stop");

    let stop_reason = openai_finish_to_anthropic_stop(finish_reason);

    let usage = resp.get("usage");
    let input_tokens = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);

    let mut content = vec![serde_json::json!({
        "type": "text",
        "text": text,
    })];

    // Convert tool_calls to tool_use blocks
    if let Some(calls) = message
        .and_then(|m| m.get("tool_calls"))
        .and_then(|c| c.as_array())
    {
        for call in calls {
            let id = call
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
            let name = call
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_owned();
            let input = call
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            content.push(serde_json::json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            }));
        }
    }

    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let result = serde_json::json!({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    });

    serde_json::to_vec(&result)
        .map(Bytes::from)
        .map_err(|e| ServerError::InternalError(format!("Serialization error: {e}")))
}

// ---------------------------------------------------------------------------
// Streaming response translation
// ---------------------------------------------------------------------------

/// Transform an Anthropic SSE stream into `OpenAI` SSE events.
#[allow(clippy::too_many_lines)]
pub fn anthropic_stream_to_openai(
    upstream: reqwest::Response,
    model: String,
) -> impl tokio_stream::Stream<Item = Result<Event, Infallible>> {
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    async_stream::stream! {
        // Role chunk
        let role_chunk = serde_json::json!({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": null,
            }]
        });
        yield Ok(Event::default().data(role_chunk.to_string()));

        let mut reader = SseReader::new(upstream);
        let mut tool_call_index: i64 = -1;
        let mut in_tool_use = false;

        while let Some((event_type, data)) = reader.next_event().await {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                match event_type.as_str() {
                    "content_block_start" => {
                        let block_type = json
                            .get("content_block")
                            .and_then(|b| b.get("type"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        if block_type == "tool_use" {
                            in_tool_use = true;
                            tool_call_index += 1;
                            let tool_id = json
                                .get("content_block")
                                .and_then(|b| b.get("id"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            let tool_name = json
                                .get("content_block")
                                .and_then(|b| b.get("name"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            let chunk = serde_json::json!({
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [{
                                            "index": tool_call_index,
                                            "id": tool_id,
                                            "type": "function",
                                            "function": {"name": tool_name, "arguments": ""},
                                        }]
                                    },
                                    "finish_reason": null,
                                }]
                            });
                            yield Ok(Event::default().data(chunk.to_string()));
                        } else {
                            in_tool_use = false;
                        }
                    }
                    "content_block_delta" => {
                        let delta = json.get("delta");
                        let delta_type = delta
                            .and_then(|d| d.get("type"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        if delta_type == "input_json_delta" && in_tool_use {
                            let partial_json = delta
                                .and_then(|d| d.get("partial_json"))
                                .and_then(|p| p.as_str())
                                .unwrap_or("");
                            if !partial_json.is_empty() {
                                let chunk = serde_json::json!({
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [{
                                                "index": tool_call_index,
                                                "function": {"arguments": partial_json},
                                            }]
                                        },
                                        "finish_reason": null,
                                    }]
                                });
                                yield Ok(Event::default().data(chunk.to_string()));
                            }
                        } else {
                            let text = delta
                                .and_then(|d| d.get("text"))
                                .and_then(|t| t.as_str())
                                .unwrap_or("");
                            if !text.is_empty() {
                                let chunk = serde_json::json!({
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": null,
                                    }]
                                });
                                yield Ok(Event::default().data(chunk.to_string()));
                            }
                        }
                    }
                    "content_block_stop" => {
                        in_tool_use = false;
                    }
                    "message_delta" => {
                        let stop_reason = json
                            .get("delta")
                            .and_then(|d| d.get("stop_reason"))
                            .and_then(|s| s.as_str())
                            .unwrap_or("end_turn");
                        let finish = anthropic_stop_to_openai_finish(stop_reason);
                        let chunk = serde_json::json!({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish,
                            }]
                        });
                        yield Ok(Event::default().data(chunk.to_string()));
                    }
                    _ => {}
                }
            }
        }

        yield Ok(Event::default().data("[DONE]"));
    }
}

/// Transform an `OpenAI` SSE stream into Anthropic SSE events.
#[allow(clippy::too_many_lines, clippy::map_entry)]
pub fn openai_stream_to_anthropic(
    upstream: reqwest::Response,
    model: String,
) -> impl tokio_stream::Stream<Item = Result<Event, Infallible>> {
    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());

    async_stream::stream! {
        // 1. message_start
        let start = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": null,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        });
        yield Ok(Event::default().event("message_start").data(start.to_string()));

        // 2. content_block_start for text
        let block_start = serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        });
        yield Ok(Event::default().event("content_block_start").data(block_start.to_string()));

        let mut reader = SseReader::new(upstream);
        let mut final_stop_reason = None;
        // Track which tool call indices we've already started (OpenAI index -> Anthropic block index)
        let mut started_tools: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        // Next Anthropic content block index (0 is text)
        let mut next_block_index: u64 = 1;
        let mut has_text = false;
        let mut text_block_closed = false;

        while let Some((_event_type, data)) = reader.next_event().await {
            if data == "[DONE]" {
                break;
            }
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                if let Some(chosen) = json
                    .get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|a| a.first())
                {
                    if let Some(text) = chosen
                        .get("delta")
                        .and_then(|d| d.get("content"))
                        .and_then(|c| c.as_str())
                    {
                        if !text.is_empty() {
                            has_text = true;
                            let delta = serde_json::json!({
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": text},
                            });
                            yield Ok(Event::default().event("content_block_delta").data(delta.to_string()));
                        }
                    }

                    // Handle tool_calls deltas
                    if let Some(tool_calls) = chosen
                        .get("delta")
                        .and_then(|d| d.get("tool_calls"))
                        .and_then(|tc| tc.as_array())
                    {
                        for tc in tool_calls {
                            let tc_index = tc.get("index").and_then(serde_json::Value::as_u64).unwrap_or(0);

                            if !started_tools.contains_key(&tc_index) {
                                // Close text block before first tool if we haven't emitted text
                                if !text_block_closed && !has_text && next_block_index == 1 {
                                    let block_stop = serde_json::json!({
                                        "type": "content_block_stop",
                                        "index": 0,
                                    });
                                    yield Ok(Event::default().event("content_block_stop").data(block_stop.to_string()));
                                    text_block_closed = true;
                                }

                                let block_idx = next_block_index;
                                next_block_index += 1;
                                started_tools.insert(tc_index, block_idx);

                                let tool_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_owned();
                                let tool_name = tc
                                    .get("function")
                                    .and_then(|f| f.get("name"))
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("")
                                    .to_owned();

                                let tool_start = serde_json::json!({
                                    "type": "content_block_start",
                                    "index": block_idx,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tool_id,
                                        "name": tool_name,
                                        "input": {},
                                    },
                                });
                                yield Ok(Event::default().event("content_block_start").data(tool_start.to_string()));
                            }

                            // Emit argument deltas
                            if let Some(args) = tc
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|a| a.as_str())
                            {
                                if !args.is_empty() {
                                    if let Some(&block_idx) = started_tools.get(&tc_index) {
                                        let delta = serde_json::json!({
                                            "type": "content_block_delta",
                                            "index": block_idx,
                                            "delta": {"type": "input_json_delta", "partial_json": args},
                                        });
                                        yield Ok(Event::default().event("content_block_delta").data(delta.to_string()));
                                    }
                                }
                            }
                        }
                    }

                    if let Some(reason) = chosen
                        .get("finish_reason")
                        .and_then(|f| f.as_str())
                    {
                        final_stop_reason = Some(openai_finish_to_anthropic_stop(reason));
                    }
                }
            }
        }

        // Close text block (index 0) if not already closed before tools
        if !text_block_closed {
            let block_stop = serde_json::json!({
                "type": "content_block_stop",
                "index": 0,
            });
            yield Ok(Event::default().event("content_block_stop").data(block_stop.to_string()));
        }

        // Close any open tool_use blocks
        for block_idx in started_tools.values() {
            let tool_stop = serde_json::json!({
                "type": "content_block_stop",
                "index": block_idx,
            });
            yield Ok(Event::default().event("content_block_stop").data(tool_stop.to_string()));
        }

        // message_delta
        let msg_delta = serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": final_stop_reason},
            "usage": {"input_tokens": 0, "output_tokens": 0},
        });
        yield Ok(Event::default().event("message_delta").data(msg_delta.to_string()));

        // message_stop
        let msg_stop = serde_json::json!({"type": "message_stop"});
        yield Ok(Event::default().event("message_stop").data(msg_stop.to_string()));
    }
}

// ---------------------------------------------------------------------------
// SSE reader helper
// ---------------------------------------------------------------------------

/// Minimal SSE parser that reads events from a reqwest byte stream.
struct SseReader {
    response: reqwest::Response,
    buffer: String,
}

impl SseReader {
    #[allow(clippy::missing_const_for_fn)]
    fn new(response: reqwest::Response) -> Self {
        Self {
            response,
            buffer: String::new(),
        }
    }

    /// Read the next SSE event, returning (`event_type`, data).
    async fn next_event(&mut self) -> Option<(String, String)> {
        loop {
            // Try to parse a complete event from the buffer
            if let Some(event) = self.try_parse_event() {
                return Some(event);
            }
            // Read more data
            match self.response.chunk().await {
                Ok(Some(chunk)) => {
                    let text = String::from_utf8_lossy(&chunk);
                    self.buffer.push_str(&text);
                }
                _ => {
                    // Try one last parse before returning None
                    return self.try_parse_event();
                }
            }
        }
    }

    fn try_parse_event(&mut self) -> Option<(String, String)> {
        // SSE events are separated by double newlines
        let separator = if self.buffer.contains("\n\n") {
            "\n\n"
        } else if self.buffer.contains("\r\n\r\n") {
            "\r\n\r\n"
        } else {
            return None;
        };

        let split_pos = self.buffer.find(separator)?;
        let event_text = self.buffer[..split_pos].to_owned();
        self.buffer = self.buffer[split_pos + separator.len()..].to_owned();

        let mut event_type = String::new();
        let mut data_parts = Vec::new();

        for line in event_text.lines() {
            if let Some(value) = line.strip_prefix("event:") {
                value.trim().clone_into(&mut event_type);
            } else if let Some(value) = line.strip_prefix("data:") {
                data_parts.push(value.trim().to_owned());
            }
        }

        if data_parts.is_empty() {
            return None;
        }

        let data = data_parts.join("\n");
        if event_type.is_empty() {
            "message".clone_into(&mut event_type);
        }

        Some((event_type, data))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_text_content(msg: &serde_json::Value) -> Option<String> {
    match msg.get("content") {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Array(parts)) => {
            let text: String = parts
                .iter()
                .filter_map(|p| {
                    if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                        p.get("text").and_then(|t| t.as_str()).map(str::to_owned)
                    } else {
                        None
                    }
                })
                .collect::<String>();
            Some(text)
        }
        _ => None,
    }
}

fn extract_text_from_blocks(blocks: &[serde_json::Value]) -> String {
    blocks
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
        .collect::<Vec<_>>()
        .join("")
}

fn extract_tool_uses(blocks: &[serde_json::Value]) -> Vec<serde_json::Value> {
    blocks
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
        .map(|b| {
            serde_json::json!({
                "id": b.get("id").cloned().unwrap_or(serde_json::Value::String(String::new())),
                "type": "function",
                "function": {
                    "name": b.get("name").cloned().unwrap_or(serde_json::Value::String(String::new())),
                    "arguments": b.get("input").map_or(String::new(), std::string::ToString::to_string),
                }
            })
        })
        .collect()
}

fn extract_tool_results(blocks: &[serde_json::Value]) -> Vec<(String, String)> {
    blocks
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_result"))
        .filter_map(|b| {
            let id = b.get("tool_use_id").and_then(|v| v.as_str())?.to_owned();
            let content = b
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_owned();
            Some((id, content))
        })
        .collect()
}

fn copy_optional_field(source: &serde_json::Value, target: &mut serde_json::Value, field: &str) {
    if let Some(value) = source.get(field) {
        if !value.is_null() {
            if let Some(obj) = target.as_object_mut() {
                obj.insert(field.to_owned(), value.clone());
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn openai_to_anthropic_basic_request() {
        let body = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 100
        });
        let result = openai_to_anthropic_request(body.to_string().as_bytes(), 1024).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["model"], "gpt-4");
        assert_eq!(parsed["max_tokens"], 100);
        assert_eq!(parsed["system"], "You are helpful.");
        assert_eq!(parsed["messages"].as_array().unwrap().len(), 1);
        assert_eq!(parsed["messages"][0]["role"], "user");
    }

    #[test]
    fn openai_to_anthropic_uses_default_max_tokens() {
        let body = serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let result = openai_to_anthropic_request(body.to_string().as_bytes(), 4096).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();
        assert_eq!(parsed["max_tokens"], 4096);
    }

    #[test]
    fn openai_to_anthropic_stop_becomes_stop_sequences() {
        let body = serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
            "stop": ["END", "STOP"]
        });
        let result = openai_to_anthropic_request(body.to_string().as_bytes(), 1024).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();
        assert_eq!(parsed["stop_sequences"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn openai_to_anthropic_tools_translation() {
        let body = serde_json::json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Weather?"}],
            "max_tokens": 100,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            }]
        });
        let result = openai_to_anthropic_request(body.to_string().as_bytes(), 1024).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();
        let tools = parsed["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "get_weather");
        assert!(tools[0].get("input_schema").is_some());
    }

    #[test]
    fn anthropic_to_openai_basic_request() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "Be concise.",
            "max_tokens": 100
        });
        let result = anthropic_to_openai_request(body.to_string().as_bytes()).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["model"], "claude-sonnet-4-6");
        assert_eq!(parsed["max_tokens"], 100);
        let messages = parsed["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "Be concise.");
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn anthropic_to_openai_stop_sequences_become_stop() {
        let body = serde_json::json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
            "stop_sequences": ["END"]
        });
        let result = anthropic_to_openai_request(body.to_string().as_bytes()).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();
        assert!(parsed["stop"].is_array());
    }

    #[test]
    fn anthropic_response_to_openai_basic() {
        let resp = serde_json::json!({
            "id": "msg_abc",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let result =
            anthropic_response_to_openai(resp.to_string().as_bytes(), "claude-sonnet-4-6").unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["object"], "chat.completion");
        assert_eq!(parsed["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
        assert_eq!(parsed["usage"]["prompt_tokens"], 10);
        assert_eq!(parsed["usage"]["completion_tokens"], 5);
    }

    #[test]
    fn openai_response_to_anthropic_basic() {
        let resp = serde_json::json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1_234_567_890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
        });
        let result = openai_response_to_anthropic(resp.to_string().as_bytes(), "gpt-4").unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["type"], "message");
        assert_eq!(parsed["role"], "assistant");
        assert_eq!(parsed["stop_reason"], "end_turn");
        assert_eq!(parsed["content"][0]["type"], "text");
        assert_eq!(parsed["content"][0]["text"], "Hi!");
        assert_eq!(parsed["usage"]["input_tokens"], 8);
        assert_eq!(parsed["usage"]["output_tokens"], 3);
    }

    #[test]
    fn anthropic_stop_reason_mapping() {
        assert_eq!(anthropic_stop_to_openai_finish("end_turn"), "stop");
        assert_eq!(anthropic_stop_to_openai_finish("max_tokens"), "length");
        assert_eq!(anthropic_stop_to_openai_finish("tool_use"), "tool_calls");
        assert_eq!(anthropic_stop_to_openai_finish("other"), "other");
    }

    #[test]
    fn openai_tool_message_to_anthropic_tool_result() {
        let body = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "72F sunny"}
            ],
            "max_tokens": 100
        });
        let result = openai_to_anthropic_request(body.to_string().as_bytes(), 1024).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();
        let messages = parsed["messages"].as_array().unwrap();

        // user, assistant with tool_use, user with tool_result
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["content"][0]["type"], "tool_use");
        assert_eq!(messages[2]["content"][0]["type"], "tool_result");
        assert_eq!(messages[2]["content"][0]["tool_use_id"], "call_1");
    }

    #[test]
    fn anthropic_tool_use_to_openai_tool_calls() {
        let resp = serde_json::json!({
            "id": "msg_abc",
            "type": "message",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "NYC"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let result = anthropic_response_to_openai(resp.to_string().as_bytes(), "claude").unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(parsed["choices"][0]["message"]["content"], "Let me check.");
        let tool_calls = parsed["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["function"]["name"], "get_weather");
    }

    #[test]
    fn multiple_system_messages_joined() {
        let body = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"}
            ],
            "max_tokens": 100
        });
        let result = openai_to_anthropic_request(body.to_string().as_bytes(), 1024).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();
        assert_eq!(parsed["system"], "Be helpful.\nBe concise.");
    }
}
