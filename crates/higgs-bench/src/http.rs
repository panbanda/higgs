//! Shared HTTP helpers used by multiple bench binaries.
//!
//! These wrap two recurring patterns: a streaming `POST
//! /v1/chat/completions` that captures TTFT + token count, and a
//! `GET /v1/models` discovery probe.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

/// One streamed chat completion result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamResult {
    /// Time from request send to the first non-empty `delta.content` SSE
    /// chunk, in milliseconds.
    pub ttft_ms: f64,
    /// Wall-clock time of the full streaming exchange, request send to
    /// last byte read, in milliseconds.
    pub total_ms: f64,
    /// Concatenated `delta.content` text across all SSE chunks.
    pub output: String,
    /// Number of non-empty content SSE chunks observed (a coarse
    /// fallback for token count when `completion_tokens` is `None`).
    pub num_tokens: u32,
    /// Server-reported prompt token count from the terminal `usage`
    /// chunk. `None` when the server omits `usage`.
    pub prompt_tokens: Option<u32>,
    /// Server-reported completion token count from the terminal `usage`
    /// chunk. `None` when the server omits `usage`. Prefer this over
    /// `num_tokens` for throughput math.
    pub completion_tokens: Option<u32>,
}

/// One non-streaming chat completion plus its wall time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResult {
    /// `choices[0].message.content` from the response body.
    pub content: String,
    /// Wall-clock time of the request, including body deserialization,
    /// in milliseconds.
    pub elapsed_ms: f64,
    /// Server-reported prompt token count from `usage.prompt_tokens`.
    pub prompt_tokens: u32,
    /// Server-reported completion token count from
    /// `usage.completion_tokens`.
    pub completion_tokens: u32,
}

/// Drives a streaming chat completion against a higgs server. Mirrors the
/// `stream_chat` helper used in the original Python benches: measure time
/// to first non-empty `delta.content`, then count the rest.
///
/// `messages` is `[{role, content}, ...]`. `max_tokens` and `temperature`
/// are passed through as-is.
pub async fn stream_chat(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    messages: &serde_json::Value,
    max_tokens: u32,
    temperature: f32,
) -> Result<StreamResult> {
    // `stream_options.include_usage` makes Higgs emit a terminal SSE
    // chunk carrying `usage: {prompt_tokens, completion_tokens, ...}`
    // and an empty `choices` array. Without it the streamed response
    // has no usage and prefill/decode tok/s collapse to 0 or to a
    // chunk-count estimate once the server emits buffered text.
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    let url = format!("{base_url}/v1/chat/completions");
    let started = Instant::now();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("{url} returned HTTP {status}: {text}");
    }

    let mut stream = resp.bytes_stream();
    let mut first_token_at: Option<Instant> = None;
    let mut tokens: Vec<String> = Vec::new();
    let mut prompt_tokens: Option<u32> = None;
    let mut completion_tokens: Option<u32> = None;
    let mut buf = String::new();

    while let Some(chunk_res) = stream.next().await {
        let bytes = chunk_res.context("read SSE chunk")?;
        buf.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(idx) = buf.find('\n') {
            let raw: String = buf.drain(..=idx).collect();
            let line = raw.trim();
            if line.is_empty() || !line.starts_with("data:") {
                continue;
            }
            let data = line.trim_start_matches("data:").trim();
            if data == "[DONE]" {
                continue;
            }
            let value: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(content) = value
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|s| s.as_str())
            {
                if !content.is_empty() {
                    if first_token_at.is_none() {
                        first_token_at = Some(Instant::now());
                    }
                    tokens.push(content.to_owned());
                }
            }
            if let Some(usage) = value.get("usage") {
                if let Some(p) = usage
                    .get("prompt_tokens")
                    .and_then(serde_json::Value::as_u64)
                {
                    prompt_tokens = Some(p as u32);
                }
                if let Some(c) = usage
                    .get("completion_tokens")
                    .and_then(serde_json::Value::as_u64)
                {
                    completion_tokens = Some(c as u32);
                }
            }
        }
    }

    let total_ms = started.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = first_token_at.map_or(total_ms, |t| {
        t.duration_since(started).as_secs_f64() * 1000.0
    });
    Ok(StreamResult {
        ttft_ms,
        total_ms,
        output: tokens.join(""),
        num_tokens: tokens.len() as u32,
        prompt_tokens,
        completion_tokens,
    })
}

/// Drives a non-streaming chat completion against a higgs server.
pub async fn chat(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    messages: &serde_json::Value,
    max_tokens: u32,
    temperature: f32,
) -> Result<ChatResult> {
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    });
    let url = format!("{base_url}/v1/chat/completions");
    let started = Instant::now();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("{url} returned HTTP {status}: {text}");
    }
    // `reqwest::Response::json` consumes the body lazily — the network
    // read happens here. Measure elapsed *after* deserialization so the
    // reported wall time matches what a caller actually waited for.
    let value: serde_json::Value = resp.json().await.context("decode chat response JSON")?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let content = value
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|s| s.as_str())
        .unwrap_or_default()
        .to_owned();
    let usage = value.get("usage");
    let prompt_tokens = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as u32;
    let completion_tokens = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as u32;
    Ok(ChatResult {
        content,
        elapsed_ms,
        prompt_tokens,
        completion_tokens,
    })
}

/// Returns the first model id reported by `GET /v1/models`.
pub async fn first_model_id(client: &reqwest::Client, base_url: &str) -> Result<String> {
    first_model_id_with_auth(client, base_url, None).await
}

/// Like `first_model_id`, but optionally sends an `Authorization: Bearer`
/// header. Used by the oMLX backend in `bench_h2h`.
pub async fn first_model_id_with_auth(
    client: &reqwest::Client,
    base_url: &str,
    api_key: Option<&str>,
) -> Result<String> {
    let url = format!("{base_url}/v1/models");
    let mut req = client.get(&url).timeout(Duration::from_secs(5));
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await.with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("{url} returned HTTP {}", resp.status());
    }
    let value: serde_json::Value = resp.json().await.context("decode /v1/models JSON")?;
    value
        .get("data")
        .and_then(|d| d.get(0))
        .and_then(|m| m.get("id"))
        .and_then(|s| s.as_str())
        .map(std::string::ToString::to_string)
        .ok_or_else(|| anyhow::anyhow!("no models reported by {url}"))
}

/// One streamed chat completion result with TTFT, decode tok/s, and the
/// flag tracking whether the server reported `usage.completion_tokens` or
/// we estimated it from the streamed character count.
///
/// Mirrors `bench_mlx_tuning.py::stream_chat` and `bench_h2h.py::stream_chat`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetrics {
    pub ttft_ms: f64,
    pub decode_tps: f64,
    pub total_ms: f64,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub completion_tokens_estimated: bool,
    pub output: String,
}

/// Optional knobs for `stream_chat_metrics`.
#[derive(Debug, Default, Clone)]
pub struct StreamChatOptions<'a> {
    pub max_tokens: u32,
    pub temperature: f32,
    pub api_key: Option<&'a str>,
    pub response_format: Option<serde_json::Value>,
    pub estimate_prompt_tokens: bool,
    /// Ask the server to emit a terminal `usage` chunk by sending
    /// `stream_options.include_usage = true`. Higgs honors this; some
    /// other backends (oMLX, in particular) reject unknown keys, so leave
    /// disabled for those and rely on the character-count estimate.
    pub include_usage: bool,
}

/// Drives a streaming chat completion and returns ttft + decode-tok/s.
///
/// Same shape as the Python `stream_chat` helpers used by `bench_h2h`
/// and `bench_mlx_tuning`. If `usage.completion_tokens` is absent the
/// completion-token count is estimated from the streamed character count
/// (`max(1, output_chars / 3.5)`); the flag is preserved on the result.
#[allow(clippy::too_many_lines)]
pub async fn stream_chat_metrics(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    messages: &serde_json::Value,
    opts: &StreamChatOptions<'_>,
) -> Result<StreamMetrics> {
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": opts.max_tokens,
        "temperature": opts.temperature,
        "stream": true,
    });
    if let Some(rf) = &opts.response_format {
        if let Some(map) = body.as_object_mut() {
            map.insert("response_format".to_owned(), rf.clone());
        }
    }
    if opts.include_usage {
        if let Some(map) = body.as_object_mut() {
            map.insert(
                "stream_options".to_owned(),
                serde_json::json!({"include_usage": true}),
            );
        }
    }

    let url = format!("{base_url}/v1/chat/completions");
    let mut req = client.post(&url).json(&body);
    if let Some(key) = opts.api_key {
        req = req.bearer_auth(key);
    }
    let started = Instant::now();
    let resp = req.send().await.with_context(|| format!("POST {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("{url} returned HTTP {status}: {text}");
    }

    let mut stream = resp.bytes_stream();
    let mut first_token_at: Option<Instant> = None;
    let mut tokens: Vec<String> = Vec::new();
    let mut prompt_tokens: u32 = 0;
    let mut completion_tokens: u32 = 0;
    let mut buf = String::new();

    while let Some(chunk_res) = stream.next().await {
        let bytes = chunk_res.context("read SSE chunk")?;
        buf.push_str(&String::from_utf8_lossy(&bytes));
        while let Some(idx) = buf.find('\n') {
            let raw: String = buf.drain(..=idx).collect();
            let line = raw.trim();
            if line.is_empty() || !line.starts_with("data:") {
                continue;
            }
            let data = line.trim_start_matches("data:").trim();
            if data == "[DONE]" {
                continue;
            }
            let value: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(content) = value
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|s| s.as_str())
            {
                if !content.is_empty() {
                    if first_token_at.is_none() {
                        first_token_at = Some(Instant::now());
                    }
                    tokens.push(content.to_owned());
                }
            }
            if let Some(usage) = value.get("usage") {
                if let Some(p) = usage
                    .get("prompt_tokens")
                    .and_then(serde_json::Value::as_u64)
                {
                    prompt_tokens = p as u32;
                }
                if let Some(c) = usage
                    .get("completion_tokens")
                    .and_then(serde_json::Value::as_u64)
                {
                    completion_tokens = c as u32;
                }
            }
        }
    }

    let ended = Instant::now();
    let first_token = first_token_at.unwrap_or(ended);
    let ttft_ms = first_token.duration_since(started).as_secs_f64() * 1000.0;
    let decode_s = ended.duration_since(first_token).as_secs_f64().max(0.001);

    let mut completion_tokens_estimated = false;
    if completion_tokens == 0 {
        let output_chars: usize = tokens.iter().map(String::len).sum();
        if output_chars > 0 {
            completion_tokens = (output_chars as f64 / 3.5).max(1.0) as u32;
            completion_tokens_estimated = true;
        }
    }

    if opts.estimate_prompt_tokens && prompt_tokens == 0 {
        let prompt_chars = messages
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
                    .map(str::len)
                    .sum::<usize>()
            })
            .unwrap_or_default();
        if prompt_chars > 0 {
            prompt_tokens = (prompt_chars as f64 / 3.5).max(1.0) as u32;
        }
    }

    let decode_tps = if completion_tokens > 0 {
        f64::from(completion_tokens.saturating_sub(1)) / decode_s
    } else {
        0.0
    };
    let total_ms = ended.duration_since(started).as_secs_f64() * 1000.0;

    Ok(StreamMetrics {
        ttft_ms,
        decode_tps,
        total_ms,
        prompt_tokens,
        completion_tokens,
        completion_tokens_estimated,
        output: tokens.join(""),
    })
}

/// Non-streaming chat with optional `response_format` and `api_key`.
pub async fn chat_with_options(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    messages: &serde_json::Value,
    max_tokens: u32,
    temperature: f32,
    response_format: Option<&serde_json::Value>,
) -> Result<ChatResult> {
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    });
    if let Some(rf) = response_format {
        if let Some(map) = body.as_object_mut() {
            map.insert("response_format".to_owned(), rf.clone());
        }
    }
    let url = format!("{base_url}/v1/chat/completions");
    let started = Instant::now();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("{url} returned HTTP {status}: {text}");
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let value: serde_json::Value = resp.json().await.context("decode chat response JSON")?;
    let content = value
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|s| s.as_str())
        .unwrap_or_default()
        .to_owned();
    let usage = value.get("usage");
    let prompt_tokens = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as u32;
    let completion_tokens = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as u32;
    Ok(ChatResult {
        content,
        elapsed_ms,
        prompt_tokens,
        completion_tokens,
    })
}
