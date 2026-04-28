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
    pub ttft_ms: f64,
    pub total_ms: f64,
    pub output: String,
    pub num_tokens: u32,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
}

/// One non-streaming chat completion plus its wall time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResult {
    pub content: String,
    pub elapsed_ms: f64,
    pub prompt_tokens: u32,
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
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": true,
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

/// Returns the first model id reported by `GET /v1/models`.
pub async fn first_model_id(client: &reqwest::Client, base_url: &str) -> Result<String> {
    let url = format!("{base_url}/v1/models");
    let resp = client
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
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
