//! HTTP client for any OpenAI-compatible chat-completions server.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use serde_json::{Map, Value, json};

use crate::compare::{
    StreamAccumulator, StreamOutcome, TrialMetrics, UsageReport, build_request_body,
};

/// Everything needed to talk to one model on one server.
#[derive(Debug, Clone)]
pub struct EndpointSettings {
    /// Server root such as `http://127.0.0.1:8870`, without a trailing path.
    pub base_url: String,
    /// Value of the request `model` field.
    pub model: String,
    pub api_key: Option<String>,
    pub temperature: f32,
    /// Merged into every streamed benchmark request body; its keys override defaults.
    pub extra_body: Map<String, Value>,
    pub timeout: Duration,
}

pub struct Endpoint {
    client: reqwest::Client,
    base_url: String,
    chat_url: String,
    api_key: Option<String>,
    model: String,
    temperature: f32,
    extra_body: Map<String, Value>,
}

/// One tool call of a finished reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    /// The `arguments` string exactly as the server sent it.
    pub arguments: String,
}

/// The logprob data of one generated token.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    pub token: String,
    /// The bytes the token adds to the output. Servers that report `bytes` give the exact text
    /// in context; without it, the token string is the best available.
    pub bytes: Vec<u8>,
    /// Logprobs of the most likely alternatives, highest first.
    pub top: Vec<f64>,
}

/// A non-streaming reply.
#[derive(Debug, Clone, PartialEq)]
pub struct Completion {
    pub content: String,
    pub reasoning: String,
    pub finish_reason: Option<String>,
    pub usage: UsageReport,
    pub tool_calls: Vec<ToolCall>,
    /// Per-token logprobs; empty unless the request asked for them and the server sent them.
    pub token_logprobs: Vec<TokenLogprob>,
}

impl Completion {
    /// Reasoning followed by visible content: everything the model produced as text.
    pub fn full_text(&self) -> String {
        [self.reasoning.as_str(), self.content.as_str()].concat()
    }
}

/// Reads a non-streaming chat-completions response body.
pub fn parse_completion(body: &Value) -> Result<Completion> {
    if let Some(error) = body.get("error").filter(|error| !error.is_null()) {
        bail!("server returned an error: {error}");
    }
    let message = body
        .pointer("/choices/0/message")
        .context("response has no choices[0].message")?;
    let text_of = |key: &str| {
        message
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    let reasoning = ["reasoning_content", "reasoning"]
        .iter()
        .map(|key| text_of(key))
        .collect::<String>();
    let tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .map(|call| ToolCall {
                    name: call
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    arguments: call
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                })
                .collect()
        })
        .unwrap_or_default();
    let token_logprobs = body
        .pointer("/choices/0/logprobs/content")
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .map(|entry| {
                    let mut top: Vec<f64> = entry
                        .get("top_logprobs")
                        .and_then(Value::as_array)
                        .map(|alternatives| {
                            alternatives
                                .iter()
                                .filter_map(|alternative| {
                                    alternative.get("logprob").and_then(Value::as_f64)
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    top.sort_by(|left, right| right.total_cmp(left));
                    let token = entry
                        .get("token")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    let bytes = entry.get("bytes").and_then(Value::as_array).map_or_else(
                        || token.clone().into_bytes(),
                        |values| {
                            values
                                .iter()
                                .filter_map(|value| {
                                    value.as_u64().and_then(|byte| u8::try_from(byte).ok())
                                })
                                .collect()
                        },
                    );
                    TokenLogprob { token, bytes, top }
                })
                .collect()
        })
        .unwrap_or_default();
    let count = |pointer: &str| {
        body.pointer(pointer)
            .and_then(Value::as_u64)
            .map(|raw| u32::try_from(raw).unwrap_or(u32::MAX))
    };
    Ok(Completion {
        content: text_of("content"),
        reasoning,
        finish_reason: body
            .pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .map(str::to_owned),
        usage: UsageReport {
            completion_tokens: count("/usage/completion_tokens"),
            prompt_tokens: count("/usage/prompt_tokens"),
            cached_tokens: count("/usage/prompt_tokens_details/cached_tokens"),
        },
        tool_calls,
        token_logprobs,
    })
}

impl Endpoint {
    pub fn new(settings: EndpointSettings) -> Result<Self> {
        let base_url = settings.base_url.trim_end_matches('/').to_owned();
        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(settings.timeout)
                .build()
                .context("build HTTP client")?,
            chat_url: format!("{base_url}/v1/chat/completions"),
            base_url,
            api_key: settings.api_key,
            model: settings.model,
            temperature: settings.temperature,
            extra_body: settings.extra_body,
        })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// The same endpoint addressing a different model.
    #[must_use]
    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    /// The ids listed by `GET /v1/models`.
    pub async fn list_model_ids(&self) -> Result<Vec<String>> {
        let listing = self.get_json("/v1/models").await?;
        Ok(listing
            .get("data")
            .and_then(Value::as_array)
            .map(|models| {
                models
                    .iter()
                    .filter_map(|model| model.get("id").and_then(Value::as_str))
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default())
    }

    fn post(&self, body: &Value) -> reqwest::RequestBuilder {
        let request = self.client.post(&self.chat_url).json(body);
        match &self.api_key {
            Some(key) => request.bearer_auth(key),
            None => request,
        }
    }

    /// `GET <base_url><path>` parsed as JSON.
    pub async fn get_json(&self, path: &str) -> Result<Value> {
        let url = format!("{}{path}", self.base_url);
        let mut request = self.client.get(&url);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.with_context(|| format!("GET {url}"))?;
        let status = response.status();
        if !status.is_success() {
            bail!("{url} returned HTTP {status}");
        }
        response
            .json()
            .await
            .with_context(|| format!("{url} did not return JSON"))
    }

    /// A streaming request for `messages` with the endpoint's temperature and extra body.
    pub async fn stream_chat(&self, messages: &Value, max_tokens: u32) -> Result<StreamOutcome> {
        let body = build_request_body(
            &self.model,
            messages,
            max_tokens,
            self.temperature,
            &self.extra_body,
        );
        self.stream_body(&body).await
    }

    pub async fn stream_user_prompt(&self, prompt: &str, max_tokens: u32) -> Result<TrialMetrics> {
        let messages = json!([{ "role": "user", "content": prompt }]);
        Ok(self.stream_chat(&messages, max_tokens).await?.metrics)
    }

    /// Sends `body` as given and digests the SSE reply.
    pub async fn stream_body(&self, body: &Value) -> Result<StreamOutcome> {
        let started = Instant::now();
        let response = self
            .post(body)
            .send()
            .await
            .with_context(|| format!("POST {}", self.chat_url))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            bail!("{} returned HTTP {status}: {text}", self.chat_url);
        }

        let mut accumulator = StreamAccumulator::default();
        let mut stream = response.bytes_stream();
        let mut pending: Vec<u8> = Vec::new();
        while let Some(chunk) = stream.next().await {
            pending.extend_from_slice(&chunk.context("read SSE chunk")?);
            while let Some(newline) = pending.iter().position(|&byte| byte == b'\n') {
                let line: Vec<u8> = pending.drain(..=newline).collect();
                accumulator.ingest_line(&line, started.elapsed());
            }
        }
        if !pending.is_empty() {
            accumulator.ingest_line(&pending, started.elapsed());
        }
        accumulator.finish(started.elapsed())
    }

    /// Sends `body` as given with `stream` forced off and parses the reply.
    pub async fn complete_body(&self, body: &Value) -> Result<Completion> {
        let mut non_streaming = body.clone();
        if let Some(object) = non_streaming.as_object_mut() {
            object.insert("stream".to_owned(), Value::Bool(false));
        }
        let response = self
            .post(&non_streaming)
            .send()
            .await
            .with_context(|| format!("POST {}", self.chat_url))?;
        let status = response.status();
        let text = response.text().await.context("read response body")?;
        if !status.is_success() {
            bail!("{} returned HTTP {status}: {text}", self.chat_url);
        }
        let parsed: Value = serde_json::from_str(&text).context("response is not JSON")?;
        parse_completion(&parsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_text_usage_and_finish_reason() -> Result<()> {
        let completion = parse_completion(&json!({
            "choices": [{
                "message": {"content": "hi", "reasoning_content": "hm "},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 32}
            }
        }))?;
        assert_eq!(completion.content, "hi");
        assert_eq!(completion.full_text(), "hm hi");
        assert_eq!(completion.finish_reason.as_deref(), Some("stop"));
        assert_eq!(completion.usage.cached_tokens, Some(32));
        assert_eq!(completion.usage.prompt_tokens, Some(50));
        assert!(completion.tool_calls.is_empty());
        Ok(())
    }

    #[test]
    fn parses_tool_calls_and_a_null_content() -> Result<()> {
        let completion = parse_completion(&json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }))?;
        assert!(completion.content.is_empty());
        assert_eq!(
            completion.tool_calls,
            [ToolCall {
                name: "get_weather".to_owned(),
                arguments: "{\"city\":\"Paris\"}".to_owned()
            }]
        );
        assert_eq!(completion.usage, UsageReport::default());
        Ok(())
    }

    #[test]
    fn parses_token_logprobs_with_alternatives_sorted() -> Result<()> {
        let completion = parse_completion(&json!({
            "choices": [{
                "message": {"content": "ab"},
                "logprobs": {"content": [
                    {"token": "a", "logprob": -0.1, "top_logprobs": [
                        {"token": "b", "logprob": -2.0}, {"token": "a", "logprob": -0.1}]},
                    {"token": "b", "logprob": -0.3, "top_logprobs": []}
                ]}
            }]
        }))?;
        assert_eq!(completion.token_logprobs.len(), 2);
        let tokens = &completion.token_logprobs;
        assert_eq!(
            tokens.first().map(|token| token.top.clone()),
            Some(vec![-0.1, -2.0])
        );
        assert_eq!(tokens.get(1).map(|token| token.token.as_str()), Some("b"));
        Ok(())
    }

    #[test]
    fn rejects_error_bodies_and_empty_choices() {
        assert!(parse_completion(&json!({"error": {"message": "boom"}})).is_err());
        assert!(parse_completion(&json!({"choices": []})).is_err());
    }

    #[test]
    fn logprob_bytes_are_read_and_default_to_the_token_text() -> Result<()> {
        let completion = parse_completion(&json!({
            "choices": [{
                "message": {"content": " ab"},
                "logprobs": {"content": [
                    {"token": "ab", "bytes": [32, 97, 98], "top_logprobs": []},
                    {"token": "c", "top_logprobs": []}
                ]}
            }]
        }))?;
        let bytes: Vec<&[u8]> = completion
            .token_logprobs
            .iter()
            .map(|token| token.bytes.as_slice())
            .collect();
        assert_eq!(bytes, [b" ab".as_slice(), b"c".as_slice()]);
        Ok(())
    }
}
