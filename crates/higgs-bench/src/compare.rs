//! Server-agnostic measurement logic for `bench_compare`.
//!
//! Everything here is pure (no I/O) so the identical code path measures every
//! server under comparison and can be unit tested without a live endpoint.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::stats;

/// Characters per token assumed when sizing synthetic prompts. The server's
/// reported `prompt_tokens` is always the ground truth.
const APPROX_CHARS_PER_TOKEN: usize = 4;

/// Sentences per paragraph in synthetic prompt bodies.
const SENTENCES_PER_PARAGRAPH: usize = 6;

const LONG_OUTPUT_REQUEST: &str = "Write a long, detailed, multi-section essay on the history and \
engineering of the transistor, from early vacuum tubes to modern nanometer-scale processes. Use at \
least eight numbered sections, each with several full paragraphs, and do not stop early or \
summarize.";

const ADJECTIVES: [&str; 20] = [
    "quiet",
    "ancient",
    "bright",
    "narrow",
    "careful",
    "restless",
    "gentle",
    "hollow",
    "distant",
    "stubborn",
    "crooked",
    "patient",
    "weathered",
    "luminous",
    "brittle",
    "frugal",
    "curious",
    "silent",
    "tangled",
    "steady",
];

const NOUNS: [&str; 24] = [
    "lighthouse",
    "merchant",
    "river",
    "library",
    "engine",
    "orchard",
    "bridge",
    "traveler",
    "workshop",
    "harbor",
    "ledger",
    "mountain",
    "signal",
    "garden",
    "carpenter",
    "village",
    "compass",
    "furnace",
    "tower",
    "market",
    "letter",
    "bell",
    "valley",
    "machine",
];

const VERBS: [&str; 20] = [
    "followed",
    "measured",
    "forgot",
    "repaired",
    "watched",
    "carried",
    "described",
    "avoided",
    "remembered",
    "crossed",
    "questioned",
    "polished",
    "gathered",
    "abandoned",
    "inspected",
    "traded",
    "mapped",
    "guarded",
    "borrowed",
    "rebuilt",
];

const ADVERBS: [&str; 12] = [
    "slowly",
    "twice",
    "at dawn",
    "without warning",
    "every winter",
    "in secret",
    "by lantern light",
    "after the storm",
    "with great care",
    "before noon",
    "against the wind",
    "for years",
];

/// Token counts the server reported in the terminal `usage` object.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UsageReport {
    pub completion_tokens: Option<u32>,
    pub prompt_tokens: Option<u32>,
    pub cached_tokens: Option<u32>,
}

impl UsageReport {
    fn merge(&mut self, later: Self) {
        self.completion_tokens = later.completion_tokens.or(self.completion_tokens);
        self.prompt_tokens = later.prompt_tokens.or(self.prompt_tokens);
        self.cached_tokens = later.cached_tokens.or(self.cached_tokens);
    }
}

/// What one SSE `data:` line contributed to the stream.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ParsedChunk {
    /// The delta carried visible content, reasoning text, or a tool call.
    pub carries_token: bool,
    /// Visible `content` text of the delta (reasoning is excluded).
    pub content: String,
    pub usage: Option<UsageReport>,
    pub error: Option<String>,
}

fn to_u32(value: &Value) -> Option<u32> {
    value
        .as_u64()
        .map(|raw| u32::try_from(raw).unwrap_or(u32::MAX))
}

fn parse_usage(usage: &Value) -> UsageReport {
    UsageReport {
        completion_tokens: usage.get("completion_tokens").and_then(to_u32),
        prompt_tokens: usage.get("prompt_tokens").and_then(to_u32),
        cached_tokens: usage
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(to_u32),
    }
}

/// Parses one raw SSE line. Returns `None` for blank lines, comments,
/// `[DONE]`, and anything that is not a JSON `data:` payload.
pub fn parse_sse_line(line: &[u8]) -> Option<ParsedChunk> {
    let text = String::from_utf8_lossy(line);
    let data = text.trim().strip_prefix("data:")?.trim();
    if data == "[DONE]" {
        return None;
    }
    let value: Value = serde_json::from_str(data).ok()?;

    let delta = value.pointer("/choices/0/delta");
    let text_field = |key: &str| {
        delta
            .and_then(|d| d.get(key))
            .and_then(Value::as_str)
            .unwrap_or("")
    };
    let content = text_field("content");
    let has_tool_calls = delta
        .and_then(|d| d.get("tool_calls"))
        .and_then(Value::as_array)
        .is_some_and(|calls| !calls.is_empty());
    let carries_token = !content.is_empty()
        || !text_field("reasoning_content").is_empty()
        || !text_field("reasoning").is_empty()
        || has_tool_calls;

    Some(ParsedChunk {
        carries_token,
        content: content.to_owned(),
        usage: value
            .get("usage")
            .filter(|usage| usage.is_object())
            .map(parse_usage),
        error: value.get("error").filter(|e| !e.is_null()).map(|e| {
            e.get("message")
                .and_then(Value::as_str)
                .map_or_else(|| e.to_string(), str::to_owned)
        }),
    })
}

/// Raw measurements and derived rates for one streamed request.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrialMetrics {
    pub ttft_ms: f64,
    /// `None` when fewer than two tokens were generated or the decode window
    /// was empty, so a degenerate trial is never reported as a number.
    pub decode_tokps: Option<f64>,
    pub e2e_tokps: f64,
    pub completion_tokens: u32,
    pub prompt_tokens: Option<u32>,
    pub cached_tokens: Option<u32>,
    pub wall_ms: f64,
    /// Token counts were estimated from delta chunks because the server sent
    /// no terminal `usage`; treat the rates as less trustworthy.
    pub usage_missing: bool,
}

/// Decode rate: tokens after the first, divided by first-token-to-end time.
pub fn decode_tokps(completion_tokens: u32, decode_window: Duration) -> Option<f64> {
    let window = decode_window.as_secs_f64();
    (completion_tokens >= 2 && window > 0.0).then(|| f64::from(completion_tokens - 1) / window)
}

/// Prefill rate derived from time to first token. Includes generation of the
/// first token, so it slightly understates pure prefill speed.
pub fn prefill_tokps(prompt_tokens: u32, ttft_ms: f64) -> Option<f64> {
    (ttft_ms > 0.0).then(|| f64::from(prompt_tokens) / (ttft_ms / 1000.0))
}

/// A finished stream: its metrics plus the visible reply text.
#[derive(Debug, Clone)]
pub struct StreamOutcome {
    pub metrics: TrialMetrics,
    pub text: String,
}

/// Incrementally digests SSE lines, stamping the first token-bearing delta.
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    first_token_offset: Option<Duration>,
    token_chunks: u32,
    usage: UsageReport,
    text: String,
    error: Option<String>,
}

impl StreamAccumulator {
    /// `elapsed` is the time since the request was sent.
    pub fn ingest_line(&mut self, line: &[u8], elapsed: Duration) {
        let Some(chunk) = parse_sse_line(line) else {
            return;
        };
        if chunk.carries_token {
            self.first_token_offset.get_or_insert(elapsed);
            self.token_chunks += 1;
        }
        self.text.push_str(&chunk.content);
        if let Some(usage) = chunk.usage {
            self.usage.merge(usage);
        }
        if chunk.error.is_some() {
            self.error = chunk.error;
        }
    }

    /// `total` is the time from request send to end of stream.
    pub fn finish(self, total: Duration) -> Result<StreamOutcome> {
        if let Some(message) = self.error {
            bail!("server reported an error mid-stream: {message}");
        }
        let first_token_offset = self
            .first_token_offset
            .context("no token-bearing delta received before the stream ended")?;
        let usage_missing = self.usage.completion_tokens.is_none();
        let completion_tokens = self.usage.completion_tokens.unwrap_or(self.token_chunks);
        if completion_tokens == 0 {
            bail!("server reported zero completion tokens");
        }

        let total_secs = total.as_secs_f64();
        let metrics = TrialMetrics {
            ttft_ms: first_token_offset.as_secs_f64() * 1000.0,
            decode_tokps: decode_tokps(completion_tokens, total.saturating_sub(first_token_offset)),
            e2e_tokps: if total_secs > 0.0 {
                f64::from(completion_tokens) / total_secs
            } else {
                0.0
            },
            completion_tokens,
            prompt_tokens: self.usage.prompt_tokens,
            cached_tokens: self.usage.cached_tokens,
            wall_ms: total_secs * 1000.0,
            usage_missing,
        };
        Ok(StreamOutcome {
            metrics,
            text: self.text,
        })
    }
}

/// Distribution summary using the shared stats helpers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Summary {
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub stdev: f64,
}

impl Summary {
    pub fn of(values: &[f64]) -> Self {
        Self {
            mean: stats::mean(values),
            median: stats::median(values),
            p95: stats::p95(values),
            stdev: stats::stdev(values),
        }
    }
}

const fn next_random(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn pick<'a>(words: &[&'a str], state: &mut u64) -> &'a str {
    let index = usize::try_from(next_random(state) % words.len() as u64).unwrap_or(0);
    words.get(index).copied().unwrap_or("")
}

/// Deterministic natural-language-like text of roughly `target_tokens` tokens
/// (assuming about four characters per token).
pub fn build_prompt_body(target_tokens: usize) -> String {
    let target_chars = target_tokens * APPROX_CHARS_PER_TOKEN;
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut body = String::with_capacity(target_chars + 128);
    let mut sentence_index = 0;
    while body.len() < target_chars {
        let sentence = format!(
            "The {} {} {} the {} {} {}.",
            pick(&ADJECTIVES, &mut state),
            pick(&NOUNS, &mut state),
            pick(&VERBS, &mut state),
            pick(&ADJECTIVES, &mut state),
            pick(&NOUNS, &mut state),
            pick(&ADVERBS, &mut state),
        );
        if !body.is_empty() {
            body.push_str(if sentence_index % SENTENCES_PER_PARAGRAPH == 0 {
                "\n\n"
            } else {
                " "
            });
        }
        body.push_str(&sentence);
        sentence_index += 1;
    }
    body
}

/// Prepends a unique nonce sentence so no prefix is shared across prompts.
pub fn nonce_prefixed(nonce: &str, text: &str) -> String {
    format!("Session reference code: {nonce}.\n\n{text}")
}

/// Prefill prompt: nonce first, filler text, then a request for a tiny reply.
pub fn build_prefill_prompt(nonce: &str, body: &str) -> String {
    nonce_prefixed(
        nonce,
        &format!("{body}\n\nAfter reading the text above, reply with the single word OK."),
    )
}

/// Prompt that elicits a long generation, with a nonce at the start.
pub fn build_long_output_prompt(nonce: &str) -> String {
    nonce_prefixed(nonce, LONG_OUTPUT_REQUEST)
}

/// The fixed (nonce-free) decode prompt.
pub const fn decode_prompt() -> &'static str {
    LONG_OUTPUT_REQUEST
}

/// 128-bit random hex string, unique per call within and across processes.
pub fn fresh_nonce() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let counter = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let high = RandomState::new().hash_one((counter, nanos, 0_u8));
    let low = RandomState::new().hash_one((counter, nanos, 1_u8));
    format!("{high:016x}{low:016x}")
}

/// Parses `--extra-body`, which must be a JSON object.
pub fn parse_extra_body(raw: &str) -> Result<Map<String, Value>> {
    let value: Value = serde_json::from_str(raw).context("--extra-body is not valid JSON")?;
    match value {
        Value::Object(map) => Ok(map),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Array(_) => {
            bail!("--extra-body must be a JSON object, got {value}")
        }
    }
}

/// Builds a streaming chat-completions body; `extra` keys override defaults.
pub fn build_request_body(
    model: &str,
    messages: &Value,
    max_tokens: u32,
    temperature: f32,
    extra: &Map<String, Value>,
) -> Value {
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "stream_options": { "include_usage": true },
        "max_tokens": max_tokens,
        "temperature": temperature,
    });
    if let Some(object) = body.as_object_mut() {
        for (key, value) in extra {
            object.insert(key.clone(), value.clone());
        }
    }
    body
}

/// Maps anything outside `[A-Za-z0-9._-]` to `_` so a model name can serve as
/// a public identifier and filename component.
pub fn sanitize_model_key(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Masks secret flag values in recorded argv so they never reach result files.
pub fn redact_secret_args(args: &[String]) -> Vec<String> {
    const SECRET_FLAGS: [&str; 1] = ["--api-key"];
    let mut redacted = Vec::with_capacity(args.len());
    let mut mask_next = false;
    for arg in args {
        if mask_next {
            redacted.push("<redacted>".to_owned());
            mask_next = false;
        } else if SECRET_FLAGS.contains(&arg.as_str()) {
            redacted.push(arg.clone());
            mask_next = true;
        } else if let Some(flag) = SECRET_FLAGS
            .iter()
            .find(|flag| arg.starts_with(&format!("{flag}=")))
        {
            redacted.push(format!("{flag}=<redacted>"));
        } else {
            redacted.push(arg.clone());
        }
    }
    redacted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(json: &str) -> Vec<u8> {
        format!("data: {json}\n").into_bytes()
    }

    #[test]
    fn parses_content_delta() -> Result<()> {
        let chunk = parse_sse_line(&line(r#"{"choices":[{"delta":{"content":"hi"}}]}"#))
            .context("chunk")?;
        assert!(chunk.carries_token);
        assert_eq!(chunk.content, "hi");
        assert!(chunk.usage.is_none());
        Ok(())
    }

    #[test]
    fn reasoning_and_tool_calls_count_as_tokens_without_content() -> Result<()> {
        for json in [
            r#"{"choices":[{"delta":{"reasoning_content":"hmm"}}]}"#,
            r#"{"choices":[{"delta":{"reasoning":"hmm"}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}"#,
        ] {
            let chunk = parse_sse_line(&line(json)).context("chunk")?;
            assert!(chunk.carries_token, "{json}");
            assert!(chunk.content.is_empty(), "{json}");
        }
        Ok(())
    }

    #[test]
    fn empty_and_role_only_deltas_are_not_tokens() -> Result<()> {
        for json in [
            r#"{"choices":[{"delta":{"role":"assistant","content":""}}]}"#,
            r#"{"choices":[{"delta":{"content":"","reasoning_content":""}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[]}}]}"#,
        ] {
            let chunk = parse_sse_line(&line(json)).context("chunk")?;
            assert!(!chunk.carries_token, "{json}");
        }
        Ok(())
    }

    #[test]
    fn parses_usage_with_cached_tokens() -> Result<()> {
        let chunk = parse_sse_line(&line(
            r#"{"choices":[],"usage":{"prompt_tokens":4100,"completion_tokens":32,"prompt_tokens_details":{"cached_tokens":4096}}}"#,
        ))
        .context("chunk")?;
        assert_eq!(
            chunk.usage,
            Some(UsageReport {
                completion_tokens: Some(32),
                prompt_tokens: Some(4100),
                cached_tokens: Some(4096),
            })
        );
        Ok(())
    }

    #[test]
    fn null_usage_and_done_and_noise_are_ignored() -> Result<()> {
        let chunk = parse_sse_line(&line(
            r#"{"choices":[{"delta":{"content":"a"}}],"usage":null}"#,
        ))
        .context("chunk")?;
        assert!(chunk.usage.is_none());
        assert!(parse_sse_line(b"data: [DONE]\n").is_none());
        assert!(parse_sse_line(b": keep-alive\n").is_none());
        assert!(parse_sse_line(b"\n").is_none());
        assert!(parse_sse_line(b"data: {not json\n").is_none());
        Ok(())
    }

    #[test]
    fn parses_error_payload() -> Result<()> {
        let chunk =
            parse_sse_line(&line(r#"{"error":{"message":"out of memory"}}"#)).context("chunk")?;
        assert_eq!(chunk.error.as_deref(), Some("out of memory"));
        Ok(())
    }

    #[test]
    fn decode_tokps_excludes_first_token() {
        let rate = decode_tokps(101, Duration::from_secs(2)).unwrap_or(f64::NAN);
        assert!((rate - 50.0).abs() < 1e-9);
        assert!(decode_tokps(1, Duration::from_secs(2)).is_none());
        assert!(decode_tokps(50, Duration::ZERO).is_none());
    }

    #[test]
    fn prefill_tokps_uses_ttft() {
        let rate = prefill_tokps(2000, 500.0).unwrap_or(f64::NAN);
        assert!((rate - 4000.0).abs() < 1e-9);
        assert!(prefill_tokps(2000, 0.0).is_none());
    }

    fn feed(accumulator: &mut StreamAccumulator, json: &str, millis: u64) {
        accumulator.ingest_line(&line(json), Duration::from_millis(millis));
    }

    #[test]
    fn accumulator_computes_metrics_from_usage() -> Result<()> {
        let mut accumulator = StreamAccumulator::default();
        feed(
            &mut accumulator,
            r#"{"choices":[{"delta":{"role":"assistant"}}]}"#,
            50,
        );
        feed(
            &mut accumulator,
            r#"{"choices":[{"delta":{"reasoning_content":"t"}}]}"#,
            200,
        );
        feed(
            &mut accumulator,
            r#"{"choices":[{"delta":{"content":"a"}}]}"#,
            300,
        );
        feed(
            &mut accumulator,
            r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":101}}"#,
            2200,
        );
        let outcome = accumulator.finish(Duration::from_millis(2200))?;
        let metrics = outcome.metrics;
        assert!((metrics.ttft_ms - 200.0).abs() < 1e-6);
        assert_eq!(metrics.completion_tokens, 101);
        assert_eq!(metrics.prompt_tokens, Some(10));
        assert!(!metrics.usage_missing);
        assert!((metrics.decode_tokps.unwrap_or(f64::NAN) - 50.0).abs() < 1e-6);
        assert!((metrics.e2e_tokps - 101.0 / 2.2).abs() < 1e-6);
        assert_eq!(outcome.text, "a");
        Ok(())
    }

    #[test]
    fn accumulator_falls_back_to_delta_count_and_flags_it() -> Result<()> {
        let mut accumulator = StreamAccumulator::default();
        for millis in [100, 200, 300] {
            feed(
                &mut accumulator,
                r#"{"choices":[{"delta":{"content":"x"}}]}"#,
                millis,
            );
        }
        let metrics = accumulator.finish(Duration::from_millis(300))?.metrics;
        assert!(metrics.usage_missing);
        assert_eq!(metrics.completion_tokens, 3);
        Ok(())
    }

    #[test]
    fn accumulator_fails_loudly_without_tokens_or_on_error() {
        assert!(
            StreamAccumulator::default()
                .finish(Duration::from_secs(1))
                .is_err()
        );

        let mut errored = StreamAccumulator::default();
        feed(
            &mut errored,
            r#"{"choices":[{"delta":{"content":"x"}}]}"#,
            10,
        );
        feed(&mut errored, r#"{"error":{"message":"boom"}}"#, 20);
        assert!(errored.finish(Duration::from_secs(1)).is_err());

        let mut zero_usage = StreamAccumulator::default();
        feed(
            &mut zero_usage,
            r#"{"choices":[{"delta":{"content":"x"}}]}"#,
            10,
        );
        feed(
            &mut zero_usage,
            r#"{"choices":[],"usage":{"completion_tokens":0}}"#,
            20,
        );
        assert!(zero_usage.finish(Duration::from_secs(1)).is_err());
    }

    #[test]
    fn prompt_body_is_deterministic_and_sized() {
        for target in [64, 512, 2048, 8192] {
            let first = build_prompt_body(target);
            assert_eq!(first, build_prompt_body(target));
            let target_chars = target * APPROX_CHARS_PER_TOKEN;
            assert!(first.len() >= target_chars);
            assert!(first.len() < target_chars + 200);
        }
    }

    #[test]
    fn prompt_body_is_varied() {
        let body = build_prompt_body(2048);
        let distinct: std::collections::HashSet<&str> = body.split_whitespace().collect();
        assert!(
            distinct.len() > 40,
            "only {} distinct words",
            distinct.len()
        );
        let sentences: std::collections::HashSet<&str> = body.split(". ").collect();
        assert!(sentences.len() > 100);
    }

    #[test]
    fn nonce_leads_prompt_and_only_prefix_differs() {
        let body = build_prompt_body(256);
        let first = build_prefill_prompt("aaaa", &body);
        let second = build_prefill_prompt("bbbb", &body);
        assert!(first.starts_with("Session reference code: aaaa."));
        assert_ne!(first, second);
        assert_eq!(
            first.trim_start_matches("Session reference code: aaaa."),
            second.trim_start_matches("Session reference code: bbbb.")
        );
        assert!(build_long_output_prompt("zz").starts_with("Session reference code: zz."));
    }

    #[test]
    fn nonces_are_unique() {
        let nonces: std::collections::HashSet<String> = (0..1000).map(|_| fresh_nonce()).collect();
        assert_eq!(nonces.len(), 1000);
        assert!(nonces.iter().all(|nonce| nonce.len() == 32));
    }

    #[test]
    fn request_body_defaults_and_extra_override() -> Result<()> {
        let extra = parse_extra_body(
            r#"{"temperature":0.5,"chat_template_kwargs":{"enable_thinking":false}}"#,
        )?;
        let messages = serde_json::json!([{"role":"user","content":"hi"}]);
        let body = build_request_body("m", &messages, 16, 0.0, &extra);
        assert_eq!(body.pointer("/model"), Some(&Value::from("m")));
        assert_eq!(body.pointer("/stream"), Some(&Value::Bool(true)));
        assert_eq!(
            body.pointer("/stream_options/include_usage"),
            Some(&Value::Bool(true))
        );
        assert_eq!(body.pointer("/max_tokens"), Some(&Value::from(16)));
        assert_eq!(body.pointer("/temperature"), Some(&Value::from(0.5)));
        assert_eq!(
            body.pointer("/chat_template_kwargs/enable_thinking"),
            Some(&Value::Bool(false))
        );
        assert!(parse_extra_body("[1]").is_err());
        assert!(parse_extra_body("nope").is_err());
        Ok(())
    }

    #[test]
    fn model_keys_are_sanitized() {
        assert_eq!(
            sanitize_model_key("mlx-community/Qwen3 1.7B:4bit"),
            "mlx-community_Qwen3_1.7B_4bit"
        );
    }

    #[test]
    fn api_key_is_redacted_from_argv() {
        let args: Vec<String> = [
            "bench",
            "--api-key",
            "secret",
            "--label",
            "x",
            "--api-key=other",
        ]
        .iter()
        .map(|arg| (*arg).to_owned())
        .collect();
        let redacted = redact_secret_args(&args);
        assert_eq!(
            redacted,
            [
                "bench",
                "--api-key",
                "<redacted>",
                "--label",
                "x",
                "--api-key=<redacted>"
            ]
        );
    }
}
