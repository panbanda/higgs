//! A quick correctness check of one model behind an OpenAI-compatible server.
//!
//! Every check uses greedy sampling and compares outputs only where a difference would be a
//! real fault: the same request must give the same text whether streamed or not, cold or from
//! the prefix cache, and (for a few tokens) alone or batched with another request. Long
//! outputs are never compared, because half-precision rounding legitimately makes them diverge.

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures::future::try_join;
use serde_json::{Value, json};

use crate::client::{Completion, Endpoint, TokenLogprob};
use crate::compare::{build_prompt_body, fresh_nonce};

/// Tokens compared between a solo and a batched run; longer outputs may diverge legitimately.
const BATCH_COMPARE_TOKENS: u32 = 16;
/// A batched run may differ from the solo run only where the solo run's best token beat the
/// runner-up by less than this many nats, since half-precision rounding can flip a near tie.
pub const NEAR_TIE_NATS: f64 = 0.5;
const SHORT_REPLY_TOKENS: u32 = 48;
const PREFIX_PROMPT_TOKENS: usize = 300;
const TOOL_NAME: &str = "get_weather";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Pass(String),
    Fail(String),
    Skip(String),
}

impl Outcome {
    const fn label(&self) -> &'static str {
        match self {
            Self::Pass(_) => "PASS",
            Self::Fail(_) => "FAIL",
            Self::Skip(_) => "SKIP",
        }
    }

    fn detail(&self) -> &str {
        match self {
            Self::Pass(detail) | Self::Fail(detail) | Self::Skip(detail) => detail,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CheckResult {
    pub name: &'static str,
    pub outcome: Outcome,
    pub elapsed: Duration,
}

/// What the server says about the model that decides which checks apply.
#[derive(Debug, Clone, Default)]
pub struct ModelFacts {
    /// `batched` or `sequential` as reported by `/v1/models/status`, when known.
    pub decode_engine: Option<String>,
    /// Prompts the prefix cache keeps, as reported by `/v1/system`; `Some(0)` means it is off.
    pub prefix_cache_entries: Option<u64>,
}

impl ModelFacts {
    /// Reads what the server reports about the model it lists as `listed_name`, from
    /// `/v1/models/status` and `/v1/system`. Anything it does not report stays unknown.
    pub async fn discover(endpoint: &Endpoint, listed_name: &str) -> Self {
        let entry_of = |body: Value, list: &str, key: &str| {
            body.get(list)?
                .as_array()?
                .iter()
                .find(|entry| entry.get(key).and_then(Value::as_str) == Some(listed_name))
                .cloned()
        };
        let decode_engine = endpoint
            .get_json("/v1/models/status")
            .await
            .ok()
            .and_then(|status| entry_of(status, "data", "name"))
            .and_then(|entry| entry.get("engine")?.as_str().map(str::to_owned));
        let prefix_cache_entries = endpoint
            .get_json("/v1/system")
            .await
            .ok()
            .and_then(|system| entry_of(system, "models", "name"))
            .and_then(|entry| entry.get("prefix_cache_entries")?.as_u64());
        Self {
            decode_engine,
            prefix_cache_entries,
        }
    }
}

fn chat_body(model: &str, prompt: &str, max_tokens: u32) -> Value {
    json!({
        "model": model,
        "messages": [{ "role": "user", "content": prompt }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
}

fn with_fields(mut body: Value, fields: &Value) -> Value {
    if let (Some(target), Some(extra)) = (body.as_object_mut(), fields.as_object()) {
        for (key, value) in extra {
            target.insert(key.clone(), value.clone());
        }
    }
    body
}

fn is_sensible_finish(finish_reason: Option<&str>) -> bool {
    matches!(finish_reason, Some("stop" | "length"))
}

fn finish_label(finish_reason: Option<&str>) -> &str {
    finish_reason.unwrap_or("none")
}

async fn check_chat(endpoint: &Endpoint) -> Result<Outcome> {
    let body = chat_body(
        endpoint.model(),
        "Name three primary colors.",
        SHORT_REPLY_TOKENS,
    );
    let reply = endpoint.complete_body(&body).await?;
    if reply.full_text().trim().is_empty() {
        return Ok(Outcome::Fail("the reply has no text".to_owned()));
    }
    if !is_sensible_finish(reply.finish_reason.as_deref()) {
        return Ok(Outcome::Fail(format!(
            "unexpected finish reason {}",
            finish_label(reply.finish_reason.as_deref())
        )));
    }
    Ok(Outcome::Pass(format!(
        "{} tokens, finish={}",
        reply.usage.completion_tokens.unwrap_or_default(),
        finish_label(reply.finish_reason.as_deref())
    )))
}

async fn check_stream_matches_non_stream(endpoint: &Endpoint) -> Result<Outcome> {
    let prompt = format!(
        "Reference {}. List the seven days of the week.",
        fresh_nonce()
    );
    let body = chat_body(endpoint.model(), &prompt, 32);
    // A cold prefill and a prefix-cache resume can round differently, so run both compared
    // requests warm; the cold-versus-cached comparison is its own check.
    endpoint.complete_body(&body).await?;
    let streamed = endpoint
        .stream_body(&with_fields(
            body.clone(),
            &json!({ "stream": true, "stream_options": { "include_usage": true } }),
        ))
        .await?;
    let whole = endpoint.complete_body(&body).await?;
    // The non-streaming path trims the reasoning and the visible text separately after parsing;
    // a stream cannot, so surrounding whitespace in each field is the one legitimate difference.
    for (field, streamed_field, whole_field) in [
        ("reasoning", &streamed.reasoning, &whole.reasoning),
        ("text", &streamed.text, &whole.content),
    ] {
        if streamed_field.trim() != whole_field.trim() {
            return Ok(Outcome::Fail(format!(
                "streamed {field} {streamed_field:?} but non-streaming gave {whole_field:?}"
            )));
        }
    }
    if streamed.finish_reason != whole.finish_reason {
        return Ok(Outcome::Fail(format!(
            "finish reason differs: streamed {}, non-streaming {}",
            finish_label(streamed.finish_reason.as_deref()),
            finish_label(whole.finish_reason.as_deref())
        )));
    }
    Ok(Outcome::Pass(format!(
        "{} characters identical",
        whole.full_text().chars().count()
    )))
}

async fn check_prefix_cache(endpoint: &Endpoint, facts: &ModelFacts) -> Result<Outcome> {
    if facts.prefix_cache_entries == Some(0) {
        return Ok(Outcome::Skip(
            "the prefix cache is off (max_entries = 0)".to_owned(),
        ));
    }
    let prompt = format!(
        "Reference {}.\n\n{}\n\nSummarize the text above in one sentence.",
        fresh_nonce(),
        build_prompt_body(PREFIX_PROMPT_TOKENS)
    );
    let body = chat_body(endpoint.model(), &prompt, 24);
    let first = endpoint.complete_body(&body).await?;
    let second = endpoint.complete_body(&body).await?;
    let Some(cached) = second.usage.cached_tokens else {
        return Ok(Outcome::Fail(
            "the server did not report cached tokens".to_owned(),
        ));
    };
    if cached == 0 {
        return Ok(Outcome::Fail(format!(
            "the repeated prompt of {} tokens hit no cached tokens",
            second.usage.prompt_tokens.unwrap_or_default()
        )));
    }
    if first.full_text() != second.full_text() {
        return Ok(Outcome::Fail(format!(
            "cold run gave {:?}, cached run gave {:?}",
            first.full_text(),
            second.full_text()
        )));
    }
    Ok(Outcome::Pass(format!(
        "cached {cached} of {} tokens, identical text",
        second.usage.prompt_tokens.unwrap_or_default()
    )))
}

fn person_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" },
            "pet": { "type": "string", "enum": ["cat", "dog", "fish"] }
        },
        "required": ["name", "age", "pet"],
        "additionalProperties": false
    })
}

async fn check_json_schema(endpoint: &Endpoint) -> Result<Outcome> {
    let schema = person_schema();
    let body = with_fields(
        chat_body(
            endpoint.model(),
            "Invent a person and describe them as JSON with their name, age and pet.",
            160,
        ),
        &json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": { "name": "person", "schema": schema }
            }
        }),
    );
    let reply = endpoint.complete_body(&body).await?;
    let value: Value = match serde_json::from_str(reply.content.trim()) {
        Ok(value) => value,
        Err(error) => {
            return Ok(Outcome::Fail(format!(
                "the reply is not JSON ({error}): {:?} (finish={})",
                reply.content,
                finish_label(reply.finish_reason.as_deref())
            )));
        }
    };
    Ok(schema_violation(&schema, &value).map_or_else(
        || Outcome::Pass(format!("valid against the schema: {value}")),
        |violation| Outcome::Fail(format!("{violation} in {value}")),
    ))
}

fn weather_tools() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }
        }
    }])
}

async fn prompt_tokens(endpoint: &Endpoint, body: &Value) -> Result<u32> {
    endpoint
        .complete_body(body)
        .await?
        .usage
        .prompt_tokens
        .context("the server did not report prompt tokens")
}

/// Whether an error is the server's HTTP 400 saying the tool-call format cannot be forced. Any
/// other error is a real failure.
fn refuses_forced_choice(error: &str) -> bool {
    error.contains("HTTP 400") && error.contains("forcing a tool call is not supported")
}

async fn check_tool_call(endpoint: &Endpoint) -> Result<Outcome> {
    let question = "What is the weather in Paris right now?";
    let plain = chat_body(endpoint.model(), question, 1);
    let with_tools = with_fields(plain.clone(), &json!({ "tools": weather_tools() }));
    let without = prompt_tokens(endpoint, &plain).await?;
    let with = prompt_tokens(endpoint, &with_tools).await?;
    if with <= without {
        return Ok(Outcome::Skip(
            "the chat template does not render tool definitions (the prompt did not grow)"
                .to_owned(),
        ));
    }

    let forced = with_fields(
        chat_body(endpoint.model(), question, 128),
        &json!({
            "tools": weather_tools(),
            "tool_choice": { "type": "function", "function": { "name": TOOL_NAME } }
        }),
    );
    let reply = match endpoint.complete_body(&forced).await {
        Ok(reply) => reply,
        Err(error) if refuses_forced_choice(&format!("{error:#}")) => {
            return Ok(Outcome::Skip(
                "the model's tool-call format cannot force a tool choice".to_owned(),
            ));
        }
        Err(error) => return Err(error),
    };
    let Some(call) = reply.tool_calls.first() else {
        return Ok(Outcome::Fail(format!(
            "no tool call; the reply was {:?} (finish={})",
            reply.content,
            finish_label(reply.finish_reason.as_deref())
        )));
    };
    if call.name != TOOL_NAME {
        return Ok(Outcome::Fail(format!(
            "called {:?}, not {TOOL_NAME}",
            call.name
        )));
    }
    let arguments: Value = match serde_json::from_str(&call.arguments) {
        Ok(arguments) => arguments,
        Err(error) => {
            return Ok(Outcome::Fail(format!(
                "arguments are not JSON ({error}): {:?}",
                call.arguments
            )));
        }
    };
    if !arguments.get("city").is_some_and(Value::is_string) {
        return Ok(Outcome::Fail(format!(
            "arguments lack a string \"city\": {arguments}"
        )));
    }
    Ok(Outcome::Pass(format!(
        "{TOOL_NAME}({arguments}), finish={}",
        finish_label(reply.finish_reason.as_deref())
    )))
}

fn reply_problem(reply: &Completion) -> Option<String> {
    if reply.full_text().trim().is_empty() {
        return Some("empty reply".to_owned());
    }
    (!is_sensible_finish(reply.finish_reason.as_deref())).then(|| {
        format!(
            "unexpected finish reason {}",
            finish_label(reply.finish_reason.as_deref())
        )
    })
}

async fn check_concurrent(endpoint: &Endpoint) -> Result<Outcome> {
    let first_body = chat_body(
        endpoint.model(),
        &format!("Reference {}. Name a river in Europe.", fresh_nonce()),
        32,
    );
    let second_body = chat_body(
        endpoint.model(),
        &format!("Reference {}. Name a mountain in Asia.", fresh_nonce()),
        32,
    );
    let (first, second) = try_join(
        endpoint.complete_body(&first_body),
        endpoint.complete_body(&second_body),
    )
    .await?;
    for (label, reply) in [("first", &first), ("second", &second)] {
        if let Some(problem) = reply_problem(reply) {
            return Ok(Outcome::Fail(format!("{label} request: {problem}")));
        }
    }
    Ok(Outcome::Pass("both requests completed".to_owned()))
}

async fn check_batched_matches_solo(endpoint: &Endpoint, facts: &ModelFacts) -> Result<Outcome> {
    match facts.decode_engine.as_deref() {
        Some("batched") => {}
        Some(other) => {
            return Ok(Outcome::Skip(format!(
                "the model decodes one request at a time ({other})"
            )));
        }
        None => {
            return Ok(Outcome::Skip(
                "the server does not report how the model decodes".to_owned(),
            ));
        }
    }
    let subject = chat_body(
        endpoint.model(),
        "Count from one to twenty in words.",
        BATCH_COMPARE_TOKENS,
    );
    let companion = chat_body(
        endpoint.model(),
        "Write one sentence about the sea.",
        BATCH_COMPARE_TOKENS,
    );
    // Requests with logprobs never join a batched pass, so the compared runs send none. A
    // separate run with logprobs supplies the margins that tell a near tie from a real fault, and
    // fills the prefix cache so the compared runs are both warm.
    let reference = endpoint
        .complete_body(&with_fields(
            subject.clone(),
            &json!({ "logprobs": true, "top_logprobs": 2 }),
        ))
        .await?;
    let solo = endpoint.complete_body(&subject).await?;
    let (batched, other) = try_join(
        endpoint.complete_body(&subject),
        endpoint.complete_body(&companion),
    )
    .await?;
    for (label, reply) in [
        ("solo", &solo),
        ("batched", &batched),
        ("companion", &other),
    ] {
        if let Some(problem) = reply_problem(reply) {
            return Ok(Outcome::Fail(format!("{label} request: {problem}")));
        }
    }
    Ok(
        match compare_texts(&reference, &solo.full_text(), &batched.full_text()) {
            Comparison::Identical => Outcome::Pass(format!(
                "{BATCH_COMPARE_TOKENS} tokens identical alone and batched"
            )),
            Comparison::NearTie { index, gap } => Outcome::Pass(format!(
                "identical until token {index}, then diverged on a near tie (gap {gap:.2})"
            )),
            Comparison::Different(detail) => Outcome::Fail(detail),
        },
    )
}

#[derive(Debug, Clone, PartialEq)]
enum Comparison {
    Identical,
    NearTie { index: usize, gap: f64 },
    Different(String),
}

/// Margin of the best token over the runner-up at a position, when the server reported both.
fn margin(token: &TokenLogprob) -> Option<f64> {
    match token.top.as_slice() {
        [best, runner_up, ..] => Some(best - runner_up),
        [] | [_] => None,
    }
}

const THINK_DELIMITERS: [&str; 2] = ["<think>", "</think>"];

/// The pieces of `raw` that survive reasoning parsing, as `(start in raw, length)`: the text
/// between `<think>` and `</think>` delimiters with surrounding whitespace removed.
fn surviving_pieces(raw: &str) -> Vec<(usize, usize)> {
    let mut pieces = Vec::new();
    let mut segment_start = 0;
    let mut push_segment = |start: usize, end: usize| {
        let segment = raw.get(start..end).unwrap_or_default();
        let trimmed = segment.trim();
        if !trimmed.is_empty() {
            pieces.push((
                start + segment.len() - segment.trim_start().len(),
                trimmed.len(),
            ));
        }
    };
    let mut cursor = 0;
    while let Some((position, delimiter)) = THINK_DELIMITERS
        .iter()
        .filter_map(|delimiter| {
            raw.get(cursor..)
                .and_then(|rest| rest.find(delimiter))
                .map(|found| (cursor + found, *delimiter))
        })
        .min_by_key(|(position, _)| *position)
    {
        push_segment(segment_start, position);
        cursor = position + delimiter.len();
        segment_start = cursor;
    }
    push_segment(segment_start, raw.len());
    pieces
}

/// Where byte `offset` of the reference's parsed text (reasoning then content) sits in the raw
/// token text the logprobs describe, which still holds the `<think>` delimiters and the
/// whitespace the parse removed. `None` when the parsed text is not what the raw text parses to.
fn raw_offset(reference: &Completion, offset: usize) -> Option<usize> {
    let raw: String = reference
        .token_logprobs
        .iter()
        .map(|token| token.token.as_str())
        .collect();
    let parsed = reference.full_text();
    if raw == parsed {
        return Some(offset);
    }
    let pieces = surviving_pieces(&raw);
    let rebuilt: String = pieces
        .iter()
        .filter_map(|&(start, length)| raw.get(start..start + length))
        .collect();
    if rebuilt != parsed {
        return None;
    }
    let mut parsed_start = 0;
    for (start, length) in pieces {
        if offset < parsed_start + length {
            return Some(start + offset - parsed_start);
        }
        parsed_start += length;
    }
    None
}

/// Compares the batched text with the solo text. A difference is accepted only where the
/// `reference` run (the same request with logprobs) shows a margin under `NEAR_TIE_NATS` for the
/// token at the first differing position.
fn compare_texts(reference: &Completion, solo: &str, batched: &str) -> Comparison {
    if solo == batched {
        return Comparison::Identical;
    }
    let offset = solo
        .char_indices()
        .zip(batched.chars())
        .find(|((_, alone), together)| alone != together)
        .map_or_else(|| solo.len().min(batched.len()), |((index, _), _)| index);
    let describe = || format!("solo gave {solo:?}, batched gave {batched:?}");
    let agrees = solo
        .get(..offset)
        .is_some_and(|prefix| reference.full_text().starts_with(prefix));
    if !agrees {
        return Comparison::Different(format!(
            "texts differ at byte {offset}, which the logprob reference does not cover: {}",
            describe()
        ));
    }
    let Some(raw_position) = raw_offset(reference, offset) else {
        return Comparison::Different(format!(
            "texts differ at byte {offset}, past the reference tokens: {}",
            describe()
        ));
    };
    let mut start = 0;
    for (index, token) in reference.token_logprobs.iter().enumerate() {
        let end = start + token.token.len();
        if raw_position < end {
            return match margin(token) {
                Some(gap) if gap < NEAR_TIE_NATS => Comparison::NearTie { index, gap },
                Some(gap) => Comparison::Different(format!(
                    "diverged at token {index} where the solo margin was {gap:.2} nats: {}",
                    describe()
                )),
                None => Comparison::Different(format!(
                    "diverged at token {index} with no runner-up reported: {}",
                    describe()
                )),
            };
        }
        start = end;
    }
    Comparison::Different(format!(
        "texts differ at byte {offset}, past the reference tokens: {}",
        describe()
    ))
}

/// The first way `value` fails to satisfy `schema`, for the schema keywords the self-test uses:
/// `type`, `enum`, `properties`, `required`, `additionalProperties: false` and `items`.
pub fn schema_violation(schema: &Value, value: &Value) -> Option<String> {
    if let Some(expected) = schema.get("type").and_then(Value::as_str) {
        let matches = match expected {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "integer" => value.is_i64() || value.is_u64(),
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            "null" => value.is_null(),
            _ => true,
        };
        if !matches {
            return Some(format!("expected {expected}, found {value}"));
        }
    }
    if let Some(allowed) = schema.get("enum").and_then(Value::as_array) {
        if !allowed.contains(value) {
            return Some(format!("{value} is not one of {allowed:?}"));
        }
    }
    if let Some(object) = value.as_object() {
        let properties = schema.get("properties").and_then(Value::as_object);
        for name in schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            if !object.contains_key(name) {
                return Some(format!("missing required property {name:?}"));
            }
        }
        for (name, member) in object {
            match properties.and_then(|known| known.get(name)) {
                Some(member_schema) => {
                    if let Some(violation) = schema_violation(member_schema, member) {
                        return Some(format!("{name}: {violation}"));
                    }
                }
                None if schema.get("additionalProperties") == Some(&Value::Bool(false)) => {
                    return Some(format!("unexpected property {name:?}"));
                }
                None => {}
            }
        }
    }
    if let (Some(items), Some(elements)) = (schema.get("items"), value.as_array()) {
        for (index, element) in elements.iter().enumerate() {
            if let Some(violation) = schema_violation(items, element) {
                return Some(format!("[{index}]: {violation}"));
            }
        }
    }
    None
}

async fn timed(
    name: &'static str,
    check: impl std::future::Future<Output = Result<Outcome>>,
) -> CheckResult {
    let started = Instant::now();
    let outcome = check
        .await
        .unwrap_or_else(|error| Outcome::Fail(format!("{error:#}")));
    CheckResult {
        name,
        outcome,
        elapsed: started.elapsed(),
    }
}

/// Runs every check in order, reporting each result through `on_result` as it finishes.
pub async fn run_selftest(
    endpoint: &Endpoint,
    facts: &ModelFacts,
    on_result: &(dyn Fn(&CheckResult) + Sync),
) -> Vec<CheckResult> {
    let mut results = Vec::new();
    let mut record = |result: CheckResult| {
        on_result(&result);
        results.push(result);
    };
    record(timed("chat completes", check_chat(endpoint)).await);
    record(
        timed(
            "stream equals non-stream",
            check_stream_matches_non_stream(endpoint),
        )
        .await,
    );
    record(timed("prefix cache hit", check_prefix_cache(endpoint, facts)).await);
    record(timed("json schema", check_json_schema(endpoint)).await);
    record(timed("forced tool call", check_tool_call(endpoint)).await);
    record(timed("concurrent requests", check_concurrent(endpoint)).await);
    record(
        timed(
            "batched equals solo",
            check_batched_matches_solo(endpoint, facts),
        )
        .await,
    );
    results
}

/// One result as `PASS  name  1.2s  detail`.
pub fn render_result(result: &CheckResult) -> String {
    format!(
        "{}  {:<26}  {:>5.1}s  {}",
        result.outcome.label(),
        result.name,
        result.elapsed.as_secs_f64(),
        result.outcome.detail()
    )
}

pub fn has_failures(results: &[CheckResult]) -> bool {
    results
        .iter()
        .any(|result| matches!(result.outcome, Outcome::Fail(_)))
}

pub fn render_summary(results: &[CheckResult]) -> String {
    let count = |wanted: &str| {
        results
            .iter()
            .filter(|result| result.outcome.label() == wanted)
            .count()
    };
    let mut summary = String::new();
    let _ = write!(
        summary,
        "{} passed, {} failed, {} skipped",
        count("PASS"),
        count("FAIL"),
        count("SKIP")
    );
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_accepts_a_conforming_value() {
        let value = json!({"name": "Ada", "age": 36, "pet": "cat"});
        assert_eq!(schema_violation(&person_schema(), &value), None);
    }

    #[test]
    fn schema_reports_each_kind_of_violation() {
        let schema = person_schema();
        let cases = [
            (json!({"name": "Ada", "age": 36}), "missing required"),
            (
                json!({"name": "Ada", "age": "36", "pet": "cat"}),
                "age: expected integer",
            ),
            (json!({"name": "Ada", "age": 36, "pet": "cow"}), "pet:"),
            (
                json!({"name": "Ada", "age": 36, "pet": "cat", "extra": 1}),
                "unexpected property",
            ),
            (json!(["x"]), "expected object"),
        ];
        for (value, needle) in cases {
            let violation = schema_violation(&schema, &value);
            assert!(
                violation
                    .as_deref()
                    .is_some_and(|text| text.contains(needle)),
                "{value}: {violation:?}"
            );
        }
    }

    #[test]
    fn schema_checks_array_items() {
        let schema = json!({"type": "array", "items": {"type": "integer"}});
        assert_eq!(schema_violation(&schema, &json!([1, 2])), None);
        assert!(
            schema_violation(&schema, &json!([1, "x"])).is_some_and(|text| text.starts_with("[1]"))
        );
    }

    fn run(tokens: &[(&str, &[f64])]) -> Completion {
        Completion {
            content: tokens.iter().map(|(token, _)| *token).collect(),
            reasoning: String::new(),
            finish_reason: None,
            usage: crate::compare::UsageReport::default(),
            tool_calls: Vec::new(),
            token_logprobs: tokens
                .iter()
                .map(|(token, top)| TokenLogprob {
                    token: (*token).to_owned(),
                    top: top.to_vec(),
                })
                .collect(),
        }
    }

    #[test]
    fn a_divergence_on_a_near_tie_is_accepted_and_a_clear_one_is_not() {
        let reference = run(&[
            ("a", &[-0.1, -3.0]),
            ("b", &[-0.5, -0.62]),
            ("c", &[-0.1, -2.0]),
        ]);
        assert_eq!(
            compare_texts(&reference, "abc", "abc"),
            Comparison::Identical
        );
        assert!(matches!(
            compare_texts(&reference, "abc", "axy"),
            Comparison::NearTie { index: 1, gap } if (gap - 0.12).abs() < 1e-9
        ));
        assert!(matches!(
            compare_texts(&reference, "abc", "zbc"),
            Comparison::Different(detail) if detail.contains("token 0")
        ));
    }

    #[test]
    fn offsets_map_through_think_delimiters_and_trimmed_whitespace() {
        let mut reference = run(&[
            ("Let", &[-0.1, -5.0]),
            (" me think", &[-0.1, -5.0]),
            ("\n", &[-0.1, -5.0]),
            ("</think>", &[-0.1, -5.0]),
            ("\n\n", &[-0.1, -5.0]),
            ("The", &[-0.5, -0.6]),
            (" answer", &[-0.1, -5.0]),
        ]);
        reference.reasoning = "Let me think".to_owned();
        reference.content = "The answer".to_owned();
        assert!(matches!(
            compare_texts(&reference, "Let me thinkThe answer", "Let me thinkA answer"),
            Comparison::NearTie { index: 5, .. }
        ));
        assert!(matches!(
            compare_texts(
                &reference,
                "Let me thinkThe answer",
                "Let me thinkThe answer."
            ),
            Comparison::Different(_)
        ));
    }

    #[test]
    fn only_the_forced_choice_refusal_counts_as_unsupported() {
        let refusal = "http://x/v1/chat/completions returned HTTP 400 Bad Request: \
            {\"error\":{\"message\":\"forcing a tool call is not supported for the gemma \
            tool-call format\"}}";
        assert!(refuses_forced_choice(refusal));
        assert!(!refuses_forced_choice(
            "returned HTTP 400 Bad Request: {\"error\":\"max_tokens is too large\"}"
        ));
        assert!(!refuses_forced_choice(
            "returned HTTP 500: forcing a tool call is not supported"
        ));
    }

    #[test]
    fn a_difference_the_reference_does_not_cover_fails() {
        let reference = run(&[("a", &[-0.1, -3.0]), ("b", &[-0.5, -0.52])]);
        assert!(matches!(
            compare_texts(&reference, "xyz1", "xyz2"),
            Comparison::Different(detail) if detail.contains("does not cover")
        ));
        assert!(matches!(
            compare_texts(&reference, "ab", "abcd"),
            Comparison::Different(detail) if detail.contains("past the reference")
        ));
    }

    fn result(name: &'static str, outcome: Outcome) -> CheckResult {
        CheckResult {
            name,
            outcome,
            elapsed: Duration::from_millis(1500),
        }
    }

    #[test]
    fn results_render_and_summarize() {
        let passed = result("a", Outcome::Pass("fine".to_owned()));
        let rendered = render_result(&passed);
        assert!(rendered.starts_with("PASS  a"), "{rendered}");
        assert!(
            rendered.contains("1.5s") && rendered.ends_with("fine"),
            "{rendered}"
        );

        let mut results = vec![passed, result("b", Outcome::Skip("n/a".to_owned()))];
        assert_eq!(render_summary(&results), "1 passed, 0 failed, 1 skipped");
        assert!(!has_failures(&results));

        results.push(result("c", Outcome::Fail("broke".to_owned())));
        assert!(has_failures(&results));
        assert_eq!(render_summary(&results), "1 passed, 1 failed, 1 skipped");
    }
}
