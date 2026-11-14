//! Drives the benchmark runner and the self-test against an in-process fake chat server, so the
//! client logic is covered without a model.
#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::panic
)]

use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use higgs_bench::client::{Endpoint, EndpointSettings};
use higgs_bench::runner::{Plan, Scenario, run_scenarios};
use higgs_bench::selftest::{ModelFacts, Outcome, has_failures, run_selftest};
use serde_json::{Value, json};

#[derive(Clone)]
#[allow(clippy::struct_excessive_bools)]
struct Behavior {
    report_cached_tokens: bool,
    engine: &'static str,
    render_tools: bool,
    prefix_cache_entries: u64,
    blank_counting_replies: bool,
    refuse_forced_tools: bool,
    seen_prompts: Arc<Mutex<Vec<String>>>,
}

impl Behavior {
    fn healthy() -> Self {
        Self {
            report_cached_tokens: true,
            engine: "batched",
            render_tools: true,
            prefix_cache_entries: 8,
            blank_counting_replies: false,
            refuse_forced_tools: false,
            seen_prompts: Arc::default(),
        }
    }
}

fn prompt_of(body: &Value) -> String {
    body.pointer("/messages")
        .and_then(Value::as_array)
        .map(|messages| {
            messages
                .iter()
                .filter_map(|message| message.get("content").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn reply_text(body: &Value) -> String {
    let max_tokens = body.get("max_tokens").and_then(Value::as_u64).unwrap_or(16);
    let words = usize::try_from(max_tokens.min(24)).unwrap_or(24);
    vec!["word"; words].join(" ")
}

async fn chat(State(behavior): State<Behavior>, Json(body): Json<Value>) -> Response {
    let prompt = prompt_of(&body);
    let is_counting_prompt = prompt.contains("Count from one");
    let has_tools = body.get("tools").is_some_and(|tools| !tools.is_null());
    let tool_tokens = if has_tools && behavior.render_tools {
        40
    } else {
        0
    };
    let prompt_tokens = 10 + prompt.len() / 4 + tool_tokens;
    let mut seen = behavior.seen_prompts.lock().unwrap();
    let cached = if seen
        .iter()
        .any(|earlier| prompt.starts_with(earlier.as_str()))
    {
        prompt_tokens - 1
    } else {
        0
    };
    seen.push(prompt);
    drop(seen);
    let mut usage = json!({ "prompt_tokens": prompt_tokens, "completion_tokens": 0 });
    if behavior.report_cached_tokens {
        usage["prompt_tokens_details"] = json!({ "cached_tokens": cached });
    }

    if behavior.refuse_forced_tools && body.pointer("/tool_choice/function/name").is_some() {
        let error = json!({ "error": { "message": "forcing a tool call is not supported for the gemma tool-call format" } });
        return (StatusCode::BAD_REQUEST, Json(error)).into_response();
    }
    let streaming = body.get("stream") == Some(&Value::Bool(true));
    let forced_tool = body
        .pointer("/tool_choice/function/name")
        .and_then(Value::as_str);
    let wants_schema = body.pointer("/response_format/type") == Some(&json!("json_schema"));
    let text = if wants_schema {
        r#"{"name":"Ada","age":36,"pet":"cat"}"#.to_owned()
    } else if behavior.blank_counting_replies && is_counting_prompt {
        String::new()
    } else {
        reply_text(&body)
    };
    let completion_tokens = text.split_whitespace().count().max(1);
    usage["completion_tokens"] = json!(completion_tokens);

    if streaming {
        let mut events: Vec<Value> = text
            .split_inclusive(' ')
            .map(|piece| json!({ "choices": [{ "delta": { "content": piece } }] }))
            .collect();
        events.push(json!({ "choices": [{ "delta": {}, "finish_reason": "length" }] }));
        events.push(json!({ "choices": [], "usage": usage }));
        let stream = futures::stream::iter(events).then(|event| async move {
            tokio::time::sleep(Duration::from_millis(2)).await;
            Ok::<_, Infallible>(Event::default().data(event.to_string()))
        });
        let done = futures::stream::once(async { Ok(Event::default().data("[DONE]")) });
        return Sse::new(stream.chain(done)).into_response();
    }

    let message = forced_tool.map_or_else(
        || json!({ "content": text }),
        |name| {
            json!({
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": name, "arguments": "{\"city\":\"Paris\"}" }
                }]
            })
        },
    );
    let finish = if forced_tool.is_some() {
        "tool_calls"
    } else if wants_schema {
        "stop"
    } else {
        "length"
    };
    Json(json!({
        "choices": [{ "message": message, "finish_reason": finish }],
        "usage": usage
    }))
    .into_response()
}

async fn start(behavior: Behavior) -> String {
    let engine = behavior.engine;
    let entries = behavior.prefix_cache_entries;
    let app =
        Router::new()
            .route("/health", get(|| async { "ok" }))
            .route(
                "/v1/models",
                get(|| async { Json(json!({ "data": [{ "id": "mock" }] })) }),
            )
            .route(
                "/v1/models/status",
                get(move || async move {
                    Json(json!({ "data": [{ "name": "mock", "engine": engine }] }))
                }),
            )
            .route(
                "/v1/system",
                get(move || async move {
                    Json(json!({ "models": [{ "name": "mock", "prefix_cache_entries": entries }] }))
                }),
            )
            .route("/v1/chat/completions", post(chat))
            .with_state(behavior);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{address}")
}

fn endpoint(base_url: String) -> Endpoint {
    Endpoint::new(EndpointSettings {
        base_url,
        model: "mock".to_owned(),
        api_key: None,
        temperature: 0.0,
        extra_body: serde_json::Map::new(),
        timeout: Duration::from_secs(30),
    })
    .unwrap()
}

fn outcome_of<'a>(results: &'a [higgs_bench::selftest::CheckResult], name: &str) -> &'a Outcome {
    &results
        .iter()
        .find(|result| result.name == name)
        .unwrap_or_else(|| panic!("no check named {name}"))
        .outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn every_check_passes_against_a_healthy_server() {
        let endpoint = endpoint(start(Behavior::healthy()).await);
        let facts = ModelFacts::discover(&endpoint, "mock").await;
        assert_eq!(facts.decode_engine.as_deref(), Some("batched"));
        let results = run_selftest(&endpoint, &facts, &|_| {}).await;
        for result in &results {
            assert!(
                matches!(result.outcome, Outcome::Pass(_)),
                "{}: {:?}",
                result.name,
                result.outcome
            );
        }
        assert_eq!(results.len(), 7);
    }

    #[tokio::test]
    async fn the_batched_check_fails_when_both_compared_replies_are_empty() {
        let behavior = Behavior {
            blank_counting_replies: true,
            ..Behavior::healthy()
        };
        let endpoint = endpoint(start(behavior).await);
        let facts = ModelFacts::discover(&endpoint, "mock").await;
        let results = run_selftest(&endpoint, &facts, &|_| {}).await;
        assert!(matches!(
            outcome_of(&results, "batched equals solo"),
            Outcome::Fail(detail) if detail.contains("solo request")
        ));
    }

    #[tokio::test]
    async fn a_format_that_cannot_force_a_tool_call_skips_the_tool_check() {
        let behavior = Behavior {
            refuse_forced_tools: true,
            ..Behavior::healthy()
        };
        let endpoint = endpoint(start(behavior).await);
        let facts = ModelFacts::discover(&endpoint, "mock").await;
        let results = run_selftest(&endpoint, &facts, &|_| {}).await;
        assert!(!has_failures(&results), "{results:?}");
        assert!(matches!(
            outcome_of(&results, "forced tool call"),
            Outcome::Skip(reason) if reason.contains("cannot force")
        ));
    }

    #[tokio::test]
    async fn the_prefix_check_skips_when_the_cache_is_off() {
        let behavior = Behavior {
            report_cached_tokens: false,
            prefix_cache_entries: 0,
            ..Behavior::healthy()
        };
        let endpoint = endpoint(start(behavior).await);
        let facts = ModelFacts::discover(&endpoint, "mock").await;
        assert_eq!(facts.prefix_cache_entries, Some(0));
        let results = run_selftest(&endpoint, &facts, &|_| {}).await;
        assert!(!has_failures(&results), "{results:?}");
        assert!(matches!(
            outcome_of(&results, "prefix cache hit"),
            Outcome::Skip(reason) if reason.contains("prefix cache is off")
        ));
    }

    #[tokio::test]
    async fn a_server_that_hides_cached_tokens_fails_the_prefix_check_only() {
        let behavior = Behavior {
            report_cached_tokens: false,
            ..Behavior::healthy()
        };
        let endpoint = endpoint(start(behavior).await);
        let facts = ModelFacts::discover(&endpoint, "mock").await;
        let results = run_selftest(&endpoint, &facts, &|_| {}).await;
        assert!(has_failures(&results));
        assert!(matches!(
            outcome_of(&results, "prefix cache hit"),
            Outcome::Fail(detail) if detail.contains("cached tokens")
        ));
        assert!(matches!(
            outcome_of(&results, "chat completes"),
            Outcome::Pass(_)
        ));
    }

    #[tokio::test]
    async fn tool_and_batching_checks_skip_when_they_do_not_apply() {
        let behavior = Behavior {
            engine: "sequential",
            render_tools: false,
            ..Behavior::healthy()
        };
        let endpoint = endpoint(start(behavior).await);
        let facts = ModelFacts::discover(&endpoint, "mock").await;
        let results = run_selftest(&endpoint, &facts, &|_| {}).await;
        assert!(!has_failures(&results), "{results:?}");
        assert!(matches!(
            outcome_of(&results, "forced tool call"),
            Outcome::Skip(reason) if reason.contains("tool definitions")
        ));
        assert!(matches!(
            outcome_of(&results, "batched equals solo"),
            Outcome::Skip(reason) if reason.contains("sequential")
        ));
    }

    #[tokio::test]
    async fn the_runner_measures_every_scenario() {
        let endpoint = endpoint(start(Behavior::healthy()).await);
        let plan = Plan {
            scenarios: Scenario::ALL.to_vec(),
            trials: 2,
            warmup: 0,
            max_tokens: 16,
            prefill_tokens: vec![64],
            prefix_cache_tokens: 64,
            concurrency: vec![1, 2],
            concurrency_rounds: 1,
        };
        let results = run_scenarios(&endpoint, &plan, None, &|_| {})
            .await
            .unwrap();
        let decode = results.decode.unwrap();
        assert_eq!(decode.trials.len(), 2);
        assert!(decode.decode_tokps.median > 0.0);
        let prefill = results.prefill.unwrap();
        assert_eq!(prefill.sizes.len(), 1);
        assert!(prefill.sizes[0].prefill_tokps_median > 0.0);
        let cache = results.prefix_cache.unwrap();
        assert!(cache.turn2_cached_tokens_median.unwrap_or_default() > 0.0);
        let concurrency = results.concurrency.unwrap();
        assert_eq!(concurrency.levels.len(), 2);
        assert!(concurrency.levels[1].aggregate_tokps_median > 0.0);
        assert!(results.rss_mb_peak.is_none());
    }

    #[tokio::test]
    async fn an_invalid_plan_is_rejected_before_any_request() {
        let endpoint = endpoint("http://127.0.0.1:1".to_owned());
        let plan = Plan {
            trials: 0,
            ..Plan::quick()
        };
        assert!(
            run_scenarios(&endpoint, &plan, None, &|_| {})
                .await
                .is_err()
        );
    }
}
