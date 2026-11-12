//! Scriptable in-process engine for route-level tests.

#![allow(clippy::unreachable)]

use std::sync::Mutex;

use higgs_engine::engine::{GenerationOutput, StreamingOutput};
use higgs_engine::error::EngineError;
use higgs_engine::tokenizers::Tokenizer;

/// Canned behavior for a [`StubEngine`].
#[derive(Default)]
pub struct StubScript {
    pub prompt_tokens: Vec<u32>,
    pub enable_thinking: bool,
    pub generation: Option<GenerationOutput>,
    pub stream: Vec<StreamingOutput>,
    /// Stamp the non-streaming output's missing timestamps at call time, as a
    /// real engine would, so durations are measured against the request.
    pub stamp_times: bool,
}

/// The arguments a generation call arrived with, for assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StubCall {
    pub max_tokens: u32,
    pub thinking_enabled: bool,
    pub has_constraint: bool,
    /// `Some` only when the request set its own budget.
    pub thinking_budget: Option<u32>,
}

pub struct StubEngine {
    pub name: String,
    pub tokenizer: Option<Tokenizer>,
    pub script: StubScript,
    pub calls: Mutex<Vec<StubCall>>,
}

impl StubEngine {
    pub fn named(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            tokenizer: None,
            script: StubScript::default(),
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn recorded_calls(&self) -> Vec<StubCall> {
        self.calls
            .lock()
            .map(|calls| calls.clone())
            .unwrap_or_default()
    }

    pub fn generate(&self, call: StubCall) -> Result<GenerationOutput, EngineError> {
        self.record(call);
        let mut output = self
            .script
            .generation
            .clone()
            .ok_or_else(|| EngineError::Generation("test stub".to_owned()))?;
        if self.script.stamp_times {
            let now = std::time::Instant::now();
            output.stats.prefill_started_at.get_or_insert(now);
            output.stats.first_token_at.get_or_insert(now);
            output.stats.finished_at.get_or_insert(now);
        }
        Ok(output)
    }

    pub fn generate_streaming(
        &self,
        call: StubCall,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
    ) -> Result<(), EngineError> {
        self.record(call);
        if self.script.stream.is_empty() {
            return Err(EngineError::Generation("test stub".to_owned()));
        }
        for output in &self.script.stream {
            if sender.blocking_send(output.clone()).is_err() {
                break;
            }
        }
        Ok(())
    }

    fn record(&self, call: StubCall) {
        if let Ok(mut calls) = self.calls.lock() {
            calls.push(call);
        }
    }
}

/// Minimal word-level tokenizer: each whitespace-separated word is one token.
pub fn word_tokenizer() -> Tokenizer {
    let json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "think": 3, "again": 4, "a": 5, "b": 6},
            "unk_token": "[UNK]"
        }
    }"#;
    json.parse::<Tokenizer>()
        .unwrap_or_else(|error| unreachable!("test tokenizer definition is valid: {error}"))
}

/// Router serving one scripted local model named `stub`.
pub fn app_with_stub(stub: StubEngine) -> (axum::Router, std::sync::Arc<crate::state::Engine>) {
    app_with_stub_and_config(stub, "")
}

/// Like [`app_with_stub`], with extra TOML appended to the server config.
pub fn app_with_stub_and_config(
    stub: StubEngine,
    extra_config: &str,
) -> (axum::Router, std::sync::Arc<crate::state::Engine>) {
    use std::collections::HashMap;
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap_or_else(|error| unreachable!("tempdir: {error}"));
    let config_path = dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        format!("[[models]]\npath = \"test/stub\"\nname = \"stub\"\n{extra_config}"),
    )
    .unwrap_or_else(|error| unreachable!("write config: {error}"));
    let config = crate::config::load_config_file(&config_path, None)
        .unwrap_or_else(|error| unreachable!("load config: {error}"));

    let engine = Arc::new(crate::state::Engine::scripted_stub(stub));
    let mut engines = HashMap::new();
    engines.insert("stub".to_owned(), Arc::clone(&engine));
    let router = crate::router::Router::from_config(&config, engines)
        .unwrap_or_else(|error| unreachable!("build router: {error}"));

    let state = Arc::new(crate::state::AppState {
        router,
        config,
        http_client: reqwest::Client::new(),
        metrics: None,
    });
    (
        crate::build_router(state, 30.0, None, 0, 1024 * 1024, None),
        engine,
    )
}

/// POST a JSON body and return the status with the raw response text.
pub async fn post_json(
    app: &axum::Router,
    path: &str,
    body: &serde_json::Value,
) -> (http::StatusCode, String) {
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    let request = http::Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap_or_else(|error| unreachable!("request: {error}"));
    let response = app
        .clone()
        .oneshot(request)
        .await
        .unwrap_or_else(|error| unreachable!("oneshot: {error}"));
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .map(http_body_util::Collected::to_bytes)
        .unwrap_or_default();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// Extract the JSON `data:` payloads of an SSE body, excluding `[DONE]`.
pub fn sse_json_events(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect()
}

pub fn token_event(
    text: &str,
    completion_tokens: u32,
    finish_reason: Option<&str>,
) -> StreamingOutput {
    StreamingOutput {
        new_text: text.to_owned(),
        finished: finish_reason.is_some(),
        finish_reason: finish_reason.map(str::to_owned),
        prompt_tokens: 0,
        completion_tokens,
        token_logprob: None,
        prefill_progress: None,
    }
}

pub fn progress_event(cached: u32, total: u32) -> StreamingOutput {
    StreamingOutput {
        new_text: String::new(),
        finished: false,
        finish_reason: None,
        prompt_tokens: total,
        completion_tokens: 0,
        token_logprob: None,
        prefill_progress: Some(higgs_engine::engine::PrefillProgress {
            processed: cached,
            cached,
            total,
        }),
    }
}

pub fn generation_output(
    text: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> GenerationOutput {
    GenerationOutput {
        text: text.to_owned(),
        finish_reason: "stop".to_owned(),
        prompt_tokens,
        completion_tokens,
        token_logprobs: None,
        stats: higgs_engine::engine::GenerationStats::default(),
    }
}
