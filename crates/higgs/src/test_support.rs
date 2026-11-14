//! Scriptable in-process engine for route-level tests.

#![allow(clippy::unreachable)]

use std::sync::{Arc, Mutex};

use higgs_engine::chat_template::ChatMessage;
use higgs_engine::constrained::ConstrainedGenerator;
use higgs_engine::engine::{GenerationOutput, StreamingOutput, ThinkingMode};
use higgs_engine::error::EngineError;
use higgs_engine::tokenizers::Tokenizer;
use higgs_engine::tool_format::ToolCallFormat;

use crate::backend::{Engine, EngineJob, LoadedEngine};

/// Canned behavior for a [`StubEngine`].
#[derive(Clone)]
pub struct StubScript {
    pub prompt_tokens: Vec<u32>,
    pub enable_thinking: bool,
    /// Defaults to the Qwen/Hermes format, like the engines' original behavior.
    pub tool_call_format: ToolCallFormat,
    pub generation: Option<GenerationOutput>,
    pub stream: Vec<StreamingOutput>,
    /// Per-call outputs, consulted before `generation`; the Nth call uses the
    /// Nth entry and later calls fall back to `generation`.
    pub generations: Vec<GenerationOutput>,
    /// Per-call streams, consulted before `stream`, indexed like `generations`.
    pub streams: Vec<Vec<StreamingOutput>>,
    /// Stamp the non-streaming output's missing timestamps at call time, as a
    /// real engine would, so durations are measured against the request.
    pub stamp_times: bool,
}

impl Default for StubScript {
    fn default() -> Self {
        Self {
            prompt_tokens: Vec::new(),
            enable_thinking: false,
            tool_call_format: ToolCallFormat::Qwen,
            generation: None,
            stream: Vec::new(),
            generations: Vec::new(),
            streams: Vec::new(),
            stamp_times: false,
        }
    }
}

/// The arguments a generation call arrived with, for assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StubCall {
    pub max_tokens: u32,
    pub thinking_enabled: bool,
    pub has_constraint: bool,
    /// `Some` when the call carried a constraint: whether it accepts the
    /// scripted output character by character (needs [`char_tokenizer`]).
    pub constraint_accepts_output: Option<bool>,
    /// `Some` only when the request set its own budget.
    pub thinking_budget: Option<u32>,
}

/// What a stub saw, shared between its clones.
type Recorded<T> = Arc<Mutex<Vec<T>>>;

/// The scripted engine every test serves models from. Clones share their
/// recorded calls, so a test keeps a handle after giving one to the pool.
#[derive(Clone)]
pub struct StubEngine {
    pub name: String,
    /// A word-level tokenizer unless a test supplies another.
    pub tokenizer: Tokenizer,
    pub script: StubScript,
    calls: Recorded<StubCall>,
    /// Tool count passed to each prompt rendering (`None` when no tools).
    prompt_tools: Recorded<Option<usize>>,
    /// Role and text of every message in each rendered prompt.
    prompt_messages: Recorded<Vec<(String, String)>>,
}

impl StubEngine {
    pub fn named(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            tokenizer: word_tokenizer(),
            script: StubScript::default(),
            calls: Arc::default(),
            prompt_tools: Arc::default(),
            prompt_messages: Arc::default(),
        }
    }

    pub fn into_loaded(self) -> LoadedEngine {
        Box::new(self)
    }

    pub fn recorded_calls(&self) -> Vec<StubCall> {
        self.calls
            .lock()
            .map(|calls| calls.clone())
            .unwrap_or_default()
    }

    pub fn recorded_prompt_tools(&self) -> Vec<Option<usize>> {
        self.prompt_tools
            .lock()
            .map(|prompts| prompts.clone())
            .unwrap_or_default()
    }

    pub fn recorded_prompt_messages(&self) -> Vec<Vec<(String, String)>> {
        self.prompt_messages
            .lock()
            .map(|rendered| rendered.clone())
            .unwrap_or_default()
    }

    /// Whether `constraint` accepts `text` when each character is one token.
    fn constraint_accepts(&self, mut constraint: ConstrainedGenerator, text: &str) -> bool {
        for character in text.chars() {
            let Some(id) = self.tokenizer.token_to_id(&character.to_string()) else {
                return false;
            };
            if !constraint.advance(id) {
                return false;
            }
        }
        constraint.is_finished()
    }

    fn call_for(&self, job: EngineJob, scripted_text: &str) -> StubCall {
        let has_constraint = job.constraint.is_some();
        let constraint_accepts_output = job
            .constraint
            .map(|constraint| self.constraint_accepts(constraint, scripted_text));
        StubCall {
            max_tokens: job.max_tokens,
            thinking_enabled: job.thinking.is_on(),
            has_constraint,
            constraint_accepts_output,
            thinking_budget: match job.thinking {
                ThinkingMode::On { budget } => budget,
                ThinkingMode::Off => None,
            },
        }
    }

    /// Record a generation call and return how many came before it.
    fn record(&self, call: StubCall) -> usize {
        self.calls.lock().map_or(0, |mut calls| {
            calls.push(call);
            calls.len() - 1
        })
    }
}

impl Engine for StubEngine {
    fn model_name(&self) -> &str {
        &self.name
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn eos_token_ids(&self) -> &[u32] {
        &[]
    }

    fn enable_thinking(&self) -> bool {
        self.script.enable_thinking
    }

    fn tool_call_format(&self) -> ToolCallFormat {
        self.script.tool_call_format
    }

    fn decode_kind(&self) -> &'static str {
        "sequential"
    }

    fn is_vlm(&self) -> bool {
        false
    }

    fn vlm_image_size(&self) -> Option<i32> {
        None
    }

    fn replace_image_tokens(&self, _tokens: &mut [u32]) {}

    fn prepare_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        _enable_thinking: bool,
    ) -> Result<Vec<u32>, EngineError> {
        if let Ok(mut rendered) = self.prompt_messages.lock() {
            rendered.push(
                messages
                    .iter()
                    .map(|message| (message.role.clone(), message.content.clone()))
                    .collect(),
            );
        }
        if let Ok(mut prompts) = self.prompt_tools.lock() {
            prompts.push(tools.map(<[_]>::len));
        }
        Ok(self.script.prompt_tokens.clone())
    }

    fn generate(&self, job: EngineJob) -> Result<GenerationOutput, EngineError> {
        let scripted_text = self
            .script
            .generation
            .as_ref()
            .map(|output| output.text.clone())
            .unwrap_or_default();
        let index = self.record(self.call_for(job, &scripted_text));
        let mut output = self
            .script
            .generations
            .get(index)
            .or(self.script.generation.as_ref())
            .cloned()
            .ok_or_else(|| EngineError::Generation("test stub".to_owned()))?;
        if self.script.stamp_times {
            let now = std::time::Instant::now();
            output.stats.prefill_started_at.get_or_insert(now);
            output.stats.first_token_at.get_or_insert(now);
            output.stats.finished_at.get_or_insert(now);
        }
        Ok(output)
    }

    fn generate_streaming(
        &self,
        job: EngineJob,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
    ) -> Result<(), EngineError> {
        let scripted_text: String = self
            .script
            .stream
            .iter()
            .map(|output| output.new_text.as_str())
            .collect();
        let index = self.record(self.call_for(job, &scripted_text));
        let events = self
            .script
            .streams
            .get(index)
            .unwrap_or(&self.script.stream);
        if events.is_empty() {
            return Err(EngineError::Generation("test stub".to_owned()));
        }
        for output in events {
            if sender.blocking_send(output.clone()).is_err() {
                break;
            }
        }
        Ok(())
    }

    fn embed(&self, _token_ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        Ok(Vec::new())
    }

    fn shutdown(self: Box<Self>) {}
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

/// Tokenizer with one token per printable ASCII character (and newline), so
/// any pattern over those characters can be compiled into a constraint.
pub fn char_tokenizer() -> Tokenizer {
    let mut vocab = serde_json::Map::new();
    vocab.insert("[UNK]".to_owned(), serde_json::json!(0));
    for (offset, byte) in std::iter::once(b'\n').chain(0x20..=0x7e).enumerate() {
        vocab.insert(char::from(byte).to_string(), serde_json::json!(offset + 1));
    }
    let definition = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "."}, "behavior": "Isolated", "invert": false},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    });
    definition
        .to_string()
        .parse::<Tokenizer>()
        .unwrap_or_else(|error| unreachable!("test tokenizer definition is valid: {error}"))
}

/// Router serving one scripted local model named `stub`.
pub fn app_with_stub(stub: StubEngine) -> (axum::Router, StubEngine) {
    app_with_stub_and_config(stub, "")
}

/// Like [`app_with_stub`], with extra TOML appended to the server config.
pub fn app_with_stub_and_config(
    stub: StubEngine,
    extra_config: &str,
) -> (axum::Router, StubEngine) {
    app_with_stub_config_and_tools(stub, extra_config, crate::tools::ToolHub::disabled())
}

/// Like [`app_with_stub_and_config`], with a tool hub for server-side tools.
pub fn app_with_stub_config_and_tools(
    stub: StubEngine,
    extra_config: &str,
    tools: crate::tools::ToolHub,
) -> (axum::Router, StubEngine) {
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

    let engine = stub.clone();
    let mut engines = HashMap::new();
    engines.insert("stub".to_owned(), Arc::new(stub.into_loaded()));
    let router = crate::router::Router::from_config(&config, engines)
        .unwrap_or_else(|error| unreachable!("build router: {error}"));

    let state = Arc::new(crate::state::AppState {
        router,
        config,
        http_client: reqwest::Client::new(),
        metrics: None,
        tools: Arc::new(tools),
    });
    (
        crate::build_router(state, 30.0, None, 0, 1024 * 1024, None),
        engine,
    )
}

/// Router over a stub model with the given tool hub, optional API key and
/// metrics store, for exercising tool endpoints.
pub fn app_with_tools(
    tools: crate::tools::ToolHub,
    api_key: Option<&str>,
    metrics: Option<std::sync::Arc<crate::metrics::MetricsStore>>,
) -> axum::Router {
    use std::collections::HashMap;
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap_or_else(|error| unreachable!("tempdir: {error}"));
    let config_path = dir.path().join("config.toml");
    std::fs::write(
        &config_path,
        "[[models]]\npath = \"test/stub\"\nname = \"stub\"\n",
    )
    .unwrap_or_else(|error| unreachable!("write config: {error}"));
    let config = crate::config::load_config_file(&config_path, None)
        .unwrap_or_else(|error| unreachable!("load config: {error}"));
    let mut engines = HashMap::new();
    engines.insert(
        "stub".to_owned(),
        Arc::new(StubEngine::named("stub").into_loaded()),
    );
    let router = crate::router::Router::from_config(&config, engines)
        .unwrap_or_else(|error| unreachable!("build router: {error}"));
    let state = Arc::new(crate::state::AppState {
        router,
        config,
        http_client: reqwest::Client::new(),
        metrics,
        tools: Arc::new(tools),
    });
    crate::build_router(
        state,
        30.0,
        api_key.map(str::to_owned),
        0,
        1024 * 1024,
        None,
    )
}

/// Send a request with an optional JSON body and bearer token.
pub async fn send_request(
    app: &axum::Router,
    method: &str,
    path: &str,
    body: Option<&serde_json::Value>,
    bearer: Option<&str>,
) -> (http::StatusCode, serde_json::Value) {
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    let mut builder = http::Request::builder().method(method).uri(path);
    if let Some(token) = bearer {
        builder = builder.header("authorization", format!("Bearer {token}"));
    }
    let request = match body {
        Some(value) => builder
            .header("content-type", "application/json")
            .body(axum::body::Body::from(value.to_string())),
        None => builder.body(axum::body::Body::empty()),
    }
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
    let parsed = serde_json::from_slice(&bytes).unwrap_or_else(|_| {
        serde_json::Value::String(String::from_utf8_lossy(&bytes).into_owned())
    });
    (status, parsed)
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

/// The terminal event an engine sends when its worker failed mid-generation.
pub fn failure_event() -> StreamingOutput {
    token_event("", 0, Some("error"))
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
