//! Server-side tool loop: let a model call MCP and built-in tools and
//! continue the generation with the results.
//!
//! A request opts in with `"mcp": {"enabled": true}`. The selected tools are
//! added to the request's `tools`; when the model answers with calls to them
//! the calls run on the server, their results are appended to the
//! conversation as `tool` messages and the model generates again, up to
//! `max_iterations` times. Calls to tools the client supplied end the loop
//! and are handed back unchanged, so a client keeps control of its own tools.
//!
//! Generation itself is [`complete_once`] and [`stream_once`]: this module
//! only runs them once per round and reads their output.

use std::collections::HashSet;
use std::time::Instant;

use higgs_engine::chat_template::ChatMessage;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_stream::StreamExt;

use super::GenerationRequest;
use super::complete::complete_once;
use super::event::{Completed, EventStream, GeneratedToolCall, GenerationEvent, ToolLoopEvent};
use super::messages::{template_tool_call, wire_tool_call};
use super::stream::{StreamMetrics, stream_once};
use crate::{
    backend::EngineLease,
    error::ServerError,
    mcp::manager::ToolOutput,
    router::ResolvedRoute,
    state::{AppState, SharedState},
    tools::openai_tool,
    types::openai::CompletionUsage,
};

pub const DEFAULT_MAX_ITERATIONS: u32 = 5;
pub const MAX_ITERATIONS_LIMIT: u32 = 25;
/// How much of a tool result the streaming events echo to the client.
const RESULT_PREVIEW_CHARS: usize = 1000;

/// The `mcp` request field.
#[derive(Debug, Clone, Deserialize)]
pub struct McpOptions {
    /// Defaults to true: sending the object at all is the opt-in.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Only offer tools of these servers (`higgs` selects the built-in web
    /// tools). Default: every available tool.
    #[serde(default)]
    pub servers: Option<Vec<String>>,
    /// How many times tool calls may be executed before the loop stops and
    /// returns the model's pending calls. Default 5.
    #[serde(default)]
    pub max_iterations: Option<u32>,
}

const fn default_enabled() -> bool {
    true
}

pub(super) struct LoopPlan {
    /// Tools to add to the request, in `OpenAI` form.
    injected_tools: Vec<Value>,
    /// Names the server executes itself.
    names: HashSet<String>,
    max_iterations: u32,
}

const fn bad_request(message: String) -> ServerError {
    ServerError::BadRequest(message)
}

fn client_tool_names(tools: &[Value]) -> HashSet<String> {
    tools
        .iter()
        .filter_map(|tool| {
            tool.get("function")
                .unwrap_or(tool)
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect()
}

/// Decide whether this request runs the loop and with which tools. `None`
/// means the request is handled exactly as if `mcp` were absent.
pub(super) fn plan(
    state: &AppState,
    request: &GenerationRequest,
) -> Result<Option<LoopPlan>, ServerError> {
    let Some(options) = request.mcp.as_ref().filter(|options| options.enabled) else {
        return Ok(None);
    };
    let Some(chat) = request.chat() else {
        return Ok(None);
    };
    if !chat.tool_policy.exposes_tools() {
        return Ok(None);
    }
    let max_iterations = options.max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS);
    if !(1..=MAX_ITERATIONS_LIMIT).contains(&max_iterations) {
        return Err(bad_request(format!(
            "mcp.max_iterations must be between 1 and {MAX_ITERATIONS_LIMIT}"
        )));
    }

    let available = state.tools.list();
    if let Some(servers) = &options.servers {
        let mut known: HashSet<String> = state
            .tools
            .mcp
            .statuses()
            .into_iter()
            .map(|status| status.name)
            .collect();
        known.extend(available.iter().map(|tool| tool.server.clone()));
        if let Some(unknown) = servers.iter().find(|name| !known.contains(name.as_str())) {
            return Err(bad_request(format!(
                "mcp.servers: unknown MCP server '{unknown}'"
            )));
        }
    }

    // A client tool with the same name as a server tool stays the client's.
    let taken = client_tool_names(&chat.tools);
    let selected: Vec<_> = available
        .iter()
        .filter(|tool| {
            options
                .servers
                .as_ref()
                .is_none_or(|servers| servers.contains(&tool.server))
        })
        .filter(|tool| !taken.contains(&tool.name))
        .collect();

    let mut injected_tools = chat.tools.clone();
    injected_tools.extend(selected.iter().map(|tool| openai_tool(tool)));
    Ok(Some(LoopPlan {
        injected_tools,
        names: selected.iter().map(|tool| tool.name.clone()).collect(),
        max_iterations,
    }))
}

/// The request as the loop's rounds send it: the loop's tools added and no
/// nested loop.
fn loop_conversation(mut request: GenerationRequest, plan: &LoopPlan) -> GenerationRequest {
    request.mcp = None;
    if let Some(chat) = request.chat_mut() {
        chat.tools.clone_from(&plan.injected_tools);
    }
    request
}

/// Prepare the conversation for the round after one that ran tools: a forced
/// `tool_choice` applied to the first generation only, so later rounds are
/// free to answer.
fn extend_conversation(
    conversation: &mut GenerationRequest,
    text: String,
    calls: &[GeneratedToolCall],
) {
    if let Some(chat) = conversation.chat_mut() {
        chat.tool_policy = chat.tool_policy.without_forced_choice();
        chat.messages.push(ChatMessage {
            role: "assistant".to_owned(),
            content: text,
            tool_calls: Some(
                calls
                    .iter()
                    .filter_map(|call| template_tool_call(&wire_tool_call(call)))
                    .collect(),
            ),
        });
    }
}

fn push_tool_result(conversation: &mut GenerationRequest, call: &ExecutedCall) {
    if let Some(chat) = conversation.chat_mut() {
        chat.messages.push(ChatMessage {
            role: "tool".to_owned(),
            content: call.content.clone(),
            tool_calls: None,
        });
    }
}

async fn reacquire(state: &AppState, model: &str) -> Result<EngineLease, ServerError> {
    match state.router.resolve(model, None).await? {
        ResolvedRoute::Higgs { engine, .. } => Ok(engine),
        ResolvedRoute::Remote { .. } => Err(ServerError::InternalError(
            "model route changed during the tool loop".to_owned(),
        )),
    }
}

/// One executed tool call, for the conversation and the trace.
struct ExecutedCall {
    id: String,
    name: String,
    arguments: Value,
    content: String,
    is_error: bool,
}

async fn execute_call(state: &AppState, call: &GeneratedToolCall) -> ExecutedCall {
    let result = state.tools.call(&call.name, call.arguments.clone()).await;
    let (content, is_error) = match result
        .map(|output| render_output(&output))
        .map_err(|error| error.to_string())
    {
        Ok((text, is_error)) => (text, is_error),
        Err(message) => (format!("Error: {message}"), true),
    };
    ExecutedCall {
        id: call.id.clone(),
        name: call.name.clone(),
        arguments: call.arguments.clone(),
        content,
        is_error,
    }
}

/// The text a model sees for a tool result. Non-text blocks cannot be shown
/// to a text model, so they are summarized.
fn render_output(output: &ToolOutput) -> (String, bool) {
    let mut lines: Vec<String> = Vec::new();
    if !output.text.is_empty() {
        lines.push(output.text.clone());
    }
    for block in &output.content {
        let kind = block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        if kind == "text" {
            continue;
        }
        let detail = block
            .get("mimeType")
            .or_else(|| {
                block
                    .get("resource")
                    .and_then(|resource| resource.get("uri"))
            })
            .and_then(Value::as_str)
            .unwrap_or_default();
        lines.push(format!("[{kind} content omitted: {detail}]"));
    }
    if lines.is_empty()
        && let Some(structured) = &output.structured_content
    {
        lines.push(structured.to_string());
    }
    if lines.is_empty() {
        lines.push("(the tool returned no output)".to_owned());
    }
    if output.truncated {
        lines.push("[output truncated]".to_owned());
    }
    (lines.join("\n"), output.is_error)
}

#[derive(Default)]
struct Trace {
    calls: Vec<Value>,
    rounds: u32,
    truncated: bool,
}

impl Trace {
    fn record(&mut self, round: u32, call: &ExecutedCall) {
        self.calls.push(json!({
            "round": round,
            "id": call.id,
            "name": call.name,
            "arguments": call.arguments,
            "is_error": call.is_error,
        }));
    }

    fn to_value(&self) -> Value {
        json!({
            "iterations": self.rounds,
            "truncated": self.truncated,
            "tool_calls": self.calls,
        })
    }

    const fn ran_tools(&self) -> bool {
        self.rounds > 0 || self.truncated
    }
}

fn split_calls(
    calls: Vec<GeneratedToolCall>,
    names: &HashSet<String>,
) -> (Vec<GeneratedToolCall>, Vec<GeneratedToolCall>) {
    calls
        .into_iter()
        .partition(|call| names.contains(&call.name))
}

/// Non-streaming generation with the tool loop.
pub(super) async fn complete(
    state: SharedState,
    request: GenerationRequest,
    engine: EngineLease,
    plan: LoopPlan,
    received_at: Instant,
) -> Result<Completed, ServerError> {
    let mut conversation = loop_conversation(request, &plan);
    let mut first_lease = Some(engine);
    let mut trace = Trace::default();
    let mut totals: (u32, u32) = (0, 0);

    for round in 0..=plan.max_iterations {
        let lease = match first_lease.take() {
            Some(lease) => lease,
            None => reacquire(&state, &conversation.model).await?,
        };
        let mut completed = complete_once(lease, &conversation, received_at).await?;

        let (previous_prompt, previous_completion) = totals;
        totals = (
            previous_prompt.saturating_add(completed.usage.prompt_tokens),
            previous_completion.saturating_add(completed.usage.completion_tokens),
        );
        completed.usage.prompt_tokens = totals.0;
        completed.usage.completion_tokens = totals.1;
        completed.usage.total_tokens = totals.0.saturating_add(totals.1);

        let (server_calls, client_calls) = split_calls(completed.tool_calls.clone(), &plan.names);
        if server_calls.is_empty() {
            return Ok(finish(completed, &trace));
        }
        let exhausted = round == plan.max_iterations;
        if !client_calls.is_empty() {
            completed.tool_calls = client_calls;
            return Ok(finish(completed, &trace));
        }
        if exhausted {
            trace.truncated = true;
            return Ok(finish(completed, &trace));
        }

        trace.rounds += 1;
        extend_conversation(&mut conversation, completed.text, &server_calls);
        for call in &server_calls {
            let executed = execute_call(&state, call).await;
            trace.record(trace.rounds, &executed);
            push_tool_result(&mut conversation, &executed);
        }
    }
    Err(ServerError::InternalError(
        "tool loop ended without a response".to_owned(),
    ))
}

fn finish(mut completed: Completed, trace: &Trace) -> Completed {
    if trace.ran_tools() {
        completed.tool_loop = Some(trace.to_value());
    }
    completed
}

/// What a stream round has seen so far.
#[derive(Default)]
struct RoundState {
    /// Only the first round announces the start of the response.
    later_round: bool,
    server_calls: Vec<GeneratedToolCall>,
    text: String,
    client_call_forwarded: bool,
    next_call_index: u32,
    held_finish: Option<GenerationEvent>,
    held_usage: Option<CompletionUsage>,
}

impl RoundState {
    /// Route one event of the round: `Some` is forwarded to the client,
    /// `None` is suppressed (or held for the end of the round).
    fn ingest(
        &mut self,
        event: GenerationEvent,
        names: &HashSet<String>,
        final_round: bool,
    ) -> Option<GenerationEvent> {
        match event {
            GenerationEvent::Usage(usage) => {
                self.held_usage = Some(usage);
                None
            }
            GenerationEvent::Finish { .. } => {
                self.held_finish = Some(event);
                None
            }
            GenerationEvent::TextDelta { ref text, .. } => {
                self.text.push_str(text);
                Some(event)
            }
            GenerationEvent::ToolCall(call) => {
                if !final_round && names.contains(&call.name) {
                    self.server_calls.push(call);
                    return None;
                }
                if !names.contains(&call.name) {
                    self.client_call_forwarded = true;
                }
                let renumbered = GeneratedToolCall {
                    index: self.next_call_index,
                    ..call
                };
                self.next_call_index += 1;
                Some(GenerationEvent::ToolCall(renumbered))
            }
            GenerationEvent::Started { .. } if self.later_round => None,
            GenerationEvent::Started { .. }
            | GenerationEvent::PrefillProgress(_)
            | GenerationEvent::ReasoningDelta(_)
            | GenerationEvent::ToolLoop(_)
            | GenerationEvent::Error(_) => Some(event),
        }
    }
}

/// Add the token counts of `next` into `total`; other usage fields (timings,
/// details) come from the latest round.
const fn merge_usage(total: &mut Option<CompletionUsage>, next: CompletionUsage) {
    let Some(previous) = total.take() else {
        *total = Some(next);
        return;
    };
    let mut merged = next;
    merged.prompt_tokens = merged.prompt_tokens.saturating_add(previous.prompt_tokens);
    merged.completion_tokens = merged
        .completion_tokens
        .saturating_add(previous.completion_tokens);
    merged.total_tokens = merged.total_tokens.saturating_add(previous.total_tokens);
    *total = Some(merged);
}

/// Streaming generation with the tool loop.
///
/// Assistant text and reasoning stream as they are produced in every round.
/// Executed tool calls never appear as `ToolCall` events; instead
/// [`ToolLoopEvent`]s announce each call, its outcome and the loop's end.
/// Preparation errors of any round, including the first, end the stream with
/// an `Error` event.
pub(super) fn stream(
    state: SharedState,
    request: GenerationRequest,
    engine: EngineLease,
    plan: LoopPlan,
    metrics: Option<StreamMetrics>,
    received_at: Instant,
) -> EventStream {
    let mut conversation = loop_conversation(request, &plan);

    Box::pin(async_stream::stream! {
        let mut first_lease = Some(engine);
        let mut trace = Trace::default();
        let mut usage_total: Option<CompletionUsage> = None;
        let mut round: u32 = 0;

        'rounds: loop {
            let final_round = round >= plan.max_iterations;
            let lease = match first_lease.take() {
                Some(lease) => lease,
                None => match reacquire(&state, &conversation.model).await {
                    Ok(lease) => lease,
                    Err(error) => {
                        yield loop_failure(&error);
                        break 'rounds;
                    }
                },
            };
            let round_metrics = if round == 0 { metrics.clone() } else { None };
            let mut events = match stream_once(lease, &conversation, received_at, round_metrics) {
                Ok(events) => events,
                Err(error) => {
                    yield loop_failure(&error);
                    break 'rounds;
                }
            };

            let mut state_of_round = RoundState {
                later_round: round > 0,
                ..RoundState::default()
            };
            while let Some(event) = events.next().await {
                if let Some(forward) = state_of_round.ingest(event, &plan.names, final_round) {
                    yield forward;
                }
            }
            drop(events);

            if let Some(usage) = state_of_round.held_usage.take() {
                merge_usage(&mut usage_total, usage);
            }
            let run_tools = !state_of_round.server_calls.is_empty()
                && !state_of_round.client_call_forwarded
                && !final_round;
            if !run_tools {
                trace.truncated = final_round
                    && !state_of_round.client_call_forwarded
                    && matches!(
                        &state_of_round.held_finish,
                        Some(GenerationEvent::Finish { reason, .. }) if reason == "tool_calls"
                    );
                if let Some(finish) = state_of_round.held_finish.take() {
                    yield finish;
                }
                break 'rounds;
            }

            round += 1;
            trace.rounds = round;
            let calls = std::mem::take(&mut state_of_round.server_calls);
            extend_conversation(
                &mut conversation,
                std::mem::take(&mut state_of_round.text),
                &calls,
            );
            for call in &calls {
                yield GenerationEvent::ToolLoop(ToolLoopEvent::Call {
                    round,
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                });
                let executed = execute_call(&state, call).await;
                trace.record(round, &executed);
                yield GenerationEvent::ToolLoop(ToolLoopEvent::Result {
                    round,
                    id: executed.id.clone(),
                    name: executed.name.clone(),
                    is_error: executed.is_error,
                    content: executed.content.chars().take(RESULT_PREVIEW_CHARS).collect(),
                });
                push_tool_result(&mut conversation, &executed);
            }
        }

        if let Some(usage) = usage_total {
            yield GenerationEvent::Usage(usage);
        }
        if trace.ran_tools() {
            yield GenerationEvent::ToolLoop(ToolLoopEvent::Summary(trace.to_value()));
        }
    })
}

fn loop_failure(error: &ServerError) -> GenerationEvent {
    tracing::error!(%error, "tool loop failed mid-stream");
    GenerationEvent::Error("tool loop failed".to_owned())
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use http::StatusCode;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::mcp::manager::McpManager;
    use crate::mcp::testing::{CallBehavior, FakeMcp, ready_manager, text_result};
    use crate::test_support::{
        StubEngine, app_with_stub_config_and_tools, generation_output, post_json, sse_json_events,
        token_event,
    };
    use crate::tools::ToolHub;
    use crate::web::WebService;
    use crate::web::fetch::WebFetcher;
    use crate::web::search::{SearchProvider, WebSearcher};
    use crate::web::ssrf::AddressPolicy;

    const CALL_ECHO: &str =
        r#"<tool_call>{"name": "files__echo", "arguments": {"text": "hi"}}</tool_call>"#;
    const CALL_CLIENT: &str =
        r#"<tool_call>{"name": "get_weather", "arguments": {"city": "Oslo"}}</tool_call>"#;

    fn stub(generations: Vec<(&str, u32, u32)>) -> StubEngine {
        let mut stub = StubEngine::named("stub");
        stub.script.prompt_tokens = vec![1, 2, 3];
        stub.script.generations = generations
            .into_iter()
            .map(|(text, prompt, completion)| generation_output(text, prompt, completion))
            .collect();
        stub
    }

    async fn files_hub(fake: &Arc<FakeMcp>, timeout: Duration) -> ToolHub {
        ToolHub::new(
            ready_manager(vec![("files", Arc::clone(fake), timeout)], 1).await,
            WebService::disabled(),
        )
    }

    async fn chat(
        stub: StubEngine,
        hub: ToolHub,
        body: &serde_json::Value,
    ) -> (StatusCode, serde_json::Value, StubEngine) {
        let (app, engine) = app_with_stub_config_and_tools(stub, "", hub);
        let (status, text) = post_json(&app, "/v1/chat/completions", body).await;
        let parsed = serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text));
        (status, parsed, engine)
    }

    fn request(extra: serde_json::Value) -> serde_json::Value {
        let mut body = json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "use the tool"}],
        });
        if let (Some(base), serde_json::Value::Object(additions)) = (body.as_object_mut(), extra) {
            base.extend(additions);
        }
        body
    }

    fn echo_fake() -> Arc<FakeMcp> {
        let fake = FakeMcp::new(&["echo", "other"]);
        fake.on_call("echo", CallBehavior::Reply(text_result("pong")));
        fake
    }

    #[tokio::test]
    async fn executes_the_tool_and_continues_to_a_final_answer() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let (status, body, engine) = chat(
            stub(vec![(CALL_ECHO, 5, 10), ("The answer is pong", 7, 3)]),
            hub,
            &request(json!({"mcp": {"enabled": true}})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let choice = &body["choices"][0];
        assert_eq!(choice["message"]["content"], "The answer is pong");
        assert_eq!(choice["finish_reason"], "stop");
        assert!(choice["message"].get("tool_calls").is_none());

        assert_eq!(body["mcp"]["iterations"], 1);
        assert_eq!(body["mcp"]["truncated"], false);
        assert_eq!(body["mcp"]["tool_calls"][0]["name"], "files__echo");
        assert_eq!(
            body["mcp"]["tool_calls"][0]["arguments"],
            json!({"text": "hi"})
        );
        assert_eq!(body["mcp"]["tool_calls"][0]["is_error"], false);

        assert_eq!(body["usage"]["prompt_tokens"], 12);
        assert_eq!(body["usage"]["completion_tokens"], 13);
        assert_eq!(body["usage"]["total_tokens"], 25);

        assert_eq!(
            fake.received_calls(),
            vec![json!({"name": "echo", "arguments": {"text": "hi"}})]
        );
        assert_eq!(engine.recorded_calls().len(), 2);
        assert_eq!(engine.recorded_prompt_tools(), vec![Some(2), Some(2)]);
        let prompts = engine.recorded_prompt_messages();
        assert_eq!(prompts[0].len(), 1);
        let roles: Vec<_> = prompts[1].iter().map(|(role, _)| role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool"]);
        assert_eq!(prompts[1][2].1, "pong");
    }

    #[tokio::test]
    async fn loop_uses_the_engines_tool_call_format() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let mut llama = stub(vec![
            (
                r#"{"name": "files__echo", "parameters": {"text": "hi"}}"#,
                5,
                10,
            ),
            ("done", 1, 1),
        ]);
        llama.script.tool_call_format = higgs_engine::tool_format::ToolCallFormat::Llama;
        let (status, body, _) = chat(llama, hub, &request(json!({"mcp": {}}))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["choices"][0]["message"]["content"], "done");
        assert_eq!(fake.received_calls().len(), 1);
    }

    #[tokio::test]
    async fn a_forced_tool_choice_applies_to_the_first_round_only() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let mut forced = stub(vec![(CALL_ECHO, 5, 10), ("done", 1, 1)]);
        forced.tokenizer = crate::test_support::char_tokenizer();
        let (status, body, engine) = chat(
            forced,
            hub,
            &request(json!({
                "mcp": {},
                "tool_choice": {"type": "function", "function": {"name": "files__echo"}}
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["choices"][0]["message"]["content"], "done");
        let constrained: Vec<_> = engine
            .recorded_calls()
            .iter()
            .map(|c| c.has_constraint)
            .collect();
        assert_eq!(constrained, vec![true, false]);
    }

    #[tokio::test]
    async fn requests_without_mcp_are_untouched() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let (status, body, engine) =
            chat(stub(vec![(CALL_ECHO, 5, 10)]), hub, &request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.get("mcp").is_none());
        assert_eq!(engine.recorded_prompt_tools(), vec![None]);
        assert_eq!(engine.recorded_calls().len(), 1);
        assert!(fake.received_calls().is_empty());

        let disabled = files_hub(&fake, Duration::from_secs(2)).await;
        let (_, off_body, off_engine) = chat(
            stub(vec![(CALL_ECHO, 5, 10)]),
            disabled,
            &request(json!({"mcp": {"enabled": false}})),
        )
        .await;
        assert!(off_body.get("mcp").is_none());
        assert_eq!(off_engine.recorded_prompt_tools(), vec![None]);
    }

    #[tokio::test]
    async fn tool_choice_none_disables_the_loop() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let (status, _, engine) = chat(
            stub(vec![(CALL_ECHO, 5, 10)]),
            hub,
            &request(json!({"mcp": {"enabled": true}, "tool_choice": "none"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(engine.recorded_prompt_tools(), vec![None]);
        assert!(fake.received_calls().is_empty());
    }

    #[tokio::test]
    async fn stops_at_max_iterations_and_returns_the_pending_call() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let mut always_calls = stub(vec![]);
        always_calls.script.generation = Some(generation_output(CALL_ECHO, 5, 10));
        let (status, body, engine) = chat(
            always_calls,
            hub,
            &request(json!({"mcp": {"max_iterations": 2}})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(engine.recorded_calls().len(), 3);
        assert_eq!(fake.received_calls().len(), 2);
        let choice = &body["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls");
        assert_eq!(
            choice["message"]["tool_calls"][0]["function"]["name"],
            "files__echo"
        );
        assert_eq!(body["mcp"]["iterations"], 2);
        assert_eq!(body["mcp"]["truncated"], true);
        assert_eq!(body["mcp"]["tool_calls"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn default_max_iterations_is_five() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let mut always_calls = stub(vec![]);
        always_calls.script.generation = Some(generation_output(CALL_ECHO, 1, 1));
        let (_, body, engine) = chat(always_calls, hub, &request(json!({"mcp": {}}))).await;
        assert_eq!(engine.recorded_calls().len(), 6);
        assert_eq!(fake.received_calls().len(), 5);
        assert_eq!(body["mcp"]["truncated"], true);
    }

    #[tokio::test]
    async fn client_tools_are_handed_back_without_running_anything() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let weather = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]);
        let (status, body, engine) = chat(
            stub(vec![(CALL_CLIENT, 5, 10)]),
            hub,
            &request(json!({"mcp": {"enabled": true}, "tools": weather})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(engine.recorded_prompt_tools(), vec![Some(3)]);
        assert_eq!(engine.recorded_calls().len(), 1);
        assert!(fake.received_calls().is_empty());
        let choice = &body["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls");
        assert_eq!(
            choice["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert!(body.get("mcp").is_none());
    }

    #[tokio::test]
    async fn a_mixed_batch_returns_only_the_client_calls() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let weather = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]);
        let both = format!("{CALL_ECHO}{CALL_CLIENT}");
        let (_, body, _) = chat(
            stub(vec![(&both, 5, 10)]),
            hub,
            &request(json!({"mcp": {"enabled": true}, "tools": weather})),
        )
        .await;
        let calls = body["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert!(fake.received_calls().is_empty());
    }

    #[tokio::test]
    async fn a_client_tool_shadowing_a_server_tool_stays_the_clients() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let shadow = json!([{"type": "function", "function": {"name": "files__echo", "parameters": {"type": "object"}}}]);
        let (_, body, engine) = chat(
            stub(vec![(CALL_ECHO, 5, 10)]),
            hub,
            &request(json!({"mcp": {"enabled": true}, "tools": shadow})),
        )
        .await;
        assert_eq!(engine.recorded_prompt_tools(), vec![Some(2)]);
        assert!(fake.received_calls().is_empty());
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "files__echo"
        );
    }

    #[tokio::test]
    async fn servers_filter_selects_which_tools_are_offered() {
        let files = echo_fake();
        let notes = FakeMcp::new(&["search"]);
        let manager = ready_manager(
            vec![
                ("files", Arc::clone(&files), Duration::from_secs(2)),
                ("notes", Arc::clone(&notes), Duration::from_secs(2)),
            ],
            2,
        )
        .await;
        let hub = ToolHub::new(manager, WebService::disabled());
        let (app, engine) =
            app_with_stub_config_and_tools(stub(vec![("plain answer", 1, 1)]), "", hub);
        let (status, text) = post_json(
            &app,
            "/v1/chat/completions",
            &request(json!({"mcp": {"servers": ["notes"]}})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{text}");
        assert_eq!(engine.recorded_prompt_tools(), vec![Some(1)]);

        let (unknown_status, unknown_text) = post_json(
            &app,
            "/v1/chat/completions",
            &request(json!({"mcp": {"servers": ["nope"]}})),
        )
        .await;
        assert_eq!(unknown_status, StatusCode::BAD_REQUEST);
        assert!(
            unknown_text.contains("unknown MCP server 'nope'"),
            "{unknown_text}"
        );
    }

    #[tokio::test]
    async fn invalid_max_iterations_is_a_400_before_generation() {
        for value in [0, 26] {
            let fake = echo_fake();
            let hub = files_hub(&fake, Duration::from_secs(2)).await;
            let (status, body, engine) = chat(
                stub(vec![("x", 1, 1)]),
                hub,
                &request(json!({"mcp": {"max_iterations": value}})),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{value}: {body}");
            assert!(engine.recorded_calls().is_empty());
        }
    }

    #[tokio::test]
    async fn tool_errors_and_timeouts_reach_the_model_as_tool_messages() {
        let fake = FakeMcp::new(&["broken", "slow"]);
        fake.on_call(
            "broken",
            CallBehavior::Reply(
                json!({"content": [{"type": "text", "text": "disk full"}], "isError": true}),
            ),
        );
        fake.on_call("slow", CallBehavior::Hang);
        let hub = files_hub(&fake, Duration::from_millis(200)).await;
        let calls = r#"<tool_call>{"name": "files__broken", "arguments": {}}</tool_call><tool_call>{"name": "files__slow", "arguments": {}}</tool_call>"#;
        let (status, body, engine) = chat(
            stub(vec![(calls, 1, 1), ("recovered", 1, 1)]),
            hub,
            &request(json!({"mcp": {}})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["choices"][0]["message"]["content"], "recovered");
        assert_eq!(body["mcp"]["tool_calls"][0]["is_error"], true);
        assert_eq!(body["mcp"]["tool_calls"][1]["is_error"], true);
        let tool_messages: Vec<_> = engine.recorded_prompt_messages()[1]
            .iter()
            .filter(|(role, _)| role == "tool")
            .map(|(_, content)| content.clone())
            .collect();
        assert_eq!(tool_messages.len(), 2);
        assert_eq!(tool_messages[0], "disk full");
        assert!(
            tool_messages[1].starts_with("Error: tool call timed out"),
            "{tool_messages:?}"
        );
    }

    #[tokio::test]
    async fn non_text_results_are_summarized_for_the_model() {
        let fake = FakeMcp::new(&["shot"]);
        fake.on_call(
            "shot",
            CallBehavior::Reply(json!({"content": [
                {"type": "text", "text": "caption"},
                {"type": "image", "data": "AAAA", "mimeType": "image/png"}
            ]})),
        );
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let call = r#"<tool_call>{"name": "files__shot", "arguments": {}}</tool_call>"#;
        let (_, _, engine) = chat(
            stub(vec![(call, 1, 1), ("done", 1, 1)]),
            hub,
            &request(json!({"mcp": {}})),
        )
        .await;
        assert_eq!(
            engine.recorded_prompt_messages()[1][2].1,
            "caption\n[image content omitted: image/png]"
        );
    }

    #[tokio::test]
    async fn built_in_web_tools_work_in_the_loop() {
        let search = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [{"title": "Higgs", "url": "https://example.com/higgs", "content": "A server"}]
            })))
            .mount(&search)
            .await;
        let searcher = WebSearcher::new(
            SearchProvider::SearxNg {
                base: reqwest::Url::parse(&search.uri()).unwrap(),
            },
            Duration::from_secs(5),
        )
        .unwrap();
        let fetcher = WebFetcher::new(AddressPolicy::default(), Duration::from_secs(5));
        let hub = ToolHub::new(
            McpManager::empty(),
            WebService::with_parts(Some(searcher), Some(fetcher), true),
        );
        let call = r#"<tool_call>{"name": "higgs__web_search", "arguments": {"query": "higgs"}}</tool_call>"#;
        let (status, body, engine) = chat(
            stub(vec![(call, 1, 1), ("It is a server.", 1, 1)]),
            hub,
            &request(json!({"mcp": {"servers": ["higgs"]}})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(engine.recorded_prompt_tools(), vec![Some(2), Some(2)]);
        assert!(
            engine.recorded_prompt_messages()[1][2]
                .1
                .contains("https://example.com/higgs")
        );
        assert_eq!(body["choices"][0]["message"]["content"], "It is a server.");
    }

    fn streaming_stub() -> StubEngine {
        let mut stub = stub(vec![]);
        stub.script.streams = vec![
            vec![
                token_event("Let me check. ", 3, None),
                token_event(CALL_ECHO, 12, Some("stop")),
            ],
            vec![token_event("Final answer", 5, Some("stop"))],
        ];
        stub
    }

    async fn stream_chat(
        stub: StubEngine,
        hub: ToolHub,
        extra: serde_json::Value,
    ) -> (StatusCode, String, StubEngine) {
        let (app, engine) = app_with_stub_config_and_tools(stub, "", hub);
        let mut body = request(extra);
        body["stream"] = json!(true);
        let (status, text) = post_json(&app, "/v1/chat/completions", &body).await;
        (status, text, engine)
    }

    fn content_of(events: &[serde_json::Value]) -> String {
        events
            .iter()
            .filter_map(|event| {
                event
                    .pointer("/choices/0/delta/content")
                    .and_then(|c| c.as_str())
            })
            .collect()
    }

    #[tokio::test]
    async fn streaming_forwards_text_and_reports_tool_activity_out_of_band() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let (status, text, engine) = stream_chat(
            streaming_stub(),
            hub,
            json!({"mcp": {}, "stream_options": {"include_usage": true}}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let events = sse_json_events(&text);

        assert_eq!(content_of(&events), "Let me check. Final answer");
        assert!(
            events
                .iter()
                .all(|event| event.pointer("/choices/0/delta/tool_calls").is_none()),
            "server tool calls must not reach the client: {text}"
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| e.pointer("/choices/0/delta/role").is_some())
                .count(),
            1
        );
        let finishes: Vec<_> = events
            .iter()
            .filter_map(|e| {
                e.pointer("/choices/0/finish_reason")
                    .and_then(|r| r.as_str())
            })
            .collect();
        assert_eq!(finishes, vec!["stop"]);

        let vendor: Vec<_> = events
            .iter()
            .filter(|e| e.get("higgs_mcp").is_some())
            .collect();
        let kinds: Vec<_> = vendor
            .iter()
            .map(|e| e["higgs_mcp"]["event"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, vec!["tool_call", "tool_result", "summary"]);
        assert_eq!(vendor[0]["higgs_mcp"]["name"], "files__echo");
        assert_eq!(vendor[0]["higgs_mcp"]["arguments"], json!({"text": "hi"}));
        assert_eq!(vendor[1]["higgs_mcp"]["content"], "pong");
        assert_eq!(vendor[1]["higgs_mcp"]["is_error"], false);
        assert_eq!(vendor[2]["higgs_mcp"]["iterations"], 1);
        assert!(vendor.iter().all(|e| e["choices"] == json!([])));

        let usage = events
            .iter()
            .find(|e| e.get("usage").is_some_and(|u| !u.is_null()))
            .unwrap();
        assert_eq!(usage["usage"]["prompt_tokens"], 6);
        assert_eq!(usage["usage"]["completion_tokens"], 17);
        assert_eq!(usage["usage"]["total_tokens"], 23);

        let ids: std::collections::HashSet<_> =
            events.iter().filter_map(|e| e["id"].as_str()).collect();
        assert_eq!(ids.len(), 1, "{ids:?}");
        assert!(text.trim_end().ends_with("data: [DONE]"));
        assert_eq!(text.matches("[DONE]").count(), 1);

        assert_eq!(fake.received_calls().len(), 1);
        let prompts = engine.recorded_prompt_messages();
        assert_eq!(prompts[1][1].0, "assistant");
        assert!(prompts[1][1].1.contains("Let me check."));
        assert_eq!(prompts[1][2], ("tool".to_owned(), "pong".to_owned()));
    }

    #[tokio::test]
    async fn streaming_hands_back_client_tool_calls() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let mut stub = stub(vec![]);
        stub.script.streams = vec![vec![token_event(CALL_CLIENT, 12, Some("stop"))]];
        let weather = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]);
        let (status, text, _) = stream_chat(stub, hub, json!({"mcp": {}, "tools": weather})).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let events = sse_json_events(&text);
        let call = events
            .iter()
            .find_map(|e| e.pointer("/choices/0/delta/tool_calls/0"))
            .unwrap();
        assert_eq!(call["function"]["name"], "get_weather");
        assert_eq!(call["index"], 0);
        let finishes: Vec<_> = events
            .iter()
            .filter_map(|e| {
                e.pointer("/choices/0/finish_reason")
                    .and_then(|r| r.as_str())
            })
            .collect();
        assert_eq!(finishes, vec!["tool_calls"]);
        assert!(fake.received_calls().is_empty());
        assert!(events.iter().all(|e| e.get("higgs_mcp").is_none()));
    }

    #[tokio::test]
    async fn streaming_stops_at_max_iterations_and_forwards_the_pending_call() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let mut stub = stub(vec![]);
        stub.script.stream = vec![token_event(CALL_ECHO, 12, Some("stop"))];
        let (status, text, engine) =
            stream_chat(stub, hub, json!({"mcp": {"max_iterations": 1}})).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let events = sse_json_events(&text);
        assert_eq!(engine.recorded_calls().len(), 2);
        assert_eq!(fake.received_calls().len(), 1);
        let pending = events
            .iter()
            .find_map(|e| e.pointer("/choices/0/delta/tool_calls/0"))
            .unwrap();
        assert_eq!(pending["function"]["name"], "files__echo");
        let finishes: Vec<_> = events
            .iter()
            .filter_map(|e| {
                e.pointer("/choices/0/finish_reason")
                    .and_then(|r| r.as_str())
            })
            .collect();
        assert_eq!(finishes, vec!["tool_calls"]);
        let summary = events
            .iter()
            .find(|e| e["higgs_mcp"]["event"] == "summary")
            .unwrap();
        assert_eq!(summary["higgs_mcp"]["truncated"], true);
        assert_eq!(summary["higgs_mcp"]["iterations"], 1);
    }

    #[tokio::test]
    async fn streaming_without_mcp_is_the_plain_path() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let mut stub = stub(vec![]);
        stub.script.stream = vec![token_event("hello", 1, Some("stop"))];
        let (_, text, engine) = stream_chat(stub, hub, json!({})).await;
        let events = sse_json_events(&text);
        assert_eq!(content_of(&events), "hello");
        assert_eq!(engine.recorded_prompt_tools(), vec![None]);
        assert!(events.iter().all(|e| e.get("higgs_mcp").is_none()));
    }

    #[tokio::test]
    async fn responses_route_shares_the_loop() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let (app, engine) = app_with_stub_config_and_tools(
            stub(vec![(CALL_ECHO, 5, 10), ("The answer is pong", 7, 3)]),
            "",
            hub,
        );
        let (status, text) = post_json(
            &app,
            "/v1/responses",
            &json!({"model": "stub", "input": "use the tool", "mcp": {"enabled": true}}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(body["output_text"], "The answer is pong");
        assert_eq!(body["usage"]["input_tokens"], 12);
        assert_eq!(fake.received_calls().len(), 1);
        assert_eq!(engine.recorded_calls().len(), 2);
    }

    #[tokio::test]
    async fn responses_streaming_completes_after_the_loop() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let (app, _) = app_with_stub_config_and_tools(streaming_stub(), "", hub);
        let (status, text) = post_json(
            &app,
            "/v1/responses",
            &json!({"model": "stub", "input": "use the tool", "stream": true, "mcp": {}}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{text}");
        assert!(text.contains("response.completed"), "{text}");
        assert!(text.contains("Final answer"), "{text}");
        assert!(!text.contains("function_call"), "{text}");
        assert_eq!(fake.received_calls().len(), 1);
    }

    #[test]
    fn mcp_options_deserialize_with_defaults() {
        let options: McpOptions = serde_json::from_value(json!({})).unwrap();
        assert!(options.enabled);
        assert!(options.servers.is_none() && options.max_iterations.is_none());
        let explicit: McpOptions = serde_json::from_value(
            json!({"enabled": false, "servers": ["a"], "max_iterations": 3}),
        )
        .unwrap();
        assert!(!explicit.enabled);
        assert_eq!(explicit.servers, Some(vec!["a".to_owned()]));
        assert_eq!(explicit.max_iterations, Some(3));
    }

    #[test]
    fn usage_is_merged_by_summing_token_counts() {
        let mut total = None;
        merge_usage(&mut total, CompletionUsage::new(3, 4));
        let mut latest = CompletionUsage::new(5, 6);
        latest.timings.total_time = Some(2.0);
        merge_usage(&mut total, latest);
        let merged = total.unwrap();
        assert_eq!(merged.prompt_tokens, 8);
        assert_eq!(merged.completion_tokens, 10);
        assert_eq!(merged.total_tokens, 18);
        assert_eq!(merged.timings.total_time, Some(2.0));
    }

    #[test]
    fn rendering_prefers_text_then_structured_content() {
        let output = ToolOutput {
            content: Vec::new(),
            text: String::new(),
            structured_content: Some(json!({"n": 1})),
            is_error: false,
            truncated: true,
        };
        assert_eq!(
            render_output(&output),
            ("{\"n\":1}\n[output truncated]".to_owned(), false)
        );
        let empty = ToolOutput {
            content: Vec::new(),
            text: String::new(),
            structured_content: None,
            is_error: true,
            truncated: false,
        };
        assert_eq!(
            render_output(&empty),
            ("(the tool returned no output)".to_owned(), true)
        );
    }
}
