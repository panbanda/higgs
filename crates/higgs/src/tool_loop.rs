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
//! Generation itself is the existing chat path: this module only calls
//! [`chat_completions_non_streaming`] and [`chat_completions_chunk_stream`]
//! once per round and reads their output.

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use serde::Deserialize;
use serde_json::{Value, json};
use tokio_stream::{Stream, StreamExt};

use crate::{
    constraint::ConstraintSpec,
    error::ServerError,
    mcp::manager::ToolOutput,
    metrics::MetricsStore,
    router::{ResolvedRoute, RoutingMethod},
    routes::chat::{chat_completions_chunk_stream, chat_completions_non_streaming},
    state::{AppState, EngineLease, SharedState},
    tool_choice::ToolChoice,
    tools::openai_tool,
    types::openai::{
        ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse, MessageContent,
        ToolCall, ToolCallFunction,
    },
};

pub const DEFAULT_MAX_ITERATIONS: u32 = 5;
pub const MAX_ITERATIONS_LIMIT: u32 = 25;
/// How much of a tool result the streaming events echo to the client.
const RESULT_PREVIEW_CHARS: usize = 1000;
/// Key under which tool-loop information is reported in responses and
/// stream events.
const VENDOR_KEY: &str = "higgs_mcp";

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

struct LoopPlan {
    /// Tools to add to the request, in `OpenAI` form.
    injected_tools: Vec<Value>,
    /// Names the server executes itself.
    names: HashSet<String>,
    max_iterations: u32,
}

const fn bad_request(message: String) -> ServerError {
    ServerError::BadRequest(message)
}

fn client_tool_names(tools: Option<&[Value]>) -> HashSet<String> {
    tools
        .unwrap_or_default()
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
fn plan(state: &AppState, req: &ChatCompletionRequest) -> Result<Option<LoopPlan>, ServerError> {
    let Some(options) = req.mcp.as_ref().filter(|options| options.enabled) else {
        return Ok(None);
    };
    if ToolChoice::from_request(req.tool_choice.as_ref())? == ToolChoice::Disabled {
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
    let taken = client_tool_names(req.tools.as_deref());
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

    let mut injected_tools = req.tools.clone().unwrap_or_default();
    injected_tools.extend(selected.iter().map(|tool| openai_tool(tool)));
    Ok(Some(LoopPlan {
        injected_tools,
        names: selected.iter().map(|tool| tool.name.clone()).collect(),
        max_iterations,
    }))
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

async fn execute_call(state: &AppState, call: &ToolCall) -> ExecutedCall {
    let raw = call.function.arguments.trim();
    let parsed: Result<Value, String> = if raw.is_empty() {
        Ok(json!({}))
    } else {
        serde_json::from_str(raw).map_err(|error| format!("arguments are not valid JSON: {error}"))
    };
    let (arguments, outcome) = match parsed {
        Ok(arguments) => {
            let result = state
                .tools
                .call(&call.function.name, arguments.clone())
                .await;
            (
                arguments,
                result
                    .map(|output| render_output(&output))
                    .map_err(|error| error.to_string()),
            )
        }
        Err(message) => (Value::String(raw.to_owned()), Err(message)),
    };
    let (content, is_error) = match outcome {
        Ok((text, is_error)) => (text, is_error),
        Err(message) => (format!("Error: {message}"), true),
    };
    ExecutedCall {
        id: call.id.clone(),
        name: call.function.name.clone(),
        arguments,
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

fn assistant_message(
    content: Option<MessageContent>,
    calls: Vec<ToolCall>,
) -> ChatCompletionMessage {
    ChatCompletionMessage {
        role: "assistant".to_owned(),
        content,
        reasoning_content: None,
        tool_calls: Some(calls),
        tool_call_id: None,
    }
}

fn tool_message(call: &ExecutedCall) -> ChatCompletionMessage {
    ChatCompletionMessage {
        role: "tool".to_owned(),
        content: Some(MessageContent::Text(call.content.clone())),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: Some(call.id.clone()),
    }
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
}

fn split_calls(calls: Vec<ToolCall>, names: &HashSet<String>) -> (Vec<ToolCall>, Vec<ToolCall>) {
    calls
        .into_iter()
        .partition(|call| names.contains(&call.function.name))
}

/// Non-streaming generation with the tool loop. Requests without `mcp` take
/// the plain path untouched.
pub(crate) async fn complete(
    state: SharedState,
    req: ChatCompletionRequest,
    engine: EngineLease,
    constraint_spec: Option<ConstraintSpec>,
    received_at: Instant,
) -> Result<ChatCompletionResponse, ServerError> {
    let Some(plan) = plan(&state, &req)? else {
        return chat_completions_non_streaming(state, req, engine, constraint_spec, received_at)
            .await;
    };

    let mut conversation = req;
    conversation.mcp = None;
    conversation.tools = Some(plan.injected_tools.clone());
    let mut first_lease = Some(engine);
    let mut trace = Trace::default();
    let mut totals: (u32, u32) = (0, 0);

    for round in 0..=plan.max_iterations {
        let lease = match first_lease.take() {
            Some(lease) => lease,
            None => reacquire(&state, &conversation.model).await?,
        };
        let mut response = chat_completions_non_streaming(
            Arc::clone(&state),
            conversation.clone(),
            lease,
            constraint_spec.clone(),
            received_at,
        )
        .await?;

        let (previous_prompt, previous_completion) = totals;
        totals = (
            previous_prompt.saturating_add(response.usage.prompt_tokens),
            previous_completion.saturating_add(response.usage.completion_tokens),
        );
        response.usage.prompt_tokens = totals.0;
        response.usage.completion_tokens = totals.1;
        response.usage.total_tokens = totals.0.saturating_add(totals.1);

        let assistant = response
            .choices
            .first()
            .map(|choice| choice.message.clone());
        let calls = assistant
            .as_ref()
            .and_then(|reply| reply.tool_calls.clone())
            .unwrap_or_default();
        let (server_calls, client_calls) = split_calls(calls, &plan.names);
        if server_calls.is_empty() {
            return Ok(finish(response, &trace));
        }
        let exhausted = round == plan.max_iterations;
        if !client_calls.is_empty() {
            if let Some(choice) = response.choices.first_mut() {
                choice.message.tool_calls = Some(client_calls);
            }
            return Ok(finish(response, &trace));
        }
        if exhausted {
            trace.truncated = true;
            return Ok(finish(response, &trace));
        }

        trace.rounds += 1;
        conversation.messages.push(assistant_message(
            assistant.and_then(|reply| reply.content),
            server_calls.clone(),
        ));
        for call in &server_calls {
            let executed = execute_call(&state, call).await;
            trace.record(trace.rounds, &executed);
            conversation.messages.push(tool_message(&executed));
        }
    }
    Err(ServerError::InternalError(
        "tool loop ended without a response".to_owned(),
    ))
}

fn finish(mut response: ChatCompletionResponse, trace: &Trace) -> ChatCompletionResponse {
    if trace.rounds > 0 || trace.truncated {
        response.mcp = Some(trace.to_value());
    }
    response
}

type ChunkStream = Pin<Box<dyn Stream<Item = String> + Send>>;

/// Id, creation time and model of the first chunk, reused on every chunk so
/// the stream looks like one response across rounds.
#[derive(Clone)]
struct Identity {
    id: Value,
    created: Value,
    model: Value,
}

/// What a stream round has seen so far.
#[derive(Default)]
struct RoundState {
    server_calls: Vec<ToolCall>,
    text: String,
    client_call_forwarded: bool,
    next_call_index: u64,
    held_finish: Option<Value>,
    held_usage: Option<Value>,
}

impl RoundState {
    /// Route one chunk of the round's stream: `Some` is forwarded to the
    /// client, `None` is suppressed (or held for the end of the round).
    fn ingest(
        &mut self,
        mut chunk: Value,
        names: &HashSet<String>,
        final_round: bool,
        role_sent: &mut bool,
    ) -> Option<Value> {
        let has_choice = chunk
            .get("choices")
            .and_then(Value::as_array)
            .is_some_and(|choices| !choices.is_empty());
        if !has_choice {
            if chunk.get("usage").is_some_and(|usage| !usage.is_null()) {
                self.held_usage = Some(chunk);
                return None;
            }
            return Some(chunk);
        }

        let finished = chunk
            .pointer("/choices/0/finish_reason")
            .is_some_and(|reason| !reason.is_null());
        if finished {
            self.held_finish = Some(chunk);
            return None;
        }

        let delta = chunk.pointer_mut("/choices/0/delta")?;
        if delta.get("role").is_some() {
            if *role_sent {
                return None;
            }
            *role_sent = true;
        }
        if let Some(text) = delta.get("content").and_then(Value::as_str) {
            self.text.push_str(text);
        }
        let Some(calls) = delta.get("tool_calls").and_then(Value::as_array).cloned() else {
            return Some(chunk);
        };

        let mut forwarded = Vec::new();
        for mut call in calls {
            let name = call
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            if !final_round && names.contains(&name) {
                self.server_calls.push(ToolCall {
                    id: call
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    r#type: "function".to_owned(),
                    function: ToolCallFunction {
                        name,
                        arguments: call
                            .pointer("/function/arguments")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    },
                });
                continue;
            }
            if !names.contains(&name) {
                self.client_call_forwarded = true;
            }
            if let Some(object) = call.as_object_mut() {
                object.insert("index".to_owned(), json!(self.next_call_index));
            }
            self.next_call_index += 1;
            forwarded.push(call);
        }
        let delta_object = delta.as_object_mut()?;
        if forwarded.is_empty() {
            delta_object.remove("tool_calls");
            let carries_anything = delta_object
                .iter()
                .any(|(key, value)| key != "role" && !value.is_null());
            return carries_anything.then_some(chunk);
        }
        delta_object.insert("tool_calls".to_owned(), Value::Array(forwarded));
        Some(chunk)
    }
}

/// Add the token counts of `next` into `total`; other usage fields (timings,
/// details) come from the latest round.
fn merge_usage(total: &mut Option<Value>, next: Value) {
    let Some(previous) = total.take() else {
        *total = Some(next);
        return;
    };
    let mut merged = next;
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"] {
        let sum = previous
            .pointer(&format!("/usage/{key}"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .saturating_add(
                merged
                    .pointer(&format!("/usage/{key}"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            );
        if let Some(slot) = merged.pointer_mut(&format!("/usage/{key}")) {
            *slot = json!(sum);
        }
    }
    *total = Some(merged);
}

/// Streaming generation with the tool loop, as raw SSE `data` payloads
/// ending in `[DONE]`.
///
/// Assistant text and reasoning stream as they are produced in every round.
/// Executed tool calls never appear as `tool_calls` deltas; instead
/// chunks with an empty `choices` array and a `higgs_mcp` field announce
/// each call (`tool_call`), its outcome (`tool_result`) and the loop's end
/// (`summary`). Standard clients ignore them, as they do usage chunks.
#[allow(clippy::too_many_lines)]
pub(crate) fn chunk_stream(
    state: SharedState,
    req: ChatCompletionRequest,
    engine: EngineLease,
    constraint_spec: Option<ConstraintSpec>,
    metrics: Option<Arc<MetricsStore>>,
    routing_method: RoutingMethod,
    received_at: Instant,
) -> Result<ChunkStream, ServerError> {
    let Some(plan) = plan(&state, &req)? else {
        return Ok(Box::pin(chat_completions_chunk_stream(
            state,
            req,
            engine,
            constraint_spec,
            metrics,
            routing_method,
            received_at,
        )?));
    };

    let mut conversation = req;
    conversation.mcp = None;
    conversation.tools = Some(plan.injected_tools.clone());

    Ok(Box::pin(async_stream::stream! {
        let mut first_lease = Some(engine);
        let mut trace = Trace::default();
        let mut role_sent = false;
        let mut identity: Option<Identity> = None;
        let mut usage_total: Option<Value> = None;
        let mut round: u32 = 0;

        'rounds: loop {
            let final_round = round >= plan.max_iterations;
            let lease = match first_lease.take() {
                Some(lease) => lease,
                None => match reacquire(&state, &conversation.model).await {
                    Ok(lease) => lease,
                    Err(error) => {
                        yield error_chunk(&error);
                        break 'rounds;
                    }
                },
            };
            let round_metrics = if round == 0 { metrics.clone() } else { None };
            let source = chat_completions_chunk_stream(
                Arc::clone(&state),
                conversation.clone(),
                lease,
                constraint_spec.clone(),
                round_metrics,
                routing_method,
                received_at,
            );
            let mut chunks = match source {
                Ok(stream) => Box::pin(stream),
                Err(error) => {
                    yield error_chunk(&error);
                    break 'rounds;
                }
            };

            let mut state_of_round = RoundState::default();
            while let Some(data) = chunks.next().await {
                if data == "[DONE]" {
                    break;
                }
                let Ok(mut chunk) = serde_json::from_str::<Value>(&data) else {
                    yield data;
                    continue;
                };
                match &identity {
                    None => {
                        identity = Some(Identity {
                            id: chunk.get("id").cloned().unwrap_or(Value::Null),
                            created: chunk.get("created").cloned().unwrap_or(Value::Null),
                            model: chunk.get("model").cloned().unwrap_or(Value::Null),
                        });
                    }
                    Some(first) => {
                        if let Some(object) = chunk.as_object_mut() {
                            object.insert("id".to_owned(), first.id.clone());
                            object.insert("created".to_owned(), first.created.clone());
                            object.insert("model".to_owned(), first.model.clone());
                        }
                    }
                }
                if let Some(forward) =
                    state_of_round.ingest(chunk, &plan.names, final_round, &mut role_sent)
                {
                    yield forward.to_string();
                }
            }
            drop(chunks);

            if let Some(usage) = state_of_round.held_usage.take() {
                merge_usage(&mut usage_total, usage);
            }
            let run_tools = !state_of_round.server_calls.is_empty()
                && !state_of_round.client_call_forwarded
                && !final_round;
            if !run_tools {
                trace.truncated =
                    final_round && state_of_round.held_finish.as_ref().is_some_and(|chunk| {
                        chunk.pointer("/choices/0/finish_reason").and_then(Value::as_str)
                            == Some("tool_calls")
                    }) && !state_of_round.client_call_forwarded;
                if let Some(finish) = state_of_round.held_finish.take() {
                    yield finish.to_string();
                }
                break 'rounds;
            }

            round += 1;
            trace.rounds = round;
            let calls = std::mem::take(&mut state_of_round.server_calls);
            let text = std::mem::take(&mut state_of_round.text);
            conversation.messages.push(assistant_message(
                (!text.is_empty()).then_some(MessageContent::Text(text)),
                calls.clone(),
            ));
            for call in &calls {
                let arguments = serde_json::from_str::<Value>(&call.function.arguments)
                    .unwrap_or_else(|_| Value::String(call.function.arguments.clone()));
                yield vendor_chunk(identity.as_ref(), &json!({
                    "event": "tool_call",
                    "round": round,
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": arguments,
                }));
                let executed = execute_call(&state, call).await;
                trace.record(round, &executed);
                yield vendor_chunk(identity.as_ref(), &json!({
                    "event": "tool_result",
                    "round": round,
                    "id": executed.id,
                    "name": executed.name,
                    "is_error": executed.is_error,
                    "content": executed.content.chars().take(RESULT_PREVIEW_CHARS).collect::<String>(),
                }));
                conversation.messages.push(tool_message(&executed));
            }
        }

        if let Some(usage) = usage_total {
            yield usage.to_string();
        }
        if trace.rounds > 0 || trace.truncated {
            let mut summary = trace.to_value();
            if let Some(object) = summary.as_object_mut() {
                object.insert("event".to_owned(), json!("summary"));
            }
            yield vendor_chunk(identity.as_ref(), &summary);
        }
        yield "[DONE]".to_owned();
    }))
}

fn vendor_chunk(identity: Option<&Identity>, payload: &Value) -> String {
    let fallback = Identity {
        id: Value::Null,
        created: Value::Null,
        model: Value::Null,
    };
    let source = identity.unwrap_or(&fallback);
    json!({
        "id": source.id,
        "object": "chat.completion.chunk",
        "created": source.created,
        "model": source.model,
        "choices": [],
        VENDOR_KEY: payload,
    })
    .to_string()
}

fn error_chunk(error: &ServerError) -> String {
    tracing::error!(%error, "tool loop failed mid-stream");
    json!({"error": {"message": "tool loop failed", "type": "server_error"}}).to_string()
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use std::time::Duration;

    use http::StatusCode;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::mcp::manager::McpManager;
    use crate::mcp::testing::{CallBehavior, FakeMcp, ready_manager, text_result};
    use crate::state::Engine;
    use crate::test_support::{
        StubEngine, app_with_stub_config_and_tools, generation_output, post_json, sse_json_events,
        token_event, word_tokenizer,
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
        stub.tokenizer = Some(word_tokenizer());
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
    ) -> (StatusCode, serde_json::Value, Arc<Engine>) {
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
        assert_eq!(engine.stub_calls().len(), 2);
        assert_eq!(engine.stub_prompt_tools(), vec![Some(2), Some(2)]);
        let prompts = engine.stub_prompt_messages();
        assert_eq!(prompts[0].len(), 1);
        let roles: Vec<_> = prompts[1].iter().map(|(role, _)| role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant", "tool"]);
        assert_eq!(prompts[1][2].1, "pong");
    }

    #[tokio::test]
    async fn requests_without_mcp_are_untouched() {
        let fake = echo_fake();
        let hub = files_hub(&fake, Duration::from_secs(2)).await;
        let (status, body, engine) =
            chat(stub(vec![(CALL_ECHO, 5, 10)]), hub, &request(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.get("mcp").is_none());
        assert_eq!(engine.stub_prompt_tools(), vec![None]);
        assert_eq!(engine.stub_calls().len(), 1);
        assert!(fake.received_calls().is_empty());

        let disabled = files_hub(&fake, Duration::from_secs(2)).await;
        let (_, off_body, off_engine) = chat(
            stub(vec![(CALL_ECHO, 5, 10)]),
            disabled,
            &request(json!({"mcp": {"enabled": false}})),
        )
        .await;
        assert!(off_body.get("mcp").is_none());
        assert_eq!(off_engine.stub_prompt_tools(), vec![None]);
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
        assert_eq!(engine.stub_prompt_tools(), vec![None]);
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
        assert_eq!(engine.stub_calls().len(), 3);
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
        assert_eq!(engine.stub_calls().len(), 6);
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
        assert_eq!(engine.stub_prompt_tools(), vec![Some(3)]);
        assert_eq!(engine.stub_calls().len(), 1);
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
        assert_eq!(engine.stub_prompt_tools(), vec![Some(2)]);
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
        assert_eq!(engine.stub_prompt_tools(), vec![Some(1)]);

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
            assert!(engine.stub_calls().is_empty());
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
        let tool_messages: Vec<_> = engine.stub_prompt_messages()[1]
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
            engine.stub_prompt_messages()[1][2].1,
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
        assert_eq!(engine.stub_prompt_tools(), vec![Some(2), Some(2)]);
        assert!(
            engine.stub_prompt_messages()[1][2]
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
    ) -> (StatusCode, String, Arc<Engine>) {
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
        let prompts = engine.stub_prompt_messages();
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
        assert_eq!(engine.stub_calls().len(), 2);
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
        assert_eq!(engine.stub_prompt_tools(), vec![None]);
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
        assert_eq!(engine.stub_calls().len(), 2);
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
    fn usage_chunks_are_merged_by_summing_token_counts() {
        let mut total = None;
        merge_usage(
            &mut total,
            json!({"usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7, "time": 1}}),
        );
        merge_usage(
            &mut total,
            json!({"usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11, "time": 2}}),
        );
        let merged = total.unwrap();
        assert_eq!(merged["usage"]["prompt_tokens"], 8);
        assert_eq!(merged["usage"]["completion_tokens"], 10);
        assert_eq!(merged["usage"]["total_tokens"], 18);
        assert_eq!(merged["usage"]["time"], 2);
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
