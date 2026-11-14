//! One request model and one event stream behind every generation API.
//!
//! The chat completions, completions, Anthropic Messages and Responses routes
//! each translate their wire format into a [`GenerationRequest`] and render
//! what comes back in their own format: a [`Completed`] answer, or a stream
//! of [`GenerationEvent`]s. Prompt building, constraints, reasoning and
//! tool-call extraction, usage accounting and the server-side tool loop are
//! shared and live here.

mod complete;
mod event;
pub(crate) mod messages;
mod prepare;
mod request;
mod stream;
pub mod tool_loop;

use std::time::Instant;

use crate::backend::EngineLease;
use crate::error::ServerError;
use crate::metrics::{MetricsStore, RequestRecord, RequestTiming};
use crate::router::RoutingMethod;
use crate::state::SharedState;
use crate::types::openai::CompletionUsage;

pub use event::{Completed, EventStream, GeneratedToolCall, GenerationEvent, ToolLoopEvent};
pub use request::{ChatPrompt, GenerationRequest, Prompt};
pub use stream::StreamMetrics;

/// Record a finished local generation on the metrics page.
pub fn record_completion(
    metrics: &MetricsStore,
    model: &str,
    routing_method: RoutingMethod,
    started: Instant,
    usage: &CompletionUsage,
) {
    metrics.record(RequestRecord {
        id: 0,
        timestamp: Instant::now(),
        wallclock: chrono::Utc::now(),
        model: Some(model.to_owned()),
        provider: Some("higgs".to_owned()),
        routing_method: routing_method.into(),
        status: 200,
        duration: started.elapsed(),
        input_tokens: u64::from(usage.prompt_tokens),
        output_tokens: u64::from(usage.completion_tokens),
        error_body: None,
        timing: RequestTiming::default(),
    });
}

/// Generate a complete answer, running server-side tools if the request asks.
pub async fn complete(
    state: SharedState,
    request: GenerationRequest,
    engine: EngineLease,
    received_at: Instant,
) -> Result<Completed, ServerError> {
    match tool_loop::plan(&state, &request)? {
        None => complete::complete_once(engine, &request, received_at).await,
        Some(plan) => tool_loop::complete(state, request, engine, plan, received_at).await,
    }
}

/// Start generating and return the events as they happen, running
/// server-side tools if the request asks.
pub fn stream(
    state: SharedState,
    request: GenerationRequest,
    engine: EngineLease,
    metrics: Option<StreamMetrics>,
    received_at: Instant,
) -> Result<EventStream, ServerError> {
    match tool_loop::plan(&state, &request)? {
        None => stream::stream_once(engine, &request, received_at, metrics),
        Some(plan) => Ok(tool_loop::stream(
            state,
            request,
            engine,
            plan,
            metrics,
            received_at,
        )),
    }
}
