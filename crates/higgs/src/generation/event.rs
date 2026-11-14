use std::pin::Pin;

use higgs_engine::engine::PrefillProgress;
use serde_json::Value;
use tokio_stream::Stream;

use crate::types::openai::{ChoiceLogprobs, CompletionUsage};

/// One thing a generation produced, in the order it happened.
///
/// A stream ends after `Finish` and `Usage`; it ends without `Finish` when
/// the engine failed partway. Every API renders these in its own format.
#[derive(Debug, Clone)]
pub enum GenerationEvent {
    /// The prompt is prepared and generation is under way; always first.
    Started {
        prompt_tokens: u32,
    },
    /// The engine evaluated part of the prompt.
    PrefillProgress(PrefillProgress),
    ReasoningDelta(String),
    TextDelta {
        text: String,
        logprobs: Option<ChoiceLogprobs>,
    },
    /// A tool call, complete: the engine's output carries calls whole.
    ToolCall(GeneratedToolCall),
    Finish {
        /// `OpenAI` vocabulary; `tool_calls` whenever the answer called a tool.
        reason: String,
        /// Carried here when the finishing token produced no visible text.
        logprobs: Option<ChoiceLogprobs>,
    },
    Usage(CompletionUsage),
    ToolLoop(ToolLoopEvent),
    /// The generation could not continue.
    Error(String),
}

/// Progress of the server-side tool loop, between generation rounds.
#[derive(Debug, Clone)]
pub enum ToolLoopEvent {
    Call {
        round: u32,
        id: String,
        name: String,
        arguments: Value,
    },
    Result {
        round: u32,
        id: String,
        name: String,
        is_error: bool,
        content: String,
    },
    Summary(Value),
    /// What the loop added to the conversation, sent once before the summary.
    Messages(Vec<LoopMessage>),
}

/// A message the tool loop appended to the conversation.
#[derive(Debug, Clone)]
pub enum LoopMessage {
    /// An assistant turn that called server-side tools.
    AssistantCalls {
        text: String,
        calls: Vec<GeneratedToolCall>,
    },
    ToolResult {
        call_id: String,
        content: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedToolCall {
    /// Position among the calls of the response, from zero.
    pub index: u32,
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

impl GeneratedToolCall {
    pub fn new(index: u32, name: String, arguments: Value) -> Self {
        Self {
            index,
            id: format!("call_{index}_{}", uuid::Uuid::new_v4()),
            name,
            arguments,
        }
    }
}

/// A finished, non-streamed generation.
#[derive(Debug, Clone)]
pub struct Completed {
    /// The visible answer, with reasoning and tool calls taken out.
    pub text: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<GeneratedToolCall>,
    /// `OpenAI` vocabulary; `tool_calls` whenever the answer called a tool.
    pub finish_reason: String,
    pub usage: CompletionUsage,
    pub logprobs: Option<ChoiceLogprobs>,
    /// What the server-side tool loop did, when it ran tools.
    pub tool_loop: Option<Value>,
    /// The messages the tool loop appended, for a caller that stores the
    /// conversation.
    pub loop_messages: Vec<LoopMessage>,
}

pub type EventStream = Pin<Box<dyn Stream<Item = GenerationEvent> + Send>>;
