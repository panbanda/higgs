use higgs_engine::chat_template::ChatMessage;
use higgs_models::SamplingParams;
use serde_json::Value;

use crate::constraint::ConstraintSpec;
use crate::tool_policy::ToolPolicy;
use crate::types::openai::ReasoningConfig;

use super::tool_loop::McpOptions;

/// What every API asks of the engine, whatever wire format it arrived in.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    /// The served name of the model that answers.
    pub model: String,
    pub prompt: Prompt,
    pub max_tokens: u32,
    pub sampling: SamplingParams,
    pub stop_sequences: Vec<String>,
    pub logprobs: bool,
    pub top_logprobs: Option<u32>,
    /// The output constraint the client asked for. A forced tool call adds its
    /// own when the prompt is prepared.
    pub constraint: Option<ConstraintSpec>,
    /// Run server-side tools and continue the generation with their results.
    pub mcp: Option<McpOptions>,
}

#[derive(Debug, Clone)]
pub enum Prompt {
    /// Tokenized as written, with no chat template and no reasoning or tool parsing.
    Text(String),
    Chat(ChatPrompt),
}

/// A conversation the model's chat template renders.
#[derive(Debug, Clone)]
pub struct ChatPrompt {
    pub messages: Vec<ChatMessage>,
    /// Decoded images, in message order; `<image>` placeholders are already in `messages`.
    pub images: Vec<Vec<u8>>,
    /// Tool definitions in `OpenAI` form; empty when the model is not offered tools.
    pub tools: Vec<Value>,
    pub tool_policy: ToolPolicy,
    /// The client's reasoning controls, when the API has them.
    pub reasoning: Option<ReasoningConfig>,
    /// A validated reasoning-token budget.
    pub thinking_budget: Option<u32>,
    /// Whether a finished (non-streamed) answer is searched for `<think>`
    /// blocks even when the prompt did not open one.
    pub parse_reasoning_without_thinking: bool,
}

impl GenerationRequest {
    pub const fn chat(&self) -> Option<&ChatPrompt> {
        match &self.prompt {
            Prompt::Chat(chat) => Some(chat),
            Prompt::Text(_) => None,
        }
    }

    pub const fn chat_mut(&mut self) -> Option<&mut ChatPrompt> {
        match &mut self.prompt {
            Prompt::Chat(chat) => Some(chat),
            Prompt::Text(_) => None,
        }
    }
}
