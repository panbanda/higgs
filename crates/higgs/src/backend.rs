//! The boundary between request handling and a loaded model.
//!
//! Routes only see [`Engine`]; the MLX-backed [`InferenceEngine`] is the one
//! production implementation, and `test_support::StubEngine` the one fake.

use higgs_engine::chat_template::ChatMessage;
use higgs_engine::constrained::ConstrainedGenerator;
use higgs_engine::engine::{GenerationOutput, StreamingOutput, ThinkingMode};
use higgs_engine::error::EngineError;
use higgs_engine::inference::InferenceEngine;
use higgs_engine::tokenizers::Tokenizer;
use higgs_engine::tool_format::ToolCallFormat;
use higgs_models::SamplingParams;
use higgs_models::siglip::PixelValues;

use crate::model_pool::ModelLease;

/// A model the pool owns: the unit it loads, leases and unloads.
pub type LoadedEngine = Box<dyn Engine>;

/// A loaded engine held for one request; the pool will not unload it until this is dropped.
pub type EngineLease = ModelLease<LoadedEngine>;

/// Everything the engine needs to run one generation.
pub struct EngineJob {
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: u32,
    pub sampling: SamplingParams,
    pub stop_sequences: Vec<String>,
    pub logprobs: bool,
    pub top_logprobs: Option<u32>,
    pub thinking: ThinkingMode,
    pub constraint: Option<ConstrainedGenerator>,
    pub pixel_values: Option<PixelValues>,
}

/// A loaded model that can render prompts and generate from them.
pub trait Engine: Send + Sync {
    fn model_name(&self) -> &str;

    fn tokenizer(&self) -> &Tokenizer;

    fn eos_token_ids(&self) -> &[u32];

    /// Whether the model thinks unless a request turns it off.
    fn enable_thinking(&self) -> bool;

    /// How this model writes tool calls.
    fn tool_call_format(&self) -> ToolCallFormat;

    /// `batched` when concurrent requests decode together, else `sequential`.
    fn decode_kind(&self) -> &'static str;

    fn is_vlm(&self) -> bool;

    fn vlm_image_size(&self) -> Option<i32>;

    fn replace_image_tokens(&self, tokens: &mut [u32]);

    /// Apply the chat template and tokenize the result.
    fn prepare_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
    ) -> Result<Vec<u32>, EngineError>;

    fn generate(&self, job: EngineJob) -> Result<GenerationOutput, EngineError>;

    /// Send each generated piece (and prefill progress) to `sender` as it is
    /// produced; returns once generation has finished.
    fn generate_streaming(
        &self,
        job: EngineJob,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
    ) -> Result<(), EngineError>;

    fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>, EngineError>;

    /// Free the engine, returning only once its MLX state has been dropped.
    fn shutdown(self: Box<Self>);
}

impl Engine for InferenceEngine {
    fn model_name(&self) -> &str {
        Self::model_name(self)
    }

    fn tokenizer(&self) -> &Tokenizer {
        Self::tokenizer(self)
    }

    fn eos_token_ids(&self) -> &[u32] {
        Self::eos_token_ids(self)
    }

    fn enable_thinking(&self) -> bool {
        Self::enable_thinking(self)
    }

    fn tool_call_format(&self) -> ToolCallFormat {
        Self::tool_call_format(self)
    }

    fn decode_kind(&self) -> &'static str {
        if self.decodes_in_batches() {
            "batched"
        } else {
            "sequential"
        }
    }

    fn is_vlm(&self) -> bool {
        Self::is_vlm(self)
    }

    fn vlm_image_size(&self) -> Option<i32> {
        Self::vlm_image_size(self)
    }

    fn replace_image_tokens(&self, tokens: &mut [u32]) {
        Self::replace_image_tokens(self, tokens);
    }

    fn prepare_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
    ) -> Result<Vec<u32>, EngineError> {
        self.prepare_chat_prompt_with_thinking(messages, tools, enable_thinking)
    }

    fn generate(&self, job: EngineJob) -> Result<GenerationOutput, EngineError> {
        self.generate_with_thinking(
            &job.prompt_tokens,
            job.max_tokens,
            &job.sampling,
            &job.stop_sequences,
            job.logprobs,
            job.top_logprobs,
            job.thinking,
            job.constraint,
            job.pixel_values,
        )
    }

    fn generate_streaming(
        &self,
        job: EngineJob,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
    ) -> Result<(), EngineError> {
        self.generate_streaming_with_thinking(
            &job.prompt_tokens,
            job.max_tokens,
            &job.sampling,
            &job.stop_sequences,
            job.logprobs,
            job.top_logprobs,
            sender,
            job.thinking,
            // Prefill progress feeds usage and metrics even for clients that never see it.
            true,
            job.constraint,
            job.pixel_values,
        )
    }

    fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        Self::embed(self, token_ids)
    }

    fn shutdown(self: Box<Self>) {
        Self::shutdown(*self);
    }
}
