use std::path::Path;
use std::sync::Arc;

use higgs_engine::batch_engine::BatchEngine;
use higgs_engine::chat_template::ChatMessage;
use higgs_engine::engine::{GenerationOutput, StreamingOutput};
use higgs_engine::error::EngineError;
use higgs_engine::simple::SimpleEngine;
use higgs_engine::tokenizers::Tokenizer;
use higgs_models::SamplingParams;
use mlx_rs::Array;

use crate::config::HiggsConfig;
use crate::metrics::MetricsStore;
use crate::router::Router;

/// Unified engine interface wrapping either the simple (serialized) or batch
/// (interleaved) engine. Route handlers interact with this enum exclusively.
pub enum Engine {
    Simple(Box<SimpleEngine>),
    Batch(Box<BatchEngine>),
}

impl Engine {
    pub fn load_simple<P: AsRef<Path>>(dir: P) -> Result<Self, EngineError> {
        SimpleEngine::load(dir).map(|e| Self::Simple(Box::new(e)))
    }

    pub fn load_batch<P: AsRef<Path>>(dir: P) -> Result<Self, EngineError> {
        BatchEngine::load(dir).map(|e| Self::Batch(Box::new(e)))
    }

    pub fn model_name(&self) -> &str {
        match self {
            Self::Simple(e) => e.model_name(),
            Self::Batch(e) => e.model_name(),
        }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        match self {
            Self::Simple(e) => e.tokenizer(),
            Self::Batch(e) => e.tokenizer(),
        }
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        match self {
            Self::Simple(e) => e.eos_token_ids(),
            Self::Batch(e) => e.eos_token_ids(),
        }
    }

    pub fn hidden_size(&self) -> i32 {
        match self {
            Self::Simple(e) => e.hidden_size(),
            Self::Batch(e) => e.hidden_size(),
        }
    }

    pub fn is_vlm(&self) -> bool {
        match self {
            Self::Simple(e) => e.is_vlm(),
            Self::Batch(_) => false,
        }
    }

    pub fn vlm_image_size(&self) -> Option<i32> {
        match self {
            Self::Simple(e) => e.vlm_image_size(),
            Self::Batch(_) => None,
        }
    }

    pub fn replace_image_tokens(&self, tokens: &mut [u32]) {
        match self {
            Self::Simple(e) => e.replace_image_tokens(tokens),
            Self::Batch(_) => {}
        }
    }

    pub fn prepare_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Vec<u32>, EngineError> {
        match self {
            Self::Simple(e) => e.prepare_chat_prompt(messages, tools),
            Self::Batch(e) => e.prepare_chat_prompt(messages, tools),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        constraint: Option<higgs_engine::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
    ) -> Result<GenerationOutput, EngineError> {
        match self {
            Self::Simple(e) => e.generate(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                constraint,
                pixel_values,
            ),
            Self::Batch(e) => e.generate(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                constraint,
                pixel_values,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        constraint: Option<higgs_engine::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
    ) -> Result<(), EngineError> {
        match self {
            Self::Simple(e) => e.generate_streaming(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                sender,
                constraint,
                pixel_values,
            ),
            Self::Batch(e) => e.generate_streaming(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                sender,
                constraint,
                pixel_values,
            ),
        }
    }

    pub fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        match self {
            Self::Simple(e) => e.embed(token_ids),
            Self::Batch(e) => e.embed(token_ids),
        }
    }
}

/// Shared application state available to all route handlers.
pub struct AppState {
    /// Routes model names to local engines or remote providers.
    pub router: Router,
    /// Full server configuration.
    pub config: HiggsConfig,
    /// HTTP client for proxying requests to remote providers.
    pub http_client: reqwest::Client,
    /// Request metrics (present in config mode, absent in simple mode).
    pub metrics: Option<Arc<MetricsStore>>,
}

/// Type alias for the shared state used by Axum handlers.
pub type SharedState = Arc<AppState>;
