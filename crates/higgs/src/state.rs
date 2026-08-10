use std::path::Path;
use std::sync::Arc;

use higgs_engine::batch_engine::BatchEngine;
use higgs_engine::chat_template::ChatMessage;
use higgs_engine::engine::{GenerationOutput, StreamingOutput};
use higgs_engine::error::EngineError;
use higgs_engine::mlx_tuning::{MlxRuntimeTuning, resolve_runtime_tuning};
use higgs_engine::simple::SimpleEngine;
use higgs_engine::tokenizers::Tokenizer;
use higgs_models::SamplingParams;
use higgs_models::turboquant::KvCacheConfig;
use mlx_rs::Array;

use crate::config::{HiggsConfig, LocalConfig, ModelConfig, resolved_model_supports_batch};
use crate::metrics::MetricsStore;
use crate::router::Router;

/// Process-wide GPU inference gate.
///
/// MLX's Metal backend keeps shared, non-stream-local state — notably the
/// output-array table mutated in `metal::CommandEncoder::set_output_array`. Two
/// co-resident models evaluating concurrently (each on its own `spawn_blocking`
/// thread, each under a fresh `with_new_default_stream(Stream::new())`) race on
/// that table and corrupt it → `EXC_BAD_ACCESS`/SIGSEGV inside
/// `set_output_array`. The per-engine `Mutex<AnyModel>` only serializes a single
/// model, not across the co-resident set (e.g. an SLM trio).
///
/// On a single-GPU host there is no real parallelism to lose, so all GPU eval is
/// serialized through this one gate. Held only for the duration of a
/// generate/embed call. NOTE: this also serializes concurrent requests to a
/// single `Batch` engine; if per-model batch interleaving is reintroduced, this
/// gate should be narrowed to cross-model boundaries.
static GPU_GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire the global GPU gate, recovering from poisoning so a panic mid-eval
/// cannot permanently wedge all inference.
fn gpu_gate() -> std::sync::MutexGuard<'static, ()> {
    GPU_GATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Unified engine interface wrapping either the simple (serialized) or batch
/// (interleaved) engine. Route handlers interact with this enum exclusively.
pub enum Engine {
    Simple(Box<SimpleEngine>),
    Batch(Box<BatchEngine>),
    #[cfg(test)]
    Stub(String),
}

impl Engine {
    pub fn load_simple<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
        tuning: MlxRuntimeTuning,
        raise_wired_limit: bool,
    ) -> Result<Self, EngineError> {
        SimpleEngine::load(dir, kv_cache_config, tuning, raise_wired_limit)
            .map(|e| Self::Simple(Box::new(e)))
    }

    pub fn load_batch<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
        raise_wired_limit: bool,
    ) -> Result<Self, EngineError> {
        BatchEngine::load(dir, kv_cache_config, raise_wired_limit).map(|e| Self::Batch(Box::new(e)))
    }

    #[cfg(test)]
    pub fn test_stub(name: &str) -> Self {
        Self::Stub(name.to_owned())
    }

    pub fn model_name(&self) -> &str {
        match self {
            Self::Simple(e) => e.model_name(),
            Self::Batch(e) => e.model_name(),
            #[cfg(test)]
            Self::Stub(name) => name,
        }
    }

    #[cfg_attr(test, allow(clippy::panic))]
    pub fn tokenizer(&self) -> &Tokenizer {
        match self {
            Self::Simple(e) => e.tokenizer(),
            Self::Batch(e) => e.tokenizer(),
            #[cfg(test)]
            Self::Stub(_) => panic!("Engine::test_stub has no tokenizer"),
        }
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        match self {
            Self::Simple(e) => e.eos_token_ids(),
            Self::Batch(e) => e.eos_token_ids(),
            #[cfg(test)]
            Self::Stub(_) => &[],
        }
    }

    pub fn hidden_size(&self) -> i32 {
        match self {
            Self::Simple(e) => e.hidden_size(),
            Self::Batch(e) => e.hidden_size(),
            #[cfg(test)]
            Self::Stub(_) => 0,
        }
    }

    pub fn enable_thinking(&self) -> bool {
        match self {
            Self::Simple(e) => e.enable_thinking(),
            Self::Batch(_) => false,
            #[cfg(test)]
            Self::Stub(_) => false,
        }
    }

    pub fn is_vlm(&self) -> bool {
        match self {
            Self::Simple(e) => e.is_vlm(),
            Self::Batch(_) => false,
            #[cfg(test)]
            Self::Stub(_) => false,
        }
    }

    pub fn vlm_image_size(&self) -> Option<i32> {
        match self {
            Self::Simple(e) => e.vlm_image_size(),
            Self::Batch(_) => None,
            #[cfg(test)]
            Self::Stub(_) => None,
        }
    }

    pub fn replace_image_tokens(&self, tokens: &mut [u32]) {
        match self {
            Self::Simple(e) => e.replace_image_tokens(tokens),
            Self::Batch(_) => {}
            #[cfg(test)]
            Self::Stub(_) => {}
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
            #[cfg(test)]
            Self::Stub(_) => Ok(Vec::new()),
        }
    }

    pub fn prepare_chat_prompt_with_thinking(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
    ) -> Result<Vec<u32>, EngineError> {
        match self {
            Self::Simple(e) => {
                e.prepare_chat_prompt_with_thinking(messages, tools, enable_thinking)
            }
            Self::Batch(e) => e.prepare_chat_prompt_with_thinking(messages, tools, enable_thinking),
            #[cfg(test)]
            Self::Stub(_) => Ok(Vec::new()),
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
        self.generate_with_thinking(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            logprobs,
            top_logprobs,
            self.enable_thinking(),
            constraint,
            pixel_values,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_thinking(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        enable_thinking: bool,
        constraint: Option<higgs_engine::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
    ) -> Result<GenerationOutput, EngineError> {
        let _gpu = gpu_gate();
        match self {
            Self::Simple(e) => e.generate_with_thinking(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                enable_thinking,
                constraint,
                pixel_values,
            ),
            Self::Batch(e) => e.generate_with_thinking(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                enable_thinking,
                constraint,
                pixel_values,
            ),
            #[cfg(test)]
            Self::Stub(_) => Err(EngineError::Generation("test stub".to_owned())),
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
        self.generate_streaming_with_thinking(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            logprobs,
            top_logprobs,
            sender,
            self.enable_thinking(),
            // /v1/completions convenience entry never streams prefill progress.
            false,
            constraint,
            pixel_values,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_streaming_with_thinking(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        enable_thinking: bool,
        return_progress: bool,
        constraint: Option<higgs_engine::constrained::ConstrainedGenerator>,
        pixel_values: Option<Array>,
    ) -> Result<(), EngineError> {
        let _gpu = gpu_gate();
        match self {
            Self::Simple(e) => e.generate_streaming_with_thinking(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                sender,
                enable_thinking,
                return_progress,
                constraint,
                pixel_values,
            ),
            Self::Batch(e) => e.generate_streaming_with_thinking(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                sender,
                enable_thinking,
                return_progress,
                constraint,
                pixel_values,
            ),
            #[cfg(test)]
            Self::Stub(_) => Err(EngineError::Generation("test stub".to_owned())),
        }
    }

    pub fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        let _gpu = gpu_gate();
        match self {
            Self::Simple(e) => e.embed(token_ids),
            Self::Batch(e) => e.embed(token_ids),
            #[cfg(test)]
            Self::Stub(_) => Ok(Vec::new()),
        }
    }
}

/// Build an engine from an already-resolved model directory and its config.
///
/// Shared by startup loading (`load_engines` in the binary) and the runtime
/// load endpoint (`POST /v1/models`). Path resolution and any download prompt
/// are the caller's responsibility, so this never blocks on stdin. Returns the
/// model's exposed name alongside the constructed engine.
pub fn build_engine(
    resolved: &Path,
    model_cfg: &ModelConfig,
    local: &LocalConfig,
) -> Result<(String, Engine), String> {
    if model_cfg.batch && !resolved_model_supports_batch(resolved)? {
        return Err(format!(
            "batch=true is only supported for transformer models (llama, mistral, qwen2, qwen3); '{}' is not supported",
            model_cfg.path
        ));
    }
    let kv_cache_config = model_cfg.kv_cache_config();
    let engine = if model_cfg.batch {
        Engine::load_batch(resolved, kv_cache_config, local.raise_wired_limit)
            .map_err(|e| e.to_string())?
    } else {
        let tuning = resolve_runtime_tuning(resolved, model_cfg.requested_mlx_profile(local));
        Engine::load_simple(resolved, kv_cache_config, tuning, local.raise_wired_limit)
            .map_err(|e| e.to_string())?
    };
    let name = model_cfg
        .name
        .clone()
        .unwrap_or_else(|| engine.model_name().to_owned());
    Ok((name, engine))
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
