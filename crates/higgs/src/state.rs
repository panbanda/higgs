use std::path::Path;
use std::sync::Arc;

use higgs_engine::batch_engine::{BatchEngine, EngineOptions};
use higgs_engine::chat_template::ChatMessage;
use higgs_engine::engine::{GenerationOutput, StreamingOutput, ThinkingMode};
use higgs_engine::error::EngineError;
use higgs_engine::mlx_tuning::MlxRuntimeTuning;
use higgs_engine::simple::SimpleEngine;
use higgs_engine::tokenizers::Tokenizer;
use higgs_engine::tool_format::ToolCallFormat;
use higgs_models::SamplingParams;
use higgs_models::siglip::PixelValues;
use higgs_models::turboquant::KvCacheConfig;

use crate::config::HiggsConfig;
use crate::metrics::MetricsStore;
use crate::model_pool::ModelLease;
use crate::router::Router;

#[cfg(test)]
const fn request_thinking_budget(thinking: ThinkingMode) -> Option<u32> {
    match thinking {
        ThinkingMode::On { budget } => budget,
        ThinkingMode::Off => None,
    }
}
/// A loaded engine held for one request; the pool will not unload it until this is dropped.
pub type EngineLease = ModelLease<Engine>;

/// Unified engine interface wrapping either the simple (serialized) or batch
/// (interleaved) engine. Route handlers interact with this enum exclusively.
pub enum Engine {
    Simple(Box<SimpleEngine>),
    Batch(Box<BatchEngine>),
    #[cfg(test)]
    Stub(Box<crate::test_support::StubEngine>),
}

impl Engine {
    pub fn load_simple<P: AsRef<Path>>(
        dir: P,
        kv_cache_config: KvCacheConfig,
        tuning: MlxRuntimeTuning,
        raise_wired_limit: bool,
        disk_prefix_dir: Option<&Path>,
        disk_prefix_budget: u64,
        default_thinking_budget: u32,
    ) -> Result<Self, EngineError> {
        SimpleEngine::load(
            dir,
            kv_cache_config,
            tuning,
            raise_wired_limit,
            disk_prefix_dir,
            disk_prefix_budget,
            default_thinking_budget,
        )
        .map(|e| Self::Simple(Box::new(e)))
    }

    pub fn load_batch<P: AsRef<Path>>(dir: P, options: EngineOptions) -> Result<Self, EngineError> {
        BatchEngine::load(dir, options).map(|e| Self::Batch(Box::new(e)))
    }

    /// Free the engine, returning only once its MLX state has been dropped.
    pub fn shutdown(self) {
        match self {
            Self::Simple(engine) => engine.shutdown(),
            Self::Batch(engine) => engine.shutdown(),
            #[cfg(test)]
            Self::Stub(_) => {}
        }
    }

    #[cfg(test)]
    pub fn test_stub(name: &str) -> Self {
        Self::Stub(Box::new(crate::test_support::StubEngine::named(name)))
    }

    #[cfg(test)]
    pub fn scripted_stub(stub: crate::test_support::StubEngine) -> Self {
        Self::Stub(Box::new(stub))
    }

    #[cfg(test)]
    pub fn stub_calls(&self) -> Vec<crate::test_support::StubCall> {
        match self {
            Self::Stub(stub) => stub.recorded_calls(),
            Self::Simple(_) | Self::Batch(_) => Vec::new(),
        }
    }

    #[cfg(test)]
    pub fn stub_prompt_messages(&self) -> Vec<Vec<(String, String)>> {
        match self {
            Self::Stub(stub) => stub.recorded_prompt_messages(),
            Self::Simple(_) | Self::Batch(_) => Vec::new(),
        }
    }

    #[cfg(test)]
    pub fn stub_prompt_tools(&self) -> Vec<Option<usize>> {
        match self {
            Self::Stub(stub) => stub.recorded_prompt_tools(),
            Self::Simple(_) | Self::Batch(_) => Vec::new(),
        }
    }

    /// `simple` (serialized) or `batch` (interleaved).
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Simple(_) => "simple",
            Self::Batch(_) => "batch",
            #[cfg(test)]
            Self::Stub(_) => "stub",
        }
    }

    pub fn model_name(&self) -> &str {
        match self {
            Self::Simple(e) => e.model_name(),
            Self::Batch(e) => e.model_name(),
            #[cfg(test)]
            Self::Stub(stub) => &stub.name,
        }
    }

    #[cfg_attr(test, allow(clippy::unreachable))]
    pub fn tokenizer(&self) -> &Tokenizer {
        match self {
            Self::Simple(e) => e.tokenizer(),
            Self::Batch(e) => e.tokenizer(),
            #[cfg(test)]
            Self::Stub(stub) => stub
                .tokenizer
                .as_ref()
                .unwrap_or_else(|| unreachable!("Engine::test_stub has no tokenizer")),
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
            Self::Batch(e) => e.enable_thinking(),
            #[cfg(test)]
            Self::Stub(stub) => stub.script.enable_thinking,
        }
    }

    /// How this model writes tool calls.
    pub fn tool_call_format(&self) -> ToolCallFormat {
        match self {
            Self::Simple(e) => e.tool_call_format(),
            Self::Batch(e) => e.tool_call_format(),
            #[cfg(test)]
            Self::Stub(stub) => stub.script.tool_call_format,
        }
    }

    pub fn is_vlm(&self) -> bool {
        match self {
            Self::Simple(e) => e.is_vlm(),
            Self::Batch(e) => e.is_vlm(),
            #[cfg(test)]
            Self::Stub(_) => false,
        }
    }

    pub fn vlm_image_size(&self) -> Option<i32> {
        match self {
            Self::Simple(e) => e.vlm_image_size(),
            Self::Batch(e) => e.vlm_image_size(),
            #[cfg(test)]
            Self::Stub(_) => None,
        }
    }

    pub fn replace_image_tokens(&self, tokens: &mut [u32]) {
        match self {
            Self::Simple(e) => e.replace_image_tokens(tokens),
            Self::Batch(e) => e.replace_image_tokens(tokens),
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
            Self::Stub(stub) => Ok(stub.render_prompt_for(messages, tools.map(<[_]>::len))),
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
        pixel_values: Option<PixelValues>,
    ) -> Result<GenerationOutput, EngineError> {
        self.generate_with_thinking(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            logprobs,
            top_logprobs,
            ThinkingMode::new(self.enable_thinking(), None),
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
        thinking: ThinkingMode,
        constraint: Option<higgs_engine::constrained::ConstrainedGenerator>,
        pixel_values: Option<PixelValues>,
    ) -> Result<GenerationOutput, EngineError> {
        match self {
            Self::Simple(e) => e.generate_with_thinking(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                thinking,
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
                thinking,
                constraint,
                pixel_values,
            ),
            #[cfg(test)]
            Self::Stub(stub) => {
                let has_constraint = constraint.is_some();
                let constraint_accepts_output = constraint.map(|generator| {
                    stub.constraint_accepts(generator, &stub.scripted_generation_text())
                });
                stub.generate(crate::test_support::StubCall {
                    max_tokens,
                    thinking_enabled: thinking.is_on(),
                    has_constraint,
                    constraint_accepts_output,
                    thinking_budget: request_thinking_budget(thinking),
                })
            }
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
        pixel_values: Option<PixelValues>,
    ) -> Result<(), EngineError> {
        self.generate_streaming_with_thinking(
            prompt_tokens,
            max_tokens,
            params,
            stop_sequences,
            logprobs,
            top_logprobs,
            sender,
            ThinkingMode::new(self.enable_thinking(), None),
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
        thinking: ThinkingMode,
        return_progress: bool,
        constraint: Option<higgs_engine::constrained::ConstrainedGenerator>,
        pixel_values: Option<PixelValues>,
    ) -> Result<(), EngineError> {
        match self {
            Self::Simple(e) => e.generate_streaming_with_thinking(
                prompt_tokens,
                max_tokens,
                params,
                stop_sequences,
                logprobs,
                top_logprobs,
                sender,
                thinking,
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
                thinking,
                return_progress,
                constraint,
                pixel_values,
            ),
            #[cfg(test)]
            Self::Stub(stub) => {
                let has_constraint = constraint.is_some();
                let constraint_accepts_output = constraint.map(|generator| {
                    stub.constraint_accepts(generator, &stub.scripted_stream_text())
                });
                stub.generate_streaming(
                    crate::test_support::StubCall {
                        max_tokens,
                        thinking_enabled: thinking.is_on(),
                        has_constraint,
                        constraint_accepts_output,
                        thinking_budget: request_thinking_budget(thinking),
                    },
                    sender,
                )
            }
        }
    }

    pub fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        match self {
            Self::Simple(e) => e.embed(token_ids),
            Self::Batch(e) => e.embed(token_ids),
            #[cfg(test)]
            Self::Stub(_) => Ok(Vec::new()),
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
    /// Tools the server can run for a model (MCP servers).
    pub tools: Arc<crate::tools::ToolHub>,
}

/// Type alias for the shared state used by Axum handlers.
pub type SharedState = Arc<AppState>;

#[cfg(test)]
#[allow(clippy::panic)]
pub(crate) mod test_support {
    use std::sync::{Arc, Mutex};

    use super::Engine;
    use crate::config::LoadMode;
    use crate::model_pool::{ModelLoader, ModelPool, ModelSpec, PoolLimits};

    /// Loads [`Engine::Stub`]s and records what the pool asked of it.
    #[derive(Default)]
    pub struct StubLoader {
        pub fail_with: Mutex<Option<String>>,
        pub events: Mutex<Vec<String>>,
    }

    impl ModelLoader<Engine> for StubLoader {
        fn load(&self, name: &str) -> Result<Engine, String> {
            let scripted_failure = self
                .fail_with
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            if let Some(reason) = scripted_failure {
                return Err(reason);
            }
            self.events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(format!("load:{name}"));
            Ok(Engine::test_stub(name))
        }

        fn unload(&self, name: &str, model: Arc<Engine>) {
            self.events
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(format!("unload:{name}"));
            drop(model);
        }

        fn estimate_bytes(&self, _name: &str) -> Option<u64> {
            None
        }

        fn memory_in_use(&self) -> Option<u64> {
            None
        }

        fn release_cached_memory(&self) {}
    }

    pub fn stub_spec(name: &str) -> ModelSpec {
        ModelSpec {
            name: name.to_owned(),
            engine_kind: "simple",
            load: LoadMode::OnDemand,
            pinned: false,
            idle_ttl: None,
        }
    }

    pub fn stub_pool_with(
        specs: Vec<ModelSpec>,
        limits: PoolLimits,
    ) -> (Arc<ModelPool<Engine>>, Arc<StubLoader>) {
        let loader = Arc::new(StubLoader::default());
        let shared_loader: Arc<dyn ModelLoader<Engine>> = Arc::<StubLoader>::clone(&loader);
        (ModelPool::new(specs, shared_loader, limits), loader)
    }

    /// App state over a pool of stub engines, with the default config.
    pub fn stub_state(
        specs: Vec<ModelSpec>,
        limits: PoolLimits,
        metrics: Option<Arc<crate::metrics::MetricsStore>>,
    ) -> (super::SharedState, Arc<StubLoader>) {
        let (pool, loader) = stub_pool_with(specs, limits);
        let config = crate::config::HiggsConfig::default();
        let router = crate::router::Router::new(&config, pool)
            .unwrap_or_else(|error| panic!("default config must build a router: {error}"));
        let state = Arc::new(super::AppState {
            router,
            config,
            http_client: reqwest::Client::new(),
            metrics,
            tools: Arc::new(crate::tools::ToolHub::disabled()),
        });
        (state, loader)
    }
}
