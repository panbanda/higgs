//! The inference engine: one worker thread that owns the model and serves
//! every request.
//!
//! The worker interleaves requests at the token level. Models that can decode
//! several sequences in one forward pass do so; each request still carries its
//! own capabilities (constraints, log-probabilities, images, speculative
//! decode), and [`decode_plan`] decides each turn how the active requests are
//! advanced together.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

use higgs_models::{
    AnyCache, AnyModel, LogprobArrays, SamplingParams, apply_penalties,
    batched_cache::AnyBatchedCache, sample, siglip::PixelValues, turboquant::KvCacheConfig,
};
use mlx_rs::{
    Array, Stream,
    ops::indexing::{IndexOp, NewAxis},
    transforms::{async_eval, eval},
    with_stream,
};
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::{
    chat_template::{ChatMessage, ChatTemplateRenderer},
    decode_plan::{
        RequestShape, RoundFacts, RoundPlan, concurrency_limit, free_slots, plan_round,
        speculation_must_yield,
    },
    detokenizer::{IncrementalDetok, find_stop_in_tail},
    disk_prefix_store::{DiskPrefixStore, StoreIdentity},
    engine::{GenerationOutput, GenerationStats, PrefillProgress, StreamingOutput, ThinkingMode},
    error::EngineError,
    generation_support::{
        PrefixResume, ThinkingSupport, embed_tokens, generation_prompt_suffix_len,
        prefix_cache_key, resume_from_prefix, tokenize_chat_prompt,
    },
    mlx_tuning::MlxRuntimeTuning,
    model_loader,
    paged_prefix_cache::{DEFAULT_BLOCK_SIZE, PagedPrefixCache},
    prefix_checkpoints::{
        Arrival, CheckpointPolicy, DEFAULT_CHECKPOINT_BUDGET_BYTES,
        DEFAULT_CHECKPOINT_INTERVAL_TOKENS, MAX_CHECKPOINTS_PER_ENTRY, PrefillCheckpoints,
        RecurrentCheckpoint,
    },
    speculative::{
        CycleBudget, Speculation, SpeculationKind, SpeculationRequest, SpeculationSettings,
        choose_speculation,
    },
};

/// Maximum number of pending requests in the submission queue.
const REQUEST_QUEUE_CAPACITY: usize = 128;

/// Most requests prefilled together in one forward pass.
const PREFILL_GROUP_MAX_ROWS: usize = 8;
/// Prompts longer than this prefill alone: a long prompt would force every
/// shorter prompt in its group to be padded up to its length.
const PREFILL_GROUP_MAX_PROMPT_TOKENS: usize = 1024;
/// Cap on `rows * longest prompt` for one grouped prefill, bounding the
/// activation memory of the padded batch.
const PREFILL_GROUP_TOKEN_BUDGET: usize = 4096;
/// Token id used for left padding. Padding is masked out of attention, so any
/// valid id works.
const PREFILL_PAD_TOKEN: u32 = 0;
static NEXT_PREFILL_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// A generation request submitted to the engine.
struct GenerationRequest {
    prompt_tokens: Vec<u32>,
    max_tokens: u32,
    params: SamplingParams,
    stop_sequences: Vec<String>,
    logprobs: bool,
    top_logprobs: Option<u32>,
    constraint: Option<crate::constrained::ConstrainedGenerator>,
    /// `None` when the request does not think.
    thinking: Option<ThinkingGate>,
    /// Preprocessed images for a vision model. Such a request prefills alone
    /// and never touches the prefix cache.
    pixel_values: Option<PixelValues>,
    response_tx: tokio::sync::mpsc::Sender<StreamingOutput>,
}

impl GenerationRequest {
    fn speculation_request(&self) -> SpeculationRequest {
        #[allow(clippy::float_cmp)]
        let plain_greedy = self.params.temperature == 0.0 && !self.params.has_penalties();
        SpeculationRequest {
            plain_greedy,
            constrained: self.constraint.is_some(),
            logprobs: self.logprobs,
            has_images: self.pixel_values.is_some(),
        }
    }
}

/// An embedding request, answered from the worker thread that owns the model.
struct EmbedRequest {
    token_ids: Vec<u32>,
    reply: std::sync::mpsc::SyncSender<Result<Vec<f32>, EngineError>>,
}

/// Work submitted to the worker thread.
enum WorkItem {
    Generate(Box<GenerationRequest>),
    Embed(EmbedRequest),
}

/// How a prompt too long for one forward pass is split so peak memory stays
/// bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrefillChunking {
    /// Prompts longer than this are split.
    threshold: usize,
    chunk_tokens: u32,
}

impl PrefillChunking {
    fn new(tuning: &MlxRuntimeTuning) -> Self {
        Self {
            threshold: usize::try_from(tuning.chunked_prefill_threshold()).unwrap_or(usize::MAX),
            chunk_tokens: u32::try_from(tuning.chunked_prefill_chunk_size()).unwrap_or(u32::MAX),
        }
    }

    /// Most tokens to feed per forward pass for a prefill of `prompt_tokens`.
    fn quantum(self, prompt_tokens: usize) -> u32 {
        if prompt_tokens > self.threshold {
            self.chunk_tokens.max(1)
        } else {
            u32::MAX
        }
    }
}

/// Most prompts the in-memory prefix cache keeps by default.
pub const DEFAULT_PREFIX_CACHE_ENTRIES: usize = 8;

/// Sizing of the in-memory prefix cache and, for hybrid models, of the recurrent-state
/// checkpoints kept with each entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixCacheOptions {
    /// Most prompts kept; `0` turns the cache off.
    pub max_entries: usize,
    /// Tokens between checkpoints, rounded up to a multiple of the prefill chunk and cache block.
    pub checkpoint_interval_tokens: usize,
    /// Most checkpoints one entry keeps; `0` turns checkpointing off.
    pub max_checkpoints_per_entry: usize,
    /// Bytes of checkpoint state kept across all entries; `0` turns checkpointing off.
    pub checkpoint_budget_bytes: usize,
}

impl Default for PrefixCacheOptions {
    fn default() -> Self {
        Self {
            max_entries: DEFAULT_PREFIX_CACHE_ENTRIES,
            checkpoint_interval_tokens: DEFAULT_CHECKPOINT_INTERVAL_TOKENS,
            max_checkpoints_per_entry: MAX_CHECKPOINTS_PER_ENTRY,
            checkpoint_budget_bytes: DEFAULT_CHECKPOINT_BUDGET_BYTES,
        }
    }
}

/// Where prefixes that outlive the process are kept.
#[derive(Debug, Clone)]
pub struct DiskPrefixConfig {
    pub dir: PathBuf,
    /// Most bytes the store may hold.
    pub budget: u64,
}

/// Everything about how an engine runs that is not the model itself.
pub struct EngineOptions {
    pub kv_cache: KvCacheConfig,
    pub tuning: MlxRuntimeTuning,
    pub raise_wired_limit: bool,
    pub prefill_yield_tokens: Option<u32>,
    pub default_thinking_budget: u32,
    pub disk_prefix: Option<DiskPrefixConfig>,
    pub prefix_cache: PrefixCacheOptions,
    /// Most requests prefilling or decoding at once; `None` is unlimited. A
    /// model that cannot batch its decode serves one request at a time
    /// regardless.
    pub max_concurrent_requests: Option<usize>,
}

/// Settings the worker needs that are fixed when the engine loads.
struct WorkerSettings {
    prefill_yield_tokens: u32,
    chunking: PrefillChunking,
    kv_cache_config: KvCacheConfig,
    /// Trailing chat-template tokens left out of prefix-cache keys.
    generation_suffix_len: usize,
    clear_cache_after_prefill: bool,
    speculation: SpeculationSettings,
    max_concurrent_requests: Option<usize>,
    disk_prefix: Option<DiskPrefixConfig>,
    prefix_cache: PrefixCacheOptions,
    model_dir: PathBuf,
}

/// What the worker reports once the model is loaded.
struct ModelFacts {
    hidden_size: i32,
    decodes_in_batches: bool,
    is_vlm: bool,
    image_size: Option<i32>,
}

/// How an active request produces its next tokens.
enum DecodeMode {
    /// One token per forward pass, alone or batched with other requests.
    Normal,
    /// Several verified tokens per cycle; only while the request is alone.
    Speculative(Speculation),
}

/// An in-flight request being actively decoded.
struct ActiveRequest {
    id: u64,
    /// Empty while the request's KV state lives in a [`DecodeBatch`].
    cache: AnyCache,
    current_token: Array,
    mode: DecodeMode,
    /// The strategy this request may return to speculative decode with after
    /// a handover; `None` when it cannot.
    resumable_speculation: Option<SpeculationKind>,
    /// A token sampled one step ahead of the ones reported, for a request
    /// decoding alone without a batch. It is neither reported nor in the cache.
    in_flight: Option<Array>,
    generated_tokens: Vec<u32>,
    max_tokens: u32,
    params: SamplingParams,
    stop_sequences: Vec<String>,
    logprob_top_n: Option<u32>,
    constraint: Option<crate::constrained::ConstrainedGenerator>,
    thinking: Option<ThinkingGate>,
    response_tx: tokio::sync::mpsc::Sender<StreamingOutput>,
    prompt_len: u32,
    detok: IncrementalDetok,
}

/// Tracks one request's reasoning block so `</think>` can be forced once the
/// budget is spent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ThinkingGate {
    close_token: u32,
    budget: u32,
    closed: bool,
}

impl ThinkingGate {
    const fn new(close_token: u32, budget: u32) -> Self {
        Self {
            close_token,
            budget,
            closed: false,
        }
    }

    /// Whether the token at 1-based `position` of the completion must be
    /// `</think>`. The first token is sampled during prefill and is never forced.
    const fn forces(&self, position: u32) -> bool {
        !self.closed && position >= 2 && position >= self.budget
    }

    const fn observe(&mut self, token_id: u32) {
        if token_id == self.close_token {
            self.closed = true;
        }
    }

    /// How many completion tokens, starting at 1-based `position`, can be
    /// emitted before `</think>` is forced; `None` once the block is closed.
    const fn room(&self, position: u32) -> Option<u32> {
        if self.closed {
            return None;
        }
        if self.forces(position) {
            return Some(0);
        }
        Some(self.budget.saturating_sub(position))
    }
}

/// Requests decoding together through one batched KV cache.
///
/// Rows follow the order of the worker's `active` list at the time the batch
/// was formed; `ids` detects when that list has changed since.
struct DecodeBatch {
    ids: Vec<u64>,
    cache: AnyBatchedCache,
    /// `[B]` most recent token of each row, fed to the next step.
    tokens: Array,
    thinking: Option<BatchThinking>,
    /// `tokens` came from a step whose results clients have not received yet.
    /// Keeping one step in flight lets the next graph be built while the GPU
    /// is still computing this one.
    undelivered: bool,
}

/// The per-row [`ThinkingGate`] state of a [`DecodeBatch`], kept on the GPU.
///
/// Decode is pipelined: the next step is sampled before this step's tokens
/// are read back, so whether a row has just closed its reasoning block is not
/// known on the host. These arrays carry that state through the sampling
/// graph, which swaps in `</think>` for rows that are out of budget.
struct BatchThinking {
    close_token: Array,
    /// `[B]` i32 reasoning budget; `i32::MAX` for rows that do not think.
    budgets: Array,
    /// `[B]` i32 completion tokens sampled so far, including the in-flight step.
    positions: Array,
    /// `[B]` bool: the row has emitted `</think>`.
    closed: Array,
}

impl BatchThinking {
    fn for_rows(active: &[ActiveRequest]) -> Result<Option<Self>, EngineError> {
        let Some(close_token) = active
            .iter()
            .find_map(|ar| ar.thinking.map(|gate| gate.close_token))
        else {
            return Ok(None);
        };
        let rows = i32::try_from(active.len())
            .map_err(|_| EngineError::Generation("Too many batched requests".to_owned()))?;
        let budgets: Vec<i32> = active
            .iter()
            .map(|ar| {
                ar.thinking.map_or(i32::MAX, |gate| {
                    i32::try_from(gate.budget).unwrap_or(i32::MAX)
                })
            })
            .collect();
        let positions: Vec<i32> = active
            .iter()
            .map(|ar| i32::try_from(ar.generated_tokens.len()).unwrap_or(i32::MAX))
            .collect();
        let closed: Vec<bool> = active
            .iter()
            .map(|ar| ar.thinking.is_none_or(|gate| gate.closed))
            .collect();
        Ok(Some(Self {
            close_token: Array::from_slice(&[close_token], &[1]),
            budgets: Array::from_slice(&budgets, &[rows]),
            positions: Array::from_slice(&positions, &[rows]),
            closed: Array::from_slice(&closed, &[rows]),
        }))
    }

    /// Replace the sampled token of every row that is out of reasoning budget
    /// with `</think>`, and advance the per-row state.
    fn apply(&mut self, sampled: &Array) -> Result<Array, EngineError> {
        let mlx = EngineError::Mlx;
        let positions = self.positions.add(Array::from_int(1)).map_err(mlx)?;
        let past_first = positions.ge(Array::from_int(2)).map_err(mlx)?;
        let over_budget = positions.ge(&self.budgets).map_err(mlx)?;
        let still_open = self.closed.logical_not().map_err(mlx)?;
        let forced = over_budget
            .logical_and(&past_first)
            .and_then(|due| due.logical_and(&still_open))
            .map_err(mlx)?;
        let tokens = mlx_rs::ops::select(&forced, &self.close_token, sampled).map_err(mlx)?;
        self.closed = tokens
            .eq(&self.close_token)
            .and_then(|is_close| self.closed.logical_or(&is_close))
            .map_err(mlx)?;
        self.positions = positions;
        Ok(tokens)
    }

    const fn eval_targets(&self) -> [&Array; 2] {
        [&self.positions, &self.closed]
    }
}

/// A request whose prompt is being fed through the model over multiple turns
/// of the worker loop.
struct PendingPrefill {
    request_id: u64,
    req: GenerationRequest,
    prompt_len: u32,
    /// Prompt tokens served from the prefix cache, so `tokens` starts after them.
    cached_tokens: u32,
    tokens: Vec<u32>,
    offset: usize,
    cache: AnyCache,
    /// Where to pause so recurrent state can be kept for the prefix cache.
    checkpoints: PrefillCheckpoints,
    /// The prefix came from the disk store, which already holds it.
    disk_loaded: bool,
    /// The request's images, fed through the vision encoder with the prompt.
    images: Option<Array>,
    /// Keep the last forward pass's hidden states to prime an MTP head.
    capture_hidden: bool,
}

/// What a finished prefill hands to [`complete_prefill`].
struct PrefillDone {
    cache: AnyCache,
    checkpoints: PrefillCheckpoints,
    last_logits: Array,
    /// Hidden states of `prefilled`, when the prefill kept them.
    hidden: Option<Array>,
    /// The prompt tokens the model actually ran, after any cached prefix.
    prefilled: Vec<u32>,
    disk_loaded: bool,
}

enum PrefillAdvance {
    InFlight(PendingPrefill),
    Complete(Option<ActiveRequest>),
}

// ---------------------------------------------------------------------------
// InferenceEngine
// ---------------------------------------------------------------------------

/// Inference engine with interleaved request processing.
///
/// All inference runs on a dedicated background thread that owns the model.
/// Concurrent requests are interleaved at the token level.
pub struct InferenceEngine {
    request_tx: tokio::sync::mpsc::Sender<WorkItem>,
    worker: std::thread::JoinHandle<()>,
    tokenizer: Tokenizer,
    template: Option<ChatTemplateRenderer>,
    model_name: String,
    eos_token_ids: Vec<u32>,
    hidden_size: AtomicI32,
    is_vlm: bool,
    image_size: Option<i32>,
    decodes_in_batches: bool,
    enable_thinking: bool,
    think_close_token: Option<u32>,
    default_thinking_budget: u32,
}

impl InferenceEngine {
    /// Load a model and start the background processing loop.
    pub fn load<P: AsRef<Path>>(dir: P, options: EngineOptions) -> Result<Self, EngineError> {
        let EngineOptions {
            kv_cache,
            tuning,
            raise_wired_limit,
            prefill_yield_tokens,
            default_thinking_budget,
            disk_prefix,
            prefix_cache,
            max_concurrent_requests,
        } = options;
        let model_dir = dir.as_ref();
        let model_name = model_loader::derive_model_name(model_dir);

        tracing::info!(model_dir = %model_dir.display(), "Loading model");

        let tokenizer = model_loader::load_tokenizer(model_dir)?;
        let template = ChatTemplateRenderer::try_from_model_dir(model_dir)?;
        if template.is_none() {
            tracing::warn!("No chat template found; /v1/chat/completions will be unavailable");
        }
        let eos_token_ids = model_loader::extract_eos_tokens(model_dir);
        let thinking = ThinkingSupport::resolve(model_dir, &tokenizer);
        let settings = WorkerSettings {
            prefill_yield_tokens: prefill_yield_tokens.unwrap_or(0),
            chunking: PrefillChunking::new(&tuning),
            kv_cache_config: kv_cache,
            generation_suffix_len: generation_prompt_suffix_len(
                template.as_ref(),
                &tokenizer,
                thinking.enabled,
            ),
            clear_cache_after_prefill: tuning.clear_cache_after_prefill(),
            speculation: SpeculationSettings::from_tuning(&tuning),
            max_concurrent_requests,
            disk_prefix,
            prefix_cache,
            model_dir: model_dir.to_path_buf(),
        };

        let (request_tx, request_rx) = tokio::sync::mpsc::channel(REQUEST_QUEUE_CAPACITY);
        let eos_ids = eos_token_ids.clone();
        let tok = tokenizer.clone();
        let worker_model_dir = model_dir.to_path_buf();
        // MLX arrays are bound to the thread that builds them (`AnyModel` is not `Send`), so
        // the worker loads the model itself and reports back only plain data.
        let (loaded_tx, loaded_rx) =
            std::sync::mpsc::sync_channel::<Result<ModelFacts, EngineError>>(1);

        let worker = std::thread::Builder::new()
            .name("engine".into())
            .spawn(move || {
                let model = match model_loader::load_model(&worker_model_dir) {
                    Ok(model) => model,
                    Err(error) => {
                        let _ = loaded_tx.send(Err(error));
                        return;
                    }
                };
                if let Err(error) = model.make_cache_with_config(settings.kv_cache_config) {
                    let _ = loaded_tx.send(Err(EngineError::Mlx(error)));
                    return;
                }
                let batched_decode = batched_decode_available(&model, settings.kv_cache_config);
                let facts = ModelFacts {
                    hidden_size: model.hidden_size(),
                    decodes_in_batches: concurrency_limit(
                        batched_decode,
                        settings.max_concurrent_requests,
                    ) > 1,
                    is_vlm: model.is_vlm(),
                    image_size: model.image_size(),
                };
                if loaded_tx.send(Ok(facts)).is_err() {
                    return;
                }
                worker_loop(model, &tok, &eos_ids, request_rx, &settings);
            })
            .map_err(|e| EngineError::Generation(format!("Failed to spawn worker: {e}")))?;

        let facts = loaded_rx.recv().map_err(|_| {
            EngineError::Generation("engine worker exited while loading the model".to_owned())
        })??;

        crate::mlx_memory::raise_wired_limit_if(raise_wired_limit);

        log_ready(&model_name, &eos_token_ids, &tuning);

        Ok(Self {
            request_tx,
            worker,
            tokenizer,
            template,
            model_name,
            eos_token_ids,
            hidden_size: AtomicI32::new(facts.hidden_size),
            is_vlm: facts.is_vlm,
            image_size: facts.image_size,
            decodes_in_batches: facts.decodes_in_batches,
            enable_thinking: thinking.enabled,
            think_close_token: thinking.close_token,
            default_thinking_budget,
        })
    }

    /// Free the model, returning only once the worker has dropped its MLX state.
    ///
    /// The worker finishes requests it already accepted, then exits when the queue closes.
    pub fn shutdown(self) {
        let Self {
            request_tx, worker, ..
        } = self;
        drop(request_tx);
        let _ = worker.join();
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// How this model writes tool calls.
    pub fn tool_call_format(&self) -> crate::tool_format::ToolCallFormat {
        self.template
            .as_ref()
            .map_or(crate::tool_format::ToolCallFormat::Auto, |template| {
                template.tool_call_format()
            })
    }

    pub const fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    pub fn hidden_size(&self) -> i32 {
        self.hidden_size.load(Ordering::Relaxed)
    }

    /// Whether the model reasons inside `<think>` tags by default.
    pub const fn enable_thinking(&self) -> bool {
        self.enable_thinking
    }

    /// Whether concurrent requests decode together in batched forward passes.
    pub const fn decodes_in_batches(&self) -> bool {
        self.decodes_in_batches
    }

    /// Whether the loaded model is a vision-language model.
    pub const fn is_vlm(&self) -> bool {
        self.is_vlm
    }

    /// The image size the VLM's vision encoder expects, or `None`.
    pub const fn vlm_image_size(&self) -> Option<i32> {
        self.image_size
    }

    /// Replace image placeholder tokens with `IMAGE_TOKEN_INDEX` in the token
    /// sequence. The `<image>` token ID is looked up from the tokenizer.
    pub fn replace_image_tokens(&self, tokens: &mut [u32]) {
        let Some(image_token_id) = self.tokenizer.token_to_id("<image>") else {
            return;
        };
        let image_token_index = higgs_models::llava_qwen2::IMAGE_TOKEN_INDEX.cast_unsigned();
        for token in tokens.iter_mut() {
            if *token == image_token_id {
                *token = image_token_index;
            }
        }
    }

    /// Apply chat template and tokenize messages with explicit thinking control.
    pub fn prepare_chat_prompt_with_thinking(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        enable_thinking: bool,
    ) -> Result<Vec<u32>, EngineError> {
        tokenize_chat_prompt(
            self.template.as_ref(),
            &self.tokenizer,
            messages,
            tools,
            enable_thinking,
        )
    }

    /// Apply chat template and tokenize messages.
    pub fn prepare_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Vec<u32>, EngineError> {
        self.prepare_chat_prompt_with_thinking(messages, tools, self.enable_thinking)
    }

    /// The gate that enforces a request's reasoning budget, or `None` when
    /// it does not think.
    fn thinking_gate(&self, thinking: ThinkingMode) -> Option<ThinkingGate> {
        let close_token = self.think_close_token?;
        thinking.is_on().then(|| {
            ThinkingGate::new(
                close_token,
                thinking.budget_or(self.default_thinking_budget),
            )
        })
    }

    /// Generate a complete (non-streaming) response.
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub fn generate_with_thinking(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        thinking: ThinkingMode,
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<higgs_models::siglip::PixelValues>,
    ) -> Result<GenerationOutput, EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation("Prompt is empty".to_owned()));
        }

        let prompt_len: u32 = prompt_tokens
            .len()
            .try_into()
            .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))?;

        if max_tokens == 0 {
            return Ok(GenerationOutput {
                text: String::new(),
                finish_reason: "length".to_owned(),
                prompt_tokens: prompt_len,
                completion_tokens: 0,
                token_logprobs: None,
                stats: crate::engine::GenerationStats::default(),
            });
        }

        // Submit request and collect all streaming outputs.
        let (internal_tx, mut internal_rx) = tokio::sync::mpsc::channel(32);

        self.request_tx
            .blocking_send(WorkItem::Generate(Box::new(GenerationRequest {
                prompt_tokens: prompt_tokens.to_vec(),
                max_tokens,
                params: params.clone(),
                stop_sequences: stop_sequences.to_vec(),
                logprobs,
                top_logprobs,
                constraint,
                thinking: self.thinking_gate(thinking),
                pixel_values,
                response_tx: internal_tx,
            })))
            .map_err(|_| EngineError::Generation("Engine shut down".to_owned()))?;

        let mut full_text = String::new();
        let mut finish_reason = "length".to_owned();
        let mut completion_tokens: u32 = 0;
        let mut all_logprobs: Option<Vec<higgs_models::TokenLogprobInfo>> = logprobs.then(Vec::new);
        let mut stats = GenerationStats::default();

        while let Some(output) = internal_rx.blocking_recv() {
            stats.observe(&output, std::time::Instant::now());
            if output.prefill_progress.is_some() {
                continue;
            }
            full_text.push_str(&output.new_text);
            completion_tokens = output.completion_tokens;
            if let Some(ref reason) = output.finish_reason {
                finish_reason.clone_from(reason);
            }
            if let (Some(all_lp), Some(lp)) = (&mut all_logprobs, output.token_logprob) {
                all_lp.push(lp);
            }
            if output.finished {
                break;
            }
        }

        Ok(GenerationOutput {
            text: full_text,
            finish_reason,
            prompt_tokens: prompt_len,
            completion_tokens,
            token_logprobs: all_logprobs,
            stats,
        })
    }

    /// Generate tokens one at a time via the provided channel.
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
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
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<higgs_models::siglip::PixelValues>,
    ) -> Result<(), EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation("Prompt is empty".to_owned()));
        }

        let prompt_len: u32 = prompt_tokens
            .len()
            .try_into()
            .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))?;

        if max_tokens == 0 {
            let _ = sender.blocking_send(StreamingOutput {
                new_text: String::new(),
                finished: true,
                finish_reason: Some("length".to_owned()),
                prompt_tokens: prompt_len,
                completion_tokens: 0,
                token_logprob: None,
                prefill_progress: None,
            });
            return Ok(());
        }

        // Submit request -- the background loop sends tokens directly to
        // an internal channel, and we forward them to the caller's sender.
        let (internal_tx, mut internal_rx) = tokio::sync::mpsc::channel(32);

        self.request_tx
            .blocking_send(WorkItem::Generate(Box::new(GenerationRequest {
                prompt_tokens: prompt_tokens.to_vec(),
                max_tokens,
                params: params.clone(),
                stop_sequences: stop_sequences.to_vec(),
                logprobs,
                top_logprobs,
                constraint,
                thinking: self.thinking_gate(thinking),
                pixel_values,
                response_tx: internal_tx,
            })))
            .map_err(|_| EngineError::Generation("Engine shut down".to_owned()))?;

        while let Some(output) = internal_rx.blocking_recv() {
            if output.prefill_progress.is_some() && !return_progress {
                continue;
            }
            let finished = output.finished;
            if sender.blocking_send(output).is_err() {
                break; // Client disconnected
            }
            if finished {
                break;
            }
        }

        Ok(())
    }

    /// Compute an embedding on the worker thread, which owns the model.
    ///
    /// The worker runs it between decode rounds on a cache of its own, so
    /// requests already generating keep their state.
    pub fn embed(&self, token_ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        let (reply, response) = std::sync::mpsc::sync_channel(1);
        self.request_tx
            .blocking_send(WorkItem::Embed(EmbedRequest {
                token_ids: token_ids.to_vec(),
                reply,
            }))
            .map_err(|_| EngineError::Generation("Engine shut down".to_owned()))?;
        response
            .recv()
            .map_err(|_| EngineError::Generation("Engine shut down".to_owned()))?
    }
}

// ---------------------------------------------------------------------------
// Background worker loop
// ---------------------------------------------------------------------------

/// Prefix cache, in memory and optionally on disk, plus the key convention
/// shared by every lookup and store.
struct PromptCache {
    entries: PagedPrefixCache,
    disk: Option<DiskPrefixStore>,
    generation_suffix_len: usize,
}

/// Disk entries are only worth their write cost for prompts this long.
const DISK_STORE_MIN_KEY_BLOCKS: usize = 8;

impl PromptCache {
    fn new(max_entries: usize, generation_suffix_len: usize) -> Self {
        Self {
            entries: PagedPrefixCache::new(max_entries, DEFAULT_BLOCK_SIZE),
            disk: None,
            generation_suffix_len,
        }
    }

    #[must_use]
    fn with_checkpoint_policy(mut self, policy: CheckpointPolicy) -> Self {
        self.entries = self.entries.with_checkpoint_policy(policy);
        self
    }

    #[must_use]
    fn with_disk_store(mut self, store: Option<DiskPrefixStore>) -> Self {
        self.disk = store;
        self
    }

    /// Where prefill of `prompt` resumes and the cache to resume from.
    ///
    /// Falls back to the disk store when memory has no prefix. The flag says
    /// the prefix came from disk.
    fn resume(
        &mut self,
        prompt: &[u32],
        fresh_cache: impl Fn() -> Result<AnyCache, EngineError>,
    ) -> Result<(PrefixResume, bool), EngineError> {
        let mut found = self.entries.find_longest_prefix(prompt);
        let mut disk_loaded = false;
        if found.is_none() {
            if let Some(from_disk) = self.load_from_disk(prompt, &fresh_cache) {
                found = Some(from_disk);
                disk_loaded = true;
            }
        }
        let resume = resume_from_prefix(found, prompt, fresh_cache)?;
        Ok((resume, disk_loaded))
    }

    fn load_from_disk(
        &mut self,
        prompt: &[u32],
        fresh_cache: &impl Fn() -> Result<AnyCache, EngineError>,
    ) -> Option<crate::paged_prefix_cache::PagedPrefixMatch> {
        let store = self.disk.as_ref()?;
        let prototype = match fresh_cache() {
            Ok(prototype) => prototype,
            Err(error) => {
                tracing::warn!(%error, "Disk prefix cache prototype failed; continuing without it");
                return None;
            }
        };
        let (prefix_len, cache) = match store.load_cache(
            prompt,
            u32::try_from(DEFAULT_BLOCK_SIZE).unwrap_or(u32::MAX),
            &prototype,
        ) {
            Ok(Some(loaded)) => loaded,
            Ok(None) => return None,
            Err(error) => {
                tracing::warn!(%error, "Disk prefix cache lookup failed; continuing without it");
                return None;
            }
        };
        self.entries
            .store(prompt.get(..prefix_len).unwrap_or_default(), &cache);
        let matched = self.entries.find_longest_prefix(prompt);
        if let Some(found) = &matched {
            tracing::info!(
                token_count = found.prefix_len,
                total_len = prompt.len(),
                "Disk prefix hit"
            );
        } else {
            tracing::debug!(
                prefix_len,
                "Disk prefix cache entry was not consumable after materialization"
            );
        }
        matched
    }

    /// Leading tokens of `prompt` the cache would serve.
    fn covered_tokens(&self, prompt: &[u32]) -> usize {
        self.entries.longest_prefix_len(prompt)
    }

    /// The pauses a prefill of `prompt` resumed at `resume_at` makes to keep
    /// state for the cache.
    fn plan_checkpoints(
        &self,
        prompt: &[u32],
        resume_at: usize,
        cache: &AnyCache,
        inherited: Vec<Arc<RecurrentCheckpoint>>,
    ) -> PrefillCheckpoints {
        PrefillCheckpoints::plan(
            self.entries.checkpoint_policy(),
            cache,
            resume_at,
            prefix_cache_key(prompt, self.generation_suffix_len).len(),
            prompt.len(),
            inherited,
        )
    }

    /// Keep the state a prefill left behind in memory, keyed without the chat
    /// template's generation suffix.
    fn remember_in_memory(
        &mut self,
        prompt: &[u32],
        cache: &AnyCache,
        checkpoints: &PrefillCheckpoints,
    ) {
        self.entries.store_with_checkpoints(
            prefix_cache_key(prompt, self.generation_suffix_len),
            cache,
            checkpoints.collected(),
        );
    }

    /// Keep a finished prefill's state in memory and, for a long prompt that
    /// did not come from there, on disk.
    fn remember(
        &mut self,
        prompt: &[u32],
        cache: &AnyCache,
        checkpoints: &PrefillCheckpoints,
        from_disk: bool,
    ) {
        self.remember_in_memory(prompt, cache, checkpoints);
        let key = prefix_cache_key(prompt, self.generation_suffix_len);
        // A hybrid cache that ran the generation suffix holds more tokens than
        // its key, and the disk format cannot cut recurrent state back.
        let cache_matches_key = !matches!(cache, AnyCache::Hybrid(_)) || key.len() == prompt.len();
        if from_disk
            || !cache_matches_key
            || key.len() < DISK_STORE_MIN_KEY_BLOCKS * DEFAULT_BLOCK_SIZE
        {
            return;
        }
        if let Some(store) = &self.disk {
            if let Err(error) = store.store_cache(
                key,
                u32::try_from(DEFAULT_BLOCK_SIZE).unwrap_or(u32::MAX),
                cache,
            ) {
                tracing::warn!(%error, "Disk prefix cache store failed; continuing without it");
            }
        }
    }
}

fn open_disk_store(settings: &WorkerSettings) -> Option<DiskPrefixStore> {
    let config = settings.disk_prefix.as_ref()?;
    match DiskPrefixStore::new(
        &config.dir,
        config.budget,
        StoreIdentity::from_model_dir(
            &settings.model_dir,
            format!("{:?}", settings.kv_cache_config),
        ),
    ) {
        Ok(store) => Some(store),
        Err(error) => {
            tracing::warn!(
                %error,
                path = %config.dir.display(),
                "Disk prefix cache disabled; continuing without it"
            );
            None
        }
    }
}

fn log_ready(model_name: &str, eos_token_ids: &[u32], tuning: &MlxRuntimeTuning) {
    tracing::info!(
        model_name,
        eos_tokens = ?eos_token_ids,
        requested_mlx_profile = tuning.requested_profile().as_str(),
        effective_mlx_profile = tuning.resolved_profile().as_str(),
        chunked_prefill_threshold = tuning.chunked_prefill_threshold(),
        chunked_prefill_chunk_size = tuning.chunked_prefill_chunk_size(),
        clear_cache_after_prefill = tuning.clear_cache_after_prefill(),
        mtp_enabled = tuning.enable_mtp(),
        mtp_draft_n_max = tuning.mtp_draft_n_max(),
        "Engine ready"
    );
}

/// Whether requests can share a batched decode pass. `TurboQuant` caches quantize per
/// request and cannot be stacked into a batched cache.
const fn batched_decode_available(model: &AnyModel, kv_cache_config: KvCacheConfig) -> bool {
    model.supports_batched_decode() && !kv_cache_config.is_turboquant()
}

/// Every MLX array the worker holds, owned by the worker thread.
struct Worker {
    model: AnyModel,
    prompt_cache: PromptCache,
    facts: RoundFacts,
    /// Most requests that may be prefilling or decoding at once.
    concurrency_limit: usize,
    grouped_prefill: bool,
    active: Vec<ActiveRequest>,
    decode_batch: Option<DecodeBatch>,
    shapes: Vec<RequestShape>,
}

#[allow(clippy::too_many_lines)]
fn worker_loop(
    model: AnyModel,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    mut request_rx: tokio::sync::mpsc::Receiver<WorkItem>,
    settings: &WorkerSettings,
) {
    let prefill_yield_tokens = settings.prefill_yield_tokens;
    let turboquant = settings.kv_cache_config.is_turboquant();
    let batched_decode = batched_decode_available(&model, settings.kv_cache_config);
    if !batched_decode {
        tracing::info!(
            "This model decodes one request at a time (no batched decode for its architecture or KV cache); requests queue"
        );
    }
    let mut worker = Worker {
        grouped_prefill: model.supports_batched_prefill() && !turboquant,
        model,
        prompt_cache: PromptCache::new(
            settings.prefix_cache.max_entries,
            settings.generation_suffix_len,
        )
        .with_checkpoint_policy(CheckpointPolicy::aligned(
            settings.prefix_cache.checkpoint_interval_tokens,
            settings.prefix_cache.max_checkpoints_per_entry,
            settings.prefix_cache.checkpoint_budget_bytes,
            usize::try_from(settings.chunking.chunk_tokens).unwrap_or(1),
        ))
        .with_disk_store(open_disk_store(settings)),
        facts: RoundFacts { batched_decode },
        concurrency_limit: concurrency_limit(batched_decode, settings.max_concurrent_requests),
        active: Vec::new(),
        decode_batch: None,
        shapes: Vec::new(),
    };
    let mut pending_prefill: Option<PendingPrefill> = None;
    let mut waiting: std::collections::VecDeque<GenerationRequest> =
        std::collections::VecDeque::new();

    loop {
        // 1. Admit queued requests, blocking only when there is nothing to do.
        while waiting.len() < REQUEST_QUEUE_CAPACITY {
            match request_rx.try_recv() {
                Ok(item) => admit(item, &mut worker.model, settings, &mut waiting),
                Err(_) => break,
            }
        }
        if worker.active.is_empty() && pending_prefill.is_none() && waiting.is_empty() {
            let Some(item) = request_rx.blocking_recv() else {
                tracing::info!("Request channel closed, worker shutting down");
                return;
            };
            admit(item, &mut worker.model, settings, &mut waiting);
        }

        // 2. Prefill. Each iteration prefills at most one request (or one
        //    group of short prompts in a single pass), so long prefills are
        //    interleaved with decode steps for requests already in flight.
        let busy = worker.active.len() + usize::from(pending_prefill.is_some());
        let slots = free_slots(worker.concurrency_limit, busy);
        if prefill_yield_tokens == 0 {
            if slots > 0 {
                worker.prefill_next(tokenizer, eos_token_ids, settings, &mut waiting, slots);
            }
        } else {
            if pending_prefill.is_none() && slots > 0 {
                if let Some(req) = waiting.pop_front() {
                    match start_prefill(&worker.model, &mut worker.prompt_cache, settings, req) {
                        Ok(prefill) => pending_prefill = Some(prefill),
                        Err(e) => tracing::error!(error = %e, "Prefill failed"),
                    }
                }
            }
            if let Some(prefill) = pending_prefill.take() {
                match advance_prefill(
                    &mut worker.model,
                    &mut worker.prompt_cache,
                    tokenizer,
                    eos_token_ids,
                    settings,
                    prefill,
                    prefill_yield_tokens,
                ) {
                    Ok(PrefillAdvance::InFlight(resumed_prefill)) => {
                        pending_prefill = Some(resumed_prefill);
                    }
                    Ok(PrefillAdvance::Complete(Some(ar))) => worker.active.push(ar),
                    Ok(PrefillAdvance::Complete(None)) => {}
                    Err(e) => tracing::error!(error = %e, "Prefill failed"),
                }
            }
        }
        if worker.active.is_empty() {
            continue;
        }

        // 3. Advance every active request one round.
        let mut finished_indices = Vec::new();
        with_stream(&Stream::new(), || {
            worker.decode_round(tokenizer, eos_token_ids, settings, &mut finished_indices);
        });

        // 4. Remove finished requests (descending order keeps indices valid).
        finished_indices.sort_unstable();
        finished_indices.dedup();
        for i in finished_indices.into_iter().rev() {
            worker.active.swap_remove(i);
        }
        if worker.active.is_empty() {
            worker.decode_batch = None;
        }
    }
}

impl Worker {
    /// Prefill from the front of `waiting`: a group of short prompts when the
    /// model can prefill them together, else one request.
    fn prefill_next(
        &mut self,
        tokenizer: &Tokenizer,
        eos_token_ids: &[u32],
        settings: &WorkerSettings,
        waiting: &mut std::collections::VecDeque<GenerationRequest>,
        slots: usize,
    ) {
        let group = if self.grouped_prefill {
            take_prefill_group(waiting, &self.prompt_cache, slots)
        } else {
            Vec::new()
        };
        if group.len() > 1 {
            prefill_group(self, tokenizer, eos_token_ids, settings, group);
        } else if let Some(req) = group.into_iter().next().or_else(|| waiting.pop_front()) {
            match prefill_request(
                &mut self.model,
                &mut self.prompt_cache,
                tokenizer,
                eos_token_ids,
                settings,
                req,
            ) {
                Ok(Some(ar)) => self.active.push(ar),
                Ok(None) => {}
                Err(e) => tracing::error!(error = %e, "Prefill failed"),
            }
        }
    }

    /// Advance every active request, following [`plan_round`].
    ///
    /// Models with batched decode step every plain request in one forward
    /// pass (a lone request included, which keeps the one-step-ahead
    /// pipelining). Constrained generation and logprobs need per-request
    /// handling and use the sequential round.
    fn decode_round(
        &mut self,
        tokenizer: &Tokenizer,
        eos_token_ids: &[u32],
        settings: &WorkerSettings,
        finished_indices: &mut Vec<usize>,
    ) {
        self.refresh_shapes();
        if speculation_must_yield(&self.shapes) {
            for (index, ar) in self.active.iter_mut().enumerate() {
                if hand_over_speculation(ar, tokenizer, eos_token_ids) {
                    finished_indices.push(index);
                }
            }
            if !finished_indices.is_empty() {
                return;
            }
            self.refresh_shapes();
        }
        let plan = plan_round(self.facts, &self.shapes);

        if plan != RoundPlan::Batched {
            if let Some(batch) = self.decode_batch.take() {
                if let Err(e) = disband_decode_batch(
                    batch,
                    &mut self.active,
                    tokenizer,
                    eos_token_ids,
                    finished_indices,
                ) {
                    tracing::error!(error = %e, "Leaving batched decode failed");
                    fail_requests(&self.active, finished_indices);
                }
                // Delivering the in-flight step may have finished requests;
                // drop those before stepping the rest.
                if !finished_indices.is_empty() {
                    return;
                }
            }
        }
        if !matches!(plan, RoundPlan::Lookahead) {
            for (index, ar) in self.active.iter_mut().enumerate() {
                if settle_in_flight(ar, tokenizer, eos_token_ids) {
                    finished_indices.push(index);
                }
            }
            if !finished_indices.is_empty() {
                return;
            }
        }

        match plan {
            RoundPlan::Batched => {
                if let Err(e) = run_batched_decode_round(
                    &mut self.model,
                    &mut self.decode_batch,
                    &mut self.active,
                    tokenizer,
                    eos_token_ids,
                    finished_indices,
                ) {
                    tracing::error!(error = %e, "Batched decode round failed");
                    self.decode_batch = None;
                    fail_requests(&self.active, finished_indices);
                }
            }
            RoundPlan::Sequential => run_pipelined_decode_round(
                &mut self.model,
                &mut self.active,
                tokenizer,
                eos_token_ids,
                finished_indices,
            ),
            RoundPlan::Lookahead | RoundPlan::Speculative | RoundPlan::ResumeSpeculation => {
                self.run_single_request_round(
                    plan,
                    tokenizer,
                    eos_token_ids,
                    settings,
                    finished_indices,
                );
            }
        }
    }

    /// Rounds that only run while exactly one request is active.
    fn run_single_request_round(
        &mut self,
        plan: RoundPlan,
        tokenizer: &Tokenizer,
        eos_token_ids: &[u32],
        settings: &WorkerSettings,
        finished_indices: &mut Vec<usize>,
    ) {
        let Some(ar) = self.active.first_mut() else {
            return;
        };
        let outcome = match plan {
            RoundPlan::Lookahead => {
                run_lookahead_round(&mut self.model, ar, tokenizer, eos_token_ids)
            }
            RoundPlan::ResumeSpeculation => {
                resume_speculation(&mut self.model, ar, &settings.speculation).map(|()| false)
            }
            RoundPlan::Speculative | RoundPlan::Batched | RoundPlan::Sequential => {
                run_speculative_round(&mut self.model, ar, tokenizer, eos_token_ids)
            }
        };
        match outcome {
            Ok(true) => finished_indices.push(0),
            Ok(false) => {}
            Err(e) => {
                tracing::error!(error = %e, "Decode round failed");
                fail_requests(&self.active, finished_indices);
            }
        }
    }

    fn refresh_shapes(&mut self) {
        self.shapes.clear();
        self.shapes
            .extend(self.active.iter().map(ActiveRequest::shape));
    }
}

/// Queue a generation request, or answer an embedding right away.
///
/// An embedding runs on its own cache, so it leaves requests that are already
/// decoding untouched.
fn admit(
    item: WorkItem,
    model: &mut AnyModel,
    settings: &WorkerSettings,
    waiting: &mut std::collections::VecDeque<GenerationRequest>,
) {
    match item {
        WorkItem::Generate(request) => waiting.push_back(*request),
        WorkItem::Embed(request) => {
            let embedding = embed_tokens(model, settings.kv_cache_config, &request.token_ids);
            let _ = request.reply.send(embedding);
        }
    }
}

impl ActiveRequest {
    const fn shape(&self) -> RequestShape {
        RequestShape {
            speculating: matches!(self.mode, DecodeMode::Speculative(_)),
            could_speculate: self.resumable_speculation.is_some()
                && matches!(self.mode, DecodeMode::Normal),
            constrained: self.constraint.is_some(),
            logprobs: self.logprob_top_n.is_some(),
        }
    }

    /// The 1-based completion position of the next token to be reported.
    fn next_position(&self) -> u32 {
        u32::try_from(self.generated_tokens.len())
            .unwrap_or(u32::MAX)
            .saturating_add(1)
    }
}

/// End speculative decode for a request that now shares the engine. Returns
/// whether the request finished.
///
/// A speculating request is at a cycle boundary: its cache holds the prompt
/// and everything reported, and the model has already settled on the next
/// token without feeding it. Reporting that token (or `</think>` when the
/// reasoning budget demands it) leaves the request exactly where ordinary
/// decode expects: the last reported token is the one not yet in the cache.
fn hand_over_speculation(
    ar: &mut ActiveRequest,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
) -> bool {
    let DecodeMode::Speculative(speculation) = std::mem::replace(&mut ar.mode, DecodeMode::Normal)
    else {
        return false;
    };
    speculation.log_summary("handover");
    ar.resumable_speculation =
        Some(speculation.kind()).filter(|kind| *kind == SpeculationKind::PromptLookup);
    let token = match ar.thinking {
        Some(gate) if gate.forces(ar.next_position()) => gate.close_token,
        _ => speculation.confirmed(),
    };
    ar.current_token = Array::from_slice(&[token], &[1]);
    deliver_token(ar, token, None, tokenizer, eos_token_ids)
}

/// Return a request on the normal path to prompt-lookup decode. Its cache
/// holds everything reported except the last token, which is what starting
/// speculative decode feeds first.
fn resume_speculation(
    model: &mut AnyModel,
    ar: &mut ActiveRequest,
    settings: &SpeculationSettings,
) -> Result<(), EngineError> {
    let Some(&last_token) = ar.generated_tokens.last() else {
        return Ok(());
    };
    ar.resumable_speculation = None;
    ar.mode = DecodeMode::Speculative(Speculation::start_prompt_lookup(
        model,
        &mut ar.cache,
        settings,
        last_token,
    )?);
    Ok(())
}

/// One speculative cycle for the lone speculating request. Returns whether the
/// request finished.
fn run_speculative_round(
    model: &mut AnyModel,
    ar: &mut ActiveRequest,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
) -> Result<bool, EngineError> {
    let DecodeMode::Speculative(mut speculation) =
        std::mem::replace(&mut ar.mode, DecodeMode::Normal)
    else {
        return Ok(false);
    };
    let completed = u32::try_from(ar.generated_tokens.len()).unwrap_or(u32::MAX);
    let to_usize = |count: u32| usize::try_from(count).unwrap_or(usize::MAX);
    let budget = CycleBudget {
        remaining: to_usize(ar.max_tokens.saturating_sub(completed)),
        thinking_room: ar
            .thinking
            .and_then(|gate| gate.room(ar.next_position()))
            .map(to_usize),
    };
    let cycle = speculation.cycle(model, &mut ar.cache, &ar.generated_tokens, budget)?;
    let Some(tokens) = cycle else {
        ar.mode = DecodeMode::Speculative(speculation);
        return Ok(hand_over_speculation(ar, tokenizer, eos_token_ids));
    };
    for token in tokens {
        if deliver_token(ar, token, None, tokenizer, eos_token_ids) {
            speculation.log_summary("finished");
            return Ok(true);
        }
    }
    ar.mode = DecodeMode::Speculative(speculation);
    Ok(false)
}

/// Sample the token that follows `input`, building the graph without evaluating it.
fn lookahead_step(
    model: &mut AnyModel,
    ar: &mut ActiveRequest,
    input: &Array,
) -> Result<Array, EngineError> {
    let decode_input = input.index((.., NewAxis));
    let logits = model
        .forward(&decode_input, None, &mut ar.cache)
        .map_err(EngineError::Mlx)?;
    let sliced = logits.index((.., -1, ..));
    let penalized =
        apply_penalties(&sliced, &ar.generated_tokens, &ar.params).map_err(EngineError::Mlx)?;
    sample(&penalized, &ar.params).map_err(EngineError::Mlx)
}

/// Replace `sampled` with `</think>` when the reasoning budget requires the
/// next reported token to be that.
fn with_forced_close(ar: &ActiveRequest, sampled: Array) -> Result<Array, EngineError> {
    match ar.thinking {
        Some(gate) if gate.forces(ar.next_position()) => {
            Array::from_slice(&[gate.close_token], sampled.shape())
                .as_dtype(sampled.dtype())
                .map_err(EngineError::Mlx)
        }
        _ => Ok(sampled),
    }
}

/// Decode a lone plain request one step ahead of the token it reports: the
/// next graph is built and submitted while the GPU is still computing the
/// previous one. Returns whether the request finished.
fn run_lookahead_round(
    model: &mut AnyModel,
    ar: &mut ActiveRequest,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
) -> Result<bool, EngineError> {
    let input = if let Some(token) = ar.in_flight.take() {
        token
    } else {
        let current = ar.current_token.clone();
        let sampled = lookahead_step(model, ar, &current)?;
        let first = with_forced_close(ar, sampled)?;
        async_eval([&first]).map_err(EngineError::Mlx)?;
        first
    };
    let following = lookahead_step(model, ar, &input)?;
    async_eval([&following]).map_err(EngineError::Mlx)?;

    let token_id: u32 = input.item_cast();
    ar.current_token = input;
    if deliver_token(ar, token_id, None, tokenizer, eos_token_ids) {
        return Ok(true);
    }
    ar.in_flight = Some(with_forced_close(ar, following)?);
    Ok(false)
}

/// Report the token a lookahead round left in flight, so the request is in the
/// state every other round starts from. Returns whether the request finished.
fn settle_in_flight(ar: &mut ActiveRequest, tokenizer: &Tokenizer, eos_token_ids: &[u32]) -> bool {
    let Some(token) = ar.in_flight.take() else {
        return false;
    };
    let token_id: u32 = token.item_cast();
    ar.current_token = token;
    deliver_token(ar, token_id, None, tokenizer, eos_token_ids)
}

/// Report a decode failure to every active request and mark them finished.
fn fail_requests(active: &[ActiveRequest], finished_indices: &mut Vec<usize>) {
    for (i, ar) in active.iter().enumerate() {
        let _ = ar.response_tx.blocking_send(StreamingOutput {
            new_text: String::new(),
            finished: true,
            finish_reason: Some("error".to_owned()),
            prompt_tokens: ar.prompt_len,
            completion_tokens: 0,
            token_logprob: None,
            prefill_progress: None,
        });
        finished_indices.push(i);
    }
}

/// Pipelined decode: build each request's computation graph with `async_eval`,
/// then materialize all results. GPU processes request N while CPU builds
/// request N+1's graph.
fn run_pipelined_decode_round(
    model: &mut AnyModel,
    active: &mut [ActiveRequest],
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    finished_indices: &mut Vec<usize>,
) {
    // Phase 1: Build computation graphs and submit to GPU.
    let mut graphs: Vec<Option<DecodeGraphResult>> = Vec::with_capacity(active.len());
    for (i, ar) in active.iter_mut().enumerate() {
        match build_decode_graph(model, ar) {
            Ok(result) => {
                let mut eval_targets: Vec<&Array> = vec![&result.next_token];
                if let Some(ref lp) = result.logprob_data {
                    eval_targets.extend(lp.eval_targets());
                }
                match async_eval(eval_targets) {
                    Ok(()) => graphs.push(Some(result)),
                    Err(e) => {
                        tracing::error!(error = %e, "async_eval failed");
                        let _ = ar.response_tx.blocking_send(StreamingOutput {
                            new_text: String::new(),
                            finished: true,
                            finish_reason: Some("error".to_owned()),
                            prompt_tokens: ar.prompt_len,
                            completion_tokens: 0,
                            token_logprob: None,
                            prefill_progress: None,
                        });
                        finished_indices.push(i);
                        graphs.push(None);
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "Decode graph build failed");
                let _ = ar.response_tx.blocking_send(StreamingOutput {
                    new_text: String::new(),
                    finished: true,
                    finish_reason: Some("error".to_owned()),
                    prompt_tokens: ar.prompt_len,
                    completion_tokens: 0,
                    token_logprob: None,
                    prefill_progress: None,
                });
                finished_indices.push(i);
                graphs.push(None);
            }
        }
    }

    // Phase 2: Materialize results.
    for (i, (ar, graph)) in active.iter_mut().zip(graphs).enumerate() {
        let Some(result) = graph else { continue };
        if materialize_decode_step(ar, result, tokenizer, eos_token_ids) {
            finished_indices.push(i);
        }
    }
}

/// Batched decode: one forward pass appends a token to every active request.
///
/// The step for round N+1 is built and submitted before round N's tokens are
/// read back, so the CPU prepares the next graph while the GPU works.
fn run_batched_decode_round(
    model: &mut AnyModel,
    decode_batch: &mut Option<DecodeBatch>,
    active: &mut [ActiveRequest],
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    finished_indices: &mut Vec<usize>,
) -> Result<(), EngineError> {
    let membership_changed = decode_batch
        .as_ref()
        .is_some_and(|batch| !batch.ids.iter().eq(active.iter().map(|ar| &ar.id)));
    if membership_changed {
        if let Some(batch) = decode_batch.take() {
            disband_decode_batch(batch, active, tokenizer, eos_token_ids, finished_indices)?;
        }
        // Delivering the in-flight step may have finished more requests; let
        // the worker drop them so the new batch forms from the survivors.
        if !finished_indices.is_empty() {
            return Ok(());
        }
    }

    let batch = match decode_batch {
        Some(batch) => batch,
        None => decode_batch.insert(form_decode_batch(active)?),
    };

    let rows = i32::try_from(batch.ids.len())
        .map_err(|_| EngineError::Generation("Too many batched requests".to_owned()))?;
    let input = batch.tokens.reshape(&[rows, 1]).map_err(EngineError::Mlx)?;
    let logits = model
        .forward_batched(&input, &mut batch.cache)
        .map_err(EngineError::Mlx)?;
    let last_logits = logits.index((.., -1, ..));
    let sampled = sample_batch(&last_logits, active)?;
    let next_tokens = match batch.thinking.as_mut() {
        Some(thinking) => thinking.apply(&sampled)?,
        None => sampled,
    };
    let mut eval_targets = vec![&next_tokens];
    if let Some(thinking) = batch.thinking.as_ref() {
        eval_targets.extend(thinking.eval_targets());
    }
    async_eval(eval_targets).map_err(EngineError::Mlx)?;

    deliver_batch_tokens(batch, active, tokenizer, eos_token_ids, finished_indices)?;
    batch.tokens = next_tokens;
    batch.undelivered = true;
    Ok(())
}

/// Sample one token per row of `[B, vocab]` logits, as a `[B]` u32 array.
fn sample_batch(last_logits: &Array, active: &[ActiveRequest]) -> Result<Array, EngineError> {
    #[allow(clippy::float_cmp)]
    let all_plain_greedy = active
        .iter()
        .all(|ar| ar.params.temperature == 0.0 && !ar.params.has_penalties());
    let tokens = if let (true, Some(first)) = (all_plain_greedy, active.first()) {
        sample(last_logits, &first.params).map_err(EngineError::Mlx)?
    } else {
        let mut row_tokens = Vec::with_capacity(active.len());
        for (ar, row) in active.iter().zip(0_i32..) {
            let row_logits = last_logits.index((row..row + 1, ..));
            let penalized = apply_penalties(&row_logits, &ar.generated_tokens, &ar.params)
                .map_err(EngineError::Mlx)?;
            row_tokens.push(sample(&penalized, &ar.params).map_err(EngineError::Mlx)?);
        }
        mlx_rs::ops::concatenate(&row_tokens.iter().collect::<Vec<_>>(), 0)
            .map_err(EngineError::Mlx)?
    };
    tokens
        .as_dtype(mlx_rs::Dtype::Uint32)
        .map_err(EngineError::Mlx)
}

/// Move every active request's KV state into one batched cache.
fn form_decode_batch(active: &mut [ActiveRequest]) -> Result<DecodeBatch, EngineError> {
    let mut rows = Vec::with_capacity(active.len());
    let mut last_tokens = Vec::with_capacity(active.len());
    for ar in active.iter_mut() {
        rows.push(std::mem::replace(&mut ar.cache, AnyCache::KV(Vec::new())));
        last_tokens.push(ar.generated_tokens.last().copied().ok_or_else(|| {
            EngineError::Generation("Active request has no generated token".to_owned())
        })?);
    }
    let rows_len = i32::try_from(last_tokens.len())
        .map_err(|_| EngineError::Generation("Too many batched requests".to_owned()))?;
    Ok(DecodeBatch {
        ids: active.iter().map(|ar| ar.id).collect(),
        cache: AnyBatchedCache::from_rows(rows).map_err(EngineError::Mlx)?,
        tokens: Array::from_slice(&last_tokens, &[rows_len]),
        thinking: BatchThinking::for_rows(active)?,
        undelivered: false,
    })
}

/// Read back the in-flight step's tokens and stream them to their requests.
/// Rows whose request has already left `active` are skipped.
fn deliver_batch_tokens(
    batch: &mut DecodeBatch,
    active: &mut [ActiveRequest],
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    finished_indices: &mut Vec<usize>,
) -> Result<(), EngineError> {
    if !batch.undelivered {
        return Ok(());
    }
    batch.undelivered = false;
    batch.tokens.eval().map_err(EngineError::Mlx)?;
    let token_ids = batch.tokens.as_slice::<u32>();
    for (request_id, &token_id) in batch.ids.iter().zip(token_ids) {
        let Some(index) = active.iter().position(|ar| ar.id == *request_id) else {
            continue;
        };
        if finished_indices.contains(&index) {
            continue;
        }
        let Some(ar) = active.get_mut(index) else {
            continue;
        };
        if deliver_token(ar, token_id, None, tokenizer, eos_token_ids) {
            finished_indices.push(index);
        }
    }
    Ok(())
}

/// Hand each surviving request its own KV cache back so it can decode alone
/// or join a differently shaped batch.
fn disband_decode_batch(
    mut batch: DecodeBatch,
    active: &mut [ActiveRequest],
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    finished_indices: &mut Vec<usize>,
) -> Result<(), EngineError> {
    deliver_batch_tokens(
        &mut batch,
        active,
        tokenizer,
        eos_token_ids,
        finished_indices,
    )?;
    let rows = batch.cache.into_rows().map_err(EngineError::Mlx)?;
    for (request_id, row) in batch.ids.iter().zip(rows) {
        let Some(ar) = active.iter_mut().find(|ar| ar.id == *request_id) else {
            continue;
        };
        ar.cache = row;
        if let Some(&last_token) = ar.generated_tokens.last() {
            ar.current_token = Array::from_slice(&[last_token], &[1]);
        }
    }
    Ok(())
}

/// Pull out the waiting requests that can share one prefill pass: short
/// prompts for which the prefix cache covers less than half. Unrelated chats
/// still share a few leading template tokens, and recomputing those inside
/// one batched pass beats prefilling each request separately. Requests left
/// behind keep their order. A request with images always prefills alone.
fn take_prefill_group(
    waiting: &mut std::collections::VecDeque<GenerationRequest>,
    prompt_cache: &PromptCache,
    slots: usize,
) -> Vec<GenerationRequest> {
    if waiting.len() < 2 || slots < 2 {
        return Vec::new();
    }
    let mut group: Vec<GenerationRequest> = Vec::new();
    let mut longest = 0;
    let mut left_behind = std::collections::VecDeque::with_capacity(waiting.len());
    for req in waiting.drain(..) {
        let tokens = req.prompt_tokens.len();
        let padded_longest = longest.max(tokens);
        let fits = group.len() < PREFILL_GROUP_MAX_ROWS.min(slots)
            && req.pixel_values.is_none()
            && tokens <= PREFILL_GROUP_MAX_PROMPT_TOKENS
            && padded_longest * (group.len() + 1) <= PREFILL_GROUP_TOKEN_BUDGET
            && prompt_cache.covered_tokens(&req.prompt_tokens) * 2 < tokens;
        if fits {
            longest = padded_longest;
            group.push(req);
        } else {
            left_behind.push_back(req);
        }
    }
    *waiting = left_behind;
    group
}

/// Prefill a group of requests in one left-padded forward pass and move them
/// to `active`. Falls back to one-at-a-time prefill if the batched pass fails.
#[allow(clippy::too_many_lines)]
fn prefill_group(
    worker: &mut Worker,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    settings: &WorkerSettings,
    group: Vec<GenerationRequest>,
) {
    let started = std::time::Instant::now();
    for req in &group {
        let prompt_len = u32::try_from(req.prompt_tokens.len()).unwrap_or(u32::MAX);
        announce_prefill(req, 0, 0, prompt_len);
    }
    let batched = with_stream(&Stream::new(), || {
        let longest = group
            .iter()
            .map(|req| req.prompt_tokens.len())
            .max()
            .unwrap_or(0);
        let mut padded = Vec::with_capacity(group.len() * longest);
        let mut lengths = Vec::with_capacity(group.len());
        for req in &group {
            padded.extend(std::iter::repeat_n(
                PREFILL_PAD_TOKEN,
                longest - req.prompt_tokens.len(),
            ));
            padded.extend_from_slice(&req.prompt_tokens);
            lengths.push(
                i32::try_from(req.prompt_tokens.len())
                    .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))?,
            );
        }
        let row_count = i32::try_from(group.len())
            .map_err(|_| EngineError::Generation("Too many grouped requests".to_owned()))?;
        let steps = i32::try_from(longest)
            .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))?;
        let input = Array::from_slice(&padded, &[row_count, steps]);
        let (logits, cache) = worker
            .model
            .prefill_batched(&input, &lengths)
            .map_err(EngineError::Mlx)?;
        let last_logits = logits.index((.., -1, ..));
        let mut eval_targets = vec![&last_logits];
        eval_targets.extend(cache.eval_targets());
        eval(eval_targets).map_err(EngineError::Mlx)?;
        let rows = cache.into_rows().map_err(EngineError::Mlx)?;
        Ok::<_, EngineError>((last_logits, rows))
    });

    let (last_logits, rows) = match batched {
        Ok(result) => result,
        Err(e) => {
            tracing::warn!(error = %e, "Grouped prefill failed; prefilling requests one at a time");
            for req in group {
                match prefill_request(
                    &mut worker.model,
                    &mut worker.prompt_cache,
                    tokenizer,
                    eos_token_ids,
                    settings,
                    req,
                ) {
                    Ok(Some(ar)) => worker.active.push(ar),
                    Ok(None) => {}
                    Err(prefill_error) => {
                        tracing::error!(error = %prefill_error, "Prefill failed");
                    }
                }
            }
            return;
        }
    };

    tracing::debug!(
        rows = group.len(),
        forward_ms = started.elapsed().as_millis(),
        "Grouped prefill forward done"
    );
    for ((req, row), row_index) in group.into_iter().zip(rows).zip(0_i32..) {
        let prompt_len = u32::try_from(req.prompt_tokens.len()).unwrap_or(u32::MAX);
        let row_logits = last_logits.index((row_index..row_index + 1, ..));
        let prefilled = req.prompt_tokens.clone();
        let completed = with_stream(&Stream::new(), || {
            complete_prefill(
                &mut worker.model,
                Some(&mut worker.prompt_cache),
                tokenizer,
                eos_token_ids,
                settings,
                req,
                prompt_len,
                PrefillDone {
                    cache: row,
                    checkpoints: PrefillCheckpoints::default(),
                    last_logits: row_logits,
                    hidden: None,
                    prefilled,
                    disk_loaded: false,
                },
            )
        });
        match completed {
            Ok(Some(ar)) => worker.active.push(ar),
            Ok(None) => {}
            Err(e) => tracing::error!(error = %e, "Prefill failed"),
        }
    }
    tracing::debug!(
        total_ms = started.elapsed().as_millis(),
        "Grouped prefill complete"
    );
}

/// Tell the caller that prompt evaluation is starting and how much of the
/// prompt the prefix cache already covers. Never blocks the worker: a full
/// channel only loses this notification.
fn announce_prefill(req: &GenerationRequest, processed: u32, cached_tokens: u32, prompt_len: u32) {
    let _ = req.response_tx.try_send(StreamingOutput {
        new_text: String::new(),
        finished: false,
        finish_reason: None,
        prompt_tokens: prompt_len,
        completion_tokens: 0,
        token_logprob: None,
        prefill_progress: Some(PrefillProgress {
            processed,
            cached: cached_tokens,
            total: prompt_len,
        }),
    });
}

/// Set up cache reuse once, before the prompt begins advancing through chunks.
fn start_prefill(
    model: &AnyModel,
    prompt_cache: &mut PromptCache,
    settings: &WorkerSettings,
    req: GenerationRequest,
) -> Result<PendingPrefill, EngineError> {
    let prompt_len = req
        .prompt_tokens
        .len()
        .try_into()
        .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))?;
    let fresh_cache = || {
        model
            .make_cache_with_config(settings.kv_cache_config)
            .map_err(EngineError::Mlx)
    };
    // Different images give different KV states for identical tokens, so a
    // request with images neither reads nor writes the prefix cache.
    let images = req.pixel_values.as_ref().map(PixelValues::to_array);
    let (
        PrefixResume {
            resume_at,
            cache,
            checkpoints: inherited,
        },
        disk_loaded,
    ) = if images.is_some() {
        (
            PrefixResume {
                resume_at: 0,
                cache: fresh_cache()?,
                checkpoints: Vec::new(),
            },
            false,
        )
    } else {
        prompt_cache.resume(&req.prompt_tokens, fresh_cache)?
    };
    let tokens = req
        .prompt_tokens
        .get(resume_at..)
        .unwrap_or_default()
        .to_vec();

    let mut checkpoints = if images.is_some() {
        PrefillCheckpoints::carrying(Vec::new())
    } else {
        prompt_cache.plan_checkpoints(&req.prompt_tokens, resume_at, &cache, inherited)
    };

    // The MTP head is primed from the hidden states of one forward pass over
    // the whole tail, so the prefill must not pause or split.
    let capture_hidden = settings.speculation.prime_mtp_from_prefill
        && settings.prefill_yield_tokens == 0
        && tokens.len() <= settings.chunking.threshold
        && choose_speculation(
            &settings.speculation,
            model.has_mtp(),
            req.speculation_request(),
        ) == Some(SpeculationKind::Mtp);
    if capture_hidden {
        checkpoints.cancel_stops();
    }

    let cached_tokens = u32::try_from(resume_at).unwrap_or(prompt_len);
    announce_prefill(&req, cached_tokens, cached_tokens, prompt_len);

    Ok(PendingPrefill {
        request_id: NEXT_PREFILL_ID.fetch_add(1, Ordering::Relaxed),
        req,
        prompt_len,
        cached_tokens,
        tokens,
        offset: 0,
        cache,
        checkpoints,
        disk_loaded,
        images,
        capture_hidden,
    })
}

#[allow(clippy::too_many_lines)]
fn advance_prefill(
    model: &mut AnyModel,
    prompt_cache: &mut PromptCache,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    settings: &WorkerSettings,
    mut prefill: PendingPrefill,
    quantum: u32,
) -> Result<PrefillAdvance, EngineError> {
    let start = prefill.offset;
    let resumed_at = usize::try_from(prefill.cached_tokens).unwrap_or(usize::MAX);
    let first_position = resumed_at.saturating_add(start);
    let chunk_tokens = usize::try_from(quantum).unwrap_or(usize::MAX);
    // Chunks end on multiples of the quantum, so the checkpoint positions
    // (multiples of a multiple of the chunk size) fall on chunk boundaries and
    // pausing for them adds no forward pass.
    let chunk_end = (first_position / chunk_tokens)
        .saturating_add(1)
        .saturating_mul(chunk_tokens);
    let mut end = chunk_end
        .saturating_sub(resumed_at)
        .min(prefill.tokens.len());
    if prefill.images.is_some() {
        // A vision model needs the whole prompt at once to place the images.
        end = prefill.tokens.len();
    } else if let Some(stop) = prefill.checkpoints.next_stop(first_position) {
        end = end.min(stop - resumed_at);
    }
    let is_complete = end == prefill.tokens.len();

    with_stream(&Stream::new(), || {
        let tokens = prefill
            .tokens
            .get(start..end)
            .ok_or_else(|| EngineError::Generation("Invalid prefill progress".to_owned()))?;
        let input = Array::from(tokens).index(NewAxis);
        let mut hidden = None;
        let logits = if let Some(images) = &prefill.images {
            model
                .forward_multimodal(&input, images, &mut prefill.cache)
                .map_err(EngineError::Mlx)?
        } else if prefill.capture_hidden && is_complete && start == 0 {
            let (states, logits) = model
                .forward_with_hidden(&input, None, &mut prefill.cache)
                .map_err(EngineError::Mlx)?;
            hidden = Some(states);
            logits
        } else {
            model
                .forward(&input, None, &mut prefill.cache)
                .map_err(EngineError::Mlx)?
        };
        let last_logits = logits.index((.., -1, ..));
        prefill.offset = end;

        tracing::debug!(
            request_id = prefill.request_id,
            tokens_advanced = end - start,
            remaining = prefill.tokens.len() - end,
            "Prefill yield"
        );

        if !is_complete {
            eval([&last_logits]).map_err(EngineError::Mlx)?;
            let processed = prefill
                .cached_tokens
                .saturating_add(u32::try_from(end).unwrap_or(u32::MAX));
            announce_prefill(
                &prefill.req,
                processed,
                prefill.cached_tokens,
                prefill.prompt_len,
            );
            let arrival = prefill
                .checkpoints
                .arrive(resumed_at + end, &prefill.cache)
                .map_err(EngineError::Mlx)?;
            if arrival == Arrival::EntryEnd {
                prompt_cache.remember_in_memory(
                    &prefill.req.prompt_tokens,
                    &prefill.cache,
                    &prefill.checkpoints,
                );
            }
            return Ok(PrefillAdvance::InFlight(prefill));
        }

        if start > 0 {
            // A prompt that went through in several chunks ends on 100%, so a
            // progress display does not stall just short of the total.
            announce_prefill(
                &prefill.req,
                prefill.prompt_len,
                prefill.cached_tokens,
                prefill.prompt_len,
            );
        }
        let PendingPrefill {
            req,
            prompt_len,
            tokens: prefilled,
            cache,
            checkpoints,
            disk_loaded,
            images,
            ..
        } = prefill;
        let done = PrefillDone {
            cache,
            checkpoints,
            last_logits,
            hidden,
            prefilled,
            disk_loaded,
        };
        complete_prefill(
            model,
            images.is_none().then_some(&mut *prompt_cache),
            tokenizer,
            eos_token_ids,
            settings,
            req,
            prompt_len,
            done,
        )
        .map(PrefillAdvance::Complete)
    })
}

/// Prefill synchronously when yielding is disabled.
fn prefill_request(
    model: &mut AnyModel,
    prompt_cache: &mut PromptCache,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    settings: &WorkerSettings,
    req: GenerationRequest,
) -> Result<Option<ActiveRequest>, EngineError> {
    let mut prefill = start_prefill(model, prompt_cache, settings, req)?;
    let quantum = settings.chunking.quantum(prefill.tokens.len());
    loop {
        match advance_prefill(
            model,
            prompt_cache,
            tokenizer,
            eos_token_ids,
            settings,
            prefill,
            quantum,
        )? {
            PrefillAdvance::Complete(active) => return Ok(active),
            PrefillAdvance::InFlight(resumed) => prefill = resumed,
        }
    }
}

/// Finish a prefill after its final forward pass has produced logits.
///
/// `prompt_cache` is `None` for a request that must not touch the prefix cache.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn complete_prefill(
    model: &mut AnyModel,
    prompt_cache: Option<&mut PromptCache>,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    settings: &WorkerSettings,
    mut req: GenerationRequest,
    prompt_len: u32,
    mut done: PrefillDone,
) -> Result<Option<ActiveRequest>, EngineError> {
    // The first token is sampled here rather than in a decode round, so
    // the constraint has to be applied and advanced here too.
    let constrained_logits = match req.constraint.as_ref() {
        Some(constraint) => constraint
            .apply_mask(&done.last_logits)
            .map_err(EngineError::Mlx)?,
        None => done.last_logits.clone(),
    };
    let current_token = sample(&constrained_logits, &req.params).map_err(EngineError::Mlx)?;

    let logprob_top_n = req.logprobs.then(|| req.top_logprobs.unwrap_or(0));
    let first_logprob_data = if let Some(top_n) = logprob_top_n {
        let scaled = if req.params.temperature <= f32::EPSILON {
            constrained_logits
        } else {
            constrained_logits
                .multiply(mlx_rs::array!(1.0 / req.params.temperature))
                .map_err(EngineError::Mlx)?
        };
        Some(
            LogprobArrays::compute(&scaled, &current_token, Some(top_n))
                .map_err(EngineError::Mlx)?,
        )
    } else {
        None
    };

    {
        let mut eval_targets: Vec<&Array> = vec![&current_token];
        if let Some(ref lp) = first_logprob_data {
            eval_targets.extend(lp.eval_targets());
        }
        if let Some(ref hidden) = done.hidden {
            eval_targets.push(hidden);
        }
        eval(eval_targets).map_err(EngineError::Mlx)?;
    }

    if let Some(cache_to_update) = prompt_cache {
        cache_to_update.remember(
            &req.prompt_tokens,
            &done.cache,
            &done.checkpoints,
            done.disk_loaded,
        );
    }
    crate::mlx_memory::clear_allocator_cache_if(settings.clear_cache_after_prefill, "post_prefill");

    let first_token_id: u32 = current_token.item_cast();
    if let Some(constraint) = req.constraint.as_mut() {
        constraint.advance(first_token_id);
    }
    if let Some(gate) = req.thinking.as_mut() {
        gate.observe(first_token_id);
    }
    let first_token_logprob = first_logprob_data
        .as_ref()
        .map(|lp| lp.materialize(first_token_id));

    // Decode the first token incrementally (routes through IncrementalDetok
    // so partial UTF-8 is held back and prefix-before-stop is correctly emitted).
    // Preserve content-bearing special tokens (e.g. MiniCPM tool-call markup)
    // while stripping control tokens.
    let skip_ids = std::sync::Arc::new(crate::detokenizer::content_preserving_skip_ids(
        tokenizer,
        eos_token_ids,
    ));
    let mut detok = IncrementalDetok::new(String::new(), 0, skip_ids);
    let first_chunk = detok
        .append(tokenizer, &[first_token_id])
        .unwrap_or_default();
    let emitted_before = detok.text.len() - first_chunk.len();

    // Check if we're done after the first token
    let is_eos = eos_token_ids.contains(&first_token_id);
    let hit_stop = !req.stop_sequences.is_empty()
        && find_stop_in_tail(&detok.text, first_chunk.len(), &req.stop_sequences).is_some();
    let at_max = req.max_tokens <= 1;

    if is_eos || hit_stop || at_max {
        let finish_reason = if is_eos || hit_stop { "stop" } else { "length" };
        let mut send_text = if hit_stop {
            // Emit any prefix text before the stop sequence
            find_stop_in_tail(&detok.text, first_chunk.len(), &req.stop_sequences)
                .and_then(|pos| detok.text.get(emitted_before..pos))
                .unwrap_or_default()
                .to_owned()
        } else {
            first_chunk
        };
        if !hit_stop {
            send_text.push_str(
                &detok
                    .flush(tokenizer, &[first_token_id])
                    .unwrap_or_default(),
            );
        }
        let _ = req.response_tx.blocking_send(StreamingOutput {
            new_text: send_text,
            finished: true,
            finish_reason: Some(finish_reason.to_owned()),
            prompt_tokens: prompt_len,
            completion_tokens: 1,
            token_logprob: first_token_logprob,
            prefill_progress: None,
        });
        return Ok(None);
    }

    // Send first token
    if req
        .response_tx
        .blocking_send(StreamingOutput {
            new_text: first_chunk,
            finished: false,
            finish_reason: None,
            prompt_tokens: prompt_len,
            completion_tokens: 1,
            token_logprob: first_token_logprob,
            prefill_progress: None,
        })
        .is_err()
    {
        return Ok(None); // Client disconnected
    }

    let speculation = choose_speculation(
        &settings.speculation,
        model.has_mtp(),
        req.speculation_request(),
    );
    let mode = if speculation == Some(SpeculationKind::Mtp) {
        DecodeMode::Speculative(Speculation::start_mtp(
            model,
            &mut done.cache,
            &settings.speculation,
            &done.prefilled,
            done.hidden.as_ref(),
            first_token_id,
        )?)
    } else {
        DecodeMode::Normal
    };

    Ok(Some(ActiveRequest {
        id: NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed),
        cache: done.cache,
        current_token,
        mode,
        resumable_speculation: speculation.filter(|kind| *kind == SpeculationKind::PromptLookup),
        in_flight: None,
        generated_tokens: vec![first_token_id],
        max_tokens: req.max_tokens,
        params: req.params,
        stop_sequences: req.stop_sequences,
        logprob_top_n,
        constraint: req.constraint,
        thinking: req.thinking,
        response_tx: req.response_tx,
        prompt_len,
        detok,
    }))
}

/// Lazy arrays from building a decode step's computation graph.
struct DecodeGraphResult {
    next_token: Array,
    logprob_data: Option<LogprobArrays>,
}

/// Build the computation graph for one decode step without evaluating.
/// The caller should `async_eval` the result, then later call
/// `materialize_decode_step` to extract concrete token values.
fn build_decode_graph(
    model: &mut AnyModel,
    ar: &mut ActiveRequest,
) -> Result<DecodeGraphResult, EngineError> {
    let decode_input = ar.current_token.index((.., NewAxis));
    let logits = model
        .forward(&decode_input, None, &mut ar.cache)
        .map_err(EngineError::Mlx)?;
    let sliced = logits.index((.., -1, ..));

    let penalized =
        apply_penalties(&sliced, &ar.generated_tokens, &ar.params).map_err(EngineError::Mlx)?;

    let constrained = if let Some(ref cg) = ar.constraint {
        cg.apply_mask(&penalized).map_err(EngineError::Mlx)?
    } else {
        penalized
    };

    let sampled = sample(&constrained, &ar.params).map_err(EngineError::Mlx)?;
    let next_position = u32::try_from(ar.generated_tokens.len())
        .unwrap_or(u32::MAX)
        .saturating_add(1);
    let next_token = match ar.thinking {
        Some(gate) if gate.forces(next_position) => {
            Array::from_slice(&[gate.close_token], sampled.shape())
                .as_dtype(sampled.dtype())
                .map_err(EngineError::Mlx)?
        }
        _ => sampled,
    };

    let logprob_data = if let Some(top_n) = ar.logprob_top_n {
        let scaled = if ar.params.temperature <= f32::EPSILON {
            constrained
        } else {
            constrained
                .multiply(mlx_rs::array!(1.0 / ar.params.temperature))
                .map_err(EngineError::Mlx)?
        };
        Some(LogprobArrays::compute(&scaled, &next_token, Some(top_n)).map_err(EngineError::Mlx)?)
    } else {
        None
    };

    Ok(DecodeGraphResult {
        next_token,
        logprob_data,
    })
}

/// Materialize a decode step's results after `async_eval` has completed.
/// Returns `true` if the request is finished, `false` to continue.
fn materialize_decode_step(
    ar: &mut ActiveRequest,
    result: DecodeGraphResult,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
) -> bool {
    let token_id: u32 = result.next_token.item_cast();
    let token_logprob = result
        .logprob_data
        .as_ref()
        .map(|lp| lp.materialize(token_id));
    ar.current_token = result.next_token;
    deliver_token(ar, token_id, token_logprob, tokenizer, eos_token_ids)
}

/// Record a generated token, stream its text, and report whether the request
/// is finished (end of sequence, length, stop sequence, or client gone).
fn deliver_token(
    ar: &mut ActiveRequest,
    token_id: u32,
    token_logprob: Option<higgs_models::TokenLogprobInfo>,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
) -> bool {
    if let Some(ref mut cg) = ar.constraint {
        cg.advance(token_id);
    }
    if let Some(gate) = ar.thinking.as_mut() {
        gate.observe(token_id);
    }

    ar.generated_tokens.push(token_id);

    let completion_len: u32 = ar.generated_tokens.len().try_into().unwrap_or(u32::MAX);

    // Decode only the trailing token window for diff and stop checking
    let new_text = ar
        .detok
        .append(tokenizer, &ar.generated_tokens)
        .unwrap_or_default();
    let emitted_before = ar.detok.text.len() - new_text.len();

    let (mut final_new_text, hit_stop) = if ar.stop_sequences.is_empty() {
        (new_text, false)
    } else if let Some(pos) = find_stop_in_tail(&ar.detok.text, new_text.len(), &ar.stop_sequences)
    {
        let emit = ar
            .detok
            .text
            .get(emitted_before..pos)
            .unwrap_or_default()
            .to_owned();
        (emit, true)
    } else {
        (new_text, false)
    };

    let is_eos = eos_token_ids.contains(&token_id);
    let at_max = completion_len >= ar.max_tokens;
    let constraint_done = ar
        .constraint
        .as_ref()
        .is_some_and(crate::constrained::ConstrainedGenerator::is_finished);

    let finished = is_eos || at_max || hit_stop || constraint_done;
    if finished && !hit_stop {
        final_new_text.push_str(
            &ar.detok
                .flush(tokenizer, &ar.generated_tokens)
                .unwrap_or_default(),
        );
    }
    let finish_reason = if is_eos || hit_stop || constraint_done {
        Some("stop".to_owned())
    } else if at_max {
        Some("length".to_owned())
    } else {
        None
    };

    let disconnected = ar
        .response_tx
        .blocking_send(StreamingOutput {
            new_text: final_new_text,
            finished,
            finish_reason,
            prompt_tokens: ar.prompt_len,
            completion_tokens: completion_len,
            token_logprob,
            prefill_progress: None,
        })
        .is_err();

    finished || disconnected
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // materialize_decode_step
    // -----------------------------------------------------------------------

    fn make_active_request(
        max_tokens: u32,
        stop_sequences: Vec<String>,
    ) -> (ActiveRequest, tokio::sync::mpsc::Receiver<StreamingOutput>) {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let ar = ActiveRequest {
            id: 0,
            cache: AnyCache::KV(vec![]),
            current_token: Array::from_slice(&[0_u32], &[1]),
            mode: DecodeMode::Normal,
            resumable_speculation: None,
            in_flight: None,
            generated_tokens: vec![],
            max_tokens,
            params: SamplingParams::default(),
            stop_sequences,
            logprob_top_n: None,
            constraint: None,
            thinking: None,
            response_tx: tx,
            prompt_len: 5,
            detok: IncrementalDetok::new(
                String::new(),
                0,
                std::sync::Arc::new(std::collections::HashSet::new()),
            ),
        };
        (ar, rx)
    }

    fn make_tokenizer() -> Tokenizer {
        Tokenizer::new(tokenizers::models::bpe::BPE::default())
    }

    #[test]
    fn load_returns_model_load_failure_to_caller() {
        // Tokenizer and template load; only the model itself fails.
        let dir = tempfile::tempdir().unwrap();
        make_tokenizer()
            .save(dir.path().join("tokenizer.json"), false)
            .unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "{{ messages }}"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "not-a-real-architecture"}"#,
        )
        .unwrap();

        let result = InferenceEngine::load(
            dir.path(),
            EngineOptions {
                kv_cache: KvCacheConfig::default(),
                tuning: MlxRuntimeTuning::from_model_dir(
                    dir.path(),
                    crate::mlx_tuning::RequestedMlxProfile::Auto,
                ),
                raise_wired_limit: false,
                prefill_yield_tokens: None,
                default_thinking_budget: 256,
                disk_prefix: None,
                prefix_cache: PrefixCacheOptions::default(),
                max_concurrent_requests: None,
            },
        );

        assert!(
            matches!(result, Err(EngineError::Model(_))),
            "a model that fails to load must surface as EngineError::Model from load()"
        );
    }

    #[test]
    fn materialize_decode_step_normal_token_returns_false() {
        let (mut ar, _rx) = make_active_request(100, vec![]);
        let tokenizer = make_tokenizer();
        let result = DecodeGraphResult {
            next_token: Array::from_slice(&[42_u32], &[1]),
            logprob_data: None,
        };
        let finished = materialize_decode_step(&mut ar, result, &tokenizer, &[0]);
        assert!(!finished, "Should not be finished after normal token");
        assert_eq!(ar.generated_tokens, vec![42]);
    }

    #[test]
    fn materialize_decode_step_eos_returns_true() {
        let (mut ar, _rx) = make_active_request(100, vec![]);
        let tokenizer = make_tokenizer();
        let eos_id = 50256_u32; // GPT-2 EOS
        let result = DecodeGraphResult {
            next_token: Array::from_slice(&[eos_id], &[1]),
            logprob_data: None,
        };
        let finished = materialize_decode_step(&mut ar, result, &tokenizer, &[eos_id]);
        assert!(finished, "Should be finished on EOS token");
    }

    #[test]
    fn materialize_decode_step_max_tokens_returns_true() {
        let (mut ar, _rx) = make_active_request(1, vec![]);
        let tokenizer = make_tokenizer();
        let result = DecodeGraphResult {
            next_token: Array::from_slice(&[42_u32], &[1]),
            logprob_data: None,
        };
        let finished = materialize_decode_step(&mut ar, result, &tokenizer, &[0]);
        assert!(finished, "Should be finished at max_tokens=1");
    }

    // -----------------------------------------------------------------------
    // Thinking budget
    // -----------------------------------------------------------------------

    const CLOSE: u32 = 99;

    #[test]
    fn gate_never_forces_the_first_token() {
        let gate = ThinkingGate::new(CLOSE, 1);
        assert!(!gate.forces(1));
        assert!(gate.forces(2));
    }

    #[test]
    fn gate_forces_from_the_budgeted_position_on() {
        let gate = ThinkingGate::new(CLOSE, 20);
        assert!(!gate.forces(19));
        assert!(gate.forces(20));
        assert!(gate.forces(21));
    }

    #[test]
    fn gate_stops_forcing_once_the_block_is_closed() {
        let mut gate = ThinkingGate::new(CLOSE, 5);
        gate.observe(7);
        assert!(gate.forces(5));
        gate.observe(CLOSE);
        assert!(!gate.forces(5));
        assert!(!gate.forces(50));
    }

    fn make_row(gate: Option<ThinkingGate>, generated: usize) -> ActiveRequest {
        let (mut ar, _rx) = make_active_request(100, vec![]);
        ar.thinking = gate;
        ar.generated_tokens = vec![1; generated];
        ar
    }

    fn apply_thinking(thinking: &mut BatchThinking, sampled: &[u32]) -> Vec<u32> {
        let rows = i32::try_from(sampled.len()).unwrap();
        let tokens = thinking
            .apply(&Array::from_slice(sampled, &[rows]))
            .unwrap();
        tokens.eval().unwrap();
        tokens.as_slice::<u32>().to_vec()
    }

    #[test]
    fn batch_thinking_forces_close_for_rows_over_budget_only() {
        let rows = [
            make_row(Some(ThinkingGate::new(CLOSE, 3)), 1),
            make_row(None, 1),
            make_row(Some(ThinkingGate::new(CLOSE, 100)), 1),
        ];
        let mut thinking = BatchThinking::for_rows(&rows).unwrap().unwrap();

        assert_eq!(apply_thinking(&mut thinking, &[10, 11, 12]), [10, 11, 12]);
        assert_eq!(
            apply_thinking(&mut thinking, &[20, 21, 22]),
            [CLOSE, 21, 22]
        );
        assert_eq!(apply_thinking(&mut thinking, &[30, 31, 32]), [30, 31, 32]);
    }

    #[test]
    fn batch_thinking_notices_a_natural_close_without_a_readback() {
        let rows = [make_row(Some(ThinkingGate::new(CLOSE, 3)), 1)];
        let mut thinking = BatchThinking::for_rows(&rows).unwrap().unwrap();

        assert_eq!(apply_thinking(&mut thinking, &[CLOSE]), [CLOSE]);
        assert_eq!(apply_thinking(&mut thinking, &[50]), [50]);
        assert_eq!(apply_thinking(&mut thinking, &[51]), [51]);
    }

    #[test]
    fn batch_thinking_respects_rows_that_already_closed() {
        let mut closed_gate = ThinkingGate::new(CLOSE, 2);
        closed_gate.observe(CLOSE);
        let rows = [make_row(Some(closed_gate), 6)];
        let mut thinking = BatchThinking::for_rows(&rows).unwrap().unwrap();

        assert_eq!(apply_thinking(&mut thinking, &[60]), [60]);
    }

    #[test]
    fn batch_thinking_is_absent_when_no_row_thinks() {
        let rows = [make_row(None, 1), make_row(None, 4)];
        assert!(BatchThinking::for_rows(&rows).unwrap().is_none());
    }

    #[test]
    fn a_lookahead_decode_emits_the_forced_close_exactly_once() {
        let mut gate = ThinkingGate::new(CLOSE, 4);
        gate.observe(1);
        let mut emitted = vec![1];
        let mut forced = None;
        for sampled in [2, 3, 4, 5, 6] {
            let token = forced.take().unwrap_or(sampled);
            emitted.push(token);
            gate.observe(token);
            let next_position = u32::try_from(emitted.len() + 1).unwrap();
            if gate.forces(next_position) {
                forced = Some(CLOSE);
            }
        }
        assert_eq!(emitted.iter().filter(|&&token| token == CLOSE).count(), 1);
        assert_eq!(emitted.iter().position(|&token| token == CLOSE), Some(3));
    }

    #[test]
    fn thinking_room_counts_the_tokens_left_before_the_forced_close() {
        let gate = ThinkingGate::new(CLOSE, 5);
        assert_eq!(gate.room(2), Some(3));
        assert_eq!(gate.room(4), Some(1));
        assert_eq!(gate.room(5), Some(0));
        assert_eq!(gate.room(9), Some(0));
        assert_eq!(ThinkingGate::new(CLOSE, 1).room(2), Some(0));
    }

    #[test]
    fn thinking_room_is_unbounded_once_the_block_closes() {
        let mut gate = ThinkingGate::new(CLOSE, 5);
        gate.observe(CLOSE);
        assert_eq!(gate.room(5), None);
    }

    // -----------------------------------------------------------------------
    // Speculation handover
    // -----------------------------------------------------------------------

    fn speculating_request(
        max_tokens: u32,
        gate: Option<ThinkingGate>,
        generated: usize,
        confirmed: u32,
    ) -> (ActiveRequest, tokio::sync::mpsc::Receiver<StreamingOutput>) {
        let (mut ar, rx) = make_active_request(max_tokens, vec![]);
        ar.thinking = gate;
        ar.generated_tokens = vec![1; generated];
        ar.mode = DecodeMode::Speculative(Speculation::prompt_lookup_confirming(confirmed));
        (ar, rx)
    }

    #[test]
    fn handover_reports_the_confirmed_token_and_leaves_the_normal_path() {
        let (mut ar, mut rx) = speculating_request(100, None, 2, 42);
        let finished = hand_over_speculation(&mut ar, &make_tokenizer(), &[0]);

        assert!(!finished);
        assert!(matches!(ar.mode, DecodeMode::Normal));
        assert_eq!(ar.generated_tokens, vec![1, 1, 42]);
        assert_eq!(
            ar.resumable_speculation,
            Some(SpeculationKind::PromptLookup)
        );
        ar.current_token.eval().unwrap();
        assert_eq!(ar.current_token.as_slice::<u32>(), [42]);
        assert_eq!(rx.try_recv().unwrap().completion_tokens, 3);
    }

    #[test]
    fn handover_reports_close_instead_when_the_reasoning_budget_is_spent() {
        let gate = ThinkingGate::new(CLOSE, 3);
        let (mut ar, _rx) = speculating_request(100, Some(gate), 2, 42);
        hand_over_speculation(&mut ar, &make_tokenizer(), &[0]);

        assert_eq!(ar.generated_tokens, vec![1, 1, CLOSE]);
        assert!(ar.thinking.is_some_and(|remaining| remaining.closed));
    }

    #[test]
    fn handover_can_finish_the_request() {
        let (mut ar, mut rx) = speculating_request(3, None, 2, 42);
        assert!(hand_over_speculation(&mut ar, &make_tokenizer(), &[0]));
        let output = rx.try_recv().unwrap();
        assert!(output.finished);
        assert_eq!(output.finish_reason.as_deref(), Some("length"));
    }

    #[test]
    fn handover_ignores_a_request_that_is_not_speculating() {
        let (mut ar, mut rx) = make_active_request(100, vec![]);
        assert!(!hand_over_speculation(&mut ar, &make_tokenizer(), &[0]));
        assert!(ar.generated_tokens.is_empty());
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn a_request_reports_its_shape_for_scheduling() {
        let (ar, _rx) = speculating_request(100, None, 1, 7);
        let shape = ar.shape();
        assert!(shape.speculating);
        assert!(!shape.constrained && !shape.logprobs);
    }

    #[test]
    fn settling_a_lookahead_reports_its_token() {
        let (mut ar, mut rx) = make_active_request(100, vec![]);
        ar.in_flight = Some(Array::from_slice(&[9_u32], &[1]));
        assert!(!settle_in_flight(&mut ar, &make_tokenizer(), &[0]));
        assert_eq!(ar.generated_tokens, vec![9]);
        assert!(ar.in_flight.is_none());
        assert_eq!(rx.try_recv().unwrap().completion_tokens, 1);
        assert!(!settle_in_flight(&mut ar, &make_tokenizer(), &[0]));
    }

    // -----------------------------------------------------------------------
    // Prompt cache
    // -----------------------------------------------------------------------

    fn dense_cache(tokens: i32) -> AnyCache {
        let keys = Array::zeros::<f32>(&[1, 2, tokens, 8]).unwrap();
        let values = Array::zeros::<f32>(&[1, 2, tokens, 8]).unwrap();
        AnyCache::KV(vec![Some(
            higgs_models::cache::SteppingKeyValueCache::from_arrays(keys, values).unwrap(),
        )])
    }

    #[test]
    fn prompt_cache_ignores_the_generation_suffix_in_its_key() {
        let mut prompt_cache = PromptCache::new(4, 3);
        let prompt: Vec<u32> = (0..99).collect();
        prompt_cache.remember(
            &prompt,
            &dense_cache(99),
            &PrefillCheckpoints::default(),
            false,
        );

        let mut repeat = prompt;
        repeat.truncate(96);
        repeat.extend([900, 901, 902]);
        assert_eq!(prompt_cache.covered_tokens(&repeat), 96);

        let (resumed, from_disk) = prompt_cache
            .resume(&repeat, || {
                Err(EngineError::Generation("expected a hit".to_owned()))
            })
            .unwrap();
        assert_eq!(resumed.resume_at, 96);
        assert!(!from_disk);
    }

    #[test]
    fn prompt_cache_reuses_the_start_of_a_diverging_prompt() {
        let mut prompt_cache = PromptCache::new(4, 3);
        let stored: Vec<u32> = (0..131).collect();
        prompt_cache.remember(
            &stored,
            &dense_cache(131),
            &PrefillCheckpoints::default(),
            false,
        );

        let mut diverging: Vec<u32> = (0..100).collect();
        diverging.extend(500..540);
        assert_eq!(prompt_cache.covered_tokens(&diverging), 96);
    }

    #[test]
    fn hybrid_prefill_pauses_at_the_cache_key() {
        let prompt_cache = PromptCache::new(4, 4);
        let prompt: Vec<u32> = (0..50).collect();
        let hybrid = AnyCache::Hybrid(Vec::new());
        let stop = |resume_at| {
            prompt_cache
                .plan_checkpoints(&prompt, resume_at, &hybrid, Vec::new())
                .next_stop(resume_at)
        };

        assert_eq!(stop(0), Some(46));
        assert_eq!(stop(10), Some(46));
        assert_eq!(stop(46), None);
        assert_eq!(
            prompt_cache
                .plan_checkpoints(&prompt, 0, &dense_cache(0), Vec::new())
                .next_stop(0),
            None
        );
    }

    #[test]
    fn hybrid_prefill_needs_no_pause_without_a_generation_suffix() {
        let prompt_cache = PromptCache::new(4, 0);
        let prompt: Vec<u32> = (0..50).collect();
        assert_eq!(
            prompt_cache
                .plan_checkpoints(&prompt, 0, &AnyCache::Hybrid(Vec::new()), Vec::new())
                .next_stop(0),
            None
        );
    }

    #[test]
    fn chunking_splits_only_prompts_past_the_threshold() {
        let chunking = PrefillChunking {
            threshold: 2048,
            chunk_tokens: 1024,
        };
        assert_eq!(chunking.quantum(100), u32::MAX);
        assert_eq!(chunking.quantum(2048), u32::MAX);
        assert_eq!(chunking.quantum(2049), 1024);
    }
}
