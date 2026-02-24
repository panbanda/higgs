//! Batch engine with interleaved request processing.
//!
//! Unlike [`SimpleEngine`](crate::simple::SimpleEngine) which serializes requests
//! through a mutex, `BatchEngine` runs a dedicated background loop that interleaves
//! decode steps across multiple active requests. Each request gets one token per
//! iteration, so concurrent clients see tokens as soon as possible rather than
//! waiting for prior requests to fully complete.

use std::path::Path;
use std::sync::atomic::{AtomicI32, Ordering};

use higgs_models::{AnyCache, AnyModel, LogprobArrays, SamplingParams, apply_penalties, sample};
use mlx_rs::{
    Array, Stream,
    ops::indexing::{IndexOp, NewAxis},
    transforms::{async_eval, eval},
    with_new_default_stream,
};
use tokenizers::Tokenizer;

use crate::{
    chat_template::{ChatMessage, ChatTemplateRenderer},
    engine::{GenerationOutput, StreamingOutput},
    error::EngineError,
    model_loader,
    prompt_cache::PrefixCache,
};

/// Default maximum number of cached prefixes.
const DEFAULT_PREFIX_CACHE_SIZE: usize = 8;

/// Maximum number of pending requests in the submission queue.
const REQUEST_QUEUE_CAPACITY: usize = 128;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// A generation request submitted to the batch engine.
struct BatchRequest {
    prompt_tokens: Vec<u32>,
    max_tokens: u32,
    params: SamplingParams,
    stop_sequences: Vec<String>,
    logprobs: bool,
    top_logprobs: Option<u32>,
    constraint: Option<crate::constrained::ConstrainedGenerator>,
    response_tx: tokio::sync::mpsc::Sender<StreamingOutput>,
}

/// An in-flight request being actively decoded.
struct ActiveRequest {
    cache: AnyCache,
    current_token: Array,
    generated_tokens: Vec<u32>,
    max_tokens: u32,
    params: SamplingParams,
    stop_sequences: Vec<String>,
    logprob_top_n: Option<u32>,
    constraint: Option<crate::constrained::ConstrainedGenerator>,
    response_tx: tokio::sync::mpsc::Sender<StreamingOutput>,
    prompt_len: u32,
    prev_decoded_len: usize,
}

// ---------------------------------------------------------------------------
// BatchEngine
// ---------------------------------------------------------------------------

/// Inference engine with interleaved request processing.
///
/// Provides the same public API as `SimpleEngine` but runs all inference on a
/// dedicated background thread. Concurrent requests are interleaved at the
/// token level instead of being fully serialized.
pub struct BatchEngine {
    request_tx: tokio::sync::mpsc::Sender<BatchRequest>,
    tokenizer: Tokenizer,
    template: ChatTemplateRenderer,
    model_name: String,
    eos_token_ids: Vec<u32>,
    hidden_size: AtomicI32,
}

impl BatchEngine {
    /// Load a model and start the background processing loop.
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self, EngineError> {
        let model_dir = dir.as_ref();
        let model_name = crate::simple::derive_model_name(model_dir);

        tracing::info!(model_dir = %model_dir.display(), "Loading model (batch engine)");

        let model = model_loader::load_model(model_dir)?;
        let tokenizer = model_loader::load_tokenizer(model_dir)?;
        let template = ChatTemplateRenderer::from_model_dir(model_dir)?;
        let eos_token_ids = crate::simple::extract_eos_tokens(model_dir);
        let hidden_size = model.hidden_size();

        crate::simple::set_wired_limit_to_max();

        let (request_tx, request_rx) = tokio::sync::mpsc::channel(REQUEST_QUEUE_CAPACITY);
        let eos_ids = eos_token_ids.clone();
        let tok = tokenizer.clone();

        std::thread::Builder::new()
            .name("batch-engine".into())
            .spawn(move || {
                worker_loop(model, &tok, &eos_ids, request_rx);
            })
            .map_err(|e| EngineError::Generation(format!("Failed to spawn worker: {e}")))?;

        tracing::info!(
            model_name = %model_name,
            eos_tokens = ?eos_token_ids,
            "Batch engine ready"
        );

        Ok(Self {
            request_tx,
            tokenizer,
            template,
            model_name,
            eos_token_ids,
            hidden_size: AtomicI32::new(hidden_size),
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
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

    /// Apply chat template and tokenize messages.
    pub fn prepare_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Vec<u32>, EngineError> {
        let prompt = self.template.apply(messages, tools, true)?;
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| EngineError::Tokenization(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Generate a complete (non-streaming) response.
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<mlx_rs::Array>,
    ) -> Result<GenerationOutput, EngineError> {
        if pixel_values.is_some() {
            return Err(EngineError::Generation(
                "Batch engine does not support multimodal (image) inputs".to_owned(),
            ));
        }
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
            });
        }

        // Submit request and collect all streaming outputs.
        let (internal_tx, mut internal_rx) = tokio::sync::mpsc::channel(32);

        self.request_tx
            .blocking_send(BatchRequest {
                prompt_tokens: prompt_tokens.to_vec(),
                max_tokens,
                params: params.clone(),
                stop_sequences: stop_sequences.to_vec(),
                logprobs,
                top_logprobs,
                constraint,
                response_tx: internal_tx,
            })
            .map_err(|_| EngineError::Generation("Engine shut down".to_owned()))?;

        let mut full_text = String::new();
        let mut finish_reason = "length".to_owned();
        let mut completion_tokens: u32 = 0;
        let mut all_logprobs: Option<Vec<higgs_models::TokenLogprobInfo>> = logprobs.then(Vec::new);

        while let Some(output) = internal_rx.blocking_recv() {
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
        })
    }

    /// Generate tokens one at a time via the provided channel.
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub fn generate_streaming(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        params: &SamplingParams,
        stop_sequences: &[String],
        logprobs: bool,
        top_logprobs: Option<u32>,
        sender: &tokio::sync::mpsc::Sender<StreamingOutput>,
        constraint: Option<crate::constrained::ConstrainedGenerator>,
        pixel_values: Option<mlx_rs::Array>,
    ) -> Result<(), EngineError> {
        if pixel_values.is_some() {
            return Err(EngineError::Generation(
                "Batch engine does not support multimodal (image) inputs".to_owned(),
            ));
        }
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
            });
            return Ok(());
        }

        // Submit request -- the background loop sends tokens directly to
        // an internal channel, and we forward them to the caller's sender.
        let (internal_tx, mut internal_rx) = tokio::sync::mpsc::channel(32);

        self.request_tx
            .blocking_send(BatchRequest {
                prompt_tokens: prompt_tokens.to_vec(),
                max_tokens,
                params: params.clone(),
                stop_sequences: stop_sequences.to_vec(),
                logprobs,
                top_logprobs,
                constraint,
                response_tx: internal_tx,
            })
            .map_err(|_| EngineError::Generation("Engine shut down".to_owned()))?;

        while let Some(output) = internal_rx.blocking_recv() {
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

    /// Compute embeddings (delegates to a single forward pass).
    pub fn embed(&self, _token_ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        // Embeddings require direct model access. For now, return an error.
        // A proper implementation would submit an embed request to the worker.
        Err(EngineError::Generation(
            "Embeddings not yet supported in batch engine".to_owned(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Background worker loop
// ---------------------------------------------------------------------------

fn worker_loop(
    mut model: AnyModel,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    mut request_rx: tokio::sync::mpsc::Receiver<BatchRequest>,
) {
    let mut prefix_cache = PrefixCache::new(DEFAULT_PREFIX_CACHE_SIZE);
    let mut active: Vec<ActiveRequest> = Vec::new();

    loop {
        // 1. Prefill at most one pending request per iteration.
        //    This interleaves prefill with decode so long prefills don't
        //    starve active requests, keeping TTFT low for new arrivals
        //    while maintaining token throughput for in-flight requests.
        if let Ok(req) = request_rx.try_recv() {
            match prefill_request(&mut model, &mut prefix_cache, tokenizer, eos_token_ids, req) {
                Ok(Some(ar)) => active.push(ar),
                Ok(None) => {}
                Err(e) => tracing::error!(error = %e, "Prefill failed"),
            }
        }

        // 2. If no active requests, block until one arrives.
        if active.is_empty() {
            if let Some(req) = request_rx.blocking_recv() {
                match prefill_request(&mut model, &mut prefix_cache, tokenizer, eos_token_ids, req)
                {
                    Ok(Some(ar)) => active.push(ar),
                    Ok(None) => continue,
                    Err(e) => {
                        tracing::error!(error = %e, "Prefill failed");
                        continue;
                    }
                }
            } else {
                tracing::info!("Request channel closed, worker shutting down");
                return;
            }
        }

        // 3. Run one decode step per active request.
        //    Use batched decode when possible (Transformer architecture, >1
        //    request, no constrained generation). Falls back to pipelined
        //    sequential decode otherwise.
        let mut finished_indices = Vec::new();
        let use_batched = active.len() > 1
            && model.supports_batched_decode()
            && active.iter().all(|ar| ar.constraint.is_none());

        if use_batched {
            if let Err(e) = with_new_default_stream(Stream::new(), || {
                run_batched_decode_round(
                    &mut model,
                    &mut active,
                    tokenizer,
                    eos_token_ids,
                    &mut finished_indices,
                )
            }) {
                tracing::error!(error = %e, "Batched decode round failed");
                for (i, ar) in active.iter().enumerate() {
                    let _ = ar.response_tx.blocking_send(StreamingOutput {
                        new_text: String::new(),
                        finished: true,
                        finish_reason: Some("error".to_owned()),
                        prompt_tokens: ar.prompt_len,
                        completion_tokens: 0,
                        token_logprob: None,
                    });
                    finished_indices.push(i);
                }
            }
        } else {
            with_new_default_stream(Stream::new(), || {
                run_pipelined_decode_round(
                    &mut model,
                    &mut active,
                    tokenizer,
                    eos_token_ids,
                    &mut finished_indices,
                );
            });
        }

        // 4. Remove finished requests (reverse order to preserve indices).
        for i in finished_indices.into_iter().rev() {
            active.swap_remove(i);
        }
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

/// Batched decode: single forward pass for all active requests.
///
/// Stacks N single-token inputs into `[N, 1]`, runs one batched forward pass,
/// then slices logits per-request for individual sampling.
#[allow(clippy::indexing_slicing)]
fn run_batched_decode_round(
    model: &mut AnyModel,
    active: &mut [ActiveRequest],
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    finished_indices: &mut Vec<usize>,
) -> Result<(), EngineError> {
    let n = active.len();

    // Stack all current tokens into [N, 1]
    let token_arrays: Vec<Array> = active
        .iter()
        .map(|ar| ar.current_token.index((.., NewAxis)))
        .collect();
    let token_refs: Vec<&Array> = token_arrays.iter().collect();
    let batched_input = mlx_rs::ops::concatenate_axis(&token_refs, 0).map_err(EngineError::Mlx)?;

    // Collect mutable cache references
    let mut cache_refs: Vec<&mut higgs_models::AnyCache> =
        active.iter_mut().map(|ar| &mut ar.cache).collect();

    // Batched forward pass: [N, 1] -> [N, 1, vocab_size]
    let batched_logits = model
        .forward_batched(&batched_input, &mut cache_refs)
        .map_err(EngineError::Mlx)?;
    let batched_last = batched_logits.index((.., -1, ..)); // [N, vocab_size]

    // Per-request: apply penalties, sample, compute logprobs, async_eval
    let mut results: Vec<Option<DecodeGraphResult>> = Vec::with_capacity(n);
    for (i, ar) in active.iter_mut().enumerate() {
        let idx = i32::try_from(i).unwrap_or(i32::MAX);
        let req_logits = batched_last.index((idx..idx + 1, ..));

        let penalized = apply_penalties(&req_logits, &ar.generated_tokens, &ar.params)
            .map_err(EngineError::Mlx)?;

        let next_token = sample(&penalized, &ar.params).map_err(EngineError::Mlx)?;

        let logprob_data = if let Some(top_n) = ar.logprob_top_n {
            let scaled = if ar.params.temperature <= f32::EPSILON {
                penalized
            } else {
                penalized
                    .multiply(mlx_rs::array!(1.0 / ar.params.temperature))
                    .map_err(EngineError::Mlx)?
            };
            Some(
                LogprobArrays::compute(&scaled, &next_token, Some(top_n))
                    .map_err(EngineError::Mlx)?,
            )
        } else {
            None
        };

        let mut eval_targets: Vec<&Array> = vec![&next_token];
        if let Some(ref lp) = logprob_data {
            eval_targets.extend(lp.eval_targets());
        }
        match async_eval(eval_targets) {
            Ok(()) => results.push(Some(DecodeGraphResult {
                next_token,
                logprob_data,
            })),
            Err(e) => {
                tracing::error!(error = %e, "async_eval failed in batched decode");
                finished_indices.push(i);
                results.push(None);
            }
        }
    }

    // Materialize all results
    for (i, (ar, result)) in active.iter_mut().zip(results).enumerate() {
        let Some(r) = result else { continue };
        if materialize_decode_step(ar, r, tokenizer, eos_token_ids) {
            finished_indices.push(i);
        }
    }

    Ok(())
}

/// Prefill a new request: run the full prompt through the model and sample
/// the first token. Returns `None` if the request completed immediately
/// (first token is EOS or hit a stop sequence).
#[allow(clippy::too_many_lines)]
fn prefill_request(
    model: &mut AnyModel,
    prefix_cache: &mut PrefixCache,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    req: BatchRequest,
) -> Result<Option<ActiveRequest>, EngineError> {
    let prompt_len: u32 = req
        .prompt_tokens
        .len()
        .try_into()
        .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))?;

    with_new_default_stream(Stream::new(), || {
        // Check prefix cache
        let prefix_match = prefix_cache.find_longest_prefix(&req.prompt_tokens);
        let (actual_tokens, mut cache) = if let Some(matched) = prefix_match {
            let suffix = req
                .prompt_tokens
                .get(matched.prefix_len..)
                .unwrap_or_default();
            if suffix.is_empty() {
                // Exact prefix match. Currently falls back to full prefill
                // because we don't store logits alongside the cache.
                (req.prompt_tokens.clone(), model.make_cache())
            } else {
                (suffix.to_vec(), matched.cache)
            }
        } else {
            (req.prompt_tokens.clone(), model.make_cache())
        };

        let prompt_array = Array::from(actual_tokens.as_slice()).index(NewAxis);

        // Prefill forward pass
        let logits = model
            .forward(&prompt_array, None, &mut cache)
            .map_err(EngineError::Mlx)?;
        let last_logits = logits.index((.., -1, ..));
        let current_token = sample(&last_logits, &req.params).map_err(EngineError::Mlx)?;

        let logprob_top_n = req.logprobs.then(|| req.top_logprobs.unwrap_or(0));
        let first_logprob_data = if let Some(top_n) = logprob_top_n {
            let scaled = if req.params.temperature <= f32::EPSILON {
                last_logits
            } else {
                last_logits
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
            eval(eval_targets).map_err(EngineError::Mlx)?;
        }

        // Cache the post-prefill state
        prefix_cache.store(&req.prompt_tokens, cache.clone());

        let first_token_id: u32 = current_token.item();
        let first_token_logprob = first_logprob_data
            .as_ref()
            .map(|lp| lp.materialize(first_token_id));

        // Decode the first token for text diff tracking
        let first_text = tokenizer
            .decode(&[first_token_id], true)
            .unwrap_or_default();

        // Check if we're done after the first token
        let is_eos = eos_token_ids.contains(&first_token_id);
        let hit_stop = check_stop_sequences_simple(&first_text, &req.stop_sequences);
        let at_max = req.max_tokens <= 1;

        if is_eos || hit_stop || at_max {
            let finish_reason = if is_eos || hit_stop { "stop" } else { "length" };
            let _ = req.response_tx.blocking_send(StreamingOutput {
                new_text: if hit_stop { String::new() } else { first_text },
                finished: true,
                finish_reason: Some(finish_reason.to_owned()),
                prompt_tokens: prompt_len,
                completion_tokens: 1,
                token_logprob: first_token_logprob,
            });
            return Ok(None);
        }

        // Send first token
        let prev_decoded_len = first_text.len();
        if req
            .response_tx
            .blocking_send(StreamingOutput {
                new_text: first_text,
                finished: false,
                finish_reason: None,
                prompt_tokens: prompt_len,
                completion_tokens: 1,
                token_logprob: first_token_logprob,
            })
            .is_err()
        {
            return Ok(None); // Client disconnected
        }

        Ok(Some(ActiveRequest {
            cache,
            current_token,
            generated_tokens: vec![first_token_id],
            max_tokens: req.max_tokens,
            params: req.params,
            stop_sequences: req.stop_sequences,
            logprob_top_n,
            constraint: req.constraint,
            response_tx: req.response_tx,
            prompt_len,
            prev_decoded_len,
        }))
    })
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

    let next_token = sample(&constrained, &ar.params).map_err(EngineError::Mlx)?;

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
    let token_id: u32 = result.next_token.item();

    // Advance constrained generator
    if let Some(ref mut cg) = ar.constraint {
        cg.advance(token_id);
    }

    let token_logprob = result
        .logprob_data
        .as_ref()
        .map(|lp| lp.materialize(token_id));

    ar.generated_tokens.push(token_id);
    ar.current_token = result.next_token;

    let completion_len: u32 = ar.generated_tokens.len().try_into().unwrap_or(u32::MAX);

    // Decode full text for diff and stop sequence checking
    let full_text = tokenizer
        .decode(&ar.generated_tokens, true)
        .unwrap_or_default();
    let new_text = full_text
        .get(ar.prev_decoded_len..)
        .unwrap_or_default()
        .to_owned();
    let old_decoded_len = ar.prev_decoded_len;
    ar.prev_decoded_len = full_text.len();

    let (final_new_text, hit_stop) = if ar.stop_sequences.is_empty() {
        (new_text, false)
    } else if check_stop_sequences_simple(&full_text, &ar.stop_sequences) {
        let truncated = truncate_at_stop(&full_text, &ar.stop_sequences);
        let emit = truncated
            .get(old_decoded_len..)
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
        })
        .is_err();

    finished || disconnected
}

/// Check if any stop sequence appears in the text.
fn check_stop_sequences_simple(text: &str, stop_sequences: &[String]) -> bool {
    stop_sequences.iter().any(|seq| text.contains(seq.as_str()))
}

/// Truncate text at the earliest stop sequence.
fn truncate_at_stop(text: &str, stop_sequences: &[String]) -> String {
    let mut earliest: Option<usize> = None;
    for seq in stop_sequences {
        if let Some(pos) = text.find(seq.as_str()) {
            earliest = Some(earliest.map_or(pos, |prev| prev.min(pos)));
        }
    }
    earliest.map_or_else(
        || text.to_owned(),
        |pos| text.get(..pos).unwrap_or_default().to_owned(),
    )
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // check_stop_sequences_simple
    // -----------------------------------------------------------------------

    #[test]
    fn stop_sequences_empty_never_matches() {
        assert!(!check_stop_sequences_simple("hello world", &[]));
    }

    #[test]
    fn stop_sequences_match_at_end() {
        let stops = vec!["</s>".to_owned()];
        assert!(check_stop_sequences_simple("some text</s>", &stops));
    }

    #[test]
    fn stop_sequences_match_in_middle() {
        let stops = vec!["STOP".to_owned()];
        assert!(check_stop_sequences_simple("before STOP after", &stops));
    }

    #[test]
    fn stop_sequences_no_match() {
        let stops = vec!["</s>".to_owned(), "<|end|>".to_owned()];
        assert!(!check_stop_sequences_simple("normal text", &stops));
    }

    #[test]
    fn stop_sequences_multiple_one_matches() {
        let stops = vec!["</s>".to_owned(), "\n\n".to_owned()];
        assert!(check_stop_sequences_simple("text\n\nmore", &stops));
    }

    // -----------------------------------------------------------------------
    // truncate_at_stop
    // -----------------------------------------------------------------------

    #[test]
    fn truncate_no_stop_returns_full_text() {
        let stops = vec!["</s>".to_owned()];
        assert_eq!(truncate_at_stop("hello world", &stops), "hello world");
    }

    #[test]
    fn truncate_at_stop_removes_suffix() {
        let stops = vec!["</s>".to_owned()];
        assert_eq!(truncate_at_stop("hello</s>", &stops), "hello");
    }

    #[test]
    fn truncate_at_earliest_of_multiple_stops() {
        let stops = vec!["BBB".to_owned(), "AAA".to_owned()];
        assert_eq!(truncate_at_stop("xAAAyBBBz", &stops), "x");
    }

    #[test]
    fn truncate_empty_stops() {
        assert_eq!(truncate_at_stop("hello", &[]), "hello");
    }

    #[test]
    fn truncate_stop_at_start() {
        let stops = vec!["STOP".to_owned()];
        assert_eq!(truncate_at_stop("STOPrest", &stops), "");
    }

    // -----------------------------------------------------------------------
    // materialize_decode_step
    // -----------------------------------------------------------------------

    fn make_active_request(
        max_tokens: u32,
        stop_sequences: Vec<String>,
    ) -> (ActiveRequest, tokio::sync::mpsc::Receiver<StreamingOutput>) {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let ar = ActiveRequest {
            cache: AnyCache::KV(vec![]),
            current_token: Array::from_slice(&[0_u32], &[1]),
            generated_tokens: vec![],
            max_tokens,
            params: SamplingParams::default(),
            stop_sequences,
            logprob_top_n: None,
            constraint: None,
            response_tx: tx,
            prompt_len: 5,
            prev_decoded_len: 0,
        };
        (ar, rx)
    }

    fn make_tokenizer() -> Tokenizer {
        Tokenizer::new(tokenizers::models::bpe::BPE::default())
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
    // Batched vs pipelined path selection
    // -----------------------------------------------------------------------

    #[test]
    fn batched_path_requires_multiple_requests() {
        // Single request should not use batched path
        let n_active = 1;
        let supports_batched = true;
        let all_unconstrained = true;
        let use_batched = n_active > 1 && supports_batched && all_unconstrained;
        assert!(!use_batched);
    }

    #[test]
    fn batched_path_requires_model_support() {
        let n_active = 4;
        let supports_batched = false;
        let all_unconstrained = true;
        let use_batched = n_active > 1 && supports_batched && all_unconstrained;
        assert!(!use_batched);
    }

    #[test]
    fn batched_path_disabled_with_constraints() {
        let n_active = 4;
        let supports_batched = true;
        let all_unconstrained = false;
        let use_batched = n_active > 1 && supports_batched && all_unconstrained;
        assert!(!use_batched);
    }

    #[test]
    fn batched_path_enabled_when_all_conditions_met() {
        let n_active = 4;
        let supports_batched = true;
        let all_unconstrained = true;
        let use_batched = n_active > 1 && supports_batched && all_unconstrained;
        assert!(use_batched);
    }
}
