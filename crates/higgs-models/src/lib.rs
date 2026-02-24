pub mod cache;
pub mod deepseek_v2;
pub mod error;
pub mod gemma2;
pub mod llava_qwen2;
pub mod phi3;
pub mod qwen3_moe;
pub mod qwen3_next;
pub mod registry;
pub mod siglip;
pub mod starcoder2;
pub mod transformer;
pub mod utils;

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::module::ModuleParametersExt;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, argmax_axis, array, categorical, error::Exception};
use serde::Deserialize;
use serde_json::Value;

use crate::error::ModelError;

// ---------------------------------------------------------------------------
// SamplingParams -- configurable sampling parameters
// ---------------------------------------------------------------------------

/// Parameters controlling token sampling behavior.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }
}

impl SamplingParams {
    /// Whether any penalty parameters are active.
    #[allow(clippy::float_cmp)]
    pub fn has_penalties(&self) -> bool {
        self.repetition_penalty.is_some_and(|p| p != 1.0)
            || self.frequency_penalty.is_some_and(|p| p != 0.0)
            || self.presence_penalty.is_some_and(|p| p != 0.0)
    }

    /// Whether any `top_k`/`min_p`/`top_p` filtering is needed beyond basic categorical.
    fn needs_filtering(&self) -> bool {
        self.top_k.is_some() || self.min_p.is_some() || self.top_p < 1.0
    }
}

pub use qwen3_next::{LayerCache, Qwen3NextCausalLM};
pub use transformer::{Model, ModelArgs};

// ---------------------------------------------------------------------------
// AnyModel / AnyCache -- unified dispatch across model architectures
// ---------------------------------------------------------------------------

/// Cache type that works for all supported model architectures.
#[derive(Debug, Clone)]
pub enum AnyCache {
    /// Standard KV cache for transformer models (Qwen2/Llama/Mistral).
    KV(Vec<Option<cache::SteppingKeyValueCache>>),
    /// Hybrid KV+SSM cache for `qwen3_next`.
    Hybrid(Vec<Option<LayerCache>>),
}

/// Unified model wrapper dispatching to the correct architecture.
pub enum AnyModel {
    /// Standard transformer architectures: Llama, Mistral, Qwen2/2.5, Qwen3.
    Transformer(Model),
    /// Qwen3-Next hybrid SSM/attention architecture with mixture-of-experts.
    Qwen3Next(Qwen3NextCausalLM),
    /// Qwen3-MoE sparse Mixture-of-Experts architecture.
    Qwen3Moe(qwen3_moe::Qwen3MoeCausalLM),
    /// Gemma 2 architecture with soft-capping and alternating sliding window.
    Gemma2(gemma2::Gemma2CausalLM),
    /// Phi-3 architecture with combined QKV and gate-up projections.
    Phi3(phi3::Phi3CausalLM),
    /// Starcoder2 architecture with `LayerNorm` and sliding window.
    Starcoder2(starcoder2::Starcoder2CausalLM),
    /// LLaVA-Qwen2 vision-language model (nanoLLaVA architecture).
    LlavaQwen2(llava_qwen2::LlavaQwen2Model),
    /// DeepSeek-V2 with Multi-head Latent Attention and sparse `MoE`.
    DeepSeekV2(deepseek_v2::DeepSeekV2CausalLM),
}

fn make_kv_cache(num_hidden_layers: i32) -> AnyCache {
    let Ok(n_layers) = usize::try_from(num_hidden_layers) else {
        tracing::warn!(
            num_hidden_layers,
            "negative num_hidden_layers; returning empty KV cache"
        );
        return AnyCache::KV(vec![]);
    };
    AnyCache::KV(
        (0..n_layers)
            .map(|_| Some(cache::SteppingKeyValueCache::new()))
            .collect(),
    )
}

impl AnyModel {
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut AnyCache,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::Transformer(m), AnyCache::KV(c)) => m.forward(inputs, mask, c),
            (Self::Qwen3Moe(m), AnyCache::KV(c)) => m.forward(inputs, mask, c),
            (Self::Gemma2(m), AnyCache::KV(c)) => m.forward(inputs, mask, c),
            (Self::Phi3(m), AnyCache::KV(c)) => m.forward(inputs, mask, c),
            (Self::Starcoder2(m), AnyCache::KV(c)) => m.forward(inputs, mask, c),
            (Self::LlavaQwen2(m), AnyCache::KV(c)) => m.forward_text(inputs, mask, c),
            (Self::DeepSeekV2(m), AnyCache::KV(c)) => m.forward(inputs, mask, c),
            (Self::Qwen3Next(m), AnyCache::Hybrid(c)) => m.forward(inputs, mask, c),
            _ => Err(Exception::custom("Model/cache type mismatch")),
        }
    }

    /// Forward pass returning hidden states before the LM head.
    pub fn forward_hidden(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut AnyCache,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::Transformer(m), AnyCache::KV(c)) => m.forward_hidden(inputs, mask, c),
            (Self::Qwen3Moe(m), AnyCache::KV(c)) => m.forward_hidden(inputs, mask, c),
            (Self::Gemma2(m), AnyCache::KV(c)) => m.forward_hidden(inputs, mask, c),
            (Self::Phi3(m), AnyCache::KV(c)) => m.forward_hidden(inputs, mask, c),
            (Self::Starcoder2(m), AnyCache::KV(c)) => m.forward_hidden(inputs, mask, c),
            (Self::LlavaQwen2(m), AnyCache::KV(c)) => m.forward_text_hidden(inputs, mask, c),
            (Self::DeepSeekV2(m), AnyCache::KV(c)) => m.forward_hidden(inputs, mask, c),
            (Self::Qwen3Next(m), AnyCache::Hybrid(c)) => m.forward_hidden(inputs, mask, c),
            _ => Err(Exception::custom("Model/cache type mismatch")),
        }
    }

    /// Batched decode forward pass for N requests each with 1 token.
    ///
    /// Only supported for the `Transformer` variant (Llama/Qwen/Mistral).
    pub fn forward_batched(
        &mut self,
        inputs: &Array,
        caches: &mut [&mut AnyCache],
    ) -> Result<Array, Exception> {
        let mut kv_refs: Vec<&mut Vec<Option<cache::SteppingKeyValueCache>>> =
            Vec::with_capacity(caches.len());
        for cache in caches.iter_mut() {
            match cache {
                AnyCache::KV(kv) => kv_refs.push(kv),
                AnyCache::Hybrid(_) => {
                    return Err(Exception::custom(
                        "Batched forward not supported for Hybrid cache",
                    ));
                }
            }
        }

        match self {
            Self::Transformer(m) => m.forward_batched(inputs, &mut kv_refs),
            Self::Qwen3Moe(_)
            | Self::Qwen3Next(_)
            | Self::Gemma2(_)
            | Self::Phi3(_)
            | Self::Starcoder2(_)
            | Self::LlavaQwen2(_)
            | Self::DeepSeekV2(_) => Err(Exception::custom(
                "Batched forward only supported for Transformer models",
            )),
        }
    }

    /// Whether this model supports batched decode.
    pub const fn supports_batched_decode(&self) -> bool {
        matches!(self, Self::Transformer(_))
    }

    /// The model's hidden dimension.
    pub const fn hidden_size(&self) -> i32 {
        match self {
            Self::Transformer(m) => m.args.hidden_size,
            Self::Qwen3Moe(m) => m.args.hidden_size,
            Self::Qwen3Next(m) => m.args.hidden_size,
            Self::Gemma2(m) => m.args.hidden_size,
            Self::Phi3(m) => m.args.hidden_size,
            Self::Starcoder2(m) => m.args.hidden_size,
            Self::LlavaQwen2(m) => m.hidden_size(),
            Self::DeepSeekV2(m) => m.args.hidden_size,
        }
    }

    pub fn make_cache(&self) -> AnyCache {
        match self {
            Self::Transformer(m) => make_kv_cache(m.args.num_hidden_layers),
            Self::Qwen3Moe(m) => make_kv_cache(m.args.num_hidden_layers),
            Self::Gemma2(m) => make_kv_cache(m.args.num_hidden_layers),
            Self::Phi3(m) => make_kv_cache(m.args.num_hidden_layers),
            Self::Starcoder2(m) => make_kv_cache(m.args.num_hidden_layers),
            Self::LlavaQwen2(m) => make_kv_cache(m.num_hidden_layers()),
            Self::DeepSeekV2(m) => make_kv_cache(m.args.num_hidden_layers),
            Self::Qwen3Next(m) => AnyCache::Hybrid(m.make_cache()),
        }
    }

    /// Whether this model is a vision-language model that supports image input.
    pub const fn is_vlm(&self) -> bool {
        matches!(self, Self::LlavaQwen2(_))
    }

    /// The expected image size for the VLM's vision encoder, or `None` for text-only models.
    pub const fn image_size(&self) -> Option<i32> {
        match self {
            Self::LlavaQwen2(m) => Some(m.image_size()),
            Self::Transformer(_)
            | Self::Qwen3Next(_)
            | Self::Qwen3Moe(_)
            | Self::Gemma2(_)
            | Self::Phi3(_)
            | Self::Starcoder2(_)
            | Self::DeepSeekV2(_) => None,
        }
    }

    /// Forward pass for multimodal input (text + image).
    ///
    /// `input_ids` should contain `IMAGE_TOKEN_INDEX` (-200 as i32) at positions
    /// where image features should be inserted.
    pub fn forward_multimodal(
        &mut self,
        input_ids: &Array,
        pixel_values: &Array,
        cache: &mut AnyCache,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (Self::LlavaQwen2(m), AnyCache::KV(c)) => {
                m.forward_multimodal(input_ids, pixel_values, c)
            }
            _ => Err(Exception::custom(
                "Model does not support multimodal forward",
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Penalty application
// ---------------------------------------------------------------------------

/// Apply repetition/frequency/presence penalties to logits based on token history.
///
/// Creates penalty adjustment arrays on CPU and applies them as MLX ops.
/// Safe to call even when no penalties are active (returns logits unchanged).
#[allow(clippy::float_cmp)]
pub fn apply_penalties(
    logits: &Array,
    generated_tokens: &[u32],
    params: &SamplingParams,
) -> Result<Array, Exception> {
    if !params.has_penalties() || generated_tokens.is_empty() {
        return Ok(logits.clone());
    }

    let vocab_size = usize::try_from(
        *logits
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("logits must have at least 1 dimension"))?,
    )
    .map_err(|_| Exception::custom("negative vocab size"))?;

    let vocab_size_i32 =
        i32::try_from(vocab_size).map_err(|_| Exception::custom("vocab size overflow for i32"))?;

    // Count token occurrences
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &tid in generated_tokens {
        if usize::try_from(tid).is_ok_and(|t| t < vocab_size) {
            *counts.entry(tid).or_insert(0) += 1;
        }
    }

    let shape: Vec<i32> = logits.shape().to_vec();
    let mut result = logits.clone();

    // Repetition penalty: for seen tokens, divide positive logits by the penalty
    // and multiply negative logits by the penalty. This moves positive logits
    // down and negative logits further negative, making seen tokens less likely.
    if let Some(rep_penalty) = params.repetition_penalty {
        if rep_penalty != 1.0 {
            let inv = 1.0 / rep_penalty;
            let mut pos_factors = vec![1.0f32; vocab_size];
            let mut neg_factors = vec![1.0f32; vocab_size];
            for &tid in counts.keys() {
                let idx = usize::try_from(tid).unwrap_or(usize::MAX);
                if let Some(slot) = pos_factors.get_mut(idx) {
                    *slot = inv;
                }
                if let Some(slot) = neg_factors.get_mut(idx) {
                    *slot = rep_penalty;
                }
            }
            let pos_arr = Array::from_slice(&pos_factors, &[vocab_size_i32]).reshape(&shape)?;
            let neg_arr = Array::from_slice(&neg_factors, &[vocab_size_i32]).reshape(&shape)?;
            let is_positive = result.gt(Array::from_f32(0.0))?;
            let factor = mlx_rs::ops::r#where(&is_positive, &pos_arr, &neg_arr)?;
            result = result.multiply(factor)?;
        }
    }

    // Frequency penalty: logits[t] -= freq_penalty * count[t]
    if let Some(freq_penalty) = params.frequency_penalty {
        if freq_penalty != 0.0 {
            let mut freq = vec![0.0f32; vocab_size];
            for (&tid, &count) in &counts {
                if let Some(slot) = freq.get_mut(usize::try_from(tid).unwrap_or(usize::MAX)) {
                    *slot = freq_penalty * f32::from(u16::try_from(count).unwrap_or(u16::MAX));
                }
            }
            let freq_array = Array::from_slice(&freq, &[vocab_size_i32]).reshape(&shape)?;
            result = result.subtract(freq_array)?;
        }
    }

    // Presence penalty: logits[t] -= pres_penalty for all seen tokens
    if let Some(pres_penalty) = params.presence_penalty {
        if pres_penalty != 0.0 {
            let mut pres = vec![0.0f32; vocab_size];
            for &tid in counts.keys() {
                if let Some(slot) = pres.get_mut(usize::try_from(tid).unwrap_or(usize::MAX)) {
                    *slot = pres_penalty;
                }
            }
            let pres_array = Array::from_slice(&pres, &[vocab_size_i32]).reshape(&shape)?;
            result = result.subtract(pres_array)?;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Sample a token from logits using the given sampling parameters.
///
/// Shared across all model architectures. Penalties should be applied to
/// `logits` via [`apply_penalties`] before calling this function.
pub fn sample(logits: &Array, params: &SamplingParams) -> Result<Array, Exception> {
    if params.temperature == 0.0 {
        return argmax_axis!(logits, -1);
    }

    let scaled = logits.multiply(array!(1.0 / params.temperature))?;

    if params.needs_filtering() {
        sample_filtered(&scaled, params)
    } else {
        categorical!(scaled)
    }
}

/// Combined top-k + min-p + top-p filtering followed by categorical sampling.
///
/// Sorts probabilities descending, applies all three filters as masks,
/// renormalizes, then samples.
#[allow(clippy::shadow_reuse)]
fn sample_filtered(logits: &Array, params: &SamplingParams) -> Result<Array, Exception> {
    use mlx_rs::ops::{argsort_axis, concatenate_axis, maximum, softmax_axis};

    let probs = softmax_axis(logits, -1, None)?;
    let n_vocab_i32 = *probs
        .shape()
        .last()
        .ok_or_else(|| Exception::custom("logits must have at least 1 dimension"))?;
    let n_vocab =
        usize::try_from(n_vocab_i32).map_err(|_| Exception::custom("negative vocab size"))?;

    // Sort descending: negate, ascending argsort
    let neg_probs = probs.negative()?;
    let sorted_indices = argsort_axis(&neg_probs, -1)?;
    let sorted_probs = probs.take_along_axis(&sorted_indices, -1)?;

    // --- Top-k mask (CPU): zero out positions beyond rank k ---
    let k = params.top_k.map_or(n_vocab, |k| {
        usize::try_from(k).unwrap_or(1).clamp(1, n_vocab)
    });
    let mut rank_mask_vec = vec![1.0f32; n_vocab];
    for slot in rank_mask_vec.get_mut(k..).into_iter().flatten() {
        *slot = 0.0;
    }
    let rank_mask = if probs.ndim() > 1 {
        Array::from_slice(&rank_mask_vec, &[n_vocab_i32]).reshape(&[1, -1])?
    } else {
        Array::from_slice(&rank_mask_vec, &[n_vocab_i32])
    };

    // --- Top-p mask (GPU): cumulative probability ---
    let cumsum = sorted_probs.cumsum(-1, None, None)?;
    let cumsum_mask = cumsum.le(array!(params.top_p))?;

    // Always keep at least the top token
    let ones = Array::ones::<f32>(&[1])?;
    let zeros = Array::zeros::<f32>(&[n_vocab_i32 - 1])?;
    let first_token_mask = if probs.ndim() > 1 {
        concatenate_axis(&[&ones, &zeros], 0)?.reshape(&[1, -1])?
    } else {
        concatenate_axis(&[&ones, &zeros], 0)?
    };
    let top_p_mask = maximum(&cumsum_mask, &first_token_mask)?;

    // --- Min-p mask (GPU): prob >= min_p * max_prob ---
    let combined = if let Some(min_p) = params.min_p.map(|v| v.clamp(0.0, 1.0)) {
        let max_prob = sorted_probs.max_axes(&[-1], true)?;
        let threshold = max_prob.multiply(array!(min_p))?;
        let min_p_mask = sorted_probs.ge(threshold)?;
        rank_mask.multiply(top_p_mask)?.multiply(min_p_mask)?
    } else {
        rank_mask.multiply(top_p_mask)?
    };

    let filtered = sorted_probs.multiply(combined)?;

    // Re-normalize
    let sum = filtered.sum_axes(&[-1], true)?;
    let normalized = filtered.divide(sum)?;

    // categorical! expects unnormalized logits (applies softmax internally),
    // so convert back to log-space
    let log_probs = normalized.log()?;
    let sampled = categorical!(log_probs)?;
    // Map back to original indices
    sorted_indices
        .take_along_axis(&sampled.reshape(&[-1, 1])?, -1)?
        .squeeze_axes(&[-1])
}

// ---------------------------------------------------------------------------
// Logprob computation
// ---------------------------------------------------------------------------

/// Logprob data for a single decoded token (no string info -- that lives in
/// the server layer which has access to the tokenizer).
#[derive(Debug, Clone)]
pub struct TokenLogprobInfo {
    pub token_id: u32,
    pub logprob: f32,
    pub top_logprobs: Vec<TopLogprobEntry>,
}

#[derive(Debug, Clone)]
pub struct TopLogprobEntry {
    pub token_id: u32,
    pub logprob: f32,
}

/// Lazy logprob arrays produced during decode, before eval.
///
/// After `async_eval`, call [`LogprobArrays::materialize`] to extract values.
pub struct LogprobArrays {
    pub token_logprob: Array,
    pub top_indices: Option<Array>,
    pub top_values: Option<Array>,
}

impl LogprobArrays {
    /// Build lazy logprob arrays from post-temperature logits and the sampled token.
    ///
    /// `scaled_logits` should already have temperature applied (i.e. the same
    /// distribution that was sampled from).
    pub fn compute(
        scaled_logits: &Array,
        sampled_token: &Array,
        top_n: Option<u32>,
    ) -> Result<Self, Exception> {
        use mlx_rs::ops::argsort_axis;

        let log_probs = mlx_rs::nn::log_softmax(scaled_logits, -1)?;

        // Logprob of the chosen token — cast to f32 so materialize can use as_slice::<f32>
        // regardless of the model's compute dtype.
        let chosen_lp = log_probs
            .take_along_axis(&sampled_token.reshape(&[-1, 1])?, -1)?
            .squeeze_axes(&[-1])?
            .as_dtype(mlx_rs::Dtype::Float32)?;

        // Top-N logprobs
        let (top_indices, top_values) = if let Some(n) = top_n {
            let vocab_size = *log_probs.shape().last().unwrap_or(&0);
            let clamped = i32::try_from(n)
                .map_err(|_| Exception::custom("top_n overflow for i32"))?
                .min(vocab_size);
            let neg = log_probs.negative()?;
            let sorted_idx = argsort_axis(&neg, -1)?;
            let top_idx = sorted_idx.index((.., ..clamped));
            let top_vals = log_probs
                .take_along_axis(&top_idx, -1)?
                .as_dtype(mlx_rs::Dtype::Float32)?;
            (Some(top_idx), Some(top_vals))
        } else {
            (None, None)
        };

        Ok(Self {
            token_logprob: chosen_lp,
            top_indices,
            top_values,
        })
    }

    /// Collect all arrays that need to be included in `async_eval`.
    pub fn eval_targets(&self) -> Vec<&Array> {
        let mut targets = vec![&self.token_logprob];
        if let Some(ref idx) = self.top_indices {
            targets.push(idx);
        }
        if let Some(ref vals) = self.top_values {
            targets.push(vals);
        }
        targets
    }

    /// Extract concrete values after eval.
    pub fn materialize(&self, token_id: u32) -> TokenLogprobInfo {
        let logprob: f32 = self.token_logprob.item();

        let top_logprobs = match (&self.top_indices, &self.top_values) {
            (Some(indices), Some(values)) => {
                let ids = indices.as_slice::<u32>();
                let vals = values.as_slice::<f32>();
                ids.iter()
                    .zip(vals.iter())
                    .map(|(&id, &lp)| TopLogprobEntry {
                        token_id: id,
                        logprob: lp,
                    })
                    .collect()
            }
            _ => vec![],
        };

        TokenLogprobInfo {
            token_id,
            logprob,
            top_logprobs,
        }
    }
}

/// Weight map index from model.safetensors.index.json.
#[derive(Debug, Clone, Deserialize)]
pub struct WeightMapIndex {
    /// Model metadata (e.g. total parameter count, dtype).
    pub metadata: HashMap<String, Value>,
    /// Maps weight tensor name to the shard filename that contains it.
    pub weight_map: HashMap<String, String>,
}

/// Load a tokenizer from a model directory.
pub fn load_tokenizer<P: AsRef<Path>>(model_dir: P) -> Result<tokenizers::Tokenizer, ModelError> {
    let file = model_dir.as_ref().join("tokenizer.json");
    tokenizers::Tokenizer::from_file(file)
        .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))
}

/// Load safetensors weights into any model that implements `ModuleParametersExt`.
pub fn load_safetensors_weights<M: ModuleParametersExt>(
    model: &mut M,
    model_path: &Path,
) -> Result<(), ModelError> {
    load_quantized_safetensors_weights(model, model_path, false)
}

/// Load safetensors weights with optional name remapping for quantized models.
///
/// Pre-quantized MLX models store weights with flat names (e.g., `q_proj.weight`)
/// but the Rust quantized types use a nested structure (`q_proj.inner.weight`).
/// When `quantized` is true, keys ending in `.weight` or `.bias` (but not
/// `.biases`) that don't match directly are retried with `.inner.` inserted.
pub fn load_quantized_safetensors_weights<M: ModuleParametersExt>(
    model: &mut M,
    model_path: &Path,
    quantized: bool,
) -> Result<(), ModelError> {
    let safetensors_files = collect_safetensors_files(model_path)?;

    let mut params = model.parameters_mut().flatten();

    for file_path in &safetensors_files {
        tracing::debug!(file = %file_path.display(), "Loading weights");
        let loaded = Array::load_safetensors(file_path)
            .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

        for (key, value) in loaded {
            if let Some(param) = params.get_mut(&*key) {
                **param = value;
            } else if quantized {
                if let Some(remapped) = remap_quantized_key(&key) {
                    if let Some(param) = params.get_mut(&*remapped) {
                        **param = value;
                    }
                }
            }
        }
    }

    model
        .eval()
        .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

    Ok(())
}

/// Load safetensors weights with prefix stripping.
///
/// Only processes weights whose key starts with `prefix`. The prefix is
/// stripped before matching against model parameters. Used for VLMs where
/// the language model weights are prefixed with `language_model.`.
pub fn load_quantized_safetensors_weights_with_prefix<M: ModuleParametersExt>(
    model: &mut M,
    model_path: &Path,
    quantized: bool,
    prefix: &str,
) -> Result<(), ModelError> {
    let safetensors_files = collect_safetensors_files(model_path)?;

    let mut params = model.parameters_mut().flatten();

    for file_path in &safetensors_files {
        tracing::debug!(file = %file_path.display(), prefix, "Loading weights with prefix");
        let loaded = Array::load_safetensors(file_path)
            .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

        for (key, value) in loaded {
            let Some(stripped) = key.strip_prefix(prefix) else {
                continue;
            };
            if let Some(param) = params.get_mut(stripped) {
                **param = value;
            } else if quantized {
                if let Some(remapped) = remap_quantized_key(stripped) {
                    if let Some(param) = params.get_mut(&*remapped) {
                        **param = value;
                    }
                }
            }
        }
    }

    model
        .eval()
        .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

    Ok(())
}

/// Collect safetensors file paths from a model directory.
fn collect_safetensors_files(model_path: &Path) -> Result<Vec<std::path::PathBuf>, ModelError> {
    let index_path = model_path.join("model.safetensors.index.json");
    if index_path.exists() {
        let json = std::fs::read_to_string(&index_path)?;
        let index: WeightMapIndex = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = index.weight_map.values().collect();
        Ok(weight_files
            .into_iter()
            .map(|f| model_path.join(f))
            .collect())
    } else {
        let single_path = model_path.join("model.safetensors");
        if single_path.exists() {
            Ok(vec![single_path])
        } else {
            Err(ModelError::MissingWeight(
                "No safetensors files found".to_owned(),
            ))
        }
    }
}

/// Remap a safetensors key for quantized model parameter names.
///
/// MLX Python saves quantized weights as `layer.weight` but Rust mlx-rs
/// QuantizedLinear/QuantizedEmbedding nest them as `layer.inner.weight`.
#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn remap_quantized_key(key: &str) -> Option<String> {
    // MaybeQuantized<nn::Linear> (used in gemma2, etc.) nests quantized params under `.inner.*`.
    if let Some(prefix) = key.strip_suffix(".weight") {
        Some(format!("{prefix}.inner.weight"))
    } else if let Some(prefix) = key.strip_suffix(".scales") {
        Some(format!("{prefix}.inner.scales"))
    } else if let Some(prefix) = key.strip_suffix(".biases") {
        Some(format!("{prefix}.inner.biases"))
    } else if key.ends_with(".bias") {
        let prefix = key.strip_suffix(".bias")?;
        Some(format!("{prefix}.inner.bias"))
    } else {
        None
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn params(temp: f32, top_p: f32) -> SamplingParams {
        SamplingParams {
            temperature: temp,
            top_p,
            ..SamplingParams::default()
        }
    }

    fn assert_sample_shape(
        logit_data: &[f32],
        logit_shape: &[i32],
        temp: f32,
        top_p: f32,
        expected_shape: &[i32],
    ) -> Array {
        let logits = Array::from_slice(logit_data, logit_shape);
        let token = sample(&logits, &params(temp, top_p)).unwrap();
        assert_eq!(token.shape(), expected_shape);
        token
    }

    #[test]
    fn test_sample_greedy() {
        let token = assert_sample_shape(&[0.1_f32, 0.9, 0.0], &[1, 3], 0.0, 1.0, &[1]);
        let val: u32 = token.item();
        assert_eq!(val, 1);
    }

    #[test]
    fn sample_2d_input_returns_1d_shape() {
        assert_sample_shape(&[0.1_f32, 0.9, 0.0], &[1, 3], 0.0, 1.0, &[1]);
    }

    #[test]
    fn sample_3d_input_returns_2d_shape() {
        // argmax on axis -1 of [1, 1, V] produces [1, 1], not [1].
        // Callers must slice to 2D before calling sample.
        assert_sample_shape(&[0.1_f32, 0.9, 0.0], &[1, 1, 3], 0.0, 1.0, &[1, 1]);
    }

    #[test]
    fn remap_quantized_key_weight() {
        assert_eq!(
            remap_quantized_key("model.layers.0.self_attn.q_proj.weight"),
            Some("model.layers.0.self_attn.q_proj.inner.weight".to_owned())
        );
    }

    #[test]
    fn remap_quantized_key_bias() {
        assert_eq!(
            remap_quantized_key("model.layers.0.self_attn.q_proj.bias"),
            Some("model.layers.0.self_attn.q_proj.inner.bias".to_owned())
        );
    }

    #[test]
    fn remap_quantized_key_biases() {
        // ".biases" (quantization bias) remaps to ".inner.biases" for MaybeQuantized layers
        assert_eq!(
            remap_quantized_key("model.layers.0.self_attn.q_proj.biases"),
            Some("model.layers.0.self_attn.q_proj.inner.biases".to_owned())
        );
    }

    #[test]
    fn remap_quantized_key_scales() {
        assert_eq!(
            remap_quantized_key("model.layers.0.self_attn.q_proj.scales"),
            Some("model.layers.0.self_attn.q_proj.inner.scales".to_owned())
        );
    }

    #[test]
    fn remap_quantized_key_embed_tokens_weight() {
        assert_eq!(
            remap_quantized_key("model.embed_tokens.weight"),
            Some("model.embed_tokens.inner.weight".to_owned())
        );
    }

    #[test]
    fn remap_quantized_key_norm_weight() {
        // RmsNorm weight should still remap -- it will just not match any param
        // and be silently skipped by the loader
        assert_eq!(
            remap_quantized_key("model.norm.weight"),
            Some("model.norm.inner.weight".to_owned())
        );
    }

    #[test]
    fn sample_with_temperature() {
        assert_sample_shape(&[0.1_f32, 100.0, 0.0], &[1, 3], 0.5, 1.0, &[1]);
    }

    #[test]
    fn sample_with_top_p() {
        assert_sample_shape(&[0.0_f32, 100.0, 0.0], &[1, 3], 1.0, 0.5, &[1]);
    }

    #[test]
    fn sample_single_element_logits() {
        let token = assert_sample_shape(&[5.0_f32], &[1, 1], 0.0, 1.0, &[1]);
        let val: u32 = token.item();
        assert_eq!(val, 0, "Single-element logits must return index 0");
    }

    #[test]
    fn sample_uniform_logits_greedy() {
        let token = assert_sample_shape(&[1.0_f32, 1.0, 1.0, 1.0], &[1, 4], 0.0, 1.0, &[1]);
        let val: u32 = token.item();
        assert_eq!(val, 0, "Tied logits should return first index via argmax");
    }

    #[test]
    fn sample_uniform_logits_with_temperature() {
        assert_sample_shape(&[1.0_f32, 1.0, 1.0, 1.0], &[1, 4], 1.0, 1.0, &[1]);
    }

    #[test]
    fn sample_top_p_with_uniform_logits() {
        assert_sample_shape(&[1.0_f32, 1.0, 1.0, 1.0], &[1, 4], 1.0, 0.3, &[1]);
    }

    #[test]
    fn load_tokenizer_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_tokenizer(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn collect_safetensors_missing_both_files() {
        let dir = tempfile::tempdir().unwrap();
        let result = collect_safetensors_files(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ModelError::MissingWeight(_)),
            "Expected MissingWeight, got: {err:?}"
        );
    }

    #[test]
    fn collect_safetensors_single_file() {
        let dir = tempfile::tempdir().unwrap();
        // Create a dummy model.safetensors file (content doesn't matter for path collection)
        std::fs::write(dir.path().join("model.safetensors"), b"dummy").unwrap();
        let result = collect_safetensors_files(dir.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert!(
            result
                .first()
                .unwrap()
                .to_string_lossy()
                .contains("model.safetensors")
        );
    }

    #[test]
    fn collect_safetensors_index_json() {
        let dir = tempfile::tempdir().unwrap();
        let index_json = r#"{
            "metadata": {"total_size": 12345},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
        std::fs::write(dir.path().join("model.safetensors.index.json"), index_json).unwrap();
        let result = collect_safetensors_files(dir.path()).unwrap();
        // Two unique shard files
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn weight_map_index_deserialization() {
        let json = r#"{
            "metadata": {"format": "pt", "total_size": 999},
            "weight_map": {
                "layer.0.weight": "shard-0.safetensors",
                "layer.1.weight": "shard-1.safetensors"
            }
        }"#;
        let index: WeightMapIndex = serde_json::from_str(json).unwrap();
        assert_eq!(index.weight_map.len(), 2);
        assert_eq!(
            index.metadata.get("format").and_then(|v| v.as_str()),
            Some("pt")
        );
    }

    #[test]
    fn weight_map_index_empty_maps() {
        let json = r#"{"metadata": {}, "weight_map": {}}"#;
        let index: WeightMapIndex = serde_json::from_str(json).unwrap();
        assert!(index.weight_map.is_empty());
        assert!(index.metadata.is_empty());
    }

    #[test]
    fn any_cache_kv_variant() {
        let cache = AnyCache::KV(vec![None, Some(cache::SteppingKeyValueCache::new())]);
        match &cache {
            AnyCache::KV(layers) => assert_eq!(layers.len(), 2),
            AnyCache::Hybrid(_) => panic!("Expected KV variant"),
        }
    }

    #[test]
    fn any_cache_hybrid_variant() {
        let cache = AnyCache::Hybrid(Vec::new());
        assert!(matches!(cache, AnyCache::Hybrid(_)));
    }

    #[test]
    fn make_kv_cache_positive_layers() {
        let cache = super::make_kv_cache(4);
        match &cache {
            AnyCache::KV(layers) => {
                assert_eq!(layers.len(), 4);
                for c in layers {
                    assert!(c.is_some());
                }
            }
            AnyCache::Hybrid(_) => panic!("Expected KV variant"),
        }
    }

    #[test]
    fn make_kv_cache_zero_layers() {
        let cache = super::make_kv_cache(0);
        match &cache {
            AnyCache::KV(layers) => assert!(layers.is_empty()),
            AnyCache::Hybrid(_) => panic!("Expected KV variant"),
        }
    }

    #[test]
    fn make_kv_cache_negative_layers() {
        let cache = super::make_kv_cache(-1);
        match &cache {
            AnyCache::KV(layers) => assert!(layers.is_empty()),
            AnyCache::Hybrid(_) => panic!("Expected KV variant"),
        }
    }

    fn small_qwen3_moe_args() -> qwen3_moe::Qwen3MoeModelArgs {
        qwen3_moe::Qwen3MoeModelArgs {
            model_type: "qwen3_moe".to_owned(),
            hidden_size: 32,
            num_hidden_layers: 2,
            intermediate_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            vocab_size: 64,
            max_position_embeddings: 128,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            attention_bias: false,
            rope_scaling: None,
            head_dim: None,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: 16,
            decoder_sparse_step: 1,
            mlp_only_layers: vec![],
            norm_topk_prob: true,
            quantization: None,
        }
    }

    #[test]
    fn any_model_qwen3_moe_make_cache_returns_kv() {
        let model = qwen3_moe::Qwen3MoeCausalLM::new(small_qwen3_moe_args()).unwrap();
        let any = AnyModel::Qwen3Moe(model);
        let cache = any.make_cache();
        match &cache {
            AnyCache::KV(layers) => assert_eq!(layers.len(), 2),
            AnyCache::Hybrid(_) => panic!("Expected KV cache for Qwen3Moe"),
        }
    }

    #[test]
    fn any_model_qwen3_moe_forward_with_hybrid_cache_errors() {
        let model = qwen3_moe::Qwen3MoeCausalLM::new(small_qwen3_moe_args()).unwrap();
        let mut any = AnyModel::Qwen3Moe(model);
        let mut cache = AnyCache::Hybrid(vec![]);
        let input = Array::from_slice(&[1_i32], &[1, 1]);
        let result = any.forward(&input, None, &mut cache);
        assert!(result.is_err(), "Qwen3Moe with Hybrid cache should error");
    }

    #[test]
    fn any_model_qwen3_moe_forward_dispatches_to_kv_cache() {
        let model = qwen3_moe::Qwen3MoeCausalLM::new(small_qwen3_moe_args()).unwrap();
        let mut any = AnyModel::Qwen3Moe(model);
        let mut cache = any.make_cache();
        let input = Array::from_slice(&[1_i32, 2, 3], &[1, 3]);
        // Forward dispatches correctly (Qwen3Moe + KV cache), but returns
        // Err because unloaded QLinear weights are float32 placeholders.
        // The important assertion: it does NOT return "Model/cache type mismatch".
        let result = any.forward(&input, None, &mut cache);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            !err_msg.contains("mismatch"),
            "Should dispatch correctly, not hit mismatch arm"
        );
    }

    // --- SamplingParams tests ---

    #[test]
    fn sampling_params_default() {
        let p = SamplingParams::default();
        assert!((p.temperature - 1.0).abs() < f32::EPSILON);
        assert!((p.top_p - 1.0).abs() < f32::EPSILON);
        assert!(p.top_k.is_none());
        assert!(p.min_p.is_none());
        assert!(!p.has_penalties());
    }

    #[test]
    fn sampling_params_has_penalties() {
        let mut p = SamplingParams::default();
        assert!(!p.has_penalties());

        p.repetition_penalty = Some(1.0);
        assert!(!p.has_penalties(), "rep_penalty=1.0 should not count");

        p.repetition_penalty = Some(1.2);
        assert!(p.has_penalties());
    }

    // --- Top-k tests ---

    #[test]
    fn sample_top_k_limits_to_k_tokens() {
        // logits: [0, 0, 0, 100] -- greedy picks index 3
        // With top_k=2, temp=1.0: only the top 2 logit positions should be sampled.
        // index 3 (logit=100) dominates, so we should still get 3.
        let logits = Array::from_slice(&[0.0_f32, 0.0, 0.0, 100.0], &[1, 4]);
        let p = SamplingParams {
            temperature: 1.0,
            top_k: Some(2),
            ..SamplingParams::default()
        };
        let token = sample(&logits, &p).unwrap();
        let val: u32 = token.item();
        assert_eq!(val, 3);
    }

    #[test]
    fn sample_top_k_larger_than_vocab_is_no_op() {
        let logits = Array::from_slice(&[0.1_f32, 100.0, 0.0], &[1, 3]);
        let p = SamplingParams {
            temperature: 1.0,
            top_k: Some(100),
            ..SamplingParams::default()
        };
        let token = sample(&logits, &p).unwrap();
        assert_eq!(token.shape(), &[1]);
    }

    // --- Min-p tests ---

    #[test]
    fn sample_min_p_filters_low_probability() {
        // logits: [100, 0, 0, 0] -- token 0 has ~100% probability
        // min_p=0.5 should filter everything except token 0
        let logits = Array::from_slice(&[100.0_f32, 0.0, 0.0, 0.0], &[1, 4]);
        let p = SamplingParams {
            temperature: 1.0,
            min_p: Some(0.5),
            ..SamplingParams::default()
        };
        let token = sample(&logits, &p).unwrap();
        let val: u32 = token.item();
        assert_eq!(val, 0);
    }

    // --- Penalty tests ---

    #[test]
    fn apply_penalties_no_penalties_returns_clone() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[1, 3]);
        let result = apply_penalties(&logits, &[0, 1], &SamplingParams::default()).unwrap();
        mlx_rs::transforms::eval([&result]).unwrap();
        assert_eq!(result.as_slice::<f32>(), logits.as_slice::<f32>());
    }

    #[test]
    fn apply_penalties_empty_tokens_returns_clone() {
        let logits = Array::from_slice(&[1.0_f32, 2.0, 3.0], &[1, 3]);
        let p = SamplingParams {
            repetition_penalty: Some(2.0),
            ..SamplingParams::default()
        };
        let result = apply_penalties(&logits, &[], &p).unwrap();
        mlx_rs::transforms::eval([&result]).unwrap();
        assert_eq!(result.as_slice::<f32>(), logits.as_slice::<f32>());
    }

    #[test]
    fn apply_repetition_penalty() {
        let logits = Array::from_slice(&[4.0_f32, 2.0, 6.0], &[1, 3]);
        let p = SamplingParams {
            repetition_penalty: Some(2.0),
            ..SamplingParams::default()
        };
        let result = apply_penalties(&logits, &[0, 2], &p).unwrap();
        mlx_rs::transforms::eval([&result]).unwrap();
        let vals = result.as_slice::<f32>();
        // token 0 (positive, seen): 4.0 / 2.0 = 2.0
        // token 1 (positive, unseen): 2.0 (unchanged)
        // token 2 (positive, seen): 6.0 / 2.0 = 3.0
        assert!((vals[0] - 2.0).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
        assert!((vals[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn apply_repetition_penalty_negative_logits() {
        let logits = Array::from_slice(&[-4.0_f32, 2.0, -6.0], &[1, 3]);
        let p = SamplingParams {
            repetition_penalty: Some(2.0),
            ..SamplingParams::default()
        };
        let result = apply_penalties(&logits, &[0, 2], &p).unwrap();
        mlx_rs::transforms::eval([&result]).unwrap();
        let vals = result.as_slice::<f32>();
        // token 0 (negative, seen): -4.0 * 2.0 = -8.0 (more negative = less likely)
        // token 1 (positive, unseen): 2.0 (unchanged)
        // token 2 (negative, seen): -6.0 * 2.0 = -12.0 (more negative = less likely)
        assert!((vals[0] - (-8.0)).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
        assert!((vals[2] - (-12.0)).abs() < 1e-5);
    }

    #[test]
    fn apply_frequency_penalty() {
        let logits = Array::from_slice(&[4.0_f32, 2.0, 6.0], &[1, 3]);
        let p = SamplingParams {
            frequency_penalty: Some(1.0),
            ..SamplingParams::default()
        };
        // token 1 appears twice
        let result = apply_penalties(&logits, &[1, 1, 2], &p).unwrap();
        mlx_rs::transforms::eval([&result]).unwrap();
        let vals = result.as_slice::<f32>();
        // token 0: 4.0 (unchanged)
        // token 1: 2.0 - 1.0*2 = 0.0
        // token 2: 6.0 - 1.0*1 = 5.0
        assert!((vals[0] - 4.0).abs() < 1e-5);
        assert!((vals[1]).abs() < 1e-5);
        assert!((vals[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn apply_presence_penalty() {
        let logits = Array::from_slice(&[4.0_f32, 2.0, 6.0], &[1, 3]);
        let p = SamplingParams {
            presence_penalty: Some(1.5),
            ..SamplingParams::default()
        };
        let result = apply_penalties(&logits, &[0, 0, 2], &p).unwrap();
        mlx_rs::transforms::eval([&result]).unwrap();
        let vals = result.as_slice::<f32>();
        // token 0: 4.0 - 1.5 = 2.5
        // token 1: 2.0 (unchanged)
        // token 2: 6.0 - 1.5 = 4.5
        assert!((vals[0] - 2.5).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
        assert!((vals[2] - 4.5).abs() < 1e-5);
    }
}
