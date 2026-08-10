//! `DFlash` block-diffusion drafter for speculative decoding.
//!
//! A 0.5B drafter that produces 16 draft tokens per round via a single
//! non-causal forward pass. The Modal drafter checkpoints verified by the
//! loader tests use 8 target-layer taps: `[1, 6, 11, 16, 22, 27, 32, 37]`.
//!
//! Architecture: 6 decoder layers with dual-stream attention —
//! Q from noise embedding, K/V from `concat(target_hidden, noise)`.
//! No `embed_tokens` or `lm_head` — uses the target model's `lm_head`.
//!
//! Reference checkpoints: `modal-labs/Qwen3.6-35B-A3B-DFlash` and
//! `modal-labs/Qwen3.5-9B-DFlash`.
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{BufReader, Read},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters as _, ModuleParametersExt as _, Param},
    nn,
    ops::{self, indexing::IndexOp},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{error::ModelError, utils::apply_rope};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct DFlashSubConfig {
    target_layer_ids: Vec<usize>,
    /// Meaning of each target-layer id. Prism's GGUF runtime captures the
    /// residual stream after the indexed layer has completed.
    #[serde(default)]
    tap_semantics: Option<DFlashTapSemantics>,
    #[serde(default)]
    mask_token_id: Option<i32>,
    /// Prism dSpark checkpoints use the `DFlash` trunk plus log-SNR conditioning
    /// and a sequential low-rank Markov resampler.
    #[serde(default)]
    dspark: bool,
    #[serde(default)]
    markov_rank: i32,
    #[serde(default)]
    log_snr_conditioning: bool,
    #[serde(default)]
    min_log_snr: f32,
    #[serde(default)]
    max_log_snr: f32,
    /// Omit dSpark's frozen Q4 output copy and use the paired target's head.
    /// Verification remains distribution-exact; this only changes proposals.
    #[serde(default)]
    reuse_target_head: bool,
    /// Exact target artifact this trained dSpark sidecar is allowed to pair
    /// with. Required for dSpark and ignored for generic DFlash checkpoints.
    #[serde(default)]
    target_binding: Option<TargetArtifactBinding>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
enum DFlashTapSemantics {
    #[serde(rename = "post_layer_residual_v1")]
    PostLayerResidualV1,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetArtifactBinding {
    format: String,
    files: Vec<TargetArtifactFile>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetArtifactFile {
    path: String,
    size: u64,
    sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DFlashConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    pub intermediate_size: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_block_size")]
    pub block_size: i32,
    pub vocab_size: i32,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub sliding_window: Option<i32>,
    /// Quantization config for the drafter. When present, linear layers use
    /// quantized matmul; when absent, they use dense matmul.
    #[serde(default)]
    pub quantization: Option<crate::qwen3_next::QuantizationConfig>,
    dflash_config: DFlashSubConfig,
}

impl DFlashConfig {
    pub(crate) const fn quant_spec(&self) -> crate::qwen3_next::QuantSpec {
        match &self.quantization {
            Some(q) => q.spec(),
            None => crate::qwen3_next::QuantSpec {
                group_size: 0,
                bits: 0,
                mode: crate::quant_mode::QuantMode::Dense,
            },
        }
    }

    pub fn target_layer_ids(&self) -> &[usize] {
        &self.dflash_config.target_layer_ids
    }

    pub const fn num_taps(&self) -> usize {
        self.dflash_config.target_layer_ids.len()
    }

    pub fn mask_token_id(&self) -> i32 {
        self.dflash_config.mask_token_id.unwrap_or(248_070)
    }

    pub const fn is_dspark(&self) -> bool {
        self.dflash_config.dspark
    }

    pub const fn reuse_target_head(&self) -> bool {
        self.dflash_config.reuse_target_head
    }
}

const fn default_head_dim() -> i32 {
    128
}

const fn default_rms_norm_eps() -> f32 {
    1e-6
}

const fn default_rope_theta() -> f32 {
    1e7
}

const fn default_block_size() -> i32 {
    16
}

// ---------------------------------------------------------------------------
// SwiGLU MLP (non-quantized)
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
struct DFlashMLP {
    #[param]
    gate_proj: crate::qwen3_next::QLinear,
    #[param]
    up_proj: crate::qwen3_next::QLinear,
    #[param]
    down_proj: crate::qwen3_next::QLinear,
}

impl DFlashMLP {
    fn new(
        _hidden_size: i32,
        _intermediate_size: i32,
        spec: crate::qwen3_next::QuantSpec,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: crate::qwen3_next::QLinear::new_spec(spec)?,
            up_proj: crate::qwen3_next::QLinear::new_spec(spec)?,
            down_proj: crate::qwen3_next::QLinear::new_spec(spec)?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = nn::sigmoid(&gate)?.multiply(&gate)?.multiply(&up)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// DFlash dual-stream attention
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
struct DFlashAttention {
    #[param]
    q_proj: crate::qwen3_next::QLinear,
    #[param]
    k_proj: crate::qwen3_next::QLinear,
    #[param]
    v_proj: crate::qwen3_next::QLinear,
    #[param]
    o_proj: crate::qwen3_next::QLinear,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
    #[param]
    rope: nn::Rope,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    scale: f32,
    is_sliding: bool,
    sliding_window: i32,
}

impl DFlashAttention {
    fn new(
        config: &DFlashConfig,
        layer_idx: usize,
        spec: crate::qwen3_next::QuantSpec,
    ) -> Result<Self, Exception> {
        let head_dim = config.head_dim;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let is_sliding = config
            .layer_types
            .as_ref()
            .and_then(|lt| lt.get(layer_idx))
            .is_some_and(|t| t == "sliding_attention");
        let sliding_window = config.sliding_window.unwrap_or(0);

        Ok(Self {
            q_proj: crate::qwen3_next::QLinear::new_spec(spec)?,
            k_proj: crate::qwen3_next::QLinear::new_spec(spec)?,
            v_proj: crate::qwen3_next::QLinear::new_spec(spec)?,
            o_proj: crate::qwen3_next::QLinear::new_spec(spec)?,
            q_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(config.rms_norm_eps)
                .build()?,
            k_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(config.rms_norm_eps)
                .build()?,
            rope: nn::RopeBuilder::new(head_dim)
                .traditional(false)
                .base(config.rope_theta)
                .scale(1.0)
                .build()
                .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?,
            num_attention_heads: n_heads,
            num_key_value_heads: n_kv_heads,
            head_dim,
            scale: f32::from(
                i16::try_from(head_dim)
                    .map_err(|_| Exception::custom("head_dim out of i16 range"))?,
            )
            .sqrt()
            .recip(),
            is_sliding,
            sliding_window,
        })
    }

    /// Project and append target tap context without running draft queries.
    /// This lets long target prompts prime the drafter cache one bounded chunk
    /// at a time instead of retaining every layer tap until the first round.
    fn append_target_context(
        &mut self,
        target_hidden: &Array,
        cache: &mut Option<(Array, Array)>,
        cache_offset: i32,
    ) -> Result<(Array, Array), Exception> {
        let batch = *target_hidden
            .shape()
            .first()
            .ok_or_else(|| Exception::custom("target context needs 3D input"))?;
        let context_len = *target_hidden
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("target context needs 3D input"))?;
        if cache_offset < 0 {
            return Err(Exception::custom(
                "drafter target-context position must be non-negative",
            ));
        }
        let context_end = cache_offset
            .checked_add(context_len)
            .ok_or_else(|| Exception::custom("drafter target-context position overflow"))?;
        let context_k = self.k_proj.forward(target_hidden)?;
        let context_v = self.v_proj.forward(target_hidden)?;
        let context_k =
            context_k.reshape(&[batch, context_len, self.num_key_value_heads, self.head_dim])?;
        let context_k = self
            .k_norm
            .forward(&context_k)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let context_v = context_v
            .reshape(&[batch, context_len, self.num_key_value_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let positions = (cache_offset..context_end).collect::<Vec<_>>();
        let positions = Array::from_slice(&positions, &[context_len]);
        let context_k = crate::qwen3_next::apply_rope_manual(
            &context_k,
            &positions,
            self.rope.dimensions,
            self.rope.base,
            self.rope.scale,
        )?;

        let (context_k, context_v) = if self.is_sliding && self.sliding_window > 1 {
            let keep = self.sliding_window - 1;
            // If the new chunk alone fills the window, old KV is provably
            // unreachable. Slice it before concatenation: besides saving a
            // graph node, this avoids making correctness depend on slicing a
            // lazy concat whose prefix will be discarded in full.
            let (context_k, context_v) = if context_len >= keep {
                (context_k, context_v)
            } else if let Some((cached_k, cached_v)) = cache.as_ref() {
                (
                    ops::concatenate_axis(&[cached_k, &context_k], 2)?,
                    ops::concatenate_axis(&[cached_v, &context_v], 2)?,
                )
            } else {
                (context_k, context_v)
            };
            let len = context_k.shape().get(2).copied().unwrap_or(0);
            if len > keep {
                let skip = len - keep;
                (
                    context_k.index((.., .., skip.., ..)),
                    context_v.index((.., .., skip.., ..)),
                )
            } else {
                (context_k, context_v)
            }
        } else if let Some((cached_k, cached_v)) = cache.as_ref() {
            (
                ops::concatenate_axis(&[cached_k, &context_k], 2)?,
                ops::concatenate_axis(&[cached_v, &context_v], 2)?,
            )
        } else {
            (context_k, context_v)
        };
        *cache = Some((context_k.clone(), context_v.clone()));
        Ok((context_k, context_v))
    }

    /// Dual-stream attention after target context has been committed: Q from
    /// noise, K/V from cached target context plus fresh noise.
    ///
    /// `noise`: `[B, block_size, hidden]` — the 16 draft positions.
    /// `cache`: target-context K/V, shape `[B, n_kv, retained_len, head_dim]`.
    /// `noise_position`: absolute first position of the draft block.
    #[allow(non_snake_case, clippy::shadow_reuse)]
    fn forward_noise(
        &mut self,
        noise: &Array,
        cache: &Option<(Array, Array)>,
        noise_position: i32,
    ) -> Result<Array, Exception> {
        let B = *noise
            .shape()
            .first()
            .ok_or_else(|| Exception::custom("need 3D"))?;
        let q_len = *noise
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("need 3D"))?;
        // Q from noise only
        let q = self.q_proj.forward(noise)?;
        let q = q.reshape(&[B, q_len, self.num_attention_heads, self.head_dim])?;
        let q = self.q_norm.forward(&q)?.transpose_axes(&[0, 2, 1, 3])?;

        // K/V from noise — freshly computed every round, never cached
        let noise_k = self.k_proj.forward(noise)?;
        let noise_v = self.v_proj.forward(noise)?;
        let noise_k = noise_k.reshape(&[B, q_len, self.num_key_value_heads, self.head_dim])?;
        let noise_k = self
            .k_norm
            .forward(&noise_k)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let noise_v = noise_v
            .reshape(&[B, q_len, self.num_key_value_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = apply_rope(&q, &self.rope, noise_position)?;
        let noise_k = apply_rope(&noise_k, &self.rope, noise_position)?;
        let (ctx_k, ctx_v) = cache
            .as_ref()
            .ok_or_else(|| Exception::custom("drafter target context cache is empty"))?;

        // Attention over cached_context + fresh_noise
        let k = ops::concatenate_axis(&[ctx_k, &noise_k], 2)?;
        let v = ops::concatenate_axis(&[ctx_v, &noise_v], 2)?;

        // Non-causal SDPA (no mask)
        let output = mlx_rs::fast::scaled_dot_product_attention(
            q,
            k,
            v,
            self.scale,
            None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            None::<&Array>,
        )?;

        // [B, n_heads, q_len, head_dim] -> [B, q_len, n_heads * head_dim]
        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[B, q_len, -1])?;
        self.o_proj.forward(&output)
    }
}

// ---------------------------------------------------------------------------
// DFlash decoder layer
// ---------------------------------------------------------------------------

#[derive(Debug, ModuleParameters)]
struct DFlashDecoderLayer {
    #[param]
    self_attn: DFlashAttention,
    #[param]
    mlp: DFlashMLP,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
}

impl DFlashDecoderLayer {
    fn new(
        config: &DFlashConfig,
        layer_idx: usize,
        spec: crate::qwen3_next::QuantSpec,
    ) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: DFlashAttention::new(config, layer_idx, spec)?,
            mlp: DFlashMLP::new(config.hidden_size, config.intermediate_size, spec)?,
            input_layernorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
        })
    }

    fn forward_noise(
        &mut self,
        noise: &Array,
        cache: &Option<(Array, Array)>,
        noise_position: i32,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(noise)?;
        let attn_out = self
            .self_attn
            .forward_noise(&normed, cache, noise_position)?;
        let h = noise.add(attn_out)?;
        let normed_post = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed_post)?;
        h.add(mlp_out)
    }

    fn prime_target_context(
        &mut self,
        target_hidden: &Array,
        cache: &mut Option<(Array, Array)>,
        cache_offset: i32,
    ) -> Result<(), Exception> {
        self.self_attn
            .append_target_context(target_hidden, cache, cache_offset)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Prism dSpark heads
// ---------------------------------------------------------------------------

const DSPARK_LOG_SNR_FEATURES: i32 = 128;

#[derive(Debug, ModuleParameters)]
struct DsparkExtras {
    #[param]
    log_snr_fc1: nn::Linear,
    #[param]
    log_snr_fc2: nn::Linear,
    /// Dense `[vocab, rank]` lookup table indexed by the previous token.
    #[param]
    markov_head_a: Param<Array>,
    /// Quantized `[vocab, rank]` projection producing the Markov logit bias.
    #[param]
    markov_head_b: crate::qwen3_next::QLinear,
    /// dSpark's frozen, higher-precision copy of the target output head.
    #[param]
    output: Option<crate::qwen3_next::QLinear>,
    log_snr_features: Array,
    /// Materialized once after checkpoint loading. The log-SNR schedule is
    /// constant for every speculative round, so rerunning its two dense MLPs
    /// would only add dispatch and bandwidth overhead.
    log_snr_embedding: Option<Array>,
}

impl DsparkExtras {
    fn new(config: &DFlashConfig, spec: crate::qwen3_next::QuantSpec) -> Result<Self, Exception> {
        if !config.dflash_config.log_snr_conditioning {
            return Err(Exception::custom(
                "Prism dSpark requires log_snr_conditioning=true",
            ));
        }
        if config.dflash_config.markov_rank <= 0 {
            return Err(Exception::custom("Prism dSpark requires markov_rank > 0"));
        }
        let min_log_snr = config.dflash_config.min_log_snr;
        let max_log_snr = config.dflash_config.max_log_snr;
        if !min_log_snr.is_finite() || !max_log_snr.is_finite() || max_log_snr <= min_log_snr {
            return Err(Exception::custom(
                "Prism dSpark requires finite min_log_snr < max_log_snr",
            ));
        }

        Ok(Self {
            log_snr_fc1: nn::LinearBuilder::new(DSPARK_LOG_SNR_FEATURES, config.hidden_size)
                .bias(true)
                .build()?,
            log_snr_fc2: nn::LinearBuilder::new(config.hidden_size, config.hidden_size)
                .bias(true)
                .build()?,
            markov_head_a: Param::new(Array::zeros::<f32>(&[1, 1])?),
            markov_head_b: crate::qwen3_next::QLinear::new_spec(spec)?,
            output: (!config.reuse_target_head())
                .then(|| crate::qwen3_next::QLinear::new_spec(spec))
                .transpose()?,
            log_snr_features: build_log_snr_features(config.block_size, min_log_snr, max_log_snr)?,
            log_snr_embedding: None,
        })
    }

    fn add_log_snr(&mut self, noise: &Array) -> Result<Array, Exception> {
        if self.log_snr_embedding.is_none() {
            let features = self.log_snr_features.as_dtype(noise.dtype())?;
            let hidden = nn::silu(&self.log_snr_fc1.forward(&features)?)?;
            let embedding = self.log_snr_fc2.forward(&hidden)?.as_dtype(noise.dtype())?;
            crate::mlx_exec::eval([&embedding])?;
            self.log_snr_embedding = Some(embedding);
        }
        noise.add(
            self.log_snr_embedding
                .as_ref()
                .ok_or_else(|| Exception::custom("dSpark log-SNR embedding cache missing"))?,
        )
    }

    /// Prism's public scheduler performs a sequential low-rank Markov resample:
    /// `argmax(base[k] + B(A(prev_token)))`, chaining each sampled token into
    /// the next position. The arrays stay lazy, so the whole four-position
    /// chain is evaluated with one host barrier by the caller.
    fn propose_tokens(
        &self,
        hidden: &Array,
        anchor: i32,
        base_logits: Option<&Array>,
    ) -> Result<Array, Exception> {
        use mlx_rs::ops::indexing::IndexOp;

        let owned_logits = if base_logits.is_none() {
            Some(
                self.output
                    .as_ref()
                    .ok_or_else(|| {
                        Exception::custom(
                            "dSpark sidecar reuses the target head; target logits are required",
                        )
                    })?
                    .forward(hidden)?,
            )
        } else {
            None
        };
        let resolved_logits = base_logits
            .or(owned_logits.as_ref())
            .ok_or_else(|| Exception::custom("dSpark base logits missing"))?;
        let block_size = *hidden
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("dSpark hidden must be [B, T, D]"))?;
        let vocab_size = *resolved_logits
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("dSpark logits must have a vocabulary axis"))?;
        let mut previous = Array::from_slice(&[anchor], &[1]);
        let mut sampled = Vec::with_capacity(usize::try_from(block_size).unwrap_or(0));

        for position in 0..block_size {
            let base = resolved_logits
                .index((.., position..position + 1, ..))
                .reshape(&[-1, vocab_size])?;
            let markov_embedding = (*self.markov_head_a).take_axis(&previous, 0)?;
            let markov_bias = self.markov_head_b.forward(&markov_embedding)?;
            let logits = base.add(&markov_bias)?;
            previous = mlx_rs::argmax_axis!(&logits, -1)?;
            sampled.push(previous.clone());
        }

        let refs: Vec<&Array> = sampled.iter().collect();
        ops::concatenate_axis(&refs, 0)?.reshape(&[1, block_size])
    }
}

fn build_log_snr_features(
    block_size: i32,
    min_log_snr: f32,
    max_log_snr: f32,
) -> Result<Array, Exception> {
    if block_size <= 0 {
        return Err(Exception::custom("dSpark block_size must be positive"));
    }
    let half = DSPARK_LOG_SNR_FEATURES / 2;
    let capacity = usize::try_from(block_size * DSPARK_LOG_SNR_FEATURES)
        .map_err(|_| Exception::custom("dSpark log-SNR feature shape overflow"))?;
    let mut features = Vec::with_capacity(capacity);
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    for position in 0..block_size {
        let log_snr = if position % block_size == 0 {
            max_log_snr
        } else {
            min_log_snr
        };
        let timestep = (log_snr - min_log_snr) / (max_log_snr - min_log_snr) * 1000.0;
        for index in 0..half {
            let frequency = (-10000.0_f32.ln() * index as f32 / half as f32).exp();
            features.push((timestep * frequency).sin());
        }
        for index in 0..half {
            let frequency = (-10000.0_f32.ln() * index as f32 / half as f32).exp();
            features.push((timestep * frequency).cos());
        }
    }
    Ok(Array::from_slice(
        &features,
        &[1, block_size, DSPARK_LOG_SNR_FEATURES],
    ))
}

// ---------------------------------------------------------------------------
// DFlash drafter (top-level)
// ---------------------------------------------------------------------------

/// `DFlash` block-diffusion drafter.
///
/// Produces `block_size` (16) draft tokens per round. Does NOT have its own
/// `embed_tokens` or `lm_head` — uses the target model's `lm_head` on the output.
#[derive(Debug, ModuleParameters)]
pub struct DFlashDrafter {
    #[param]
    fc: crate::qwen3_next::QLinear,
    #[param]
    hidden_norm: nn::RmsNorm,
    #[param]
    layers: Vec<DFlashDecoderLayer>,
    #[param]
    norm: nn::RmsNorm,
    #[param]
    dspark: Option<DsparkExtras>,
    pub config: DFlashConfig,
}

/// Transactional per-layer context owned by a DFlash/dSpark drafter.
///
/// `position` is the absolute number of target-context rows consumed.  It is
/// deliberately independent of retained KV length: sliding-attention layers
/// evict old rows, so their tensor length is not a valid RoPE position.
#[derive(Debug)]
pub struct DFlashCache {
    layers: Vec<Option<(Array, Array)>>,
    /// Raw concatenated target taps not yet projected into a complete fixed
    /// context tile. Bounded to fewer than `DSPARK_CONTEXT_TILE_ROWS` rows.
    pending_taps: Option<Array>,
    /// Total target rows ingested, including `pending_taps`.
    position: i32,
    /// Unique identity for this live branch. A staged transaction may commit
    /// only to the exact branch it was created from.
    branch: DFlashBranchId,
    /// Monotonic mutation counter within one live branch.
    revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DFlashBranchId(u64);

static NEXT_DFLASH_BRANCH_ID: AtomicU64 = AtomicU64::new(1);

fn try_next_dflash_branch_id(counter: &AtomicU64) -> Option<DFlashBranchId> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .ok()
        .map(DFlashBranchId)
}

fn next_dflash_branch_id() -> DFlashBranchId {
    let Some(branch) = try_next_dflash_branch_id(&NEXT_DFLASH_BRANCH_ID) else {
        // Wrapping would allow a stale transaction to alias a live branch.
        // Exhausting u64 identities is unrecoverable, so fail closed.
        std::process::abort();
    };
    branch
}

impl Default for DFlashCache {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            pending_taps: None,
            position: 0,
            branch: next_dflash_branch_id(),
            revision: 0,
        }
    }
}

/// Immutable, evaluated drafter context at one exact target-token boundary.
///
/// Snapshots expose no forward or mutation API. A session may move the stored
/// arrays into a fresh live branch with [`Self::into_live`]; radix reuse calls
/// [`Self::fork_live`] to obtain an independent device-side copy.
#[derive(Debug)]
pub struct DFlashSnapshot {
    layers: Vec<Option<(Array, Array)>>,
    pending_taps: Option<Array>,
    position: i32,
}

impl DFlashSnapshot {
    #[must_use]
    pub const fn position(&self) -> i32 {
        self.position
    }

    /// Estimated device bytes retained by this immutable snapshot.
    ///
    /// MLX reports each array's logical buffer size. The estimate deliberately
    /// excludes small Rust-side vector/enum metadata.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        let layer_bytes = self.layers.iter().fold(0usize, |total, layer| {
            layer.as_ref().map_or(total, |(keys, values)| {
                total
                    .saturating_add(keys.nbytes())
                    .saturating_add(values.nbytes())
            })
        });
        self.pending_taps.as_ref().map_or(layer_bytes, |pending| {
            layer_bytes.saturating_add(pending.nbytes())
        })
    }

    /// Move this uniquely-owned snapshot into a new live branch without copying
    /// its evaluated arrays.
    #[must_use]
    pub fn into_live(self) -> DFlashCache {
        DFlashCache {
            layers: self.layers,
            pending_taps: self.pending_taps,
            position: self.position,
            branch: next_dflash_branch_id(),
            revision: 0,
        }
    }

    /// Deep-copy this immutable snapshot into an independent live branch.
    ///
    /// Every MLX array is copied and evaluated before publication so later
    /// mutation or buffer donation in the live branch cannot affect the stored
    /// snapshot.
    pub fn fork_live(&self) -> Result<DFlashCache, Exception> {
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                layer.as_ref().map_or(Ok(None), |(keys, values)| {
                    Ok(Some((
                        try_eval_device_copy(keys)?,
                        try_eval_device_copy(values)?,
                    )))
                })
            })
            .collect::<Result<Vec<_>, Exception>>()?;
        let pending_taps = self
            .pending_taps
            .as_ref()
            .map(try_eval_device_copy)
            .transpose()?;
        Ok(DFlashCache {
            layers,
            pending_taps,
            position: self.position,
            branch: next_dflash_branch_id(),
            revision: 0,
        })
    }
}

#[allow(unsafe_code)]
fn try_eval_device_copy(array: &Array) -> Result<Array, Exception> {
    let mut result = unsafe { mlx_sys::mlx_array_new() };
    let status = unsafe {
        mlx_sys::mlx_copy(
            &raw mut result,
            array.as_ptr(),
            mlx_rs::Stream::task_local_or_default().as_ptr(),
        )
    };
    if status != 0 {
        unsafe { mlx_sys::mlx_array_free(result) };
        return Err(Exception::custom(format!(
            "failed to copy drafter snapshot array: MLX status {status}"
        )));
    }
    let copy = unsafe { Array::from_ptr(result) };
    copy.eval()?;
    Ok(copy)
}

/// A lazy drafter forward whose cache update has not been published yet.
///
/// The proposal and target-verification graphs may evaluate `hidden` before
/// committing this transaction. If graph construction or evaluation fails,
/// dropping the value leaves the live drafter cache unchanged.
#[derive(Debug)]
pub struct DFlashForwardTransaction {
    hidden: Array,
    layers: Vec<Option<(Array, Array)>>,
    base_branch: DFlashBranchId,
    base_revision: u64,
    base_position: i32,
    appended_rows: i32,
}

impl DFlashForwardTransaction {
    #[must_use]
    pub const fn hidden(&self) -> &Array {
        &self.hidden
    }

    pub fn position(&self) -> Result<i32, Exception> {
        self.base_position
            .checked_add(self.appended_rows)
            .ok_or_else(|| Exception::custom("drafter staged context position overflow"))
    }

    pub fn commit(self, cache: &mut DFlashCache) -> Result<(), Exception> {
        if cache.branch != self.base_branch {
            return Err(Exception::custom(format!(
                "stale drafter transaction branch: staged={} live={}",
                self.base_branch.0, cache.branch.0
            )));
        }
        if cache.revision != self.base_revision {
            return Err(Exception::custom(format!(
                "stale drafter transaction revision: staged={} live={}",
                self.base_revision, cache.revision
            )));
        }
        if cache.position != self.base_position {
            return Err(Exception::custom(format!(
                "stale drafter transaction: base={} live={}",
                self.base_position, cache.position
            )));
        }
        cache.commit(self.layers, self.appended_rows, None)
    }
}

impl DFlashCache {
    #[must_use]
    pub const fn position(&self) -> i32 {
        self.position
    }

    fn pending_rows(&self) -> Result<i32, Exception> {
        self.pending_taps.as_ref().map_or(Ok(0), |pending| {
            pending
                .shape()
                .get(1)
                .copied()
                .ok_or_else(|| Exception::custom("drafter pending taps have no row axis"))
        })
    }

    fn projected_position(&self) -> Result<i32, Exception> {
        self.position
            .checked_sub(self.pending_rows()?)
            .ok_or_else(|| Exception::custom("drafter pending rows exceed absolute position"))
    }

    fn staged_layers(&self, expected: usize) -> Result<Vec<Option<(Array, Array)>>, Exception> {
        if self.layers.len() != expected {
            return Err(Exception::custom(format!(
                "drafter cache has {} layers, expected {expected}",
                self.layers.len()
            )));
        }
        Ok(self.layers.clone())
    }

    fn commit(
        &mut self,
        layers: Vec<Option<(Array, Array)>>,
        appended_rows: i32,
        pending_taps: Option<Array>,
    ) -> Result<(), Exception> {
        let position = self
            .position
            .checked_add(appended_rows)
            .ok_or_else(|| Exception::custom("drafter absolute context position overflow"))?;
        let revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| Exception::custom("drafter cache revision overflow"))?;
        self.layers = layers;
        self.pending_taps = pending_taps;
        self.position = position;
        self.revision = revision;
        Ok(())
    }

    fn validate_at_boundary(
        &self,
        expected_position: i32,
        expected_layers: usize,
        expected_pending_width: i32,
    ) -> Result<(), Exception> {
        if expected_position < 0 {
            return Err(Exception::custom(format!(
                "drafter boundary must be non-negative, got {expected_position}"
            )));
        }
        if self.position != expected_position {
            return Err(Exception::custom(format!(
                "drafter boundary mismatch: cache={} expected={expected_position}",
                self.position
            )));
        }
        if self.layers.len() != expected_layers {
            return Err(Exception::custom(format!(
                "drafter cache has {} layers, expected {expected_layers}",
                self.layers.len()
            )));
        }
        let pending_rows = self.pending_rows()?;
        if !(0..DFLASH_CONTEXT_TILE_ROWS).contains(&pending_rows) {
            return Err(Exception::custom(format!(
                "drafter pending tail has {pending_rows} rows, expected 0..{}",
                DFLASH_CONTEXT_TILE_ROWS - 1
            )));
        }
        let projected_position = self.projected_position()?;
        if projected_position > 0 && self.layers.iter().any(Option::is_none) {
            return Err(Exception::custom(
                "drafter projected context is missing one or more layer caches",
            ));
        }
        if let Some(pending) = self.pending_taps.as_ref() {
            let shape = pending.shape();
            if shape.len() != 3
                || shape[0] <= 0
                || shape[1] != pending_rows
                || shape[2] != expected_pending_width
            {
                return Err(Exception::custom(format!(
                    "drafter pending tail shape mismatch: expected [B, {pending_rows}, {expected_pending_width}], got {shape:?}"
                )));
            }
        }
        Ok(())
    }
}

/// Context projection granularity is part of the dSpark numerical contract.
/// Raw tap rows carry across outer target-prefill chunks until a full tile is
/// available, so changing the memory-oriented prefill chunk size cannot select
/// a different Q4 projection schedule or alter proposal quality.
const DFLASH_CONTEXT_TILE_ROWS: i32 = 32;

impl DFlashDrafter {
    pub fn new(config: DFlashConfig) -> Result<Self, Exception> {
        if config.is_dspark()
            && config.dflash_config.tap_semantics != Some(DFlashTapSemantics::PostLayerResidualV1)
        {
            return Err(Exception::custom(
                "Prism dSpark requires tap_semantics=post_layer_residual_v1",
            ));
        }
        let spec = config.quant_spec();
        let _fc_in = i32::try_from(config.num_taps())
            .map_err(|e| Exception::custom(format!("num_taps too large for i32: {e}")))?
            * config.hidden_size;
        let layers = (0..config.num_hidden_layers)
            .map(|i| DFlashDecoderLayer::new(&config, usize::try_from(i).unwrap_or(0), spec))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            fc: crate::qwen3_next::QLinear::new_spec(spec)?,
            hidden_norm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size)
                .eps(config.rms_norm_eps)
                .build()?,
            dspark: config
                .is_dspark()
                .then(|| DsparkExtras::new(&config, spec))
                .transpose()?,
            config,
        })
    }

    /// Create an empty per-layer KV cache for the drafter.
    pub fn make_cache(&self) -> DFlashCache {
        DFlashCache {
            layers: vec![None; self.layers.len()],
            pending_taps: None,
            position: 0,
            branch: next_dflash_branch_id(),
            revision: 0,
        }
    }

    fn validate_context_taps(&self, taps: &[Array], operation: &str) -> Result<i32, Exception> {
        if taps.len() != self.config.num_taps() {
            return Err(Exception::custom(format!(
                "drafter {operation} transaction mismatch: taps={}/{}",
                taps.len(),
                self.config.num_taps()
            )));
        }
        let Some(first) = taps.first() else {
            return Err(Exception::custom(format!(
                "drafter {operation} requires at least one configured tap"
            )));
        };
        let shape = first.shape();
        if shape.len() != 3 || shape[1] <= 0 || shape[2] != self.config.hidden_size {
            return Err(Exception::custom(format!(
                "drafter {operation} tap must be [B, T, {}] with T > 0, got {shape:?}",
                self.config.hidden_size
            )));
        }
        let batch = shape[0];
        let rows = shape[1];
        for (index, tap) in taps.iter().enumerate().skip(1) {
            if tap.shape() != [batch, rows, self.config.hidden_size] {
                return Err(Exception::custom(format!(
                    "drafter {operation} tap {index} shape mismatch: expected [{batch}, {rows}, {}], got {:?}",
                    self.config.hidden_size,
                    tap.shape()
                )));
            }
        }
        Ok(rows)
    }

    fn validate_noise(&self, noise: &Array, taps: &[Array]) -> Result<(), Exception> {
        let batch = taps
            .first()
            .and_then(|tap| tap.shape().first())
            .copied()
            .ok_or_else(|| Exception::custom("drafter noise validation requires target taps"))?;
        let expected = [batch, self.config.block_size, self.config.hidden_size];
        if noise.shape() != expected {
            return Err(Exception::custom(format!(
                "drafter noise must be [B, block_size, hidden] = {expected:?}, got {:?}",
                noise.shape()
            )));
        }
        Ok(())
    }

    fn validate_cache_layer_shapes(&self, cache: &DFlashCache) -> Result<(), Exception> {
        let projected_position = cache.projected_position()?;
        let mut expected_batch = cache
            .pending_taps
            .as_ref()
            .and_then(|pending| pending.shape().first().copied());

        for (index, (state, layer)) in cache.layers.iter().zip(&self.layers).enumerate() {
            let Some((keys, values)) = state else {
                continue;
            };
            let key_shape = keys.shape();
            let value_shape = values.shape();
            if key_shape.len() != 4 || value_shape.len() != 4 {
                return Err(Exception::custom(format!(
                    "drafter layer {index} cache must be rank 4 [B, Hkv, T, D], got keys={key_shape:?} values={value_shape:?}"
                )));
            }
            if key_shape[0] <= 0 || value_shape[0] <= 0 {
                return Err(Exception::custom(format!(
                    "drafter layer {index} cache batch must be positive, got keys={} values={}",
                    key_shape[0], value_shape[0]
                )));
            }
            if key_shape[0] != value_shape[0] {
                return Err(Exception::custom(format!(
                    "drafter layer {index} key/value batch mismatch: {} vs {}",
                    key_shape[0], value_shape[0]
                )));
            }
            match expected_batch {
                Some(batch) if key_shape[0] != batch => {
                    return Err(Exception::custom(format!(
                        "drafter layer {index} cache batch mismatch: expected {batch}, got {}",
                        key_shape[0]
                    )));
                }
                None => expected_batch = Some(key_shape[0]),
                Some(_) => {}
            }

            let attention = &layer.self_attn;
            let retained = if attention.is_sliding && attention.sliding_window > 1 {
                projected_position.min(attention.sliding_window - 1)
            } else {
                projected_position
            };
            let expected_shape = [
                key_shape[0],
                attention.num_key_value_heads,
                retained,
                attention.head_dim,
            ];
            if key_shape != expected_shape || value_shape != expected_shape {
                return Err(Exception::custom(format!(
                    "drafter layer {index} cache shape mismatch: expected {expected_shape:?}, got keys={key_shape:?} values={value_shape:?}"
                )));
            }
        }
        Ok(())
    }

    fn merge_pending_taps(cache: &DFlashCache, target_cat: &Array) -> Result<Array, Exception> {
        cache.pending_taps.as_ref().map_or_else(
            || Ok(target_cat.clone()),
            |pending| ops::concatenate_axis(&[pending, target_cat], 1),
        )
    }

    fn append_context_tiles(
        &mut self,
        raw_taps: &Array,
        staged: &mut [Option<(Array, Array)>],
        start_position: i32,
        flush_remainder: bool,
    ) -> Result<Option<Array>, Exception> {
        let rows = raw_taps
            .shape()
            .get(1)
            .copied()
            .ok_or_else(|| Exception::custom("drafter raw taps have no row axis"))?;
        let processed = if flush_remainder {
            rows
        } else {
            rows - rows.rem_euclid(DFLASH_CONTEXT_TILE_ROWS)
        };
        let mut offset = 0_i32;
        while offset < processed {
            let end = (offset + DFLASH_CONTEXT_TILE_ROWS).min(processed);
            let tile = raw_taps.index((.., offset..end, ..));
            let projected = self.fc.forward(&tile)?;
            let target_hidden = self.hidden_norm.forward(&projected)?;
            let absolute = start_position
                .checked_add(offset)
                .ok_or_else(|| Exception::custom("drafter context tile position overflow"))?;
            for (layer, layer_cache) in self.layers.iter_mut().zip(staged.iter_mut()) {
                layer.prime_target_context(&target_hidden, layer_cache, absolute)?;
            }
            offset = end;
        }
        if processed < rows {
            Ok(Some(raw_taps.index((.., processed.., ..))))
        } else {
            Ok(None)
        }
    }

    /// Append target tap context to every drafter layer without running the
    /// draft trunk. Used by chunked target prefill to keep peak tap memory
    /// bounded by one chunk.
    pub fn prime_taps(&mut self, taps: &[Array], cache: &mut DFlashCache) -> Result<(), Exception> {
        let rows = self.validate_context_taps(taps, "prime")?;
        let projected_position = cache.projected_position()?;
        let mut staged = cache.staged_layers(self.layers.len())?;
        let tap_refs: Vec<&Array> = taps.iter().collect();
        let target_cat = ops::concatenate_axis(&tap_refs, -1)?;
        let combined = Self::merge_pending_taps(cache, &target_cat)?;
        let pending =
            self.append_context_tiles(&combined, &mut staged, projected_position, false)?;
        Self::eval_parts(&staged, pending.as_ref())?;
        cache.commit(staged, rows, pending)?;
        Ok(())
    }

    /// Materialize a primed context cache between prompt chunks so lazy graphs
    /// cannot retain the entire target prefill.
    fn eval_parts(
        layers: &[Option<(Array, Array)>],
        pending_taps: Option<&Array>,
    ) -> Result<(), Exception> {
        let mut targets = layers
            .iter()
            .flatten()
            .flat_map(|(keys, values)| [keys, values])
            .collect::<Vec<_>>();
        targets.extend(pending_taps);
        mlx_rs::transforms::eval(targets)
    }

    pub fn eval_cache(cache: &DFlashCache) -> Result<(), Exception> {
        Self::eval_parts(&cache.layers, cache.pending_taps.as_ref())
    }

    /// Consume a live drafter cache and publish one immutable boundary snapshot.
    ///
    /// `external_taps` is the engine-owned tap frontier not yet represented in
    /// `cache`. It is appended through [`Self::prime_taps`], which projects only
    /// complete 32-row tiles and intentionally retains the final sub-tile tail.
    /// Empty `external_taps` means the cache is already caught up.
    pub fn seal_after_taps(
        &mut self,
        mut cache: DFlashCache,
        external_taps: &[Array],
        expected_position: i32,
    ) -> Result<DFlashSnapshot, Exception> {
        if expected_position < 0 {
            return Err(Exception::custom(format!(
                "drafter boundary must be non-negative, got {expected_position}"
            )));
        }
        if !external_taps.is_empty() {
            self.prime_taps(external_taps, &mut cache)?;
        }
        let pending_width = i32::try_from(self.config.num_taps())
            .map_err(|_| Exception::custom("drafter tap count exceeds i32"))?
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| Exception::custom("drafter pending tap width overflow"))?;
        cache.validate_at_boundary(expected_position, self.layers.len(), pending_width)?;
        self.validate_cache_layer_shapes(&cache)?;
        Self::eval_cache(&cache)?;
        Ok(DFlashSnapshot {
            layers: cache.layers,
            pending_taps: cache.pending_taps,
            position: cache.position,
        })
    }

    /// Initialize all `QLinear` weights to zero at the correct shapes.
    /// Only for tests — `QLinear` (unlike `nn::Linear`) starts with [1] placeholders.
    #[cfg(test)]
    #[allow(clippy::unwrap_used)]
    pub fn init_test_weights(&mut self) {
        use mlx_rs::module::Param;
        let h = self.config.hidden_size;
        let n_heads = self.config.num_attention_heads;
        let n_kv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;
        let inter = self.config.intermediate_size;
        let fc_in = i32::try_from(self.config.num_taps()).unwrap_or(1) * h;
        let qo = n_heads * hd;
        let kv = n_kv * hd;
        for layer in &mut self.layers {
            layer.self_attn.q_proj.weight = Param::new(Array::zeros::<f32>(&[qo, h]).unwrap());
            layer.self_attn.k_proj.weight = Param::new(Array::zeros::<f32>(&[kv, h]).unwrap());
            layer.self_attn.v_proj.weight = Param::new(Array::zeros::<f32>(&[kv, h]).unwrap());
            layer.self_attn.o_proj.weight = Param::new(Array::zeros::<f32>(&[h, qo]).unwrap());
            layer.mlp.gate_proj.weight = Param::new(Array::zeros::<f32>(&[inter, h]).unwrap());
            layer.mlp.up_proj.weight = Param::new(Array::zeros::<f32>(&[inter, h]).unwrap());
            layer.mlp.down_proj.weight = Param::new(Array::zeros::<f32>(&[h, inter]).unwrap());
        }
        self.fc.weight = Param::new(Array::zeros::<f32>(&[h, fc_in]).unwrap());
    }

    /// Run the drafter forward pass.
    ///
    /// - `noise`: `[B, block_size, hidden_size]` — embedded block tokens.
    /// - `taps`: slice of hidden states from the target model at tap layers,
    ///   each `[B, T, target_hidden_size]`. Concatenated along the last dim,
    ///   projected via `fc`, then normalized.
    /// - `cache`: transactional fixed-tile context cache.
    ///
    /// Returns `[B, block_size, hidden_size]` — pass to target's `lm_head` for logits.
    #[allow(non_snake_case)]
    pub fn stage_forward(
        &mut self,
        noise: &Array,
        taps: &[Array],
        cache: &DFlashCache,
    ) -> Result<DFlashForwardTransaction, Exception> {
        let rows = self.validate_context_taps(taps, "forward")?;
        self.validate_noise(noise, taps)?;
        let projected_position = cache.projected_position()?;
        let mut staged = cache.staged_layers(self.layers.len())?;

        let tap_refs: Vec<&Array> = taps.iter().collect();
        let target_cat = ops::concatenate_axis(&tap_refs, -1)?;
        let combined = Self::merge_pending_taps(cache, &target_cat)?;
        let pending =
            self.append_context_tiles(&combined, &mut staged, projected_position, true)?;
        if pending.is_some() {
            return Err(Exception::custom(
                "drafter forward failed to flush its context-tile remainder",
            ));
        }
        let noise_position = cache
            .position
            .checked_add(rows)
            .ok_or_else(|| Exception::custom("drafter noise position overflow"))?;

        let mut h = match self.dspark.as_mut() {
            Some(dspark) => dspark.add_log_snr(noise)?,
            None => noise.clone(),
        };
        for (layer, lc) in self.layers.iter_mut().zip(staged.iter_mut()) {
            h = layer.forward_noise(&h, lc, noise_position)?;
        }
        let hidden = self.norm.forward(&h)?;
        Ok(DFlashForwardTransaction {
            hidden,
            layers: staged,
            base_branch: cache.branch,
            base_revision: cache.revision,
            base_position: cache.position,
            appended_rows: rows,
        })
    }

    /// Materialized convenience path for callers that do not fuse proposal
    /// evaluation with target verification. Production speculative decoding
    /// uses [`Self::stage_forward`] and commits after its synchronization
    /// barrier instead.
    pub fn forward(
        &mut self,
        noise: &Array,
        taps: &[Array],
        cache: &mut DFlashCache,
    ) -> Result<Array, Exception> {
        let transaction = self.stage_forward(noise, taps, cache)?;
        let mut targets = transaction
            .layers
            .iter()
            .flatten()
            .flat_map(|(keys, values)| [keys, values])
            .collect::<Vec<_>>();
        targets.push(transaction.hidden());
        mlx_rs::transforms::eval(targets)?;
        let hidden = transaction.hidden.clone();
        transaction.commit(cache)?;
        Ok(hidden)
    }

    /// Produce Prism dSpark draft tokens from the normalized trunk output.
    /// Returns `None` for ordinary Modal `DFlash` checkpoints.
    pub fn propose_dspark_tokens(
        &self,
        hidden: &Array,
        anchor: i32,
        base_logits: Option<&Array>,
    ) -> Result<Option<Array>, Exception> {
        self.dspark
            .as_ref()
            .map(|dspark| dspark.propose_tokens(hidden, anchor, base_logits))
            .transpose()
    }
}

// ---------------------------------------------------------------------------
// GDN state save/restore for hybrid models (Qwen3.5)
// ---------------------------------------------------------------------------

/// Saved state for all GDN/linear-attention layers in the target model.
///
/// Much smaller than cloning the full KV cache — only stores `conv_state`,
/// `ssm_state`, and offset for each `ArraysCache` layer.
pub struct GdnStateBackup {
    states: Vec<(Option<Array>, Option<Array>, i32)>,
}

impl GdnStateBackup {
    /// Save GDN (`ArraysCache`) state from all layers. Call BEFORE verify forward.
    /// KV layers are not saved — they use cheap offset-based rollback instead.
    pub fn save(kv_cache: &[Option<crate::qwen3_next::LayerCache>]) -> Result<Self, Exception> {
        let mut states = Vec::with_capacity(kv_cache.len());
        for lc in kv_cache {
            match lc {
                Some(crate::qwen3_next::LayerCache::Arrays(ac)) => {
                    ac.eval_arrays()?;
                    states.push((ac.conv_state.clone(), ac.ssm_state.clone(), ac.offset));
                }
                _ => states.push((None, None, 0)),
            }
        }
        Ok(Self { states })
    }

    /// Restore GDN state and rollback KV offsets. On rejection, call this
    /// BEFORE re-running the accepted tokens.
    pub fn restore_and_rollback(
        &self,
        kv_cache: &mut [Option<crate::qwen3_next::LayerCache>],
        rollback: i32,
    ) {
        for (lc, (conv, ssm, offset)) in kv_cache.iter_mut().zip(self.states.iter()) {
            match lc {
                Some(crate::qwen3_next::LayerCache::Arrays(ac)) => {
                    ac.conv_state.clone_from(conv);
                    ac.ssm_state.clone_from(ssm);
                    ac.offset = *offset;
                }
                Some(crate::qwen3_next::LayerCache::KV(kv)) if rollback > 0 => {
                    kv.trim_by(rollback.unsigned_abs().try_into().unwrap_or(usize::MAX));
                }
                Some(crate::qwen3_next::LayerCache::KV(_)) | None => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

fn validate_shape(key: &str, actual: &Array, expected: &[i32]) -> Result<(), ModelError> {
    if actual.shape() == expected {
        return Ok(());
    }
    Err(ModelError::ShapeMismatch(format!(
        "DFlash tensor {key}: expected {expected:?}, got {:?}",
        actual.shape()
    )))
}

fn validate_qlinear_shape(
    key: &str,
    linear: &crate::qwen3_next::QLinear,
    out_features: i32,
    in_features: i32,
) -> Result<(), ModelError> {
    if linear.mode.is_dense() {
        return validate_shape(
            &format!("{key}.weight"),
            &linear.weight,
            &[out_features, in_features],
        );
    }
    if linear.bits <= 0 || linear.group_size <= 0 || in_features % linear.group_size != 0 {
        return Err(ModelError::ShapeMismatch(format!(
            "DFlash tensor {key}: invalid quantization bits={}, group_size={} for input {in_features}",
            linear.bits, linear.group_size
        )));
    }
    let packed_bits = in_features.checked_mul(linear.bits).ok_or_else(|| {
        ModelError::ShapeMismatch(format!("DFlash tensor {key}: packed shape overflow"))
    })?;
    if packed_bits % 32 != 0 {
        return Err(ModelError::ShapeMismatch(format!(
            "DFlash tensor {key}: {in_features} inputs at {} bits do not fill u32 words",
            linear.bits
        )));
    }
    let packed_columns = packed_bits / 32;
    let groups = in_features / linear.group_size;
    validate_shape(
        &format!("{key}.weight"),
        &linear.weight,
        &[out_features, packed_columns],
    )?;
    validate_shape(
        &format!("{key}.scales"),
        &linear.scales,
        &[out_features, groups],
    )?;
    if !linear.mode.is_mxfp4() {
        validate_shape(
            &format!("{key}.biases"),
            &linear.biases,
            &[out_features, groups],
        )?;
    }
    Ok(())
}

fn validate_loaded_shapes(drafter: &DFlashDrafter) -> Result<(), ModelError> {
    let config = &drafter.config;
    let hidden = config.hidden_size;
    let query_width = config
        .num_attention_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| ModelError::ShapeMismatch("DFlash query width overflow".to_owned()))?;
    let kv_width = config
        .num_key_value_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| ModelError::ShapeMismatch("DFlash KV width overflow".to_owned()))?;
    let tap_count = i32::try_from(config.num_taps())
        .map_err(|_| ModelError::ShapeMismatch("DFlash tap count overflow".to_owned()))?;
    let tap_width = tap_count
        .checked_mul(hidden)
        .ok_or_else(|| ModelError::ShapeMismatch("DFlash tap width overflow".to_owned()))?;

    validate_qlinear_shape("fc", &drafter.fc, hidden, tap_width)?;
    for (index, layer) in drafter.layers.iter().enumerate() {
        let prefix = format!("layers.{index}");
        validate_qlinear_shape(
            &format!("{prefix}.self_attn.q_proj"),
            &layer.self_attn.q_proj,
            query_width,
            hidden,
        )?;
        validate_qlinear_shape(
            &format!("{prefix}.self_attn.k_proj"),
            &layer.self_attn.k_proj,
            kv_width,
            hidden,
        )?;
        validate_qlinear_shape(
            &format!("{prefix}.self_attn.v_proj"),
            &layer.self_attn.v_proj,
            kv_width,
            hidden,
        )?;
        validate_qlinear_shape(
            &format!("{prefix}.self_attn.o_proj"),
            &layer.self_attn.o_proj,
            hidden,
            query_width,
        )?;
        validate_qlinear_shape(
            &format!("{prefix}.mlp.gate_proj"),
            &layer.mlp.gate_proj,
            config.intermediate_size,
            hidden,
        )?;
        validate_qlinear_shape(
            &format!("{prefix}.mlp.up_proj"),
            &layer.mlp.up_proj,
            config.intermediate_size,
            hidden,
        )?;
        validate_qlinear_shape(
            &format!("{prefix}.mlp.down_proj"),
            &layer.mlp.down_proj,
            hidden,
            config.intermediate_size,
        )?;
    }

    if let Some(dspark) = drafter.dspark.as_ref() {
        let rank = config.dflash_config.markov_rank;
        validate_shape(
            "dspark.markov_head_a",
            &dspark.markov_head_a,
            &[config.vocab_size, rank],
        )?;
        validate_qlinear_shape(
            "dspark.markov_head_b",
            &dspark.markov_head_b,
            config.vocab_size,
            rank,
        )?;
        if let Some(output) = dspark.output.as_ref() {
            validate_qlinear_shape("dspark.output", output, config.vocab_size, hidden)?;
        }
    }
    Ok(())
}

fn load_dflash_weights(drafter: &mut DFlashDrafter, model_path: &Path) -> Result<(), ModelError> {
    let expected_shapes: HashMap<String, Vec<i32>> = drafter
        .parameters()
        .flatten()
        .iter()
        .filter(|(_, value)| value.shape().iter().product::<i32>() > 0)
        .map(|(name, value)| (name.to_string(), value.shape().to_vec()))
        .collect();
    let mut missing: std::collections::HashSet<String> = expected_shapes.keys().cloned().collect();
    let mut params = drafter.parameters_mut().flatten();

    for file_path in crate::collect_safetensors_files(model_path)? {
        tracing::debug!(file = %file_path.display(), "Loading DFlash weights");
        for (key, value) in Array::load_safetensors(&file_path)? {
            let Some(param) = params.get_mut(&*key) else {
                tracing::warn!(key = %key, "Weight key not found in DFlash parameters");
                continue;
            };
            if let Some(expected) = expected_shapes.get(&*key)
                && expected.iter().product::<i32>() != 1
                && value.shape() != expected.as_slice()
            {
                return Err(ModelError::ShapeMismatch(format!(
                    "DFlash tensor {key}: expected {expected:?}, got {:?}",
                    value.shape()
                )));
            }
            **param = value;
            missing.remove(&*key);
        }
    }

    if !missing.is_empty() {
        let mut names: Vec<_> = missing.into_iter().collect();
        names.sort_unstable();
        let examples = names
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        return Err(ModelError::MissingWeight(format!(
            "{} DFlash parameters were not loaded; examples: {examples}",
            names.len()
        )));
    }
    validate_loaded_shapes(drafter)?;
    drafter.eval()?;
    Ok(())
}

const DSPARK_TARGET_BINDING_FORMAT: &str = "higgs-target-artifact-v1";

fn target_binding_error(message: impl Into<String>) -> ModelError {
    ModelError::UnsupportedModel(format!("dSpark target binding: {}", message.into()))
}

fn binding_relative_path(root: &Path, file: &Path) -> Result<String, ModelError> {
    let relative = file.strip_prefix(root).map_err(|_| {
        target_binding_error(format!(
            "{} is outside target directory {}",
            file.display(),
            root.display()
        ))
    })?;
    let mut parts = Vec::new();
    for component in relative.components() {
        let Component::Normal(part) = component else {
            return Err(target_binding_error(format!(
                "non-normal target artifact path {}",
                relative.display()
            )));
        };
        parts.push(
            part.to_str()
                .ok_or_else(|| target_binding_error("target artifact path is not UTF-8"))?,
        );
    }
    if parts.is_empty() {
        return Err(target_binding_error("target artifact path is empty"));
    }
    Ok(parts.join("/"))
}

fn selected_target_artifacts(target_path: &Path) -> Result<Vec<(String, PathBuf)>, ModelError> {
    let mut files = vec![target_path.join("config.json")];
    files.extend(crate::collect_base_safetensors_files(target_path)?);
    let mut artifacts = files
        .into_iter()
        .map(|path| binding_relative_path(target_path, &path).map(|name| (name, path)))
        .collect::<Result<Vec<_>, _>>()?;
    artifacts.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(artifacts)
}

fn sha256_file(path: &Path) -> Result<(u64, String), ModelError> {
    let file = File::open(path).map_err(|error| {
        ModelError::Io(std::io::Error::other(format!(
            "opening target artifact {}: {error}",
            path.display()
        )))
    })?;
    let size = file.metadata()?.len();
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut digest = Sha256::new();
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(buffer.get(..count).ok_or_else(|| {
            ModelError::Io(std::io::Error::other("SHA-256 read buffer bounds failure"))
        })?);
    }
    Ok((size, format!("{:x}", digest.finalize())))
}

fn verify_dspark_target_binding(
    config: &DFlashConfig,
    target_path: &Path,
) -> Result<(), ModelError> {
    let binding = config
        .dflash_config
        .target_binding
        .as_ref()
        .ok_or_else(|| target_binding_error("missing target_binding manifest"))?;
    if binding.format != DSPARK_TARGET_BINDING_FORMAT {
        return Err(target_binding_error(format!(
            "unsupported format {:?}",
            binding.format
        )));
    }

    let mut declared = BTreeMap::new();
    for artifact in &binding.files {
        let path = Path::new(&artifact.path);
        if path.is_absolute()
            || path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(target_binding_error(format!(
                "manifest path {:?} must be normalized and relative",
                artifact.path
            )));
        }
        if artifact.sha256.len() != 64
            || !artifact
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(target_binding_error(format!(
                "manifest SHA-256 for {:?} must be 64 lowercase hex characters",
                artifact.path
            )));
        }
        if declared.insert(artifact.path.as_str(), artifact).is_some() {
            return Err(target_binding_error(format!(
                "duplicate manifest path {:?}",
                artifact.path
            )));
        }
    }

    let selected = selected_target_artifacts(target_path)?;
    let selected_names = selected
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    let declared_names = declared.keys().copied().collect::<Vec<_>>();
    if selected_names != declared_names {
        return Err(target_binding_error(format!(
            "artifact set mismatch: selected={selected_names:?} manifest={declared_names:?}"
        )));
    }

    tracing::info!(target = %target_path.display(), files = selected.len(), "Verifying dSpark target artifact binding");
    for (name, path) in selected {
        let expected = declared
            .get(name.as_str())
            .ok_or_else(|| target_binding_error(format!("missing manifest entry {name:?}")))?;
        let (size, digest) = sha256_file(&path)?;
        if size != expected.size || digest != expected.sha256 {
            return Err(target_binding_error(format!(
                "artifact {name:?} mismatch: expected {} bytes/{}, got {size} bytes/{digest}",
                expected.size, expected.sha256
            )));
        }
    }
    Ok(())
}

fn load_dflash_drafter_inner(
    model_path: &Path,
    target_path: Option<&Path>,
) -> Result<DFlashDrafter, ModelError> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| ModelError::Io(std::io::Error::other(format!("reading config.json: {e}"))))?;
    let config: DFlashConfig = serde_json::from_str(&config_str)
        .map_err(|e| ModelError::Io(std::io::Error::other(format!("parsing config.json: {e}"))))?;

    if config.is_dspark() {
        let target_path = target_path.ok_or_else(|| {
            target_binding_error("dSpark must be loaded through the paired-target API")
        })?;
        verify_dspark_target_binding(&config, target_path)?;
    }

    let mut drafter = DFlashDrafter::new(config)
        .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

    load_dflash_weights(&mut drafter, model_path)?;

    Ok(drafter)
}

/// Load a generic DFlash drafter. Trained dSpark checkpoints fail closed here
/// because they require an explicitly attested target artifact.
pub fn load_dflash_drafter(model_path: &Path) -> Result<DFlashDrafter, ModelError> {
    load_dflash_drafter_inner(model_path, None)
}

/// Load a dSpark/DFlash drafter paired with the target checkpoint directory.
pub fn load_dflash_drafter_for_target(
    model_path: &Path,
    target_path: &Path,
) -> Result<DFlashDrafter, ModelError> {
    load_dflash_drafter_inner(model_path, Some(target_path))
}

// ---------------------------------------------------------------------------
// Speculative-decode acceptance helper
// ---------------------------------------------------------------------------

/// Greedy speculative-decode acceptance.
///
/// Compares each drafted token to the target's verified argmax. Accepts the
/// longest matching prefix and appends one bonus token (the target's argmax at
/// the position immediately after the rejected draft, or after the last accept
/// if all tokens accepted).
///
/// Returns at least 1 token, at most `draft.len() + 1`.
///
/// # Panics in debug
/// Panics if `verify_argmax.len() != draft.len() + 1`.
pub fn accept_prefix(draft: &[u32], verify_argmax: &[u32]) -> Vec<u32> {
    debug_assert_eq!(verify_argmax.len(), draft.len() + 1);

    let accepted = draft
        .iter()
        .zip(verify_argmax.iter())
        .take_while(|(d, v)| **d == **v)
        .count();
    let mut out: Vec<u32> = draft.get(..accepted).unwrap_or_default().to_vec();
    if let Some(&bonus) = verify_argmax.get(accepted) {
        out.push(bonus);
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// The full DFlash test suite (~3.8K lines, 30+ end-to-end tests covering the
// draft-verify loop against a real target model) lives on `feat/magic-canvas`
// and will be ported in a follow-up PR alongside the engine glue
// (`SimpleEngine::generate_dflash_inner`).
//
// What's tested here: only the pure helpers (`accept_prefix`) — the
// rest depends on MLX-loaded model weights which aren't part of unit-test
// surface.

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::indexing_slicing
)]
mod tests {
    use super::accept_prefix;

    fn tiny_dspark_config() -> super::DFlashConfig {
        super::DFlashConfig {
            hidden_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            intermediate_size: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 1e7,
            block_size: 2,
            vocab_size: 4,
            layer_types: None,
            sliding_window: None,
            quantization: None,
            dflash_config: super::DFlashSubConfig {
                target_layer_ids: vec![0],
                tap_semantics: Some(super::DFlashTapSemantics::PostLayerResidualV1),
                mask_token_id: Some(3),
                dspark: true,
                markov_rank: 1,
                log_snr_conditioning: true,
                min_log_snr: -9.0,
                max_log_snr: 9.0,
                reuse_target_head: true,
                target_binding: None,
            },
        }
    }

    fn tiny_dense_config(sliding_window: Option<i32>) -> super::DFlashConfig {
        super::DFlashConfig {
            hidden_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            intermediate_size: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 1e7,
            block_size: 2,
            vocab_size: 8,
            layer_types: sliding_window.map(|_| vec!["sliding_attention".to_owned()]),
            sliding_window,
            quantization: None,
            dflash_config: super::DFlashSubConfig {
                target_layer_ids: vec![0],
                tap_semantics: None,
                mask_token_id: Some(7),
                dspark: false,
                markov_rank: 0,
                log_snr_conditioning: false,
                min_log_snr: 0.0,
                max_log_snr: 0.0,
                reuse_target_head: false,
                target_binding: None,
            },
        }
    }

    fn patterned(rows: i32, columns: i32, salt: usize) -> mlx_rs::Array {
        let count = usize::try_from(rows * columns).unwrap();
        let values = (0..count)
            .map(|index| {
                let centered = i32::try_from((index * 17 + salt) % 29).unwrap() - 14;
                centered as f32 * 0.003
            })
            .collect::<Vec<_>>();
        mlx_rs::Array::from_slice(&values, &[rows, columns])
    }

    fn init_patterned_weights(drafter: &mut super::DFlashDrafter) {
        use mlx_rs::module::Param;

        let hidden = drafter.config.hidden_size;
        let intermediate = drafter.config.intermediate_size;
        let query = drafter.config.num_attention_heads * drafter.config.head_dim;
        let kv = drafter.config.num_key_value_heads * drafter.config.head_dim;
        let fc_in = i32::try_from(drafter.config.num_taps()).unwrap() * hidden;
        drafter.fc.weight = Param::new(patterned(hidden, fc_in, 1));
        for (index, layer) in drafter.layers.iter_mut().enumerate() {
            let salt = index * 31;
            layer.self_attn.q_proj.weight = Param::new(patterned(query, hidden, salt + 2));
            layer.self_attn.k_proj.weight = Param::new(patterned(kv, hidden, salt + 3));
            layer.self_attn.v_proj.weight = Param::new(patterned(kv, hidden, salt + 4));
            layer.self_attn.o_proj.weight = Param::new(patterned(hidden, query, salt + 5));
            layer.mlp.gate_proj.weight = Param::new(patterned(intermediate, hidden, salt + 6));
            layer.mlp.up_proj.weight = Param::new(patterned(intermediate, hidden, salt + 7));
            layer.mlp.down_proj.weight = Param::new(patterned(hidden, intermediate, salt + 8));
        }
    }

    fn init_patterned_q4_weights(drafter: &mut super::DFlashDrafter) {
        use mlx_rs::module::Param;

        fn install(linear: &mut crate::qwen3_next::QLinear, rows: i32, columns: i32, salt: usize) {
            let words = usize::try_from(rows * columns / 8).unwrap();
            let packed = (0..words)
                .map(|index| {
                    0xA5A5_5A5A_u32.rotate_left(u32::try_from((index + salt) % 31).unwrap())
                })
                .collect::<Vec<_>>();
            let groups = usize::try_from(rows * columns / 32).unwrap();
            let scales = (0..groups)
                .map(|index| 0.01_f32 + ((index + salt) % 7) as f32 * 0.002)
                .collect::<Vec<_>>();
            let biases = scales.iter().map(|scale| -0.5 * scale).collect::<Vec<_>>();
            linear.weight = Param::new(mlx_rs::Array::from_slice(&packed, &[rows, columns / 8]));
            linear.scales = Param::new(mlx_rs::Array::from_slice(&scales, &[rows, columns / 32]));
            linear.biases = Param::new(mlx_rs::Array::from_slice(&biases, &[rows, columns / 32]));
        }

        let hidden = drafter.config.hidden_size;
        let intermediate = drafter.config.intermediate_size;
        let query = drafter.config.num_attention_heads * drafter.config.head_dim;
        let kv = drafter.config.num_key_value_heads * drafter.config.head_dim;
        let fc_in = i32::try_from(drafter.config.num_taps()).unwrap() * hidden;
        install(&mut drafter.fc, hidden, fc_in, 1);
        for (index, layer) in drafter.layers.iter_mut().enumerate() {
            let salt = index * 31;
            install(&mut layer.self_attn.q_proj, query, hidden, salt + 2);
            install(&mut layer.self_attn.k_proj, kv, hidden, salt + 3);
            install(&mut layer.self_attn.v_proj, kv, hidden, salt + 4);
            install(&mut layer.self_attn.o_proj, hidden, query, salt + 5);
            install(&mut layer.mlp.gate_proj, intermediate, hidden, salt + 6);
            install(&mut layer.mlp.up_proj, intermediate, hidden, salt + 7);
            install(&mut layer.mlp.down_proj, hidden, intermediate, salt + 8);
        }
    }

    fn input(rows: i32, hidden: i32, salt: usize) -> mlx_rs::Array {
        let count = usize::try_from(rows * hidden).unwrap();
        let values = (0..count)
            .map(|index| ((index * 11 + salt) % 23) as f32 * 0.02 - 0.2)
            .collect::<Vec<_>>();
        mlx_rs::Array::from_slice(&values, &[1, rows, hidden])
    }

    fn assert_f32_bits_equal(left: &mlx_rs::Array, right: &mlx_rs::Array, label: &str) {
        assert_eq!(left.shape(), right.shape(), "{label} shape");
        let left_bits = left
            .as_slice::<f32>()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let right_bits = right
            .as_slice::<f32>()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(left_bits, right_bits, "{label} values");
    }

    fn assert_cache_exact(left: &super::DFlashCache, right: &super::DFlashCache, label: &str) {
        super::DFlashDrafter::eval_cache(left).unwrap();
        super::DFlashDrafter::eval_cache(right).unwrap();
        assert_eq!(left.position(), right.position(), "{label} position");
        assert_eq!(left.layers.len(), right.layers.len(), "{label} layers");
        for (index, (left_layer, right_layer)) in
            left.layers.iter().zip(right.layers.iter()).enumerate()
        {
            match (left_layer, right_layer) {
                (Some((left_k, left_v)), Some((right_k, right_v))) => {
                    assert_f32_bits_equal(left_k, right_k, &format!("{label} layer {index} keys"));
                    assert_f32_bits_equal(
                        left_v,
                        right_v,
                        &format!("{label} layer {index} values"),
                    );
                }
                (None, None) => {}
                _ => panic!("{label} layer {index} presence mismatch"),
            }
        }
        match (&left.pending_taps, &right.pending_taps) {
            (Some(left_pending), Some(right_pending)) => {
                assert_f32_bits_equal(
                    left_pending,
                    right_pending,
                    &format!("{label} pending taps"),
                );
            }
            (None, None) => {}
            _ => panic!("{label} pending-tap presence mismatch"),
        }
    }

    fn target_fixture() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("config.json"), b"target-config-v1").unwrap();
        std::fs::write(
            directory.path().join("model.safetensors"),
            b"target-weights-v1",
        )
        .unwrap();
        directory
    }

    fn fixture_binding(target: &std::path::Path) -> super::TargetArtifactBinding {
        let files = super::selected_target_artifacts(target)
            .unwrap()
            .into_iter()
            .map(|(path, file)| {
                let (size, sha256) = super::sha256_file(&file).unwrap();
                super::TargetArtifactFile { path, size, sha256 }
            })
            .collect();
        super::TargetArtifactBinding {
            format: super::DSPARK_TARGET_BINDING_FORMAT.to_owned(),
            files,
        }
    }

    #[test]
    fn accept_prefix_full_match_returns_draft_plus_bonus() {
        let draft = vec![1, 2, 3];
        let verify = vec![1, 2, 3, 4];
        assert_eq!(accept_prefix(&draft, &verify), vec![1, 2, 3, 4]);
    }

    #[test]
    fn dspark_target_binding_attests_exact_selected_artifacts() {
        let target = target_fixture();
        let mut config = tiny_dspark_config();
        config.dflash_config.target_binding = Some(fixture_binding(target.path()));

        // Optional MTP files do not alter the base-target identity.
        std::fs::write(target.path().join("mtp.safetensors"), b"optional-head").unwrap();
        super::verify_dspark_target_binding(&config, target.path()).unwrap();

        // Same file name and size is insufficient: every byte is attested.
        std::fs::write(
            target.path().join("model.safetensors"),
            b"target-weights-v2",
        )
        .unwrap();
        let error = super::verify_dspark_target_binding(&config, target.path()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("artifact \"model.safetensors\" mismatch")
        );
    }

    #[test]
    fn dspark_target_binding_fails_closed_for_missing_or_unknown_manifest() {
        let target = target_fixture();
        let config = tiny_dspark_config();
        let missing = super::verify_dspark_target_binding(&config, target.path()).unwrap_err();
        assert!(missing.to_string().contains("missing target_binding"));

        let mut unknown = config;
        let mut binding = fixture_binding(target.path());
        binding.format = "future-unreviewed-format".to_owned();
        unknown.dflash_config.target_binding = Some(binding);
        let error = super::verify_dspark_target_binding(&unknown, target.path()).unwrap_err();
        assert!(error.to_string().contains("unsupported format"));
    }

    #[test]
    fn dspark_requires_pinned_post_layer_tap_semantics() {
        let mut config = tiny_dspark_config();
        config.dflash_config.tap_semantics = None;

        let error = super::DFlashDrafter::new(config).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("tap_semantics=post_layer_residual_v1")
        );
    }

    #[test]
    fn unpaired_dspark_loader_fails_before_reading_weights() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            r#"{
                "model_type":"dspark",
                "hidden_size":4,
                "num_hidden_layers":1,
                "num_attention_heads":1,
                "num_key_value_heads":1,
                "head_dim":4,
                "intermediate_size":8,
                "block_size":2,
                "vocab_size":4,
                "dflash_config":{
                    "target_layer_ids":[0],
                    "tap_semantics":"post_layer_residual_v1",
                    "mask_token_id":3,
                    "dspark":true,
                    "markov_rank":1,
                    "log_snr_conditioning":true,
                    "min_log_snr":-9.0,
                    "max_log_snr":9.0
                }
            }"#,
        )
        .unwrap();

        let error = super::load_dflash_drafter(directory.path()).unwrap_err();

        assert!(error.to_string().contains("paired-target API"));
        assert!(!error.to_string().contains("weight"));
    }

    #[test]
    fn accept_prefix_first_token_rejects_returns_correction() {
        let draft = vec![1, 2, 3];
        let verify = vec![9, 5, 5, 5];
        assert_eq!(accept_prefix(&draft, &verify), vec![9]);
    }

    #[test]
    fn accept_prefix_partial_match_returns_prefix_plus_correction() {
        let draft = vec![1, 2, 3];
        let verify = vec![1, 2, 9, 0];
        assert_eq!(accept_prefix(&draft, &verify), vec![1, 2, 9]);
    }

    #[test]
    fn accept_prefix_empty_draft_returns_single_verify_token() {
        let draft: Vec<u32> = vec![];
        let verify = vec![42];
        assert_eq!(accept_prefix(&draft, &verify), vec![42]);
    }

    #[test]
    #[should_panic(expected = "left == right")]
    fn accept_prefix_mismatched_lengths_panic_in_debug() {
        let draft = vec![1, 2, 3];
        let verify = vec![1, 2]; // wrong length
        let _ = accept_prefix(&draft, &verify);
    }

    #[test]
    fn dspark_log_snr_schedule_uses_max_then_min() {
        let features = super::build_log_snr_features(2, -9.0, 9.0).unwrap();
        mlx_rs::transforms::eval([&features]).unwrap();
        let values = features.as_slice::<f32>();
        assert_eq!(features.shape(), &[1, 2, 128]);
        assert!((values[0] - 1000.0_f32.sin()).abs() < 1e-6);
        assert!((values[64] - 1000.0_f32.cos()).abs() < 1e-6);
        assert!(values[128].abs() < f32::EPSILON);
        assert!((values[192] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn dspark_markov_resampler_chains_sampled_token() {
        use mlx_rs::{Array, module::Param};

        let mut drafter = super::DFlashDrafter::new(tiny_dspark_config()).unwrap();
        let extras = drafter.dspark.as_mut().unwrap();
        // A(anchor=3)=3 makes token 1 beat base token 2 at position zero.
        // Chaining the sampled token gives A(1)=1 at position one, where base
        // token 2 wins. Reusing the anchor incorrectly would yield [1, 1].
        extras.markov_head_a = Param::new(Array::from_slice(&[0.0, 1.0, 0.0, 3.0], &[4, 1]));
        extras.markov_head_b.weight = Param::new(Array::from_slice(&[0.0, 1.0, 0.0, 0.0], &[4, 1]));
        let hidden = Array::zeros::<f32>(&[1, 2, 4]).unwrap();
        let base = Array::from_slice(&[0.0_f32, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0], &[1, 2, 4]);
        let tokens = drafter
            .propose_dspark_tokens(&hidden, 3, Some(&base))
            .unwrap()
            .unwrap();
        mlx_rs::transforms::eval([&tokens]).unwrap();
        assert_eq!(tokens.as_slice::<u32>(), &[1, 2]);
    }

    #[test]
    fn chunked_context_priming_is_exact_with_only_sliding_layers() {
        use mlx_rs::ops::indexing::IndexOp;

        let config = tiny_dense_config(Some(3));
        let mut full = super::DFlashDrafter::new(config.clone()).unwrap();
        let mut chunked = super::DFlashDrafter::new(config).unwrap();
        init_patterned_weights(&mut full);
        init_patterned_weights(&mut chunked);

        let taps = input(6, 4, 3);
        let prefix = taps.index((.., ..4, ..));
        let suffix = taps.index((.., 4.., ..));
        let noise = input(2, 4, 19);
        let mut full_cache = full.make_cache();
        let mut chunked_cache = chunked.make_cache();

        let full_output = full.forward(&noise, &[taps], &mut full_cache).unwrap();
        chunked.prime_taps(&[prefix], &mut chunked_cache).unwrap();
        assert_eq!(chunked_cache.position(), 4);
        // Incomplete fixed context tiles stay as bounded raw taps. Absolute
        // position must still advance before any layer KV exists.
        assert_eq!(chunked_cache.pending_rows().unwrap(), 4);
        assert!(chunked_cache.layers[0].is_none());
        let chunked_output = chunked
            .forward(&noise, &[suffix], &mut chunked_cache)
            .unwrap();

        super::DFlashDrafter::eval_cache(&full_cache).unwrap();
        super::DFlashDrafter::eval_cache(&chunked_cache).unwrap();
        mlx_rs::transforms::eval([&full_output, &chunked_output]).unwrap();
        assert_eq!(full_cache.position(), 6);
        assert_eq!(chunked_cache.position(), 6);
        assert_eq!(chunked_cache.pending_rows().unwrap(), 0);
        assert_f32_bits_equal(&full_output, &chunked_output, "draft hidden");
        let (full_k, full_v) = full_cache.layers[0].as_ref().unwrap();
        let (chunked_k, chunked_v) = chunked_cache.layers[0].as_ref().unwrap();
        // Once forward flushes the tile remainder, retained sliding KV is
        // shorter than absolute position. Tensor length cannot be used as the
        // next RoPE offset.
        assert_eq!(chunked_k.shape()[2], 2);
        assert_f32_bits_equal(full_k, chunked_k, "context keys");
        assert_f32_bits_equal(full_v, chunked_v, "context values");
    }

    #[test]
    fn uneven_multilayer_priming_and_next_round_are_exact() {
        use mlx_rs::ops::indexing::IndexOp;

        let mut config = tiny_dense_config(Some(4));
        config.hidden_size = 32;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 2;
        config.head_dim = 8;
        config.intermediate_size = 32;
        config.num_hidden_layers = 6;
        config.layer_types = Some(
            (0..6)
                .map(|index| {
                    if index % 2 == 0 {
                        "sliding_attention".to_owned()
                    } else {
                        "full_attention".to_owned()
                    }
                })
                .collect(),
        );
        config.dflash_config.target_layer_ids = vec![0, 1, 2, 3, 4];
        config.quantization = Some(crate::qwen3_next::QuantizationConfig {
            group_size: 32,
            bits: 4,
            mode: crate::quant_mode::QuantMode::Affine,
        });
        let mut full = super::DFlashDrafter::new(config.clone()).unwrap();
        let mut chunked = super::DFlashDrafter::new(config).unwrap();
        init_patterned_q4_weights(&mut full);
        init_patterned_q4_weights(&mut chunked);
        let taps = (0..5)
            .map(|tap| input(11, 32, tap * 7 + 1))
            .collect::<Vec<_>>();
        let noise = input(2, 32, 29);
        let mut full_cache = full.make_cache();
        let mut chunked_cache = chunked.make_cache();

        let full_output = full.forward(&noise, &taps, &mut full_cache).unwrap();
        for (start, end) in [(0_i32, 1_i32), (1, 4), (4, 6)] {
            let chunk = taps
                .iter()
                .map(|tap| tap.index((.., start..end, ..)))
                .collect::<Vec<_>>();
            chunked.prime_taps(&chunk, &mut chunked_cache).unwrap();
        }
        let final_taps = taps
            .iter()
            .map(|tap| tap.index((.., 6.., ..)))
            .collect::<Vec<_>>();
        let chunked_output = chunked
            .forward(&noise, &final_taps, &mut chunked_cache)
            .unwrap();
        mlx_rs::transforms::eval([&full_output, &chunked_output]).unwrap();
        assert_f32_bits_equal(&full_output, &chunked_output, "uneven first round");
        assert_cache_exact(&full_cache, &chunked_cache, "uneven first round cache");

        let next_taps = (0..5)
            .map(|tap| input(3, 32, tap * 11 + 2))
            .collect::<Vec<_>>();
        let next_noise = input(2, 32, 41);
        let full_next = full
            .forward(&next_noise, &next_taps, &mut full_cache)
            .unwrap();
        let chunked_next = chunked
            .forward(&next_noise, &next_taps, &mut chunked_cache)
            .unwrap();
        mlx_rs::transforms::eval([&full_next, &chunked_next]).unwrap();
        assert_f32_bits_equal(&full_next, &chunked_next, "next draft round");
        assert_cache_exact(&full_cache, &chunked_cache, "next draft round cache");
    }

    #[test]
    fn fixed_context_tiles_are_exact_across_32_and_64_row_boundaries() {
        use mlx_rs::ops::indexing::IndexOp;

        let mut config = tiny_dense_config(Some(16));
        config.hidden_size = 32;
        config.num_attention_heads = 4;
        config.num_key_value_heads = 2;
        config.head_dim = 8;
        config.intermediate_size = 32;
        config.num_hidden_layers = 2;
        config.layer_types = Some(vec![
            "sliding_attention".to_owned(),
            "full_attention".to_owned(),
        ]);
        config.dflash_config.target_layer_ids = vec![0, 1];
        config.quantization = Some(crate::qwen3_next::QuantizationConfig {
            group_size: 32,
            bits: 4,
            mode: crate::quant_mode::QuantMode::Affine,
        });

        let mut full = super::DFlashDrafter::new(config.clone()).unwrap();
        let mut chunked = super::DFlashDrafter::new(config).unwrap();
        init_patterned_q4_weights(&mut full);
        init_patterned_q4_weights(&mut chunked);
        let taps = (0..2)
            .map(|tap| input(67, 32, tap * 13 + 1))
            .collect::<Vec<_>>();
        let noise = input(2, 32, 37);
        let mut full_cache = full.make_cache();
        let mut chunked_cache = chunked.make_cache();

        let full_output = full.forward(&noise, &taps, &mut full_cache).unwrap();
        let mut start = 0_i32;
        for end in [1_i32, 31, 32, 33, 64, 65] {
            let chunk = taps
                .iter()
                .map(|tap| tap.index((.., start..end, ..)))
                .collect::<Vec<_>>();
            chunked.prime_taps(&chunk, &mut chunked_cache).unwrap();
            assert_eq!(chunked_cache.position(), end);
            assert!(chunked_cache.pending_rows().unwrap() < super::DFLASH_CONTEXT_TILE_ROWS);
            start = end;
        }
        let final_taps = taps
            .iter()
            .map(|tap| tap.index((.., start.., ..)))
            .collect::<Vec<_>>();
        let chunked_output = chunked
            .forward(&noise, &final_taps, &mut chunked_cache)
            .unwrap();

        mlx_rs::transforms::eval([&full_output, &chunked_output]).unwrap();
        assert_f32_bits_equal(&full_output, &chunked_output, "multi-tile output");
        assert_cache_exact(&full_cache, &chunked_cache, "multi-tile cache");
        assert_eq!(chunked_cache.position(), 67);
        assert_eq!(chunked_cache.pending_rows().unwrap(), 0);
    }

    #[test]
    fn sealed_snapshot_preserves_fixed_tile_tail_and_next_round_exactness() {
        use mlx_rs::ops::indexing::IndexOp;

        for boundary in [31_i32, 32, 33, 63, 64, 65] {
            let config = tiny_dense_config(Some(16));
            let mut reference = super::DFlashDrafter::new(config.clone()).unwrap();
            let mut sealing = super::DFlashDrafter::new(config).unwrap();
            init_patterned_weights(&mut reference);
            init_patterned_weights(&mut sealing);

            let all_taps = input(boundary, 4, usize::try_from(boundary).unwrap() + 1);
            let split = boundary - 1;
            let prefix = all_taps.index((.., ..split, ..));
            let external = all_taps.index((.., split.., ..));

            let mut reference_cache = reference.make_cache();
            reference
                .prime_taps(&[all_taps], &mut reference_cache)
                .unwrap();

            let mut live = sealing.make_cache();
            sealing.prime_taps(&[prefix], &mut live).unwrap();
            let snapshot = sealing
                .seal_after_taps(live, &[external], boundary)
                .unwrap();

            assert_eq!(snapshot.position(), boundary);
            let pending_rows = snapshot
                .pending_taps
                .as_ref()
                .map_or(0, |pending| pending.shape()[1]);
            assert_eq!(
                pending_rows,
                boundary.rem_euclid(super::DFLASH_CONTEXT_TILE_ROWS),
                "boundary {boundary} must retain its raw fixed-tile tail"
            );

            let mut resumed = snapshot.into_live();
            assert_cache_exact(
                &reference_cache,
                &resumed,
                &format!("sealed boundary {boundary}"),
            );

            let next_taps = input(2, 4, usize::try_from(boundary).unwrap() + 19);
            let next_noise = input(2, 4, usize::try_from(boundary).unwrap() + 37);
            let reference_hidden = reference
                .forward(&next_noise, &[next_taps.clone()], &mut reference_cache)
                .unwrap();
            let resumed_hidden = sealing
                .forward(&next_noise, &[next_taps], &mut resumed)
                .unwrap();
            mlx_rs::transforms::eval([&reference_hidden, &resumed_hidden]).unwrap();
            assert_f32_bits_equal(
                &reference_hidden,
                &resumed_hidden,
                &format!("sealed boundary {boundary} next hidden"),
            );
            assert_cache_exact(
                &reference_cache,
                &resumed,
                &format!("sealed boundary {boundary} next cache"),
            );
        }
    }

    #[test]
    fn snapshot_forks_remain_independent_after_one_live_branch_advances() {
        let mut drafter = super::DFlashDrafter::new(tiny_dense_config(Some(16))).unwrap();
        init_patterned_weights(&mut drafter);
        let mut live = drafter.make_cache();
        drafter.prime_taps(&[input(33, 4, 5)], &mut live).unwrap();
        let snapshot = drafter.seal_after_taps(live, &[], 33).unwrap();

        let stable_before = snapshot.fork_live().unwrap();
        let mut advanced = snapshot.fork_live().unwrap();
        drafter
            .prime_taps(&[input(2, 4, 17)], &mut advanced)
            .unwrap();
        let stable_after = snapshot.fork_live().unwrap();

        assert_eq!(advanced.position(), 35);
        assert_eq!(stable_after.position(), 33);
        assert_cache_exact(
            &stable_before,
            &stable_after,
            "immutable snapshot after fork advance",
        );
    }

    #[test]
    fn snapshot_estimated_bytes_sums_retained_arrays() {
        let keys = mlx_rs::Array::zeros::<f32>(&[1, 1, 2, 4]).unwrap();
        let values = mlx_rs::Array::zeros::<f32>(&[1, 1, 2, 4]).unwrap();
        let pending = mlx_rs::Array::zeros::<f32>(&[1, 3, 4]).unwrap();
        let expected = keys
            .nbytes()
            .saturating_add(values.nbytes())
            .saturating_add(pending.nbytes());
        let snapshot = super::DFlashSnapshot {
            layers: vec![Some((keys, values)), None],
            pending_taps: Some(pending),
            position: 5,
        };

        assert_eq!(snapshot.estimated_bytes(), expected);
    }

    #[test]
    fn dflash_branch_counter_rejects_wraparound() {
        let counter = std::sync::atomic::AtomicU64::new(u64::MAX);

        assert!(super::try_next_dflash_branch_id(&counter).is_none());
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn staged_transaction_rejects_a_different_same_position_branch() {
        let mut drafter = super::DFlashDrafter::new(tiny_dense_config(None)).unwrap();
        init_patterned_weights(&mut drafter);
        let snapshot = drafter
            .seal_after_taps(drafter.make_cache(), &[], 0)
            .unwrap();
        let left = snapshot.fork_live().unwrap();
        let mut right = snapshot.fork_live().unwrap();
        let taps = input(2, 4, 23);
        let noise = input(2, 4, 29);

        let transaction = drafter.stage_forward(&noise, &[taps], &left).unwrap();
        let error = transaction.commit(&mut right).unwrap_err();

        assert!(error.to_string().contains("branch"));
        assert_eq!(right.position(), 0);
    }

    #[test]
    fn staged_transaction_rejects_a_same_position_revision_change() {
        let mut drafter = super::DFlashDrafter::new(tiny_dense_config(None)).unwrap();
        init_patterned_weights(&mut drafter);
        let mut cache = drafter.make_cache();
        let taps = input(2, 4, 31);
        let noise = input(2, 4, 41);
        let transaction = drafter.stage_forward(&noise, &[taps], &cache).unwrap();
        cache.revision += 1;

        let error = transaction.commit(&mut cache).unwrap_err();

        assert!(error.to_string().contains("revision"));
        assert_eq!(cache.position(), 0);
    }

    #[test]
    fn seal_rejects_an_incorrect_absolute_boundary() {
        let mut drafter = super::DFlashDrafter::new(tiny_dense_config(None)).unwrap();
        init_patterned_weights(&mut drafter);

        let error = drafter
            .seal_after_taps(drafter.make_cache(), &[], 1)
            .unwrap_err();

        assert!(error.to_string().contains("boundary"));
    }

    #[test]
    fn seal_accepts_empty_and_single_pending_boundaries() {
        let mut drafter = super::DFlashDrafter::new(tiny_dense_config(None)).unwrap();
        init_patterned_weights(&mut drafter);

        let empty = drafter
            .seal_after_taps(drafter.make_cache(), &[], 0)
            .unwrap();
        assert_eq!(empty.position(), 0);
        assert!(empty.layers.iter().all(Option::is_none));
        assert!(empty.pending_taps.is_none());

        let one = drafter
            .seal_after_taps(drafter.make_cache(), &[input(1, 4, 47)], 1)
            .unwrap();
        assert_eq!(one.position(), 1);
        assert!(one.layers.iter().all(Option::is_none));
        assert_eq!(one.pending_taps.as_ref().unwrap().shape(), &[1, 1, 4]);
    }

    #[test]
    fn seal_rejects_wrong_full_and_sliding_retained_lengths() {
        for (sliding_window, expected_retained) in [(None, 32_i32), (Some(16), 15)] {
            let mut drafter = super::DFlashDrafter::new(tiny_dense_config(sliding_window)).unwrap();
            init_patterned_weights(&mut drafter);
            let mut cache = drafter.make_cache();
            drafter.prime_taps(&[input(32, 4, 53)], &mut cache).unwrap();
            let wrong_retained = expected_retained - 1;
            let wrong = mlx_rs::Array::zeros::<f32>(&[1, 1, wrong_retained, 4]).unwrap();
            cache.layers[0] = Some((wrong.clone(), wrong));

            let error = drafter.seal_after_taps(cache, &[], 32).unwrap_err();

            assert!(
                error.to_string().contains("cache shape mismatch"),
                "unexpected error for sliding_window={sliding_window:?}: {error}"
            );
        }
    }

    #[test]
    fn invalid_tap_batch_leaves_cache_transaction_unchanged() {
        let mut config = tiny_dense_config(None);
        config.dflash_config.target_layer_ids = vec![0, 1];
        let mut drafter = super::DFlashDrafter::new(config).unwrap();
        init_patterned_weights(&mut drafter);
        let mut cache = drafter.make_cache();
        let good = input(2, 4, 1);
        drafter
            .prime_taps(&[good.clone(), good], &mut cache)
            .unwrap();
        let snapshot = drafter.seal_after_taps(cache, &[], 2).unwrap();
        let before = snapshot.fork_live().unwrap();
        let mut cache = snapshot.into_live();

        let first = input(1, 4, 2);
        let bad_batch = mlx_rs::Array::zeros::<f32>(&[2, 1, 4]).unwrap();
        let error = drafter
            .prime_taps(&[first, bad_batch], &mut cache)
            .unwrap_err();
        assert!(error.to_string().contains("shape mismatch"));
        assert_cache_exact(&cache, &before, "invalid transaction");
    }

    #[test]
    fn invalid_noise_shape_leaves_cache_transaction_unchanged() {
        let mut drafter = super::DFlashDrafter::new(tiny_dense_config(None)).unwrap();
        init_patterned_weights(&mut drafter);
        let mut cache = drafter.make_cache();
        let taps = input(2, 4, 1);
        drafter.prime_taps(&[taps.clone()], &mut cache).unwrap();
        let snapshot = drafter.seal_after_taps(cache, &[], 2).unwrap();
        let before = snapshot.fork_live().unwrap();
        let mut cache = snapshot.into_live();
        let wrong_block = input(1, 4, 7);

        let error = drafter
            .forward(&wrong_block, &[taps], &mut cache)
            .unwrap_err();

        assert!(error.to_string().contains("drafter noise must be"));
        assert_cache_exact(&cache, &before, "invalid noise transaction");
    }

    #[test]
    fn staged_forward_publishes_cache_only_after_successful_commit() {
        let mut drafter = super::DFlashDrafter::new(tiny_dense_config(None)).unwrap();
        init_patterned_weights(&mut drafter);
        let empty = drafter.make_cache();
        let snapshot = drafter.seal_after_taps(empty, &[], 0).unwrap();
        let before = snapshot.fork_live().unwrap();
        let mut cache = snapshot.into_live();
        let taps = input(2, 4, 3);
        let noise = input(2, 4, 11);

        let transaction = drafter.stage_forward(&noise, &[taps], &cache).unwrap();
        mlx_rs::transforms::eval([transaction.hidden()]).unwrap();
        assert_eq!(transaction.position().unwrap(), 2);
        assert_cache_exact(&cache, &before, "uncommitted forward");

        transaction.commit(&mut cache).unwrap();
        assert_eq!(cache.position(), 2);
        assert!(cache.layers[0].is_some());
        assert_eq!(cache.pending_rows().unwrap(), 0);
    }

    #[test]
    fn stale_staged_forward_cannot_overwrite_newer_cache() {
        let mut drafter = super::DFlashDrafter::new(tiny_dense_config(None)).unwrap();
        init_patterned_weights(&mut drafter);
        let mut cache = drafter.make_cache();
        let taps = input(2, 4, 5);
        let noise = input(2, 4, 13);
        let first = drafter
            .stage_forward(&noise, &[taps.clone()], &cache)
            .unwrap();
        let stale = drafter.stage_forward(&noise, &[taps], &cache).unwrap();

        first.commit(&mut cache).unwrap();
        let error = stale.commit(&mut cache).unwrap_err();

        assert!(error.to_string().contains("stale drafter transaction"));
        assert_eq!(cache.position(), 2);
    }

    /// Sliding-window eviction: a tiny random drafter with layer 0 = sliding
    /// (window 4) and layer 1 = full. Driving several rounds must cap the
    /// sliding layer's context KV at `sliding_window - 1` while the full layer
    /// accumulates all context — and the rope offset (max cache length across
    /// layers) must keep advancing absolutely. No weights needed.
    #[test]
    fn sliding_layer_cache_caps_while_full_layer_grows() {
        use super::{DFlashConfig, DFlashDrafter, DFlashSubConfig};
        use mlx_rs::Array;

        let hidden = 8i32;
        let config = DFlashConfig {
            hidden_size: hidden,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            intermediate_size: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 1e7,
            block_size: 4,
            vocab_size: 32,
            layer_types: Some(vec![
                "sliding_attention".to_owned(),
                "full_attention".to_owned(),
            ]),
            sliding_window: Some(4),
            quantization: None,
            dflash_config: DFlashSubConfig {
                target_layer_ids: vec![0],
                tap_semantics: None,
                mask_token_id: Some(1),
                dspark: false,
                markov_rank: 0,
                log_snr_conditioning: false,
                min_log_snr: 0.0,
                max_log_snr: 0.0,
                reuse_target_head: false,
                target_binding: None,
            },
        };
        let mut drafter = DFlashDrafter::new(config).unwrap();
        // QLinear (unlike nn::Linear) starts with [1] placeholder weights.
        // Initialize them to zero at the correct shapes so forward() works.
        drafter.init_test_weights();
        let mut cache = drafter.make_cache();

        let ctx_len = 2i32; // context positions added per round
        let rounds = 6;
        let zeros = |t: i32| Array::from_slice(&vec![0f32; (t * hidden) as usize], &[1, t, hidden]);
        for _ in 0..rounds {
            let taps = vec![zeros(ctx_len)]; // num_taps = 1
            let noise = zeros(4); // block_size noise positions
            drafter.forward(&noise, &taps, &mut cache).unwrap();
        }

        let sliding_len = cache.layers[0].as_ref().unwrap().0.shape()[2];
        let full_len = cache.layers[1].as_ref().unwrap().0.shape()[2];
        assert!(
            sliding_len <= 3,
            "sliding cache must cap at window-1=3, got {sliding_len}"
        );
        assert_eq!(
            full_len,
            ctx_len * rounds,
            "full-attn cache must accumulate all {} context positions, got {full_len}",
            ctx_len * rounds
        );
        assert!(
            full_len > sliding_len,
            "eviction must make the sliding cache shorter than the full one"
        );
    }

    /// The "drop-in drafter, no MLX port" claim: the config-driven loader must
    /// parse Modal's config.json and populate every module from Modal's bf16
    /// safetensors by key name. Gated on the real weights dir.
    #[test]
    #[ignore = "load: set HIGGS_DFLASH_DRAFTER_DIR to the Modal drafter snapshot dir"]
    #[allow(clippy::print_stderr)]
    fn loads_modal_drafter_against_real_weights() {
        let Ok(dir) = std::env::var("HIGGS_DFLASH_DRAFTER_DIR") else {
            eprintln!("skip: set HIGGS_DFLASH_DRAFTER_DIR");
            return;
        };
        let d = super::load_dflash_drafter(std::path::Path::new(&dir))
            .expect("load Modal DFlash drafter");
        let c = &d.config;
        assert_eq!(c.num_hidden_layers, 6, "layers");
        assert_eq!(c.hidden_size, 2048, "hidden");
        assert_eq!(c.num_taps(), 8, "taps");
        assert_eq!(
            c.target_layer_ids(),
            &[1, 6, 11, 16, 22, 27, 32, 37],
            "tap ids"
        );
        assert_eq!(c.mask_token_id(), 248_077, "mask token");
        eprintln!(
            "Modal drafter loaded OK: {} layers, hidden {}, {} taps, fc_in {}",
            c.num_hidden_layers,
            c.hidden_size,
            c.num_taps(),
            c.num_taps() as i32 * c.hidden_size
        );
    }
}

#[cfg(test)]
mod weight_check {
    #[test]
    #[ignore = "needs HIGGS_DFLASH_DRAFTER_DIR"]
    fn check_weights_loaded() {
        let dir = std::env::var("HIGGS_DFLASH_DRAFTER_DIR").unwrap();
        let d = super::load_dflash_drafter(std::path::Path::new(&dir)).unwrap();
        let fc_w = &d.fc.weight;
        let fc_s = &d.fc.scales;
        let q_w = &d.layers[0].self_attn.q_proj.weight;
        let q_s = &d.layers[0].self_attn.q_proj.scales;
        eprintln!("fc.weight: {:?} {:?}", fc_w.shape(), fc_w.dtype());
        eprintln!("fc.scales: {:?} {:?}", fc_s.shape(), fc_s.dtype());
        eprintln!("q_proj.weight: {:?} {:?}", q_w.shape(), q_w.dtype());
        eprintln!("q_proj.scales: {:?} {:?}", q_s.shape(), q_s.dtype());
        let fc_prod: i32 = fc_w.shape().iter().product();
        let q_prod: i32 = q_w.shape().iter().product();
        assert!(
            fc_prod > 1,
            "fc.weight placeholder! shape={:?}",
            fc_w.shape()
        );
        assert!(
            q_prod > 1,
            "q_proj.weight placeholder! shape={:?}",
            q_w.shape()
        );
        eprintln!("ALL WEIGHTS LOADED OK");
    }
}
