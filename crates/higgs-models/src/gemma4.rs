// Model forward passes eval under the engine MLX gate (structurally on-gate); see clippy.toml.
#![allow(clippy::disallowed_methods)]

//! Gemma 4 (text) model implementation.
//!
//! Key differences from Gemma 2/3:
//! - Standard `RMSNorm` (NO +1 convention)
//! - Per-layer-type geometry: global layers use `global_head_dim` + partial rotary 0.25;
//!   sliding layers use `head_dim` + full rotary
//! - Cross-layer KV sharing: shared layers (`i >= first_kv_shared`) reuse the post-cache
//!   K/V of the most recent earlier layer of the SAME layer type
//! - Optional per-layer input embeddings (small/edge models)
//! - Optional `MoE` block (dense MLP + experts in parallel, summed)
//! - `attention_k_eq_v`: global layers may use keys as values (no `v_proj`)
//! - Final logit soft-capping at 30.0
//! - `layer_scalar` (shape `[1]`, ones) multiplies each layer output
//! - Sliding window via masking (full KV retained)

use std::path::Path;

use mlx_rs::{
    Array, array,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters as _, ModuleParametersExt as _, Param},
    nn, ops,
    ops::indexing::IndexOp,
    quantization::MaybeQuantized,
};
use serde::Deserialize;

use crate::{
    cache::KeyValueCache,
    error::ModelError,
    utils::{apply_rope, create_causal_mask, create_windowed_causal_mask},
};

// ---------------------------------------------------------------------------
// Config defaults
// ---------------------------------------------------------------------------

const fn default_head_dim() -> i32 {
    256
}

const fn default_global_head_dim() -> i32 {
    512
}

const fn default_global_partial_rotary_factor() -> f32 {
    0.25
}

const fn default_partial_rotary_factor() -> f32 {
    1.0
}

const fn default_rms_norm_eps() -> f32 {
    1e-6
}

const fn default_vocab_size() -> i32 {
    262_144
}

const fn default_vocab_size_per_layer_input() -> i32 {
    262_144
}

const fn default_num_attention_heads() -> i32 {
    8
}

const fn default_num_key_value_heads() -> i32 {
    1
}

const fn default_sliding_window() -> i32 {
    512
}

const fn default_sliding_window_pattern() -> i32 {
    5
}

const fn default_final_logit_softcapping() -> f32 {
    30.0
}

const fn default_tie_word_embeddings() -> bool {
    true
}

const fn default_use_double_wide_mlp() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Gemma 4 model configuration.
///
/// HF ships two shapes:
/// - Flat: all fields at top level.
/// - Wrapped: `{"model_type":"gemma4","text_config":{...}}`.
///
/// `load_gemma4_model_args` normalizes these; this struct covers the flat layout.
#[derive(Debug, Clone, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct Gemma4ModelArgs {
    #[serde(default)]
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    #[serde(default = "default_global_head_dim")]
    pub global_head_dim: i32,
    #[serde(default = "default_global_partial_rotary_factor")]
    pub global_partial_rotary_factor: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    #[serde(default = "default_vocab_size_per_layer_input")]
    pub vocab_size_per_layer_input: i32,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: i32,
    #[serde(default)]
    pub num_global_key_value_heads: Option<i32>,
    #[serde(default)]
    pub num_kv_shared_layers: i32,
    #[serde(default)]
    pub hidden_size_per_layer_input: i32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: i32,
    #[serde(default)]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub attention_k_eq_v: bool,
    #[serde(default = "default_final_logit_softcapping")]
    pub final_logit_softcapping: f32,
    #[serde(default = "default_use_double_wide_mlp")]
    pub use_double_wide_mlp: bool,
    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: Option<i32>,
    #[serde(default)]
    pub top_k_experts: Option<i32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    /// Explicit list of layer types; if absent, derived from `sliding_window_pattern`.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<crate::gemma2::QuantizationConfig>,
}

impl Gemma4ModelArgs {
    /// Index of the first KV-shared layer. Layers `[0, first_kv_shared)` own K/V;
    /// layers `[first_kv_shared, N)` reuse the K/V of an earlier owner.
    pub fn first_kv_shared(&self) -> i32 {
        (self.num_hidden_layers - self.num_kv_shared_layers).max(0)
    }

    /// True if layer `i` uses full (global) attention.
    pub fn is_full_attention(&self, layer_idx: i32) -> bool {
        self.layer_types.as_ref().map_or_else(
            || {
                // Default pattern: every `sliding_window_pattern`-th layer (1-indexed) is global.
                self.sliding_window_pattern > 0
                    && (layer_idx + 1) % self.sliding_window_pattern == 0
            },
            |types| {
                let i = usize::try_from(layer_idx).unwrap_or(0);
                types.get(i).is_some_and(|t| t == "full_attention")
            },
        )
    }

    /// Effective rope theta and `partial_rotary_factor` for a given layer type.
    #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
    pub fn rope_params_for(&self, full_attention: bool) -> (f32, f32) {
        let key = if full_attention {
            "full_attention"
        } else {
            "sliding_attention"
        };

        if let Some(rp_val) = &self.rope_parameters {
            if let Some(entry) = rp_val.get(key) {
                // f64->f32 is intentionally lossy: RoPE theta values fit in f32.
                #[allow(clippy::cast_possible_truncation)]
                let theta = entry
                    .get("rope_theta")
                    .and_then(serde_json::Value::as_f64)
                    .map_or(
                        if full_attention {
                            1_000_000.0
                        } else {
                            10_000.0
                        },
                        |v| v as f32,
                    );
                #[allow(clippy::cast_possible_truncation)]
                let prf = entry
                    .get("partial_rotary_factor")
                    .and_then(serde_json::Value::as_f64)
                    .map_or(if full_attention { 0.25 } else { 1.0 }, |v| v as f32);
                return (theta, prf);
            }
        }

        if full_attention {
            (1_000_000.0, self.global_partial_rotary_factor)
        } else {
            (10_000.0, self.partial_rotary_factor)
        }
    }

    /// Per-layer-type `head_dim`.
    pub const fn head_dim_for(&self, full_attention: bool) -> i32 {
        if full_attention {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }

    /// Per-layer-type `n_kv_heads`.
    pub fn n_kv_heads_for(&self, full_attention: bool) -> i32 {
        if full_attention && self.attention_k_eq_v {
            self.num_global_key_value_heads
                .unwrap_or(self.num_key_value_heads)
        } else {
            self.num_key_value_heads
        }
    }

    /// Build the `previous_kvs` routing table.
    ///
    /// Owner layers (`i < first_kv_shared`) map to themselves.
    /// Shared layers map to the most recent owner of the same layer type.
    pub fn previous_kvs(&self) -> Vec<usize> {
        let n = usize::try_from(self.num_hidden_layers).unwrap_or(0);
        let m = usize::try_from(self.first_kv_shared()).unwrap_or(n);

        let mut table: Vec<usize> = (0..n).collect();

        if m < n {
            let mut last_full: Option<usize> = None;
            let mut last_sliding: Option<usize> = None;

            for i in 0..m {
                let i32_i = i32::try_from(i).unwrap_or(i32::MAX);
                if self.is_full_attention(i32_i) {
                    last_full = Some(i);
                } else {
                    last_sliding = Some(i);
                }
            }

            for j in m..n {
                let i32_j = i32::try_from(j).unwrap_or(i32::MAX);
                let prev = if self.is_full_attention(i32_j) {
                    last_full
                } else {
                    last_sliding
                };
                if let Some(slot) = table.get_mut(j) {
                    *slot = prev.unwrap_or(j);
                }
            }
        }

        table
    }
}

// ---------------------------------------------------------------------------
// No-scale RMSNorm (v_norm)
// ---------------------------------------------------------------------------

/// Normalizes with `RMSNorm` but applies no learnable scale weight.
///
/// Matches mlx-lm's `RMSNormNoScale` used for `v_norm` in Gemma 4.
struct RmsNormNoScale {
    eps: f32,
}

impl RmsNormNoScale {
    const fn new(eps: f32) -> Self {
        Self { eps }
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let x_sq = x.square()?;
        let mean_sq = x_sq.mean_axes(&[-1], true)?;
        let rms = mean_sq.add(array!(self.eps))?.rsqrt()?;
        x.multiply(rms)
    }
}

// ---------------------------------------------------------------------------
// MLP (GeGLU)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma4Mlp {
    #[quantizable]
    #[param]
    gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    down_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    up_proj: MaybeQuantized<nn::Linear>,
}

impl Gemma4Mlp {
    fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(dim, hidden_dim)
                    .bias(false)
                    .build()?,
            ),
            down_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(hidden_dim, dim)
                    .bias(false)
                    .build()?,
            ),
            up_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(dim, hidden_dim)
                    .bias(false)
                    .build()?,
            ),
        })
    }
}

impl Module<&Array> for Gemma4Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        // GeGLU: gelu_approx(gate) * up, then project down
        let gated = nn::gelu_approximate(self.gate_proj.forward(input)?)?
            .multiply(self.up_proj.forward(input)?)?;
        self.down_proj.forward(&gated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// MoE Router
// ---------------------------------------------------------------------------

/// Gemma 4 `MoE` router — `rms_norm` -> linear -> top-k -> softmax.
///
/// `scale` and `per_expert_scale` are learnable (loaded from checkpoint keys
/// `router.scale` / `router.per_expert_scale`). The RMS normalization weight is
/// `scale * hidden^-0.5` and the top-k softmax weights are multiplied by the
/// selected `per_expert_scale` entries.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma4Router {
    #[quantizable]
    #[param]
    proj: MaybeQuantized<nn::Linear>,
    #[param]
    scale: Param<Array>,
    #[param]
    per_expert_scale: Param<Array>,

    num_experts: i32,
    top_k: i32,
    eps: f32,
    root: f32,
    cached_norm_weight: Option<Array>,
}

impl Gemma4Router {
    #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
    fn new(hidden: i32, num_experts: i32, top_k: i32, eps: f32) -> Result<Self, Exception> {
        let proj = nn::LinearBuilder::new(hidden, num_experts)
            .bias(false)
            .build()?;
        Ok(Self {
            proj: MaybeQuantized::Original(proj),
            scale: Param::new(Array::ones::<f32>(&[hidden])?),
            per_expert_scale: Param::new(Array::ones::<f32>(&[num_experts])?),
            num_experts,
            top_k,
            eps,
            root: (hidden as f32).powf(-0.5),
            cached_norm_weight: None,
        })
    }

    /// Returns `(expert_indices [B,L,top_k], weights [B,L,top_k])`.
    fn forward(&mut self, x: &Array) -> Result<(Array, Array), Exception> {
        let x_sq = x.square()?;
        let mean_sq = x_sq.mean_axes(&[-1], true)?;
        let rms = mean_sq.add(array!(self.eps))?.rsqrt()?;

        // RMS norm weight = scale * hidden^-0.5, cached and dtype-matched to the
        // running activation dtype to avoid f32 promotion.
        let needs_refresh = self
            .cached_norm_weight
            .as_ref()
            .is_none_or(|w| w.dtype() != x.dtype());
        if needs_refresh {
            let weight = self
                .scale
                .multiply(array!(self.root))?
                .as_dtype(x.dtype())?;
            self.cached_norm_weight = Some(weight);
        }
        let norm_weight = self
            .cached_norm_weight
            .as_ref()
            .ok_or_else(|| Exception::custom("cached_norm_weight not initialized"))?;
        let x_normed = x.multiply(rms)?.multiply(norm_weight)?;

        let logits = self.proj.forward(&x_normed)?;

        let k = self.top_k;
        let neg_logits = logits.negative()?;
        let top_k_indices = ops::argpartition_axis(&neg_logits, k - 1, -1)?.index((.., .., ..k));

        let top_k_scores = logits.take_along_axis(&top_k_indices, -1)?;
        let weights = ops::softmax_axis(&top_k_scores, -1, None)?;

        // Multiply by per-expert scale gathered at the selected expert indices.
        // `per_expert_scale` is 1-D `[num_experts]`; gathering along axis 0 with
        // `[B, L, top_k]` indices yields `[B, L, top_k]`.
        let pes = self.per_expert_scale.as_dtype(weights.dtype())?;
        let pes_sel = ops::indexing::take_axis(&pes, &top_k_indices, 0)?;
        let scaled_weights = weights.multiply(&pes_sel)?;

        Ok((top_k_indices, scaled_weights))
    }
}

// ---------------------------------------------------------------------------
// MoE Experts (GeGLU)
// ---------------------------------------------------------------------------

/// Gemma 4 `MoE` expert block.
///
/// All expert weights are stacked: gate/up projections have
/// `out_features = num_experts * moe_inter`, and down projection has
/// `out_features = num_experts * hidden`. Forward uses "project all, gather
/// selected" to dispatch tokens to their top-k experts.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma4Experts {
    #[quantizable]
    #[param]
    gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    up_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    down_proj: MaybeQuantized<nn::Linear>,

    num_experts: i32,
    moe_inter: i32,
    hidden: i32,
}

impl Gemma4Experts {
    fn new(hidden: i32, moe_inter: i32, num_experts: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(hidden, num_experts * moe_inter)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(hidden, num_experts * moe_inter)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(moe_inter, num_experts * hidden)
            .bias(false)
            .build()?;
        Ok(Self {
            gate_proj: MaybeQuantized::Original(gate_proj),
            up_proj: MaybeQuantized::Original(up_proj),
            down_proj: MaybeQuantized::Original(down_proj),
            num_experts,
            moe_inter,
            hidden,
        })
    }

    #[allow(non_snake_case)]
    fn forward(&mut self, x: &Array, indices: &Array, weights: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let err = || Exception::custom("experts input must have shape [B, L, H]");
        let B = *shape.first().ok_or_else(err)?;
        let L = *shape.get(1).ok_or_else(err)?;
        let top_k = *indices
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("indices must have last dim"))?;

        let E = self.num_experts;
        let M = self.moe_inter;
        let H = self.hidden;

        let gate_all = self.gate_proj.forward(x)?.reshape(&[B, L, E, M])?;
        let up_all = self.up_proj.forward(x)?.reshape(&[B, L, E, M])?;

        // Gather for top-k experts
        let idx_gate = ops::broadcast_to(&indices.reshape(&[B, L, top_k, 1])?, &[B, L, top_k, M])?;
        let gate_sel = gate_all.take_along_axis(&idx_gate, 2)?;
        let up_sel = up_all.take_along_axis(&idx_gate, 2)?;

        // GeGLU
        let activated = nn::gelu_approximate(gate_sel)?.multiply(up_sel)?;

        // Down: project all, gather correct expert
        let flat_activated = activated.reshape(&[B * L * top_k, M])?;
        let down_all = self
            .down_proj
            .forward(&flat_activated)?
            .reshape(&[B * L * top_k, E, H])?;

        let flat_indices = indices.flatten(None, None)?;
        let idx_down = ops::broadcast_to(
            &flat_indices.reshape(&[B * L * top_k, 1, 1])?,
            &[B * L * top_k, 1, H],
        )?;
        let down_sel = down_all
            .take_along_axis(&idx_down, 1)?
            .reshape(&[B, L, top_k, H])?;

        // Weighted sum over experts
        let w_exp = ops::broadcast_to(&weights.reshape(&[B, L, top_k, 1])?, &[B, L, top_k, H])?;
        down_sel.multiply(w_exp)?.sum_axes(&[2], false)
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Gemma 4 attention — unified struct supporting owner and shared layers.
///
/// Owner layers have K/V projections and update the cache.
/// Shared layers have only Q projection and reuse the owner's cached K/V.
/// `is_owner` distinguishes the two modes at runtime.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma4Attention {
    #[quantizable]
    #[param]
    q_proj: MaybeQuantized<nn::Linear>,
    // None for shared layers
    #[quantizable]
    #[param]
    k_proj: Option<MaybeQuantized<nn::Linear>>,
    // None for shared layers or when attention_k_eq_v is true for full-attention layers
    #[quantizable]
    #[param]
    v_proj: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    q_norm: nn::RmsNorm,
    // None for shared layers
    #[param]
    k_norm: Option<nn::RmsNorm>,
    #[param]
    rope: nn::Rope,

    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    is_owner: bool,
    k_eq_v: bool,
    v_norm_eps: f32,
}

impl Gemma4Attention {
    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn new_owner(args: &Gemma4ModelArgs, full_attention: bool) -> Result<Self, Exception> {
        let head_dim = args.head_dim_for(full_attention);
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.n_kv_heads_for(full_attention);
        let k_eq_v = full_attention && args.attention_k_eq_v;

        let q_proj = nn::LinearBuilder::new(args.hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(args.hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = if k_eq_v {
            None
        } else {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, n_kv_heads * head_dim)
                    .bias(false)
                    .build()?,
            ))
        };
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, args.hidden_size)
            .bias(false)
            .build()?;
        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;

        let (rope_theta, partial_rotary_factor) = args.rope_params_for(full_attention);
        let rope_dims = (head_dim as f32 * partial_rotary_factor).round() as i32;
        let rope = nn::RopeBuilder::new(rope_dims)
            .traditional(false)
            .base(rope_theta)
            .scale(1.0)
            .build()
            .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?;

        Ok(Self {
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: Some(MaybeQuantized::Original(k_proj)),
            v_proj,
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm: Some(k_norm),
            rope,
            n_heads,
            n_kv_heads,
            head_dim,
            is_owner: true,
            k_eq_v,
            v_norm_eps: args.rms_norm_eps,
        })
    }

    #[allow(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn new_shared(args: &Gemma4ModelArgs, full_attention: bool) -> Result<Self, Exception> {
        let head_dim = args.head_dim_for(full_attention);
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.n_kv_heads_for(full_attention);

        let q_proj = nn::LinearBuilder::new(args.hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, args.hidden_size)
            .bias(false)
            .build()?;
        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;

        let (rope_theta, partial_rotary_factor) = args.rope_params_for(full_attention);
        let rope_dims = (head_dim as f32 * partial_rotary_factor).round() as i32;
        let rope = nn::RopeBuilder::new(rope_dims)
            .traditional(false)
            .base(rope_theta)
            .scale(1.0)
            .build()
            .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?;

        Ok(Self {
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: None,
            v_proj: None,
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm: None,
            rope,
            n_heads,
            n_kv_heads,
            head_dim,
            is_owner: false,
            k_eq_v: false,
            v_norm_eps: args.rms_norm_eps,
        })
    }

    /// Forward for owner layers.
    ///
    /// Returns `(output, (keys, values), offset_after_update)`.
    #[allow(non_snake_case)]
    fn forward_owner<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut C>,
    ) -> Result<(Array, (Array, Array), i32), Exception> {
        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        let v_norm = RmsNormNoScale::new(self.v_norm_eps);

        let k_proj = self
            .k_proj
            .as_mut()
            .ok_or_else(|| Exception::custom("forward_owner called on shared-only attention"))?;
        let k_norm = self
            .k_norm
            .as_mut()
            .ok_or_else(|| Exception::custom("forward_owner called on shared-only attention"))?;

        let q_reshaped = self
            .q_proj
            .forward(x)?
            .reshape(&[B, L, self.n_heads, self.head_dim])?;
        let q_normed = self
            .q_norm
            .forward(&q_reshaped)?
            .transpose_axes(&[0, 2, 1, 3])?;

        let k_raw = k_proj
            .forward(x)?
            .reshape(&[B, L, self.n_kv_heads, self.head_dim])?;
        let k_normed = k_norm.forward(&k_raw)?.transpose_axes(&[0, 2, 1, 3])?;

        let v_unnormed = match &mut self.v_proj {
            Some(vp) => vp
                .forward(x)?
                .reshape(&[B, L, self.n_kv_heads, self.head_dim])?,
            None => k_raw.clone(), // k_eq_v: values = (pre-norm) keys
        };
        let values_new = v_norm.forward(&v_unnormed)?.transpose_axes(&[0, 2, 1, 3])?;

        // The PRE-update offset is the absolute position of this layer's queries.
        // It is threaded to shared layers so they rope their queries at the same
        // positions. Returning the post-update offset (old + L) would shift shared
        // layers by L in prefill and by 1 per decode step.
        let offset = kv_cache.as_ref().map_or(0, KeyValueCache::offset);
        let keys_roped = apply_rope(&k_normed, &self.rope, offset)?;

        let (keys, values) = if let Some(cache) = kv_cache {
            cache.update_and_fetch(keys_roped, values_new)?
        } else {
            (keys_roped, values_new)
        };

        let q_final = apply_rope(&q_normed, &self.rope, offset)?;

        let sdpa_mask = mask.map(mlx_rs::fast::ScaledDotProductAttentionMask::Array);
        let attn_raw = mlx_rs::fast::scaled_dot_product_attention(
            q_final,
            keys.clone(),
            values.clone(),
            1.0,
            sdpa_mask,
            None::<&Array>,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        let attn_out = self.o_proj.forward(&attn_raw)?;
        Ok((attn_out, (keys, values), offset))
    }

    /// Forward for shared layers — reuses pre-computed K/V from an owner layer.
    #[allow(non_snake_case)]
    fn forward_shared(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        shared_kv: &(Array, Array),
        offset: i32,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        let (keys, values) = shared_kv;

        let q_reshaped = self
            .q_proj
            .forward(x)?
            .reshape(&[B, L, self.n_heads, self.head_dim])?;
        let q_normed = self
            .q_norm
            .forward(&q_reshaped)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let q_final = apply_rope(&q_normed, &self.rope, offset)?;

        let sdpa_mask = mask.map(mlx_rs::fast::ScaledDotProductAttentionMask::Array);
        let attn_raw = mlx_rs::fast::scaled_dot_product_attention(
            q_final,
            keys.clone(),
            values.clone(),
            1.0,
            sdpa_mask,
            None::<&Array>,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        self.o_proj.forward(&attn_raw)
    }
}

// ---------------------------------------------------------------------------
// Decoder block
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma4Block {
    // Field name must be `self_attn` to match checkpoint weight keys
    // (`model.layers.N.self_attn.*`).
    #[quantizable]
    #[param]
    self_attn: Gemma4Attention,
    #[quantizable]
    #[param]
    mlp: Gemma4Mlp,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
    #[param]
    pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    post_feedforward_layernorm: nn::RmsNorm,

    // MoE fields (None unless enable_moe_block=true)
    #[quantizable]
    #[param]
    router: Option<Gemma4Router>,
    #[quantizable]
    #[param]
    experts: Option<Gemma4Experts>,
    #[param]
    pre_feedforward_layernorm_2: Option<nn::RmsNorm>,
    #[param]
    post_feedforward_layernorm_1: Option<nn::RmsNorm>,
    #[param]
    post_feedforward_layernorm_2: Option<nn::RmsNorm>,

    // Per-layer input gating (None unless hidden_size_per_layer_input > 0)
    #[quantizable]
    #[param]
    per_layer_input_gate: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable]
    #[param]
    per_layer_projection: Option<MaybeQuantized<nn::Linear>>,
    #[param]
    post_per_layer_input_norm: Option<nn::RmsNorm>,

    // Learnable scalar multiplied onto each layer's output (init=1)
    #[param]
    layer_scalar: Param<Array>,
}

impl Gemma4Block {
    fn new(args: &Gemma4ModelArgs, layer_idx: i32) -> Result<Self, Exception> {
        let full_attention = args.is_full_attention(layer_idx);
        let is_owner = layer_idx < args.first_kv_shared();

        let attn = if is_owner {
            Gemma4Attention::new_owner(args, full_attention)?
        } else {
            Gemma4Attention::new_shared(args, full_attention)?
        };

        let is_kv_shared = !is_owner;
        let inter = if args.use_double_wide_mlp && is_kv_shared {
            args.intermediate_size * 2
        } else {
            args.intermediate_size
        };
        let mlp = Gemma4Mlp::new(args.hidden_size, inter)?;

        let make_norm = |d: i32| nn::RmsNormBuilder::new(d).eps(args.rms_norm_eps).build();
        let hidden = args.hidden_size;

        let (router, experts, pffn2, pffn1_post, pffn2_post) = if args.enable_moe_block {
            let num_experts = args.num_experts.ok_or_else(|| {
                Exception::custom("enable_moe_block=true but num_experts is None")
            })?;
            let top_k = args.top_k_experts.ok_or_else(|| {
                Exception::custom("enable_moe_block=true but top_k_experts is None")
            })?;
            let moe_inter = args.moe_intermediate_size.ok_or_else(|| {
                Exception::custom("enable_moe_block=true but moe_intermediate_size is None")
            })?;
            (
                Some(Gemma4Router::new(
                    hidden,
                    num_experts,
                    top_k,
                    args.rms_norm_eps,
                )?),
                Some(Gemma4Experts::new(hidden, moe_inter, num_experts)?),
                Some(make_norm(hidden)?),
                Some(make_norm(hidden)?),
                Some(make_norm(hidden)?),
            )
        } else {
            (None, None, None, None, None)
        };

        let (per_layer_input_gate, per_layer_projection, post_per_layer_input_norm) =
            if args.hidden_size_per_layer_input > 0 {
                let hp = args.hidden_size_per_layer_input;
                (
                    Some(MaybeQuantized::Original(
                        nn::LinearBuilder::new(hidden, hp).bias(false).build()?,
                    )),
                    Some(MaybeQuantized::Original(
                        nn::LinearBuilder::new(hp, hidden).bias(false).build()?,
                    )),
                    Some(make_norm(hidden)?),
                )
            } else {
                (None, None, None)
            };

        Ok(Self {
            self_attn: attn,
            mlp,
            input_layernorm: make_norm(hidden)?,
            post_attention_layernorm: make_norm(hidden)?,
            pre_feedforward_layernorm: make_norm(hidden)?,
            post_feedforward_layernorm: make_norm(hidden)?,
            router,
            experts,
            pre_feedforward_layernorm_2: pffn2,
            post_feedforward_layernorm_1: pffn1_post,
            post_feedforward_layernorm_2: pffn2_post,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            layer_scalar: Param::new(Array::ones::<f32>(&[1])?),
        })
    }

    fn forward_owner<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut C>,
        per_layer_input: Option<&Array>,
    ) -> Result<(Array, (Array, Array), i32), Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let (attn_raw, kv, offset) = self.self_attn.forward_owner(&normed, mask, kv_cache)?;
        let attn_normed = self.post_attention_layernorm.forward(&attn_raw)?;
        let mut h = x.add(attn_normed)?;

        let residual2 = h.clone();
        h = self.forward_ff(&h)?;
        h = self.post_feedforward_layernorm.forward(&h)?;
        h = residual2.add(h)?;

        h = self.apply_per_layer_gate(h, per_layer_input)?;

        let scalar = self.layer_scalar.as_dtype(h.dtype())?;
        h = h.multiply(scalar)?;

        Ok((h, kv, offset))
    }

    fn forward_shared(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        shared_kv: &(Array, Array),
        offset: i32,
        per_layer_input: Option<&Array>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_raw = self
            .self_attn
            .forward_shared(&normed, mask, shared_kv, offset)?;
        let attn_normed = self.post_attention_layernorm.forward(&attn_raw)?;
        let mut h = x.add(attn_normed)?;

        let residual2 = h.clone();
        h = self.forward_ff(&h)?;
        h = self.post_feedforward_layernorm.forward(&h)?;
        h = residual2.add(h)?;

        h = self.apply_per_layer_gate(h, per_layer_input)?;

        let scalar = self.layer_scalar.as_dtype(h.dtype())?;
        h.multiply(scalar)
    }

    /// Dense-only or dense+`MoE` feed-forward, with `MoE` outputs summed in parallel.
    fn forward_ff(&mut self, h: &Array) -> Result<Array, Exception> {
        if self.router.is_some() {
            // MoE: dense path and expert path each have their own norms and are summed.
            let pre_ff1 = self.pre_feedforward_layernorm.forward(h)?;
            let h1_raw = self.mlp.forward(&pre_ff1)?;
            let h1 = self
                .post_feedforward_layernorm_1
                .as_mut()
                .ok_or_else(|| Exception::custom("post_feedforward_layernorm_1 missing"))?
                .forward(&h1_raw)?;

            // The router normalizes internally (rms_norm with its own scale), so it
            // receives the RAW residual `h`. Only the experts get the pre-FF2-normed
            // input. (mlx-lm: `router(h)` but `experts(pre_feedforward_layernorm_2(h))`.)
            let (indices, weights) = self
                .router
                .as_mut()
                .ok_or_else(|| Exception::custom("router missing"))?
                .forward(h)?;
            let pre_ff2 = self
                .pre_feedforward_layernorm_2
                .as_mut()
                .ok_or_else(|| Exception::custom("pre_feedforward_layernorm_2 missing"))?
                .forward(h)?;
            let h2_raw = self
                .experts
                .as_mut()
                .ok_or_else(|| Exception::custom("experts missing"))?
                .forward(&pre_ff2, &indices, &weights)?;
            let h2 = self
                .post_feedforward_layernorm_2
                .as_mut()
                .ok_or_else(|| Exception::custom("post_feedforward_layernorm_2 missing"))?
                .forward(&h2_raw)?;

            h1.add(h2)
        } else {
            let pre = self.pre_feedforward_layernorm.forward(h)?;
            self.mlp.forward(&pre)
        }
    }

    fn apply_per_layer_gate(
        &mut self,
        h: Array,
        per_layer_input: Option<&Array>,
    ) -> Result<Array, Exception> {
        if let (Some(gate_proj), Some(proj), Some(norm)) = (
            self.per_layer_input_gate.as_mut(),
            self.per_layer_projection.as_mut(),
            self.post_per_layer_input_norm.as_mut(),
        ) {
            if let Some(pli) = per_layer_input {
                let residual = h.clone();
                let gate_gated = nn::gelu_approximate(gate_proj.forward(&h)?)?.multiply(pli)?;
                let gate_projected = proj.forward(&gate_gated)?;
                let gate_normed = norm.forward(&gate_projected)?;
                return residual.add(gate_normed);
            }
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Text model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma4TextModel {
    #[quantizable]
    #[param]
    embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    layers: Vec<Gemma4Block>,
    #[param]
    norm: nn::RmsNorm,

    #[quantizable]
    #[param]
    embed_tokens_per_layer: Option<MaybeQuantized<nn::Embedding>>,
    // Not quantizable: MLX Gemma 4 checkpoints store this projection in full
    // precision (only `.weight`, no `.scales`), unlike the other Linears.
    #[param]
    per_layer_model_projection: Option<nn::Linear>,
    #[param]
    per_layer_projection_norm: Option<nn::RmsNorm>,

    // Non-parameter cached scalars, refreshed lazily per dtype
    cached_embed_scale: Option<Array>,
    cached_per_layer_embed_scale: Option<Array>,
    cached_per_layer_projection_scale: Option<Array>,

    hidden_size: i32,
    hidden_size_per_layer_input: i32,
    sliding_window: i32,
    num_hidden_layers: i32,
    previous_kvs: Vec<usize>,
    layer_type_is_full: Vec<bool>,
}

impl Gemma4TextModel {
    fn new(args: &Gemma4ModelArgs) -> Result<Self, Exception> {
        if !args.vocab_size.is_positive() {
            return Err(Exception::custom("vocab_size must be positive"));
        }
        if !args.num_hidden_layers.is_positive() {
            return Err(Exception::custom("num_hidden_layers must be positive"));
        }

        let layers = (0..args.num_hidden_layers)
            .map(|i| Gemma4Block::new(args, i))
            .collect::<Result<Vec<_>, _>>()?;

        let layer_type_is_full: Vec<bool> = (0..args.num_hidden_layers)
            .map(|i| args.is_full_attention(i))
            .collect();

        let (embed_tokens_per_layer, per_layer_model_projection, per_layer_projection_norm) =
            if args.hidden_size_per_layer_input > 0 {
                let hp = args.hidden_size_per_layer_input;
                let n32 = args.num_hidden_layers;
                (
                    Some(MaybeQuantized::Original(nn::Embedding::new(
                        args.vocab_size_per_layer_input,
                        n32 * hp,
                    )?)),
                    Some(
                        nn::LinearBuilder::new(args.hidden_size, n32 * hp)
                            .bias(false)
                            .build()?,
                    ),
                    Some(nn::RmsNormBuilder::new(hp).eps(args.rms_norm_eps).build()?),
                )
            } else {
                (None, None, None)
            };

        Ok(Self {
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(
                args.vocab_size,
                args.hidden_size,
            )?),
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            cached_embed_scale: None,
            cached_per_layer_embed_scale: None,
            cached_per_layer_projection_scale: None,
            hidden_size: args.hidden_size,
            hidden_size_per_layer_input: args.hidden_size_per_layer_input,
            sliding_window: args.sliding_window,
            num_hidden_layers: args.num_hidden_layers,
            previous_kvs: args.previous_kvs(),
            layer_type_is_full,
        })
    }

    /// Per-layer input embedding lookup, scaled by `sqrt(hidden_size_per_layer_input)`.
    ///
    /// Returns `None` if `hidden_size_per_layer_input == 0`, otherwise `[B, L, N, hp]`.
    #[allow(non_snake_case, clippy::as_conversions, clippy::cast_precision_loss)]
    fn get_per_layer_inputs(&mut self, inputs: &Array) -> Result<Option<Array>, Exception> {
        if self.hidden_size_per_layer_input <= 0 {
            return Ok(None);
        }
        let embed = self.embed_tokens_per_layer.as_mut().ok_or_else(|| {
            Exception::custom(
                "embed_tokens_per_layer missing despite hidden_size_per_layer_input>0",
            )
        })?;
        let raw = embed.forward(inputs)?;

        let needs_refresh = self
            .cached_per_layer_embed_scale
            .as_ref()
            .is_none_or(|e| e.dtype() != raw.dtype());
        if needs_refresh {
            let scale = (self.hidden_size_per_layer_input as f32).sqrt();
            self.cached_per_layer_embed_scale = Some(array!(scale).as_dtype(raw.dtype())?);
        }
        let scale = self
            .cached_per_layer_embed_scale
            .as_ref()
            .ok_or_else(|| Exception::custom("cached_per_layer_embed_scale not initialized"))?;
        let scaled = raw.multiply(scale)?;

        let shape = scaled.shape();
        let err = || Exception::custom("embed output must have >= 2 dims");
        let B = *shape.first().ok_or_else(err)?;
        let L = *shape.get(1).ok_or_else(err)?;
        let hp = self.hidden_size_per_layer_input;
        let n = self.num_hidden_layers;
        scaled.reshape(&[B, L, n, hp]).map(Some)
    }

    /// Project hidden state to per-layer representations and sum with embedding-based inputs.
    ///
    /// Returns `None` if `hidden_size_per_layer_input == 0`.
    #[allow(non_snake_case, clippy::as_conversions, clippy::cast_precision_loss)]
    fn project_per_layer_inputs(
        &mut self,
        h: &Array,
        per_layer_inputs: Option<Array>,
    ) -> Result<Option<Array>, Exception> {
        if self.hidden_size_per_layer_input <= 0 {
            return Ok(None);
        }

        let proj = self
            .per_layer_model_projection
            .as_mut()
            .ok_or_else(|| Exception::custom("per_layer_model_projection missing"))?;
        let p_raw = proj.forward(h)?;

        let needs_refresh = self
            .cached_per_layer_projection_scale
            .as_ref()
            .is_none_or(|e| e.dtype() != p_raw.dtype());
        if needs_refresh {
            let scale = (self.hidden_size as f32).sqrt().recip();
            self.cached_per_layer_projection_scale = Some(array!(scale).as_dtype(p_raw.dtype())?);
        }
        let scale = self
            .cached_per_layer_projection_scale
            .as_ref()
            .ok_or_else(|| {
                Exception::custom("cached_per_layer_projection_scale not initialized")
            })?;
        let p_scaled = p_raw.multiply(scale)?;

        let shape = p_scaled.shape();
        let err = || Exception::custom("proj output must have >= 2 dims");
        let B = *shape.first().ok_or_else(err)?;
        let L = *shape.get(1).ok_or_else(err)?;
        let n = self.num_hidden_layers;
        let hp = self.hidden_size_per_layer_input;
        let p_r = p_scaled.reshape(&[B, L, n, hp])?;

        let norm = self
            .per_layer_projection_norm
            .as_mut()
            .ok_or_else(|| Exception::custom("per_layer_projection_norm missing"))?;
        let p_normed = norm.forward(&p_r)?;

        let result = if let Some(pli) = per_layer_inputs {
            let combined = p_normed.add(pli)?;
            combined.multiply(array!(2.0_f32.sqrt().recip()).as_dtype(combined.dtype())?)?
        } else {
            p_normed
        };
        Ok(Some(result))
    }

    #[allow(
        non_snake_case,
        clippy::too_many_lines,
        clippy::as_conversions,
        clippy::cast_precision_loss
    )]
    fn forward_internal<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        let mut h = self.embed_tokens.forward(inputs)?;

        let needs_refresh = self
            .cached_embed_scale
            .as_ref()
            .is_none_or(|e| e.dtype() != h.dtype());
        if needs_refresh {
            let scale = (self.hidden_size as f32).sqrt();
            self.cached_embed_scale = Some(array!(scale).as_dtype(h.dtype())?);
        }
        let embed_scale = self
            .cached_embed_scale
            .as_ref()
            .ok_or_else(|| Exception::custom("cached_embed_scale not initialized"))?;
        h = h.multiply(embed_scale)?;

        if cache.is_empty() {
            // Cache initialization happens externally via AnyModel::make_cache.
            // For a bare forward call without prior initialization, fill with None.
            // Shared layers never write to their cache slot (they use intermediates).
            *cache = (0..self.layers.len()).map(|_| None).collect();
        } else if cache.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "kv_cache length ({}) must match num layers ({})",
                cache.len(),
                self.layers.len()
            )));
        }

        let T = *h
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("hidden state must have >= 2 dims"))?;

        let offset = cache
            .iter()
            .find_map(Option::as_ref)
            .map_or(0, KeyValueCache::offset);

        let full_mask = (T > 1)
            .then(|| create_causal_mask(T, Some(offset)))
            .transpose()?;
        let sliding_mask = if T > 1 {
            Some(create_windowed_causal_mask(T, offset, self.sliding_window)?)
        } else {
            let kv_len = offset + 1;
            (kv_len > self.sliding_window)
                .then(|| create_windowed_causal_mask(1, offset, self.sliding_window))
                .transpose()?
        };

        let pli_all = self.get_per_layer_inputs(inputs)?;
        let projected = self.project_per_layer_inputs(&h, pli_all)?;

        let n_layers = self.layers.len();
        let mut intermediates: Vec<Option<((Array, Array), i32)>> = vec![None; n_layers];

        for i in 0..n_layers {
            let i32_i = i32::try_from(i).map_err(|_| Exception::custom("layer idx overflow"))?;
            let is_full = *self
                .layer_type_is_full
                .get(i)
                .ok_or_else(|| Exception::custom("layer_type_is_full out of bounds"))?;

            let mask = if is_full {
                full_mask.as_ref()
            } else {
                sliding_mask.as_ref()
            };

            let per_layer_in = projected.as_ref().map(|p| p.index((.., .., i32_i, ..)));

            let prev_idx = *self
                .previous_kvs
                .get(i)
                .ok_or_else(|| Exception::custom("previous_kvs out of bounds"))?;

            let layer = self
                .layers
                .get_mut(i)
                .ok_or_else(|| Exception::custom("layer index out of bounds"))?;

            if layer.self_attn.is_owner {
                let cache_ref = cache
                    .get_mut(i)
                    .ok_or_else(|| Exception::custom("cache slot out of bounds"))?
                    .as_mut();

                let (new_h, kv, off) =
                    layer.forward_owner(&h, mask, cache_ref, per_layer_in.as_ref())?;
                h = new_h;
                if let Some(slot) = intermediates.get_mut(i) {
                    *slot = Some((kv, off));
                }
            } else {
                let (shared_kv, shared_offset) = intermediates
                    .get(prev_idx)
                    .and_then(|opt| opt.as_ref())
                    .ok_or_else(|| {
                        Exception::custom(format!(
                            "layer {i} requires intermediates[{prev_idx}] but it is not available"
                        ))
                    })?;

                h = layer.forward_shared(
                    &h,
                    mask,
                    shared_kv,
                    *shared_offset,
                    per_layer_in.as_ref(),
                )?;
            }
        }

        self.norm.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Causal LM
// ---------------------------------------------------------------------------

/// Gemma 4 causal language model.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Gemma4CausalLM {
    pub args: Gemma4ModelArgs,

    #[quantizable]
    #[param]
    model: Gemma4TextModel,

    #[quantizable]
    #[param]
    lm_head: Option<MaybeQuantized<nn::Linear>>,

    // Cached dtype-specific scalars for final logit soft-capping
    cached_final_inv_cap: Option<Array>,
    cached_final_cap: Option<Array>,
}

impl Gemma4CausalLM {
    pub fn new(args: Gemma4ModelArgs) -> Result<Self, Exception> {
        let model = Gemma4TextModel::new(&args)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        };
        Ok(Self {
            args,
            model,
            lm_head,
            cached_final_inv_cap: None,
            cached_final_cap: None,
        })
    }

    fn project_hidden(&mut self, hidden: &Array) -> Result<Array, Exception> {
        let mut logits = match self.lm_head.as_mut() {
            Some(head) => head.forward(hidden)?,
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed) => embed.as_linear(hidden)?,
                MaybeQuantized::Quantized(q_embed) => q_embed.as_linear(hidden)?,
            },
        };

        let cap = self.args.final_logit_softcapping;
        if cap > 0.0 {
            let needs_refresh = self
                .cached_final_inv_cap
                .as_ref()
                .is_none_or(|c| c.dtype() != logits.dtype());
            if needs_refresh {
                self.cached_final_inv_cap = Some(array!(1.0 / cap).as_dtype(logits.dtype())?);
                self.cached_final_cap = Some(array!(cap).as_dtype(logits.dtype())?);
            }
            let inv_cap = self
                .cached_final_inv_cap
                .as_ref()
                .ok_or_else(|| Exception::custom("cached_final_inv_cap not initialized"))?;
            let cap_arr = self
                .cached_final_cap
                .as_ref()
                .ok_or_else(|| Exception::custom("cached_final_cap not initialized"))?;
            logits = ops::tanh(&logits.multiply(inv_cap)?)?.multiply(cap_arr)?;
        }
        Ok(logits)
    }

    pub fn forward<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        let _ = mask;
        let hidden_all = self.model.forward_internal(inputs, kv_cache)?;
        let seq_len = inputs.shape().get(1).copied().unwrap_or(1);
        let lm_input = if seq_len > 1 {
            hidden_all.index((.., -1.., ..))
        } else {
            hidden_all
        };
        self.project_hidden(&lm_input)
    }

    pub fn forward_all_logits<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        let _ = mask;
        let hidden = self.model.forward_internal(inputs, kv_cache)?;
        self.project_hidden(&hidden)
    }

    pub fn forward_hidden<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        let _ = mask;
        self.model.forward_internal(inputs, kv_cache)
    }
}

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct Gemma4TopLevel {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    text_config: Option<serde_json::Value>,
    #[serde(default)]
    quantization: Option<crate::gemma2::QuantizationConfig>,
}

/// Load `Gemma4ModelArgs` from a model directory.
///
/// Supports both flat and `text_config`-wrapped HF config formats.
pub fn load_gemma4_model_args<P: AsRef<Path>>(model_dir: P) -> Result<Gemma4ModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let text = std::fs::read_to_string(config_path)?;
    let raw: serde_json::Value = serde_json::from_str(&text)?;
    let top: Gemma4TopLevel = serde_json::from_str(&text)?;

    let mut args: Gemma4ModelArgs = if let Some(inner) = top.text_config {
        let mut a: Gemma4ModelArgs = serde_json::from_value(inner)?;
        if a.model_type.is_empty() {
            if let Some(mt) = top.model_type {
                a.model_type = mt;
            }
        }
        if a.quantization.is_none() {
            a.quantization = top.quantization;
        }
        a
    } else {
        serde_json::from_value(raw)?
    };

    if args.model_type.is_empty() {
        "gemma4_text".clone_into(&mut args.model_type);
    }

    Ok(args)
}

/// Load a Gemma 4 model from a directory.
///
/// Gemma 4 uses standard `RMSNorm` — no +1 weight shifting is applied.
pub fn load_gemma4_model<P: AsRef<Path>>(model_dir: P) -> Result<Gemma4CausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_gemma4_model_args(model_path)?;

    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_kv_shared_layers = args.num_kv_shared_layers,
        first_kv_shared = args.first_kv_shared(),
        enable_moe = args.enable_moe_block,
        has_per_layer_input = args.hidden_size_per_layer_input > 0,
        "Loading Gemma 4 model"
    );

    let quantization = args.quantization.clone();
    let enable_moe = args.enable_moe_block;
    let raw_model = Gemma4CausalLM::new(args)?;

    let mut model = if let Some(ref qc) = quantization {
        tracing::info!(
            group_size = qc.group_size,
            bits = qc.bits,
            "Applying quantization"
        );
        mlx_rs::nn::quantize(raw_model, qc.group_size, qc.bits).map_err(|e| {
            ModelError::ShapeMismatch(format!("Failed to quantize model structure: {e}"))
        })?
    } else {
        raw_model
    };

    // `gemma4` (multimodal wrapper) checkpoints nest the text model under
    // `language_model.`; `gemma4_text` checkpoints start at `model.`. Strip the
    // prefix when present so both load, and skip the vision/audio tower weights.
    crate::load_quantized_safetensors_weights_optional_prefix(
        &mut model,
        model_path,
        quantization.is_some(),
        "language_model.",
    )?;

    if enable_moe {
        load_gemma4_moe_expert_weights(&mut model, model_path, quantization.is_some())?;
    }

    tracing::info!("Gemma 4 model loaded successfully");
    Ok(model)
}

/// Load and remap fused `MoE` expert weights for Gemma 4.
///
/// The checkpoint stores experts in mlx `SwitchGLU` layout:
/// - `model.layers.{i}.experts.gate_up_proj` `[E, 2*M, hidden]` (fused gate+up,
///   `SwitchLinear` `[num_experts, out, in]`)
/// - `model.layers.{i}.experts.down_proj` `[E, hidden, M]`
///
/// higgs uses stacked `nn::Linear` weights instead:
/// - `gate_proj.weight` / `up_proj.weight` `[E*M, hidden]`
/// - `down_proj.weight` `[E*hidden, M]`
///
/// This splits `gate_up_proj` along axis -2 into gate/up `[E, M, hidden]`, reshapes
/// each to `[E*M, hidden]`, and reshapes `down_proj` `[E, hidden, M]` to `[E*hidden, M]`.
/// Per-expert grouping is preserved: expert `e`'s rows occupy `[e*M, (e+1)*M)`.
///
/// The standard loader runs first and already loads everything that matches directly
/// (router params, norms, attention). This second pass only handles the fused expert
/// tensors that the direct match misses, then re-evals.
fn load_gemma4_moe_expert_weights(
    model: &mut Gemma4CausalLM,
    model_path: &Path,
    quantized: bool,
) -> Result<(), ModelError> {
    if quantized {
        // Packed uint32 expert tensors (+ scales/biases) cannot be reshaped along the
        // expert axis without dequantizing; without a reference checkpoint to validate
        // against, loading them would silently corrupt the experts. Reject instead.
        return Err(ModelError::UnsupportedModel(
            "quantized-expert Gemma 4 MoE is not yet supported".to_owned(),
        ));
    }

    let files = crate::collect_safetensors_files(model_path)?;
    let mut params = model.parameters_mut().flatten();
    let mut remapped = 0usize;

    for file_path in &files {
        let loaded = Array::load_safetensors(file_path)
            .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

        for (key, value) in loaded {
            if let Some(prefix) = key.strip_suffix(".experts.gate_up_proj") {
                let (gate_key, up_key, gate_w, up_w) = split_fused_gate_up(prefix, &value)?;
                assign_param(&mut params, &gate_key, gate_w)?;
                assign_param(&mut params, &up_key, up_w)?;
                remapped += 2;
            } else if let Some(prefix) = key.strip_suffix(".experts.down_proj") {
                let (down_key, down_w) = reshape_down(prefix, &value)?;
                assign_param(&mut params, &down_key, down_w)?;
                remapped += 1;
            }
        }
    }

    drop(params);
    model
        .eval()
        .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;
    tracing::info!(remapped, "Remapped fused MoE expert weights");
    Ok(())
}

/// Split a fused `gate_up_proj` `[E, 2*M, hidden]` into gate/up `[E*M, hidden]` each.
#[allow(non_snake_case)]
fn split_fused_gate_up(
    prefix: &str,
    fused: &Array,
) -> Result<(String, String, Array, Array), ModelError> {
    let shape = fused.shape();
    let [E, two_m, hidden] = *shape else {
        return Err(ModelError::ShapeMismatch(format!(
            "{prefix}.experts.gate_up_proj expected 3D [E, 2*M, hidden], got {shape:?}"
        )));
    };
    if two_m % 2 != 0 {
        return Err(ModelError::ShapeMismatch(format!(
            "{prefix}.experts.gate_up_proj middle dim {two_m} is not even"
        )));
    }
    let m = two_m / 2;

    let parts = fused.split(2, 1).map_err(ModelError::Mlx)?;
    let [gate, up] = <[Array; 2]>::try_from(parts).map_err(|_| {
        ModelError::ShapeMismatch("gate_up_proj split did not yield 2 parts".to_owned())
    })?;

    let gate_w = gate.reshape(&[E * m, hidden]).map_err(ModelError::Mlx)?;
    let up_w = up.reshape(&[E * m, hidden]).map_err(ModelError::Mlx)?;

    Ok((
        format!("{prefix}.experts.gate_proj.weight"),
        format!("{prefix}.experts.up_proj.weight"),
        gate_w,
        up_w,
    ))
}

/// Reshape `down_proj` `[E, hidden, M]` to `[E*hidden, M]`.
#[allow(non_snake_case)]
fn reshape_down(prefix: &str, down: &Array) -> Result<(String, Array), ModelError> {
    let shape = down.shape();
    let [E, hidden, m] = *shape else {
        return Err(ModelError::ShapeMismatch(format!(
            "{prefix}.experts.down_proj expected 3D [E, hidden, M], got {shape:?}"
        )));
    };
    let down_w = down.reshape(&[E * hidden, m]).map_err(ModelError::Mlx)?;
    Ok((format!("{prefix}.experts.down_proj.weight"), down_w))
}

/// Assign a remapped weight into the flattened parameter map, erroring on shape mismatch.
fn assign_param(
    params: &mut std::collections::HashMap<std::rc::Rc<str>, &mut Array>,
    key: &str,
    value: Array,
) -> Result<(), ModelError> {
    let param = params.get_mut(key).ok_or_else(|| {
        ModelError::MissingWeight(format!(
            "remapped MoE key {key} not found in model parameters"
        ))
    })?;
    if param.shape() != value.shape() {
        return Err(ModelError::ShapeMismatch(format!(
            "remapped MoE key {key}: model expects {:?}, got {:?}",
            param.shape(),
            value.shape()
        )));
    }
    **param = value;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::redundant_type_annotations,
    clippy::shadow_unrelated,
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::suboptimal_flops,
    clippy::unnecessary_cast,
    clippy::cast_lossless,
    clippy::doc_markdown,
    clippy::float_cmp
)]
mod tests {
    use super::*;
    use crate::cache::SteppingKeyValueCache;

    fn small_args() -> Gemma4ModelArgs {
        Gemma4ModelArgs {
            model_type: "gemma4_text".to_owned(),
            hidden_size: 64,
            num_hidden_layers: 5,
            intermediate_size: 128,
            num_attention_heads: 4,
            head_dim: 16,
            global_head_dim: 32,
            global_partial_rotary_factor: 0.25,
            rms_norm_eps: 1e-6,
            vocab_size: 256,
            vocab_size_per_layer_input: 256,
            num_key_value_heads: 1,
            num_global_key_value_heads: None,
            num_kv_shared_layers: 2,
            hidden_size_per_layer_input: 0,
            partial_rotary_factor: 1.0,
            rope_parameters: None,
            sliding_window: 8,
            sliding_window_pattern: 5,
            max_position_embeddings: 128,
            attention_k_eq_v: false,
            final_logit_softcapping: 30.0,
            use_double_wide_mlp: true,
            enable_moe_block: false,
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
            layer_types: None,
            tie_word_embeddings: true,
            quantization: None,
        }
    }

    fn small_model_args() -> Gemma4ModelArgs {
        Gemma4ModelArgs {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_attention_heads: 2,
            head_dim: 8,
            global_head_dim: 16,
            num_key_value_heads: 1,
            sliding_window: 8,
            use_double_wide_mlp: false,
            // Set to 0 so no layer is promoted to full attention.
            // With 5 layers and 2 shared, all shared layers are sliding
            // and can reference the last sliding owner (layer 2).
            sliding_window_pattern: 0,
            ..small_args()
        }
    }

    // -----------------------------------------------------------------------
    // Parameter-key naming (must match HF/MLX checkpoint keys)
    // -----------------------------------------------------------------------

    /// Weight keys must match the checkpoint layout, or weights silently load as
    /// random init. Forward unit tests use in-memory weights and cannot catch
    /// this — only real checkpoint loading (or this test) does.
    #[test]
    fn parameter_keys_match_checkpoint_naming() {
        let mut args = small_model_args();
        args.hidden_size_per_layer_input = 16;
        let mut model = Gemma4CausalLM::new(args).unwrap();
        let params = model.parameters_mut().flatten();
        let keys: Vec<String> = params.keys().map(ToString::to_string).collect();

        assert!(
            keys.iter().any(|k| k.contains("self_attn.q_proj")),
            "attention submodule must be named `self_attn` to match checkpoint keys"
        );
        let stray: Vec<&String> = keys.iter().filter(|k| k.contains(".attn.")).collect();
        assert!(stray.is_empty(), "no param key may use `.attn.`: {stray:?}");

        assert!(
            keys.iter()
                .any(|k| k == "model.per_layer_model_projection.weight"),
            "per_layer_model_projection must be an unquantized plain Linear \
             (`.weight`, no `.inner.`)"
        );
    }

    // -----------------------------------------------------------------------
    // Config deserialization
    // -----------------------------------------------------------------------

    #[test]
    fn config_flat_deserialization() {
        let json = r#"{
            "model_type": "gemma4_text",
            "hidden_size": 1536,
            "num_hidden_layers": 35,
            "intermediate_size": 6144,
            "num_attention_heads": 8,
            "head_dim": 256,
            "global_head_dim": 512,
            "rms_norm_eps": 1e-6,
            "vocab_size": 262144,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 20,
            "sliding_window": 512,
            "sliding_window_pattern": 5,
            "tie_word_embeddings": true
        }"#;
        let args: Gemma4ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "gemma4_text");
        assert_eq!(args.hidden_size, 1536);
        assert_eq!(args.num_hidden_layers, 35);
        assert_eq!(args.global_head_dim, 512);
        assert_eq!(args.num_kv_shared_layers, 20);
        assert_eq!(args.sliding_window_pattern, 5);
        assert!(args.tie_word_embeddings);
        assert!(!args.enable_moe_block);
    }

    #[test]
    fn config_text_config_wrapper() {
        let json = r#"{
            "model_type": "gemma4",
            "text_config": {
                "hidden_size": 1536,
                "num_hidden_layers": 35,
                "intermediate_size": 6144,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-6,
                "vocab_size": 262144,
                "num_kv_shared_layers": 20,
                "sliding_window_pattern": 5
            }
        }"#;
        let top: Gemma4TopLevel = serde_json::from_str(json).unwrap();
        assert!(top.text_config.is_some());
        let inner: Gemma4ModelArgs = serde_json::from_value(top.text_config.unwrap()).unwrap();
        assert_eq!(inner.hidden_size, 1536);
        assert_eq!(inner.num_kv_shared_layers, 20);
    }

    #[test]
    fn config_moe_variant() {
        let json = r#"{
            "model_type": "gemma4_text",
            "hidden_size": 2048,
            "num_hidden_layers": 46,
            "intermediate_size": 8192,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "vocab_size": 262144,
            "num_kv_shared_layers": 35,
            "enable_moe_block": true,
            "num_experts": 128,
            "top_k_experts": 2,
            "moe_intermediate_size": 1024
        }"#;
        let args: Gemma4ModelArgs = serde_json::from_str(json).unwrap();
        assert!(args.enable_moe_block);
        assert_eq!(args.num_experts, Some(128));
        assert_eq!(args.top_k_experts, Some(2));
        assert_eq!(args.moe_intermediate_size, Some(1024));
    }

    #[test]
    fn config_defaults_for_rope_and_layer_types() {
        let args = small_args();
        assert!(args.rope_parameters.is_none());
        assert!(args.layer_types.is_none());

        for i in 0..4 {
            assert!(!args.is_full_attention(i), "layer {i} should be sliding");
        }
        assert!(args.is_full_attention(4), "layer 4 should be full");

        let (theta_full, prf_full) = args.rope_params_for(true);
        assert!((theta_full - 1_000_000.0).abs() < 1.0);
        assert!((prf_full - 0.25).abs() < 1e-6);

        let (theta_slide, prf_slide) = args.rope_params_for(false);
        assert!((theta_slide - 10_000.0).abs() < 1.0);
        assert!((prf_slide - 1.0).abs() < 1e-6);
    }

    #[test]
    fn config_per_layer_edge_model() {
        let json = r#"{
            "model_type": "gemma4_text",
            "hidden_size": 1024,
            "num_hidden_layers": 18,
            "intermediate_size": 4096,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "vocab_size": 262144,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 256,
            "vocab_size_per_layer_input": 262144
        }"#;
        let args: Gemma4ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.hidden_size_per_layer_input, 256);
        assert_eq!(args.vocab_size_per_layer_input, 262_144);
        assert_eq!(args.num_kv_shared_layers, 0);
        assert_eq!(args.first_kv_shared(), 18);
    }

    // -----------------------------------------------------------------------
    // is_full_attention
    // -----------------------------------------------------------------------

    #[test]
    fn is_full_attention_pattern_5() {
        let args = small_args();
        for i in 0..4 {
            assert!(!args.is_full_attention(i));
        }
        assert!(args.is_full_attention(4));
    }

    #[test]
    fn is_full_attention_explicit_layer_types() {
        let mut args = small_args();
        args.layer_types = Some(vec![
            "sliding_attention".to_owned(),
            "full_attention".to_owned(),
            "sliding_attention".to_owned(),
        ]);
        assert!(!args.is_full_attention(0));
        assert!(args.is_full_attention(1));
        assert!(!args.is_full_attention(2));
    }

    // -----------------------------------------------------------------------
    // first_kv_shared
    // -----------------------------------------------------------------------

    #[test]
    fn first_kv_shared_basic() {
        let args = small_args();
        assert_eq!(args.first_kv_shared(), 3);
    }

    #[test]
    fn first_kv_shared_zero_shared() {
        let mut args = small_args();
        args.num_kv_shared_layers = 0;
        assert_eq!(args.first_kv_shared(), args.num_hidden_layers);
    }

    #[test]
    fn first_kv_shared_all_shared() {
        let mut args = small_args();
        args.num_kv_shared_layers = args.num_hidden_layers;
        assert_eq!(args.first_kv_shared(), 0);
    }

    // -----------------------------------------------------------------------
    // previous_kvs
    // -----------------------------------------------------------------------

    #[test]
    fn previous_kvs_basic() {
        let mut args = small_args();
        args.num_hidden_layers = 8;
        args.num_kv_shared_layers = 3;
        args.sliding_window_pattern = 5;

        let pvs = args.previous_kvs();
        for (i, &pv) in pvs.iter().enumerate().take(5) {
            assert_eq!(pv, i);
        }
        // Layers 5,6,7 are sliding and map to last sliding owner (3)
        assert_eq!(pvs[5], 3);
        assert_eq!(pvs[6], 3);
        assert_eq!(pvs[7], 3);
    }

    #[test]
    fn previous_kvs_with_full_shared_layer() {
        let mut args = small_args();
        args.num_hidden_layers = 6;
        args.num_kv_shared_layers = 2;
        args.sliding_window_pattern = 3;

        // Full at layers 2, 5 (pattern 3: (i+1)%3==0)
        // Owners: 0,1,2,3; last_full=2, last_sliding=3
        // Shared: layer 4 (sliding -> 3), layer 5 (full -> 2)
        let pvs = args.previous_kvs();
        assert_eq!(pvs[4], 3);
        assert_eq!(pvs[5], 2);
    }

    // -----------------------------------------------------------------------
    // head_dim / n_kv_heads selection
    // -----------------------------------------------------------------------

    #[test]
    fn head_dim_selection() {
        let args = small_args();
        assert_eq!(args.head_dim_for(false), 16);
        assert_eq!(args.head_dim_for(true), 32);
    }

    #[test]
    fn n_kv_heads_k_eq_v() {
        let mut args = small_args();
        args.attention_k_eq_v = true;
        args.num_global_key_value_heads = Some(2);
        assert_eq!(args.n_kv_heads_for(true), 2);
        assert_eq!(args.n_kv_heads_for(false), 1);
    }

    // -----------------------------------------------------------------------
    // Model construction
    // -----------------------------------------------------------------------

    #[test]
    fn model_construction_tied_embeddings() {
        let args = small_args();
        let model = Gemma4CausalLM::new(args).unwrap();
        assert!(model.lm_head.is_none());
    }

    #[test]
    fn model_construction_untied_embeddings() {
        let mut args = small_args();
        args.tie_word_embeddings = false;
        let model = Gemma4CausalLM::new(args).unwrap();
        assert!(model.lm_head.is_some());
    }

    #[test]
    fn model_rejects_zero_vocab_size() {
        let mut args = small_args();
        args.vocab_size = 0;
        assert!(Gemma4CausalLM::new(args).is_err());
    }

    #[test]
    fn model_rejects_zero_layers() {
        let mut args = small_args();
        args.num_hidden_layers = 0;
        assert!(Gemma4CausalLM::new(args).is_err());
    }

    // -----------------------------------------------------------------------
    // Final logit soft-capping
    // -----------------------------------------------------------------------

    #[test]
    fn final_logit_softcap_bounded() {
        let x = Array::from_slice(&[1000.0_f32, -1000.0, 0.0, 15.0], &[1, 4]);
        let cap = 30.0_f32;
        let capped = ops::tanh(x.multiply(array!(1.0 / cap)).unwrap())
            .unwrap()
            .multiply(array!(cap))
            .unwrap();
        mlx_rs::transforms::eval([&capped]).unwrap();
        let vals: Vec<f32> = capped.as_slice().to_vec();
        for v in &vals {
            assert!(v.abs() <= cap);
        }
        assert!(vals[2].abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Smoke: forward passes
    // -----------------------------------------------------------------------

    #[test]
    fn smoke_forward_dense_kv_share() {
        let mut model = Gemma4CausalLM::new(small_model_args()).unwrap();

        let input = Array::from_slice(&[0i32, 1, 2, 3], &[1, 4]);
        let mut kv_cache: Vec<Option<SteppingKeyValueCache>> = vec![];
        let logits = model.forward(&input, None, &mut kv_cache).unwrap();

        let shape = logits.shape();
        assert_eq!(shape[0], 1);
        assert_eq!(shape[2], 64);

        mlx_rs::transforms::eval([&logits]).unwrap();
        let vals: Vec<f32> = logits.as_slice().to_vec();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn smoke_forward_decode_step() {
        let mut model = Gemma4CausalLM::new(small_model_args()).unwrap();

        let prefill = Array::from_slice(&[0i32, 1, 2], &[1, 3]);
        let mut kv_cache: Vec<Option<SteppingKeyValueCache>> = vec![];
        let _ = model.forward(&prefill, None, &mut kv_cache).unwrap();

        let token = Array::from_slice(&[3i32], &[1, 1]);
        let logits = model.forward(&token, None, &mut kv_cache).unwrap();
        mlx_rs::transforms::eval([&logits]).unwrap();
        let vals: Vec<f32> = logits.as_slice().to_vec();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn smoke_forward_per_layer_input() {
        let mut args = small_model_args();
        args.hidden_size_per_layer_input = 8;
        args.vocab_size_per_layer_input = 64;
        args.num_kv_shared_layers = 0; // all own

        let mut model = Gemma4CausalLM::new(args).unwrap();

        let input = Array::from_slice(&[0i32, 1, 2], &[1, 3]);
        let mut kv_cache: Vec<Option<SteppingKeyValueCache>> = vec![];
        let logits = model.forward(&input, None, &mut kv_cache).unwrap();
        mlx_rs::transforms::eval([&logits]).unwrap();
        let vals: Vec<f32> = logits.as_slice().to_vec();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn smoke_forward_all_logits() {
        let mut model = Gemma4CausalLM::new(small_model_args()).unwrap();

        let input = Array::from_slice(&[0i32, 1, 2, 3], &[1, 4]);
        let mut kv_cache: Vec<Option<SteppingKeyValueCache>> = vec![];
        let logits = model
            .forward_all_logits(&input, None, &mut kv_cache)
            .unwrap();

        let shape = logits.shape();
        assert_eq!(shape[0], 1);
        assert_eq!(shape[1], 4);
        assert_eq!(shape[2], 64);

        mlx_rs::transforms::eval([&logits]).unwrap();
        let vals: Vec<f32> = logits.as_slice().to_vec();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    // -----------------------------------------------------------------------
    // RoPE dims
    // -----------------------------------------------------------------------

    #[test]
    fn rope_dims_partial_rotary_global() {
        let args = small_args();
        let (_, prf) = args.rope_params_for(true);
        let rope_dims = (32.0_f32 * prf).round() as i32;
        assert_eq!(rope_dims, 8); // 32 * 0.25 = 8
    }

    #[test]
    fn rope_dims_full_rotary_sliding() {
        let args = small_args();
        let (_, prf) = args.rope_params_for(false);
        let rope_dims = (16.0_f32 * prf).round() as i32;
        assert_eq!(rope_dims, 16); // 16 * 1.0 = 16
    }

    // -----------------------------------------------------------------------
    // MoE expert-weight remap (fused SwitchGLU layout -> stacked Linear)
    // -----------------------------------------------------------------------

    #[allow(non_snake_case)]
    fn moe_args() -> Gemma4ModelArgs {
        Gemma4ModelArgs {
            enable_moe_block: true,
            num_experts: Some(4),
            top_k_experts: Some(2),
            moe_intermediate_size: Some(16),
            num_hidden_layers: 2,
            num_kv_shared_layers: 0,
            sliding_window_pattern: 0,
            ..small_model_args()
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn moe_remap_gate_up_shapes_and_grouping() {
        // Fused gate_up_proj is [E, 2*M, hidden]. Build a tensor where each
        // expert/row has a unique value so we can verify per-expert grouping
        // survives the split + reshape.
        let E = 3;
        let M = 4;
        let hidden = 5;
        let two_m = 2 * M;

        // value[e, r, c] = e*1000 + r*10 + c  -> easy to identify origin
        let mut data = Vec::with_capacity((E * two_m * hidden) as usize);
        for e in 0..E {
            for r in 0..two_m {
                for c in 0..hidden {
                    data.push((e * 1000 + r * 10 + c) as f32);
                }
            }
        }
        let fused = Array::from_slice(&data, &[E, two_m, hidden]);

        let (gate_key, up_key, gate_w, up_w) = split_fused_gate_up("m.l.0", &fused).unwrap();
        assert_eq!(gate_key, "m.l.0.experts.gate_proj.weight");
        assert_eq!(up_key, "m.l.0.experts.up_proj.weight");
        assert_eq!(gate_w.shape(), &[E * M, hidden]);
        assert_eq!(up_w.shape(), &[E * M, hidden]);

        mlx_rs::transforms::eval([&gate_w, &up_w]).unwrap();
        let gate_vals: Vec<f32> = gate_w.as_slice().to_vec();
        let up_vals: Vec<f32> = up_w.as_slice().to_vec();

        // Gate is the first M of the 2M output rows for each expert; expert e's gate
        // rows must land at [e*M, (e+1)*M). Row r of expert e, col c -> e*1000+r*10+c.
        for e in 0..E {
            for r in 0..M {
                for c in 0..hidden {
                    let flat = ((e * M + r) * hidden + c) as usize;
                    assert_eq!(gate_vals[flat], (e * 1000 + r * 10 + c) as f32);
                    // Up rows are the second half: original row index r + M.
                    assert_eq!(up_vals[flat], (e * 1000 + (r + M) * 10 + c) as f32);
                }
            }
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn moe_remap_down_shape_and_grouping() {
        let E = 3;
        let hidden = 5;
        let M = 4;

        let mut data = Vec::with_capacity((E * hidden * M) as usize);
        for e in 0..E {
            for r in 0..hidden {
                for c in 0..M {
                    data.push((e * 1000 + r * 10 + c) as f32);
                }
            }
        }
        let down = Array::from_slice(&data, &[E, hidden, M]);

        let (down_key, down_w) = reshape_down("m.l.0", &down).unwrap();
        assert_eq!(down_key, "m.l.0.experts.down_proj.weight");
        assert_eq!(down_w.shape(), &[E * hidden, M]);

        mlx_rs::transforms::eval([&down_w]).unwrap();
        let vals: Vec<f32> = down_w.as_slice().to_vec();
        for e in 0..E {
            for r in 0..hidden {
                for c in 0..M {
                    let flat = ((e * hidden + r) * M + c) as usize;
                    assert_eq!(vals[flat], (e * 1000 + r * 10 + c) as f32);
                }
            }
        }
    }

    #[test]
    fn moe_remap_rejects_wrong_rank() {
        let bad = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(split_fused_gate_up("m.l.0", &bad).is_err());
        assert!(reshape_down("m.l.0", &bad).is_err());
    }

    #[test]
    fn moe_remap_rejects_odd_gate_up_middle_dim() {
        // 2*M dim must be even; an odd middle dimension is malformed.
        let bad = Array::from_slice(&[0.0_f32; 15], &[1, 3, 5]);
        assert!(split_fused_gate_up("m.l.0", &bad).is_err());
    }

    #[test]
    fn router_applies_per_expert_scale() {
        let hidden = 8;
        let num_experts = 4;
        let top_k = 2;
        let mut router = Gemma4Router::new(hidden, num_experts, top_k, 1e-6).unwrap();

        // Default per_expert_scale is ones -> weights unchanged and sum to 1 (softmax).
        let x = Array::ones::<f32>(&[1, 1, hidden]).unwrap();
        let (idx, w) = router.forward(&x).unwrap();
        assert_eq!(idx.shape(), &[1, 1, top_k]);
        assert_eq!(w.shape(), &[1, 1, top_k]);
        mlx_rs::transforms::eval([&w]).unwrap();
        let w_sum: f32 = w.as_slice::<f32>().iter().sum();
        assert!((w_sum - 1.0).abs() < 1e-4);

        // Doubling per_expert_scale doubles the resulting weights.
        *router.per_expert_scale = Array::full::<f32>(&[num_experts], array!(2.0_f32)).unwrap();
        router.cached_norm_weight = None;
        let (_, w2) = router.forward(&x).unwrap();
        mlx_rs::transforms::eval([&w2]).unwrap();
        let w2_sum: f32 = w2.as_slice::<f32>().iter().sum();
        assert!((w2_sum - 2.0).abs() < 1e-4);
    }

    #[test]
    fn smoke_forward_moe() {
        let mut model = Gemma4CausalLM::new(moe_args()).unwrap();

        let input = Array::from_slice(&[0i32, 1, 2, 3], &[1, 4]);
        let mut kv_cache: Vec<Option<SteppingKeyValueCache>> = vec![];
        let logits = model.forward(&input, None, &mut kv_cache).unwrap();

        let shape = logits.shape();
        assert_eq!(shape[0], 1);
        assert_eq!(shape[2], 64);

        mlx_rs::transforms::eval([&logits]).unwrap();
        let vals: Vec<f32> = logits.as_slice().to_vec();
        assert!(vals.iter().all(|v| v.is_finite()));
    }
}
