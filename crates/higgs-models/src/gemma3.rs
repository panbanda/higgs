// Model forward passes eval under the engine MLX gate (structurally on-gate); see clippy.toml.
#![allow(clippy::disallowed_methods)]

//! Gemma 3 (text) model implementation.
//!
//! Key differences from Gemma 2:
//! - QK-norm (per-head `RMSNorm` on queries and keys, same +1 convention)
//! - Dual `RoPE`: local layers use `theta=10_000`, global layers use `theta=1_000_000`
//! - `query_pre_attn_scalar`-based attention scale (no soft-capping)
//! - `clip_residual` on both residual adds to prevent fp16 overflow
//! - Sliding window via masking (full KV retained, windowed positions masked)
//! - Supports flat config.json or nested `text_config` wrapper

use std::path::Path;

use mlx_rs::{
    Array, Dtype, array,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters},
    nn, ops,
    ops::indexing::IndexOp,
    quantization::MaybeQuantized,
};
use serde::Deserialize;

use crate::{
    cache::{KeyValueCache, KvCacheView},
    error::ModelError,
    utils::{apply_rope, create_causal_mask, create_windowed_causal_mask},
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const fn default_rope_theta() -> f32 {
    1_000_000.0
}

const fn default_rope_local_base_freq() -> f32 {
    10_000.0
}

const fn default_sliding_window() -> i32 {
    512
}

const fn default_sliding_window_pattern() -> i32 {
    6
}

const fn default_tie_word_embeddings() -> bool {
    true
}

const fn default_head_dim() -> i32 {
    256
}

const fn default_query_pre_attn_scalar() -> f32 {
    256.0
}

// Multimodal `gemma3` checkpoints carry a minimal `text_config` that omits fields
// equal to HF `Gemma3TextConfig` defaults. These supply those defaults.
const fn default_num_attention_heads() -> i32 {
    8
}

const fn default_num_key_value_heads() -> i32 {
    4
}

const fn default_rms_norm_eps() -> f32 {
    1e-6
}

const fn default_vocab_size() -> i32 {
    262_208
}

const fn default_max_position_embeddings() -> i32 {
    131_072
}

/// Gemma 3 model configuration.
///
/// HF ships two config shapes:
/// - Flat: top-level fields directly.
/// - Wrapped: `{"model_type": "gemma3", "text_config": { ... }}`.
///
/// `load_gemma3_model_args` handles the wrapping; this struct is for
/// the inner (flat) layer only.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma3ModelArgs {
    #[serde(default)]
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: i32,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: i32,
    #[serde(default = "default_head_dim")]
    pub head_dim: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: i32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    /// `RoPE` theta for global (full-attention) layers.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// `RoPE` theta for sliding-window (local) layers.
    #[serde(default = "default_rope_local_base_freq")]
    pub rope_local_base_freq: f32,
    /// Attention scale = `query_pre_attn_scalar ** -0.5`.
    #[serde(default = "default_query_pre_attn_scalar")]
    pub query_pre_attn_scalar: f32,
    /// Number of tokens visible to sliding-window attention layers.
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    /// Layer period: every `sliding_window_pattern`-th layer is global.
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: i32,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<crate::gemma2::QuantizationConfig>,
    /// Ignored: Gemma 3 has no logit soft-capping.
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
}

impl Gemma3ModelArgs {
    pub fn attn_scale(&self) -> f32 {
        self.query_pre_attn_scalar.sqrt().recip()
    }

    /// Layer at `idx` is global (full attention) when
    /// `(idx + 1) % sliding_window_pattern == 0`.
    pub const fn is_global_layer(&self, layer_idx: i32) -> bool {
        self.sliding_window_pattern > 0 && (layer_idx + 1) % self.sliding_window_pattern == 0
    }
}

// ---------------------------------------------------------------------------
// clip_residual helper
// ---------------------------------------------------------------------------

/// Add `x + y`, clamping to `f16::MAX` in absolute value when both are f16.
///
/// Prevents fp16 overflow on the residual adds in deep models.
fn clip_residual(x: &Array, y: &Array) -> Result<Array, Exception> {
    if x.dtype() == Dtype::Float16 {
        // Widen to f32, add, clamp to f16 range, then narrow back.
        const F16_MAX: f32 = 65504.0;
        let sum = x
            .as_dtype(Dtype::Float32)?
            .add(y.as_dtype(Dtype::Float32)?)?;
        let clamped = ops::clip(&sum, (-F16_MAX, F16_MAX))?;
        clamped.as_dtype(Dtype::Float16)
    } else {
        x.add(y)
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma3Attention {
    n_heads: i32,
    n_kv_heads: i32,
    n_rep: i32,
    scale: f32,
    is_sliding: bool,
    sliding_window: i32,

    #[quantizable]
    #[param]
    q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
    #[param]
    rope: nn::Rope,
}

impl Gemma3Attention {
    fn new(args: &Gemma3ModelArgs, layer_idx: i32) -> Result<Self, Exception> {
        let head_dim = args.head_dim;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let is_sliding = !args.is_global_layer(layer_idx);

        let q_proj = nn::LinearBuilder::new(args.hidden_size, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(args.hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(args.hidden_size, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, args.hidden_size)
            .bias(false)
            .build()?;

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;

        // Sliding layers use the local (short-range) theta; global layers use the long-range theta.
        let rope_base = if is_sliding {
            args.rope_local_base_freq
        } else {
            args.rope_theta
        };
        let rope = nn::RopeBuilder::new(head_dim)
            .traditional(false)
            .base(rope_base)
            .scale(1.0)
            .build()
            .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            n_rep: n_heads / n_kv_heads,
            scale: args.attn_scale(),
            is_sliding,
            sliding_window: args.sliding_window,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm,
            rope,
        })
    }
}

struct Gemma3AttentionInput<'a, C> {
    x: &'a Array,
    /// Pre-computed mask for this layer (global or sliding).
    mask: Option<&'a Array>,
    cache: Option<&'a mut C>,
}

impl<C> Module<Gemma3AttentionInput<'_, C>> for Gemma3Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: Gemma3AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Gemma3AttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // QK-norm is applied in [B, L, heads, head_dim] order so the last axis
        // (head_dim) is normalized. We apply it before transposing to [B, heads, L, D].
        let q_normed = self
            .q_norm
            .forward(&self.q_proj.forward(x)?.reshape(&[B, L, self.n_heads, -1])?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k_normed = self
            .k_norm
            .forward(
                &self
                    .k_proj
                    .forward(x)?
                    .reshape(&[B, L, self.n_kv_heads, -1])?,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v_proj = self
            .v_proj
            .forward(x)?
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let (queries, keys, values) = if let Some(ref mut kv_cache) = cache {
            let offset = kv_cache.offset();
            let q = apply_rope(&q_normed, &self.rope, offset)?;
            let k = apply_rope(&k_normed, &self.rope, offset)?;
            // Materialize dense KV (also works for TurboQuant via into_dense).
            // Gemma 3 already uses the mask path for windowed attention, so the
            // TurboQuant decode path isn't needed here.
            let (ck, cv) = match kv_cache.update_and_view(k, v_proj)? {
                view @ (KvCacheView::Dense { .. } | KvCacheView::TurboQuant(_)) => {
                    view.into_dense()?
                }
            };
            (q, ck, cv)
        } else {
            let q = apply_rope(&q_normed, &self.rope, 0)?;
            let k = apply_rope(&k_normed, &self.rope, 0)?;
            (q, k, v_proj)
        };

        // Use MLX fast SDPA: boolean mask (true=attend, None=attend-all).
        let sdpa_mask = mask.map(mlx_rs::fast::ScaledDotProductAttentionMask::Array);
        let output = mlx_rs::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            self.scale,
            sdpa_mask,
            None::<&Array>,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        <nn::RmsNorm as Module<&Array>>::training_mode(&mut self.q_norm, mode);
        <nn::RmsNorm as Module<&Array>>::training_mode(&mut self.k_norm, mode);
        <nn::Rope as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

// ---------------------------------------------------------------------------
// MLP (GeGLU, identical structure to Gemma 2)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma3Mlp {
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

impl Gemma3Mlp {
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

impl Module<&Array> for Gemma3Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
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
// Block (4 norms + clip_residual)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma3Block {
    #[quantizable]
    #[param]
    self_attn: Gemma3Attention,
    #[quantizable]
    #[param]
    mlp: Gemma3Mlp,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
    #[param]
    pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    post_feedforward_layernorm: nn::RmsNorm,
}

impl Gemma3Block {
    fn new(args: &Gemma3ModelArgs, layer_idx: i32) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: Gemma3Attention::new(args, layer_idx)?,
            mlp: Gemma3Mlp::new(args.hidden_size, args.intermediate_size)?,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            pre_feedforward_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_feedforward_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

impl<C> Module<Gemma3AttentionInput<'_, C>> for Gemma3Block
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: Gemma3AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Gemma3AttentionInput { x, mask, cache } = input;

        // Gemma 3 block: input norm -> attn -> post-attn norm -> clip_residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(Gemma3AttentionInput {
            x: &normed,
            mask,
            cache,
        })?;
        let attn_normed = self.post_attention_layernorm.forward(&attn_out)?;
        let h = clip_residual(x, &attn_normed)?;

        // Pre-ff norm -> MLP -> post-ff norm -> clip_residual
        let ff_normed = self.pre_feedforward_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&ff_normed)?;
        let ff_post_normed = self.post_feedforward_layernorm.forward(&mlp_out)?;
        clip_residual(&h, &ff_post_normed)
    }

    fn training_mode(&mut self, mode: bool) {
        <Gemma3Attention as Module<Gemma3AttentionInput<'_, C>>>::training_mode(
            &mut self.self_attn,
            mode,
        );
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.pre_feedforward_layernorm.training_mode(mode);
        self.post_feedforward_layernorm.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma3Model {
    #[quantizable]
    #[param]
    embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    layers: Vec<Gemma3Block>,
    #[param]
    norm: nn::RmsNorm,

    hidden_size: i32,
    sliding_window: i32,
    sliding_window_pattern: i32,
    cached_embed_scale: Option<Array>,
}

struct Gemma3ModelInput<'a, C> {
    inputs: &'a Array,
    cache: &'a mut Vec<Option<C>>,
}

impl Gemma3Model {
    fn new(args: &Gemma3ModelArgs) -> Result<Self, Exception> {
        if !args.vocab_size.is_positive() {
            return Err(Exception::custom("vocab_size must be positive"));
        }
        if !args.num_hidden_layers.is_positive() {
            return Err(Exception::custom("num_hidden_layers must be positive"));
        }

        let layers = (0..args.num_hidden_layers)
            .map(|i| Gemma3Block::new(args, i))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(
                args.vocab_size,
                args.hidden_size,
            )?),
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            hidden_size: args.hidden_size,
            sliding_window: args.sliding_window,
            sliding_window_pattern: args.sliding_window_pattern,
            cached_embed_scale: None,
        })
    }
}

impl<C> Module<Gemma3ModelInput<'_, C>> for Gemma3Model
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: Gemma3ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Gemma3ModelInput { inputs, cache } = input;

        // Scale embeddings by sqrt(hidden_size) — common to all Gemma variants.
        let mut h = self.embed_tokens.forward(inputs)?;
        if self.cached_embed_scale.is_none() {
            let hidden_f32 = f32::from(
                i16::try_from(self.hidden_size)
                    .map_err(|_| Exception::custom("hidden_size out of i16 range"))?,
            );
            self.cached_embed_scale = Some(array!(hidden_f32.sqrt()).as_dtype(h.dtype())?);
        }
        let embed_scale = self
            .cached_embed_scale
            .as_ref()
            .ok_or_else(|| Exception::custom("cached_embed_scale not initialized"))?;
        h = h.multiply(embed_scale)?;

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| None).collect();
        } else if cache.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "kv_cache length ({}) must match num layers ({})",
                cache.len(),
                self.layers.len()
            )));
        }

        // Determine sequence length and cache offset.
        let T = *h
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("hidden state must have >= 2 dims"))?;

        let offset = cache
            .first()
            .and_then(|c| c.as_ref())
            .map_or(0, KeyValueCache::offset);

        // Build the two masks once and reuse across layers.
        // Global layers get a standard causal mask; sliding layers get the windowed variant.
        //
        // For T == 1 (decode step): the standard path returns None for the causal mask
        // (single token attends to all cached keys). But sliding layers must still bound
        // the window — we build a [1, kv_len] boolean row mask when offset+1 > window.
        let global_mask = (T > 1)
            .then(|| create_causal_mask(T, Some(offset)))
            .transpose()?;

        let sliding_mask = if T > 1 {
            Some(create_windowed_causal_mask(T, offset, self.sliding_window)?)
        } else {
            // T == 1: only need a mask if the KV context exceeds the window
            let kv_len = offset + 1;
            (kv_len > self.sliding_window)
                .then(|| create_windowed_causal_mask(1, offset, self.sliding_window))
                .transpose()?
        };

        let pattern = self.sliding_window_pattern;

        for (i, (layer, layer_cache)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            let layer_idx =
                i32::try_from(i).map_err(|_| Exception::custom("too many layers for i32 index"))?;
            let is_global = pattern > 0 && (layer_idx + 1) % pattern == 0;
            let mask = if is_global {
                global_mask.as_ref()
            } else {
                sliding_mask.as_ref()
            };

            h = layer.forward(Gemma3AttentionInput {
                x: &h,
                mask,
                cache: layer_cache.as_mut(),
            })?;
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <Gemma3Block as Module<Gemma3AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Causal LM
// ---------------------------------------------------------------------------

/// Gemma 3 causal language model.
///
/// `tie_word_embeddings` defaults to true in HF Gemma 3 text configs. When
/// true, `embed_tokens` is used as the LM head via `as_linear`; when false a
/// separate `lm_head` weight is loaded and used instead.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Gemma3CausalLM {
    pub args: Gemma3ModelArgs,

    #[quantizable]
    #[param]
    model: Gemma3Model,

    /// Present only when `tie_word_embeddings == false`.
    #[quantizable]
    #[param]
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Gemma3CausalLM {
    pub fn new(args: Gemma3ModelArgs) -> Result<Self, Exception> {
        let model = Gemma3Model::new(&args)?;
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
        })
    }

    fn project_hidden(&mut self, hidden: &Array) -> Result<Array, Exception> {
        match self.lm_head.as_mut() {
            Some(head) => head.forward(hidden),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed) => embed.as_linear(hidden),
                MaybeQuantized::Quantized(q_embed) => q_embed.as_linear(hidden),
            },
        }
    }

    pub fn forward<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        let _ = mask; // Gemma 3 builds its own masks internally
        let hidden_all = self.model.forward(Gemma3ModelInput {
            inputs,
            cache: kv_cache,
        })?;
        // Slice to last token for decode efficiency
        let seq_len = inputs.shape().get(1).copied().unwrap_or(1);
        let hidden_last = hidden_all.index((.., -1.., ..));
        let lm_input = if seq_len > 1 {
            hidden_last.index((.., -1.., ..))
        } else {
            hidden_last
        };
        self.project_hidden(&lm_input)
    }

    /// Forward pass producing logits for every input position.
    pub fn forward_all_logits<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        let _ = mask;
        let hidden = self.model.forward(Gemma3ModelInput {
            inputs,
            cache: kv_cache,
        })?;
        self.project_hidden(&hidden)
    }

    pub fn forward_hidden<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        let _ = mask;
        self.model.forward(Gemma3ModelInput {
            inputs,
            cache: kv_cache,
        })
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Top-level shape of a `text_config`-wrapped Gemma 3 config.json.
#[derive(Debug, Deserialize)]
struct Gemma3TopLevel {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    text_config: Option<serde_json::Value>,
    #[serde(default)]
    quantization: Option<crate::gemma2::QuantizationConfig>,
}

/// Load `Gemma3ModelArgs` from a `config.json`, supporting both flat and
/// `text_config`-wrapped layouts.
pub fn load_gemma3_model_args<P: AsRef<Path>>(model_dir: P) -> Result<Gemma3ModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let text = std::fs::read_to_string(config_path)?;
    let raw: serde_json::Value = serde_json::from_str(&text)?;

    // Detect wrapping: if a `text_config` object exists, deserialize from it
    // and fill in `model_type` from the top level when absent.
    let top: Gemma3TopLevel = serde_json::from_str(&text)?;

    let mut args: Gemma3ModelArgs = if let Some(inner) = top.text_config {
        let mut a: Gemma3ModelArgs = serde_json::from_value(inner)?;
        if a.model_type.is_empty() {
            if let Some(mt) = top.model_type {
                a.model_type = mt;
            }
        }
        // Quantization usually lives at the top level of the wrapper.
        if a.quantization.is_none() {
            a.quantization = top.quantization;
        }
        a
    } else {
        serde_json::from_value(raw)?
    };

    // Ensure model_type is populated (may be missing in some inner configs).
    if args.model_type.is_empty() {
        "gemma3".clone_into(&mut args.model_type);
    }

    Ok(args)
}

/// Load a Gemma 3 model from a directory.
///
/// Applies the `RMSNorm` +1 convention (same as Gemma 2): Gemma 3 stores norm
/// weights pre-shifted by −1, so 1.0 is added to every `*.weight` key whose
/// path contains "norm" after the safetensors weights are loaded.
pub fn load_gemma3_model<P: AsRef<Path>>(model_dir: P) -> Result<Gemma3CausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let mut args = load_gemma3_model_args(model_path)?;

    // HF Gemma 3 text configs omit `tie_word_embeddings`; MLX checkpoints that ship
    // a separate (often separately-quantized) `lm_head` are untied. Honor the
    // checkpoint — reusing the tied embedding as the output projection corrupts
    // low-margin logits (e.g. the stop-token decision) and degrades generation.
    if crate::checkpoint_has_key_suffix(model_path, "lm_head.weight") {
        args.tie_word_embeddings = false;
    }

    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        head_dim = args.head_dim,
        vocab_size = args.vocab_size,
        sliding_window = args.sliding_window,
        sliding_window_pattern = args.sliding_window_pattern,
        "Loading Gemma 3 model"
    );

    let quantization = args.quantization.clone();
    let raw_model = Gemma3CausalLM::new(args)?;

    let mut model = if let Some(ref qc) = quantization {
        tracing::info!(
            group_size = qc.group_size,
            bits = qc.bits,
            "Applying quantization structure"
        );
        mlx_rs::nn::quantize(raw_model, qc.group_size, qc.bits).map_err(|e| {
            ModelError::ShapeMismatch(format!("Failed to quantize model structure: {e}"))
        })?
    } else {
        raw_model
    };

    // Multimodal `gemma3` checkpoints nest the text model under `language_model.`
    // (alongside a vision tower); text-only `gemma3_text` checkpoints start at
    // `model.`. Strip the prefix when present so both load, skipping vision weights.
    crate::load_quantized_safetensors_weights_optional_prefix(
        &mut model,
        model_path,
        quantization.is_some(),
        "language_model.",
    )?;

    // Apply RMSNorm +1 convention — Gemma 3 uses the same shifted-weight storage
    // as Gemma 2. This includes q_norm and k_norm (both keys contain "norm").
    apply_rmsnorm_plus_one(&mut model)
        .map_err(|e| ModelError::ShapeMismatch(format!("Failed to apply RMSNorm +1: {e}")))?;

    tracing::info!("Gemma 3 model loaded successfully");
    Ok(model)
}

/// Add 1.0 to all `RMSNorm` weight parameters (the Gemma +1 convention).
fn apply_rmsnorm_plus_one(model: &mut Gemma3CausalLM) -> Result<(), Exception> {
    use std::rc::Rc;

    let one = array!(1.0_f32);
    let mut params = model.parameters_mut().flatten();

    let norm_keys: Vec<Rc<str>> = params
        .keys()
        .filter(|k| k.ends_with(".weight") && k.contains("norm"))
        .cloned()
        .collect();

    for key in &norm_keys {
        if let Some(param) = params.get_mut(&**key) {
            let shifted = param.add(&one)?;
            **param = shifted;
        }
    }

    let eval_targets: Vec<&Array> = norm_keys
        .iter()
        .filter_map(|k| params.get(&**k).map(|p| &**p))
        .collect();

    mlx_rs::transforms::eval(eval_targets)?;

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
    clippy::doc_markdown
)]
mod tests {
    use super::*;
    use crate::cache::SteppingKeyValueCache;

    fn default_args() -> Gemma3ModelArgs {
        Gemma3ModelArgs {
            model_type: "gemma3_text".to_owned(),
            hidden_size: 128,
            num_hidden_layers: 6,
            intermediate_size: 256,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 32,
            rms_norm_eps: 1e-6,
            vocab_size: 1024,
            max_position_embeddings: 512,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            query_pre_attn_scalar: 256.0,
            sliding_window: 64,
            sliding_window_pattern: 6,
            tie_word_embeddings: true,
            quantization: None,
            rope_scaling: None,
        }
    }

    // -----------------------------------------------------------------------
    // Config deserialization
    // -----------------------------------------------------------------------

    #[test]
    fn config_flat_deserialization() {
        let json = r#"{
            "model_type": "gemma3_text",
            "hidden_size": 1152,
            "num_hidden_layers": 26,
            "intermediate_size": 6912,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
            "vocab_size": 262144,
            "max_position_embeddings": 131072,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "query_pre_attn_scalar": 256.0,
            "sliding_window": 512,
            "sliding_window_pattern": 6,
            "tie_word_embeddings": true
        }"#;
        let args: Gemma3ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "gemma3_text");
        assert_eq!(args.hidden_size, 1152);
        assert_eq!(args.num_hidden_layers, 26);
        assert_eq!(args.head_dim, 256);
        assert_eq!(args.sliding_window, 512);
        assert_eq!(args.sliding_window_pattern, 6);
        assert!(args.tie_word_embeddings);
    }

    #[test]
    fn config_text_config_wrapper() {
        // Simulate the multimodal wrapper shape: top-level model_type + text_config blob
        let json = r#"{
            "model_type": "gemma3",
            "text_config": {
                "hidden_size": 1152,
                "num_hidden_layers": 26,
                "intermediate_size": 6912,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "rms_norm_eps": 1e-6,
                "vocab_size": 262144,
                "max_position_embeddings": 131072,
                "rope_theta": 1000000.0,
                "rope_local_base_freq": 10000.0,
                "query_pre_attn_scalar": 256.0,
                "sliding_window": 512,
                "sliding_window_pattern": 6,
                "tie_word_embeddings": true
            }
        }"#;
        let top: Gemma3TopLevel = serde_json::from_str(json).unwrap();
        assert!(top.text_config.is_some());
        let inner: Gemma3ModelArgs = serde_json::from_value(top.text_config.unwrap()).unwrap();
        assert_eq!(inner.hidden_size, 1152);
        assert_eq!(inner.sliding_window_pattern, 6);
    }

    #[test]
    fn config_defaults_for_missing_optional_fields() {
        let json = r#"{
            "model_type": "gemma3_text",
            "hidden_size": 1152,
            "num_hidden_layers": 26,
            "intermediate_size": 6912,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "vocab_size": 262144,
            "max_position_embeddings": 131072
        }"#;
        let args: Gemma3ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.head_dim, 256); // default
        assert!((args.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!((args.rope_local_base_freq - 10_000.0).abs() < 1.0);
        assert_eq!(args.sliding_window, 512);
        assert_eq!(args.sliding_window_pattern, 6);
        assert!(args.tie_word_embeddings);
    }

    /// Multimodal `gemma3` checkpoints ship a minimal `text_config` that omits
    /// fields equal to HF `Gemma3TextConfig` defaults (e.g. Gemma 3 4B). Those
    /// must fill in, or the config fails to parse.
    #[test]
    fn config_minimal_text_config_uses_hf_defaults() {
        let json = r#"{
            "hidden_size": 2560,
            "num_hidden_layers": 34,
            "intermediate_size": 10240,
            "sliding_window": 1024
        }"#;
        let args: Gemma3ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.num_attention_heads, 8);
        assert_eq!(args.num_key_value_heads, 4);
        assert_eq!(args.head_dim, 256);
        assert_eq!(args.vocab_size, 262_208);
        assert_eq!(args.max_position_embeddings, 131_072);
        assert!((args.rms_norm_eps - 1e-6).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // Attention scale
    // -----------------------------------------------------------------------

    #[test]
    fn attn_scale_uses_query_pre_attn_scalar() {
        let args = default_args();
        let expected = (256.0_f32).sqrt().recip();
        assert!((args.attn_scale() - expected).abs() < 1e-6);
    }

    #[test]
    fn attn_scale_reflects_custom_scalar() {
        let mut args = default_args();
        args.query_pre_attn_scalar = 64.0;
        let expected = (64.0_f32).sqrt().recip();
        assert!((args.attn_scale() - expected).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Sliding-window / global layer pattern
    // -----------------------------------------------------------------------

    #[test]
    fn layer_pattern_with_pattern_6() {
        let args = default_args(); // pattern=6
        // Layers 0..4 => sliding; layer 5 => global
        for i in 0..5 {
            assert!(!args.is_global_layer(i), "layer {i} should be sliding");
        }
        assert!(args.is_global_layer(5), "layer 5 should be global");
        // Pattern repeats
        for i in 6..11 {
            assert!(!args.is_global_layer(i), "layer {i} should be sliding");
        }
        assert!(args.is_global_layer(11), "layer 11 should be global");
    }

    #[test]
    fn layer_pattern_with_pattern_2() {
        let mut args = default_args();
        args.sliding_window_pattern = 2;
        // Layers 1,3,5,... are global
        assert!(!args.is_global_layer(0));
        assert!(args.is_global_layer(1));
        assert!(!args.is_global_layer(2));
        assert!(args.is_global_layer(3));
    }

    #[test]
    fn layer_pattern_zero_means_no_global() {
        let mut args = default_args();
        args.sliding_window_pattern = 0;
        for i in 0..10 {
            assert!(!args.is_global_layer(i));
        }
    }

    // -----------------------------------------------------------------------
    // Model construction
    // -----------------------------------------------------------------------

    #[test]
    fn model_construction_tied_embeddings() {
        let args = default_args();
        let model = Gemma3CausalLM::new(args).unwrap();
        assert!(model.lm_head.is_none());
    }

    #[test]
    fn model_construction_untied_embeddings() {
        let mut args = default_args();
        args.tie_word_embeddings = false;
        let model = Gemma3CausalLM::new(args).unwrap();
        assert!(model.lm_head.is_some());
    }

    #[test]
    fn model_rejects_zero_vocab_size() {
        let mut args = default_args();
        args.vocab_size = 0;
        assert!(Gemma3CausalLM::new(args).is_err());
    }

    #[test]
    fn model_rejects_zero_layers() {
        let mut args = default_args();
        args.num_hidden_layers = 0;
        assert!(Gemma3CausalLM::new(args).is_err());
    }

    // -----------------------------------------------------------------------
    // Windowed mask values (hand-computed small cases)
    // -----------------------------------------------------------------------

    #[test]
    fn windowed_mask_shape() {
        let mask = create_windowed_causal_mask(4, 0, 2).unwrap();
        assert_eq!(mask.shape(), &[4, 4]);
    }

    #[test]
    fn windowed_mask_decode_single_token_exceeds_window() {
        // N=1, offset=10, window=4: keys 7,8,9,10 are visible; 0..6 are not
        let mask = create_windowed_causal_mask(1, 10, 4).unwrap();
        assert_eq!(mask.shape(), &[1, 11]);
        mlx_rs::transforms::eval([&mask]).unwrap();
        let flat: Vec<bool> = mask.as_slice().to_vec();
        let expected = [
            false, false, false, false, false, false, false, true, true, true, true,
        ];
        assert_eq!(flat, expected);
    }

    #[test]
    fn windowed_mask_is_subset_of_causal() {
        // Every position that is true in the windowed mask must also be true in
        // the plain causal mask.
        let n = 5i32;
        let offset = 3i32;
        let window = 3i32;
        let windowed = create_windowed_causal_mask(n, offset, window).unwrap();
        let causal = create_causal_mask(n, Some(offset)).unwrap();
        mlx_rs::transforms::eval([&windowed, &causal]).unwrap();
        let w: Vec<bool> = windowed.as_slice().to_vec();
        let c: Vec<bool> = causal.as_slice().to_vec();
        for (wv, cv) in w.iter().zip(c.iter()) {
            if *wv {
                assert!(*cv, "windowed=true but causal=false — mask is wrong");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Smoke test: random-weight forward pass produces finite output
    // -----------------------------------------------------------------------

    #[test]
    fn smoke_forward_produces_finite_output() {
        // Tiny model to keep the test fast
        let mut args = default_args();
        args.num_hidden_layers = 2; // one sliding + one ... pattern=6 so both sliding
        args.sliding_window_pattern = 2; // layer 1 is global
        args.vocab_size = 64;
        args.hidden_size = 32;
        args.intermediate_size = 64;
        args.num_attention_heads = 2;
        args.num_key_value_heads = 1;
        args.head_dim = 16;
        args.sliding_window = 8;

        let mut model = Gemma3CausalLM::new(args).unwrap();

        let input = Array::from_slice(&[0i32, 1, 2, 3], &[1, 4]);
        let mut kv_cache: Vec<Option<SteppingKeyValueCache>> = vec![];
        let logits = model.forward(&input, None, &mut kv_cache).unwrap();

        // Shape: [1, 1, vocab_size] (last token only)
        let shape = logits.shape();
        assert_eq!(shape[0], 1);
        assert_eq!(shape[2], 64);

        mlx_rs::transforms::eval([&logits]).unwrap();
        let vals: Vec<f32> = logits.as_slice().to_vec();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "forward pass produced non-finite logits"
        );
    }

    #[test]
    fn smoke_forward_decode_step() {
        let mut args = default_args();
        args.num_hidden_layers = 2;
        args.sliding_window_pattern = 2;
        args.vocab_size = 64;
        args.hidden_size = 32;
        args.intermediate_size = 64;
        args.num_attention_heads = 2;
        args.num_key_value_heads = 1;
        args.head_dim = 16;
        args.sliding_window = 8;

        let mut model = Gemma3CausalLM::new(args).unwrap();

        // Prefill
        let prefill = Array::from_slice(&[0i32, 1, 2], &[1, 3]);
        let mut kv_cache: Vec<Option<SteppingKeyValueCache>> = vec![];
        let _ = model.forward(&prefill, None, &mut kv_cache).unwrap();

        // Decode step (T=1)
        let token = Array::from_slice(&[3i32], &[1, 1]);
        let logits = model.forward(&token, None, &mut kv_cache).unwrap();
        mlx_rs::transforms::eval([&logits]).unwrap();
        let vals: Vec<f32> = logits.as_slice().to_vec();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "decode step produced non-finite logits"
        );
    }
}
