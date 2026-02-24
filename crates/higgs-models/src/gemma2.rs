//! Gemma 2 model implementation.
//!
//! Differs from the standard transformer in several ways:
//! - Explicit `head_dim` (not derived from `hidden_size / num_attention_heads`)
//! - 4 layer norms per block (pre/post attention + pre/post feedforward)
//! - Attention logit soft-capping via tanh
//! - Final logit soft-capping
//! - `GeGLU` activation (GELU-gated instead of SiLU-gated)
//! - `RMSNorm` with +1 convention (weights stored as w-1)
//! - Alternating sliding window / global attention layers

use std::path::Path;

use mlx_rs::{
    Array, array,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters},
    nn, ops,
    quantization::MaybeQuantized,
};
use serde::Deserialize;

use crate::{
    cache::KeyValueCache,
    error::ModelError,
    utils::{AttentionMask, apply_rope, create_attention_mask},
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const fn default_rope_theta() -> f32 {
    10000.0
}

const fn default_sliding_window_pattern() -> i32 {
    2
}

// Gemma 2 uses tied word embeddings by default (no separate lm_head weight).
const fn default_tie_word_embeddings() -> bool {
    true
}

/// Quantization parameters from config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

/// Gemma 2 model configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma2ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,

    /// Scale factor for attention logits: `1 / sqrt(query_pre_attn_scalar)`.
    /// Defaults to `head_dim` if not present.
    #[serde(default)]
    pub query_pre_attn_scalar: Option<i32>,

    /// Tanh soft-capping for attention logits before softmax.
    #[serde(default)]
    pub attn_logit_softcapping: Option<f32>,

    /// Tanh soft-capping for final output logits.
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,

    /// Sliding window size for local attention layers.
    #[serde(default)]
    pub sliding_window: Option<i32>,

    /// How many layers between sliding window layers (default 2 = alternating).
    #[serde(default = "default_sliding_window_pattern")]
    pub sliding_window_pattern: i32,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl Gemma2ModelArgs {
    fn attn_scale(&self) -> f32 {
        let scalar = self.query_pre_attn_scalar.unwrap_or(self.head_dim);
        let scalar_f32 = f32::from(i16::try_from(scalar).unwrap_or(i16::MAX));
        scalar_f32.sqrt().recip()
    }

    /// Whether layer at `idx` uses sliding window attention.
    const fn is_sliding_window_layer(&self, layer_idx: i32) -> bool {
        if self.sliding_window.is_none() || self.sliding_window_pattern <= 0 {
            return false;
        }
        layer_idx % self.sliding_window_pattern == 0
    }
}

// ---------------------------------------------------------------------------
// Repeat KV heads for manual GQA attention
// ---------------------------------------------------------------------------

fn repeat_kv(x: &Array, n_rep: i32) -> Result<Array, Exception> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let shape = x.shape();
    let b = *shape
        .first()
        .ok_or_else(|| Exception::custom("repeat_kv: empty shape"))?;
    let n_kv = *shape
        .get(1)
        .ok_or_else(|| Exception::custom("repeat_kv: need 4D input"))?;
    let s = *shape
        .get(2)
        .ok_or_else(|| Exception::custom("repeat_kv: need 4D input"))?;
    let d = *shape
        .get(3)
        .ok_or_else(|| Exception::custom("repeat_kv: need 4D input"))?;

    let expanded = x.reshape(&[b, n_kv, 1, s, d])?;
    let repeated = ops::broadcast_to(&expanded, &[b, n_kv, n_rep, s, d])?;
    repeated.reshape(&[b, n_kv * n_rep, s, d])
}

// ---------------------------------------------------------------------------
// Attention (manual with soft-capping)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma2Attention {
    n_heads: i32,
    n_kv_heads: i32,
    n_rep: i32,
    scale: f32,
    attn_logit_softcapping: Option<f32>,
    sliding_window: Option<i32>,

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
    rope: nn::Rope,
}

impl Gemma2Attention {
    fn new(args: &Gemma2ModelArgs, sliding_window: bool) -> Result<Self, Exception> {
        let head_dim = args.head_dim;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let scale = args.attn_scale();

        let bias = args.attention_bias;
        let q_proj = nn::LinearBuilder::new(args.hidden_size, n_heads * head_dim)
            .bias(bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(args.hidden_size, n_kv_heads * head_dim)
            .bias(bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(args.hidden_size, n_kv_heads * head_dim)
            .bias(bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, args.hidden_size)
            .bias(bias)
            .build()?;

        let rope = nn::RopeBuilder::new(head_dim)
            .traditional(false)
            .base(args.rope_theta)
            .scale(1.0)
            .build()
            .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?;

        let window = if sliding_window {
            args.sliding_window
        } else {
            None
        };

        Ok(Self {
            n_heads,
            n_kv_heads,
            n_rep: n_heads / n_kv_heads,
            scale,
            attn_logit_softcapping: args.attn_logit_softcapping,
            sliding_window: window,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            rope,
        })
    }
}

struct Gemma2AttentionInput<'a, C> {
    x: &'a Array,
    mask: Option<&'a Array>,
    cache: Option<&'a mut C>,
}

impl<C> Module<Gemma2AttentionInput<'_, C>> for Gemma2Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: Gemma2AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Gemma2AttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        let mut queries = self
            .q_proj
            .forward(x)?
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = self
            .k_proj
            .forward(x)?
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = self
            .v_proj
            .forward(x)?
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        if let Some(ref mut kv_cache) = cache {
            queries = apply_rope(&queries, &self.rope, kv_cache.offset())?;
            keys = apply_rope(&keys, &self.rope, kv_cache.offset())?;

            let (cached_keys, cached_values) = kv_cache.update_and_fetch(keys, values)?;
            keys = cached_keys;
            values = cached_values;
        } else {
            queries = apply_rope(&queries, &self.rope, 0)?;
            keys = apply_rope(&keys, &self.rope, 0)?;
        }

        // Expand KV heads for GQA
        keys = repeat_kv(&keys, self.n_rep)?;
        values = repeat_kv(&values, self.n_rep)?;

        // Manual attention with soft-capping
        // scores: [B, n_heads, L, S]
        let mut scores = queries
            .matmul(&keys.transpose_axes(&[0, 1, 3, 2])?)?
            .multiply(array!(self.scale))?;

        // Soft-capping: tanh(scores / cap) * cap
        if let Some(cap) = self.attn_logit_softcapping {
            let inv_cap = 1.0 / cap;
            scores = ops::tanh(&scores.multiply(array!(inv_cap))?)?.multiply(array!(cap))?;
        }

        // Apply sliding window mask (additive: -inf for out-of-window positions)
        if let Some(window) = self.sliding_window {
            let s_len = *scores
                .shape()
                .last()
                .ok_or_else(|| Exception::custom("scores must have >= 1 dim"))?;
            if s_len > window {
                let window_mask = create_sliding_window_mask(L, s_len, window)?;
                scores = ops::r#where(&window_mask, &scores, &array!(f32::NEG_INFINITY))?;
            }
        }

        // Apply causal mask (boolean: true = attend, false = mask out)
        if let Some(m) = mask {
            scores = ops::r#where(m, &scores, &array!(f32::NEG_INFINITY))?;
        }

        let weights = ops::softmax_axis(&scores, -1, None)?;
        let output = weights
            .matmul(&values)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        <nn::Rope as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

/// Create a boolean mask for sliding window attention.
///
/// For each query position q (with absolute position = offset + `q_local`),
/// only keys within `[q_abs - window + 1, q_abs]` are visible.
#[allow(non_snake_case)]
fn create_sliding_window_mask(L: i32, S: i32, window: i32) -> Result<Array, Exception> {
    // Query positions: last L of the S total positions
    // offset = S - L
    let offset = S - L;
    // For query at local index i (absolute position = offset + i),
    // key at index j is visible if j >= (offset + i) - window + 1 AND j <= (offset + i)
    // The causal mask already handles j <= (offset + i).
    // We just need: j >= (offset + i) - window + 1

    let query_positions = mlx_rs::arange!(start = offset, stop = offset + L)?;
    let key_positions = mlx_rs::arange!(stop = S)?;

    // lower_bound[i] = query_pos[i] - window + 1
    let lower_bounds = query_positions.subtract(array!(window - 1))?;
    // Reshape for broadcasting: [L, 1] vs [1, S]
    let lower_expanded = lower_bounds.reshape(&[L, 1])?;
    let key_expanded = key_positions.reshape(&[1, S])?;

    // mask[i, j] = key_positions[j] >= lower_bound[i]
    key_expanded.ge(&lower_expanded)
}

// ---------------------------------------------------------------------------
// MLP (GeGLU)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma2Mlp {
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

impl Gemma2Mlp {
    fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_proj: MaybeQuantized::Original(gate_proj),
            down_proj: MaybeQuantized::Original(down_proj),
            up_proj: MaybeQuantized::Original(up_proj),
        })
    }
}

impl Module<&Array> for Gemma2Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        // GeGLU: gelu(gate) * up, then down
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
// Block (4 norms)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Gemma2Block {
    #[quantizable]
    #[param]
    self_attn: Gemma2Attention,
    #[quantizable]
    #[param]
    mlp: Gemma2Mlp,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
    #[param]
    pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    post_feedforward_layernorm: nn::RmsNorm,
}

impl Gemma2Block {
    fn new(args: &Gemma2ModelArgs, layer_idx: i32) -> Result<Self, Exception> {
        let sliding = args.is_sliding_window_layer(layer_idx);
        Ok(Self {
            self_attn: Gemma2Attention::new(args, sliding)?,
            mlp: Gemma2Mlp::new(args.hidden_size, args.intermediate_size)?,
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

impl<C> Module<Gemma2AttentionInput<'_, C>> for Gemma2Block
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: Gemma2AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Gemma2AttentionInput { x, mask, cache } = input;

        // Pre-attention norm -> attention -> post-attention norm -> residual
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(Gemma2AttentionInput {
            x: &normed,
            mask,
            cache,
        })?;
        let attn_normed = self.post_attention_layernorm.forward(&attn_out)?;
        let h = x.add(attn_normed)?;

        // Pre-feedforward norm -> MLP -> post-feedforward norm -> residual
        let ff_normed = self.pre_feedforward_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&ff_normed)?;
        let ff_post_normed = self.post_feedforward_layernorm.forward(&mlp_out)?;
        h.add(ff_post_normed)
    }

    fn training_mode(&mut self, mode: bool) {
        <Gemma2Attention as Module<Gemma2AttentionInput<'_, C>>>::training_mode(
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
struct Gemma2Model {
    #[quantizable]
    #[param]
    embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    layers: Vec<Gemma2Block>,
    #[param]
    norm: nn::RmsNorm,

    hidden_size: i32,
}

struct Gemma2ModelInput<'a, C> {
    inputs: &'a Array,
    mask: Option<&'a Array>,
    cache: &'a mut Vec<Option<C>>,
}

impl Gemma2Model {
    fn new(args: &Gemma2ModelArgs) -> Result<Self, Exception> {
        if !args.vocab_size.is_positive() {
            return Err(Exception::custom("vocab_size must be positive"));
        }
        if !args.num_hidden_layers.is_positive() {
            return Err(Exception::custom("num_hidden_layers must be positive"));
        }

        let layers = (0..args.num_hidden_layers)
            .map(|i| Gemma2Block::new(args, i))
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
        })
    }
}

impl<C> Module<Gemma2ModelInput<'_, C>> for Gemma2Model
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: Gemma2ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Gemma2ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        // Gemma 2 scales embeddings by sqrt(hidden_size)
        let hidden_size_f32 = f32::from(
            i16::try_from(self.hidden_size)
                .map_err(|_| Exception::custom("hidden_size out of i16 range"))?,
        );
        let mut h = self
            .embed_tokens
            .forward(inputs)?
            .multiply(array!(hidden_size_f32.sqrt()))?;

        let computed_mask = match mask {
            Some(m) => Some(m.clone()),
            None => match create_attention_mask(&h, cache, Some(true))? {
                Some(AttentionMask::Array(a)) => Some(a),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Only Array mask is supported"));
                }
                None => None,
            },
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| None).collect();
        } else if cache.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "kv_cache length ({}) must match num layers ({})",
                cache.len(),
                self.layers.len()
            )));
        }

        for (layer, layer_cache) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(Gemma2AttentionInput {
                x: &h,
                mask: computed_mask.as_ref(),
                cache: layer_cache.as_mut(),
            })?;
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <Gemma2Block as Module<Gemma2AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Causal LM
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Gemma2CausalLM {
    pub args: Gemma2ModelArgs,

    #[quantizable]
    #[param]
    model: Gemma2Model,

    #[quantizable]
    #[param]
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Gemma2CausalLM {
    pub fn new(args: Gemma2ModelArgs) -> Result<Self, Exception> {
        let model = Gemma2Model::new(&args)?;
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

    pub fn forward<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        let hidden = self.forward_hidden(inputs, mask, kv_cache)?;

        let mut logits = match self.lm_head.as_mut() {
            Some(head) => head.forward(&hidden)?,
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed) => embed.as_linear(&hidden)?,
                MaybeQuantized::Quantized(q_embed) => q_embed.as_linear(&hidden)?,
            },
        };

        // Final logit soft-capping
        if let Some(cap) = self.args.final_logit_softcapping {
            let inv_cap = 1.0 / cap;
            logits = ops::tanh(&logits.multiply(array!(inv_cap))?)?.multiply(array!(cap))?;
        }

        Ok(logits)
    }

    pub fn forward_hidden<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        self.model.forward(Gemma2ModelInput {
            inputs,
            mask,
            cache: kv_cache,
        })
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

pub fn load_gemma2_model_args<P: AsRef<Path>>(model_dir: P) -> Result<Gemma2ModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

/// Load a Gemma 2 model from a directory.
///
/// After loading weights, applies the `RMSNorm` +1 convention by adding 1.0
/// to all `RmsNorm` weight parameters. Gemma 2 stores norm weights as `w - 1`,
/// so `(w + 1) * rms_norm(x)` is equivalent to standard `RmsNorm` with the
/// shifted weight.
pub fn load_gemma2_model<P: AsRef<Path>>(model_dir: P) -> Result<Gemma2CausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_gemma2_model_args(model_path)?;

    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        head_dim = args.head_dim,
        vocab_size = args.vocab_size,
        attn_softcap = ?args.attn_logit_softcapping,
        final_softcap = ?args.final_logit_softcapping,
        "Loading Gemma 2 model"
    );

    let quantization = args.quantization.clone();
    let raw_model = Gemma2CausalLM::new(args)?;

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

    crate::load_quantized_safetensors_weights(&mut model, model_path, quantization.is_some())?;

    // Apply RMSNorm +1 convention: add 1.0 to all norm weights
    apply_rmsnorm_plus_one(&mut model)
        .map_err(|e| ModelError::ShapeMismatch(format!("Failed to apply RMSNorm +1: {e}")))?;

    tracing::info!("Gemma 2 model loaded successfully");
    Ok(model)
}

/// Add 1.0 to all `RmsNorm` weight parameters.
///
/// Gemma 2 stores norm weights pre-shifted by -1. Standard `RmsNorm` computes
/// `weight * rms_norm(x)`, so adding 1.0 to the stored weights gives the
/// correct Gemma 2 behavior: `(stored_weight + 1) * rms_norm(x)`.
fn apply_rmsnorm_plus_one(model: &mut Gemma2CausalLM) -> Result<(), Exception> {
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

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn default_gemma2_args() -> Gemma2ModelArgs {
        Gemma2ModelArgs {
            model_type: "gemma2".to_owned(),
            hidden_size: 256,
            num_hidden_layers: 2,
            intermediate_size: 512,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 64,
            rms_norm_eps: 1e-6,
            vocab_size: 1000,
            max_position_embeddings: 512,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            attention_bias: false,
            query_pre_attn_scalar: None,
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            sliding_window: Some(128),
            sliding_window_pattern: 2,
            quantization: None,
        }
    }

    #[test]
    fn config_deserialization() {
        let json = r#"{
            "model_type": "gemma2",
            "hidden_size": 2304,
            "num_hidden_layers": 26,
            "intermediate_size": 9216,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
            "vocab_size": 256000,
            "max_position_embeddings": 8192,
            "rope_theta": 10000.0,
            "tie_word_embeddings": true,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "query_pre_attn_scalar": 256,
            "sliding_window": 4096
        }"#;

        let args: Gemma2ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "gemma2");
        assert_eq!(args.hidden_size, 2304);
        assert_eq!(args.head_dim, 256);
        assert_eq!(args.num_attention_heads, 8);
        assert_eq!(args.num_key_value_heads, 4);
        assert_eq!(args.attn_logit_softcapping, Some(50.0));
        assert_eq!(args.final_logit_softcapping, Some(30.0));
        assert_eq!(args.sliding_window, Some(4096));
    }

    #[test]
    fn attn_scale_uses_query_pre_attn_scalar() {
        let mut args = default_gemma2_args();
        args.query_pre_attn_scalar = Some(256);
        let expected = (256.0_f32).sqrt().recip();
        assert!((args.attn_scale() - expected).abs() < 1e-6);
    }

    #[test]
    fn attn_scale_defaults_to_head_dim() {
        let args = default_gemma2_args();
        let expected = (64.0_f32).sqrt().recip();
        assert!((args.attn_scale() - expected).abs() < 1e-6);
    }

    #[test]
    fn sliding_window_layer_pattern() {
        let args = default_gemma2_args();
        // pattern=2: layers 0, 2, 4... are sliding window
        assert!(args.is_sliding_window_layer(0));
        assert!(!args.is_sliding_window_layer(1));
        assert!(args.is_sliding_window_layer(2));
        assert!(!args.is_sliding_window_layer(3));
    }

    #[test]
    fn sliding_window_disabled_without_window() {
        let mut args = default_gemma2_args();
        args.sliding_window = None;
        assert!(!args.is_sliding_window_layer(0));
        assert!(!args.is_sliding_window_layer(1));
    }

    #[test]
    fn model_construction() {
        let args = default_gemma2_args();
        let model = Gemma2CausalLM::new(args).unwrap();
        assert!(model.lm_head.is_none()); // tied embeddings
    }

    #[test]
    fn model_construction_untied_embeddings() {
        let mut args = default_gemma2_args();
        args.tie_word_embeddings = false;
        let model = Gemma2CausalLM::new(args).unwrap();
        assert!(model.lm_head.is_some());
    }

    #[test]
    fn model_rejects_zero_vocab_size() {
        let mut args = default_gemma2_args();
        args.vocab_size = 0;
        assert!(Gemma2CausalLM::new(args).is_err());
    }

    #[test]
    fn model_rejects_zero_layers() {
        let mut args = default_gemma2_args();
        args.num_hidden_layers = 0;
        assert!(Gemma2CausalLM::new(args).is_err());
    }

    #[test]
    fn repeat_kv_no_op_for_n_rep_1() {
        let x = Array::ones::<f32>(&[1, 4, 3, 8]).unwrap();
        let result = repeat_kv(&x, 1).unwrap();
        assert_eq!(result.shape(), &[1, 4, 3, 8]);
    }

    #[test]
    fn repeat_kv_doubles_heads() {
        let x = Array::ones::<f32>(&[1, 2, 3, 8]).unwrap();
        let result = repeat_kv(&x, 2).unwrap();
        assert_eq!(result.shape(), &[1, 4, 3, 8]);
    }

    #[test]
    fn sliding_window_mask_shape() {
        let mask = create_sliding_window_mask(4, 10, 3).unwrap();
        assert_eq!(mask.shape(), &[4, 10]);
    }

    #[test]
    fn sliding_window_mask_values() {
        // L=3, S=5, window=2
        // Query positions (absolute): 2, 3, 4 (offset = S-L = 2)
        // Key positions: 0, 1, 2, 3, 4
        // The mask only enforces the lower bound (j >= q - window + 1).
        // The causal mask separately enforces the upper bound (j <= q).
        // q=2: j >= 1 -> [F, T, T, T, T]
        // q=3: j >= 2 -> [F, F, T, T, T]
        // q=4: j >= 3 -> [F, F, F, T, T]
        let mask = create_sliding_window_mask(3, 5, 2).unwrap();
        mlx_rs::transforms::eval([&mask]).unwrap();
        let flat: Vec<bool> = mask.as_slice().to_vec();
        let expected = [
            false, true, true, true, true, false, false, true, true, true, false, false, false,
            true, true,
        ];
        assert_eq!(flat, expected);
    }

    #[test]
    fn config_defaults_without_optional_fields() {
        let json = r#"{
            "model_type": "gemma2",
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "rms_norm_eps": 1e-6,
            "vocab_size": 1000,
            "max_position_embeddings": 512
        }"#;

        let args: Gemma2ModelArgs = serde_json::from_str(json).unwrap();
        assert!(args.attn_logit_softcapping.is_none());
        assert!(args.final_logit_softcapping.is_none());
        assert!(args.sliding_window.is_none());
        assert!(args.quantization.is_none());
        assert_eq!(args.sliding_window_pattern, 2);
        assert!((args.rope_theta - 10000.0).abs() < f32::EPSILON);
        assert!(args.tie_word_embeddings); // Gemma 2 default: tied embeddings
    }
}
