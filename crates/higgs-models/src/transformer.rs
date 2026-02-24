//! Unified transformer model implementation.
//!
//! Supports Qwen2, Llama, and Mistral architectures. Architecture-specific
//! behavior (e.g., Q/K/V bias) is parameterized through `ModelArgs`.

use std::path::Path;

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn, ops,
    ops::indexing::IndexOp,
    quantization::MaybeQuantized,
};
use serde::Deserialize;

use crate::{
    cache::{KeyValueCache, SteppingKeyValueCache},
    error::ModelError,
    utils::{
        AttentionMask, apply_rope, create_attention_mask, create_batched_decode_mask,
        scaled_dot_product_attention,
    },
};

const fn default_rope_theta() -> f32 {
    10000.0
}

/// Deserialize an `Option<i32>` that may appear as the string `"None"` in
/// some `HuggingFace` configs (e.g., `nanoLLaVA`'s `sliding_window`).
fn deserialize_optional_i32<'de, D>(deserializer: D) -> Result<Option<i32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value: serde_json::Value = Deserialize::deserialize(deserializer)?;
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(n) => n
            .as_i64()
            .and_then(|v| i32::try_from(v).ok())
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("invalid number for i32")),
        serde_json::Value::String(ref s) if s == "None" || s == "null" => Ok(None),
        serde_json::Value::String(_)
        | serde_json::Value::Bool(_)
        | serde_json::Value::Array(_)
        | serde_json::Value::Object(_) => Err(serde::de::Error::custom(format!(
            "expected i32 or null, got {value}"
        ))),
    }
}

/// Quantization parameters from config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

/// Unified model configuration, deserialized from config.json.
///
/// Architecture-specific fields use serde defaults so that configs from
/// Qwen2, Llama, and Mistral all deserialize into the same struct.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // Architecture-specific optional fields
    #[serde(default)]
    pub attention_bias: Option<bool>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default, deserialize_with = "deserialize_optional_i32")]
    pub sliding_window: Option<i32>,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    // Quantization (present in pre-quantized MLX models)
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl ModelArgs {
    /// Whether Q/K/V projections should have bias.
    ///
    /// Uses the config's `attention_bias` field when present, otherwise falls
    /// back to architecture defaults (only qwen2 uses bias by default).
    pub fn qkv_bias(&self) -> bool {
        self.attention_bias
            .unwrap_or(matches!(self.model_type.as_str(), "qwen2"))
    }

    /// Head dimension, computed from `hidden_size / num_attention_heads`.
    ///
    /// Panics in debug builds if not evenly divisible.
    pub fn head_dim(&self) -> i32 {
        debug_assert!(
            self.num_attention_heads != 0 && self.hidden_size % self.num_attention_heads == 0,
            "hidden_size ({}) must be divisible by num_attention_heads ({})",
            self.hidden_size,
            self.num_attention_heads
        );
        self.hidden_size / self.num_attention_heads
    }

    /// Validated head dimension that returns an error if not evenly divisible.
    pub fn checked_head_dim(&self) -> Result<i32, ModelError> {
        if self.num_attention_heads == 0 {
            return Err(ModelError::ShapeMismatch(
                "num_attention_heads must be positive".to_owned(),
            ));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(ModelError::ShapeMismatch(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        Ok(self.hidden_size / self.num_attention_heads)
    }
}

/// Multi-head attention module.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub scale: f32,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub q_norm: Option<nn::RmsNorm>,
    #[param]
    pub k_norm: Option<nn::RmsNorm>,
    #[param]
    pub rope: nn::Rope,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args
            .checked_head_dim()
            .map_err(|e| Exception::custom(e.to_string()))?;
        let head_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        let scale = head_dim_f32.sqrt().recip();

        let qkv_bias = args.qkv_bias();
        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(qkv_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(qkv_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(qkv_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
            .build()?;

        let qk_norm = matches!(args.model_type.as_str(), "qwen3");
        let q_norm = qk_norm
            .then(|| {
                nn::RmsNormBuilder::new(head_dim)
                    .eps(args.rms_norm_eps)
                    .build()
            })
            .transpose()?;
        let k_norm = qk_norm
            .then(|| {
                nn::RmsNormBuilder::new(head_dim)
                    .eps(args.rms_norm_eps)
                    .build()
            })
            .transpose()?;

        let rope = nn::RopeBuilder::new(head_dim)
            .traditional(false)
            .base(args.rope_theta)
            .scale(1.0)
            .build()
            .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            scale,
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

/// Input to the attention module.
pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have at least 2 dimensions"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have at least 2 dimensions"))?;

        let q_raw = self.q_proj.forward(x)?;
        let k_raw = self.k_proj.forward(x)?;
        let v_raw = self.v_proj.forward(x)?;

        let mut queries = q_raw.reshape(&[B, L, self.n_heads, -1])?;
        let mut keys = k_raw.reshape(&[B, L, self.n_kv_heads, -1])?;

        if let Some(ref mut qn) = self.q_norm {
            queries = qn.forward(&queries)?;
        }
        if let Some(ref mut kn) = self.k_norm {
            keys = kn.forward(&keys)?;
        }

        queries = queries.transpose_axes(&[0, 2, 1, 3])?;
        keys = keys.transpose_axes(&[0, 2, 1, 3])?;
        let mut values = v_raw
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

        let output = scaled_dot_product_attention(queries, keys, values, self.scale, mask)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        if let Some(ref mut qn) = self.q_norm {
            qn.training_mode(mode);
        }
        if let Some(ref mut kn) = self.k_norm {
            kn.training_mode(mode);
        }
        <nn::Rope as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

/// SiLU-gated MLP.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
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

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let gated =
            nn::silu(self.gate_proj.forward(input)?)?.multiply(self.up_proj.forward(input)?)?;
        self.down_proj.forward(&gated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

/// A single transformer block (attention + MLP with residual connections).
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TransformerBlock {
    pub num_attention_heads: i32,
    pub hidden_size: i32,

    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: Mlp,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            num_attention_heads: args.num_attention_heads,
            hidden_size: args.hidden_size,
            self_attn: Attention::new(args)?,
            mlp: Mlp::new(args.hidden_size, args.intermediate_size)?,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;

        let normed = self.input_layernorm.forward(x)?;
        let residual = self.self_attn.forward(AttentionInput {
            x: &normed,
            mask,
            cache,
        })?;
        let h = x.add(residual)?;

        let normed_post = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed_post)?;
        h.add(mlp_out)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

/// Transformer model (embedding + layers + norm, without LM head).
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct TransformerModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl TransformerModel {
    fn new(args: &ModelArgs) -> Result<Self, Exception> {
        if !args.vocab_size.is_positive() {
            return Err(Exception::custom("vocab_size must be positive"));
        }
        if !args.num_hidden_layers.is_positive() {
            return Err(Exception::custom("num_hidden_layers must be positive"));
        }
        if !args.num_key_value_heads.is_positive() {
            return Err(Exception::custom("num_key_value_heads must be positive"));
        }

        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(
                args.vocab_size,
                args.hidden_size,
            )?),
            layers: (0..args.num_hidden_layers)
                .map(|_| TransformerBlock::new(args))
                .collect::<Result<Vec<_>, _>>()?,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

/// Input to the transformer model.
struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for TransformerModel
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        let mut h = self.embed_tokens.forward(inputs)?;

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
            h = layer.forward(AttentionInput {
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
            <TransformerBlock as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

/// Full causal language model with LM head.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    model: TransformerModel,

    #[quantizable]
    #[param]
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = TransformerModel::new(&args)?;
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

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    /// Run a forward pass returning hidden states before the LM head.
    pub fn forward_hidden<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        self.model.forward(ModelInput {
            inputs,
            mask,
            cache: kv_cache,
        })
    }

    /// Run a forward pass producing logits.
    pub fn forward<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        let out = self.forward_hidden(inputs, mask, kv_cache)?;
        self.apply_lm_head(&out)
    }

    /// Get the hidden size.
    pub const fn hidden_size(&self) -> i32 {
        self.args.hidden_size
    }

    /// Number of transformer layers.
    pub const fn num_layers(&self) -> i32 {
        self.args.num_hidden_layers
    }

    /// Look up token embeddings without running the transformer.
    pub fn embed_tokens(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        self.model.embed_tokens.forward(input_ids)
    }

    /// Forward pass starting from pre-computed embeddings (skips embedding lookup).
    /// Used by VLMs that merge text + image embeddings before running the transformer.
    pub fn forward_from_embeddings<C: KeyValueCache>(
        &mut self,
        embeddings: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        let computed_mask = match mask {
            Some(m) => Some(m.clone()),
            None => match create_attention_mask(embeddings, kv_cache, Some(true))? {
                Some(AttentionMask::Array(a)) => Some(a),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Only Array mask is supported"));
                }
                None => None,
            },
        };

        if kv_cache.is_empty() {
            *kv_cache = (0..self.model.layers.len()).map(|_| None).collect();
        } else if kv_cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "kv_cache length ({}) must match num layers ({})",
                kv_cache.len(),
                self.model.layers.len()
            )));
        }

        let mut h = embeddings.clone();
        for (layer, layer_cache) in self.model.layers.iter_mut().zip(kv_cache.iter_mut()) {
            h = layer.forward(AttentionInput {
                x: &h,
                mask: computed_mask.as_ref(),
                cache: layer_cache.as_mut(),
            })?;
        }

        let out = self.model.norm.forward(&h)?;
        self.apply_lm_head(&out)
    }

    /// Batched decode: one forward pass for N requests each with 1 token.
    ///
    /// Heavy ops (projections, MLP, LM head) run batched. Per-request ops
    /// (`RoPE`, KV cache update) loop over individual requests since each has
    /// a different position offset and cache state.
    #[allow(clippy::too_many_lines, clippy::indexing_slicing)]
    pub fn forward_batched(
        &mut self,
        inputs: &Array,
        kv_caches: &mut [&mut Vec<Option<SteppingKeyValueCache>>],
    ) -> Result<Array, Exception> {
        let n = *inputs
            .shape()
            .first()
            .ok_or_else(|| Exception::custom("inputs must have batch dimension"))?;
        let num_layers = self.model.layers.len();
        let n_usize = usize::try_from(n).map_err(|_| Exception::custom("batch size overflow"))?;
        if kv_caches.len() != n_usize {
            return Err(Exception::custom("kv_caches length must match batch size"));
        }
        for (i, cache) in kv_caches.iter().enumerate() {
            if cache.len() != num_layers {
                return Err(Exception::custom(format!(
                    "kv_cache[{i}] length ({}) must match num layers ({num_layers})",
                    cache.len()
                )));
            }
        }
        let head_dim = self.args.head_dim();

        // Per-request offsets (from layer 0's cache, all layers have the same offset)
        let offsets: Vec<i32> = kv_caches
            .iter()
            .map(|req| {
                req.first()
                    .and_then(Option::as_ref)
                    .map_or(0, KeyValueCache::offset)
            })
            .collect();
        let max_kv_len = offsets.iter().map(|&o| o + 1).max().unwrap_or(1);
        let kv_lengths: Vec<i32> = offsets.iter().map(|&o| o + 1).collect();

        let mut h = self.model.embed_tokens.forward(inputs)?;

        for (layer_idx, layer) in self.model.layers.iter_mut().enumerate() {
            let n_heads = layer.self_attn.n_heads;
            let n_kv_heads = layer.self_attn.n_kv_heads;
            let scale = layer.self_attn.scale;

            // Extract RoPE params as scalars (avoids borrow conflict with mutable layer)
            let rope_dims = layer.self_attn.rope.dimensions;
            let rope_traditional = layer.self_attn.rope.traditional;
            let rope_base = layer.self_attn.rope.base;
            let rope_scale = layer.self_attn.rope.scale;

            // --- Batched: layernorm + Q/K/V projections ---
            let normed = layer.input_layernorm.forward(&h)?;
            let q_raw = layer.self_attn.q_proj.forward(&normed)?;
            let k_raw = layer.self_attn.k_proj.forward(&normed)?;
            let v_raw = layer.self_attn.v_proj.forward(&normed)?;

            // [N, 1, proj_dim] -> [N, heads, 1, head_dim]
            let mut queries = q_raw
                .reshape(&[n, 1, n_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let mut keys = k_raw
                .reshape(&[n, 1, n_kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;
            let values = v_raw
                .reshape(&[n, 1, n_kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?;

            // --- Batched: QK norm (Qwen3) ---
            if let Some(ref mut qn) = layer.self_attn.q_norm {
                queries = qn.forward(&queries)?;
            }
            if let Some(ref mut kn) = layer.self_attn.k_norm {
                keys = kn.forward(&keys)?;
            }

            // --- Per-request: RoPE + KV cache update + pad ---
            // Flatten to 2D for reliable per-request slicing
            let q_flat = queries.reshape(&[n, n_heads * head_dim])?;
            let k_flat = keys.reshape(&[n, n_kv_heads * head_dim])?;
            let v_flat = values.reshape(&[n, n_kv_heads * head_dim])?;

            let mut all_queries = Vec::with_capacity(n_usize);
            let mut all_keys = Vec::with_capacity(n_usize);
            let mut all_values = Vec::with_capacity(n_usize);

            for (req_idx, &offset) in offsets.iter().enumerate() {
                let i = i32::try_from(req_idx)
                    .map_err(|_| Exception::custom("request index overflow"))?;

                let q_i = q_flat
                    .index((i..i + 1, ..))
                    .reshape(&[1, n_heads, 1, head_dim])?;
                let k_i = k_flat
                    .index((i..i + 1, ..))
                    .reshape(&[1, n_kv_heads, 1, head_dim])?;
                let v_i = v_flat
                    .index((i..i + 1, ..))
                    .reshape(&[1, n_kv_heads, 1, head_dim])?;

                // RoPE with this request's offset
                let q_rope = mlx_rs::fast::rope(
                    &q_i,
                    rope_dims,
                    rope_traditional,
                    rope_base,
                    rope_scale,
                    offset,
                    None,
                )?;
                let k_rope = mlx_rs::fast::rope(
                    &k_i,
                    rope_dims,
                    rope_traditional,
                    rope_base,
                    rope_scale,
                    offset,
                    None,
                )?;

                // Update this request's KV cache
                let cache = kv_caches[req_idx][layer_idx]
                    .as_mut()
                    .ok_or_else(|| Exception::custom("Cache not initialized"))?;
                let (full_k, full_v) = cache.update_and_fetch(k_rope, v_i)?;

                // Right-pad shorter caches to max_kv_len
                let seq_len = full_k.shape()[2];
                if seq_len < max_kv_len {
                    let pad_len = max_kv_len - seq_len;
                    let pad_k =
                        ops::zeros_dtype(&[1, n_kv_heads, pad_len, head_dim], full_k.dtype())?;
                    let pad_v =
                        ops::zeros_dtype(&[1, n_kv_heads, pad_len, head_dim], full_v.dtype())?;
                    all_keys.push(ops::concatenate_axis(&[&full_k, &pad_k], 2)?);
                    all_values.push(ops::concatenate_axis(&[&full_v, &pad_v], 2)?);
                } else {
                    all_keys.push(full_k);
                    all_values.push(full_v);
                }
                all_queries.push(q_rope);
            }

            // --- Batched: stack + SDPA + output proj + MLP ---
            let stacked_q = ops::concatenate_axis(&all_queries.iter().collect::<Vec<_>>(), 0)?;
            let stacked_k = ops::concatenate_axis(&all_keys.iter().collect::<Vec<_>>(), 0)?;
            let stacked_v = ops::concatenate_axis(&all_values.iter().collect::<Vec<_>>(), 0)?;

            let mask = create_batched_decode_mask(&kv_lengths, max_kv_len)?;

            let attn_out =
                scaled_dot_product_attention(stacked_q, stacked_k, stacked_v, scale, Some(&mask))?;

            let attn_flat = attn_out
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[n, 1, -1])?;
            let residual = layer.self_attn.o_proj.forward(&attn_flat)?;
            h = h.add(residual)?;

            let normed_post = layer.post_attention_layernorm.forward(&h)?;
            let mlp_out = layer.mlp.forward(&normed_post)?;
            h = h.add(mlp_out)?;
        }

        let out = self.model.norm.forward(&h)?;
        self.apply_lm_head(&out)
    }

    /// Apply the LM head to hidden states.
    fn apply_lm_head(&mut self, hidden: &Array) -> Result<Array, Exception> {
        match self.lm_head.as_mut() {
            Some(head) => head.forward(hidden),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed) => embed.as_linear(hidden),
                MaybeQuantized::Quantized(q_embed) => q_embed.as_linear(hidden),
            },
        }
    }
}

// --- Loading ---

/// Load model args from config.json.
pub fn load_model_args<P: AsRef<Path>>(model_dir: P) -> Result<ModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

/// Load model args from the `text_config` section of config.json (used by VLMs).
pub fn load_text_config_args<P: AsRef<Path>>(model_dir: P) -> Result<ModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    let config: serde_json::Value = serde_json::from_reader(file)?;

    let text_config = config
        .get("text_config")
        .ok_or_else(|| ModelError::UnsupportedModel("missing text_config in config.json".into()))?;

    // Merge top-level quantization config into text_config
    let mut text_obj = text_config.clone();
    if let Some(quant) = config.get("quantization") {
        if let Some(obj) = text_obj.as_object_mut() {
            obj.insert("quantization".to_owned(), quant.clone());
        }
    }
    // Also merge tie_word_embeddings from top level if not in text_config
    if text_obj.get("tie_word_embeddings").is_none() {
        if let Some(tie) = config.get("tie_word_embeddings") {
            if let Some(obj) = text_obj.as_object_mut() {
                obj.insert("tie_word_embeddings".to_owned(), tie.clone());
            }
        }
    }

    Ok(serde_json::from_value(text_obj)?)
}

/// Load a language model for a VLM.
///
/// Reads `text_config` from config.json and loads weights from safetensors
/// files, stripping the `language_model.` prefix from weight keys.
pub fn load_vlm_language_model<P: AsRef<Path>>(model_dir: P) -> Result<Model, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_text_config_args(model_path)?;

    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        "Loading VLM language model"
    );

    let quantization = args.quantization.clone();
    let raw_model = Model::new(args)?;

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

    crate::load_quantized_safetensors_weights_with_prefix(
        &mut model,
        model_path,
        quantization.is_some(),
        "language_model.",
    )?;

    tracing::info!("VLM language model loaded successfully");
    Ok(model)
}

/// Load a model from a directory containing safetensors + config.json.
pub fn load_model<P: AsRef<Path>>(model_dir: P) -> Result<Model, ModelError> {
    let model_path = model_dir.as_ref();

    let args = load_model_args(model_path)?;
    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        vocab_size = args.vocab_size,
        qkv_bias = args.qkv_bias(),
        "Loading model"
    );

    let quantization = args.quantization.clone();
    let raw_model = Model::new(args)?;

    // Pre-quantized models need the MaybeQuantized fields converted to
    // Quantized variants before loading weights, so that the parameter
    // names (inner.weight, scales, biases) match the safetensors keys.
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

    tracing::info!("Model loaded successfully");
    Ok(model)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Create a `ModelArgs` with sensible defaults. Only fields that vary
    /// between tests need to be overridden after construction.
    fn default_model_args() -> ModelArgs {
        ModelArgs {
            model_type: "llama".to_owned(),
            hidden_size: 256,
            num_hidden_layers: 2,
            intermediate_size: 512,
            num_attention_heads: 4,
            rms_norm_eps: 1e-6,
            vocab_size: 1000,
            num_key_value_heads: 2,
            max_position_embeddings: 512,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            attention_bias: None,
            use_sliding_window: false,
            sliding_window: None,
            rope_scaling: None,
            quantization: None,
        }
    }

    /// Create a `ModelArgs` with the given core parameters and defaults for
    /// everything else.
    fn make_model_args(
        model_type: &str,
        hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        vocab_size: i32,
        num_layers: i32,
    ) -> ModelArgs {
        ModelArgs {
            model_type: model_type.to_owned(),
            hidden_size,
            num_attention_heads: num_heads,
            num_key_value_heads: num_kv_heads,
            vocab_size,
            num_hidden_layers: num_layers,
            ..default_model_args()
        }
    }

    #[test]
    fn test_qwen2_config_deserialization() {
        let json = r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "intermediate_size": 8960,
            "num_attention_heads": 12,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true,
            "use_sliding_window": false,
            "sliding_window": 32768
        }"#;

        let args = assert_model_config(json, "qwen2", 1536, 128, true);
        assert_eq!(args.num_hidden_layers, 28);
        assert_eq!(args.num_attention_heads, 12);
        assert_eq!(args.num_key_value_heads, 2);
        assert!(args.tie_word_embeddings);
    }

    fn assert_model_config(
        json: &str,
        expected_type: &str,
        expected_hidden: i32,
        expected_head_dim: i32,
        expected_qkv_bias: bool,
    ) -> ModelArgs {
        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, expected_type);
        assert_eq!(args.hidden_size, expected_hidden);
        assert_eq!(args.head_dim(), expected_head_dim);
        assert_eq!(args.qkv_bias(), expected_qkv_bias);
        args
    }

    #[test]
    fn test_llama_config_deserialization() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "rms_norm_eps": 1e-06,
            "vocab_size": 32000,
            "num_key_value_heads": 32,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }"#;

        let args = assert_model_config(json, "llama", 4096, 128, false);
        assert!(!args.tie_word_embeddings);
    }

    #[test]
    fn test_llama_config_defaults() {
        // Verify serde defaults when optional fields are omitted
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 22,
            "intermediate_size": 5632,
            "num_attention_heads": 32,
            "rms_norm_eps": 1e-05,
            "vocab_size": 32000,
            "num_key_value_heads": 4,
            "max_position_embeddings": 2048
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert!((args.rope_theta - 10000.0).abs() < f32::EPSILON);
        assert!(!args.tie_word_embeddings);
    }

    #[test]
    fn test_mistral_config_deserialization() {
        let json = r#"{
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "rms_norm_eps": 1e-05,
            "vocab_size": 32000,
            "num_key_value_heads": 8,
            "max_position_embeddings": 32768,
            "rope_theta": 10000.0,
            "sliding_window": 4096
        }"#;

        let args = assert_model_config(json, "mistral", 4096, 128, false);
        assert_eq!(args.sliding_window, Some(4096));
    }

    #[test]
    fn test_mistral_config_no_sliding_window() {
        let args = make_model_args("mistral", 2048, 32, 4, 32000, 22);
        assert!(args.sliding_window.is_none());
        assert!((args.rope_theta - 10000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_head_dim_computation() {
        let args = make_model_args("qwen2", 768, 12, 4, 32000, 12);
        assert_eq!(args.head_dim(), 64);
    }

    #[test]
    fn test_checked_head_dim_zero_heads() {
        let args = make_model_args("llama", 768, 0, 4, 32000, 12);
        assert!(args.checked_head_dim().is_err());
    }

    #[test]
    fn test_checked_head_dim_not_divisible() {
        let args = make_model_args("llama", 100, 3, 1, 32000, 12);
        assert!(args.checked_head_dim().is_err());
    }

    #[test]
    fn test_qkv_bias_for_qwen3() {
        // qwen3 uses bias=false (unlike qwen2)
        let args = make_model_args("qwen3", 2048, 16, 2, 151_936, 28);
        assert!(!args.qkv_bias());
    }

    #[test]
    fn test_load_model_args_missing_file_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_model_args(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_model_args_invalid_json_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "not json").unwrap();
        let result = load_model_args(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_quantization_config_deserialization() {
        let mut args = default_model_args();
        args.model_type = "qwen2".to_owned();
        args.quantization = Some(QuantizationConfig {
            group_size: 64,
            bits: 4,
        });
        let qc = args.quantization.unwrap();
        assert_eq!(qc.group_size, 64);
        assert_eq!(qc.bits, 4);
    }

    #[test]
    fn test_no_quantization_config() {
        let args = make_model_args("llama", 4096, 32, 32, 32000, 32);
        assert!(args.quantization.is_none());
    }

    #[test]
    fn test_model_args_missing_optional_fields_use_defaults() {
        let args = default_model_args();
        assert!((args.rope_theta - 10000.0).abs() < f32::EPSILON);
        assert!(!args.tie_word_embeddings);
        assert!(!args.use_sliding_window);
        assert!(args.sliding_window.is_none());
        assert!(args.rope_scaling.is_none());
        assert!(args.quantization.is_none());
    }

    #[test]
    fn test_checked_head_dim_valid_cases() {
        // 768 / 12 = 64
        let args = make_model_args("qwen2", 768, 12, 4, 32000, 12);
        assert_eq!(args.checked_head_dim().unwrap(), 64);

        // 4096 / 32 = 128
        let args2 = make_model_args("llama", 4096, 32, 32, 32000, 32);
        assert_eq!(args2.checked_head_dim().unwrap(), 128);

        // 256 / 4 = 64
        let args3 = make_model_args("mistral", 256, 4, 2, 1000, 2);
        assert_eq!(args3.checked_head_dim().unwrap(), 64);
    }

    #[test]
    fn test_checked_head_dim_error_messages() {
        // Zero heads
        let args = make_model_args("llama", 768, 0, 4, 32000, 12);
        let err = args.checked_head_dim().unwrap_err();
        assert!(err.to_string().contains("positive"));

        // Not divisible
        let args2 = make_model_args("llama", 100, 7, 1, 1000, 2);
        let err2 = args2.checked_head_dim().unwrap_err();
        assert!(err2.to_string().contains("divisible"));
    }

    #[test]
    fn test_model_new_zero_num_hidden_layers() {
        let args = make_model_args("llama", 256, 4, 2, 1000, 0);
        let result = Model::new(args);
        assert!(result.is_err(), "Should reject num_hidden_layers == 0");
    }

    #[test]
    fn test_model_new_zero_num_key_value_heads() {
        let args = make_model_args("llama", 256, 4, 0, 1000, 2);
        let result = Model::new(args);
        assert!(result.is_err(), "Should reject num_key_value_heads == 0");
    }

    #[test]
    fn test_model_new_zero_vocab_size() {
        let args = make_model_args("llama", 256, 4, 2, 0, 2);
        let result = Model::new(args);
        assert!(result.is_err(), "Should reject vocab_size == 0");
    }

    #[test]
    fn test_model_new_valid_with_tied_embeddings() {
        let mut args = default_model_args();
        args.tie_word_embeddings = true;
        let model = Model::new(args).unwrap();
        assert_eq!(model.model_type(), "llama");
        assert!(model.lm_head.is_none());
    }

    #[test]
    fn test_model_new_valid_without_tied_embeddings() {
        let args = make_model_args("qwen2", 256, 4, 2, 1000, 2);
        let model = Model::new(args).unwrap();
        assert_eq!(model.model_type(), "qwen2");
        assert!(model.lm_head.is_some());
    }

    /// Write a minimal config.json to a tempdir and return the directory.
    fn write_model_config(model_type: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let config = format!(
            r#"{{"model_type":"{model_type}","hidden_size":256,"num_hidden_layers":2,"intermediate_size":512,"num_attention_heads":4,"rms_norm_eps":1e-06,"vocab_size":1000,"num_key_value_heads":2,"max_position_embeddings":512}}"#
        );
        std::fs::write(dir.path().join("config.json"), config).unwrap();
        dir
    }

    fn assert_loaded_model_config(model_type: &str, expected_qkv_bias: bool) {
        let dir = write_model_config(model_type);
        let args = load_model_args(dir.path()).unwrap();
        assert_eq!(args.model_type, model_type);
        assert_eq!(args.qkv_bias(), expected_qkv_bias);
    }

    #[test]
    fn test_load_model_args_valid_qwen3_config() {
        assert_loaded_model_config("qwen3", false);
    }

    #[test]
    fn test_load_model_args_valid_llama_config() {
        assert_loaded_model_config("llama", false);
    }

    #[test]
    fn test_qkv_bias_explicit_attention_bias_overrides_default() {
        let mut args = make_model_args("llama", 256, 4, 2, 1000, 2);
        // llama defaults to no bias, but explicit config overrides
        args.attention_bias = Some(true);
        assert!(args.qkv_bias());

        let mut args2 = make_model_args("qwen2", 256, 4, 2, 1000, 2);
        // qwen2 defaults to bias, but explicit config overrides
        args2.attention_bias = Some(false);
        assert!(!args2.qkv_bias());
    }

    #[test]
    fn test_qkv_bias_for_unsupported_types() {
        let args = make_model_args("custom_arch", 256, 4, 2, 1000, 2);
        assert!(!args.qkv_bias());
    }

    #[test]
    fn test_qwen3_config_deserialization_with_attention_bias() {
        let json = r#"{
            "model_type": "qwen3",
            "hidden_size": 2048,
            "num_hidden_layers": 36,
            "intermediate_size": 11008,
            "num_attention_heads": 16,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "num_key_value_heads": 2,
            "max_position_embeddings": 40960,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true,
            "attention_bias": false
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.attention_bias, Some(false));
        assert!(!args.qkv_bias());
    }

    #[test]
    fn test_model_new_negative_num_hidden_layers() {
        let args = make_model_args("llama", 256, 4, 2, 1000, -1);
        let result = Model::new(args);
        assert!(result.is_err(), "Should reject negative num_hidden_layers");
    }
}
