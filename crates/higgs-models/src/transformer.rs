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
    nn,
    ops::indexing::IndexOp,
    quantization::MaybeQuantized,
};
use serde::Deserialize;

use crate::qwen3_next::{QEmbedding, QLinear};
use crate::{
    batched_cache::BatchedKvCache,
    cache::KeyValueCache,
    error::ModelError,
    utils::{
        AttentionMask, apply_rope, apply_rope_at_cache, cached_scaled_dot_product_attention,
        create_attention_mask, scaled_dot_product_attention,
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

pub use crate::quant_config::QuantizationSettings as QuantizationConfig;

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
    /// Explicit attention head dimension (`head_dim` in config.json). Most
    /// Llama-family configs omit it, in which case `head_dim()` falls back to
    /// `hidden_size / num_attention_heads`. Some models set it to a value that
    /// differs from that ratio — e.g. `MiniCPM5` uses 128 while 1536/16 = 96 —
    /// and the attention projections, `RoPE`, and scale must use the explicit
    /// value or the model produces garbage.
    #[serde(default, rename = "head_dim")]
    pub head_dim_override: Option<i32>,

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

    /// Head dimension. Uses the explicit `head_dim` from config when present,
    /// otherwise `hidden_size / num_attention_heads`.
    ///
    /// Panics in debug builds if the fallback is not evenly divisible.
    pub fn head_dim(&self) -> i32 {
        if let Some(head_dim) = self.head_dim_override {
            return head_dim;
        }
        debug_assert!(
            self.num_attention_heads != 0 && self.hidden_size % self.num_attention_heads == 0,
            "hidden_size ({}) must be divisible by num_attention_heads ({})",
            self.hidden_size,
            self.num_attention_heads
        );
        self.hidden_size / self.num_attention_heads
    }

    /// Validated head dimension. Honours an explicit `head_dim` from config;
    /// otherwise returns an error if `hidden_size` is not evenly divisible by
    /// `num_attention_heads`.
    pub fn checked_head_dim(&self) -> Result<i32, ModelError> {
        if let Some(head_dim) = self.head_dim_override {
            if head_dim <= 0 {
                return Err(ModelError::ShapeMismatch(
                    "explicit head_dim must be positive".to_owned(),
                ));
            }
            return Ok(head_dim);
        }
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
            .scale(1.0_f32)
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
        let values = v_raw
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        if let Some(ref mut kv_cache) = cache {
            queries = apply_rope_at_cache(&queries, &self.rope, kv_cache)?;
            keys = apply_rope_at_cache(&keys, &self.rope, kv_cache)?;

            let output = cached_scaled_dot_product_attention(
                queries, kv_cache, keys, values, self.scale, mask,
            )?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;

            return self.o_proj.forward(&output);
        }
        queries = apply_rope(&queries, &self.rope, 0)?;
        keys = apply_rope(&keys, &self.rope, 0)?;

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

    #[param]
    pub embed_tokens: QEmbedding,
    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl TransformerModel {
    fn new(args: &ModelArgs, embed_group_size: i32, embed_bits: i32) -> Result<Self, Exception> {
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
            embed_tokens: QEmbedding::new(embed_group_size, embed_bits)?,
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

    #[param]
    lm_head: Option<QLinear>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let default_quant = args.quantization.as_ref();
        let embed_quant = default_quant.map_or_else(
            || crate::quant_config::TensorQuant::Unquantized,
            |settings| settings.resolve("model.embed_tokens"),
        );
        let model = TransformerModel::new(&args, embed_quant.group_size(), embed_quant.bits())?;
        let lm_head_quant = default_quant.map_or_else(
            || crate::quant_config::TensorQuant::Unquantized,
            |settings| settings.resolve("lm_head"),
        );
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(QLinear::new(
                lm_head_quant.group_size(),
                lm_head_quant.bits(),
            )?)
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
        let hidden = self.forward_hidden(inputs, mask, kv_cache)?;
        let last = hidden.index((.., -1.., ..));
        self.apply_lm_head(&last)
    }

    /// Run a forward pass producing logits for every input position.
    pub fn forward_all_logits<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        let hidden = self.forward_hidden(inputs, mask, kv_cache)?;
        self.apply_lm_head_all(&hidden)
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

    /// Batched decode: one forward pass appending one token to each of the
    /// `B` sequences held in `cache`. `inputs` is `[B, 1]`.
    pub fn forward_batched(
        &mut self,
        inputs: &Array,
        cache: &mut BatchedKvCache,
    ) -> Result<Array, Exception> {
        let mask = cache.attention_mask(1)?;
        self.forward(inputs, mask.as_ref(), cache.layers_mut())
    }

    /// Prefill several prompts in one forward pass.
    ///
    /// `inputs` is `[B, T]` with each prompt **left-padded** to the longest
    /// (`T`), and `lengths` holds the true token counts. Returns last-position
    /// logits `[B, 1, vocab]` and the filled cache. Real tokens are rotated at
    /// their true positions, so the rows are interchangeable with caches
    /// built by an unbatched prefill.
    pub fn prefill_batched(
        &mut self,
        inputs: &Array,
        lengths: &[i32],
    ) -> Result<(Array, BatchedKvCache), Exception> {
        let steps = crate::batched_cache::validate_prefill_shape(inputs, lengths)?;
        let mut cache = BatchedKvCache::for_prefill(self.model.layers.len(), lengths, steps);
        let mask = BatchedKvCache::prefill_mask(lengths, steps)?;
        let logits = self.forward(inputs, Some(&mask), cache.layers_mut())?;
        Ok((logits, cache))
    }

    /// Apply the LM head to hidden states (last position only during prefill).
    #[allow(non_snake_case)]
    fn apply_lm_head(&mut self, hidden: &Array) -> Result<Array, Exception> {
        let t = hidden.shape().get(1).copied().unwrap_or(1);
        let lm_input = if t > 1 {
            hidden.index((.., -1.., ..))
        } else {
            hidden.clone()
        };
        match self.lm_head.as_mut() {
            Some(head) => head.forward(&lm_input),
            None => self.model.embed_tokens.as_linear(&lm_input),
        }
    }

    fn apply_lm_head_all(&mut self, hidden: &Array) -> Result<Array, Exception> {
        match self.lm_head.as_mut() {
            Some(head) => head.forward(hidden),
            None => self.model.embed_tokens.as_linear(hidden),
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

pub(crate) fn model_args_from_value(config: &serde_json::Value) -> Result<ModelArgs, ModelError> {
    Ok(serde_json::from_value(config.clone())?)
}

/// Load model args from the `text_config` section of config.json (used by VLMs).
pub fn load_text_config_args<P: AsRef<Path>>(model_dir: P) -> Result<ModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    let config: serde_json::Value = serde_json::from_reader(file)?;
    text_model_args_from_value(&config)
}

pub(crate) fn text_model_args_from_value(
    config: &serde_json::Value,
) -> Result<ModelArgs, ModelError> {
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

    model_args_from_value(&text_obj)
}

/// Load a language model for a VLM.
///
/// Reads `text_config` from config.json and loads weights from safetensors
/// files, stripping the `language_model.` prefix from weight keys.
pub fn load_vlm_language_model<P: AsRef<Path>>(model_dir: P) -> Result<Model, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_text_config_args(model_path)?;
    load_vlm_language_model_with_args(model_path, args)
}

pub(crate) fn load_vlm_language_model_with_args(
    model_path: &Path,
    args: ModelArgs,
) -> Result<Model, ModelError> {
    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        "Loading VLM language model"
    );

    let quantization = args.quantization.clone();
    if let Some(settings) = quantization.as_ref() {
        crate::validate_per_tensor_quantization_support(
            settings,
            &["model.embed_tokens", "lm_head"],
        )?;
    }
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

    crate::load_quantized_safetensors_weights_with_prefix_and_settings(
        &mut model,
        model_path,
        quantization.is_some(),
        "language_model.",
        quantization.as_ref(),
    )?;

    tracing::info!("VLM language model loaded successfully");
    Ok(model)
}

/// Load a model from a directory containing safetensors + config.json.
pub fn load_model<P: AsRef<Path>>(model_dir: P) -> Result<Model, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_model_args(model_path)?;
    load_model_with_args(model_path, args)
}

pub(crate) fn load_model_with_args(
    model_path: &Path,
    args: ModelArgs,
) -> Result<Model, ModelError> {
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
    if let Some(settings) = quantization.as_ref() {
        crate::validate_per_tensor_quantization_support(
            settings,
            &["model.embed_tokens", "lm_head"],
        )?;
    }
    let raw_model = Model::new(args)?;

    // Pre-quantized scalar layers need their MaybeQuantized fields converted
    // to Quantized variants before loading weights, so that the parameter
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

    crate::load_quantized_safetensors_weights_with_settings(
        &mut model,
        model_path,
        quantization.is_some(),
        quantization.as_ref(),
    )?;

    tracing::info!("Model loaded successfully");
    Ok(model)
}

#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::shadow_reuse,
    clippy::shadow_unrelated
)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::SteppingKeyValueCache;
    use mlx_rs::module::ModuleParameters as _;

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
            head_dim_override: None,
            quantization: None,
        }
    }

    #[test]
    fn keeps_dense_embedding_and_lm_head_outside_global_quantization()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut args = default_model_args();
        args.quantization = Some(serde_json::from_str(
            r#"{"group_size":64,"bits":4,"model.embed_tokens":false,"lm_head":false}"#,
        )?);
        let model = Model::new(args)?;
        let quantized = mlx_rs::nn::quantize(model, 64, 4)?;
        let params = quantized.parameters().flatten();

        assert!(params.contains_key("model.embed_tokens.weight"));
        assert!(!params.contains_key("model.embed_tokens.inner.weight"));
        assert!(params.contains_key("lm_head.weight"));
        assert!(!params.contains_key("lm_head.inner.weight"));
        assert!(params.contains_key("model.layers.0.self_attn.q_proj.inner.weight"));
        Ok(())
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
        args.quantization = Some(QuantizationConfig::new(64, 4));
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
    fn test_explicit_head_dim_honored() {
        // MiniCPM5-1B: head_dim=128 even though hidden/heads = 1536/16 = 96.
        // Must use the explicit value or attention/RoPE are wrong.
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 1536,
            "num_hidden_layers": 24,
            "intermediate_size": 4608,
            "num_attention_heads": 16,
            "rms_norm_eps": 1e-06,
            "vocab_size": 130560,
            "num_key_value_heads": 2,
            "max_position_embeddings": 131072,
            "head_dim": 128
        }"#;
        let args = assert_model_config(json, "llama", 1536, 128, false);
        assert_eq!(args.head_dim_override, Some(128));
        assert_eq!(args.checked_head_dim().unwrap(), 128);
    }

    #[test]
    fn test_head_dim_falls_back_when_config_omits_it() {
        // Standard Llama config has no `head_dim` → fall back to 4096/32 = 128.
        let args = make_model_args("llama", 4096, 32, 32, 32000, 32);
        assert_eq!(args.head_dim_override, None);
        assert_eq!(args.head_dim(), 128);
        assert_eq!(args.checked_head_dim().unwrap(), 128);
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

    /// Largest absolute difference between two same-shaped f32 arrays.
    fn max_abs_difference(left: &Array, right: &Array) -> f32 {
        let left = left.contiguous().unwrap();
        let right = right.contiguous().unwrap();
        left.as_slice::<f32>()
            .iter()
            .zip(right.as_slice::<f32>())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    /// Batched decode must reproduce one-at-a-time decode. `Model::new` only
    /// creates placeholder embedding weights, so this needs a real checkpoint:
    /// `HIGGS_MODEL_PATH=<llama/qwen/mistral dir> cargo test -p higgs-models -- --ignored forward_batched`
    #[test]
    #[ignore = "requires real model weights (set HIGGS_MODEL_PATH)"]
    fn forward_batched_matches_per_sequence_forward() {
        let Ok(model_path) = std::env::var("HIGGS_MODEL_PATH") else {
            println!("HIGGS_MODEL_PATH not set, skipping");
            return;
        };
        let mut model = load_model(&model_path).unwrap();
        let layers = model.model.layers.len();
        // Different prompt lengths exercise left padding, masking, and
        // per-row rotary positions.
        let prompts: [&[u32]; 3] = [
            &[101, 2002, 3003, 404, 505, 6006, 707],
            &[909, 808, 7007],
            &[1100, 1200, 130, 1400, 150],
        ];
        let steps: [[u32; 3]; 3] = [[210, 2200, 230], [3100, 320, 3300], [41, 4200, 430]];

        let prefill = |model: &mut Model, prompt: &[u32]| {
            let mut cache: Vec<Option<SteppingKeyValueCache>> = (0..layers)
                .map(|_| Some(SteppingKeyValueCache::new()))
                .collect();
            let length = i32::try_from(prompt.len()).unwrap();
            let logits = model
                .forward(&Array::from_slice(prompt, &[1, length]), None, &mut cache)
                .unwrap();
            logits.eval().unwrap();
            cache
        };

        let mut reference_caches: Vec<_> = prompts.iter().map(|p| prefill(&mut model, p)).collect();
        let rows: Vec<_> = prompts.iter().map(|p| prefill(&mut model, p)).collect();
        let mut batch = BatchedKvCache::from_rows(rows).unwrap();

        let mut worst_relative = 0.0_f32;
        for step_tokens in steps {
            let batched = model
                .forward_batched(&Array::from_slice(&step_tokens, &[3, 1]), &mut batch)
                .unwrap()
                .as_dtype(mlx_rs::Dtype::Float32)
                .unwrap();
            for ((cache, &token), row) in reference_caches.iter_mut().zip(&step_tokens).zip(0_i32..)
            {
                let expected = model
                    .forward(&Array::from_slice(&[token], &[1, 1]), None, cache)
                    .unwrap()
                    .as_dtype(mlx_rs::Dtype::Float32)
                    .unwrap();
                let actual = batched.index((row..row + 1, .., ..));
                let largest_logit = expected
                    .as_slice::<f32>()
                    .iter()
                    .fold(0.0_f32, |max, logit| max.max(logit.abs()));
                worst_relative =
                    worst_relative.max(max_abs_difference(&expected, &actual) / largest_logit);
            }
        }
        println!("largest logit difference relative to the largest logit: {worst_relative}");
        // Half-precision kernels round differently at batch 3 than at batch 1:
        // the standard forward shows ~1.5% on Qwen3-1.7B-4bit and ~0.4% on
        // Llama-3.2-1B-4bit. A positional or masking bug is an order of
        // magnitude larger.
        assert!(
            worst_relative < 0.03,
            "batched logits differ from unbatched by {worst_relative} of the largest logit"
        );
        assert_eq!(batch.lengths(), vec![10, 6, 8]);
    }

    /// Batched prefill must leave the same logits and the same decode
    /// behavior as prefilling each prompt alone.
    /// `HIGGS_MODEL_PATH=<llama/qwen/mistral dir> cargo test -p higgs-models -- --ignored prefill_batched`
    #[test]
    #[ignore = "requires real model weights (set HIGGS_MODEL_PATH)"]
    fn prefill_batched_matches_per_sequence_prefill() {
        let Ok(model_path) = std::env::var("HIGGS_MODEL_PATH") else {
            println!("HIGGS_MODEL_PATH not set, skipping");
            return;
        };
        let mut model = load_model(&model_path).unwrap();
        let layers = model.model.layers.len();
        let prompts: [&[u32]; 3] = [
            &[101, 2002, 3003, 404, 505, 6006, 707],
            &[909, 808, 7007],
            &[1100, 1200, 130, 1400, 150],
        ];
        let next_tokens = [210_u32, 2200, 230];
        const LONGEST: usize = 7;
        const PAD_TOKEN: u32 = 0;

        let mut padded = Vec::new();
        for prompt in prompts {
            padded.extend(std::iter::repeat_n(PAD_TOKEN, LONGEST - prompt.len()));
            padded.extend_from_slice(prompt);
        }
        let lengths: Vec<i32> = prompts
            .iter()
            .map(|p| i32::try_from(p.len()).unwrap())
            .collect();
        let (batched_logits, mut batch) = model
            .prefill_batched(&Array::from_slice(&padded, &[3, 7]), &lengths)
            .unwrap();
        let batched_logits = batched_logits.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let batched_step = model
            .forward_batched(&Array::from_slice(&next_tokens, &[3, 1]), &mut batch)
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap();

        let relative_difference = |expected: &Array, actual: &Array| {
            let largest_logit = expected
                .as_slice::<f32>()
                .iter()
                .fold(0.0_f32, |max, logit| max.max(logit.abs()));
            max_abs_difference(expected, actual) / largest_logit
        };
        let mut worst_relative = 0.0_f32;
        for ((prompt, &next_token), row) in prompts.iter().zip(&next_tokens).zip(0_i32..) {
            let mut cache: Vec<Option<SteppingKeyValueCache>> = (0..layers)
                .map(|_| Some(SteppingKeyValueCache::new()))
                .collect();
            let length = i32::try_from(prompt.len()).unwrap();
            let expected_prefill = model
                .forward(&Array::from_slice(prompt, &[1, length]), None, &mut cache)
                .unwrap()
                .as_dtype(mlx_rs::Dtype::Float32)
                .unwrap();
            let expected_step = model
                .forward(&Array::from_slice(&[next_token], &[1, 1]), None, &mut cache)
                .unwrap()
                .as_dtype(mlx_rs::Dtype::Float32)
                .unwrap();
            let row_range = row..row + 1;
            worst_relative = worst_relative
                .max(relative_difference(
                    &expected_prefill,
                    &batched_logits.index((row_range.clone(), .., ..)),
                ))
                .max(relative_difference(
                    &expected_step,
                    &batched_step.index((row_range, .., ..)),
                ));
        }
        println!("largest logit difference relative to the largest logit: {worst_relative}");
        assert!(
            worst_relative < 0.03,
            "batched prefill differs from unbatched by {worst_relative} of the largest logit"
        );
        assert!(
            batch.eval_targets().iter().all(|array| !array
                .as_dtype(mlx_rs::Dtype::Float32)
                .unwrap()
                .sum(None)
                .unwrap()
                .item_cast::<f32>()
                .is_nan()),
            "padding must not leave NaN in the cache"
        );
    }

    /// Steady-state decode step time by batch size, for the batched path and
    /// the standard forward at the same batch size.
    /// `HIGGS_MODEL_PATH=<dir> cargo test --release -p higgs-models -- --ignored bench_batched_decode_step --nocapture`
    #[test]
    #[ignore = "benchmark, requires real model weights (set HIGGS_MODEL_PATH)"]
    fn bench_batched_decode_step() {
        let Ok(model_path) = std::env::var("HIGGS_MODEL_PATH") else {
            println!("HIGGS_MODEL_PATH not set, skipping");
            return;
        };
        let mut model = load_model(&model_path).unwrap();
        let layers = model.model.layers.len();
        const PROMPT_TOKENS: i32 = 64;
        const STEPS: usize = 200;

        for batch_size in [1_i32, 2, 4, 8] {
            let prompt: Vec<u32> = (0..batch_size * PROMPT_TOKENS)
                .map(|i| 100 + u32::try_from(i % 5000).unwrap())
                .collect();
            let fresh = || -> Vec<Option<SteppingKeyValueCache>> {
                (0..layers)
                    .map(|_| Some(SteppingKeyValueCache::new()))
                    .collect()
            };

            let mut standard_cache = fresh();
            let prefill = Array::from_slice(&prompt, &[batch_size, PROMPT_TOKENS]);
            let mut tokens =
                argmax_last(&model.forward(&prefill, None, &mut standard_cache).unwrap());
            tokens.eval().unwrap();
            let started = std::time::Instant::now();
            for _ in 0..STEPS {
                let logits = model.forward(&tokens, None, &mut standard_cache).unwrap();
                tokens = argmax_last(&logits);
                tokens.eval().unwrap();
            }
            let standard_ms = started.elapsed().as_secs_f64() * 1000.0 / STEPS as f64;

            let rows: Vec<_> = (0..batch_size)
                .map(|row| {
                    let mut cache = fresh();
                    let start = usize::try_from(row * PROMPT_TOKENS).unwrap();
                    let end = start + usize::try_from(PROMPT_TOKENS).unwrap();
                    model
                        .forward(
                            &Array::from_slice(&prompt[start..end], &[1, PROMPT_TOKENS]),
                            None,
                            &mut cache,
                        )
                        .unwrap()
                        .eval()
                        .unwrap();
                    cache
                })
                .collect();
            let mut batch = BatchedKvCache::from_rows(rows).unwrap();
            let mut tokens = Array::from_slice(
                &vec![7_u32; usize::try_from(batch_size).unwrap()],
                &[batch_size, 1],
            );
            let started = std::time::Instant::now();
            for _ in 0..STEPS {
                let logits = model.forward_batched(&tokens, &mut batch).unwrap();
                tokens = argmax_last(&logits);
                tokens.eval().unwrap();
            }
            let batched_ms = started.elapsed().as_secs_f64() * 1000.0 / STEPS as f64;
            println!(
                "batch {batch_size}: standard forward {standard_ms:.3} ms/step, forward_batched {batched_ms:.3} ms/step"
            );
        }
    }

    /// Greedy next tokens as `[B, 1]` from `[B, T, vocab]` logits.
    fn argmax_last(logits: &Array) -> Array {
        mlx_rs::ops::indexing::argmax_axis(logits.index((.., -1, ..)), -1, true).unwrap()
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
