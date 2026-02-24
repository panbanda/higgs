//! Starcoder2 model implementation.
//!
//! Differs from the standard transformer in several ways:
//! - `LayerNorm` (not `RMSNorm`) with both weight and bias
//! - Standard 2-layer MLP with `c_fc`/`c_proj` (not gated)
//! - GELU activation (`gelu_pytorch_tanh`)
//! - `use_bias` for all linear projections
//! - Sliding window attention on all layers

use std::path::Path;

use mlx_rs::{
    Array, arange, array,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    quantization::MaybeQuantized,
};
use serde::Deserialize;

use crate::{
    cache::KeyValueCache,
    error::ModelError,
    utils::{apply_rope, create_causal_mask, scaled_dot_product_attention},
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const fn default_rope_theta() -> f32 {
    10000.0
}

/// Quantization parameters from config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

/// Starcoder2 model configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Starcoder2ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "LayerNorm::default_eps")]
    pub norm_epsilon: f32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_bias: bool,
    #[serde(default)]
    pub sliding_window: Option<i32>,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

/// Wrapper to provide a default fn for serde.
struct LayerNorm;
impl LayerNorm {
    const fn default_eps() -> f32 {
        1e-5
    }
}

impl Starcoder2ModelArgs {
    fn head_dim(&self) -> i32 {
        debug_assert!(
            self.num_attention_heads > 0,
            "num_attention_heads must be positive"
        );
        self.hidden_size / self.num_attention_heads
    }

    fn checked_head_dim(&self) -> Result<i32, ModelError> {
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

// ---------------------------------------------------------------------------
// Sliding window + causal mask
// ---------------------------------------------------------------------------

/// Create a combined causal + sliding window boolean mask.
///
/// For query at position `q_abs`, key at position `k` is visible when
/// `k <= q_abs` (causal) AND `k >= q_abs - window + 1` (sliding window).
#[allow(non_snake_case)]
fn create_sliding_causal_mask(
    query_len: i32,
    kv_len: i32,
    window: i32,
) -> Result<Array, Exception> {
    let offset = kv_len - query_len;

    let q_pos = arange!(start = offset, stop = offset + query_len)?.reshape(&[query_len, 1])?;
    let k_pos = arange!(stop = kv_len)?.reshape(&[1, kv_len])?;

    // diff[q, k] = q_abs - k; visible when 0 <= diff < window
    let diff = q_pos.subtract(&k_pos)?;
    let lower = diff.ge(&array!(0_i32))?;
    let upper = diff.lt(&array!(window))?;
    lower.multiply(upper)
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Starcoder2Attention {
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,

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

impl Starcoder2Attention {
    fn new(args: &Starcoder2ModelArgs) -> Result<Self, Exception> {
        let head_dim = args
            .checked_head_dim()
            .map_err(|e| Exception::custom(e.to_string()))?;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        let scale = head_dim_f32.sqrt().recip();
        let bias = args.use_bias;

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

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            rope,
        })
    }
}

struct Starcoder2AttentionInput<'a, C> {
    x: &'a Array,
    mask: Option<&'a Array>,
    cache: Option<&'a mut C>,
}

impl<C> Module<Starcoder2AttentionInput<'_, C>> for Starcoder2Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        input: Starcoder2AttentionInput<'_, C>,
    ) -> Result<Self::Output, Self::Error> {
        let Starcoder2AttentionInput { x, mask, mut cache } = input;

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
        <nn::Rope as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

// ---------------------------------------------------------------------------
// MLP (standard 2-layer with GELU)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Starcoder2Mlp {
    #[quantizable]
    #[param]
    c_fc: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    c_proj: MaybeQuantized<nn::Linear>,
}

impl Starcoder2Mlp {
    fn new(dim: i32, intermediate_size: i32, bias: bool) -> Result<Self, Exception> {
        let c_fc = nn::LinearBuilder::new(dim, intermediate_size)
            .bias(bias)
            .build()?;
        let c_proj = nn::LinearBuilder::new(intermediate_size, dim)
            .bias(bias)
            .build()?;

        Ok(Self {
            c_fc: MaybeQuantized::Original(c_fc),
            c_proj: MaybeQuantized::Original(c_proj),
        })
    }
}

impl Module<&Array> for Starcoder2Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let h = nn::gelu_approximate(self.c_fc.forward(input)?)?;
        self.c_proj.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.c_fc.training_mode(mode);
        self.c_proj.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Starcoder2Block {
    #[quantizable]
    #[param]
    self_attn: Starcoder2Attention,
    #[quantizable]
    #[param]
    mlp: Starcoder2Mlp,
    #[param]
    input_layernorm: nn::LayerNorm,
    #[param]
    post_attention_layernorm: nn::LayerNorm,
}

impl Starcoder2Block {
    fn new(args: &Starcoder2ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: Starcoder2Attention::new(args)?,
            mlp: Starcoder2Mlp::new(args.hidden_size, args.intermediate_size, args.use_bias)?,
            input_layernorm: nn::LayerNormBuilder::new(args.hidden_size)
                .eps(args.norm_epsilon)
                .build()?,
            post_attention_layernorm: nn::LayerNormBuilder::new(args.hidden_size)
                .eps(args.norm_epsilon)
                .build()?,
        })
    }
}

impl<C> Module<Starcoder2AttentionInput<'_, C>> for Starcoder2Block
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(
        &mut self,
        input: Starcoder2AttentionInput<'_, C>,
    ) -> Result<Self::Output, Self::Error> {
        let Starcoder2AttentionInput { x, mask, cache } = input;

        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(Starcoder2AttentionInput {
            x: &normed,
            mask,
            cache,
        })?;
        let h = x.add(attn_out)?;

        let normed_post = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed_post)?;
        h.add(mlp_out)
    }

    fn training_mode(&mut self, mode: bool) {
        <Starcoder2Attention as Module<Starcoder2AttentionInput<'_, C>>>::training_mode(
            &mut self.self_attn,
            mode,
        );
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Starcoder2Model {
    // Not marked #[quantizable]: starcoder2 safetensors stores embed_tokens as float16
    // (not uint32), so we keep it unquantized and let it load via direct key match.
    #[param]
    embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    layers: Vec<Starcoder2Block>,
    #[param]
    norm: nn::LayerNorm,

    sliding_window: Option<i32>,
}

struct Starcoder2ModelInput<'a, C> {
    inputs: &'a Array,
    mask: Option<&'a Array>,
    cache: &'a mut Vec<Option<C>>,
}

impl Starcoder2Model {
    fn new(args: &Starcoder2ModelArgs) -> Result<Self, Exception> {
        if !args.vocab_size.is_positive() {
            return Err(Exception::custom("vocab_size must be positive"));
        }
        if !args.num_hidden_layers.is_positive() {
            return Err(Exception::custom("num_hidden_layers must be positive"));
        }

        Ok(Self {
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(
                args.vocab_size,
                args.hidden_size,
            )?),
            layers: (0..args.num_hidden_layers)
                .map(|_| Starcoder2Block::new(args))
                .collect::<Result<Vec<_>, _>>()?,
            norm: nn::LayerNormBuilder::new(args.hidden_size)
                .eps(args.norm_epsilon)
                .build()?,
            sliding_window: args.sliding_window,
        })
    }
}

impl<C> Module<Starcoder2ModelInput<'_, C>> for Starcoder2Model
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: Starcoder2ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Starcoder2ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        let mut h = self.embed_tokens.forward(inputs)?;

        let shape = h.shape();
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Hidden state must have at least 2 dims"))?;

        let offset = cache
            .first()
            .and_then(|c| c.as_ref())
            .map_or(0, KeyValueCache::offset);
        let kv_len = offset + T;

        let computed_mask = match mask {
            Some(m) => Some(m.clone()),
            None => {
                if let Some(window) = self.sliding_window {
                    // Combined causal + sliding window mask for prefill,
                    // or window-only mask for decode when KV exceeds window.
                    if T > 1 || kv_len > window {
                        Some(create_sliding_causal_mask(T, kv_len, window)?)
                    } else {
                        None
                    }
                } else if T > 1 {
                    Some(create_causal_mask(T, Some(offset))?)
                } else {
                    None
                }
            }
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
            h = layer.forward(Starcoder2AttentionInput {
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
            <Starcoder2Block as Module<Starcoder2AttentionInput<'_, C>>>::training_mode(
                layer, mode,
            );
        }
        self.norm.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Causal LM
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Starcoder2CausalLM {
    pub args: Starcoder2ModelArgs,

    #[quantizable]
    #[param]
    model: Starcoder2Model,

    #[quantizable]
    #[param]
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Starcoder2CausalLM {
    pub fn new(args: Starcoder2ModelArgs) -> Result<Self, Exception> {
        let model = Starcoder2Model::new(&args)?;
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
        let out = self.forward_hidden(inputs, mask, kv_cache)?;

        match self.lm_head.as_mut() {
            Some(head) => head.forward(&out),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed) => embed.as_linear(&out),
                MaybeQuantized::Quantized(q_embed) => q_embed.as_linear(&out),
            },
        }
    }

    pub fn forward_hidden<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        self.model.forward(Starcoder2ModelInput {
            inputs,
            mask,
            cache: kv_cache,
        })
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

pub fn load_starcoder2_model_args<P: AsRef<Path>>(
    model_dir: P,
) -> Result<Starcoder2ModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

pub fn load_starcoder2_model<P: AsRef<Path>>(
    model_dir: P,
) -> Result<Starcoder2CausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_starcoder2_model_args(model_path)?;

    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        head_dim = args.head_dim(),
        vocab_size = args.vocab_size,
        sliding_window = ?args.sliding_window,
        use_bias = args.use_bias,
        "Loading Starcoder2 model"
    );

    let quantization = args.quantization.clone();
    let raw_model = Starcoder2CausalLM::new(args)?;

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

    tracing::info!("Starcoder2 model loaded successfully");
    Ok(model)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn default_starcoder2_args() -> Starcoder2ModelArgs {
        Starcoder2ModelArgs {
            model_type: "starcoder2".to_owned(),
            hidden_size: 256,
            num_hidden_layers: 2,
            intermediate_size: 512,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            norm_epsilon: 1e-5,
            vocab_size: 1000,
            max_position_embeddings: 512,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            use_bias: true,
            sliding_window: Some(128),
            rope_scaling: None,
            quantization: None,
        }
    }

    #[test]
    fn config_deserialization() {
        let json = r#"{
            "model_type": "starcoder2",
            "hidden_size": 3072,
            "num_hidden_layers": 30,
            "intermediate_size": 12288,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "norm_epsilon": 1e-5,
            "vocab_size": 49152,
            "max_position_embeddings": 16384,
            "rope_theta": 100000.0,
            "tie_word_embeddings": false,
            "use_bias": true,
            "sliding_window": 4096
        }"#;

        let args: Starcoder2ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "starcoder2");
        assert_eq!(args.hidden_size, 3072);
        assert_eq!(args.num_attention_heads, 24);
        assert_eq!(args.num_key_value_heads, 2);
        assert_eq!(args.sliding_window, Some(4096));
        assert!(args.use_bias);
    }

    #[test]
    fn head_dim_computation() {
        let args = default_starcoder2_args();
        assert_eq!(args.head_dim(), 64);
    }

    #[test]
    fn checked_head_dim_zero_heads() {
        let mut args = default_starcoder2_args();
        args.num_attention_heads = 0;
        assert!(args.checked_head_dim().is_err());
    }

    #[test]
    fn checked_head_dim_not_divisible() {
        let mut args = default_starcoder2_args();
        args.hidden_size = 100;
        args.num_attention_heads = 3;
        assert!(args.checked_head_dim().is_err());
    }

    #[test]
    fn model_construction() {
        let args = default_starcoder2_args();
        let model = Starcoder2CausalLM::new(args).unwrap();
        assert!(model.lm_head.is_some()); // not tied
    }

    #[test]
    fn model_construction_tied_embeddings() {
        let mut args = default_starcoder2_args();
        args.tie_word_embeddings = true;
        let model = Starcoder2CausalLM::new(args).unwrap();
        assert!(model.lm_head.is_none());
    }

    #[test]
    fn model_rejects_zero_vocab_size() {
        let mut args = default_starcoder2_args();
        args.vocab_size = 0;
        assert!(Starcoder2CausalLM::new(args).is_err());
    }

    #[test]
    fn model_rejects_zero_layers() {
        let mut args = default_starcoder2_args();
        args.num_hidden_layers = 0;
        assert!(Starcoder2CausalLM::new(args).is_err());
    }

    #[test]
    fn config_defaults_without_optional_fields() {
        let json = r#"{
            "model_type": "starcoder2",
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 512
        }"#;

        let args: Starcoder2ModelArgs = serde_json::from_str(json).unwrap();
        assert!(args.quantization.is_none());
        assert!(args.sliding_window.is_none());
        assert!(args.rope_scaling.is_none());
        assert!(!args.use_bias);
        assert!(!args.tie_word_embeddings);
        assert!((args.rope_theta - 10000.0).abs() < f32::EPSILON);
        assert!((args.norm_epsilon - 1e-5).abs() < f32::EPSILON);
    }

    #[test]
    fn sliding_causal_mask_shape() {
        let mask = create_sliding_causal_mask(4, 10, 3).unwrap();
        assert_eq!(mask.shape(), &[4, 10]);
    }

    #[test]
    fn sliding_causal_mask_values() {
        // query_len=3, kv_len=5, window=2
        // offset = 5 - 3 = 2
        // Query positions (absolute): 2, 3, 4
        // Key positions: 0, 1, 2, 3, 4
        // q=2: visible if 0 <= 2-k < 2 -> k in {1, 2}
        // q=3: visible if 0 <= 3-k < 2 -> k in {2, 3}
        // q=4: visible if 0 <= 4-k < 2 -> k in {3, 4}
        let mask = create_sliding_causal_mask(3, 5, 2).unwrap();
        mlx_rs::transforms::eval([&mask]).unwrap();
        let flat: Vec<bool> = mask.as_slice().to_vec();
        let expected = [
            false, true, true, false, false, false, false, true, true, false, false, false, false,
            true, true,
        ];
        assert_eq!(flat, expected);
    }

    #[test]
    fn sliding_causal_mask_large_window() {
        // When window >= kv_len, the mask is just a causal mask
        let mask = create_sliding_causal_mask(3, 3, 100).unwrap();
        mlx_rs::transforms::eval([&mask]).unwrap();
        let flat: Vec<bool> = mask.as_slice().to_vec();
        // Standard causal: lower-triangular
        let expected = [true, false, false, true, true, false, true, true, true];
        assert_eq!(flat, expected);
    }

    #[test]
    fn sliding_causal_mask_single_token_decode() {
        // T=1, kv_len=5, window=3: should see positions 2,3,4
        let mask = create_sliding_causal_mask(1, 5, 3).unwrap();
        mlx_rs::transforms::eval([&mask]).unwrap();
        let flat: Vec<bool> = mask.as_slice().to_vec();
        let expected = [false, false, true, true, true];
        assert_eq!(flat, expected);
    }
}
