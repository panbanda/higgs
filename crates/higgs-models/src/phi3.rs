//! Phi-3 model implementation.
//!
//! Differs from the standard transformer in two ways:
//! - Combined QKV projection (single `qkv_proj` instead of separate `q_proj`/`k_proj`/`v_proj`)
//! - Combined gate-up MLP projection (single `gate_up_proj` instead of separate `gate_proj`/`up_proj`)

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

use crate::{
    cache::KeyValueCache,
    error::ModelError,
    utils::{AttentionMask, apply_rope, create_attention_mask, scaled_dot_product_attention},
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

/// Phi-3 model configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Phi3ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl Phi3ModelArgs {
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
// Attention (combined QKV projection)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Phi3Attention {
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,

    #[quantizable]
    #[param]
    qkv_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    rope: nn::Rope,
}

impl Phi3Attention {
    fn new(args: &Phi3ModelArgs) -> Result<Self, Exception> {
        let head_dim = args
            .checked_head_dim()
            .map_err(|e| Exception::custom(e.to_string()))?;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        let scale = head_dim_f32.sqrt().recip();

        // Combined QKV: output = (n_heads + 2 * n_kv_heads) * head_dim
        let qkv_out = (n_heads + 2 * n_kv_heads) * head_dim;
        let qkv_proj = nn::LinearBuilder::new(args.hidden_size, qkv_out)
            .bias(args.attention_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, args.hidden_size)
            .bias(false)
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
            qkv_proj: MaybeQuantized::Original(qkv_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            rope,
        })
    }
}

struct Phi3AttentionInput<'a, C> {
    x: &'a Array,
    mask: Option<&'a Array>,
    cache: Option<&'a mut C>,
}

impl<C> Module<Phi3AttentionInput<'_, C>> for Phi3Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: Phi3AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Phi3AttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // Combined QKV projection, then slice along last axis
        let qkv = self.qkv_proj.forward(x)?;

        let q_end = self.n_heads * self.head_dim;
        let k_end = q_end + self.n_kv_heads * self.head_dim;

        let mut queries = qkv
            .index((.., .., ..q_end))
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = qkv
            .index((.., .., q_end..k_end))
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = qkv
            .index((.., .., k_end..))
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
        self.qkv_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        <nn::Rope as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

// ---------------------------------------------------------------------------
// MLP (combined gate-up projection)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Phi3Mlp {
    intermediate_size: i32,

    #[quantizable]
    #[param]
    gate_up_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    down_proj: MaybeQuantized<nn::Linear>,
}

impl Phi3Mlp {
    fn new(dim: i32, intermediate_size: i32) -> Result<Self, Exception> {
        let gate_up_proj = nn::LinearBuilder::new(dim, 2 * intermediate_size)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(intermediate_size, dim)
            .bias(false)
            .build()?;

        Ok(Self {
            intermediate_size,
            gate_up_proj: MaybeQuantized::Original(gate_up_proj),
            down_proj: MaybeQuantized::Original(down_proj),
        })
    }
}

impl Module<&Array> for Phi3Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let up_states = self.gate_up_proj.forward(input)?;
        // Slice into gate and up halves along last axis
        let gate = up_states.index((.., .., ..self.intermediate_size));
        let up = up_states.index((.., .., self.intermediate_size..));
        let gated = nn::silu(gate)?.multiply(up)?;
        self.down_proj.forward(&gated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct Phi3Block {
    #[quantizable]
    #[param]
    self_attn: Phi3Attention,
    #[quantizable]
    #[param]
    mlp: Phi3Mlp,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
}

impl Phi3Block {
    fn new(args: &Phi3ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: Phi3Attention::new(args)?,
            mlp: Phi3Mlp::new(args.hidden_size, args.intermediate_size)?,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

impl<C> Module<Phi3AttentionInput<'_, C>> for Phi3Block
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: Phi3AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Phi3AttentionInput { x, mask, cache } = input;

        let normed = self.input_layernorm.forward(x)?;
        let residual = self.self_attn.forward(Phi3AttentionInput {
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
        <Phi3Attention as Module<Phi3AttentionInput<'_, C>>>::training_mode(
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
struct Phi3Model {
    #[quantizable]
    #[param]
    embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    layers: Vec<Phi3Block>,
    #[param]
    norm: nn::RmsNorm,
}

struct Phi3ModelInput<'a, C> {
    inputs: &'a Array,
    mask: Option<&'a Array>,
    cache: &'a mut Vec<Option<C>>,
}

impl Phi3Model {
    fn new(args: &Phi3ModelArgs) -> Result<Self, Exception> {
        if !args.vocab_size.is_positive() {
            return Err(Exception::custom("vocab_size must be positive"));
        }
        if !args.num_hidden_layers.is_positive() {
            return Err(Exception::custom("num_hidden_layers must be positive"));
        }
        if args.num_key_value_heads <= 0 || args.num_key_value_heads > args.num_attention_heads {
            return Err(Exception::custom(format!(
                "num_key_value_heads ({}) must be in [1, num_attention_heads ({})]",
                args.num_key_value_heads, args.num_attention_heads
            )));
        }

        Ok(Self {
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(
                args.vocab_size,
                args.hidden_size,
            )?),
            layers: (0..args.num_hidden_layers)
                .map(|_| Phi3Block::new(args))
                .collect::<Result<Vec<_>, _>>()?,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

impl<C> Module<Phi3ModelInput<'_, C>> for Phi3Model
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: Phi3ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let Phi3ModelInput {
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
            h = layer.forward(Phi3AttentionInput {
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
            <Phi3Block as Module<Phi3AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

// ---------------------------------------------------------------------------
// Causal LM
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Phi3CausalLM {
    pub args: Phi3ModelArgs,

    #[quantizable]
    #[param]
    model: Phi3Model,

    #[quantizable]
    #[param]
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Phi3CausalLM {
    pub fn new(args: Phi3ModelArgs) -> Result<Self, Exception> {
        if args.rope_scaling.is_some() {
            return Err(Exception::custom(
                "Phi-3 rope_scaling (SuRoPE/longrope) is not yet supported",
            ));
        }
        let model = Phi3Model::new(&args)?;
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
        self.model.forward(Phi3ModelInput {
            inputs,
            mask,
            cache: kv_cache,
        })
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

pub fn load_phi3_model_args<P: AsRef<Path>>(model_dir: P) -> Result<Phi3ModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

pub fn load_phi3_model<P: AsRef<Path>>(model_dir: P) -> Result<Phi3CausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_phi3_model_args(model_path)?;

    let quantization = args.quantization.clone();
    let raw_model = Phi3CausalLM::new(args)?;

    tracing::info!(
        model_type = %raw_model.args.model_type,
        hidden_size = raw_model.args.hidden_size,
        num_layers = raw_model.args.num_hidden_layers,
        num_heads = raw_model.args.num_attention_heads,
        num_kv_heads = raw_model.args.num_key_value_heads,
        head_dim = raw_model.args.head_dim(),
        vocab_size = raw_model.args.vocab_size,
        "Loading Phi-3 model"
    );

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

    tracing::info!("Phi-3 model loaded successfully");
    Ok(model)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn default_phi3_args() -> Phi3ModelArgs {
        Phi3ModelArgs {
            model_type: "phi3".to_owned(),
            hidden_size: 256,
            num_hidden_layers: 2,
            intermediate_size: 512,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-6,
            vocab_size: 1000,
            max_position_embeddings: 512,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            attention_bias: false,
            rope_scaling: None,
            quantization: None,
        }
    }

    #[test]
    fn config_deserialization() {
        let json = r#"{
            "model_type": "phi3",
            "hidden_size": 3072,
            "num_hidden_layers": 32,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "rms_norm_eps": 1e-5,
            "vocab_size": 32064,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "attention_bias": false
        }"#;

        let args: Phi3ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "phi3");
        assert_eq!(args.hidden_size, 3072);
        assert_eq!(args.head_dim(), 96);
        assert_eq!(args.num_attention_heads, 32);
    }

    #[test]
    fn head_dim_computation() {
        let args = default_phi3_args();
        assert_eq!(args.head_dim(), 64);
    }

    #[test]
    fn checked_head_dim_zero_heads() {
        let mut args = default_phi3_args();
        args.num_attention_heads = 0;
        assert!(args.checked_head_dim().is_err());
    }

    #[test]
    fn checked_head_dim_not_divisible() {
        let mut args = default_phi3_args();
        args.hidden_size = 100;
        args.num_attention_heads = 3;
        assert!(args.checked_head_dim().is_err());
    }

    #[test]
    fn model_construction() {
        let args = default_phi3_args();
        let model = Phi3CausalLM::new(args).unwrap();
        assert!(model.lm_head.is_some()); // not tied
    }

    #[test]
    fn model_construction_tied_embeddings() {
        let mut args = default_phi3_args();
        args.tie_word_embeddings = true;
        let model = Phi3CausalLM::new(args).unwrap();
        assert!(model.lm_head.is_none());
    }

    #[test]
    fn model_rejects_zero_vocab_size() {
        let mut args = default_phi3_args();
        args.vocab_size = 0;
        assert!(Phi3CausalLM::new(args).is_err());
    }

    #[test]
    fn model_rejects_zero_layers() {
        let mut args = default_phi3_args();
        args.num_hidden_layers = 0;
        assert!(Phi3CausalLM::new(args).is_err());
    }

    #[test]
    fn config_defaults_without_optional_fields() {
        let json = r#"{
            "model_type": "phi3",
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "vocab_size": 1000,
            "max_position_embeddings": 512
        }"#;

        let args: Phi3ModelArgs = serde_json::from_str(json).unwrap();
        assert!(args.quantization.is_none());
        assert!(args.rope_scaling.is_none());
        assert!(!args.attention_bias);
        assert!(!args.tie_word_embeddings);
        assert!((args.rope_theta - 10000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn config_with_rope_scaling() {
        let json = r#"{
            "model_type": "phi3",
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "vocab_size": 1000,
            "max_position_embeddings": 131072,
            "rope_scaling": {"type": "longrope", "long_factor": [1.0, 1.0]}
        }"#;

        let args: Phi3ModelArgs = serde_json::from_str(json).unwrap();
        assert!(args.rope_scaling.is_some());
    }
}
