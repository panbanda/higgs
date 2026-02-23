//! Qwen3-MoE model implementation.
//!
//! Standard qwen3-style attention (with QK norm) paired with sparse
//! Mixture-of-Experts on most layers. Differs from `qwen3_next` in having
//! no shared expert, no SSM layers, and per-layer MoE/dense selection.

use std::path::Path;

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops::{self, indexing::IndexOp},
};
use serde::Deserialize;

use crate::{
    cache::{KeyValueCache, SteppingKeyValueCache},
    error::ModelError,
    qwen3_next::{
        QEmbedding, QLinear, QuantizationConfig, SwitchMlpWeights, new_mlp_projections, swiglu,
    },
    utils::{AttentionMask, apply_rope, create_attention_mask, scaled_dot_product_attention},
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const fn default_rope_theta() -> f32 {
    10000.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3MoeModelArgs {
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
    pub head_dim: Option<i32>,

    // MoE params
    #[serde(default)]
    pub num_experts: i32,
    #[serde(default)]
    pub num_experts_per_tok: i32,
    #[serde(default)]
    pub moe_intermediate_size: i32,
    #[serde(default)]
    pub decoder_sparse_step: i32,
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
    #[serde(default)]
    pub norm_topk_prob: bool,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl Qwen3MoeModelArgs {
    fn head_dim(&self) -> i32 {
        self.head_dim.unwrap_or_else(|| {
            debug_assert!(
                self.num_attention_heads > 0,
                "num_attention_heads must be positive"
            );
            self.hidden_size / self.num_attention_heads
        })
    }

    fn is_moe_layer(&self, layer_idx: i32) -> bool {
        if self.decoder_sparse_step <= 0 {
            return false;
        }
        if self.mlp_only_layers.contains(&layer_idx) {
            return false;
        }
        (layer_idx + 1) % self.decoder_sparse_step == 0
    }
}

// ---------------------------------------------------------------------------
// Attention (QLinear-based, qwen3-style with QK norm)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3MoeAttention {
    #[param]
    q_proj: QLinear,
    #[param]
    k_proj: QLinear,
    #[param]
    v_proj: QLinear,
    #[param]
    o_proj: QLinear,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
    #[param]
    rope: nn::Rope,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    scale: f32,
}

impl Qwen3MoeAttention {
    fn new(args: &Qwen3MoeModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let head_dim = args.head_dim();
        let head_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        let scale = head_dim_f32.sqrt().recip();

        Ok(Self {
            q_proj: QLinear::new(ql, qb)?,
            k_proj: QLinear::new(ql, qb)?,
            v_proj: QLinear::new(ql, qb)?,
            o_proj: QLinear::new(ql, qb)?,
            q_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            k_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            rope: nn::RopeBuilder::new(head_dim)
                .traditional(false)
                .base(args.rope_theta)
                .scale(1.0)
                .build()
                .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?,
            num_attention_heads: args.num_attention_heads,
            num_key_value_heads: args.num_key_value_heads,
            scale,
        })
    }

    #[allow(non_snake_case)]
    fn forward<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut C>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        let mut queries = self
            .q_norm
            .forward(
                &self
                    .q_proj
                    .forward(x)?
                    .reshape(&[B, L, self.num_attention_heads, -1])?,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = self
            .k_norm
            .forward(
                &self
                    .k_proj
                    .forward(x)?
                    .reshape(&[B, L, self.num_key_value_heads, -1])?,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = self
            .v_proj
            .forward(x)?
            .reshape(&[B, L, self.num_key_value_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        if let Some(kv_cache) = cache {
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
}

// ---------------------------------------------------------------------------
// Unified MLP block (dense or sparse MoE, field named `mlp` to match safetensors)
//
// Python's Qwen3MoE uses a single `self.mlp` attribute for both dense and MoE
// layers, so safetensors keys are always prefixed with `mlp.`. MoE layers have
// `mlp.gate.*` (router) + `mlp.switch_mlp.*` (experts); dense layers have
// `mlp.gate_proj.*`, `mlp.down_proj.*`, `mlp.up_proj.*`.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3MoeMlpBlock {
    // MoE fields (None for dense layers)
    #[param]
    gate: Option<QLinear>,
    #[param]
    switch_mlp: Option<SwitchMlpWeights>,
    // Dense fields (None for MoE layers)
    #[param]
    gate_proj: Option<QLinear>,
    #[param]
    down_proj: Option<QLinear>,
    #[param]
    up_proj: Option<QLinear>,

    num_experts: i32,
    top_k: i32,
    norm_topk_prob: bool,
    is_moe: bool,
}

impl Qwen3MoeMlpBlock {
    fn new_moe(args: &Qwen3MoeModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        if args.num_experts <= 0 {
            return Err(Exception::custom("num_experts must be > 0"));
        }
        if args.num_experts_per_tok <= 0 {
            return Err(Exception::custom("num_experts_per_tok must be > 0"));
        }
        if args.num_experts_per_tok > args.num_experts {
            return Err(Exception::custom(
                "num_experts_per_tok must be <= num_experts",
            ));
        }
        Ok(Self {
            gate: Some(QLinear::new(ql, qb)?),
            switch_mlp: Some(SwitchMlpWeights::new(ql, qb)?),
            gate_proj: None,
            down_proj: None,
            up_proj: None,
            num_experts: args.num_experts,
            top_k: args.num_experts_per_tok,
            norm_topk_prob: args.norm_topk_prob,
            is_moe: true,
        })
    }

    fn new_dense(ql: i32, qb: i32) -> Result<Self, Exception> {
        let (gate_proj, down_proj, up_proj) = new_mlp_projections(ql, qb)?;
        Ok(Self {
            gate: None,
            switch_mlp: None,
            gate_proj: Some(gate_proj),
            down_proj: Some(down_proj),
            up_proj: Some(up_proj),
            num_experts: 0,
            top_k: 0,
            norm_topk_prob: false,
            is_moe: false,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        if self.is_moe {
            self.forward_moe(x)
        } else {
            self.forward_dense(x)
        }
    }

    fn forward_moe(&self, x: &Array) -> Result<Array, Exception> {
        let router = self
            .gate
            .as_ref()
            .ok_or_else(|| Exception::custom("MoE router gate missing"))?;
        let experts = self
            .switch_mlp
            .as_ref()
            .ok_or_else(|| Exception::custom("MoE switch_mlp missing"))?;

        let gates = ops::softmax_axis(&router.forward(x)?, -1, true)?;

        let neg_k = -self.top_k;
        let all_inds = ops::argpartition_axis(&gates, neg_k, -1)?;
        let top_k_start = self.num_experts - self.top_k;
        let top_inds = all_inds.index((.., .., top_k_start..));
        let raw_scores = gates.take_along_axis(&top_inds, -1)?;

        let top_scores = if self.norm_topk_prob {
            let score_sum = raw_scores.sum_axes(&[-1], true)?;
            raw_scores.divide(score_sum)?
        } else {
            raw_scores
        };

        let y = experts.forward_gather(x, &top_inds, false)?;

        y.multiply(&top_scores.expand_dims(-1)?)?
            .sum_axes(&[-2], false)
    }

    fn forward_dense(&self, x: &Array) -> Result<Array, Exception> {
        let gp = self
            .gate_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense gate_proj missing"))?;
        let dp = self
            .down_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense down_proj missing"))?;
        let up = self
            .up_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense up_proj missing"))?;

        let activated = swiglu(&gp.forward(x)?, &up.forward(x)?)?;
        dp.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3MoeDecoderLayer {
    #[param]
    self_attn: Qwen3MoeAttention,
    #[param]
    mlp: Qwen3MoeMlpBlock,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
}

impl Qwen3MoeDecoderLayer {
    fn new(args: &Qwen3MoeModelArgs, layer_idx: i32, ql: i32, qb: i32) -> Result<Self, Exception> {
        let mlp = if args.is_moe_layer(layer_idx) {
            Qwen3MoeMlpBlock::new_moe(args, ql, qb)?
        } else {
            Qwen3MoeMlpBlock::new_dense(ql, qb)?
        };

        Ok(Self {
            self_attn: Qwen3MoeAttention::new(args, ql, qb)?,
            mlp,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }

    fn forward<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: Option<&mut C>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        let h = x.add(attn_out)?;

        let normed_post = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed_post)?;
        h.add(mlp_out)
    }
}

// ---------------------------------------------------------------------------
// Inner model (embed + layers + norm)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3MoeInner {
    #[param]
    embed_tokens: QEmbedding,
    #[param]
    layers: Vec<Qwen3MoeDecoderLayer>,
    #[param]
    norm: nn::RmsNorm,
}

impl Qwen3MoeInner {
    fn new(args: &Qwen3MoeModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let layers = (0..args.num_hidden_layers)
            .map(|i| Qwen3MoeDecoderLayer::new(args, i, ql, qb))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            embed_tokens: QEmbedding::new(ql, qb)?,
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

// ---------------------------------------------------------------------------
// Qwen3MoeCausalLM (the public model type)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen3MoeCausalLM {
    pub args: Qwen3MoeModelArgs,
    #[param]
    model: Qwen3MoeInner,
    #[param]
    lm_head: Option<QLinear>,
}

impl Qwen3MoeCausalLM {
    pub fn new(args: Qwen3MoeModelArgs) -> Result<Self, Exception> {
        if !args.num_hidden_layers.is_positive() {
            return Err(Exception::custom("num_hidden_layers must be positive"));
        }
        if !args.vocab_size.is_positive() {
            return Err(Exception::custom("vocab_size must be positive"));
        }
        if !args.num_attention_heads.is_positive() {
            return Err(Exception::custom("num_attention_heads must be positive"));
        }
        if args.head_dim.is_none() && args.hidden_size % args.num_attention_heads != 0 {
            return Err(Exception::custom(
                "hidden_size must be divisible by num_attention_heads",
            ));
        }

        let ql = args.quantization.as_ref().map_or(64, |q| q.group_size);
        let qb = args.quantization.as_ref().map_or(4, |q| q.bits);

        let model = Qwen3MoeInner::new(&args, ql, qb)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(QLinear::new(ql, qb)?)
        };

        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

    /// Forward pass returning hidden states before the LM head.
    #[allow(non_snake_case)]
    pub fn forward_hidden(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<SteppingKeyValueCache>>,
    ) -> Result<Array, Exception> {
        let mut h = self.model.embed_tokens.forward(inputs)?;

        let computed_mask = match mask {
            Some(m) => Some(m.clone()),
            None => match create_attention_mask(&h, kv_cache, Some(true))? {
                Some(AttentionMask::Array(a)) => Some(a),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Only Array mask is supported"));
                }
                None => None,
            },
        };

        if kv_cache.is_empty() {
            *kv_cache = (0..self.model.layers.len())
                .map(|_| Some(SteppingKeyValueCache::new()))
                .collect();
        } else if kv_cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "kv_cache length ({}) must match num layers ({})",
                kv_cache.len(),
                self.model.layers.len()
            )));
        }

        for (layer, layer_cache) in self.model.layers.iter_mut().zip(kv_cache.iter_mut()) {
            h = layer.forward(&h, computed_mask.as_ref(), layer_cache.as_mut())?;
        }

        self.model.norm.forward(&h)
    }

    /// Forward pass producing logits.
    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<SteppingKeyValueCache>>,
    ) -> Result<Array, Exception> {
        let h = self.forward_hidden(inputs, mask, kv_cache)?;

        match self.lm_head.as_ref() {
            Some(head) => head.forward(&h),
            None => self.model.embed_tokens.as_linear(&h),
        }
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

pub fn load_model_args<P: AsRef<Path>>(model_dir: P) -> Result<Qwen3MoeModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

pub fn load_qwen3_moe_model<P: AsRef<Path>>(model_dir: P) -> Result<Qwen3MoeCausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_model_args(model_path)?;

    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        num_experts = args.num_experts,
        num_experts_per_tok = args.num_experts_per_tok,
        vocab_size = args.vocab_size,
        "Loading qwen3_moe model"
    );

    let mut model = Qwen3MoeCausalLM::new(args)?;

    crate::load_safetensors_weights(&mut model, model_path)?;

    tracing::info!("Qwen3MoE model loaded successfully");
    Ok(model)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn default_moe_args() -> Qwen3MoeModelArgs {
        Qwen3MoeModelArgs {
            model_type: "qwen3_moe".to_owned(),
            hidden_size: 4096,
            num_hidden_layers: 94,
            intermediate_size: 18944,
            num_attention_heads: 64,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-6,
            vocab_size: 151_936,
            max_position_embeddings: 40960,
            rope_theta: 1e6,
            tie_word_embeddings: false,
            attention_bias: false,
            rope_scaling: None,
            head_dim: None,
            num_experts: 128,
            num_experts_per_tok: 8,
            moe_intermediate_size: 1536,
            decoder_sparse_step: 1,
            mlp_only_layers: vec![],
            norm_topk_prob: true,
            quantization: Some(QuantizationConfig {
                group_size: 64,
                bits: 4,
            }),
        }
    }

    #[test]
    fn test_config_deserialization() {
        let json = r#"{
            "model_type": "qwen3_moe",
            "hidden_size": 4096,
            "num_hidden_layers": 94,
            "intermediate_size": 18944,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "head_dim": 128,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1536,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "norm_topk_prob": true,
            "quantization": {
                "group_size": 32,
                "bits": 4
            }
        }"#;

        let args: Qwen3MoeModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "qwen3_moe");
        assert_eq!(args.hidden_size, 4096);
        assert_eq!(args.num_hidden_layers, 94);
        assert_eq!(args.num_experts, 128);
        assert_eq!(args.num_experts_per_tok, 8);
        assert_eq!(args.moe_intermediate_size, 1536);
        assert_eq!(args.decoder_sparse_step, 1);
        assert!(args.mlp_only_layers.is_empty());
        assert!(args.norm_topk_prob);
        assert!(!args.tie_word_embeddings);
        assert!(!args.attention_bias);
        assert_eq!(args.head_dim(), 128);
        let q = args.quantization.as_ref().unwrap();
        assert_eq!(q.group_size, 32);
        assert_eq!(q.bits, 4);
    }

    #[test]
    fn test_head_dim_explicit_overrides_computed() {
        let mut args = default_moe_args();
        assert_eq!(args.head_dim(), 64); // 4096 / 64
        args.head_dim = Some(128);
        assert_eq!(args.head_dim(), 128);
    }

    #[test]
    fn test_config_defaults() {
        let json = r#"{
            "model_type": "qwen3_moe",
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "vocab_size": 1000,
            "max_position_embeddings": 512
        }"#;

        let args: Qwen3MoeModelArgs = serde_json::from_str(json).unwrap();
        assert!((args.rope_theta - 10000.0).abs() < f32::EPSILON);
        assert!(!args.tie_word_embeddings);
        assert!(!args.attention_bias);
        assert_eq!(args.num_experts, 0);
        assert_eq!(args.num_experts_per_tok, 0);
        assert_eq!(args.moe_intermediate_size, 0);
        assert_eq!(args.decoder_sparse_step, 0);
        assert!(args.mlp_only_layers.is_empty());
        assert!(!args.norm_topk_prob);
        assert!(args.quantization.is_none());
    }

    #[test]
    fn is_moe_layer_step_1_layer_0() {
        assert!(default_moe_args().is_moe_layer(0));
    }

    #[test]
    fn is_moe_layer_step_1_layer_5() {
        assert!(default_moe_args().is_moe_layer(5));
    }

    #[test]
    fn is_moe_layer_step_1_layer_93() {
        assert!(default_moe_args().is_moe_layer(93));
    }

    #[test]
    fn is_moe_layer_step_0_layer_0() {
        let mut args = default_moe_args();
        args.decoder_sparse_step = 0;
        assert!(!args.is_moe_layer(0));
    }

    #[test]
    fn is_moe_layer_step_0_layer_5() {
        let mut args = default_moe_args();
        args.decoder_sparse_step = 0;
        assert!(!args.is_moe_layer(5));
    }

    #[test]
    fn is_moe_layer_excluded_layer_0() {
        let mut args = default_moe_args();
        args.mlp_only_layers = vec![0, 5];
        assert!(!args.is_moe_layer(0));
    }

    #[test]
    fn is_moe_layer_excluded_layer_5() {
        let mut args = default_moe_args();
        args.mlp_only_layers = vec![0, 5];
        assert!(!args.is_moe_layer(5));
    }

    #[test]
    fn is_moe_layer_non_excluded_layer_1() {
        let mut args = default_moe_args();
        args.mlp_only_layers = vec![0, 5];
        assert!(args.is_moe_layer(1));
    }

    #[test]
    fn is_moe_layer_non_excluded_layer_6() {
        let mut args = default_moe_args();
        args.mlp_only_layers = vec![0, 5];
        assert!(args.is_moe_layer(6));
    }

    #[test]
    fn test_head_dim() {
        let args = default_moe_args();
        assert_eq!(args.head_dim(), 64);
    }

    #[test]
    fn test_model_new_zero_layers() {
        let mut args = default_moe_args();
        args.num_hidden_layers = 0;
        assert!(Qwen3MoeCausalLM::new(args).is_err());
    }

    #[test]
    fn test_model_new_zero_vocab() {
        let mut args = default_moe_args();
        args.vocab_size = 0;
        assert!(Qwen3MoeCausalLM::new(args).is_err());
    }

    #[test]
    fn test_load_model_args_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_model_args(dir.path()).is_err());
    }

    #[test]
    fn test_model_new_zero_attention_heads() {
        let mut args = default_moe_args();
        args.num_attention_heads = 0;
        assert!(Qwen3MoeCausalLM::new(args).is_err());
    }

    #[test]
    fn test_model_new_hidden_size_not_divisible() {
        let mut args = default_moe_args();
        args.hidden_size = 100;
        args.num_attention_heads = 3;
        args.head_dim = None;
        assert!(Qwen3MoeCausalLM::new(args).is_err());
    }

    #[test]
    fn test_model_new_explicit_head_dim_skips_divisibility_check() {
        let mut args = default_moe_args();
        args.hidden_size = 100;
        args.num_attention_heads = 3;
        args.head_dim = Some(128);
        // Should not fail: explicit head_dim bypasses hidden_size/num_heads check.
        // Construction still works because QLinear is a placeholder.
        assert!(Qwen3MoeCausalLM::new(args).is_ok());
    }

    fn small_moe_args() -> Qwen3MoeModelArgs {
        Qwen3MoeModelArgs {
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
    fn test_model_new_small_config() {
        let args = small_moe_args();
        let model = Qwen3MoeCausalLM::new(args).unwrap();
        assert_eq!(model.args.num_hidden_layers, 2);
        assert!(model.lm_head.is_none(), "tied embeddings => no lm_head");
    }

    #[test]
    fn test_model_new_untied_embeddings() {
        let mut args = small_moe_args();
        args.tie_word_embeddings = false;
        let model = Qwen3MoeCausalLM::new(args).unwrap();
        assert!(model.lm_head.is_some());
    }

    #[test]
    fn test_forward_preserves_pre_initialized_cache() {
        let args = small_moe_args();
        let mut model = Qwen3MoeCausalLM::new(args).unwrap();
        let mut cache: Vec<Option<SteppingKeyValueCache>> =
            (0..2).map(|_| Some(SteppingKeyValueCache::new())).collect();

        let input = Array::from_slice(&[1_i32, 2, 3], &[1, 3]);
        // Forward errors on unloaded quantized weights, but the
        // pre-initialized cache should not be cleared or resized.
        let _ = model.forward(&input, None, &mut cache);
        assert_eq!(cache.len(), 2, "cache should still have 2 entries");
        for (i, c) in cache.iter().enumerate() {
            assert!(c.is_some(), "layer {i} cache should be Some");
        }
    }

    #[test]
    fn test_forward_unloaded_model_returns_error() {
        let args = small_moe_args();
        let mut model = Qwen3MoeCausalLM::new(args).unwrap();
        let mut cache = vec![];

        let input = Array::from_slice(&[1_i32, 2, 3], &[1, 3]);
        // Unloaded QLinear weights are float32 placeholders;
        // quantized_matmul requires uint32, so forward should error
        // (not panic).
        let result = model.forward(&input, None, &mut cache);
        assert!(result.is_err());
    }

    #[test]
    fn test_forward_cache_length_mismatch() {
        let args = small_moe_args();
        let mut model = Qwen3MoeCausalLM::new(args).unwrap();
        // 3 entries but model has 2 layers
        let mut cache = vec![
            Some(SteppingKeyValueCache::new()),
            Some(SteppingKeyValueCache::new()),
            Some(SteppingKeyValueCache::new()),
        ];

        let input = Array::from_slice(&[1_i32], &[1, 1]);
        let result = model.forward(&input, None, &mut cache);
        assert!(result.is_err(), "mismatched cache length should error");
    }

    #[test]
    fn test_load_qwen3_moe_model_missing_dir() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_qwen3_moe_model(dir.path()).is_err());
    }

    #[test]
    fn test_load_qwen3_moe_model_invalid_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "not json").unwrap();
        assert!(load_qwen3_moe_model(dir.path()).is_err());
    }

    #[test]
    fn test_load_model_args_valid_small_config() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{
            "model_type": "qwen3_moe",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "vocab_size": 64,
            "max_position_embeddings": 128
        }"#;
        std::fs::write(dir.path().join("config.json"), json).unwrap();
        let args = load_model_args(dir.path()).unwrap();
        assert_eq!(args.model_type, "qwen3_moe");
        assert_eq!(args.hidden_size, 32);
        assert_eq!(args.num_hidden_layers, 2);
    }
}
