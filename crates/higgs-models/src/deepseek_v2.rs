//! DeepSeek-V2 model implementation.
//!
//! Multi-head Latent Attention (MLA) compresses KV into a low-rank latent,
//! paired with sparse Mixture-of-Experts (shared + routed experts).
//! Supports both DeepSeek-V2-Lite (no query compression, no `YaRN`) and
//! full DeepSeek-V2 (query compression, `YaRN` `RoPE`, group-limited routing).

use std::f32::consts::PI;
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
    utils::{AttentionMask, create_attention_mask, scaled_dot_product_attention},
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const fn default_rope_theta() -> f32 {
    10000.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV2ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    #[serde(default)]
    pub num_key_value_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    #[serde(default)]
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    // MLA params
    pub kv_lora_rank: i32,
    #[serde(default)]
    pub q_lora_rank: Option<i32>,
    pub qk_rope_head_dim: i32,
    pub v_head_dim: i32,
    pub qk_nope_head_dim: i32,

    // MoE params
    #[serde(default)]
    pub n_routed_experts: Option<i32>,
    #[serde(default)]
    pub n_shared_experts: Option<i32>,
    #[serde(default)]
    pub num_experts_per_tok: Option<i32>,
    #[serde(default)]
    pub moe_intermediate_size: Option<i32>,
    #[serde(default)]
    pub routed_scaling_factor: Option<f32>,
    #[serde(default)]
    pub topk_method: Option<String>,
    #[serde(default)]
    pub n_group: Option<i32>,
    #[serde(default)]
    pub topk_group: Option<i32>,
    #[serde(default)]
    pub first_k_dense_replace: i32,
    #[serde(default)]
    pub moe_layer_freq: Option<i32>,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl DeepSeekV2ModelArgs {
    const fn q_head_dim(&self) -> i32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    fn is_moe_layer(&self, layer_idx: i32) -> bool {
        let Some(n_routed) = self.n_routed_experts else {
            return false;
        };
        if n_routed <= 0 {
            return false;
        }
        if layer_idx < self.first_k_dense_replace {
            return false;
        }
        let freq = self.moe_layer_freq.unwrap_or(1);
        if freq <= 0 {
            return false;
        }
        layer_idx % freq == 0
    }
}

// ---------------------------------------------------------------------------
// YaRN RoPE helpers
// ---------------------------------------------------------------------------

fn yarn_find_correction_dim(num_rotations: f32, dim: i32, base: f32, max_pos: i32) -> f32 {
    let dim_f = f32::from(i16::try_from(dim).unwrap_or(i16::MAX));
    let max_pos_f = f32::from(i16::try_from(max_pos).unwrap_or(i16::MAX));
    (dim_f * (max_pos_f / (num_rotations * 2.0 * PI)).ln()) / (2.0 * base.ln())
}

#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn yarn_find_correction_range(
    low_rot: f32,
    high_rot: f32,
    dim: i32,
    base: f32,
    max_pos: i32,
) -> (i32, i32) {
    let low = yarn_find_correction_dim(low_rot, dim, base, max_pos).floor() as i32;
    let high = yarn_find_correction_dim(high_rot, dim, base, max_pos).ceil() as i32;
    (low.max(0), high.min(dim - 1))
}

fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        (0.1 * mscale).mul_add(scale.ln(), 1.0)
    }
}

/// Precompute `YaRN`-interpolated `RoPE` frequencies.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing
)]
fn compute_yarn_freqs(
    dim: i32,
    base: f32,
    scaling_factor: f32,
    orig_max_pos: i32,
    beta_fast: f32,
    beta_slow: f32,
) -> Array {
    let half_dim = dim / 2;
    let dim_f = f32::from(i16::try_from(dim).unwrap_or(i16::MAX));

    // freq_extra = base^(arange(0, dim, 2) / dim) -- standard theta
    // freq_inter = scaling_factor * freq_extra -- extended theta
    let mut freq_extra = Vec::with_capacity(half_dim as usize);
    let mut freq_inter = Vec::with_capacity(half_dim as usize);
    for i in 0..half_dim {
        let exp = f32::from(i16::try_from(2 * i).unwrap_or(0)) / dim_f;
        let theta = base.powf(exp);
        freq_extra.push(theta);
        freq_inter.push(scaling_factor * theta);
    }

    let (low, high) = yarn_find_correction_range(beta_fast, beta_slow, dim, base, orig_max_pos);

    // Linear ramp mask: 0 at low, 1 at high
    let low_f = f32::from(i16::try_from(low).unwrap_or(0));
    let high_f = f32::from(i16::try_from(high).unwrap_or(0));
    let range = if (high_f - low_f).abs() < 0.001 {
        high_f - low_f + 0.001
    } else {
        high_f - low_f
    };

    // Compute interpolated freqs: blend between freq_inter and freq_extra
    // freq_mask = 1 - ramp (high mask = use freq_extra, low mask = use freq_inter)
    let mut freqs = Vec::with_capacity(half_dim as usize);
    for i in 0..half_dim as usize {
        let idx_f = f32::from(i16::try_from(i).unwrap_or(0));
        let ramp = ((idx_f - low_f) / range).clamp(0.0, 1.0);
        let mask = 1.0 - ramp;
        let inter = freq_inter[i];
        let extra = freq_extra[i];
        let denom = inter * mask + extra * (1.0 - mask);
        freqs.push((inter * extra) / denom);
    }

    Array::from_slice(&freqs, &[half_dim])
}

fn apply_deepseek_rope(
    x: &Array,
    dim: i32,
    base: f32,
    yarn_freqs: Option<&Array>,
    yarn_mscale: f32,
    offset: i32,
) -> Result<Array, Exception> {
    let x_scaled = if (yarn_mscale - 1.0).abs() > f32::EPSILON {
        x.multiply(mlx_rs::array!(yarn_mscale))?
    } else {
        x.clone()
    };
    yarn_freqs.map_or_else(
        || mlx_rs::fast::rope(&x_scaled, dim, true, base, 1.0, offset, None::<&Array>),
        |freqs| mlx_rs::fast::rope(&x_scaled, dim, true, None::<f32>, 1.0, offset, Some(freqs)),
    )
}

// ---------------------------------------------------------------------------
// MLA Attention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct DeepSeekV2Attention {
    // Compressed query path (q_lora_rank is Some)
    #[param]
    q_a_proj: Option<QLinear>,
    #[param]
    q_a_layernorm: Option<nn::RmsNorm>,
    #[param]
    q_b_proj: Option<QLinear>,
    // Direct query path (q_lora_rank is None)
    #[param]
    q_proj: Option<QLinear>,

    // KV compression
    #[param]
    kv_a_proj_with_mqa: QLinear,
    #[param]
    kv_a_layernorm: nn::RmsNorm,
    #[param]
    kv_b_proj: QLinear,
    #[param]
    o_proj: QLinear,

    // Non-parameter fields
    num_heads: i32,
    kv_lora_rank: i32,
    qk_nope_head_dim: i32,
    qk_rope_head_dim: i32,
    _v_head_dim: i32,
    scale: f32,
    rope_base: f32,
    yarn_freqs: Option<Array>,
    yarn_mscale: f32,
    use_compressed_query: bool,
}

impl DeepSeekV2Attention {
    fn new(args: &DeepSeekV2ModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let q_head_dim = args.q_head_dim();
        let q_head_dim_f = f32::from(
            i16::try_from(q_head_dim)
                .map_err(|_| Exception::custom("q_head_dim out of i16 range"))?,
        );
        let mut scale = q_head_dim_f.sqrt().recip();

        let use_compressed_query = args.q_lora_rank.is_some();

        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) =
            if let Some(_q_lora_rank) = args.q_lora_rank {
                (
                    Some(QLinear::new(ql, qb)?),
                    Some(
                        nn::RmsNormBuilder::new(args.q_lora_rank.unwrap_or(0))
                            .eps(1e-6)
                            .build()?,
                    ),
                    Some(QLinear::new(ql, qb)?),
                    None,
                )
            } else {
                (None, None, None, Some(QLinear::new(ql, qb)?))
            };

        // YaRN RoPE
        #[allow(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            clippy::option_if_let_else
        )]
        let (yarn_freqs, yarn_mscale) = if let Some(ref scaling) = args.rope_scaling {
            let factor = scaling
                .get("factor")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1.0) as f32;
            let orig_max_pos = scaling
                .get("original_max_position_embeddings")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(4096) as i32;
            let beta_fast = scaling
                .get("beta_fast")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(32.0) as f32;
            let beta_slow = scaling
                .get("beta_slow")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1.0) as f32;
            let mscale_param = scaling
                .get("mscale")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1.0) as f32;
            let mscale_all_dim = scaling
                .get("mscale_all_dim")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0) as f32;

            // Attention scale adjustment
            if mscale_all_dim > 0.0 {
                let ms = yarn_get_mscale(factor, mscale_all_dim);
                scale *= ms * ms;
            }

            // RoPE mscale: applied to x before rotation
            let rope_mscale =
                yarn_get_mscale(factor, mscale_param) / yarn_get_mscale(factor, mscale_all_dim);

            let freqs = compute_yarn_freqs(
                args.qk_rope_head_dim,
                args.rope_theta,
                factor,
                orig_max_pos,
                beta_fast,
                beta_slow,
            );

            (Some(freqs), rope_mscale)
        } else {
            (None, 1.0)
        };

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa: QLinear::new(ql, qb)?,
            kv_a_layernorm: nn::RmsNormBuilder::new(args.kv_lora_rank)
                .eps(1e-6)
                .build()?,
            kv_b_proj: QLinear::new(ql, qb)?,
            o_proj: QLinear::new(ql, qb)?,
            num_heads: args.num_attention_heads,
            kv_lora_rank: args.kv_lora_rank,
            qk_nope_head_dim: args.qk_nope_head_dim,
            qk_rope_head_dim: args.qk_rope_head_dim,
            _v_head_dim: args.v_head_dim,
            scale,
            rope_base: args.rope_theta,
            yarn_freqs,
            yarn_mscale,
            use_compressed_query,
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

        // Query projection
        let q_projected = if self.use_compressed_query {
            let qa = self
                .q_a_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("q_a_proj missing"))?;
            let qa_ln = self
                .q_a_layernorm
                .as_mut()
                .ok_or_else(|| Exception::custom("q_a_layernorm missing"))?;
            let qb = self
                .q_b_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("q_b_proj missing"))?;
            qb.forward(&qa_ln.forward(&qa.forward(x)?)?)?
        } else {
            let qp = self
                .q_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("q_proj missing"))?;
            qp.forward(x)?
        };

        let q = q_projected
            .reshape(&[
                B,
                L,
                self.num_heads,
                self.qk_nope_head_dim + self.qk_rope_head_dim,
            ])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let q_nope = q.index((.., .., .., ..self.qk_nope_head_dim));
        let q_pe_raw = q.index((.., .., .., self.qk_nope_head_dim..));

        // KV compression
        let compressed_kv = self.kv_a_proj_with_mqa.forward(x)?;
        let kv_latent = compressed_kv.index((.., .., ..self.kv_lora_rank));
        let k_pe_raw = compressed_kv
            .index((.., .., self.kv_lora_rank..))
            .reshape(&[B, L, 1, self.qk_rope_head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Decompress KV
        let kv = self
            .kv_b_proj
            .forward(&self.kv_a_layernorm.forward(&kv_latent)?)?
            .reshape(&[B, L, self.num_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k_nope = kv.index((.., .., .., ..self.qk_nope_head_dim));
        let v_decompressed = kv.index((.., .., .., self.qk_nope_head_dim..));

        // Apply RoPE
        let offset = cache.as_ref().map_or(0, |c| KeyValueCache::offset(*c));

        let q_pe = apply_deepseek_rope(
            &q_pe_raw,
            self.qk_rope_head_dim,
            self.rope_base,
            self.yarn_freqs.as_ref(),
            self.yarn_mscale,
            offset,
        )?;
        let k_pe = apply_deepseek_rope(
            &k_pe_raw,
            self.qk_rope_head_dim,
            self.rope_base,
            self.yarn_freqs.as_ref(),
            self.yarn_mscale,
            offset,
        )?;

        // Repeat k_pe for all heads: [B, 1, L, D] -> [B, num_heads, L, D]
        let k_pe_expanded = {
            let refs: Vec<Array> = (0..self.num_heads).map(|_| k_pe.clone()).collect();
            let ref_refs: Vec<&Array> = refs.iter().collect();
            ops::concatenate_axis(&ref_refs, 1)?
        };

        // Combine nope + pe components
        let keys_combined = ops::concatenate_axis(&[&k_nope, &k_pe_expanded], -1)?;
        let queries = ops::concatenate_axis(&[&q_nope, &q_pe], -1)?;

        // Update cache
        let (keys, values) = if let Some(kv_cache) = cache {
            kv_cache.update_and_fetch(keys_combined, v_decompressed)?
        } else {
            (keys_combined, v_decompressed)
        };

        let output = scaled_dot_product_attention(queries, keys, values, self.scale, mask)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }
}

// ---------------------------------------------------------------------------
// Shared experts MLP (always-on experts combined into a single MLP)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct SharedExperts {
    #[param]
    gate_proj: QLinear,
    #[param]
    down_proj: QLinear,
    #[param]
    up_proj: QLinear,
}

impl SharedExperts {
    fn new(ql: i32, qb: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: QLinear::new(ql, qb)?,
            down_proj: QLinear::new(ql, qb)?,
            up_proj: QLinear::new(ql, qb)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let activated = swiglu(&self.gate_proj.forward(x)?, &self.up_proj.forward(x)?)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// MLP block (dense or sparse MoE)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct DeepSeekV2MlpBlock {
    // MoE fields
    // The router gate is a plain float16 linear (never quantized in mlx-community checkpoints).
    #[param]
    gate: Option<nn::Linear>,
    #[param]
    switch_mlp: Option<SwitchMlpWeights>,
    #[param]
    shared_experts: Option<SharedExperts>,
    // Dense fields
    #[param]
    gate_proj: Option<QLinear>,
    #[param]
    down_proj: Option<QLinear>,
    #[param]
    up_proj: Option<QLinear>,

    num_experts: i32,
    top_k: i32,
    scaling_factor: f32,
    is_moe: bool,
}

impl DeepSeekV2MlpBlock {
    fn new_moe(args: &DeepSeekV2ModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let n_routed = args
            .n_routed_experts
            .ok_or_else(|| Exception::custom("n_routed_experts required for MoE layer"))?;
        let top_k = args.num_experts_per_tok.unwrap_or(6);

        let shared = if args.n_shared_experts.is_some() {
            Some(SharedExperts::new(ql, qb)?)
        } else {
            None
        };

        Ok(Self {
            gate: Some(
                nn::LinearBuilder::new(args.hidden_size, n_routed)
                    .bias(false)
                    .build()?,
            ),
            switch_mlp: Some(SwitchMlpWeights::new(ql, qb)?),
            shared_experts: shared,
            gate_proj: None,
            down_proj: None,
            up_proj: None,
            num_experts: n_routed,
            top_k,
            scaling_factor: args.routed_scaling_factor.unwrap_or(1.0),
            is_moe: true,
        })
    }

    fn new_dense(ql: i32, qb: i32) -> Result<Self, Exception> {
        let (gate_proj, down_proj, up_proj) = new_mlp_projections(ql, qb)?;
        Ok(Self {
            gate: None,
            switch_mlp: None,
            shared_experts: None,
            gate_proj: Some(gate_proj),
            down_proj: Some(down_proj),
            up_proj: Some(up_proj),
            num_experts: 0,
            top_k: 0,
            scaling_factor: 1.0,
            is_moe: false,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        if self.is_moe {
            self.forward_moe(x)
        } else {
            self.forward_dense(x)
        }
    }

    fn forward_moe(&mut self, x: &Array) -> Result<Array, Exception> {
        // Borrow each sub-module in its own scope to avoid simultaneous mutable borrows.

        // Softmax routing (V2 style)
        let gates_raw = self
            .gate
            .as_mut()
            .ok_or_else(|| Exception::custom("MoE router gate missing"))?
            .forward(x)?;
        let gates = ops::softmax_axis(&gates_raw, -1, true)?;

        // Top-k selection
        let neg_k = -self.top_k;
        let all_inds = ops::argpartition_axis(&gates, neg_k, -1)?;
        let top_k_start = self.num_experts - self.top_k;
        let top_inds = all_inds.index((.., .., top_k_start..));
        let top_scores = gates.take_along_axis(&top_inds, -1)?;

        // Scale scores
        let scaled_scores = if (self.scaling_factor - 1.0).abs() > f32::EPSILON {
            top_scores.multiply(mlx_rs::array!(self.scaling_factor))?
        } else {
            top_scores
        };

        // Expert computation
        let y = self
            .switch_mlp
            .as_ref()
            .ok_or_else(|| Exception::custom("MoE switch_mlp missing"))?
            .forward_gather(x, &top_inds, false)?;
        let mut result = y
            .multiply(&scaled_scores.expand_dims(-1)?)?
            .sum_axes(&[-2], false)?;

        // Add shared experts
        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(x)?;
            result = result.add(shared_out)?;
        }

        Ok(result)
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
struct DeepSeekV2DecoderLayer {
    #[param]
    self_attn: DeepSeekV2Attention,
    #[param]
    mlp: DeepSeekV2MlpBlock,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
}

impl DeepSeekV2DecoderLayer {
    fn new(
        args: &DeepSeekV2ModelArgs,
        layer_idx: i32,
        ql: i32,
        qb: i32,
    ) -> Result<Self, Exception> {
        let mlp = if args.is_moe_layer(layer_idx) {
            DeepSeekV2MlpBlock::new_moe(args, ql, qb)?
        } else {
            DeepSeekV2MlpBlock::new_dense(ql, qb)?
        };

        Ok(Self {
            self_attn: DeepSeekV2Attention::new(args, ql, qb)?,
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
struct DeepSeekV2Inner {
    #[param]
    embed_tokens: QEmbedding,
    #[param]
    layers: Vec<DeepSeekV2DecoderLayer>,
    #[param]
    norm: nn::RmsNorm,
}

impl DeepSeekV2Inner {
    fn new(args: &DeepSeekV2ModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let layers = (0..args.num_hidden_layers)
            .map(|i| DeepSeekV2DecoderLayer::new(args, i, ql, qb))
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
// DeepSeekV2CausalLM (the public model type)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct DeepSeekV2CausalLM {
    pub args: DeepSeekV2ModelArgs,
    #[param]
    model: DeepSeekV2Inner,
    #[param]
    lm_head: Option<QLinear>,
}

impl DeepSeekV2CausalLM {
    pub fn new(args: DeepSeekV2ModelArgs) -> Result<Self, Exception> {
        if !args.num_hidden_layers.is_positive() {
            return Err(Exception::custom("num_hidden_layers must be positive"));
        }
        if !args.vocab_size.is_positive() {
            return Err(Exception::custom("vocab_size must be positive"));
        }
        if !args.num_attention_heads.is_positive() {
            return Err(Exception::custom("num_attention_heads must be positive"));
        }

        let ql = args.quantization.as_ref().map_or(64, |q| q.group_size);
        let qb = args.quantization.as_ref().map_or(4, |q| q.bits);

        let model = DeepSeekV2Inner::new(&args, ql, qb)?;
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

pub fn load_model_args<P: AsRef<Path>>(model_dir: P) -> Result<DeepSeekV2ModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

pub fn load_deepseek_v2_model<P: AsRef<Path>>(
    model_dir: P,
) -> Result<DeepSeekV2CausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_model_args(model_path)?;

    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        kv_lora_rank = args.kv_lora_rank,
        q_lora_rank = ?args.q_lora_rank,
        n_routed_experts = ?args.n_routed_experts,
        n_shared_experts = ?args.n_shared_experts,
        vocab_size = args.vocab_size,
        "Loading deepseek_v2 model"
    );

    let mut model = DeepSeekV2CausalLM::new(args)?;

    crate::load_safetensors_weights(&mut model, model_path)?;

    tracing::info!("DeepSeek-V2 model loaded successfully");
    Ok(model)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn v2_lite_args() -> DeepSeekV2ModelArgs {
        DeepSeekV2ModelArgs {
            model_type: "deepseek_v2".to_owned(),
            hidden_size: 2048,
            num_hidden_layers: 27,
            intermediate_size: 10944,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            rms_norm_eps: 1e-6,
            vocab_size: 102_400,
            max_position_embeddings: 163_840,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            attention_bias: false,
            rope_scaling: None,
            kv_lora_rank: 512,
            q_lora_rank: None,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            qk_nope_head_dim: 128,
            n_routed_experts: Some(64),
            n_shared_experts: Some(2),
            num_experts_per_tok: Some(6),
            moe_intermediate_size: Some(1408),
            routed_scaling_factor: Some(1.0),
            topk_method: Some("greedy".to_owned()),
            n_group: Some(1),
            topk_group: Some(1),
            first_k_dense_replace: 1,
            moe_layer_freq: Some(1),
            quantization: Some(QuantizationConfig {
                group_size: 64,
                bits: 4,
            }),
        }
    }

    fn small_args() -> DeepSeekV2ModelArgs {
        DeepSeekV2ModelArgs {
            model_type: "deepseek_v2".to_owned(),
            hidden_size: 64,
            num_hidden_layers: 2,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-6,
            vocab_size: 128,
            max_position_embeddings: 256,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            attention_bias: false,
            rope_scaling: None,
            kv_lora_rank: 32,
            q_lora_rank: None,
            qk_rope_head_dim: 8,
            v_head_dim: 16,
            qk_nope_head_dim: 16,
            n_routed_experts: Some(4),
            n_shared_experts: Some(1),
            num_experts_per_tok: Some(2),
            moe_intermediate_size: Some(32),
            routed_scaling_factor: Some(1.0),
            topk_method: Some("greedy".to_owned()),
            n_group: Some(1),
            topk_group: Some(1),
            first_k_dense_replace: 1,
            moe_layer_freq: Some(1),
            quantization: None,
        }
    }

    #[test]
    fn test_config_deserialization_v2_lite() {
        let json = r#"{
            "model_type": "deepseek_v2",
            "hidden_size": 2048,
            "num_hidden_layers": 27,
            "intermediate_size": 10944,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "rms_norm_eps": 1e-06,
            "vocab_size": 102400,
            "max_position_embeddings": 163840,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "qk_nope_head_dim": 128,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 1408,
            "routed_scaling_factor": 1.0,
            "topk_method": "greedy",
            "n_group": 1,
            "topk_group": 1,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1
        }"#;

        let args: DeepSeekV2ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "deepseek_v2");
        assert_eq!(args.hidden_size, 2048);
        assert_eq!(args.kv_lora_rank, 512);
        assert!(args.q_lora_rank.is_none());
        assert_eq!(args.qk_nope_head_dim, 128);
        assert_eq!(args.qk_rope_head_dim, 64);
        assert_eq!(args.v_head_dim, 128);
        assert_eq!(args.q_head_dim(), 192);
        assert_eq!(args.n_routed_experts, Some(64));
        assert_eq!(args.n_shared_experts, Some(2));
        assert!(args.rope_scaling.is_none());
    }

    #[test]
    fn test_config_deserialization_v2_full() {
        let json = r#"{
            "model_type": "deepseek_v2",
            "hidden_size": 5120,
            "num_hidden_layers": 60,
            "intermediate_size": 12288,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "rms_norm_eps": 1e-06,
            "vocab_size": 102400,
            "max_position_embeddings": 163840,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "qk_nope_head_dim": 128,
            "n_routed_experts": 160,
            "n_shared_experts": 2,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 1536,
            "routed_scaling_factor": 16.0,
            "topk_method": "group_limited_greedy",
            "n_group": 8,
            "topk_group": 3,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "rope_scaling": {
                "type": "yarn",
                "factor": 40,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 0.707,
                "mscale_all_dim": 0.707
            }
        }"#;

        let args: DeepSeekV2ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.q_lora_rank, Some(1536));
        assert_eq!(args.n_routed_experts, Some(160));
        assert!(args.rope_scaling.is_some());
        assert!((args.routed_scaling_factor.unwrap() - 16.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_q_head_dim() {
        let args = v2_lite_args();
        assert_eq!(args.q_head_dim(), 192);
    }

    #[test]
    fn test_is_moe_layer_first_dense() {
        let args = v2_lite_args();
        assert!(
            !args.is_moe_layer(0),
            "layer 0 should be dense (first_k_dense_replace=1)"
        );
    }

    #[test]
    fn test_is_moe_layer_after_dense() {
        let args = v2_lite_args();
        assert!(args.is_moe_layer(1));
        assert!(args.is_moe_layer(10));
        assert!(args.is_moe_layer(26));
    }

    #[test]
    fn test_is_moe_layer_no_experts() {
        let mut args = v2_lite_args();
        args.n_routed_experts = None;
        assert!(!args.is_moe_layer(5));
    }

    #[test]
    fn test_model_new_small_config() {
        let args = small_args();
        let model = DeepSeekV2CausalLM::new(args).unwrap();
        assert_eq!(model.args.num_hidden_layers, 2);
        assert!(model.lm_head.is_none(), "tied embeddings => no lm_head");
    }

    #[test]
    fn test_model_new_untied_embeddings() {
        let mut args = small_args();
        args.tie_word_embeddings = false;
        let model = DeepSeekV2CausalLM::new(args).unwrap();
        assert!(model.lm_head.is_some());
    }

    #[test]
    fn test_model_new_zero_layers() {
        let mut args = small_args();
        args.num_hidden_layers = 0;
        assert!(DeepSeekV2CausalLM::new(args).is_err());
    }

    #[test]
    fn test_model_new_zero_vocab() {
        let mut args = small_args();
        args.vocab_size = 0;
        assert!(DeepSeekV2CausalLM::new(args).is_err());
    }

    #[test]
    fn test_model_new_zero_attention_heads() {
        let mut args = small_args();
        args.num_attention_heads = 0;
        assert!(DeepSeekV2CausalLM::new(args).is_err());
    }

    #[test]
    fn test_forward_preserves_pre_initialized_cache() {
        let args = small_args();
        let mut model = DeepSeekV2CausalLM::new(args).unwrap();
        let mut cache: Vec<Option<SteppingKeyValueCache>> =
            (0..2).map(|_| Some(SteppingKeyValueCache::new())).collect();

        let input = Array::from_slice(&[1_i32, 2, 3], &[1, 3]);
        let _ = model.forward(&input, None, &mut cache);
        assert_eq!(cache.len(), 2);
        for (i, c) in cache.iter().enumerate() {
            assert!(c.is_some(), "layer {i} cache should be Some");
        }
    }

    #[test]
    fn test_forward_cache_length_mismatch() {
        let args = small_args();
        let mut model = DeepSeekV2CausalLM::new(args).unwrap();
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
    fn test_load_model_args_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_model_args(dir.path()).is_err());
    }

    #[test]
    fn test_load_model_args_invalid_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "not json").unwrap();
        assert!(load_model_args(dir.path()).is_err());
    }

    #[test]
    fn test_yarn_get_mscale_no_scaling() {
        assert!((yarn_get_mscale(1.0, 0.707) - 1.0).abs() < f32::EPSILON);
        assert!((yarn_get_mscale(0.5, 0.707) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_yarn_get_mscale_with_scaling() {
        let result = yarn_get_mscale(40.0, 0.707);
        assert!(result > 1.0);
        // 0.1 * 0.707 * ln(40) + 1.0 = 0.1 * 0.707 * 3.689 + 1.0 ≈ 1.261
        assert!((result - 1.261).abs() < 0.01);
    }

    #[test]
    fn test_compute_yarn_freqs_shape() {
        let freqs = compute_yarn_freqs(64, 10000.0, 40.0, 4096, 32.0, 1.0);
        assert_eq!(freqs.shape(), &[32]); // half_dim = 64/2 = 32
    }

    #[test]
    fn test_model_new_with_compressed_query() {
        let mut args = small_args();
        args.q_lora_rank = Some(16);
        let model = DeepSeekV2CausalLM::new(args).unwrap();
        assert_eq!(model.args.q_lora_rank, Some(16));
    }

    #[test]
    fn test_model_new_dense_only() {
        let mut args = small_args();
        args.n_routed_experts = None;
        args.first_k_dense_replace = 0;
        let model = DeepSeekV2CausalLM::new(args).unwrap();
        assert!(model.args.n_routed_experts.is_none());
    }
}
