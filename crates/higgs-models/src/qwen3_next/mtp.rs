//! Multi-token-prediction (MTP) heads.

use super::{
    attention::{DenseQwen3NextAttention, Qwen3NextAttention},
    config::Qwen3NextModelArgs,
    linear::{QLinear, dense_linear_no_bias_forward},
    mlp::{DenseQwen3NextMLP, Qwen3NextMLP},
    moe::SparseMoeBlock,
};
use mlx_rs::{
    Array, builder::Builder, error::Exception, macros::ModuleParameters, module::Param, nn,
};

/// Single MTP transformer layer (full attention + dense MLP).
#[derive(Debug, Clone, ModuleParameters)]
pub(super) struct MtpTransformerLayer {
    #[param]
    pub(super) self_attn: Qwen3NextAttention,
    #[param]
    pub(super) input_layernorm: nn::RmsNorm,
    #[param]
    pub(super) post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub(super) mlp: Qwen3NextMLP,
}

/// Single dense MTP transformer layer for sidecar checkpoints that store
/// full-precision projection weights without quantization metadata.
#[derive(Debug, Clone, ModuleParameters)]
pub(super) struct DenseMtpTransformerLayer {
    #[param]
    pub(super) self_attn: DenseQwen3NextAttention,
    #[param]
    pub(super) input_layernorm: nn::RmsNorm,
    #[param]
    pub(super) post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub(super) mlp: DenseQwen3NextMLP,
}

/// Multi-Token Prediction head.
///
/// Predicts the token at position t+2 given:
/// - The backbone's hidden state at position t (`h_t`)
/// - The embedding of the confirmed token at position t+1
///
/// Forward:
///   `fc(concat(norm_h(h_t), norm_e(embed(tok_{t+1})))) → transformer layer → shared lm_head`
#[derive(Debug, Clone, ModuleParameters)]
pub struct MtpHead {
    #[param]
    pub(super) pre_fc_norm_hidden: nn::RmsNorm,
    #[param]
    pub(super) pre_fc_norm_embedding: nn::RmsNorm,
    #[param]
    pub(super) fc: MtpFc,
    #[param]
    pub(super) layers: Vec<MtpTransformerLayer>,
    #[param]
    pub(super) norm: nn::RmsNorm,
}

#[derive(Debug, Clone, ModuleParameters)]
pub(super) struct DenseMtpHead {
    #[param]
    pub(super) pre_fc_norm_hidden: nn::RmsNorm,
    #[param]
    pub(super) pre_fc_norm_embedding: nn::RmsNorm,
    #[param]
    pub(super) fc: MtpFc,
    #[param]
    pub(super) layers: Vec<DenseMtpTransformerLayer>,
    #[param]
    pub(super) norm: nn::RmsNorm,
}

/// MTP fusion projection — kept in full precision (fp16) for accuracy.
///
/// mlx-lm's `quant_predicate` excludes `mtp.fc` from quantization because
/// quantizing the fusion layer destroys MTP prediction quality (0% acceptance).
#[derive(Debug, Clone, ModuleParameters)]
pub struct MtpFc {
    #[param]
    weight: Param<Array>,
}

impl MtpFc {
    fn new() -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::new(Array::zeros::<f32>(&[1, 1])?),
        })
    }

    pub(super) fn forward(&self, x: &Array) -> Result<Array, Exception> {
        dense_linear_no_bias_forward(&self.weight, x)
    }
}

impl MtpHead {
    pub(super) fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let n = usize::try_from(args.mtp_num_hidden_layers)
            .map_err(|_| Exception::custom("mtp_num_hidden_layers must be non-negative"))?;

        let layers = (0..n)
            .map(|_| {
                Ok(MtpTransformerLayer {
                    self_attn: Qwen3NextAttention::new(args, ql, qb)?,
                    input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    mlp: Qwen3NextMLP::new(ql, qb)?,
                })
            })
            .collect::<Result<Vec<_>, Exception>>()?;

        Ok(Self {
            pre_fc_norm_hidden: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            pre_fc_norm_embedding: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            fc: MtpFc::new()?,
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

impl DenseMtpHead {
    pub(super) fn new(args: &Qwen3NextModelArgs) -> Result<Self, Exception> {
        let n = usize::try_from(args.mtp_num_hidden_layers)
            .map_err(|_| Exception::custom("mtp_num_hidden_layers must be non-negative"))?;

        let layers = (0..n)
            .map(|_| {
                Ok(DenseMtpTransformerLayer {
                    self_attn: DenseQwen3NextAttention::new(args)?,
                    input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    mlp: DenseQwen3NextMLP::new(),
                })
            })
            .collect::<Result<Vec<_>, Exception>>()?;

        Ok(Self {
            pre_fc_norm_hidden: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            pre_fc_norm_embedding: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            fc: MtpFc::new()?,
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

/// Single `MoE` MTP transformer layer (Qwen3.6-A3B style).
///
/// Qwen3.6-A3B sidecars ship the MTP layer as a full `MoE` decoder layer:
/// full attention (with q/k norms) + `SparseMoeBlock`
/// (router gate + stacked experts + shared expert + shared-expert gate).
#[derive(Debug, Clone, ModuleParameters)]
pub(super) struct MoeMtpTransformerLayer {
    #[param]
    pub(super) self_attn: Qwen3NextAttention,
    #[param]
    pub(super) input_layernorm: nn::RmsNorm,
    #[param]
    pub(super) post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub(super) mlp: SparseMoeBlock,
}

/// MTP head with an `MoE` transformer layer (Qwen3.6-A3B style sidecars).
///
/// Unlike [`MtpHead`], the fusion projection `fc` is a quantized [`QLinear`]
/// (these sidecars ship `fc.{weight,scales,biases}` triples), and the MLP is
/// a [`SparseMoeBlock`]. All projections use the checkpoint's uniform
/// quantization — the main model's `gate_quantization` override must NOT be
/// applied here (the sidecar's router gate is quantized at the default width).
#[derive(Debug, Clone, ModuleParameters)]
pub struct MoeMtpHead {
    #[param]
    pub(super) pre_fc_norm_hidden: nn::RmsNorm,
    #[param]
    pub(super) pre_fc_norm_embedding: nn::RmsNorm,
    #[param]
    pub(super) fc: QLinear,
    #[param]
    pub(super) layers: Vec<MoeMtpTransformerLayer>,
    #[param]
    pub(super) norm: nn::RmsNorm,
}

impl MoeMtpHead {
    pub(super) fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let n = usize::try_from(args.mtp_num_hidden_layers)
            .map_err(|_| Exception::custom("mtp_num_hidden_layers must be non-negative"))?;

        // The sidecar's MoE block is uniformly quantized at the default width;
        // strip the main model's per-layer gate override so the router gate's
        // QLinear dequantizes with the right parameters.
        let mut mtp_args = args.clone();
        mtp_args.gate_quantization = None;

        let layers = (0..n)
            .map(|layer_idx| {
                Ok(MoeMtpTransformerLayer {
                    self_attn: Qwen3NextAttention::new(args, ql, qb)?,
                    input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                        .eps(args.rms_norm_eps)
                        .build()?,
                    mlp: SparseMoeBlock::new(
                        &mtp_args,
                        i32::try_from(layer_idx)
                            .map_err(|_| Exception::custom("MTP layer index exceeds i32"))?,
                        ql,
                        qb,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, Exception>>()?;

        Ok(Self {
            pre_fc_norm_hidden: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            pre_fc_norm_embedding: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            fc: QLinear::new(ql, qb)?,
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}
