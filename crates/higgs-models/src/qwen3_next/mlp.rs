//! Dense `SwiGLU` feed-forward layers and shared projection helpers.

use super::{
    activations::swiglu,
    config::Qwen3NextModelArgs,
    linear::{DenseLinearNoBias, QLinear},
};
use mlx_rs::{Array, error::Exception, macros::ModuleParameters};

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen3NextMLP {
    #[param]
    pub(super) gate_proj: QLinear,
    #[param]
    pub(super) down_proj: QLinear,
    #[param]
    pub(super) up_proj: QLinear,
}

pub fn new_mlp_projections(ql: i32, qb: i32) -> Result<(QLinear, QLinear, QLinear), Exception> {
    Ok((
        QLinear::new(ql, qb)?,
        QLinear::new(ql, qb)?,
        QLinear::new(ql, qb)?,
    ))
}

pub fn moe_expert_projection_quantization(
    args: &Qwen3NextModelArgs,
    layer_idx: i32,
    projection: &str,
    fallback_group_size: i32,
    fallback_bits: i32,
) -> Result<(i32, i32), Exception> {
    let Some(settings) = args.quantization.as_ref() else {
        return Ok((fallback_group_size, fallback_bits));
    };
    let paths = (0..args.num_experts)
        .map(|expert_idx| format!("model.layers.{layer_idx}.mlp.experts.{expert_idx}.{projection}"))
        .collect::<Vec<_>>();
    let quant = settings
        .resolve_uniform(paths.iter().map(String::as_str))
        .map_err(Exception::custom)?;
    if matches!(quant, crate::quant_config::TensorQuant::Unquantized) {
        return Err(Exception::custom(format!(
            "dense (unquantized) expert projections are not supported by the fused MoE path; tensor {}",
            paths.first().map_or(projection, String::as_str)
        )));
    }
    Ok((quant.group_size(), quant.bits()))
}

/// Every checkpoint tensor path this architecture actually resolves against
/// a per-tensor quantization override while constructing the model.
///
/// Attention projections, `GatedDeltaNet` linear-attention projections, and
/// dense-layer `mlp.{gate,up,down}_proj` are always built from the scalar
/// `group_size`/`bits` fallback — an override for one of those paths must be
/// rejected rather than silently dropped. The MTP sidecar (`mtp.*`,
/// `dense_mtp.*`, `moe_mtp.*`) is likewise scalar-only.
pub fn consumed_quantization_paths(args: &Qwen3NextModelArgs) -> std::collections::HashSet<String> {
    let mut consumed = std::collections::HashSet::new();
    consumed.insert("model.embed_tokens".to_owned());
    consumed.insert("lm_head".to_owned());

    if args.num_experts > 0 {
        // Every layer's `mlp` block resolves per-expert overrides via
        // `moe_expert_projection_quantization` (see `FfnBlock::new_moe`).
        let mut moe_layer_count = args.num_hidden_layers;
        // The MoE-structured MTP sidecar (Qwen3.6-A3B style) reuses the same
        // `model.layers.{idx}.mlp.experts.*` path prefix for its own
        // `SparseMoeBlock` (see `MoeMtpHead::new`), keyed by 0..mtp layer
        // count rather than the main model's layer indices — cover it too.
        if args.use_moe_mtp {
            moe_layer_count = moe_layer_count.max(args.mtp_num_hidden_layers);
        }
        for layer_idx in 0..moe_layer_count {
            for expert_idx in 0..args.num_experts {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    consumed.insert(format!(
                        "model.layers.{layer_idx}.mlp.experts.{expert_idx}.{projection}"
                    ));
                }
            }
            // `gate_quantization_override` (see `load_qwen3_next_args_from_value`)
            // reads only layer 0's `mlp.gate` override and applies it
            // uniformly as the scalar `gate_quantization` fallback for
            // every router gate and shared-expert gate — real
            // mlx-community checkpoints (e.g. Qwen3-Next 5-bit) still list
            // a per-layer override for both projections at every MoE
            // layer. Those per-layer entries are read (layer 0) or
            // deliberately superseded by the uniform fallback (other
            // layers), not silently dropped — mark them all consumed.
            for prefix in ["model", "language_model.model"] {
                consumed.insert(format!("{prefix}.layers.{layer_idx}.mlp.gate"));
                consumed.insert(format!(
                    "{prefix}.layers.{layer_idx}.mlp.shared_expert_gate"
                ));
            }
        }
    }

    consumed
}

impl Qwen3NextMLP {
    pub(super) fn new(ql: i32, qb: i32) -> Result<Self, Exception> {
        let (gate_proj, down_proj, up_proj) = new_mlp_projections(ql, qb)?;
        Ok(Self {
            gate_proj,
            down_proj,
            up_proj,
        })
    }

    pub(super) fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let gate_out = self.gate_proj.forward(x)?;
        let up_out = self.up_proj.forward(x)?;
        let activated = swiglu(&gate_out, &up_out)?;
        self.down_proj.forward(&activated)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(super) struct DenseQwen3NextMLP {
    #[param]
    gate_proj: DenseLinearNoBias,
    #[param]
    down_proj: DenseLinearNoBias,
    #[param]
    up_proj: DenseLinearNoBias,
}

impl DenseQwen3NextMLP {
    pub(super) fn new() -> Self {
        Self {
            gate_proj: DenseLinearNoBias::new(),
            down_proj: DenseLinearNoBias::new(),
            up_proj: DenseLinearNoBias::new(),
        }
    }

    pub(super) fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let gate_out = self.gate_proj.forward(x)?;
        let up_out = self.up_proj.forward(x)?;
        let activated = swiglu(&gate_out, &up_out)?;
        self.down_proj.forward(&activated)
    }
}
