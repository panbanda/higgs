//! Decoder layers and the feed-forward block that selects sparse or dense.

#[cfg(test)]
use super::cache::LayerCache;
use super::{
    activations::silu_mul,
    attention::Qwen3NextAttention,
    config::Qwen3NextModelArgs,
    flags::{
        DenseFfnGemvMode, apple_cpu_brand, dense_ffn_fuse_gate_up, dense_ffn_gemv_mode,
        moe_ffn_fuse_gate_up, should_force_dense_decode_safe_defaults_for_brand,
    },
    gated_delta_net::GatedDeltaNet,
    linear::{QLinear, quantized_forward},
    mlp::Qwen3NextMLP,
    moe::{MOE_SORT_MIN_LOOKUPS, SparseMoeBlock, SwitchMlpWeights},
    qgemv::qgemv_4bit,
};
#[cfg(test)]
use crate::utils::AttentionMask;
#[cfg(test)]
use mlx_rs::module::Module;
use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    nn,
    ops::{self, indexing::IndexOp},
};

/// Wrapper for the FFN block: either sparse `MoE` or dense `SwiGLU`.
/// Both share the `mlp` parameter namespace in safetensors — their sub-keys
/// don't overlap (`MoE`: gate, `switch_mlp`, `shared_expert`; Dense: `gate_proj`, `up_proj`, `down_proj`).
#[derive(Debug, Clone, ModuleParameters)]
pub(super) struct FfnBlock {
    #[param]
    gate: Option<QLinear>,
    #[param]
    switch_mlp: Option<SwitchMlpWeights>,
    #[param]
    shared_expert: Option<Qwen3NextMLP>,
    #[param]
    shared_expert_gate: Option<QLinear>,
    #[param]
    pub(super) gate_proj: Option<QLinear>,
    #[param]
    pub(super) up_proj: Option<QLinear>,
    #[param]
    pub(super) down_proj: Option<QLinear>,
    is_moe: bool,
    top_k: i32,
    norm_topk_prob: bool,
    /// Cached fused gate+up weights for dense layers (lazily computed on first forward).
    fused_gate_up: Option<(Array, Array, Array, i32)>,
}

impl FfnBlock {
    fn new_moe(
        args: &Qwen3NextModelArgs,
        layer_idx: i32,
        ql: i32,
        qb: i32,
    ) -> Result<Self, Exception> {
        let moe = SparseMoeBlock::new(args, layer_idx, ql, qb)?;
        Ok(Self {
            gate: Some(moe.gate),
            switch_mlp: Some(moe.switch_mlp),
            shared_expert: Some(moe.shared_expert),
            shared_expert_gate: Some(moe.shared_expert_gate),
            gate_proj: None,
            up_proj: None,
            down_proj: None,
            is_moe: true,
            top_k: moe.top_k,
            norm_topk_prob: moe.norm_topk_prob,
            fused_gate_up: None,
        })
    }

    pub(super) fn new_dense(ql: i32, qb: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate: None,
            switch_mlp: None,
            shared_expert: None,
            shared_expert_gate: None,
            gate_proj: Some(QLinear::new(ql, qb)?),
            up_proj: Some(QLinear::new(ql, qb)?),
            down_proj: Some(QLinear::new(ql, qb)?),
            is_moe: false,
            top_k: 0,
            norm_topk_prob: false,
            fused_gate_up: None,
        })
    }

    pub(super) fn dense_hidden_fused(
        &mut self,
        x: &Array,
        use_fused_gemv: bool,
    ) -> Result<Array, Exception> {
        if self.fused_gate_up.is_none() {
            let gp = self
                .gate_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("dense gate_proj missing"))?;
            let up = self
                .up_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("dense up_proj missing"))?;
            let intermediate = *gp
                .weight
                .shape()
                .first()
                .ok_or_else(|| Exception::custom("gate_proj weight has no dims"))?;
            let fw = ops::concatenate(&[&*gp.weight, &*up.weight], 0)?;
            let fs = ops::concatenate(&[&*gp.scales, &*up.scales], 0)?;
            let fb = ops::concatenate(&[&*gp.biases, &*up.biases], 0)?;
            fw.eval()?;
            fs.eval()?;
            fb.eval()?;
            self.fused_gate_up = Some((fw, fs, fb, intermediate));
        }

        let (fw, fs, fb, intermediate) = self
            .fused_gate_up
            .as_ref()
            .ok_or_else(|| Exception::custom("fused_gate_up missing after init"))?;
        let gp = self
            .gate_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense gate_proj missing"))?;

        let fused_out = if use_fused_gemv {
            qgemv_4bit(x, fw, fs, fb, gp.group_size)?
        } else {
            quantized_forward(x, fw, fs, fb, gp.group_size, gp.bits)?
        };
        let parts = fused_out.split_at_indices(&[*intermediate], Some(-1))?;
        let gate_out = parts
            .first()
            .ok_or_else(|| Exception::custom("fused split failed"))?;
        let up_out = parts
            .get(1)
            .ok_or_else(|| Exception::custom("fused split failed"))?;
        silu_mul(gate_out, up_out)
    }

    pub(super) fn dense_hidden_separate(&self, x: &Array) -> Result<Array, Exception> {
        let gp = self
            .gate_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense gate_proj missing"))?;
        let up = self
            .up_proj
            .as_ref()
            .ok_or_else(|| Exception::custom("dense up_proj missing"))?;
        let gate_out = gp.forward_decode_fast(x)?;
        let up_out = up.forward_decode_fast(x)?;
        silu_mul(&gate_out, &up_out)
    }

    /// Pick each token's top-k experts from the router logits `[B, L, E]`.
    ///
    /// Returns the expert indices `[B, L, top_k]`, their mixing scores, and
    /// whether the indices were sorted per token (callers then sort the
    /// lookups globally before `gather_qmm`).
    fn route_experts(&self, router_logits: &Array) -> Result<(Array, Array, bool), Exception> {
        // With renormalized top-k scores the result equals a softmax over
        // just the selected logits, so the softmax over every expert (and
        // the sum and divide after it) is skipped. Softmax is monotonic, so
        // selecting on logits picks the same experts.
        let selection_key = if self.norm_topk_prob {
            router_logits.clone()
        } else {
            ops::softmax_axis(router_logits, -1, true)?
        };

        let all_inds = ops::argpartition_axis(&selection_key, -self.top_k, -1)?;
        let num_experts = *selection_key
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("gates must have last dim"))?;
        let top_inds = all_inds.index((.., .., (num_experts - self.top_k)..));
        // Sorting lookups by expert makes weight access coalesced, which
        // pays off across many tokens. For a handful of lookups (one decode
        // token) the argsorts and gathers cost more than they save, so
        // those go to `gather_qmm` as selected.
        let sort_lookups = moe_ffn_fuse_gate_up() || top_inds.size() >= MOE_SORT_MIN_LOOKUPS;
        let inds = if sort_lookups {
            ops::sort_axis(&top_inds, -1)?
        } else {
            top_inds
        };
        let selected = selection_key.take_along_axis(&inds, -1)?;
        let scores = if self.norm_topk_prob {
            ops::softmax_axis(&selected, -1, true)?
        } else {
            selected
        };
        Ok((inds, scores, sort_lookups))
    }

    pub(super) fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        if self.is_moe {
            // Delegate to SparseMoeBlock logic
            let gate_ref = self
                .gate
                .as_ref()
                .ok_or_else(|| Exception::custom("MoE gate missing"))?;
            let seg_ref = self
                .shared_expert_gate
                .as_ref()
                .ok_or_else(|| Exception::custom("MoE shared_expert_gate missing"))?;

            let (inds, scores, sort_lookups) = self.route_experts(&gate_ref.forward(x)?)?;

            let switch_ref = self
                .switch_mlp
                .as_mut()
                .ok_or_else(|| Exception::custom("MoE switch_mlp missing"))?;
            let y = if moe_ffn_fuse_gate_up() {
                switch_ref.forward_gather_fused(x, &inds)?
            } else if sort_lookups {
                switch_ref.forward_gather_global_sort(x, &inds)?
            } else {
                switch_ref.forward_gather(x, &inds, false)?
            };

            let expert_sum = y
                .multiply(&scores.expand_dims(-1)?)?
                .sum_axes(&[-2], false)?;

            let se_ref = self
                .shared_expert
                .as_ref()
                .ok_or_else(|| Exception::custom("MoE shared_expert missing"))?;
            let shared_y = se_ref.forward(x)?;

            let shared_gate_val = nn::sigmoid(&seg_ref.forward(x)?)?;
            let shared_out = shared_y.multiply(&shared_gate_val)?;

            expert_sum.add(shared_out)
        } else {
            // Dense SwiGLU with configurable gate/up path so we can benchmark
            // whether one large fused matmul or two smaller matmuls are faster
            // on a given Apple GPU.
            let gp = self
                .gate_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("dense gate_proj missing"))?;

            let seq_len = *x.shape().get(1).unwrap_or(&0);
            // `qgemv_4bit` handles exactly one row.
            let use_decode_gemv = matches!(x.shape(), [1, 1, _]) && gp.bits == 4;
            let gemv_mode = if std::env::var_os("HIGGS_QGEMV_FFN_MODE").is_none()
                && should_force_dense_decode_safe_defaults_for_brand(apple_cpu_brand())
            {
                DenseFfnGemvMode::Off
            } else {
                dense_ffn_gemv_mode()
            };
            let use_fused_gemv = use_decode_gemv
                && matches!(
                    gemv_mode,
                    DenseFfnGemvMode::Both | DenseFfnGemvMode::FusedOnly
                );
            let use_down_gemv = use_decode_gemv
                && matches!(
                    gemv_mode,
                    DenseFfnGemvMode::Both | DenseFfnGemvMode::DownOnly
                );

            let hidden = if dense_ffn_fuse_gate_up() {
                self.dense_hidden_fused(x, use_fused_gemv)?
            } else {
                self.dense_hidden_separate(x)?
            };

            // Down projection
            let dp = self
                .down_proj
                .as_ref()
                .ok_or_else(|| Exception::custom("dense down_proj missing"))?;
            let out = if use_down_gemv {
                qgemv_4bit(&hidden, &dp.weight, &dp.scales, &dp.biases, dp.group_size)
            } else {
                dp.forward(&hidden)
            }?;
            if seq_len == 1 {
                mlx_rs::stop_gradient(&out)
            } else {
                Ok(out)
            }
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(super) struct DecoderLayer {
    #[param]
    pub(super) linear_attn: Option<GatedDeltaNet>,
    #[param]
    pub(super) self_attn: Option<Qwen3NextAttention>,
    #[param]
    pub(super) input_layernorm: nn::RmsNorm,
    #[param]
    pub(super) post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub(super) mlp: FfnBlock,
    pub(super) is_linear: bool,
}

impl DecoderLayer {
    pub(super) fn new(
        args: &Qwen3NextModelArgs,
        layer_idx: i32,
        ql: i32,
        qb: i32,
    ) -> Result<Self, Exception> {
        let is_linear = (layer_idx + 1) % args.full_attention_interval != 0;

        let linear_attn = is_linear
            .then(|| GatedDeltaNet::new(args, ql, qb))
            .transpose()?;
        let self_attn = (!is_linear)
            .then(|| Qwen3NextAttention::new(args, ql, qb))
            .transpose()?;

        let ffn = if args.num_experts > 0 {
            FfnBlock::new_moe(args, layer_idx, ql, qb)?
        } else {
            FfnBlock::new_dense(ql, qb)?
        };
        Ok(Self {
            linear_attn,
            self_attn,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            mlp: ffn,
            is_linear,
        })
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: &mut LayerCache,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let r = if self.is_linear {
            let attn = self
                .linear_attn
                .as_mut()
                .ok_or_else(|| Exception::custom("linear_attn missing on linear layer"))?;
            let LayerCache::Arrays(ssm_cache) = cache else {
                return Err(Exception::custom("Expected ArraysCache for linear layer"));
            };
            attn.forward(&normed, mask, ssm_cache)?
        } else {
            let attn = self
                .self_attn
                .as_mut()
                .ok_or_else(|| Exception::custom("self_attn missing on attention layer"))?;
            let LayerCache::KV(kv_cache) = cache else {
                return Err(Exception::custom("Expected KVCache for attention layer"));
            };
            attn.forward(&normed, mask, kv_cache)?
        };

        let h = x.add(r)?;
        let normed_post = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed_post)?;
        h.add(mlp_out)
    }
}
