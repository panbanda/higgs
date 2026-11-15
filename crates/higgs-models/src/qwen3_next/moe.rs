//! Sparse mixture-of-experts: stacked expert weights, router, and shared expert.

use super::{
    activations::swiglu,
    config::Qwen3NextModelArgs,
    ffi::gather_qmm,
    linear::QLinear,
    mlp::{Qwen3NextMLP, moe_expert_projection_quantization},
};
use mlx_rs::{
    Array,
    error::Exception,
    macros::ModuleParameters,
    nn,
    ops::{self, indexing::IndexOp},
};

/// Expert lookups (`tokens * top_k`) at which sorting by expert starts to pay
/// for itself. Same threshold as mlx-lm's `SwitchGLU`.
pub(super) const MOE_SORT_MIN_LOOKUPS: usize = 64;

#[derive(Debug, Clone, ModuleParameters)]
pub struct SwitchMlpWeights {
    #[param]
    pub(super) gate_proj: QLinear,
    #[param]
    pub(super) up_proj: QLinear,
    #[param]
    pub(super) down_proj: QLinear,
    /// Lazily fused gate+up weights for `MoE` `gather_qmm` (3→2 calls per layer).
    pub(super) fused_gate_up: Option<(Array, Array, Array, i32)>,
}

impl SwitchMlpWeights {
    pub(crate) fn new(ql: i32, qb: i32) -> Result<Self, Exception> {
        Self::new_with_projection_quantization((ql, qb), (ql, qb), (ql, qb))
    }

    pub(crate) fn new_with_projection_quantization(
        gate: (i32, i32),
        up: (i32, i32),
        down: (i32, i32),
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: QLinear::new(gate.0, gate.1)?,
            up_proj: QLinear::new(up.0, up.1)?,
            down_proj: QLinear::new(down.0, down.1)?,
            fused_gate_up: None,
        })
    }

    /// Apply the full `SwiGLU` `MoE` block for all selected experts in one shot
    /// using `gather_qmm` (fused expert-indexed quantized matmul).
    ///
    /// `x`: `[..., D]` input
    /// `indices`: `[..., top_k]` expert indices
    /// Returns: `[..., top_k, D]`
    pub(crate) fn forward_gather(
        &self,
        x: &Array,
        indices: &Array,
        sorted: bool,
    ) -> Result<Array, Exception> {
        // Reshape so x batch dims broadcast with the indices shape.
        // x: [B, L, D] -> [B, L, 1, 1, D]
        //   batch = [B, L, 1], M=1, K=D
        // indices: [B, L, top_k]
        //   broadcast([B, L, 1], [B, L, top_k]) -> [B, L, top_k]
        let shape = x.shape();
        let err = || Exception::custom("forward_gather input must be [B, L, D]");
        let b = *shape.first().ok_or_else(err)?;
        let l = *shape.get(1).ok_or_else(err)?;
        let d = *shape.get(2).ok_or_else(err)?;
        let x_exp = x.reshape(&[b, l, 1, 1, d])?;

        // Gate/up projections: [B, L, top_k, 1, intermediate]
        let gate_out = gather_qmm(
            &x_exp,
            &self.gate_proj.weight,
            &self.gate_proj.scales,
            &self.gate_proj.biases,
            indices,
            true,
            self.gate_proj.group_size,
            self.gate_proj.bits,
            sorted,
        )?;
        let up_out = gather_qmm(
            &x_exp,
            &self.up_proj.weight,
            &self.up_proj.scales,
            &self.up_proj.biases,
            indices,
            true,
            self.up_proj.group_size,
            self.up_proj.bits,
            sorted,
        )?;

        let activated = swiglu(&gate_out, &up_out)?;

        // Down projection: [B, L, top_k, 1, D]
        // activated batch=[B,L,top_k] broadcasts with indices [B,L,top_k] exactly
        let down_out = gather_qmm(
            &activated,
            &self.down_proj.weight,
            &self.down_proj.scales,
            &self.down_proj.biases,
            indices,
            true,
            self.down_proj.group_size,
            self.down_proj.bits,
            sorted,
        )?;

        // Squeeze M=1: [B, L, top_k, D]
        down_out.squeeze_axes(&[-2])
    }

    /// Like `forward_gather` but reorders tokens globally by expert index
    /// before calling `gather_qmm`, matching mlx-lm's `_gather_sort` pattern.
    ///
    /// This gives coalesced GPU memory access and is 3-6x faster for prefill
    /// (L >= 32). For single-token decode (L=1) it's equivalent.
    ///
    /// `x`: `[B, L, D]`
    /// `indices`: `[B, L, top_k]` expert indices (need NOT be pre-sorted)
    /// Returns: `[B, L, top_k, D]`
    pub(crate) fn forward_gather_global_sort(
        &self,
        x: &Array,
        indices: &Array,
    ) -> Result<Array, Exception> {
        let x_shape = x.shape();
        let err = || Exception::custom("forward_gather_global_sort input must be [B, L, D]");
        let b = *x_shape.first().ok_or_else(err)?;
        let l = *x_shape.get(1).ok_or_else(err)?;
        let d = *x_shape.get(2).ok_or_else(err)?;
        let top_k = *indices
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("indices must have last dim"))?;

        // --- Global sort: flatten, argsort, reorder tokens by expert ---
        // indices: [B, L, top_k] -> [N] where N = B*L*top_k
        let idx_flat = indices.flatten(None, None)?;
        let order = ops::argsort_axis(&idx_flat, 0)?;
        let inv_order = ops::argsort_axis(&order, 0)?;

        // Map each sorted position back to its source token: order / top_k
        let top_k_u32 =
            u32::try_from(top_k).map_err(|_| Exception::custom("top_k must fit in u32"))?;
        let top_k_arr = Array::from_slice(&[top_k_u32], &[1]);
        let token_idx = order.floor_divide(&top_k_arr)?;

        // x_flat: [B*L, 1, D] -> x_sorted: [N, 1, D]
        let x_flat = x.reshape(&[b * l, 1, d])?;
        let x_sorted = x_flat.take_axis(&token_idx, 0)?;

        // idx_sorted: [N] — monotonically non-decreasing expert indices
        let idx_sorted = idx_flat.take_axis(&order, 0)?;

        // --- gather_qmm with coalesced access ---
        let gate_out = gather_qmm(
            &x_sorted,
            &self.gate_proj.weight,
            &self.gate_proj.scales,
            &self.gate_proj.biases,
            &idx_sorted,
            true,
            self.gate_proj.group_size,
            self.gate_proj.bits,
            true, // indices are globally sorted
        )?;
        let up_out = gather_qmm(
            &x_sorted,
            &self.up_proj.weight,
            &self.up_proj.scales,
            &self.up_proj.biases,
            &idx_sorted,
            true,
            self.up_proj.group_size,
            self.up_proj.bits,
            true,
        )?;

        let activated = swiglu(&gate_out, &up_out)?;

        let down_out = gather_qmm(
            &activated,
            &self.down_proj.weight,
            &self.down_proj.scales,
            &self.down_proj.biases,
            &idx_sorted,
            true,
            self.down_proj.group_size,
            self.down_proj.bits,
            true,
        )?;

        // down_out: [N, 1, D] -> squeeze M -> [N, D]
        let out_flat = down_out.squeeze_axes(&[-2])?;

        // --- Unsort: restore original token order ---
        let out_unsorted = out_flat.take_axis(&inv_order, 0)?;

        // Reshape back to [B, L, top_k, D]
        out_unsorted.reshape(&[b, l, top_k, d])
    }

    /// Like `forward_gather_global_sort` but fuses gate+up into a single
    /// `gather_qmm` call (3→2 per layer). Lazy-inits fused weights on first call.
    /// Production routing gates this behind `HIGGS_MOE_FFN_GATE_UP` because the
    /// fused cache duplicates the resident gate/up tensors.
    pub(crate) fn forward_gather_fused(
        &mut self,
        x: &Array,
        indices: &Array,
    ) -> Result<Array, Exception> {
        // Lazy-init: concatenate gate+up weights along axis 1 (intermediate dim).
        // MoE weights are [num_experts, intermediate_packed, hidden].
        if self.fused_gate_up.is_none() {
            let intermediate = *self
                .gate_proj
                .weight
                .shape()
                .get(1)
                .ok_or_else(|| Exception::custom("gate_proj weight missing dim 1"))?;
            let fw = ops::concatenate(&[&*self.gate_proj.weight, &*self.up_proj.weight], 1)?;
            let fs = ops::concatenate(&[&*self.gate_proj.scales, &*self.up_proj.scales], 1)?;
            let fb = ops::concatenate(&[&*self.gate_proj.biases, &*self.up_proj.biases], 1)?;
            fw.eval()?;
            fs.eval()?;
            fb.eval()?;
            self.fused_gate_up = Some((fw, fs, fb, intermediate));
        }
        let (fw, fs, fb, intermediate) = self
            .fused_gate_up
            .as_ref()
            .ok_or_else(|| Exception::custom("fused_gate_up missing after init"))?;

        // --- Global sort (same as forward_gather_global_sort) ---
        let x_shape = x.shape();
        let err = || Exception::custom("forward_gather_fused input must be [B, L, D]");
        let b = *x_shape.first().ok_or_else(err)?;
        let l = *x_shape.get(1).ok_or_else(err)?;
        let d = *x_shape.get(2).ok_or_else(err)?;
        let top_k = *indices
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("indices must have last dim"))?;

        let idx_flat = indices.flatten(None, None)?;
        let order = ops::argsort_axis(&idx_flat, 0)?;
        let inv_order = ops::argsort_axis(&order, 0)?;

        let top_k_u32 =
            u32::try_from(top_k).map_err(|_| Exception::custom("top_k must fit in u32"))?;
        let top_k_arr = Array::from_slice(&[top_k_u32], &[1]);
        let token_idx = order.floor_divide(&top_k_arr)?;

        let x_flat = x.reshape(&[b * l, 1, d])?;
        let x_sorted = x_flat.take_axis(&token_idx, 0)?;
        let idx_sorted = idx_flat.take_axis(&order, 0)?;

        // --- Fused gate+up: ONE gather_qmm instead of TWO ---
        let fused_out = gather_qmm(
            &x_sorted,
            fw,
            fs,
            fb,
            &idx_sorted,
            true,
            self.gate_proj.group_size,
            self.gate_proj.bits,
            true,
        )?;
        // Split at intermediate boundary → gate_out, up_out
        let parts = fused_out.split_at_indices(&[*intermediate], Some(-1))?;
        let gate_out = parts
            .first()
            .ok_or_else(|| Exception::custom("fused split failed"))?;
        let up_out = parts
            .get(1)
            .ok_or_else(|| Exception::custom("fused split failed"))?;
        let activated = swiglu(gate_out, up_out)?;

        // --- down_proj: unchanged ---
        let down_out = gather_qmm(
            &activated,
            &self.down_proj.weight,
            &self.down_proj.scales,
            &self.down_proj.biases,
            &idx_sorted,
            true,
            self.down_proj.group_size,
            self.down_proj.bits,
            true,
        )?;

        // down_out: [N, 1, D] -> squeeze M -> [N, D]
        let out_flat = down_out.squeeze_axes(&[-2])?;

        // --- Unsort: restore original token order ---
        let out_unsorted = out_flat.take_axis(&inv_order, 0)?;

        // Reshape back to [B, L, top_k, D]
        out_unsorted.reshape(&[b, l, top_k, d])
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(super) struct SparseMoeBlock {
    #[param]
    pub(super) gate: QLinear,
    #[param]
    pub(super) switch_mlp: SwitchMlpWeights,
    #[param]
    pub(super) shared_expert: Qwen3NextMLP,
    #[param]
    pub(super) shared_expert_gate: QLinear,
    pub(super) top_k: i32,
    pub(super) norm_topk_prob: bool,
}

impl SparseMoeBlock {
    pub(super) fn new(
        args: &Qwen3NextModelArgs,
        layer_idx: i32,
        ql: i32,
        qb: i32,
    ) -> Result<Self, Exception> {
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
        // Gate quantization: use per-layer override if present, else global
        let (gate_ql, gate_qb) = args
            .gate_quantization
            .as_ref()
            .map_or((ql, qb), |gq| (gq.group_size, gq.bits));
        let expert_gate = moe_expert_projection_quantization(args, layer_idx, "gate_proj", ql, qb)?;
        let expert_up = moe_expert_projection_quantization(args, layer_idx, "up_proj", ql, qb)?;
        let expert_down = moe_expert_projection_quantization(args, layer_idx, "down_proj", ql, qb)?;
        Ok(Self {
            gate: QLinear::new(gate_ql, gate_qb)?,
            switch_mlp: SwitchMlpWeights::new_with_projection_quantization(
                expert_gate,
                expert_up,
                expert_down,
            )?,
            shared_expert: Qwen3NextMLP::new(ql, qb)?,
            shared_expert_gate: QLinear::new(gate_ql, gate_qb)?,
            top_k: args.num_experts_per_tok,
            norm_topk_prob: args.norm_topk_prob,
        })
    }

    #[allow(dead_code)]
    pub(super) fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let gates = ops::softmax_axis(&self.gate.forward(x)?, -1, true)?;

        // Top-K selection via argpartition
        let neg_k = -self.top_k;
        let all_inds = ops::argpartition_axis(&gates, neg_k, -1)?;
        let num_experts = *gates
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("gates must have last dim"))?;
        let top_k_start = num_experts - self.top_k;
        let top_inds = ops::sort_axis(all_inds.index((.., .., top_k_start..)), -1)?;
        let raw_scores = gates.take_along_axis(&top_inds, -1)?;

        let top_scores = if self.norm_topk_prob {
            let score_sum = raw_scores.sum_axes(&[-1], true)?;
            raw_scores.divide(score_sum)?
        } else {
            raw_scores
        };

        // Expert computation via fused gather_qmm (global sort for coalesced access)
        let y = self.switch_mlp.forward_gather_global_sort(x, &top_inds)?;

        // Weighted sum over experts: [B, L, top_k, D] * [B, L, top_k, 1] -> sum -> [B, L, D]
        let expert_sum = y
            .multiply(&top_scores.expand_dims(-1)?)?
            .sum_axes(&[-2], false)?;

        // Shared expert
        let shared_y = self.shared_expert.forward(x)?;
        let shared_gate_val = nn::sigmoid(&self.shared_expert_gate.forward(x)?)?;
        let shared_out = shared_y.multiply(&shared_gate_val)?;

        expert_sum.add(shared_out)
    }
}
