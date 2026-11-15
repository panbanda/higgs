//! The `GatedDeltaNet` SSM-like linear attention layer.

use super::{
    activations::{gdn_output_gate, silu_direct},
    cache::{ArraysCache, chronological_conv_history},
    config::Qwen3NextModelArgs,
    flags::{async_layer_state_eval_enabled, compiled_gdn_decode_enabled},
    gated_delta_kernel::gated_delta_kernel_ffi,
    linear::QLinear,
};
use crate::utils::AttentionMask;
use mlx_rs::{
    Array, Dtype,
    builder::Builder,
    error::Exception,
    fast,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        self,
        indexing::{IndexOp, TryIndexMutOp},
    },
    transforms::compile::compile_with_state,
};
use std::cell::RefCell;

type CompiledGdnDecodeFn =
    dyn for<'a> FnMut(&mut ArraysCache, &'a [Array]) -> Result<Vec<Array>, Exception>;

thread_local! {
    static COMPILED_GDN_DECODE_FN: RefCell<Option<Box<CompiledGdnDecodeFn>>> = RefCell::new(None);
}

pub(super) fn make_compiled_gdn_decode() -> Box<CompiledGdnDecodeFn> {
    Box::new(compile_with_state(compiled_gdn_decode_step, true))
}

fn run_compiled_gdn_decode(cache: &mut ArraysCache, inputs: &[Array]) -> Result<Array, Exception> {
    COMPILED_GDN_DECODE_FN.with(|cell| {
        let mut guard = cell.borrow_mut();
        let compiled = guard.get_or_insert_with(make_compiled_gdn_decode);
        let mut out = compiled(cache, inputs)?;
        out.pop()
            .ok_or_else(|| Exception::custom("compiled GDN decode returned no outputs"))
    })
}

fn compute_g_direct(a_log: &Array, a: &Array, dt_bias: &Array) -> Result<Array, Exception> {
    let a_plus_bias = a.add(dt_bias)?;
    let sp = nn::softplus(&a_plus_bias)?;
    let neg_decay = a_log.exp()?.negative()?.multiply(sp)?;
    neg_decay.exp()
}

fn compiled_gdn_decode_step(
    cache: &mut ArraysCache,
    inputs: &[Array],
) -> Result<Vec<Array>, Exception> {
    let [q, k, v, g, beta, z, norm_weight] = inputs else {
        return Err(Exception::custom("compiled GDN decode expects 7 inputs"));
    };
    // q: [B, 1, Hv, Dk]
    // k: [B, 1, Hv, Dk]
    // v: [B, 1, Hv, Dv]
    // g: [B, 1, Hv]
    // beta: [B, 1, Hv]
    // z: [B, 1, Hv, Dv]
    // norm_weight: [Dv]

    let state = cache
        .ssm_state
        .as_ref()
        .ok_or_else(|| Exception::custom("compiled GDN decode requires initialized state"))?;

    let q_t = q.squeeze_axes(&[1])?;
    let k_t = k.squeeze_axes(&[1])?;
    let v_t = v.squeeze_axes(&[1])?;
    let g_t = g.squeeze_axes(&[1])?.expand_dims(-1)?.expand_dims(-1)?;
    let beta_t = beta.squeeze_axes(&[1])?.expand_dims(-1)?;

    let decayed_state = state.multiply(&g_t)?;
    let k_expanded = k_t.expand_dims(-2)?;
    let kv_mem = decayed_state
        .multiply(&k_expanded)?
        .sum_axes(&[-1], false)?;
    let delta = v_t.subtract(&kv_mem)?.multiply(&beta_t)?;
    let new_state = decayed_state.add(&k_expanded.multiply(&delta.expand_dims(-1)?)?)?;

    *cache
        .ssm_state
        .as_mut()
        .ok_or_else(|| Exception::custom("compiled GDN decode lost cache state"))? = new_state;

    let y_t = cache
        .ssm_state
        .as_ref()
        .ok_or_else(|| Exception::custom("compiled GDN decode missing updated state"))?
        .multiply(&q_t.expand_dims(-2)?)?
        .sum_axes(&[-1], false)?;
    let y = y_t.expand_dims(1)?;
    let normed = fast::rms_norm(&y, Some(norm_weight), 1e-6)?;
    let gated = nn::silu(z)?.multiply(&normed)?;
    Ok(vec![gated])
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, ModuleParameters)]
pub(super) struct GatedDeltaNet {
    #[param]
    in_proj_qkvz: QLinear,
    #[param]
    in_proj_ba: QLinear,
    // Separate projections for qwen3_5-style models (flat split, not per-head)
    #[param]
    in_proj_qkv: Option<QLinear>,
    #[param]
    in_proj_z: Option<QLinear>,
    #[param]
    in_proj_a: Option<QLinear>,
    #[param]
    in_proj_b: Option<QLinear>,
    #[param]
    pub(super) conv1d: nn::Conv1d,
    #[param]
    norm: nn::RmsNorm,
    #[param]
    out_proj: QLinear,
    #[param]
    A_log: Param<Array>,
    #[param]
    dt_bias: Param<Array>,
    num_k_heads: i32,
    pub(super) num_v_heads: i32,
    pub(super) head_k_dim: i32,
    pub(super) head_v_dim: i32,
    key_dim: i32,
    pub(super) conv_dim: i32,
    pub(super) conv_kernel_size: i32,
    use_separate_projections: bool,
    qk_norm_weight_q: Array,
    qk_norm_weight_k: Array,
    /// Pre-transposed conv weight for fast T=1 decode: [`kernel_size`, `conv_dim`].
    pub(super) conv_weight_t: Option<Array>,
}

impl GatedDeltaNet {
    pub(super) fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let num_k_heads = args.linear_num_key_heads;
        let num_v_heads = args.linear_num_value_heads;
        let head_k_dim = args.linear_key_head_dim;
        let head_v_dim = args.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel_size = args.linear_conv_kernel_dim;

        let use_sep = args.use_separate_gdn_projections;
        let in_proj_qkvz = QLinear::new(ql, qb)?;
        let in_proj_ba = QLinear::new(ql, qb)?;
        let in_proj_qkv = use_sep.then(|| QLinear::new(ql, qb)).transpose()?;
        let in_proj_z = use_sep.then(|| QLinear::new(ql, qb)).transpose()?;
        let in_proj_a = use_sep.then(|| QLinear::new(ql, qb)).transpose()?;
        let in_proj_b = use_sep.then(|| QLinear::new(ql, qb)).transpose()?;
        let conv1d = nn::Conv1dBuilder::new(conv_dim, conv_dim, conv_kernel_size)
            .bias(false)
            .groups(conv_dim)
            .padding(0)
            .build()?;
        let norm = nn::RmsNormBuilder::new(head_v_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let out_proj = QLinear::new(ql, qb)?;
        let a_log = Param::new(Array::zeros::<f32>(&[num_v_heads])?);
        let dt_bias = Param::new(Array::zeros::<f32>(&[num_v_heads])?);
        Ok(Self {
            in_proj_qkvz,
            in_proj_ba,
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            conv1d,
            norm,
            out_proj,
            A_log: a_log,
            dt_bias,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            conv_dim,
            conv_kernel_size,
            use_separate_projections: use_sep,
            qk_norm_weight_q: {
                let dim_f32 = f32::from(
                    i16::try_from(head_k_dim)
                        .map_err(|_| Exception::custom("head_k_dim out of i16 range"))?,
                );
                let value = dim_f32.sqrt().recip().powi(2);
                let values = vec![
                    value;
                    usize::try_from(head_k_dim).map_err(|_| Exception::custom(
                        "head_k_dim out of usize range"
                    ))?
                ];
                Array::from_slice(&values, &[head_k_dim])
            },
            qk_norm_weight_k: {
                let dim_f32 = f32::from(
                    i16::try_from(head_k_dim)
                        .map_err(|_| Exception::custom("head_k_dim out of i16 range"))?,
                );
                let value = dim_f32.sqrt().recip();
                let values = vec![
                    value;
                    usize::try_from(head_k_dim).map_err(|_| Exception::custom(
                        "head_k_dim out of usize range"
                    ))?
                ];
                Array::from_slice(&values, &[head_k_dim])
            },
            conv_weight_t: None,
        })
    }

    pub(super) fn decode_conv1d_step(
        &mut self,
        mixed_qkv: &Array,
        cache: &mut ArraysCache,
        batch: i32,
    ) -> Result<Array, Exception> {
        let history_len = self.conv_kernel_size.saturating_sub(1);
        let wt = if let Some(w) = &self.conv_weight_t {
            w.clone()
        } else {
            // Conv1d weight: [conv_dim, kernel_size, 1] -> [kernel_size, conv_dim]
            let raw_w = self.conv1d.weight.squeeze_axes(&[-1])?.transpose()?;
            let typed_w = raw_w.as_dtype(mixed_qkv.dtype())?;
            typed_w.eval()?;
            self.conv_weight_t = Some(typed_w.clone());
            typed_w
        };

        let current_flat = mixed_qkv.reshape(&[batch, self.conv_dim])?;
        let current_weight = wt.index((self.conv_kernel_size - 1, ..));
        let mut conv_flat = current_flat.multiply(&current_weight)?;

        if history_len > 0 {
            if cache.conv_state.is_none() {
                cache.conv_state = Some(ops::zeros_dtype(
                    &[batch, history_len, self.conv_dim],
                    mixed_qkv.dtype(),
                )?);
                cache.conv_pos = -1;
            }

            let history = cache
                .conv_state
                .as_mut()
                .ok_or_else(|| Exception::custom("decode conv history missing"))?;

            let available = cache.offset.min(history_len);
            if cache.conv_pos >= 0 {
                for lag in 0..available {
                    let idx = (cache.conv_pos - lag).rem_euclid(history_len);
                    let prev = history
                        .index((.., idx..idx + 1, ..))
                        .reshape(&[batch, self.conv_dim])?;
                    let weight = wt.index((history_len - 1 - lag, ..));
                    conv_flat = conv_flat.add(&prev.multiply(&weight)?)?;
                }
            }

            let next_pos = if cache.conv_pos < 0 {
                0
            } else {
                (cache.conv_pos + 1).rem_euclid(history_len)
            };
            history.try_index_mut((.., next_pos..next_pos + 1, ..), mixed_qkv.clone())?;
            cache.conv_pos = next_pos;
        }

        silu_direct(&conv_flat.reshape(&[batch, 1, self.conv_dim])?)
    }

    pub(super) fn chronological_conv_state(
        &self,
        cache: &mut ArraysCache,
        batch: i32,
        dtype: Dtype,
    ) -> Result<Array, Exception> {
        let n_keep = self.conv_kernel_size - 1;
        let Some(state) = cache.conv_state.take() else {
            return ops::zeros_dtype(&[batch, n_keep, self.conv_dim], dtype);
        };
        chronological_conv_history(&state, cache.conv_pos, cache.offset)
    }

    #[allow(non_snake_case, clippy::too_many_lines)]
    pub(super) fn forward(
        &mut self,
        inputs: &Array,
        _mask: Option<&AttentionMask>,
        cache: &mut ArraysCache,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let S = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // Project inputs and split into q, k, v, z, b, a
        let (q, k, v, z, b, a) = if self.use_separate_projections {
            // qwen3.5-style: 4 separate projections, flat split
            let qkv_proj = self
                .in_proj_qkv
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_qkv missing"))?;
            let z_proj = self
                .in_proj_z
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_z missing"))?;
            let b_proj = self
                .in_proj_b
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_b missing"))?;
            let a_proj = self
                .in_proj_a
                .as_ref()
                .ok_or_else(|| Exception::custom("in_proj_a missing"))?;

            let qkv = qkv_proj.forward_decode_fast(inputs)?;
            let z = z_proj.forward_decode_fast(inputs)?.reshape(&[
                B,
                S,
                self.num_v_heads,
                self.head_v_dim,
            ])?;
            let b = b_proj.forward_decode_fast(inputs)?;
            let a = a_proj.forward_decode_fast(inputs)?;

            let split_indices = &[self.key_dim, self.key_dim * 2];
            let qkv_parts = qkv.split_at_indices(split_indices, Some(-1))?;
            let q = qkv_parts
                .first()
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
            let k = qkv_parts
                .get(1)
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
            let v = qkv_parts
                .get(2)
                .ok_or_else(|| Exception::custom("qkv split failed"))?
                .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;

            (q, k, v, z, b, a)
        } else {
            // qwen3_next-style: combined projections, per-head reshape
            let mixed_qkvz = self.in_proj_qkvz.forward_decode_fast(inputs)?;
            let mixed_ba = self.in_proj_ba.forward_decode_fast(inputs)?;
            self.fix_query_key_value_ordering(&mixed_qkvz, &mixed_ba, B, S)?
        };

        // Concatenate q, k, v for conv input
        let q_flat = q.reshape(&[B, S, -1])?;
        let k_flat = k.reshape(&[B, S, -1])?;
        let v_flat = v.reshape(&[B, S, -1])?;
        let mixed_qkv = ops::concatenate(&[&q_flat, &k_flat, &v_flat], -1)?;
        let n_keep = self.conv_kernel_size - 1;

        let conv_out = if S == 1 {
            self.decode_conv1d_step(&mixed_qkv, cache, B)?
        } else {
            let conv_state = self.chronological_conv_state(cache, B, inputs.dtype())?;
            let conv_input = ops::concatenate(&[&conv_state, &mixed_qkv], 1)?;
            let conv_input_len = *conv_input
                .shape()
                .get(1)
                .ok_or_else(|| Exception::custom("conv_input missing seq dim"))?;
            let keep_start = conv_input_len - n_keep;
            cache.conv_state = Some(conv_input.index((.., keep_start.., ..)));
            cache.conv_pos = if n_keep > 0 { n_keep - 1 } else { -1 };
            silu_direct(&self.conv1d.forward(&conv_input)?)?
        };

        if S == 1 && async_layer_state_eval_enabled() {
            if let Some(conv_state) = cache.conv_state.as_ref() {
                mlx_rs::transforms::async_eval([conv_state])?;
            }
        }

        // Split conv output back to q, k, v
        let split_indices = &[self.key_dim, self.key_dim * 2];
        let conv_parts = conv_out.split_at_indices(split_indices, Some(-1))?;
        let conv_q = conv_parts
            .first()
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_k = conv_parts
            .get(1)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_v = conv_parts
            .get(2)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;

        // On first call, convert weight vectors to match input dtype.
        let in_dt = inputs.dtype();
        if self.qk_norm_weight_q.dtype() != in_dt {
            self.qk_norm_weight_q = self.qk_norm_weight_q.as_dtype(in_dt)?;
            self.qk_norm_weight_k = self.qk_norm_weight_k.as_dtype(in_dt)?;
        }

        let norm_q = fast::rms_norm(&conv_q, Some(&self.qk_norm_weight_q), 1e-6)?;
        let norm_k = fast::rms_norm(&conv_k, Some(&self.qk_norm_weight_k), 1e-6)?;

        // The compiled step is traced for a single sequence.
        let use_compiled_decode = compiled_gdn_decode_enabled() && S == 1 && B == 1;
        if use_compiled_decode {
            if cache.ssm_state.is_none() {
                cache.ssm_state = Some(ops::zeros_dtype(
                    &[B, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                    Dtype::Float32,
                )?);
            }

            let repeat_factor = self.num_v_heads / self.num_k_heads;
            let q_decode = if repeat_factor > 1 {
                ops::repeat_axis::<f32>(norm_q, repeat_factor, -2)?
            } else {
                norm_q
            };
            let k_decode = if repeat_factor > 1 {
                ops::repeat_axis::<f32>(norm_k, repeat_factor, -2)?
            } else {
                norm_k
            };
            let g = compute_g_direct(self.A_log.as_ref(), &a, self.dt_bias.as_ref())?;
            let beta = nn::sigmoid(&b)?;
            let kernel_inputs = [
                q_decode,
                k_decode,
                conv_v,
                g,
                beta,
                z,
                self.norm.weight.as_ref().clone(),
            ];
            let gated_out = run_compiled_gdn_decode(cache, &kernel_inputs)?;
            cache.offset += S;

            let out_flat = gated_out.reshape(&[B, S, -1])?;
            let out = self.out_proj.forward_decode_fast(&out_flat)?;
            return mlx_rs::stop_gradient(&out);
        }

        // Get or initialize SSM state: [B, Hv, Dv, Dk]
        let state = match cache.ssm_state.take() {
            Some(state) => state,
            None => ops::zeros_dtype(
                &[B, self.num_v_heads, self.head_v_dim, self.head_k_dim],
                Dtype::Float32,
            )?,
        };

        // Fused kernel: computes g, beta, AND runs the full recurrence in one dispatch.
        let (y, new_state) = gated_delta_kernel_ffi(
            &norm_q,
            &norm_k,
            &conv_v,
            &self.A_log,
            &a,
            &self.dt_bias,
            &b,
            &state,
            B,
            S,
            self.num_k_heads,
            self.head_k_dim,
            self.num_v_heads,
            self.head_v_dim,
        )?;
        cache.ssm_state = Some(new_state);
        cache.offset += S;

        if S == 1 && async_layer_state_eval_enabled() {
            if let Some(ssm_state) = cache.ssm_state.as_ref() {
                mlx_rs::transforms::async_eval([ssm_state])?;
            }
        }

        // Fused RMSNorm + gated output: silu(z) * rms_norm(y)
        // nn::silu is pre-compiled in MLX (1 fused dispatch vs 3 for manual swiglu)
        let gated_out = gdn_output_gate(&y, self.norm.weight.as_ref(), self.norm.eps, &z)?;

        // Output projection
        let out_flat = gated_out.reshape(&[B, S, -1])?;
        let out = self.out_proj.forward_decode_fast(&out_flat)?;
        if S == 1 {
            mlx_rs::stop_gradient(&out)
        } else {
            Ok(out)
        }
    }

    /// Reorder the projected qkvz and ba tensors into separate heads.
    #[allow(non_snake_case, clippy::type_complexity)]
    fn fix_query_key_value_ordering(
        &self,
        mixed_qkvz: &Array,
        mixed_ba: &Array,
        B: i32,
        S: i32,
    ) -> Result<(Array, Array, Array, Array, Array, Array), Exception> {
        let nk = self.num_k_heads;
        let dn = self.head_k_dim;
        let nv = self.num_v_heads;
        let dv = self.head_v_dim;
        let v_per_k = nv / nk;

        // Reshape to [B, S, nk, -1]
        let qkvz = mixed_qkvz.reshape(&[B, S, nk, -1])?;
        let ba = mixed_ba.reshape(&[B, S, nk, -1])?;

        // Split qkvz at [dn, 2*dn, 2*dn + v_per_k*dv]
        let split_at = &[dn, 2 * dn, 2 * dn + v_per_k * dv];
        let qkvz_parts = qkvz.split_at_indices(split_at, Some(-1))?;
        let q = qkvz_parts
            .first()
            .ok_or_else(|| Exception::custom("qkvz split failed"))?
            .clone();
        let k = qkvz_parts
            .get(1)
            .ok_or_else(|| Exception::custom("qkvz split failed"))?
            .clone();
        let v_raw = qkvz_parts
            .get(2)
            .ok_or_else(|| Exception::custom("qkvz split failed"))?;
        let z_raw = qkvz_parts
            .get(3)
            .ok_or_else(|| Exception::custom("qkvz split failed"))?;

        let v = v_raw.reshape(&[B, S, nv, dv])?;
        let z = z_raw.reshape(&[B, S, nv, dv])?;

        // Split ba at [v_per_k]
        let ba_parts = ba.split_at_indices(&[v_per_k], Some(-1))?;
        let b_raw = ba_parts
            .first()
            .ok_or_else(|| Exception::custom("ba split failed"))?;
        let a_raw = ba_parts
            .get(1)
            .ok_or_else(|| Exception::custom("ba split failed"))?;

        let b = b_raw.reshape(&[B, S, nv])?;
        let a = a_raw.reshape(&[B, S, nv])?;

        Ok((q, k, v, z, b, a))
    }
}

/// Reference implementation of gate computation (used by tests).
/// Production code uses `compute_g_beta_kernel_ffi` instead.
#[cfg(test)]
pub(super) fn compute_g_compiled(
    (a_log, a, dt_bias): (&Array, &Array, &Array),
) -> Result<Array, Exception> {
    let a_plus_bias = a.add(dt_bias)?;
    let sp = nn::softplus(&a_plus_bias)?;
    let neg_decay = a_log.exp()?.negative()?.multiply(sp)?;
    neg_decay.exp()
}
