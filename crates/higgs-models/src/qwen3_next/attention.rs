//! Full attention layers with gated queries and partial `RoPE`.

use super::{
    activations::sigmoid_mul,
    config::Qwen3NextModelArgs,
    flags::async_layer_state_eval_enabled,
    linear::{DenseLinearNoBias, QLinear},
};
use crate::{
    cache::{KeyValueCache, SteppingKeyValueCache},
    utils::{AttentionMask, apply_rope, apply_rope_at_cache},
};
use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    fast,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops::{self, indexing::IndexOp},
};

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen3NextAttention {
    #[param]
    pub(super) q_proj: QLinear,
    #[param]
    pub(super) k_proj: QLinear,
    #[param]
    pub(super) v_proj: QLinear,
    #[param]
    pub(super) o_proj: QLinear,
    #[param]
    pub(super) q_norm: nn::RmsNorm,
    #[param]
    pub(super) k_norm: nn::RmsNorm,
    #[param]
    rope: nn::Rope,
    pub(super) num_attention_heads: i32,
    pub(super) num_key_value_heads: i32,
    pub(super) scale: f32,
}

impl Qwen3NextAttention {
    pub(super) fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let head_dim = args.head_dim;
        let head_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        let scale = head_dim_f32.sqrt().recip();
        let rope_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        // partial_rotary_factor * head_dim is always a small positive integer (e.g. 64)
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let partial_dim = (rope_dim_f32 * args.partial_rotary_factor).round() as i32;

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
            rope: nn::RopeBuilder::new(partial_dim)
                .traditional(false)
                .base(args.rope_theta)
                .scale(1.0_f32)
                .build()
                .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?,
            num_attention_heads: args.num_attention_heads,
            num_key_value_heads: args.num_key_value_heads,
            scale,
        })
    }

    #[allow(non_snake_case)]
    pub(super) fn forward<C: KeyValueCache>(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: &mut C,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // Q is projected to 2 * num_heads * head_dim (doubled for gating)
        let q_proj_output = self.q_proj.forward_decode_fast(x)?;
        let q_reshaped = q_proj_output.reshape(&[B, L, self.num_attention_heads, -1])?;
        let q_halves = q_reshaped.split_equal(2, Some(-1))?;
        let queries_pre = q_halves
            .first()
            .ok_or_else(|| Exception::custom("split produced empty result"))?;
        let gate = q_halves
            .get(1)
            .ok_or_else(|| Exception::custom("split produced empty result"))?
            .reshape(&[B, L, -1])?;

        let keys_raw = self.k_proj.forward_decode_fast(x)?;
        let values_raw = self.v_proj.forward_decode_fast(x)?;

        // Per-head RmsNorm then transpose to [B, H, L, D]
        let mut queries = self
            .q_norm
            .forward(queries_pre)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = self
            .k_norm
            .forward(&keys_raw.reshape(&[B, L, self.num_key_value_heads, -1])?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values_raw
            .reshape(&[B, L, self.num_key_value_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        queries = apply_rope_at_cache(&queries, &self.rope, cache)?;
        keys = apply_rope_at_cache(&keys, &self.rope, cache)?;
        let view = cache.update_and_view(keys, values)?;
        let try_tq_decode = mask.is_none() && L == 1;
        let output = match view {
            crate::cache::KvCacheView::TurboQuant(tq_view) if try_tq_decode => {
                let scores = tq_view.decode_scores(&queries, self.num_attention_heads)?;
                let scale_arr = Array::from_f32(self.scale).as_dtype(scores.dtype())?;
                let weights = ops::softmax_axis(&scores.multiply(&scale_arr)?, -1, true)?;
                tq_view
                    .decode_values(&weights, self.num_attention_heads)?
                    .transpose_axes(&[0, 2, 1, 3])?
                    .reshape(&[B, L, -1])?
            }
            other @ (crate::cache::KvCacheView::Dense { .. }
            | crate::cache::KvCacheView::TurboQuant(_)) => {
                let (cached_keys, cached_values) = other.into_dense()?;
                let sdpa_mask = mask.map(fast::ScaledDotProductAttentionMask::from);
                fast::scaled_dot_product_attention(
                    queries,
                    cached_keys,
                    cached_values,
                    self.scale,
                    sdpa_mask,
                    None::<&Array>,
                )?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[B, L, -1])?
            }
        };
        if L == 1 && async_layer_state_eval_enabled() {
            mlx_rs::transforms::async_eval(cache.eval_targets())?;
        }

        let gated = sigmoid_mul(&gate, &output)?;
        let out = self.o_proj.forward_decode_fast(&gated)?;
        if L == 1 {
            mlx_rs::stop_gradient(&out)
        } else {
            Ok(out)
        }
    }

    /// Apply `RoPE` at custom positions using `rope_dynamic`.
    ///
    /// # Arguments
    /// * `queries` - Query tensor [B, `n_heads`, L, `head_dim`]
    /// * `keys` - Key tensor [B, `n_kv_heads`, L, `head_dim`]
    /// * `positions` - Position indices [L] (can be non-contiguous)
    ///
    /// # Returns
    /// (queries, keys) with `RoPE` applied at specified positions
    pub fn apply_rope_at_positions(
        &self,
        queries: &Array,
        keys: &Array,
        positions: &Array,
    ) -> Result<(Array, Array), Exception> {
        // Use manual RoPE implementation for per-token positions
        let queries_with_rope = apply_rope_manual(
            queries,
            positions,
            self.rope.dimensions,
            self.rope.base,
            self.rope.scale,
        )?;

        let keys_with_rope = apply_rope_manual(
            keys,
            positions,
            self.rope.dimensions,
            self.rope.base,
            self.rope.scale,
        )?;

        Ok((queries_with_rope, keys_with_rope))
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(super) struct DenseQwen3NextAttention {
    #[param]
    q_proj: DenseLinearNoBias,
    #[param]
    k_proj: DenseLinearNoBias,
    #[param]
    v_proj: DenseLinearNoBias,
    #[param]
    o_proj: DenseLinearNoBias,
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

impl DenseQwen3NextAttention {
    pub(super) fn new(args: &Qwen3NextModelArgs) -> Result<Self, Exception> {
        let head_dim = args.head_dim;
        let head_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        let scale = head_dim_f32.sqrt().recip();
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let partial_dim = (head_dim_f32 * args.partial_rotary_factor).round() as i32;

        Ok(Self {
            q_proj: DenseLinearNoBias::new(),
            k_proj: DenseLinearNoBias::new(),
            v_proj: DenseLinearNoBias::new(),
            o_proj: DenseLinearNoBias::new(),
            q_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            k_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            rope: nn::RopeBuilder::new(partial_dim)
                .traditional(false)
                .base(args.rope_theta)
                .scale(1.0_f32)
                .build()
                .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?,
            num_attention_heads: args.num_attention_heads,
            num_key_value_heads: args.num_key_value_heads,
            scale,
        })
    }

    #[allow(non_snake_case)]
    pub(super) fn forward(
        &mut self,
        x: &Array,
        mask: Option<&AttentionMask>,
        cache: &mut SteppingKeyValueCache,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        let q_proj_output = self.q_proj.forward(x)?;
        let q_reshaped = q_proj_output.reshape(&[B, L, self.num_attention_heads, -1])?;
        let q_halves = q_reshaped.split_equal(2, Some(-1))?;
        let queries_pre = q_halves
            .first()
            .ok_or_else(|| Exception::custom("split produced empty result"))?;
        let gate = q_halves
            .get(1)
            .ok_or_else(|| Exception::custom("split produced empty result"))?
            .reshape(&[B, L, -1])?;

        let keys_raw = self.k_proj.forward(x)?;
        let values_raw = self.v_proj.forward(x)?;

        let mut queries = self
            .q_norm
            .forward(queries_pre)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = self
            .k_norm
            .forward(&keys_raw.reshape(&[B, L, self.num_key_value_heads, -1])?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = values_raw
            .reshape(&[B, L, self.num_key_value_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let offset = cache.offset();
        queries = apply_rope(&queries, &self.rope, offset)?;
        keys = apply_rope(&keys, &self.rope, offset)?;

        let view = cache.update_and_view(keys, values)?;
        let try_tq_decode = mask.is_none() && L == 1;

        let output = match view {
            crate::cache::KvCacheView::TurboQuant(tq_view) if try_tq_decode => {
                let scores = tq_view.decode_scores(&queries, self.num_attention_heads)?;
                let scale_arr = Array::from_f32(self.scale).as_dtype(scores.dtype())?;
                let weights = ops::softmax_axis(&scores.multiply(&scale_arr)?, -1, true)?;
                tq_view
                    .decode_values(&weights, self.num_attention_heads)?
                    .transpose_axes(&[0, 2, 1, 3])?
                    .reshape(&[B, L, -1])?
            }
            other @ (crate::cache::KvCacheView::Dense { .. }
            | crate::cache::KvCacheView::TurboQuant(_)) => {
                let (cached_keys, cached_values) = other.into_dense()?;
                let sdpa_mask = mask.map(fast::ScaledDotProductAttentionMask::from);
                fast::scaled_dot_product_attention(
                    queries,
                    cached_keys,
                    cached_values,
                    self.scale,
                    sdpa_mask,
                    None::<&Array>,
                )?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[B, L, -1])?
            }
        };

        if L == 1 && async_layer_state_eval_enabled() {
            mlx_rs::transforms::async_eval(cache.eval_targets())?;
        }

        let gated = sigmoid_mul(&gate, &output)?;
        let out = self.o_proj.forward(&gated)?;
        if L == 1 {
            mlx_rs::stop_gradient(&out)
        } else {
            Ok(out)
        }
    }
}

// Manual RoPE implementation for arbitrary positions
/// Manual `RoPE` implementation for arbitrary positions
#[allow(dead_code)]
fn apply_rope_manual(
    x: &Array,
    positions: &Array,
    dimensions: i32,
    base: f32,
    _scale: f32,
) -> Result<Array, Exception> {
    use mlx_rs::ops;

    // x shape: [B, H, L, D] or [B, L, D]
    let shape = x.shape();
    let ndim = shape.len();

    if ndim < 2 {
        return Err(Exception::custom("Input must have at least 2 dimensions"));
    }

    let half_dim = dimensions / 2;
    let half_dim_i32 = half_dim;
    #[allow(clippy::cast_precision_loss)]
    let dimensions_f32 = f32::from(i16::try_from(dimensions).unwrap_or(i16::MAX));

    // Compute frequencies: base^(-2i/dimensions) for i in [0, half_dim)
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let i_f32 = f32::from(i16::try_from(i).unwrap_or(i16::MAX));
            let power = -2.0 * i_f32 / dimensions_f32;
            base.powf(power)
        })
        .collect();
    let inv_freq_arr = Array::from_slice(&inv_freq, &[half_dim_i32]);

    // Get positions as [L] or [B, L]
    let pos_shape = positions.shape();
    let l_dim = *pos_shape
        .last()
        .ok_or_else(|| Exception::custom("positions must have at least 1 dim"))?;

    tracing::debug!(
        "apply_rope_manual: x.shape={:?}, positions.shape={:?}, dimensions={}, base={}",
        x.shape(),
        positions.shape(),
        dimensions,
        base
    );
    // Compute angles: positions * inv_freq
    // positions: [L], inv_freq: [half_dim] -> angles: [L, half_dim]
    let positions_expanded = positions.reshape(&[l_dim, 1])?;
    let inv_freq_expanded = inv_freq_arr.reshape(&[1, half_dim_i32])?;
    let angles = ops::multiply(&positions_expanded, &inv_freq_expanded)?;

    // Compute cos and sin
    let cos_raw = ops::cos(&angles)?;
    let sin_raw = ops::sin(&angles)?;

    // Reshape for broadcasting: [1, 1, L, half_dim] or [1, L, half_dim]
    let cos_shape: Vec<i32> = if ndim == 4 {
        vec![1, 1, l_dim, half_dim_i32]
    } else {
        vec![1, l_dim, half_dim_i32]
    };
    let cos = cos_raw.reshape(&cos_shape)?;
    let sin = sin_raw.reshape(&cos_shape)?;

    // Split x into two halves along last dimension
    let x_first = x.index((.., .., .., ..half_dim));
    let x_second = x.index((.., .., .., half_dim..));

    // Apply RoPE rotation
    // output_first = x_first * cos - x_second * sin
    // output_second = x_first * sin + x_second * cos
    let output_first = ops::subtract(
        &ops::multiply(&x_first, &cos)?,
        &ops::multiply(&x_second, &sin)?,
    )?;
    let output_second = ops::add(
        &ops::multiply(&x_first, &sin)?,
        &ops::multiply(&x_second, &cos)?,
    )?;

    // Concatenate back
    let last_axis = i32::try_from(ndim.saturating_sub(1))
        .map_err(|_| Exception::custom("ndim too large for i32"))?;
    ops::concatenate(&[&output_first, &output_second], last_axis)
}
