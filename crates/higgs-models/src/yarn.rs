//! YaRN RoPE helpers shared across models.
//!
//! Extracted verbatim from the original site in `deepseek_v2.rs`. The
//! `apply_yarn_rope` wrapper adds a `traditional` flag so Qwen3-family models
//! (Bonsai) can reuse the same freq precomputation with `traditional=false`,
//! while DeepSeek stays on `traditional=true`.

#![allow(clippy::doc_markdown)] // YaRN, RoPE, etc. are domain terms, not items.

use std::f32::consts::PI;

use mlx_rs::{Array, error::Exception, fast};

#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
fn yarn_find_correction_dim(num_rotations: f32, dim: i32, base: f32, max_pos: i32) -> f32 {
    let dim_f = dim as f32;
    let max_pos_f = max_pos as f32;
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

pub(crate) fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        (0.1 * mscale).mul_add(scale.ln(), 1.0)
    }
}

#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing
)]
pub(crate) fn compute_yarn_freqs(
    dim: i32,
    base: f32,
    scaling_factor: f32,
    orig_max_pos: i32,
    beta_fast: f32,
    beta_slow: f32,
) -> Array {
    let half_dim = dim / 2;
    let dim_f = dim as f32;

    let mut freq_extra = Vec::with_capacity(half_dim as usize);
    let mut freq_inter = Vec::with_capacity(half_dim as usize);
    for i in 0..half_dim {
        let exp = (2 * i) as f32 / dim_f;
        let theta = base.powf(exp);
        freq_extra.push(theta);
        freq_inter.push(scaling_factor * theta);
    }

    let (low, high) = yarn_find_correction_range(beta_fast, beta_slow, dim, base, orig_max_pos);

    let low_f = low as f32;
    let high_f = high as f32;
    let range = if (high_f - low_f).abs() < 0.001 {
        high_f - low_f + 0.001
    } else {
        high_f - low_f
    };

    let mut freqs = Vec::with_capacity(half_dim as usize);
    for i in 0..half_dim as usize {
        let idx_f = i as f32;
        let ramp = ((idx_f - low_f) / range).clamp(0.0, 1.0);
        let mask = 1.0 - ramp;
        let inter = freq_inter[i];
        let extra = freq_extra[i];
        let denom = inter * mask + extra * (1.0 - mask);
        freqs.push((inter * extra) / denom);
    }

    Array::from_slice(&freqs, &[half_dim])
}

/// Apply YaRN-scaled RoPE.
///
/// When `mscale != 1.0`, inputs are pre-scaled before rotation (matches the
/// DeepSeek reference). `traditional=false` matches the Qwen3 / LLaMA rope
/// layout; `traditional=true` matches DeepSeek's packed complex layout.
///
/// `offset` is a scalar `Array` (not an `i32`) so the value is not baked into
/// compiled traces — required for `compile_with_state` wrapping of decode.
pub(crate) fn apply_yarn_rope(
    x: &Array,
    dim: i32,
    base: f32,
    yarn_freqs: Option<&Array>,
    mscale: f32,
    offset: &Array,
    traditional: bool,
) -> Result<Array, Exception> {
    let x_scaled = if (mscale - 1.0).abs() > f32::EPSILON {
        // Match x's dtype to avoid silent upcast (fp16 → f32) that bleeds into
        // the entire attention path (rope, sdpa, o_proj inputs). For Bonsai
        // with rope_yarn_factor>1, mscale ≈ 1.14, so this branch fires every
        // rope call; without the cast the whole decode runs in f32 and pays
        // ~28 ms/step on 8B. See bisect_decode v6 vs v7.
        let scalar = Array::from_f32(mscale).as_dtype(x.dtype())?;
        x.multiply(&scalar)?
    } else {
        x.clone()
    };
    yarn_freqs.map_or_else(
        || {
            fast::rope_dynamic(
                &x_scaled,
                dim,
                traditional,
                base,
                1.0,
                offset,
                None::<&Array>,
            )
        },
        |freqs| {
            fast::rope_dynamic(
                &x_scaled,
                dim,
                traditional,
                None::<f32>,
                1.0,
                offset,
                Some(freqs),
            )
        },
    )
}

#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::random;

    /// Parity: dynamic-offset rope (via `apply_yarn_rope`) must match
    /// static-offset `fast::rope` for every offset in 0..64, both with and
    /// without precomputed YaRN freqs. Guards the B1 step 1 migration to
    /// `fast::rope_dynamic` (prerequisite for `compile_with_state`-wrapped
    /// decode — the static `offset: i32` was being baked into the compile
    /// trace, forcing a recompile every step).
    #[test]
    #[ignore = "passes targeted (`cargo test yarn::`) but fails when run after other MLX tests in the same process — global Metal/RNG state contamination, pre-existing harness limitation"]
    fn rope_dynamic_matches_static_offset_0_to_64() {
        random::seed(71).unwrap();
        // [B=2, H=4, T=1, head_dim=16] — matches decode shape (T=1).
        let head_dim: i32 = 16;
        let base: f32 = 10_000.0;
        let x = random::uniform::<_, f32>(0.0, 1.0, &[2, 4, 1, head_dim], None).unwrap();

        // Case A: no yarn_freqs (base path).
        for offset in 0_i32..64 {
            let off_arr = Array::from_int(offset);
            let got = apply_yarn_rope(&x, head_dim, base, None, 1.0, &off_arr, false).unwrap();
            let want = fast::rope(&x, head_dim, false, base, 1.0, offset, None::<&Array>).unwrap();
            let diff = (&got - &want)
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            assert!(
                diff < 1e-5,
                "offset={offset} no-freqs: max_diff={diff} >= 1e-5"
            );
        }

        // Case B: with precomputed yarn_freqs (Bonsai path).
        let freqs = compute_yarn_freqs(head_dim, base, 1.0, 2048, 32.0, 1.0);
        for offset in 0_i32..64 {
            let off_arr = Array::from_int(offset);
            let got =
                apply_yarn_rope(&x, head_dim, base, Some(&freqs), 1.0, &off_arr, false).unwrap();
            let want =
                fast::rope(&x, head_dim, false, None::<f32>, 1.0, offset, Some(&freqs)).unwrap();
            let diff = (&got - &want)
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            assert!(
                diff < 1e-5,
                "offset={offset} with-freqs: max_diff={diff} >= 1e-5"
            );
        }
    }
}
