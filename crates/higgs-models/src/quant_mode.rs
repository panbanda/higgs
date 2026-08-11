//! Quantization mode dispatch for `quantized_matmul` / `quantize` / `dequantize`.
//!
//! `mlx-rs` pins `mode = "affine"` in its wrappers
//! (`mlx-rs/src/ops/quantization.rs:11`). MLX's C core supports `"mxfp4"`
//! (E2M1 with shared block exponent) natively via the same C functions — only
//! the Rust wrapper hides it. This module calls `mlx_sys` directly, mirroring
//! the FFI pattern in `cache.rs::slice_axis`, so we can pass either mode
//! without an upstream `mlx-rs` change.

use std::ffi::CStr;

use mlx_rs::{Array, Stream, error::Exception};

/// Quantization format for a packed weight tensor.
///
/// `serde` deserialises from the lowercase strings MLX stores in `config.json`
/// (`"affine"`, `"mxfp4"`). Missing / unknown values fall back to `Affine`
/// via [`QuantMode::default`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantMode {
    /// Per-group affine: `w_q * scale + bias`.
    #[default]
    Affine,
    /// MX-format E2M1 (4-bit) with a shared 8-bit block exponent, no bias.
    MxFp4,
    /// Unquantized dense weights (bf16/fp16). Used when a tensor has no
    /// `.scales` in the checkpoint (e.g. GDN dynamics in mixed-precision models).
    /// Not a config.json value — set by the loader after checkpoint pre-scan.
    Dense,
}

impl QuantMode {
    /// The C-string MLX expects for this mode.
    pub const fn cstr(&self) -> &'static CStr {
        match self {
            // Dense never reaches the C FFI — QLinear uses plain matmul.
            Self::Affine | Self::Dense => c"affine",
            Self::MxFp4 => c"mxfp4",
        }
    }

    /// Parse from the `"mode"` string stored in `config.json` quantization.
    ///
    /// Unknown / missing values fall back to `Affine` (the historical MLX
    /// default and the only mode older checkpoints declare).
    pub fn parse(s: &str) -> Self {
        match s.trim() {
            "mxfp4" => Self::MxFp4,
            _ => Self::Affine,
        }
    }

    pub const fn is_mxfp4(&self) -> bool {
        matches!(self, Self::MxFp4)
    }

    pub const fn is_dense(&self) -> bool {
        matches!(self, Self::Dense)
    }
}

/// Wrap an `i32` as MLX's optional-int C struct.
const fn opt_int(value: i32) -> mlx_sys::mlx_optional_int {
    mlx_sys::mlx_optional_int {
        value,
        has_value: true,
    }
}

/// "No dtype override" sentinel — matches mlx-rs's private `optional_dtype_none`.
const fn optional_dtype_none() -> mlx_sys::mlx_optional_dtype {
    mlx_sys::mlx_optional_dtype {
        value: mlx_sys::mlx_dtype__MLX_FLOAT32,
        has_value: false,
    }
}

/// Quantize `w` with an explicit mode. Returns `(w_q, scales, biases)`.
///
/// MLX returns 3 arrays for `Affine` (w, scales, biases) but only 2 for `MxFp4`
/// (w, scales) — the C library omits the unused biases. We synthesise a
/// placeholder so callers get a uniform triple and can ignore `biases` for
/// mxfp4.
///
/// # Safety
///
/// Calls MLX's C FFI directly; the result vector and its arrays are either
/// adopted into `Array` or freed on error.
#[allow(unsafe_code)]
pub fn quantize(
    w: &Array,
    group_size: i32,
    bits: i32,
    mode: QuantMode,
) -> Result<(Array, Array, Array), Exception> {
    let gs_opt = opt_int(group_size);
    let bits_opt = opt_int(bits);

    unsafe {
        let mut vec = mlx_sys::mlx_vector_array_new();
        let status = mlx_sys::mlx_quantize(
            &raw mut vec,
            w.as_ptr(),
            gs_opt,
            bits_opt,
            mode.cstr().as_ptr(),
            Stream::task_local_or_default().as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_vector_array_free(vec);
            return Err(Exception::custom(format!(
                "mlx_quantize failed (mode={}, bits={bits})",
                mode.cstr().to_string_lossy()
            )));
        }

        let n = mlx_sys::mlx_vector_array_size(vec);
        // Affine: (w, scales, biases). MxFp4: (w, scales) — biases omitted.
        if !(2..=3).contains(&n) {
            mlx_sys::mlx_vector_array_free(vec);
            return Err(Exception::custom(format!(
                "mlx_quantize returned {n} arrays, expected 2 (mxfp4) or 3 (affine)"
            )));
        }

        let take = |idx: usize| -> Result<Array, Exception> {
            let mut arr = mlx_sys::mlx_array_new();
            let s = mlx_sys::mlx_vector_array_get(&raw mut arr, vec, idx);
            if s != 0 {
                mlx_sys::mlx_array_free(arr);
                return Err(Exception::custom(format!(
                    "mlx_vector_array_get({idx}) failed"
                )));
            }
            Ok(Array::from_ptr(arr))
        };

        let wq = take(0)?;
        let scales = take(1)?;
        // For mxfp4 (n==2) there is no biases array; return a placeholder so the
        // triple shape is uniform. Callers pass None to matmul for mxfp4.
        let biases = if n >= 3 {
            take(2)?
        } else {
            Array::from_slice(&[0.0_f32], &[1])
        };
        mlx_sys::mlx_vector_array_free(vec);
        Ok((wq, scales, biases))
    }
}

/// Dequantize packed weights back to a dense array, honouring `mode`.
///
/// # Safety
///
/// Calls MLX's C FFI directly.
#[allow(unsafe_code)]
pub fn dequantize(
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: i32,
    bits: i32,
    mode: QuantMode,
) -> Result<Array, Exception> {
    let gs_opt = opt_int(group_size);
    let bits_opt = opt_int(bits);

    unsafe {
        let mut result = mlx_sys::mlx_array_new();
        #[allow(clippy::option_if_let_else)]
        let biases_ptr = match biases {
            Some(b) => b.as_ptr(),
            None => mlx_sys::mlx_array_new(),
        };
        let status = mlx_sys::mlx_dequantize(
            &raw mut result,
            w.as_ptr(),
            scales.as_ptr(),
            biases_ptr,
            gs_opt,
            bits_opt,
            mode.cstr().as_ptr(),
            optional_dtype_none(),
            Stream::task_local_or_default().as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_array_free(result);
            return Err(Exception::custom(format!(
                "mlx_dequantize failed (mode={}, bits={bits})",
                mode.cstr().to_string_lossy()
            )));
        }
        Ok(Array::from_ptr(result))
    }
}

/// Quantized matmul that honours the per-tensor `mode`.
///
/// Direct FFI into `mlx_quantized_matmul`. `biases` may be `None` for `MxFp4`
/// tensors (the C signature allows a null biases array); for `Affine` it is
/// required for correctness.
///
/// # Safety
///
/// Calls MLX's C FFI directly.
#[allow(unsafe_code, clippy::too_many_arguments)]
pub fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: i32,
    bits: i32,
    mode: QuantMode,
) -> Result<Array, Exception> {
    let gs_opt = opt_int(group_size);
    let bits_opt = opt_int(bits);

    unsafe {
        let mut result = mlx_sys::mlx_array_new();
        // mxfp4 has no zero-point; pass a fresh (null) array handle for biases.
        #[allow(clippy::option_if_let_else)]
        let biases_ptr = match biases {
            Some(b) => b.as_ptr(),
            None => mlx_sys::mlx_array_new(),
        };
        let status = mlx_sys::mlx_quantized_matmul(
            &raw mut result,
            x.as_ptr(),
            w.as_ptr(),
            scales.as_ptr(),
            biases_ptr,
            transpose,
            gs_opt,
            bits_opt,
            mode.cstr().as_ptr(),
            Stream::task_local_or_default().as_ptr(),
        );
        if status != 0 {
            mlx_sys::mlx_array_free(result);
            return Err(Exception::custom(format!(
                "mlx_quantized_matmul failed (mode={}, bits={bits})",
                mode.cstr().to_string_lossy()
            )));
        }
        Ok(Array::from_ptr(result))
    }
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    unsafe_code,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::needless_borrow
)]
mod tests {
    use super::*;

    #[test]
    fn parse_known_modes() {
        assert_eq!(QuantMode::parse("affine"), QuantMode::Affine);
        assert_eq!(QuantMode::parse("mxfp4"), QuantMode::MxFp4);
        assert_eq!(QuantMode::parse("  mxfp4  "), QuantMode::MxFp4);
    }

    #[test]
    fn parse_unknown_falls_back_to_affine() {
        assert_eq!(QuantMode::parse(""), QuantMode::Affine);
        assert_eq!(QuantMode::parse("int4"), QuantMode::Affine);
        assert_eq!(QuantMode::parse("MXFP4"), QuantMode::Affine);
    }

    #[test]
    fn default_is_affine() {
        assert_eq!(QuantMode::default(), QuantMode::Affine);
    }

    #[test]
    fn cstr_values_are_stable() {
        assert_eq!(QuantMode::Affine.cstr(), c"affine");
        assert_eq!(QuantMode::MxFp4.cstr(), c"mxfp4");
    }

    /// `quantized_matmul(mxfp4)` must produce output close to the *unquantized*
    /// matmul `x @ w`. This is the real correctness criterion: the FFI bypass
    /// calls the right C kernel with the right mode. Compares against the
    /// original dense weights, not against dequant (which would be circular).
    #[test]
    fn mxfp4_matmul_close_to_unquantized() {
        let (out_features, in_features): (i32, i32) = (32, 64);
        // Weights in [-1, 1]; in_features must divide mxfp4 group_size (32).
        let w_data: Vec<f32> = (0..(out_features * in_features))
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        let w = Array::from_slice(&w_data, &[out_features, in_features]);
        let x_data: Vec<f32> = (0..in_features).map(|i| (i as f32 * 0.07).sin()).collect();
        let x = Array::from_slice(&x_data, &[1, in_features]);

        // Unquantized reference: x @ w^T
        let ref_out = x.matmul(w.transpose().unwrap()).unwrap();
        let want = ref_out.as_slice::<f32>();

        // mxfp4 path
        let (wq, sq, _bq) = quantize(&w, 32, 4, QuantMode::MxFp4).unwrap();
        let got_arr = quantized_matmul(&x, &wq, &sq, None, true, 32, 4, QuantMode::MxFp4).unwrap();
        let got = got_arr.as_slice::<f32>();

        assert_eq!(got.len(), want.len(), "output length mismatch");
        // Each output element sums 64 dot products of E2M1-quantized products.
        // E2M1 has ~12.5% per-weight noise, but matmul averages it down. Check
        // that the sign and magnitude order match (correlation), not exact
        // round-off. A mode-mismatch bug would give ~100% error or NaN.
        let mut max_abs_err = 0.0_f32;
        let mut max_ref = 1e-6_f32;
        for (&g, &r) in got.iter().zip(want.iter()) {
            max_abs_err = max_abs_err.max((g - r).abs());
            max_ref = max_ref.max(r.abs());
        }
        let rel = max_abs_err / max_ref;
        assert!(
            rel < 0.35,
            "mxfp4 matmul too far from unquantized: max_abs_err={max_abs_err:.4}, max_ref={max_ref:.4}, rel={rel:.4}"
        );
    }

    /// Same equivalence check for affine mode, to confirm the bypass works for
    /// the existing path too (not just mxfp4).
    #[test]
    fn affine_matmul_matches_dequant_reference() {
        let (out_features, in_features): (i32, i32) = (64, 64);
        let w_data: Vec<f32> = (0..(out_features * in_features))
            .map(|i| (i as f32 * 0.1).rem_euclid(2.0) - 1.0)
            .collect();
        let w = Array::from_slice(&w_data, &[out_features, in_features]);
        let x_data: Vec<f32> = (0..in_features)
            .map(|i| (i as f32 * 0.07).rem_euclid(2.0) - 1.0)
            .collect();
        let x = Array::from_slice(&x_data, &[1, in_features]);

        let (wq, sq, bq) = quantize(&w, 64, 4, QuantMode::Affine).unwrap();
        let affine_out = quantized_matmul(&x, &wq, &sq, Some(&bq), true, 64, 4, QuantMode::Affine)
            .expect("affine quantized_matmul");

        let w_deq = dequantize(&wq, &sq, Some(&bq), 64, 4, QuantMode::Affine).unwrap();
        let ref_out = x.matmul(w_deq.transpose().unwrap()).unwrap();

        let got = affine_out.as_slice::<f32>();
        let want = ref_out.as_slice::<f32>();
        assert_eq!(got.len(), want.len());

        let mut max_abs_err = 0.0_f32;
        for (&g, &r) in got.iter().zip(want.iter()) {
            max_abs_err = max_abs_err.max((g - r).abs());
        }
        assert!(
            max_abs_err < 1e-3,
            "affine matmul disagrees with dequant reference: max_abs_err={max_abs_err:.6}"
        );
    }

    /// `quantize` → `dequantize` round-trip recovers the original weights
    /// within mxfp4's E2M1 precision envelope. Guards against mode-mismatch
    /// bugs where mxfp4 quantize output is fed to affine dequant (or vice versa).
    #[test]
    fn mxfp4_roundtrip_within_precision() {
        let n: i32 = 64;
        // Values spread across [-1, 1] — E2M1's useful range for a single scale
        // group. Step avoids exact zeros (which blow up relative error).
        let w_data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + 1.0) * 0.03).rem_euclid(2.0) - 1.0)
            .collect();
        let w = Array::from_slice(&w_data, &[1, n]);

        let (wq, sq, _bq) = quantize(&w, 32, 4, QuantMode::MxFp4).unwrap();
        let w_rec = dequantize(&wq, &sq, None, 32, 4, QuantMode::MxFp4)
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap();

        let orig = w.as_slice::<f32>();
        let rec = w_rec.as_slice::<f32>();
        // E2M1 with group_size=32 has one shared exponent per 32 elements and
        // ~8 mantissa levels per octave, so worst-case absolute error for a
        // group spanning [-1, 1] is ~0.25. Use absolute error (not relative —
        // values near zero make relative error unbounded for any 4-bit format).
        // A mode-mismatch bug would give ~1.0 error or NaN.
        let mut max_abs = 0.0_f32;
        for (&o, &r) in orig.iter().zip(rec.iter()) {
            max_abs = max_abs.max((o - r).abs());
        }
        assert!(
            max_abs < 0.35,
            "mxfp4 round-trip absolute error too high: {max_abs:.4}"
        );
    }
}
