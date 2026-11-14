//! Tests for quantized linear layers, embeddings, GEMV, and feature flags.

use crate::qwen3_next::{
    activations::swiglu,
    flags::{
        DenseFfnGemvMode, parse_dense_ffn_gemv_mode,
        should_force_dense_decode_safe_defaults_for_brand,
    },
    linear::{QLinear, quantized_forward},
    qgemv::qgemv_4bit,
};
use mlx_rs::{Array, Stream, ops::indexing::IndexOp};

#[test]
fn qlinear_zero_bits_uses_dense_weight() {
    let mut linear = QLinear::new(0, 0).unwrap();
    *linear.weight = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
    let input = Array::from_slice(&[2.0_f32, 1.0], &[1, 2]);
    let output = linear.forward(&input).unwrap();
    assert_eq!(output.shape(), &[1, 2]);
    assert_eq!(output.as_slice::<f32>(), &[4.0, 10.0]);
}

#[test]
fn test_swiglu() {
    let gate = Array::from_slice(&[1.0_f32, -1.0, 0.5], &[1, 3]);
    let x = Array::from_slice(&[2.0_f32, 3.0, 4.0], &[1, 3]);
    let result = swiglu(&gate, &x).unwrap();
    assert_eq!(result.shape(), &[1, 3]);
    // silu(1.0) * 2.0 = 0.7311 * 2.0 ~= 1.462
    let first: f32 = result.index((.., 0..1)).item_cast();
    assert!(first > 1.0);
}

#[test]
fn test_swiglu_numeric_correctness() {
    // silu(x) = x * sigmoid(x)
    // silu(0) = 0 * 0.5 = 0
    // silu(1) = 1 * sigmoid(1) = 1 * 0.7310586 = 0.7310586
    // silu(-1) = -1 * sigmoid(-1) = -1 * 0.2689414 = -0.2689414

    // swiglu(gate, x) = silu(gate) * x

    // gate=0, x=5 => silu(0) * 5 = 0
    let gate = Array::from_slice(&[0.0_f32], &[1, 1]);
    let x = Array::from_slice(&[5.0_f32], &[1, 1]);
    let result = swiglu(&gate, &x).unwrap();
    let val: f32 = result.item_cast();
    assert!((val - 0.0).abs() < 1e-6, "silu(0)*5 should be 0, got {val}");

    // gate=1, x=1 => silu(1) * 1 = 0.7310586
    let gate2 = Array::from_slice(&[1.0_f32], &[1, 1]);
    let x2 = Array::from_slice(&[1.0_f32], &[1, 1]);
    let result2 = swiglu(&gate2, &x2).unwrap();
    let val2: f32 = result2.item_cast();
    assert!(
        (val2 - 0.731_058_6).abs() < 1e-4,
        "silu(1)*1 should be ~0.7311, got {val2}"
    );

    // gate=-1, x=2 => silu(-1) * 2 = -0.2689414 * 2 = -0.5378828
    let gate3 = Array::from_slice(&[-1.0_f32], &[1, 1]);
    let x3 = Array::from_slice(&[2.0_f32], &[1, 1]);
    let result3 = swiglu(&gate3, &x3).unwrap();
    let val3: f32 = result3.item_cast();
    assert!(
        (val3 - (-0.537_882_8)).abs() < 1e-4,
        "silu(-1)*2 should be ~-0.5379, got {val3}"
    );
}

/// QEmbedding equivalence: dequantize-then-gather produces same result as
/// the full dequantize path (validates that gather on quantized storage
/// is safe for future optimisation).
#[test]
fn test_qembedding_gather_then_dequantize_equivalence() {
    use mlx_rs::transforms::eval;

    let group_size = 64i32;
    let bits = 4i32;
    let vocab = 256i32;
    let hidden = 128i32;

    // Create a random float matrix and quantize it
    let float_weight =
        mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[vocab, hidden], None).unwrap();
    eval([&float_weight].into_iter()).unwrap();
    let (qw, qs, qb) =
        crate::test_support::quantize_affine(&float_weight, group_size, bits).unwrap();
    eval([&qw, &qs, &qb].into_iter()).unwrap();

    let indices = Array::from_slice(&[0i32, 5, 42, 255, 5], &[5]);
    eval([&indices].into_iter()).unwrap();

    // Path A: dequantize full vocab, then gather (current QEmbedding::forward)
    let full_deq = crate::test_support::dequantize_affine(&qw, &qs, &qb, group_size, bits).unwrap();
    let path_a = full_deq.take_axis(&indices, 0).unwrap();
    eval([&path_a].into_iter()).unwrap();

    // Path B: gather quantized rows first, then dequantize only selected
    let sel_w = qw.take_axis(&indices, 0).unwrap();
    let sel_s = qs.take_axis(&indices, 0).unwrap();
    let sel_b = qb.take_axis(&indices, 0).unwrap();
    let path_b =
        crate::test_support::dequantize_affine(&sel_w, &sel_s, &sel_b, group_size, bits).unwrap();
    eval([&path_b].into_iter()).unwrap();

    // They should be identical (both round-trip through the same quantized repr)
    let diff = path_a.subtract(&path_b).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max(None).unwrap().item_cast();
    assert!(
        max_diff < 1e-6,
        "gather-then-dequantize should match dequantize-then-gather, max diff: {max_diff}"
    );
}

/// Helper: run qgemv_4bit against quantized_matmul reference and assert max abs error.
fn assert_qgemv_matches_reference(n: i32, k: i32, group_size: i32, label: &str) {
    use mlx_rs::Dtype;

    let x = mlx_rs::with_stream(&Stream::default(), || {
        mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[1, 1, k], None)
    })
    .unwrap()
    .as_dtype(Dtype::Bfloat16)
    .unwrap();

    let w_dense = mlx_rs::with_stream(&Stream::default(), || {
        mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[n, k], None)
    })
    .unwrap();
    let (w_q, scales, biases) =
        crate::test_support::quantize_affine(&w_dense, group_size, 4).unwrap();
    mlx_rs::transforms::eval([&w_q, &scales, &biases, &x]).unwrap();

    // Reference: MLX quantized_matmul
    let ref_out = quantized_forward(&x, &w_q, &scales, &biases, group_size, 4).unwrap();
    mlx_rs::transforms::eval([&ref_out]).unwrap();

    // Custom GEMV kernel — native dtypes, zero conversions
    let custom_out = qgemv_4bit(&x, &w_q, &scales, &biases, group_size).unwrap();
    mlx_rs::transforms::eval([&custom_out]).unwrap();

    let ref_f32 = ref_out.as_dtype(Dtype::Float32).unwrap();
    let cust_f32 = custom_out.as_dtype(Dtype::Float32).unwrap();
    mlx_rs::transforms::eval([&ref_f32, &cust_f32]).unwrap();

    let ref_vals = ref_f32.as_slice::<f32>();
    let cust_vals = cust_f32.as_slice::<f32>();

    assert_eq!(
        ref_out.shape(),
        custom_out.shape(),
        "[{label}] shape mismatch: ref={:?} vs custom={:?}",
        ref_out.shape(),
        custom_out.shape()
    );
    assert_eq!(ref_vals.len(), cust_vals.len(), "[{label}] length mismatch");

    let mut max_diff = 0.0f32;
    for i in 0..ref_vals.len() {
        let diff = (ref_vals[i] - cust_vals[i]).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff < 0.5,
            "[{label}] mismatch at {i}: ref={}, custom={}, diff={diff}",
            ref_vals[i],
            cust_vals[i]
        );
    }
    println!("[{label}] PASS — N={n} K={k} gs={group_size} max_diff={max_diff:.4}");
}

#[test]
fn test_qgemv_native_dtype_matches_reference() {
    for &k in &[256, 512, 1024, 4096] {
        let n = 16;
        let gs = 64;
        assert_qgemv_matches_reference(n, k, gs, &format!("K={k}"));
    }
}

#[test]
fn test_qgemv_various_group_sizes() {
    let k = 512;
    let n = 32;
    for &gs in &[32, 64, 128] {
        assert_qgemv_matches_reference(n, k, gs, &format!("gs={gs}"));
    }
}

#[test]
fn test_qgemv_large_n_rows() {
    // Realistic dims: gate+up fused (2*intermediate) and down projection
    assert_qgemv_matches_reference(512, 1024, 64, "N=512 K=1024");
    assert_qgemv_matches_reference(2048, 1024, 64, "N=2048 K=1024");
}

#[test]
fn parse_dense_ffn_gemv_mode_defaults_to_both() {
    assert_eq!(parse_dense_ffn_gemv_mode(None), DenseFfnGemvMode::Both);
    assert_eq!(
        parse_dense_ffn_gemv_mode(Some("unexpected")),
        DenseFfnGemvMode::Both
    );
}

#[test]
fn parse_dense_ffn_gemv_mode_supports_all_variants() {
    assert_eq!(
        parse_dense_ffn_gemv_mode(Some("fused")),
        DenseFfnGemvMode::FusedOnly
    );
    assert_eq!(
        parse_dense_ffn_gemv_mode(Some("fused_only")),
        DenseFfnGemvMode::FusedOnly
    );
    assert_eq!(
        parse_dense_ffn_gemv_mode(Some("down")),
        DenseFfnGemvMode::DownOnly
    );
    assert_eq!(
        parse_dense_ffn_gemv_mode(Some("down_only")),
        DenseFfnGemvMode::DownOnly
    );
    assert_eq!(
        parse_dense_ffn_gemv_mode(Some("off")),
        DenseFfnGemvMode::Off
    );
    assert_eq!(
        parse_dense_ffn_gemv_mode(Some("none")),
        DenseFfnGemvMode::Off
    );
}

#[test]
fn base_m4_forces_dense_decode_safe_defaults() {
    assert!(should_force_dense_decode_safe_defaults_for_brand(Some(
        "Apple M4"
    )));
    assert!(should_force_dense_decode_safe_defaults_for_brand(Some(
        " Apple M4 "
    )));
}

#[test]
fn non_base_m4_keeps_dense_decode_fastpaths_available() {
    assert!(!should_force_dense_decode_safe_defaults_for_brand(Some(
        "Apple M4 Pro"
    )));
    assert!(!should_force_dense_decode_safe_defaults_for_brand(Some(
        "Apple M4 Max"
    )));
    assert!(!should_force_dense_decode_safe_defaults_for_brand(Some(
        "Apple M5"
    )));
    assert!(!should_force_dense_decode_safe_defaults_for_brand(None));
}
