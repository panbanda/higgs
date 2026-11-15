//! Tests for the dense feed-forward fast paths.

use crate::qwen3_next::{decoder::FfnBlock, linear::QLinear};
use mlx_rs::{module::Param, ops::indexing::IndexOp};

/// The dense FFN has a single-row decode kernel; a batch must fall back
/// to the general path and produce what each row gets alone.
#[test]
fn dense_ffn_batched_decode_matches_per_row() {
    use mlx_rs::Dtype;

    fn assign_qlinear(layer: &mut QLinear, out_dim: i32, in_dim: i32) {
        let raw = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[out_dim, in_dim], None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let (w, s, b) = crate::test_support::quantize_affine(&raw, 32, 4).unwrap();
        layer.weight = Param::new(w);
        layer.scales = Param::new(s);
        layer.biases = Param::new(b);
        layer.group_size = 32;
        layer.bits = 4;
    }

    let mut block = FfnBlock::new_dense(64, 4).unwrap();
    assign_qlinear(block.gate_proj.as_mut().unwrap(), 96, 64);
    assign_qlinear(block.up_proj.as_mut().unwrap(), 96, 64);
    assign_qlinear(block.down_proj.as_mut().unwrap(), 64, 96);

    let x = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[3, 1, 64], None)
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();
    let batched = block.forward(&x).unwrap();
    assert_eq!(batched.shape(), &[3, 1, 64]);
    for row in 0..3 {
        let alone = block.forward(&x.index((row..=row, .., ..))).unwrap();
        let diff = batched
            .index((row..=row, .., ..))
            .subtract(&alone)
            .unwrap()
            .abs()
            .unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item_cast();
        let largest: f32 = alone
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .item_cast();
        assert!(
            max_diff < 1e-2 * largest,
            "row {row}: batched FFN differs by {max_diff} of {largest}"
        );
    }
}

#[test]
fn dense_hidden_fused_matches_separate_path() {
    use mlx_rs::{Dtype, module::Param};

    fn assign_qlinear(layer: &mut QLinear, out_dim: i32, in_dim: i32) {
        let raw = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[out_dim, in_dim], None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let (w, s, b) = crate::test_support::quantize_affine(&raw, 32, 4).unwrap();
        layer.weight = Param::new(w);
        layer.scales = Param::new(s);
        layer.biases = Param::new(b);
        layer.group_size = 32;
        layer.bits = 4;
    }

    let mut block = FfnBlock::new_dense(64, 4).unwrap();
    assign_qlinear(block.gate_proj.as_mut().unwrap(), 96, 64);
    assign_qlinear(block.up_proj.as_mut().unwrap(), 96, 64);
    assign_qlinear(block.down_proj.as_mut().unwrap(), 64, 96);

    let x = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, 1, 64], None)
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();

    let fused = block.dense_hidden_fused(&x, false).unwrap();
    let separate = block.dense_hidden_separate(&x).unwrap();
    mlx_rs::transforms::eval([&fused, &separate]).unwrap();

    let diff = fused.subtract(&separate).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max(None).unwrap().item_cast();
    assert!(
        max_diff < 1e-3,
        "dense fused/separate hidden mismatch by {max_diff}"
    );
}
