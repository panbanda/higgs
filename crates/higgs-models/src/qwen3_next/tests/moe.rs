//! Tests for the sparse mixture-of-experts path and `gather_qmm`.

use super::{minimal_qwen3_next_args, quantize_weights};
use crate::qwen3_next::{
    config::{QuantizationConfig, Qwen3NextModelArgs},
    ffi::gather_qmm,
    linear::quantized_forward,
    moe::{SparseMoeBlock, SwitchMlpWeights},
};
use mlx_rs::Array;

#[test]
fn test_sparse_moe_rejects_top_k_exceeding_num_experts() {
    assert_sparse_moe_rejects(
        |a| {
            a.num_experts = 4;
            a.num_experts_per_tok = 8;
        },
        "num_experts_per_tok",
    );
}

#[test]
fn test_sparse_moe_accepts_top_k_equal_to_num_experts() {
    let mut args = minimal_qwen3_next_args();
    args.num_experts = 4;
    args.num_experts_per_tok = 4; // top_k == num_experts is fine
    let result = SparseMoeBlock::new(&args, 0, 64, 4);
    assert!(result.is_ok());
}

fn assert_sparse_moe_rejects(
    mutate: impl FnOnce(&mut Qwen3NextModelArgs),
    expected_substring: &str,
) {
    let mut args = minimal_qwen3_next_args();
    mutate(&mut args);
    let result = SparseMoeBlock::new(&args, 0, 64, 4);
    assert!(result.is_err(), "Should reject invalid args");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains(expected_substring),
        "Expected error about {expected_substring}, got: {msg}"
    );
}

#[test]
fn test_sparse_moe_rejects_zero_num_experts() {
    assert_sparse_moe_rejects(|a| a.num_experts = 0, "num_experts");
}

#[test]
fn test_sparse_moe_rejects_zero_num_experts_per_tok() {
    assert_sparse_moe_rejects(|a| a.num_experts_per_tok = 0, "num_experts_per_tok");
}

#[test]
fn test_sparse_moe_happy_path_construction() {
    let args = minimal_qwen3_next_args();
    let result = SparseMoeBlock::new(&args, 0, 64, 4);
    assert!(result.is_ok());
    let block = result.unwrap();
    assert_eq!(block.top_k, args.num_experts_per_tok);
    assert!(block.norm_topk_prob);
}

#[test]
fn test_gather_qmm_basic() {
    // 2 experts, out=64, in=64 (dims must be multiples of 32 for quantize)
    let w_float = Array::ones::<f32>(&[2, 64, 64]).unwrap();
    let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

    // Input [1, 1, 1, 64], select expert 0
    let x = Array::ones::<f32>(&[1, 1, 1, 64]).unwrap();
    let indices = Array::from_slice(&[0_u32], &[1, 1, 1]);

    let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
    // Force evaluation to run the Metal kernel (MLX is lazy)
    result.eval().unwrap();
    // Output: [1, 1, 1, 1, 64] (batch broadcast with indices, M=1, N=64)
    assert_eq!(result.ndim(), 5);
    assert_eq!(*result.shape().last().unwrap(), 64);
}

#[test]
fn test_gather_qmm_multi_expert() {
    // 4 experts, out=64, in=64
    let w_float = Array::ones::<f32>(&[4, 64, 64]).unwrap();
    let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

    let x = Array::ones::<f32>(&[1, 1, 1, 64]).unwrap();
    let indices = Array::from_slice(&[0_u32, 2, 3], &[1, 1, 3]);

    let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
    result.eval().unwrap();
    // Output: [1, 1, 3, 1, 64] — 3 experts selected
    assert_eq!(*result.shape().get(2).unwrap(), 3);
}

#[test]
fn test_gather_qmm_matches_per_expert() {
    // Verify that gather_qmm produces the same result as the old
    // take_axis + quantized_matmul path for a single expert.
    let w_float = mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[4, 64, 64], None).unwrap();
    let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

    let x = mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[1, 64], None).unwrap();
    let expert_idx = Array::from_slice(&[2_u32], &[1]);

    // Old path: take_axis + quantized_matmul
    let ew = qw
        .take_axis(&expert_idx, 0)
        .unwrap()
        .squeeze_axes(&[0])
        .unwrap();
    let es = scales
        .take_axis(&expert_idx, 0)
        .unwrap()
        .squeeze_axes(&[0])
        .unwrap();
    let eb = biases
        .take_axis(&expert_idx, 0)
        .unwrap()
        .squeeze_axes(&[0])
        .unwrap();
    let old_result = quantized_forward(&x, &ew, &es, &eb, 64, 4).unwrap();

    // New path: gather_qmm
    let x_expanded = x.expand_dims(-2).unwrap(); // [1, 1, 64]
    let indices = Array::from_slice(&[2_u32], &[1, 1]);
    let new_result = gather_qmm(
        &x_expanded,
        &qw,
        &scales,
        &biases,
        &indices,
        true,
        64,
        4,
        false,
    )
    .unwrap()
    .squeeze_axes(&[-2])
    .unwrap()
    .squeeze_axes(&[-2])
    .unwrap();

    // Compare element-wise (both are quantized, should be exact match)
    let diff = old_result.subtract(&new_result).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max(None).unwrap().item_cast();
    assert!(
        max_diff < 1e-5,
        "gather_qmm and per-expert path differ by {max_diff}"
    );
}

#[test]
fn test_forward_gather_global_sort_shape() {
    // RED: forward_gather_global_sort should produce [B, L, top_k, D]
    let mut block = SwitchMlpWeights::new(64, 4).unwrap();

    let gate_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
    let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
    *block.gate_proj.weight = gw;
    *block.gate_proj.scales = gs;
    *block.gate_proj.biases = gb;

    let up_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
    let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
    *block.up_proj.weight = uw;
    *block.up_proj.scales = us;
    *block.up_proj.biases = ub;

    let down_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
    let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
    *block.down_proj.weight = dw;
    *block.down_proj.scales = ds;
    *block.down_proj.biases = db;

    // B=1, L=4, top_k=2 — enough tokens to exercise the sort path
    let x = Array::ones::<f32>(&[1, 4, 64]).unwrap();
    let indices = Array::from_slice(&[2u32, 0, 1, 3, 0, 2, 3, 1], &[1, 4, 2]);

    let result = block.forward_gather_global_sort(&x, &indices).unwrap();
    assert_eq!(result.shape(), &[1, 4, 2, 64]);
}

#[test]
fn test_forward_gather_global_sort_equivalence() {
    // RED: global sort must produce the same values as forward_gather
    let mut block = SwitchMlpWeights::new(64, 4).unwrap();

    let gate_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
    let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
    *block.gate_proj.weight = gw;
    *block.gate_proj.scales = gs;
    *block.gate_proj.biases = gb;

    let up_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
    let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
    *block.up_proj.weight = uw;
    *block.up_proj.scales = us;
    *block.up_proj.biases = ub;

    let down_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
    let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
    *block.down_proj.weight = dw;
    *block.down_proj.scales = ds;
    *block.down_proj.biases = db;

    let x = Array::ones::<f32>(&[1, 4, 64]).unwrap();
    let indices = Array::from_slice(&[2u32, 0, 1, 3, 0, 2, 3, 1], &[1, 4, 2]);

    let baseline = block.forward_gather(&x, &indices, false).unwrap();
    let sorted = block.forward_gather_global_sort(&x, &indices).unwrap();
    baseline.eval().unwrap();
    sorted.eval().unwrap();

    let diff = baseline.subtract(&sorted).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max(None).unwrap().item_cast();
    assert!(
        max_diff < 1e-5,
        "global sort and baseline differ by {max_diff}"
    );
}

#[test]
fn test_forward_gather_global_sort_random_weights() {
    // Harder: random weights + distinct per-token inputs + more experts
    // Verifies the sort/unsort cycle preserves per-token identity.
    let num_experts = 8;
    let hidden = 64;
    let top_k = 3;
    let b = 1;
    let l = 16;

    let mut block = SwitchMlpWeights::new(64, 4).unwrap();

    let gate_w =
        mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, hidden, hidden], None)
            .unwrap();
    let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
    *block.gate_proj.weight = gw;
    *block.gate_proj.scales = gs;
    *block.gate_proj.biases = gb;

    let up_w = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, hidden, hidden], None)
        .unwrap();
    let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
    *block.up_proj.weight = uw;
    *block.up_proj.scales = us;
    *block.up_proj.biases = ub;

    let down_w =
        mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, hidden, hidden], None)
            .unwrap();
    let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
    *block.down_proj.weight = dw;
    *block.down_proj.scales = ds;
    *block.down_proj.biases = db;

    // Random input — each token is distinct
    let x = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[b, l, hidden], None).unwrap();
    // Random expert indices in [0, num_experts)
    let idx_data: Vec<u32> = (0..(b * l * top_k) as u32)
        .map(|i| i % num_experts as u32)
        .collect();
    let indices = Array::from_slice(&idx_data, &[b, l, top_k]);
    x.eval().unwrap();
    indices.eval().unwrap();

    let baseline = block.forward_gather(&x, &indices, false).unwrap();
    let sorted = block.forward_gather_global_sort(&x, &indices).unwrap();
    baseline.eval().unwrap();
    sorted.eval().unwrap();

    assert_eq!(baseline.shape(), sorted.shape());
    assert_eq!(sorted.shape(), &[b, l, top_k, hidden]);

    let diff = baseline.subtract(&sorted).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max(None).unwrap().item_cast();
    assert!(
        max_diff < 1e-4,
        "random weights: global sort differs by {max_diff}"
    );
}

#[test]
fn test_moe_gate_up_fusion_parity() {
    // Fused gate+up (2 gather_qmm) must match unfused (3 gather_qmm).
    // Uses random weights + distinct per-token inputs to stress sort/unsort.
    let num_experts = 8;
    let hidden = 128;
    let intermediate = 64;
    let top_k = 3;
    let b = 1;
    let l = 16;

    let mut block = SwitchMlpWeights::new(64, 4).unwrap();

    let gate_w =
        mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, intermediate, hidden], None)
            .unwrap();
    let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
    *block.gate_proj.weight = gw;
    *block.gate_proj.scales = gs;
    *block.gate_proj.biases = gb;

    let up_w =
        mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, intermediate, hidden], None)
            .unwrap();
    let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
    *block.up_proj.weight = uw;
    *block.up_proj.scales = us;
    *block.up_proj.biases = ub;

    let down_w =
        mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[num_experts, hidden, intermediate], None)
            .unwrap();
    let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
    *block.down_proj.weight = dw;
    *block.down_proj.scales = ds;
    *block.down_proj.biases = db;

    let x = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[b, l, hidden], None).unwrap();
    let idx_data: Vec<u32> = (0..(b * l * top_k) as u32)
        .map(|i| i % num_experts as u32)
        .collect();
    let indices = Array::from_slice(&idx_data, &[b, l, top_k]);
    x.eval().unwrap();
    indices.eval().unwrap();

    // Reference: unfused 3-call path
    let reference = block.forward_gather_global_sort(&x, &indices).unwrap();
    // Fused: 2-call path
    let fused = block.forward_gather_fused(&x, &indices).unwrap();
    reference.eval().unwrap();
    fused.eval().unwrap();

    assert_eq!(reference.shape(), fused.shape());
    assert_eq!(fused.shape(), &[b, l, top_k, hidden]);

    let diff = reference.subtract(&fused).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max(None).unwrap().item_cast();
    assert!(
        max_diff < 1e-5,
        "fused gate+up differs from unfused by {max_diff}"
    );
}

#[test]
fn test_switch_mlp_forward_gather_shapes() {
    // Verify forward_gather produces the correct output shape with the
    // double expand_dims pattern matching Python's SwitchGLU.
    let mut block = SwitchMlpWeights::new(64, 4).unwrap();

    // 4 experts, intermediate=64, hidden=64
    let gate_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
    let (gw, gs, gb) = quantize_weights(&gate_w, 64, 4);
    *block.gate_proj.weight = gw;
    *block.gate_proj.scales = gs;
    *block.gate_proj.biases = gb;

    let up_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
    let (uw, us, ub) = quantize_weights(&up_w, 64, 4);
    *block.up_proj.weight = uw;
    *block.up_proj.scales = us;
    *block.up_proj.biases = ub;

    let down_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
    let (dw, ds, db) = quantize_weights(&down_w, 64, 4);
    *block.down_proj.weight = dw;
    *block.down_proj.scales = ds;
    *block.down_proj.biases = db;

    let x = Array::ones::<f32>(&[1, 1, 64]).unwrap();
    let indices = Array::from_slice(&[0_u32, 1, 2], &[1, 1, 3]);

    let result = block.forward_gather(&x, &indices, false).unwrap();
    // [B=1, L=1, top_k=3, D=64]
    assert_eq!(result.shape(), &[1, 1, 3, 64]);
}

#[test]
fn test_sparse_moe_forward_output_shape() {
    // Build a SparseMoeBlock with quantized dummy weights and verify the
    // full forward pass produces the correct output shape.
    let mut args = minimal_qwen3_next_args();
    args.num_experts = 4;
    args.num_experts_per_tok = 2;
    args.moe_intermediate_size = 64;
    args.shared_expert_intermediate_size = 64;
    args.hidden_size = 64;
    args.gate_quantization = Some(QuantizationConfig::new(64, 8));

    let mut block = SparseMoeBlock::new(&args, 0, 64, 4).unwrap();

    // Set router gate weights: [num_experts, hidden_size]
    let gate_w = Array::ones::<f32>(&[4, 64]).unwrap();
    let (gw, gs, gb) = quantize_weights(&gate_w, 64, 8);
    *block.gate.weight = gw;
    *block.gate.scales = gs;
    *block.gate.biases = gb;

    // Set switch_mlp expert weights: [4, intermediate, hidden] and [4, hidden, intermediate]
    let proj_w = Array::ones::<f32>(&[4, 64, 64]).unwrap();
    let (pw, ps, pb) = quantize_weights(&proj_w, 64, 4);
    for proj in [
        &mut block.switch_mlp.gate_proj,
        &mut block.switch_mlp.up_proj,
    ] {
        *proj.weight = pw.clone();
        *proj.scales = ps.clone();
        *proj.biases = pb.clone();
    }
    *block.switch_mlp.down_proj.weight = pw;
    *block.switch_mlp.down_proj.scales = ps;
    *block.switch_mlp.down_proj.biases = pb;

    // Set shared expert weights
    let shared_w = Array::ones::<f32>(&[64, 64]).unwrap();
    let (sw, ss, sb) = quantize_weights(&shared_w, 64, 4);
    for proj in [
        &mut block.shared_expert.gate_proj,
        &mut block.shared_expert.up_proj,
        &mut block.shared_expert.down_proj,
    ] {
        *proj.weight = sw.clone();
        *proj.scales = ss.clone();
        *proj.biases = sb.clone();
    }

    // Set shared expert gate weights
    let sgate_w = Array::ones::<f32>(&[1, 64]).unwrap();
    let (sgw, sgs, sgb) = quantize_weights(&sgate_w, 64, 8);
    *block.shared_expert_gate.weight = sgw;
    *block.shared_expert_gate.scales = sgs;
    *block.shared_expert_gate.biases = sgb;

    let x = Array::ones::<f32>(&[1, 1, 64]).unwrap();
    let result = block.forward(&x).unwrap();
    assert_eq!(result.shape(), &[1, 1, 64]);
}

#[test]
fn test_gather_qmm_model_scale() {
    // Reproduce actual Qwen3-Next-4bit shapes: 512 experts, hidden=2048,
    // intermediate=512, group_size=64, bits=4, top_k=10.
    // Use smaller dims to keep test fast but same expert count.
    let num_experts = 512;
    let hidden = 128; // Smaller than 2048 for test speed
    let intermediate = 64;

    let w_float =
        mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[num_experts, intermediate, hidden], None)
            .unwrap();
    let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

    // Decode shape: B=1, L=1, M=1
    let x = mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[1, 1, 1, hidden], None).unwrap();
    let indices = Array::from_slice(
        &[0_u32, 10, 50, 100, 200, 300, 400, 450, 500, 511],
        &[1, 1, 10],
    );

    let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
    // Force actual Metal kernel evaluation
    result.eval().unwrap();
    assert_eq!(result.shape(), &[1, 1, 10, 1, intermediate]);
}

#[test]
fn test_gather_qmm_prefill_broadcast() {
    // Prefill case: L > 1 requires the double expand_dims pattern.
    // x batch [B, L, 1] must broadcast with indices [B, L, top_k].
    let w_float = Array::ones::<f32>(&[8, 64, 64]).unwrap();
    let (qw, scales, biases) = quantize_weights(&w_float, 64, 4);

    // Prefill: B=1, L=9
    let x = Array::ones::<f32>(&[1, 9, 1, 1, 64]).unwrap(); // double expand
    let indices = Array::from_slice(
        &[0_u32, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 7],
        &[1, 9, 2],
    );

    let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
    result.eval().unwrap();
    // [1, 9, 2, 1, 64]: broadcast batch [1,9,1] with [1,9,2] -> [1,9,2], M=1, N=64
    assert_eq!(result.shape(), &[1, 9, 2, 1, 64]);
}

#[test]
fn test_gather_qmm_bfloat16() {
    // Model uses bfloat16 for scales/biases and input activations.
    // Verify gather_qmm works with bfloat16 dtypes.
    use mlx_rs::Dtype;

    let num_experts = 8;
    let hidden = 128;
    let intermediate = 64;

    let w_float =
        mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[num_experts, intermediate, hidden], None)
            .unwrap();
    let (qw, scales_f32, biases_f32) = quantize_weights(&w_float, 64, 4);

    // Convert scales/biases to bfloat16 (matching model file dtype)
    let scales = scales_f32.as_dtype(Dtype::Bfloat16).unwrap();
    let biases = biases_f32.as_dtype(Dtype::Bfloat16).unwrap();

    // Input in bfloat16
    let x_f32 = mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &[1, 1, 1, hidden], None).unwrap();
    let x = x_f32.as_dtype(Dtype::Bfloat16).unwrap();
    let indices = Array::from_slice(&[0_u32, 3, 7], &[1, 1, 3]);

    let result = gather_qmm(&x, &qw, &scales, &biases, &indices, true, 64, 4, false).unwrap();
    result.eval().unwrap();
    assert_eq!(result.shape(), &[1, 1, 3, 1, intermediate]);
}
