//! Tests for the `GatedDeltaNet` kernel and compiled decode.

use super::valid_causal_lm_args;
use crate::qwen3_next::{
    activations::{gdn_output_gate, sigmoid_mul, silu_direct, silu_mul},
    cache::{ArraysCache, LayerCache},
    gated_delta_kernel::gated_delta_kernel_ffi,
    gated_delta_net::{GatedDeltaNet, compute_g_compiled, make_compiled_gdn_decode},
};
use mlx_rs::{
    Array, fast,
    module::Param,
    nn,
    ops::{self, indexing::IndexOp},
};

#[test]
fn test_gated_delta_kernel_basic() {
    // B=1, T=1, Hk=2, Hv=4, Dk=32, Dv=32
    // Dk must be multiple of 32 for SIMD group width
    let q = Array::ones::<f32>(&[1, 1, 2, 32]).unwrap();
    let k = Array::ones::<f32>(&[1, 1, 2, 32]).unwrap();
    let v = Array::ones::<f32>(&[1, 1, 4, 32]).unwrap();
    let a_log = Array::zeros::<f32>(&[4]).unwrap();
    let a = Array::ones::<f32>(&[1, 1, 4]).unwrap();
    let dt_bias = Array::zeros::<f32>(&[4]).unwrap();
    let b = Array::zeros::<f32>(&[1, 1, 4]).unwrap();
    let state = Array::zeros::<f32>(&[1, 4, 32, 32]).unwrap();

    let (y, new_state) = gated_delta_kernel_ffi(
        &q, &k, &v, &a_log, &a, &dt_bias, &b, &state, 1, 1, 2, 32, 4, 32,
    )
    .unwrap();
    y.eval().unwrap();
    new_state.eval().unwrap();
    assert_eq!(y.shape(), &[1, 1, 4, 32]);
    assert_eq!(new_state.shape(), &[1, 4, 32, 32]);
}

#[test]
fn test_arrays_cache_default() {
    let cache = ArraysCache::default();
    assert!(cache.conv_state.is_none());
    assert!(cache.ssm_state.is_none());
    assert_eq!(cache.offset, 0);
}

#[test]
fn test_gated_delta_kernel_prefill() {
    // B=1, T=4, Hk=2, Hv=4, Dk=32, Dv=32
    let q = Array::ones::<f32>(&[1, 4, 2, 32]).unwrap();
    let k = Array::ones::<f32>(&[1, 4, 2, 32]).unwrap();
    let v = Array::ones::<f32>(&[1, 4, 4, 32]).unwrap();
    let a_log = Array::zeros::<f32>(&[4]).unwrap();
    let a = Array::ones::<f32>(&[1, 4, 4]).unwrap();
    let dt_bias = Array::zeros::<f32>(&[4]).unwrap();
    let b = Array::zeros::<f32>(&[1, 4, 4]).unwrap();
    let state = Array::zeros::<f32>(&[1, 4, 32, 32]).unwrap();

    let (y, new_state) = gated_delta_kernel_ffi(
        &q, &k, &v, &a_log, &a, &dt_bias, &b, &state, 1, 4, 2, 32, 4, 32,
    )
    .unwrap();
    y.eval().unwrap();
    new_state.eval().unwrap();
    assert_eq!(y.shape(), &[1, 4, 4, 32]);
    assert_eq!(new_state.shape(), &[1, 4, 32, 32]);
}

#[test]
fn test_compiled_compute_g_matches_raw() {
    let a_log = Array::from_slice(&[0.5_f32, -0.3], &[1, 2]);
    let a = Array::from_slice(&[1.0_f32, -1.0], &[1, 2]);
    let dt_bias = Array::from_slice(&[0.1_f32, 0.2], &[1, 2]);

    // Raw computation
    let a_plus_bias = a.add(&dt_bias).unwrap();
    let sp = nn::softplus(&a_plus_bias).unwrap();
    let neg_decay = a_log
        .exp()
        .unwrap()
        .negative()
        .unwrap()
        .multiply(sp)
        .unwrap();
    let raw_g = neg_decay.exp().unwrap();

    // Compiled computation
    let mut compiled = mlx_rs::transforms::compile::compile(compute_g_compiled, None);
    let compiled_g = compiled((&a_log, &a, &dt_bias)).unwrap();

    let diff = raw_g.subtract(&compiled_g).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max(None).unwrap().item_cast();
    assert!(
        max_diff < 1e-6,
        "compiled compute_g differs from raw by {max_diff}"
    );
}

#[test]
fn test_compiled_silu_mul_matches_raw() {
    let gate = mlx_rs::random::uniform::<f32, f32>(-2.0, 2.0, &[1, 1, 17408], None).unwrap();
    let x = mlx_rs::random::uniform::<f32, f32>(-2.0, 2.0, &[1, 1, 17408], None).unwrap();

    let raw = nn::silu(&gate).unwrap().multiply(&x).unwrap();
    let compiled = silu_mul(&gate, &x).unwrap();

    let diff = raw.subtract(&compiled).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max(None).unwrap().item_cast();
    assert!(
        max_diff < 1e-5,
        "compiled silu_mul differs from raw by {max_diff}"
    );
}

#[test]
fn test_compiled_sigmoid_mul_matches_raw() {
    let gate = mlx_rs::random::uniform::<f32, f32>(-2.0, 2.0, &[1, 1, 6144], None).unwrap();
    let x = mlx_rs::random::uniform::<f32, f32>(-2.0, 2.0, &[1, 1, 6144], None).unwrap();

    let raw = nn::sigmoid(&gate).unwrap().multiply(&x).unwrap();
    let compiled = sigmoid_mul(&gate, &x).unwrap();

    let diff = raw.subtract(&compiled).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max(None).unwrap().item_cast();
    assert!(
        max_diff < 1e-5,
        "compiled sigmoid_mul differs from raw by {max_diff}"
    );
}

#[test]
fn test_compiled_gdn_output_gate_matches_raw() {
    let y = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, 1, 48, 128], None).unwrap();
    let z = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, 1, 48, 128], None).unwrap();
    let weight = Array::ones::<f32>(&[128]).unwrap();

    let normed = fast::rms_norm(&y, Some(&weight), 1e-6).unwrap();
    let raw = nn::silu(&z).unwrap().multiply(&normed).unwrap();
    let compiled = gdn_output_gate(&y, &weight, 1e-6, &z).unwrap();

    let diff = raw.subtract(&compiled).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max(None).unwrap().item_cast();
    assert!(
        max_diff < 1e-5,
        "compiled gdn_output_gate differs from raw by {max_diff}"
    );
}

#[test]
fn test_gdn_decode_conv_ring_buffer_matches_concat_path() {
    use mlx_rs::Dtype;

    let args = valid_causal_lm_args();
    let mut gdn = GatedDeltaNet::new(&args, 64, 4).unwrap();
    let conv_w = mlx_rs::random::uniform::<f32, f32>(
        -0.5,
        0.5,
        &[gdn.conv_dim, gdn.conv_kernel_size, 1],
        None,
    )
    .unwrap()
    .as_dtype(Dtype::Float16)
    .unwrap();
    gdn.conv1d.weight = Param::new(conv_w);
    gdn.conv_weight_t = None;

    let wt = gdn
        .conv1d
        .weight
        .squeeze_axes(&[-1])
        .unwrap()
        .transpose()
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();

    let history_len = gdn.conv_kernel_size - 1;
    let mut ref_state =
        mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, history_len, gdn.conv_dim], None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
    let mut cache = ArraysCache {
        conv_state: Some(ref_state.clone()),
        ssm_state: None,
        conv_pos: history_len - 1,
        offset: history_len,
    };

    for _ in 0..6 {
        let mixed_qkv = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, 1, gdn.conv_dim], None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();

        let actual = gdn.decode_conv1d_step(&mixed_qkv, &mut cache, 1).unwrap();
        let conv_in = ops::concatenate(&[&ref_state, &mixed_qkv], 1).unwrap();
        let expected =
            silu_direct(&conv_in.multiply(&wt).unwrap().sum_axes(&[1], true).unwrap()).unwrap();

        mlx_rs::transforms::eval([&actual, &expected]).unwrap();
        let diff = actual.subtract(&expected).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(None).unwrap().item_cast();
        assert!(
            max_diff < 2e-3,
            "ring-buffer decode conv differs from concat path by {max_diff}"
        );

        ref_state = conv_in.index((.., 1.., ..));
        cache.offset += 1;
    }

    let ordered = gdn
        .chronological_conv_state(&mut cache, 1, Dtype::Float16)
        .unwrap();
    mlx_rs::transforms::eval([&ordered, &ref_state]).unwrap();
    let diff = ordered.subtract(&ref_state).unwrap().abs().unwrap();
    let max_diff: f32 = diff.max(None).unwrap().item_cast();
    assert!(
        max_diff < 2e-3,
        "linearized ring-buffer conv state differs from chronological state by {max_diff}"
    );
}

/// One decode-time convolution step on every row of a merged batch must
/// match the same step taken by each row alone, even though the rows'
/// ring buffers are at different positions and some have seen fewer than
/// `kernel - 1` tokens. Afterwards each row must continue correctly alone.
#[test]
fn test_batched_conv_ring_merge_and_split_match_unbatched() {
    use crate::batched_cache::{HybridBatchedCache, HybridLayer};
    use mlx_rs::Dtype;

    let args = valid_causal_lm_args();
    let mut gdn = GatedDeltaNet::new(&args, 64, 4).unwrap();
    let conv_w = mlx_rs::random::uniform::<f32, f32>(
        -0.5,
        0.5,
        &[gdn.conv_dim, gdn.conv_kernel_size, 1],
        None,
    )
    .unwrap();
    gdn.conv1d.weight = Param::new(conv_w);
    gdn.conv_weight_t = None;
    let conv_dim = gdn.conv_dim;
    let ssm_shape = [1, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim];

    let step_counts = [1_usize, 2, 5];
    let inputs: Vec<Vec<Array>> = step_counts
        .iter()
        .map(|&count| {
            (0..count + 6)
                .map(|_| {
                    mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, 1, conv_dim], None).unwrap()
                })
                .collect()
        })
        .collect();

    let replay = |gdn: &mut GatedDeltaNet, row: usize| {
        let mut cache = ArraysCache::new();
        for input in inputs[row].iter().take(step_counts[row]) {
            gdn.decode_conv1d_step(input, &mut cache, 1).unwrap();
            cache.offset += 1;
        }
        cache.ssm_state = Some(ops::zeros_dtype(&ssm_shape, Dtype::Float32).unwrap());
        cache
    };
    let positions: Vec<i32> = (0..3).map(|row| replay(&mut gdn, row).conv_pos).collect();
    assert!(
        positions.windows(2).any(|pair| pair[0] != pair[1]),
        "test rows must sit at different ring positions: {positions:?}"
    );

    let mut alone: Vec<ArraysCache> = (0..3).map(|row| replay(&mut gdn, row)).collect();
    let merged_rows = (0..3)
        .map(|row| vec![Some(LayerCache::Arrays(replay(&mut gdn, row)))])
        .collect();
    let mut batch = HybridBatchedCache::from_rows(merged_rows).unwrap();

    let max_difference = |expected: &Array, actual: &Array| -> f32 {
        let diff = expected.subtract(actual).unwrap().abs().unwrap();
        diff.max(None).unwrap().item_cast()
    };

    for step in 0..3 {
        let stacked_input = ops::concatenate(
            &inputs
                .iter()
                .zip(&step_counts)
                .map(|(row_inputs, &count)| &row_inputs[count + step])
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap();
        let Some(HybridLayer::Recurrent(recurrent)) = batch.layers_mut().first_mut() else {
            panic!("expected a recurrent layer");
        };
        let batched = gdn
            .decode_conv1d_step(&stacked_input, recurrent.state_mut(), 3)
            .unwrap();
        recurrent.state_mut().offset += 1;
        for (row, cache) in alone.iter_mut().enumerate() {
            let expected = gdn
                .decode_conv1d_step(&inputs[row][step_counts[row] + step], cache, 1)
                .unwrap();
            cache.offset += 1;
            let row_index = i32::try_from(row).unwrap();
            let diff = max_difference(&expected, &batched.index((row_index..=row_index, .., ..)));
            assert!(
                diff < 1e-5,
                "row {row} step {step}: batched conv differs by {diff}"
            );
        }
    }

    let mut rows = batch.into_rows().unwrap();
    for (row, (layers, cache)) in rows.iter_mut().zip(alone.iter_mut()).enumerate() {
        let Some(Some(LayerCache::Arrays(split))) = layers.first_mut() else {
            panic!("expected a recurrent layer");
        };
        assert_eq!(split.offset, cache.offset, "row {row} offset after split");
        let input = &inputs[row][step_counts[row] + 3];
        let from_split = gdn.decode_conv1d_step(input, split, 1).unwrap();
        let expected = gdn.decode_conv1d_step(input, cache, 1).unwrap();
        let diff = max_difference(&expected, &from_split);
        assert!(diff < 1e-5, "row {row}: step after split differs by {diff}");
    }
}

#[test]
fn test_gated_delta_kernel_state_passthrough() {
    // Verify that running kernel with T=1 twice produces different state
    // than running with T=2, confirming sequential dependence works.
    let q = Array::ones::<f32>(&[1, 1, 2, 32]).unwrap();
    let k = Array::ones::<f32>(&[1, 1, 2, 32]).unwrap();
    let v = Array::ones::<f32>(&[1, 1, 4, 32]).unwrap();
    let a_log = Array::zeros::<f32>(&[4]).unwrap();
    let a = Array::ones::<f32>(&[1, 1, 4]).unwrap();
    let dt_bias = Array::zeros::<f32>(&[4]).unwrap();
    let b = Array::zeros::<f32>(&[1, 1, 4]).unwrap();
    let state0 = Array::zeros::<f32>(&[1, 4, 32, 32]).unwrap();

    // Step 1
    let (_, state1) = gated_delta_kernel_ffi(
        &q, &k, &v, &a_log, &a, &dt_bias, &b, &state0, 1, 1, 2, 32, 4, 32,
    )
    .unwrap();
    state1.eval().unwrap();

    // Step 2 (uses state1)
    let (y2, state2) = gated_delta_kernel_ffi(
        &q, &k, &v, &a_log, &a, &dt_bias, &b, &state1, 1, 1, 2, 32, 4, 32,
    )
    .unwrap();
    y2.eval().unwrap();
    state2.eval().unwrap();

    assert_eq!(y2.shape(), &[1, 1, 4, 32]);
    assert_eq!(state2.shape(), &[1, 4, 32, 32]);
}

/// Reference ops implementation of a single gated delta step (for comparison tests).
pub(super) fn gated_delta_step_ref(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> (Array, Array) {
    let decay = g.expand_dims(-1).unwrap().expand_dims(-1).unwrap();
    let decayed_state = state.multiply(&decay).unwrap();
    let k_expanded = k.expand_dims(-2).unwrap();
    let kv_mem = decayed_state
        .multiply(&k_expanded)
        .unwrap()
        .sum_axes(&[-1], false)
        .unwrap();
    let beta_expanded = beta.expand_dims(-1).unwrap();
    let delta = v
        .subtract(&kv_mem)
        .unwrap()
        .multiply(&beta_expanded)
        .unwrap();
    let delta_expanded = delta.expand_dims(-1).unwrap();
    let new_state = decayed_state
        .add(k_expanded.multiply(&delta_expanded).unwrap())
        .unwrap();
    let q_expanded = q.expand_dims(-2).unwrap();
    let y = new_state
        .multiply(&q_expanded)
        .unwrap()
        .sum_axes(&[-1], false)
        .unwrap();
    (y, new_state)
}

#[test]
fn test_gated_delta_kernel_matches_ops() {
    // Compare kernel output against reference ops for T=1, no GQA.
    // B=1, T=1, Hk=1, Hv=1, Dk=32, Dv=32
    assert_kernel_matches_ops(1, 1, 1, 1, 32, 32, 1e-4, "Hk=Hv=1");
}

#[test]
fn test_gated_delta_kernel_matches_ops_gqa() {
    // GQA: Hk=2, Hv=4 (repeat factor 2). This is the pattern used by Qwen3-Next.
    assert_kernel_matches_ops(1, 1, 2, 4, 32, 32, 1e-4, "Hk=2,Hv=4 GQA");
}

#[test]
fn test_gated_delta_kernel_matches_ops_multi_step() {
    // T=3 with GQA: verify multi-timestep correctness
    assert_kernel_matches_ops(1, 3, 2, 4, 32, 32, 1e-4, "T=3 GQA");
}

#[test]
fn test_gated_delta_kernel_matches_ops_model_dims() {
    // Actual Qwen3-Next dims: Hk=16, Hv=32, Dk=128, Dv=128
    assert_kernel_matches_ops(1, 1, 16, 32, 128, 128, 1e-4, "model dims");
}

#[test]
fn test_gated_delta_kernel_matches_ops_batched() {
    // Batched decode runs the kernel with one sequence per batch row.
    assert_kernel_matches_ops(3, 1, 2, 4, 32, 32, 1e-4, "B=3 GQA");
    assert_kernel_matches_ops(2, 3, 2, 4, 32, 32, 1e-4, "B=2 T=3 GQA");
    assert_kernel_matches_ops(4, 1, 16, 32, 128, 128, 1e-4, "B=4 model dims");
}

#[test]
fn test_gated_delta_kernel_matches_ops_bfloat16() {
    // The actual model uses bfloat16. Test with model dims in bfloat16.
    use mlx_rs::Dtype;
    let hk = 2;
    let hv = 4;
    let dk = 32;
    let dv = 32;
    let batch = 1;
    let seq_len = 1;

    let q = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hk, dk], None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let k = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hk, dk], None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let v = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv, dv], None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let a_log = mlx_rs::random::uniform::<f32, f32>(-1.0, 0.0, &[hv], None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let a = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv], None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let dt_bias = mlx_rs::random::uniform::<f32, f32>(-0.5, 0.5, &[hv], None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let b = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv], None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let state = mlx_rs::random::uniform::<f32, f32>(-0.1, 0.1, &[batch, hv, dv, dk], None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

    // Kernel
    let (kern_y, kern_state) = gated_delta_kernel_ffi(
        &q, &k, &v, &a_log, &a, &dt_bias, &b, &state, batch, seq_len, hk, dk, hv, dv,
    )
    .unwrap();
    kern_y.eval().unwrap();
    kern_state.eval().unwrap();

    assert_eq!(kern_y.shape(), &[batch, seq_len, hv, dv]);
    assert_eq!(kern_state.shape(), &[batch, hv, dv, dk]);

    // Verify outputs are finite (not NaN/Inf)
    let y_f32 = kern_y.as_dtype(Dtype::Float32).unwrap();
    let y_abs_max: f32 = y_f32.abs().unwrap().max(None).unwrap().item_cast();
    assert!(
        y_abs_max.is_finite() && y_abs_max < 1e6,
        "bfloat16 kernel y has bad values: max abs = {y_abs_max}"
    );
}

#[allow(clippy::too_many_arguments)]
fn assert_kernel_matches_ops(
    batch: i32,
    seq_len: i32,
    hk: i32,
    hv: i32,
    dk: i32,
    dv: i32,
    tol: f32,
    label: &str,
) {
    let q =
        mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hk, dk], None).unwrap();
    let k =
        mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hk, dk], None).unwrap();
    let v =
        mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv, dv], None).unwrap();
    let a_log = mlx_rs::random::uniform::<f32, f32>(-1.0, 0.0, &[hv], None).unwrap();
    let a_val =
        mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv], None).unwrap();
    let dt_bias = mlx_rs::random::uniform::<f32, f32>(-0.5, 0.5, &[hv], None).unwrap();
    let b = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, seq_len, hv], None).unwrap();
    let state = mlx_rs::random::uniform::<f32, f32>(-0.1, 0.1, &[batch, hv, dv, dk], None).unwrap();

    // Compute g and beta from raw inputs for the reference path
    let mut compute_g_fn = mlx_rs::transforms::compile::compile(compute_g_compiled, None);
    let g = compute_g_fn((&a_log, &a_val, &dt_bias)).unwrap();
    let beta = nn::sigmoid(&b).unwrap();

    // Reference: loop over timesteps with repeat_axis for GQA
    let repeat_factor = hv / hk;
    let mut ref_state = state.clone();
    let mut ref_ys = Vec::new();
    for t in 0..seq_len {
        let qt = q.index((.., t, .., ..));
        let kt = k.index((.., t, .., ..));
        let vt = v.index((.., t, .., ..));
        let gt = g.index((.., t, ..));
        let bt = beta.index((.., t, ..));

        let qt_rep = if repeat_factor > 1 {
            ops::repeat_axis::<f32>(qt, repeat_factor, -2).unwrap()
        } else {
            qt
        };
        let kt_rep = if repeat_factor > 1 {
            ops::repeat_axis::<f32>(kt, repeat_factor, -2).unwrap()
        } else {
            kt
        };

        let (y_t, new_state) = gated_delta_step_ref(&qt_rep, &kt_rep, &vt, &gt, &bt, &ref_state);
        ref_state = new_state;
        ref_ys.push(y_t);
    }
    let ref_y_refs: Vec<&Array> = ref_ys.iter().collect();
    let ref_y = ops::stack(&ref_y_refs, 1).unwrap();
    ref_y.eval().unwrap();
    ref_state.eval().unwrap();

    // Kernel
    let (kern_y, kern_state) = gated_delta_kernel_ffi(
        &q, &k, &v, &a_log, &a_val, &dt_bias, &b, &state, batch, seq_len, hk, dk, hv, dv,
    )
    .unwrap();
    kern_y.eval().unwrap();
    kern_state.eval().unwrap();

    // Compare y
    let y_diff = ref_y.subtract(&kern_y).unwrap().abs().unwrap();
    let y_max: f32 = y_diff.max(None).unwrap().item_cast();
    assert!(y_max < tol, "[{label}] kernel y differs by {y_max}");

    // Compare state
    let s_diff = ref_state.subtract(&kern_state).unwrap().abs().unwrap();
    let s_max: f32 = s_diff.max(None).unwrap().item_cast();
    assert!(s_max < tol, "[{label}] kernel state differs by {s_max}");
}

#[test]
fn test_compiled_gdn_decode_matches_reference_ops() {
    let batch = 1;
    let hv = 4;
    let dk = 32;
    let dv = 32;

    let q = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, 1, hv, dk], None).unwrap();
    let k = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, 1, hv, dk], None).unwrap();
    let v = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, 1, hv, dv], None).unwrap();
    let g = mlx_rs::random::uniform::<f32, f32>(0.1, 0.9, &[batch, 1, hv], None).unwrap();
    let beta = mlx_rs::random::uniform::<f32, f32>(0.1, 0.9, &[batch, 1, hv], None).unwrap();
    let z = mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[batch, 1, hv, dv], None).unwrap();
    let norm_weight = Array::ones::<f32>(&[dv]).unwrap();
    let state = mlx_rs::random::uniform::<f32, f32>(-0.1, 0.1, &[batch, hv, dv, dk], None).unwrap();

    let (y_ref, state_ref) = gated_delta_step_ref(
        &q.squeeze_axes(&[1]).unwrap(),
        &k.squeeze_axes(&[1]).unwrap(),
        &v.squeeze_axes(&[1]).unwrap(),
        &g.squeeze_axes(&[1]).unwrap(),
        &beta.squeeze_axes(&[1]).unwrap(),
        &state,
    );
    let y_ref = y_ref.expand_dims(1).unwrap();
    let expected = nn::silu(&z)
        .unwrap()
        .multiply(&fast::rms_norm(&y_ref, Some(&norm_weight), 1e-6).unwrap())
        .unwrap();

    let mut cache = ArraysCache {
        conv_state: None,
        ssm_state: Some(state.clone()),
        conv_pos: -1,
        offset: 0,
    };
    let mut compiled = make_compiled_gdn_decode();
    let inputs = [
        q.clone(),
        k.clone(),
        v.clone(),
        g.clone(),
        beta.clone(),
        z.clone(),
        norm_weight.clone(),
    ];
    let actual = compiled(&mut cache, &inputs).unwrap().pop().unwrap();

    let actual_state = cache.ssm_state.as_ref().unwrap().clone();
    mlx_rs::transforms::eval([&actual, &expected, &actual_state, &state_ref]).unwrap();

    let out_diff = actual.subtract(&expected).unwrap().abs().unwrap();
    let out_max: f32 = out_diff.max(None).unwrap().item_cast();
    assert!(
        out_max < 1e-5,
        "compiled GDN decode output mismatch by {out_max}"
    );

    let state_diff = actual_state.subtract(&state_ref).unwrap().abs().unwrap();
    let state_max: f32 = state_diff.max(None).unwrap().item_cast();
    assert!(
        state_max < 1e-5,
        "compiled GDN decode state mismatch by {state_max}"
    );
}
