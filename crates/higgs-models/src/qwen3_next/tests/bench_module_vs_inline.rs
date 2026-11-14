//! Ignored benchmark comparing module and inline execution.

use crate::qwen3_next::{
    activations::swiglu,
    ffi::gather_qmm,
    linear::{QLinear, quantized_forward},
    mlp::Qwen3NextMLP,
    moe::{SparseMoeBlock, SwitchMlpWeights},
};
use mlx_rs::{
    Array, fast, nn,
    ops::{self, indexing::IndexOp},
};

/// Isolate what causes the module vs inline performance gap.
/// Tests three variants at 48 layers:
/// A) Module forward with multiply-by-zero attention (baseline slow path)
/// B) Inline forward with multiply-by-zero attention (tests if graph structure matters)
/// C) Inline forward with real quantized_matmul attention (original fast path)
/// D) Extract weights from modules into tuples, run inline (tests Param<Array> access)
#[test]
#[ignore = "benchmark, requires GPU"]
fn bench_module_vs_inline() {
    use mlx_rs::Dtype;
    use mlx_rs::module::Param;

    let d = 2048i32;
    let d_inter = 512i32;
    let n_experts = 512i32;
    let top_k = 10i32;
    let gs = 64i32;
    let bits = 4i32;
    let shared_inter = 512i32;

    let make_ql = |d_in: i32, d_out: i32, gs: i32, bits: i32| -> QLinear {
        let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let (w, s, b) = crate::test_support::quantize_affine(&raw, gs, bits).unwrap();
        QLinear {
            weight: Param::new(w),
            scales: Param::new(s),
            biases: Param::new(b),
            group_size: gs,
            bits,
        }
    };

    let make_switch_ql = |d_in: i32, d_out: i32| -> QLinear {
        let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let (w, s, b) = crate::test_support::quantize_affine(&raw, gs, bits).unwrap();
        QLinear {
            weight: Param::new(w),
            scales: Param::new(s),
            biases: Param::new(b),
            group_size: gs,
            bits,
        }
    };

    // Build 48 SparseMoeBlock instances with random weights
    let moe_blocks: Vec<SparseMoeBlock> = (0..48)
        .map(|_| SparseMoeBlock {
            gate: make_ql(d, n_experts, gs, bits),
            switch_mlp: SwitchMlpWeights {
                gate_proj: make_switch_ql(d, d_inter),
                up_proj: make_switch_ql(d, d_inter),
                down_proj: make_switch_ql(d_inter, d),
                fused_gate_up: None,
            },
            shared_expert: Qwen3NextMLP {
                gate_proj: make_ql(d, shared_inter * 2, gs, bits),
                up_proj: make_ql(d, shared_inter * 2, gs, bits),
                down_proj: make_ql(shared_inter * 2, d, gs, bits),
            },
            shared_expert_gate: make_ql(d, 1, gs, bits),
            top_k,
            norm_topk_prob: true,
        })
        .collect();

    // Eval all module weights
    {
        use mlx_rs::module::ModuleParameters;
        let mut all_w: Vec<&Array> = Vec::new();
        for moe in &moe_blocks {
            for (_, arr) in moe.parameters().flatten() {
                all_w.push(arr);
            }
        }
        mlx_rs::transforms::eval(all_w).unwrap();
    }

    // Extract module weights into bare tuples for variant D
    struct ExtractedWeights {
        gate: (Array, Array, Array),
        sw_gate: (Array, Array, Array),
        sw_up: (Array, Array, Array),
        sw_down: (Array, Array, Array),
        se_gate: (Array, Array, Array),
        se_up: (Array, Array, Array),
        se_down: (Array, Array, Array),
        se_gate_proj: (Array, Array, Array),
    }
    let extracted: Vec<ExtractedWeights> = moe_blocks
        .iter()
        .map(|moe| {
            // Clone the Array handles (cheap refcount bump, same underlying MLX data)
            ExtractedWeights {
                gate: (
                    moe.gate.weight.value.clone(),
                    moe.gate.scales.value.clone(),
                    moe.gate.biases.value.clone(),
                ),
                sw_gate: (
                    moe.switch_mlp.gate_proj.weight.value.clone(),
                    moe.switch_mlp.gate_proj.scales.value.clone(),
                    moe.switch_mlp.gate_proj.biases.value.clone(),
                ),
                sw_up: (
                    moe.switch_mlp.up_proj.weight.value.clone(),
                    moe.switch_mlp.up_proj.scales.value.clone(),
                    moe.switch_mlp.up_proj.biases.value.clone(),
                ),
                sw_down: (
                    moe.switch_mlp.down_proj.weight.value.clone(),
                    moe.switch_mlp.down_proj.scales.value.clone(),
                    moe.switch_mlp.down_proj.biases.value.clone(),
                ),
                se_gate: (
                    moe.shared_expert.gate_proj.weight.value.clone(),
                    moe.shared_expert.gate_proj.scales.value.clone(),
                    moe.shared_expert.gate_proj.biases.value.clone(),
                ),
                se_up: (
                    moe.shared_expert.up_proj.weight.value.clone(),
                    moe.shared_expert.up_proj.scales.value.clone(),
                    moe.shared_expert.up_proj.biases.value.clone(),
                ),
                se_down: (
                    moe.shared_expert.down_proj.weight.value.clone(),
                    moe.shared_expert.down_proj.scales.value.clone(),
                    moe.shared_expert.down_proj.biases.value.clone(),
                ),
                se_gate_proj: (
                    moe.shared_expert_gate.weight.value.clone(),
                    moe.shared_expert_gate.scales.value.clone(),
                    moe.shared_expert_gate.biases.value.clone(),
                ),
            }
        })
        .collect();

    let norm_w = Array::ones::<f32>(&[d])
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();
    let x = ops::ones_dtype(&[1, 1, d], Dtype::Float16).unwrap();
    mlx_rs::transforms::eval([&x, &norm_w]).unwrap();

    let n_layers = 48usize;
    let n = 20;

    // Helper: run N warmups then N timed evals
    let bench = |label: &str, forward: &dyn Fn(&Array) -> Array| {
        for _ in 0..5 {
            let y = forward(&x);
            mlx_rs::transforms::eval([&y]).unwrap();
        }
        let mut total = 0u128;
        for _ in 0..n {
            let y = forward(&x);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&y]).unwrap();
            total += t0.elapsed().as_nanos();
        }
        let ms = total as f64 / n as f64 / 1e6;
        eprintln!(
            "{label}: eval={ms:.2}ms per_layer={:.2}ms",
            ms / n_layers as f64
        );
    };

    // A) Module forward + multiply-by-zero attention
    bench("A) module+zero_attn", &|x: &Array| {
        let mut h = x.clone();
        for moe in moe_blocks.iter().take(n_layers) {
            let normed = fast::rms_norm(&h, Some(&norm_w), 1e-6).unwrap();
            let dummy_attn = normed.multiply(Array::from_f32(0.0)).unwrap();
            let h2 = h.add(dummy_attn).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&norm_w), 1e-6).unwrap();
            let mlp_out = moe.forward(&normed2).unwrap();
            h = h2.add(mlp_out).unwrap();
        }
        h
    });

    // B) Inline forward + multiply-by-zero attention (same extracted weights)
    bench("B) inline+zero_attn", &|x: &Array| {
        let mut h = x.clone();
        for l in extracted.iter().take(n_layers) {
            let normed = fast::rms_norm(&h, Some(&norm_w), 1e-6).unwrap();
            let dummy_attn = normed.multiply(Array::from_f32(0.0)).unwrap();
            let h2 = h.add(dummy_attn).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&norm_w), 1e-6).unwrap();

            // Inline MoE (same code as bench_simulated_forward)
            let gate_out =
                quantized_forward(&normed2, &l.gate.0, &l.gate.1, &l.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_start = n_experts - top_k;
            let top_inds = all_inds.index((.., .., top_start..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
            let scores = raw_scores.divide(score_sum).unwrap();

            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &l.sw_gate.0,
                &l.sw_gate.1,
                &l.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &l.sw_down.0,
                &l.sw_down.1,
                &l.sw_down.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let expert_sum = d_out
                .squeeze_axes(&[-2])
                .unwrap()
                .multiply(&scores.expand_dims(-1).unwrap())
                .unwrap()
                .sum_axes(&[-2], false)
                .unwrap();

            let sh_g =
                quantized_forward(&normed2, &l.se_gate.0, &l.se_gate.1, &l.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &l.se_down.0, &l.se_down.1, &l.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                &quantized_forward(
                    &normed2,
                    &l.se_gate_proj.0,
                    &l.se_gate_proj.1,
                    &l.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

            let mlp_out = expert_sum.add(shared_out).unwrap();
            h = h2.add(mlp_out).unwrap();
        }
        h
    });

    // C) Inline forward + real quantized_matmul for attention (per-layer attn weights)
    // This matches the bench_simulated_forward test structure
    let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
        let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        crate::test_support::quantize_affine(&raw, gs, bits).unwrap()
    };
    let attn_weights: Vec<(Array, Array, Array)> = (0..48).map(|_| make_qw(d, d)).collect();
    let per_layer_norms: Vec<Array> = (0..48)
        .map(|_| {
            Array::ones::<f32>(&[d])
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap()
        })
        .collect();
    {
        let mut all_w: Vec<&Array> = Vec::new();
        for (w, s, b) in &attn_weights {
            all_w.extend_from_slice(&[w, s, b]);
        }
        for nw in &per_layer_norms {
            all_w.push(nw);
        }
        mlx_rs::transforms::eval(all_w).unwrap();
    }

    bench("C) inline+real_attn+per_layer_norm", &|x: &Array| {
        let mut h = x.clone();
        for (i, l) in extracted.iter().take(n_layers).enumerate() {
            let normed = fast::rms_norm(&h, Some(&per_layer_norms[i]), 1e-6).unwrap();
            let attn_out = quantized_forward(
                &normed,
                &attn_weights[i].0,
                &attn_weights[i].1,
                &attn_weights[i].2,
                gs,
                bits,
            )
            .unwrap();
            let h2 = h.add(attn_out).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&per_layer_norms[i]), 1e-6).unwrap();

            let gate_out =
                quantized_forward(&normed2, &l.gate.0, &l.gate.1, &l.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_start = n_experts - top_k;
            let top_inds = all_inds.index((.., .., top_start..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
            let scores = raw_scores.divide(score_sum).unwrap();

            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &l.sw_gate.0,
                &l.sw_gate.1,
                &l.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &l.sw_down.0,
                &l.sw_down.1,
                &l.sw_down.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let expert_sum = d_out
                .squeeze_axes(&[-2])
                .unwrap()
                .multiply(&scores.expand_dims(-1).unwrap())
                .unwrap()
                .sum_axes(&[-2], false)
                .unwrap();

            let sh_g =
                quantized_forward(&normed2, &l.se_gate.0, &l.se_gate.1, &l.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &l.se_down.0, &l.se_down.1, &l.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                &quantized_forward(
                    &normed2,
                    &l.se_gate_proj.0,
                    &l.se_gate_proj.1,
                    &l.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

            let mlp_out = expert_sum.add(shared_out).unwrap();
            h = h2.add(mlp_out).unwrap();
        }
        h
    });

    // D) Inline + zero_attn + per_layer_norm (isolates norm_w sharing vs attn method)
    bench("D) inline+zero_attn+per_layer_norm", &|x: &Array| {
        let mut h = x.clone();
        for (i, l) in extracted.iter().take(n_layers).enumerate() {
            let normed = fast::rms_norm(&h, Some(&per_layer_norms[i]), 1e-6).unwrap();
            let dummy_attn = normed.multiply(Array::from_f32(0.0)).unwrap();
            let h2 = h.add(dummy_attn).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&per_layer_norms[i]), 1e-6).unwrap();

            let gate_out =
                quantized_forward(&normed2, &l.gate.0, &l.gate.1, &l.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_start = n_experts - top_k;
            let top_inds = all_inds.index((.., .., top_start..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
            let scores = raw_scores.divide(score_sum).unwrap();

            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &l.sw_gate.0,
                &l.sw_gate.1,
                &l.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &l.sw_down.0,
                &l.sw_down.1,
                &l.sw_down.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let expert_sum = d_out
                .squeeze_axes(&[-2])
                .unwrap()
                .multiply(&scores.expand_dims(-1).unwrap())
                .unwrap()
                .sum_axes(&[-2], false)
                .unwrap();

            let sh_g =
                quantized_forward(&normed2, &l.se_gate.0, &l.se_gate.1, &l.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &l.se_down.0, &l.se_down.1, &l.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                &quantized_forward(
                    &normed2,
                    &l.se_gate_proj.0,
                    &l.se_gate_proj.1,
                    &l.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

            let mlp_out = expert_sum.add(shared_out).unwrap();
            h = h2.add(mlp_out).unwrap();
        }
        h
    });

    // E) Inline + multiply-by-ONE + shared norm (is zero specifically the issue?)
    bench("E) inline+mul_one_attn", &|x: &Array| {
        let mut h = x.clone();
        for l in extracted.iter().take(n_layers) {
            let normed = fast::rms_norm(&h, Some(&norm_w), 1e-6).unwrap();
            let dummy_attn = normed.multiply(Array::from_f32(1.0)).unwrap();
            let h2 = h.add(dummy_attn).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&norm_w), 1e-6).unwrap();

            let gate_out =
                quantized_forward(&normed2, &l.gate.0, &l.gate.1, &l.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_start = n_experts - top_k;
            let top_inds = all_inds.index((.., .., top_start..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
            let scores = raw_scores.divide(score_sum).unwrap();

            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &l.sw_gate.0,
                &l.sw_gate.1,
                &l.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &l.sw_down.0,
                &l.sw_down.1,
                &l.sw_down.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let expert_sum = d_out
                .squeeze_axes(&[-2])
                .unwrap()
                .multiply(&scores.expand_dims(-1).unwrap())
                .unwrap()
                .sum_axes(&[-2], false)
                .unwrap();

            let sh_g =
                quantized_forward(&normed2, &l.se_gate.0, &l.se_gate.1, &l.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &l.se_down.0, &l.se_down.1, &l.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                &quantized_forward(
                    &normed2,
                    &l.se_gate_proj.0,
                    &l.se_gate_proj.1,
                    &l.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

            let mlp_out = expert_sum.add(shared_out).unwrap();
            h = h2.add(mlp_out).unwrap();
        }
        h
    });

    // F) Inline + zeros_like (skip normed entirely, just add zeros)
    bench("F) inline+zeros_like_attn", &|x: &Array| {
        let mut h = x.clone();
        for l in extracted.iter().take(n_layers) {
            let normed = fast::rms_norm(&h, Some(&norm_w), 1e-6).unwrap();
            let _ = &normed; // normed computed but not used for attn
            let dummy_attn = ops::zeros_like(&normed).unwrap();
            let h2 = h.add(dummy_attn).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&norm_w), 1e-6).unwrap();

            let gate_out =
                quantized_forward(&normed2, &l.gate.0, &l.gate.1, &l.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_start = n_experts - top_k;
            let top_inds = all_inds.index((.., .., top_start..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
            let scores = raw_scores.divide(score_sum).unwrap();

            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &l.sw_gate.0,
                &l.sw_gate.1,
                &l.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &l.sw_down.0,
                &l.sw_down.1,
                &l.sw_down.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let expert_sum = d_out
                .squeeze_axes(&[-2])
                .unwrap()
                .multiply(&scores.expand_dims(-1).unwrap())
                .unwrap()
                .sum_axes(&[-2], false)
                .unwrap();

            let sh_g =
                quantized_forward(&normed2, &l.se_gate.0, &l.se_gate.1, &l.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &l.se_down.0, &l.se_down.1, &l.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                &quantized_forward(
                    &normed2,
                    &l.se_gate_proj.0,
                    &l.se_gate_proj.1,
                    &l.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

            let mlp_out = expert_sum.add(shared_out).unwrap();
            h = h2.add(mlp_out).unwrap();
        }
        h
    });

    // G) Inline + skip normed entirely, h2 = h (no ops for attention)
    bench("G) inline+h2_equals_h", &|x: &Array| {
        let mut h = x.clone();
        for l in extracted.iter().take(n_layers) {
            // Skip first rms_norm entirely
            let h2 = h.clone();
            let normed2 = fast::rms_norm(&h2, Some(&norm_w), 1e-6).unwrap();

            let gate_out =
                quantized_forward(&normed2, &l.gate.0, &l.gate.1, &l.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_start = n_experts - top_k;
            let top_inds = all_inds.index((.., .., top_start..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let score_sum = raw_scores.sum_axes(&[-1], true).unwrap();
            let scores = raw_scores.divide(score_sum).unwrap();

            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &l.sw_gate.0,
                &l.sw_gate.1,
                &l.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &l.sw_up.0, &l.sw_up.1, &l.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &l.sw_down.0,
                &l.sw_down.1,
                &l.sw_down.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let expert_sum = d_out
                .squeeze_axes(&[-2])
                .unwrap()
                .multiply(&scores.expand_dims(-1).unwrap())
                .unwrap()
                .sum_axes(&[-2], false)
                .unwrap();

            let sh_g =
                quantized_forward(&normed2, &l.se_gate.0, &l.se_gate.1, &l.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &l.se_up.0, &l.se_up.1, &l.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &l.se_down.0, &l.se_down.1, &l.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                &quantized_forward(
                    &normed2,
                    &l.se_gate_proj.0,
                    &l.se_gate_proj.1,
                    &l.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(&sh_gate_val).unwrap();

            let mlp_out = expert_sum.add(shared_out).unwrap();
            h = h2.add(mlp_out).unwrap();
        }
        h
    });
}
