//! Ignored benchmarks for `gather_qmm` and MoE kernels.

use crate::qwen3_next::{
    activations::swiglu, ffi::gather_qmm, gated_delta_kernel::gated_delta_kernel_ffi,
    linear::quantized_forward,
};
use mlx_rs::{
    Array, fast, nn,
    ops::{self, indexing::IndexOp},
};

/// Benchmark: chain 48 layers of 3x gather_qmm + SwiGLU, single eval.
/// Compare with Python's 0.378ms (48 layers, single eval).
#[test]
#[ignore = "benchmark, requires GPU"]
fn bench_gather_qmm_chain() {
    let num_experts = 512;
    let d = 2048;
    let intermediate = 512;
    let top_k = 10;

    // Create quantized expert weights (same as model)
    let gate_w = Array::zeros::<u32>(&[num_experts, intermediate, d * 4 / 32]).unwrap();
    let gate_s = Array::ones::<f32>(&[num_experts, intermediate, d / 64]).unwrap();
    let gate_b = Array::zeros::<f32>(&[num_experts, intermediate, d / 64]).unwrap();

    let up_w = Array::zeros::<u32>(&[num_experts, intermediate, d * 4 / 32]).unwrap();
    let up_s = Array::ones::<f32>(&[num_experts, intermediate, d / 64]).unwrap();
    let up_b = Array::zeros::<f32>(&[num_experts, intermediate, d / 64]).unwrap();

    let down_w = Array::zeros::<u32>(&[num_experts, d, intermediate * 4 / 32]).unwrap();
    let down_s = Array::ones::<f32>(&[num_experts, d, intermediate / 64]).unwrap();
    let down_b = Array::zeros::<f32>(&[num_experts, d, intermediate / 64]).unwrap();

    let x = Array::ones::<f32>(&[1, 1, 1, 1, d]).unwrap();
    let indices = Array::from_slice(&[0_i32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, top_k]);
    mlx_rs::transforms::eval([
        &gate_w, &gate_s, &gate_b, &up_w, &up_s, &up_b, &down_w, &down_s, &down_b, &x, &indices,
    ])
    .unwrap();

    // Warm up
    for _ in 0..3 {
        let mut y = x.clone();
        for _ in 0..48 {
            let g =
                gather_qmm(&y, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false).unwrap();
            let u = gather_qmm(&y, &up_w, &up_s, &up_b, &indices, true, 64, 4, false).unwrap();
            let activated = swiglu(&g, &u).unwrap();
            y = gather_qmm(
                &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
            )
            .unwrap();
        }
        mlx_rs::transforms::eval([&y]).unwrap();
    }

    // Benchmark: 48 layers, single eval -- split graph build vs eval
    let n = 50;
    let mut total_build_ns = 0u128;
    let mut total_eval_ns = 0u128;
    for _ in 0..n {
        let t0 = std::time::Instant::now();
        let mut y = x.clone();
        for _ in 0..48 {
            let g =
                gather_qmm(&y, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false).unwrap();
            let u = gather_qmm(&y, &up_w, &up_s, &up_b, &indices, true, 64, 4, false).unwrap();
            let activated = swiglu(&g, &u).unwrap();
            y = gather_qmm(
                &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
            )
            .unwrap();
        }
        let t1 = std::time::Instant::now();
        mlx_rs::transforms::eval([&y]).unwrap();
        let t2 = std::time::Instant::now();
        total_build_ns += (t1 - t0).as_nanos();
        total_eval_ns += (t2 - t1).as_nanos();
    }
    let build_ms = total_build_ns as f64 / n as f64 / 1_000_000.0;
    let eval_ms = total_eval_ns as f64 / n as f64 / 1_000_000.0;
    eprintln!(
        "48 layers * 3 gather_qmm + SwiGLU: build={build_ms:.2}ms eval={eval_ms:.2}ms total={:.2}ms",
        build_ms + eval_ms
    );

    // Also test with mlx-rs ops::add chain (no FFI gather_qmm)
    let n3 = 50;
    let x_simple = Array::ones::<f32>(&[1, 1, d]).unwrap();
    mlx_rs::transforms::eval([&x_simple]).unwrap();
    let mut total_simple_ns = 0u128;
    for _ in 0..n3 {
        let t0 = std::time::Instant::now();
        let mut y2 = x_simple.clone();
        for _ in 0..(48 * 5) {
            y2 = y2.add(&x_simple).unwrap();
        }
        mlx_rs::transforms::eval([&y2]).unwrap();
        total_simple_ns += t0.elapsed().as_nanos();
    }
    let simple_ms = total_simple_ns as f64 / n3 as f64 / 1_000_000.0;
    eprintln!("240 chained adds (single eval): {simple_ms:.2}ms");

    // Test with the shared gather_qmm wrapper
    let n4 = 50;
    let mut total_builtin_build = 0u128;
    let mut total_builtin_eval = 0u128;
    for _ in 0..n4 {
        let t0 = std::time::Instant::now();
        let mut y3 = x.clone();
        for _ in 0..48 {
            let g =
                gather_qmm(&y3, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false).unwrap();
            let u = gather_qmm(&y3, &up_w, &up_s, &up_b, &indices, true, 64, 4, false).unwrap();
            let activated = swiglu(&g, &u).unwrap();
            y3 = gather_qmm(
                &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
            )
            .unwrap();
        }
        let t1 = std::time::Instant::now();
        mlx_rs::transforms::eval([&y3]).unwrap();
        let t2 = std::time::Instant::now();
        total_builtin_build += (t1 - t0).as_nanos();
        total_builtin_eval += (t2 - t1).as_nanos();
    }
    let builtin_build = total_builtin_build as f64 / n4 as f64 / 1_000_000.0;
    let builtin_eval = total_builtin_eval as f64 / n4 as f64 / 1_000_000.0;
    eprintln!(
        "48 layers mlx-rs gather_qmm: build={builtin_build:.2}ms eval={builtin_eval:.2}ms total={:.2}ms",
        builtin_build + builtin_eval
    );

    // Test with quantized_matmul (not gather) - 144 chained calls
    let qm_w = Array::zeros::<u32>(&[d, d * 4 / 32]).unwrap();
    let qm_s = Array::ones::<f32>(&[d, d / 64]).unwrap();
    let qm_b = Array::zeros::<f32>(&[d, d / 64]).unwrap();
    let x_qm = Array::ones::<f32>(&[1, 1, d]).unwrap();
    mlx_rs::transforms::eval([&qm_w, &qm_s, &qm_b, &x_qm]).unwrap();

    // Warm up
    for _ in 0..3 {
        let mut y4 = x_qm.clone();
        for _ in 0..144 {
            y4 = quantized_forward(&y4, &qm_w, &qm_s, &qm_b, 64, 4).unwrap();
        }
        mlx_rs::transforms::eval([&y4]).unwrap();
    }

    let n5 = 50;
    let mut total_qm_build = 0u128;
    let mut total_qm_eval = 0u128;
    for _ in 0..n5 {
        let t0 = std::time::Instant::now();
        let mut y4 = x_qm.clone();
        for _ in 0..144 {
            y4 = quantized_forward(&y4, &qm_w, &qm_s, &qm_b, 64, 4).unwrap();
        }
        let t1 = std::time::Instant::now();
        mlx_rs::transforms::eval([&y4]).unwrap();
        let t2 = std::time::Instant::now();
        total_qm_build += (t1 - t0).as_nanos();
        total_qm_eval += (t2 - t1).as_nanos();
    }
    let qm_build = total_qm_build as f64 / n5 as f64 / 1_000_000.0;
    let qm_eval = total_qm_eval as f64 / n5 as f64 / 1_000_000.0;
    eprintln!(
        "144 chained quantized_matmul: build={qm_build:.2}ms eval={qm_eval:.2}ms total={:.2}ms",
        qm_build + qm_eval
    );

    // Benchmark: single layer, per-call eval
    let n2 = 200;
    let start2 = std::time::Instant::now();
    for _ in 0..n2 {
        let g = gather_qmm(&x, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false).unwrap();
        let u = gather_qmm(&x, &up_w, &up_s, &up_b, &indices, true, 64, 4, false).unwrap();
        let activated = swiglu(&g, &u).unwrap();
        let y = gather_qmm(
            &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
        )
        .unwrap();
        mlx_rs::transforms::eval([&y]).unwrap();
    }
    let per_layer_ms = start2.elapsed().as_millis() as f64 / n2 as f64;
    eprintln!("1 layer * 3 gather_qmm + SwiGLU (per-call eval): {per_layer_ms:.2} ms");

    // Test eval overhead: 1000 chained adds (Python: build=0.23ms eval=1.87ms)
    let n_ops = 1000;
    let x_add = Array::ones::<f32>(&[1, 1, 2048]).unwrap();
    mlx_rs::transforms::eval([&x_add]).unwrap();
    // Warmup
    for _ in 0..3 {
        let mut y = x_add.clone();
        for _ in 0..n_ops {
            y = y.add(&x_add).unwrap();
        }
        mlx_rs::transforms::eval([&y]).unwrap();
    }
    let n6 = 50;
    let mut total_add_build = 0u128;
    let mut total_add_eval = 0u128;
    for _ in 0..n6 {
        let t0 = std::time::Instant::now();
        let mut y = x_add.clone();
        for _ in 0..n_ops {
            y = y.add(&x_add).unwrap();
        }
        let t1 = std::time::Instant::now();
        mlx_rs::transforms::eval([&y]).unwrap();
        let t2 = std::time::Instant::now();
        total_add_build += (t1 - t0).as_nanos();
        total_add_eval += (t2 - t1).as_nanos();
    }
    let add_build = total_add_build as f64 / n6 as f64 / 1_000_000.0;
    let add_eval = total_add_eval as f64 / n6 as f64 / 1_000_000.0;
    eprintln!(
        "{n_ops} chained adds: build={add_build:.2}ms eval={add_eval:.2}ms total={:.2}ms",
        add_build + add_eval
    );
    eprintln!(
        "Per op: build={:.1}us eval={:.1}us",
        add_build * 1000.0 / n_ops as f64,
        add_eval * 1000.0 / n_ops as f64
    );

    // Test with task-local default stream
    let stream = mlx_rs::Stream::new();
    let gather_with_stream = || {
        mlx_rs::with_stream(&stream.clone(), || {
            let mut total_b = 0u128;
            let mut total_e = 0u128;
            let n7 = 50;
            for _ in 0..n7 {
                let t0 = std::time::Instant::now();
                let mut y = x.clone();
                for _ in 0..48 {
                    let g = gather_qmm(&y, &gate_w, &gate_s, &gate_b, &indices, true, 64, 4, false)
                        .unwrap();
                    let u =
                        gather_qmm(&y, &up_w, &up_s, &up_b, &indices, true, 64, 4, false).unwrap();
                    let activated = swiglu(&g, &u).unwrap();
                    y = gather_qmm(
                        &activated, &down_w, &down_s, &down_b, &indices, true, 64, 4, false,
                    )
                    .unwrap();
                }
                let t1 = std::time::Instant::now();
                mlx_rs::transforms::eval([&y]).unwrap();
                let t2 = std::time::Instant::now();
                total_b += (t1 - t0).as_nanos();
                total_e += (t2 - t1).as_nanos();
            }
            let b = total_b as f64 / n7 as f64 / 1_000_000.0;
            let e = total_e as f64 / n7 as f64 / 1_000_000.0;
            eprintln!(
                "48 layers gather_qmm (with task-local stream): build={b:.2}ms eval={e:.2}ms total={:.2}ms",
                b + e
            );
        });
    };
    gather_with_stream();
}

/// Benchmark: 200 chained quantized_matmul ops (matching Python bench).
/// Python: build=0.05ms eval=1.40ms total=1.45ms
#[test]
#[ignore = "benchmark, requires GPU"]
fn bench_chained_quantized_matmul() {
    use mlx_rs::Dtype;

    let x = ops::ones_dtype(&[1, 1, 2048], Dtype::Float16).unwrap();
    let raw_w = ops::ones_dtype(&[2048, 2048], Dtype::Float16).unwrap();
    let (w, s, b) = crate::test_support::quantize_affine(&raw_w, 64, 4).unwrap();
    mlx_rs::transforms::eval([&x, &w, &s, &b]).unwrap();

    let n_ops = 200;
    let n = 50;

    // Warmup
    for _ in 0..10 {
        let mut y = x.clone();
        for _ in 0..n_ops {
            y = quantized_forward(&y, &w, &s, &b, 64, 4).unwrap();
        }
        mlx_rs::transforms::eval([&y]).unwrap();
    }

    let mut total_build = 0u128;
    let mut total_eval = 0u128;
    for _ in 0..n {
        let t0 = std::time::Instant::now();
        let mut y = x.clone();
        for _ in 0..n_ops {
            y = quantized_forward(&y, &w, &s, &b, 64, 4).unwrap();
        }
        let t1 = std::time::Instant::now();
        mlx_rs::transforms::eval([&y]).unwrap();
        let t2 = std::time::Instant::now();
        total_build += (t1 - t0).as_nanos();
        total_eval += (t2 - t1).as_nanos();
    }
    let build = total_build as f64 / n as f64 / 1e6;
    let eval = total_eval as f64 / n as f64 / 1e6;
    eprintln!(
        "Rust 200 qmm: build={build:.2}ms eval={eval:.2}ms total={:.2}ms",
        build + eval
    );

    // 200 chained adds
    for _ in 0..10 {
        let mut y = x.clone();
        for _ in 0..n_ops {
            y = y.add(&x).unwrap();
        }
        mlx_rs::transforms::eval([&y]).unwrap();
    }
    let mut total_build = 0u128;
    let mut total_eval = 0u128;
    for _ in 0..n {
        let t0 = std::time::Instant::now();
        let mut y = x.clone();
        for _ in 0..n_ops {
            y = y.add(&x).unwrap();
        }
        let t1 = std::time::Instant::now();
        mlx_rs::transforms::eval([&y]).unwrap();
        let t2 = std::time::Instant::now();
        total_build += (t1 - t0).as_nanos();
        total_eval += (t2 - t1).as_nanos();
    }
    let build = total_build as f64 / n as f64 / 1e6;
    let eval = total_eval as f64 / n as f64 / 1e6;
    eprintln!(
        "Rust 200 add: build={build:.2}ms eval={eval:.2}ms total={:.2}ms",
        build + eval
    );
}

/// Simulate 48-layer forward pass with per-layer weights.
/// Python shared-weight sim: build=0.59ms eval=8.08ms
#[test]
#[ignore = "benchmark, requires GPU"]
fn bench_simulated_forward() {
    use mlx_rs::Dtype;

    let d = 2048i32;
    let d_inter = 512i32; // moe_intermediate_size from config
    let n_experts = 512i32;
    let top_k = 10i32; // num_experts_per_tok from config
    let gs = 64i32;
    let bits = 4i32;
    let shared_inter = 512i32; // shared_expert_intermediate_size

    // Use random weights to test realistic memory access patterns.
    // ops::ones_dtype creates constant data that artificially benefits from GPU cache.
    let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
        let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        crate::test_support::quantize_affine(&raw, gs, bits).unwrap()
    };
    let make_sw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
        let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        crate::test_support::quantize_affine(&raw, gs, bits).unwrap()
    };

    let hk = 16i32;
    let dk = 128i32;
    let hv = 32i32;
    let dv = 128i32;

    struct LayerWeights {
        q_proj: (Array, Array, Array),
        k_proj: (Array, Array, Array),
        v_proj: (Array, Array, Array),
        o_proj: (Array, Array, Array),
        g_proj: (Array, Array, Array),
        beta_proj: (Array, Array, Array),
        gate: (Array, Array, Array),
        sw_gate: (Array, Array, Array),
        sw_up: (Array, Array, Array),
        sw_down: (Array, Array, Array),
        se_gate: (Array, Array, Array),
        se_up: (Array, Array, Array),
        se_down: (Array, Array, Array),
        se_gate_proj: (Array, Array, Array),
        norm_w: Array,
    }

    let layers: Vec<LayerWeights> = (0..48)
        .map(|_| LayerWeights {
            q_proj: make_qw(d, hk * dk),
            k_proj: make_qw(d, hk * dk),
            v_proj: make_qw(d, hv * dv),
            o_proj: make_qw(hv * dv, d),
            g_proj: make_qw(d, hv),
            beta_proj: make_qw(d, hv),
            gate: make_qw(d, n_experts),
            sw_gate: make_sw(d, d_inter),
            sw_up: make_sw(d, d_inter),
            sw_down: make_sw(d_inter, d),
            se_gate: make_qw(d, shared_inter * 2),
            se_up: make_qw(d, shared_inter * 2),
            se_down: make_qw(shared_inter * 2, d),
            se_gate_proj: make_qw(d, 1),
            norm_w: Array::ones::<f32>(&[d])
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap(),
        })
        .collect();

    let mut all_w: Vec<&Array> = Vec::new();
    for l in &layers {
        for (w, s, b) in [
            &l.q_proj,
            &l.k_proj,
            &l.v_proj,
            &l.o_proj,
            &l.g_proj,
            &l.beta_proj,
            &l.gate,
            &l.sw_gate,
            &l.sw_up,
            &l.sw_down,
            &l.se_gate,
            &l.se_up,
            &l.se_down,
            &l.se_gate_proj,
        ] {
            all_w.extend_from_slice(&[w, s, b]);
        }
        all_w.push(&l.norm_w);
    }
    mlx_rs::transforms::eval(all_w).unwrap();

    // Check actual memory usage to verify weights are materialized
    let active_mem = {
        let mut res: usize = 0;
        #[allow(unsafe_code)]
        unsafe {
            mlx_sys::mlx_get_active_memory(&mut res as *mut _);
        }
        res
    };
    eprintln!(
        "Active memory after weight eval: {:.2} GB",
        active_mem as f64 / 1e9
    );

    // Print one switch weight shape to verify
    eprintln!(
        "sw_gate[0] shape: {:?} dtype: {:?}",
        layers[0].sw_gate.0.shape(),
        layers[0].sw_gate.0.dtype()
    );

    let x = ops::ones_dtype(&[1, 1, d], Dtype::Float16).unwrap();
    mlx_rs::transforms::eval([&x]).unwrap();

    let forward_n_inline = |x: &Array, n_layers: usize| -> Array {
        let mut h = x.clone();
        for l in layers.iter().take(n_layers) {
            let normed = fast::rms_norm(&h, Some(&l.norm_w), 1e-6).unwrap();

            // Attention projections (matching real model's GDN layer ops)
            let _q = quantized_forward(&normed, &l.q_proj.0, &l.q_proj.1, &l.q_proj.2, gs, bits)
                .unwrap();
            let _k = quantized_forward(&normed, &l.k_proj.0, &l.k_proj.1, &l.k_proj.2, gs, bits)
                .unwrap();
            let v = quantized_forward(&normed, &l.v_proj.0, &l.v_proj.1, &l.v_proj.2, gs, bits)
                .unwrap();
            let g = quantized_forward(&normed, &l.g_proj.0, &l.g_proj.1, &l.g_proj.2, gs, bits)
                .unwrap();
            let _beta = quantized_forward(
                &normed,
                &l.beta_proj.0,
                &l.beta_proj.1,
                &l.beta_proj.2,
                gs,
                bits,
            )
            .unwrap();
            let attn_proxy = v
                .multiply(&nn::sigmoid(&g.sum_axes(&[-1], true).unwrap()).unwrap())
                .unwrap();
            let o = quantized_forward(&attn_proxy, &l.o_proj.0, &l.o_proj.1, &l.o_proj.2, gs, bits)
                .unwrap();

            let h2 = h.add(o).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&l.norm_w), 1e-6).unwrap();

            // Router
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

            // Switch MLP (per-layer switch weights)
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

            // Shared expert (per-layer weights)
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
    };

    for n_layers in [1, 4, 8, 16, 24, 48] {
        for _ in 0..5 {
            let y = forward_n_inline(&x, n_layers);
            mlx_rs::transforms::eval([&y]).unwrap();
        }
        let n = 20;
        let mut total_eval = 0u128;
        for _ in 0..n {
            let y = forward_n_inline(&x, n_layers);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&y]).unwrap();
            total_eval += t0.elapsed().as_nanos();
        }
        let eval = total_eval as f64 / n as f64 / 1e6;
        eprintln!(
            "Inline {n_layers} layers: eval={eval:.2}ms per_layer={:.2}ms",
            eval / n_layers as f64
        );
    }
}

/// Test gather_qmm with loaded vs random weights to isolate memory effects.
#[test]
#[ignore = "benchmark, requires GPU"]
fn bench_gather_qmm_loaded_vs_random() {
    use mlx_rs::Dtype;
    let Some(model_dir) = std::env::var_os("HIGGS_QWEN3_NEXT_BENCH_MODEL") else {
        eprintln!("Skipping: set HIGGS_QWEN3_NEXT_BENCH_MODEL to a local model directory");
        return;
    };
    let model_dir = std::path::PathBuf::from(model_dir);
    let Some(shard) = find_gather_qmm_bench_shard(&model_dir) else {
        eprintln!("Skipping: no safetensors shard with switch_mlp gate weights found");
        return;
    };
    let path = shard.as_path();

    // Load one safetensors shard
    let loaded = Array::load_safetensors(path).unwrap();
    mlx_rs::transforms::eval(loaded.values()).unwrap();

    // Find a switch_mlp weight (should be large [512, intermediate, ...])
    let mut sw_key = None;
    for key in loaded.keys() {
        if key.contains("switch_mlp") && key.contains("gate_proj") && key.contains(".weight") {
            sw_key = Some(key.clone());
            break;
        }
    }
    let sw_key = sw_key.expect("No switch_mlp weight found in shard");
    let w_loaded = &loaded[&sw_key];
    eprintln!(
        "Loaded weight '{sw_key}': shape={:?} dtype={:?}",
        w_loaded.shape(),
        w_loaded.dtype()
    );

    // Find corresponding scales and biases
    let scales_key = sw_key.replace(".weight", ".scales");
    let biases_key = sw_key.replace(".weight", ".biases");
    let s_loaded = &loaded[&scales_key];
    let b_loaded = &loaded[&biases_key];
    eprintln!(
        "Scales: {:?}, Biases: {:?}",
        s_loaded.shape(),
        b_loaded.shape()
    );

    // Create random weights of the same shape/dtype
    let w_shape = w_loaded.shape().to_vec();
    let s_shape = s_loaded.shape().to_vec();
    let b_shape = b_loaded.shape().to_vec();

    let w_random = mlx_rs::random::normal::<f32>(&w_shape, None, None, None)
        .unwrap()
        .as_dtype(w_loaded.dtype())
        .unwrap();
    let s_random = mlx_rs::random::normal::<f32>(&s_shape, None, None, None)
        .unwrap()
        .as_dtype(s_loaded.dtype())
        .unwrap();
    let b_random = mlx_rs::random::normal::<f32>(&b_shape, None, None, None)
        .unwrap()
        .as_dtype(b_loaded.dtype())
        .unwrap();
    mlx_rs::transforms::eval([&w_random, &s_random, &b_random]).unwrap();

    // Test input
    let x = ops::ones_dtype(&[1, 1, 1, 1, 2048], Dtype::Float16).unwrap();
    let indices = Array::from_slice(&[0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, 10]);
    mlx_rs::transforms::eval([&x, &indices]).unwrap();

    let gs = 64i32;
    let bits = 4i32;
    let n = 100;

    // Benchmark loaded weights
    for _ in 0..10 {
        let y = gather_qmm(
            &x, w_loaded, s_loaded, b_loaded, &indices, true, gs, bits, false,
        )
        .unwrap();
        mlx_rs::transforms::eval([&y]).unwrap();
    }
    let mut total_loaded = 0u128;
    for _ in 0..n {
        let t0 = std::time::Instant::now();
        let y = gather_qmm(
            &x, w_loaded, s_loaded, b_loaded, &indices, true, gs, bits, false,
        )
        .unwrap();
        mlx_rs::transforms::eval([&y]).unwrap();
        total_loaded += t0.elapsed().as_nanos();
    }

    // Benchmark random weights
    for _ in 0..10 {
        let y = gather_qmm(
            &x, &w_random, &s_random, &b_random, &indices, true, gs, bits, false,
        )
        .unwrap();
        mlx_rs::transforms::eval([&y]).unwrap();
    }
    let mut total_random = 0u128;
    for _ in 0..n {
        let t0 = std::time::Instant::now();
        let y = gather_qmm(
            &x, &w_random, &s_random, &b_random, &indices, true, gs, bits, false,
        )
        .unwrap();
        mlx_rs::transforms::eval([&y]).unwrap();
        total_random += t0.elapsed().as_nanos();
    }

    let loaded_us = total_loaded as f64 / n as f64 / 1e3;
    let random_us = total_random as f64 / n as f64 / 1e3;
    eprintln!(
        "gather_qmm single layer: loaded={loaded_us:.1}us random={random_us:.1}us ratio={:.2}x",
        loaded_us / random_us
    );
}

fn find_gather_qmm_bench_shard(model_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut candidates: Vec<_> = std::fs::read_dir(model_dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    candidates.sort_by(|a, b| {
        let a_name = a.file_name().and_then(|name| name.to_str()).unwrap_or("");
        let b_name = b.file_name().and_then(|name| name.to_str()).unwrap_or("");
        let a_shard = a_name.starts_with("model-") && a_name.contains("-of-");
        let b_shard = b_name.starts_with("model-") && b_name.contains("-of-");
        b_shard.cmp(&a_shard).then_with(|| a_name.cmp(b_name))
    });

    candidates.into_iter().find(|path| {
        Array::load_safetensors(path).is_ok_and(|loaded| {
            loaded.keys().any(|key| {
                key.contains("switch_mlp") && key.contains("gate_proj") && key.contains(".weight")
            })
        })
    })
}

#[test]
#[ignore = "requires GPU"]
#[cfg(any())]
fn bench_gather_mm_interleave() {
    use mlx_rs::Dtype;
    let n_layers = 48usize;
    let n_gdn = 36usize;
    let hv = 32i32;
    let dv = 128i32;
    let dk = 128i32;
    let d = 256i32; // Small dim to avoid OOM (float weights are not quantized)
    let n_experts = 64i32;
    let top_k = 10i32;

    // gather_mm: a=[..., M, K] @ b=[batch, K, N] -> [..., batch_sel, M, N]
    let float_weights: Vec<Array> = (0..n_layers)
        .map(|_| {
            mlx_rs::random::normal::<f32>(&[n_experts, d, d], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap()
        })
        .collect();
    mlx_rs::transforms::eval(float_weights.iter().collect::<Vec<_>>()).unwrap();

    let x = Array::ones::<f32>(&[1, 1, d])
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();
    let states: Vec<Array> = (0..n_gdn)
        .map(|_| {
            Array::zeros::<f32>(&[1, hv, dv, dk])
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap()
        })
        .collect();
    x.eval().unwrap();
    for s in &states {
        s.eval().unwrap();
    }

    let n = 20;

    // gather_mm only (no state)
    let forward_gather_only = |h_in: &Array| -> Array {
        let mut h = h_in.clone();
        for i in 0..n_layers {
            let rhs_inds = Array::from_slice(&[0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, top_k]);
            let x_exp = h.expand_dims(-2).unwrap();
            let out =
                ops::gather_mm(&x_exp, &float_weights[i], None::<&Array>, &rhs_inds, None).unwrap();
            let out_sq = out.squeeze_axes(&[-2]).unwrap();
            let expert_sum = out_sq.sum_axes(&[-2], false).unwrap();
            h = h.add(expert_sum).unwrap();
        }
        h
    };

    for _ in 0..5 {
        let r = forward_gather_only(&x);
        mlx_rs::transforms::eval([&r]).unwrap();
    }
    let mut total = 0u128;
    for _ in 0..n {
        let r = forward_gather_only(&x);
        let t0 = std::time::Instant::now();
        mlx_rs::transforms::eval([&r]).unwrap();
        total += t0.elapsed().as_nanos();
    }
    println!(
        "gather_mm only (48 layers): {:.2}ms",
        total as f64 / n as f64 / 1e6
    );

    // gather_mm interleaved with state
    let forward_interleaved = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
        let mut h = h_in.clone();
        let mut gdn_idx = 0usize;
        for i in 0..n_layers {
            if gdn_idx < n_gdn && (i + 1) % 4 != 0 {
                let g = h.sum_axes(&[-1], true).unwrap();
                let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                let new_state = ss[gdn_idx]
                    .multiply(decay)
                    .unwrap()
                    .add(Array::from_f32(0.01))
                    .unwrap();
                let y = new_state
                    .sum_axes(&[-1], false)
                    .unwrap()
                    .reshape(&[1, 1, -1])
                    .unwrap()
                    .index((.., .., ..d));
                ss[gdn_idx] = new_state;
                h = h.add(y).unwrap();
                gdn_idx += 1;
            }

            let rhs_inds = Array::from_slice(&[0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, top_k]);
            let x_exp = h.expand_dims(-2).unwrap();
            let out =
                ops::gather_mm(&x_exp, &float_weights[i], None::<&Array>, &rhs_inds, None).unwrap();
            let out_sq = out.squeeze_axes(&[-2]).unwrap();
            let expert_sum = out_sq.sum_axes(&[-2], false).unwrap();
            h = h.add(expert_sum).unwrap();
        }
        h
    };

    for _ in 0..5 {
        let mut ss = states.clone();
        let r = forward_interleaved(&x, &mut ss);
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        mlx_rs::transforms::eval(t).unwrap();
    }
    total = 0;
    for _ in 0..n {
        let mut ss = states.clone();
        let r = forward_interleaved(&x, &mut ss);
        let t0 = std::time::Instant::now();
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        mlx_rs::transforms::eval(t).unwrap();
        total += t0.elapsed().as_nanos();
    }
    println!(
        "gather_mm interleaved (48 layers): {:.2}ms",
        total as f64 / n as f64 / 1e6
    );
}

#[test]
#[ignore = "benchmark, requires GPU"]
fn bench_metal_kernel_gather_qmm_interleaving() {
    let b: i32 = 1;
    let d: i32 = 2048;
    let n_layers: i32 = 48;
    let n_gdn = 36;
    let n_experts: i32 = 512;
    let d_inter: i32 = 512;
    let top_k: i32 = 10;
    let gs: i32 = 64;
    let bits: i32 = 4;
    let hk: i32 = 16;
    let hv: i32 = 32;
    let dk: i32 = 128;
    let dv: i32 = 128;

    let x = Array::from_slice(&vec![0.1f32; (b * d) as usize], &[b, 1, d]);

    fn make_qw3d(n: i32, out_d: i32, in_d: i32, gs: i32, bits: i32) -> (Array, Array, Array) {
        let raw = Array::from_slice(
            &vec![0.01f32; (n * out_d * in_d) as usize],
            &[n, out_d, in_d],
        );
        let (w, s, b_arr) = crate::test_support::quantize_affine(&raw, gs, bits).unwrap();
        mlx_rs::transforms::eval([&w, &s, &b_arr]).unwrap();
        (w, s, b_arr)
    }

    let gate_w: Vec<_> = (0..n_layers)
        .map(|_| make_qw3d(n_experts, d_inter, d, gs, bits))
        .collect();
    let up_w: Vec<_> = (0..n_layers)
        .map(|_| make_qw3d(n_experts, d_inter, d, gs, bits))
        .collect();
    let down_w: Vec<_> = (0..n_layers)
        .map(|_| make_qw3d(n_experts, d, d_inter, gs, bits))
        .collect();

    let q = Array::from_slice(&vec![0.1f32; (b * hk * dk) as usize], &[b, 1, hk, dk]);
    let k = Array::from_slice(&vec![0.1f32; (b * hk * dk) as usize], &[b, 1, hk, dk]);
    let v = Array::from_slice(&vec![0.1f32; (b * hv * dv) as usize], &[b, 1, hv, dv]);
    let a_log_arr = Array::zeros::<f32>(&[hv]).unwrap();
    let a_arr = Array::from_slice(&vec![1.0f32; (b * hv) as usize], &[b, 1, hv]);
    let dt_bias_arr = Array::zeros::<f32>(&[hv]).unwrap();
    let b_arr = Array::zeros::<f32>(&[b, 1, hv]).unwrap();
    let state = Array::zeros::<f32>(&[b, hv, dv, dk]).unwrap();
    mlx_rs::transforms::eval([&q, &k, &v, &a_log_arr, &a_arr, &dt_bias_arr, &b_arr, &state])
        .unwrap();

    let indices = Array::from_slice(&[0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 1, top_k]);

    // Test 1: gather_qmm ONLY
    let build_gqmm_only = |h_in: &Array| -> Array {
        let mut h = h_in.clone();
        for i in 0..n_layers as usize {
            let xe = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &xe,
                &gate_w[i].0,
                &gate_w[i].1,
                &gate_w[i].2,
                &indices,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &xe, &up_w[i].0, &up_w[i].1, &up_w[i].2, &indices, true, gs, bits, false,
            )
            .unwrap();
            let act = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &act,
                &down_w[i].0,
                &down_w[i].1,
                &down_w[i].2,
                &indices,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let expert = d_out
                .squeeze_axes(&[-2])
                .unwrap()
                .sum_axes(&[-2], false)
                .unwrap();
            h = h.add(expert).unwrap();
        }
        h
    };

    for _ in 0..5 {
        let r = build_gqmm_only(&x);
        mlx_rs::transforms::eval([&r]).unwrap();
    }
    let n = 10;
    let mut total = 0u128;
    for _ in 0..n {
        let r = build_gqmm_only(&x);
        let t0 = std::time::Instant::now();
        mlx_rs::transforms::eval([&r]).unwrap();
        total += t0.elapsed().as_nanos();
    }
    println!(
        "gather_qmm only (48 layers): {:.2}ms",
        total as f64 / n as f64 / 1e6
    );

    // Test 2: Metal kernel + gather_qmm interleaved
    let build_interleaved = |h_in: &Array| -> (Array, Vec<Array>) {
        let mut h = h_in.clone();
        let mut states_out = Vec::new();
        let mut gdn_idx = 0usize;
        for i in 0..n_layers as usize {
            if gdn_idx < n_gdn as usize && (i + 1) % 4 != 0 {
                let (y, s_out) = gated_delta_kernel_ffi(
                    &q,
                    &k,
                    &v,
                    &a_log_arr,
                    &a_arr,
                    &dt_bias_arr,
                    &b_arr,
                    &state,
                    b,
                    1,
                    hk,
                    dk,
                    hv,
                    dv,
                )
                .unwrap();
                let y_flat = y.reshape(&[b, 1, -1]).unwrap();
                let y_trunc = y_flat.index((.., .., ..d));
                h = h.add(y_trunc).unwrap();
                states_out.push(s_out);
                gdn_idx += 1;
            }
            let xe = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &xe,
                &gate_w[i].0,
                &gate_w[i].1,
                &gate_w[i].2,
                &indices,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &xe, &up_w[i].0, &up_w[i].1, &up_w[i].2, &indices, true, gs, bits, false,
            )
            .unwrap();
            let act = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &act,
                &down_w[i].0,
                &down_w[i].1,
                &down_w[i].2,
                &indices,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let expert = d_out
                .squeeze_axes(&[-2])
                .unwrap()
                .sum_axes(&[-2], false)
                .unwrap();
            h = h.add(expert).unwrap();
        }
        (h, states_out)
    };

    for _ in 0..5 {
        let (r, s) = build_interleaved(&x);
        let mut ev: Vec<&Array> = vec![&r];
        ev.extend(s.iter());
        mlx_rs::transforms::eval(ev).unwrap();
    }
    total = 0;
    for _ in 0..n {
        let (r, s) = build_interleaved(&x);
        let mut ev: Vec<&Array> = vec![&r];
        ev.extend(s.iter());
        let t0 = std::time::Instant::now();
        mlx_rs::transforms::eval(ev).unwrap();
        total += t0.elapsed().as_nanos();
    }
    println!(
        "Metal kernel + gather_qmm (eval h+states): {:.2}ms",
        total as f64 / n as f64 / 1e6
    );

    // Test 3: Metal kernel + gather_qmm, eval h only
    for _ in 0..5 {
        let (r, _) = build_interleaved(&x);
        mlx_rs::transforms::eval([&r]).unwrap();
    }
    total = 0;
    for _ in 0..n {
        let (r, _) = build_interleaved(&x);
        let t0 = std::time::Instant::now();
        mlx_rs::transforms::eval([&r]).unwrap();
        total += t0.elapsed().as_nanos();
    }
    println!(
        "Metal kernel + gather_qmm (eval h only): {:.2}ms",
        total as f64 / n as f64 / 1e6
    );
}
