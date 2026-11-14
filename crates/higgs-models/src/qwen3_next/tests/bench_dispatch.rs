//! Ignored benchmarks for dispatch and evaluation overhead.

use crate::qwen3_next::{activations::swiglu, ffi::gather_qmm, linear::quantized_forward};
use mlx_rs::{
    Array, Stream, fast,
    ops::{self, indexing::IndexOp},
};

/// Minimal reproducer: state ops + gather_qmm, nothing else.
#[test]
#[ignore = "requires GPU"]
fn bench_minimal_state_moe_interaction() {
    use mlx_rs::Dtype;
    let n_layers = 48usize;
    let n_gdn = 36usize;
    let hv = 32i32;
    let dv = 128i32;
    let dk = 128i32;
    let d = 2048i32;
    let gs = 64i32;
    let bits = 4i32;
    let n_experts = 512i32;
    let d_inter = 512i32;
    let top_k = 10i32;

    // Expert weights for gather_qmm
    let make_sw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
        let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        crate::test_support::quantize_affine(&raw, gs, bits).unwrap()
    };
    let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
        let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        crate::test_support::quantize_affine(&raw, gs, bits).unwrap()
    };

    let sw_gate: Vec<_> = (0..n_layers).map(|_| make_sw(d, d_inter)).collect();
    let sw_up: Vec<_> = (0..n_layers).map(|_| make_sw(d, d_inter)).collect();
    let sw_down: Vec<_> = (0..n_layers).map(|_| make_sw(d_inter, d)).collect();
    let gate_proj: Vec<_> = (0..n_layers).map(|_| make_qw(d, n_experts)).collect();
    let mut all_w: Vec<Array> = Vec::new();
    for i in 0..n_layers {
        all_w.extend([
            sw_gate[i].0.clone(),
            sw_gate[i].1.clone(),
            sw_gate[i].2.clone(),
        ]);
        all_w.extend([sw_up[i].0.clone(), sw_up[i].1.clone(), sw_up[i].2.clone()]);
        all_w.extend([
            sw_down[i].0.clone(),
            sw_down[i].1.clone(),
            sw_down[i].2.clone(),
        ]);
        all_w.extend([
            gate_proj[i].0.clone(),
            gate_proj[i].1.clone(),
            gate_proj[i].2.clone(),
        ]);
    }
    mlx_rs::transforms::eval(all_w.iter().collect::<Vec<_>>()).unwrap();

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

    // Test 1: state ops only (no MoE)
    let forward_state_only = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
        let mut h = h_in.clone();
        for gdn_idx in 0..n_gdn {
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
        }
        h
    };

    for _ in 0..5 {
        let mut ss = states.clone();
        let r = forward_state_only(&x, &mut ss);
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        mlx_rs::transforms::eval(t).unwrap();
    }
    let mut total = 0u128;
    for _ in 0..n {
        let mut ss = states.clone();
        let r = forward_state_only(&x, &mut ss);
        let t0 = std::time::Instant::now();
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        mlx_rs::transforms::eval(t).unwrap();
        total += t0.elapsed().as_nanos();
    }
    println!(
        "State ops only (36 layers): {:.2}ms",
        total as f64 / n as f64 / 1e6
    );

    // Test 2: MoE only (no state)
    let forward_moe_only = |h_in: &Array| -> Array {
        let mut h = h_in.clone();
        for i in 0..n_layers {
            let gate_out = quantized_forward(
                &h,
                &gate_proj[i].0,
                &gate_proj[i].1,
                &gate_proj[i].2,
                gs,
                bits,
            )
            .unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let all_inds = ops::argpartition_axis(&gates, -top_k, -1).unwrap();
            let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let scores = raw_scores
                .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                .unwrap();
            let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &sw_gate[i].0,
                &sw_gate[i].1,
                &sw_gate[i].2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp,
                &sw_up[i].0,
                &sw_up[i].1,
                &sw_up[i].2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &sw_down[i].0,
                &sw_down[i].1,
                &sw_down[i].2,
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
                .multiply(scores.expand_dims(-1).unwrap())
                .unwrap()
                .sum_axes(&[-2], false)
                .unwrap();
            h = h.add(expert_sum).unwrap();
        }
        h
    };

    for _ in 0..5 {
        let r = forward_moe_only(&x);
        mlx_rs::transforms::eval([&r]).unwrap();
    }
    total = 0;
    for _ in 0..n {
        let r = forward_moe_only(&x);
        let t0 = std::time::Instant::now();
        mlx_rs::transforms::eval([&r]).unwrap();
        total += t0.elapsed().as_nanos();
    }
    println!(
        "MoE ops only (48 layers): {:.2}ms",
        total as f64 / n as f64 / 1e6
    );

    // Test 3: interleaved state + MoE
    let forward_interleaved = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
        let mut h = h_in.clone();
        let mut gdn_idx = 0usize;
        for i in 0..n_layers {
            // State ops (for GDN layers)
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

            // MoE ops
            let gate_out = quantized_forward(
                &h,
                &gate_proj[i].0,
                &gate_proj[i].1,
                &gate_proj[i].2,
                gs,
                bits,
            )
            .unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let all_inds = ops::argpartition_axis(&gates, -top_k, -1).unwrap();
            let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let scores = raw_scores
                .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                .unwrap();
            let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &sw_gate[i].0,
                &sw_gate[i].1,
                &sw_gate[i].2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp,
                &sw_up[i].0,
                &sw_up[i].1,
                &sw_up[i].2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &sw_down[i].0,
                &sw_down[i].1,
                &sw_down[i].2,
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
                .multiply(scores.expand_dims(-1).unwrap())
                .unwrap()
                .sum_axes(&[-2], false)
                .unwrap();
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
        "Interleaved state + MoE (48 layers): {:.2}ms",
        total as f64 / n as f64 / 1e6
    );

    // Test 3c: keep ALL intermediates alive (prevent drops during graph construction)
    let forward_keep_alive = |h_in: &Array, ss: &mut Vec<Array>| -> (Array, Vec<Array>) {
        let mut h = h_in.clone();
        let mut gdn_idx = 0usize;
        let mut keep: Vec<Array> = Vec::with_capacity(n_layers * 20);
        for i in 0..n_layers {
            if gdn_idx < n_gdn && (i + 1) % 4 != 0 {
                let g = h.sum_axes(&[-1], true).unwrap();
                let decay = g.reshape(&[1, 1, 1, 1]).unwrap();
                let new_state = ss[gdn_idx]
                    .multiply(&decay)
                    .unwrap()
                    .add(Array::from_f32(0.01))
                    .unwrap();
                let y = new_state
                    .sum_axes(&[-1], false)
                    .unwrap()
                    .reshape(&[1, 1, -1])
                    .unwrap()
                    .index((.., .., ..d));
                keep.push(g);
                keep.push(decay);
                keep.push(y.clone());
                ss[gdn_idx] = new_state;
                h = h.add(y).unwrap();
                gdn_idx += 1;
            }

            let gate_out = quantized_forward(
                &h,
                &gate_proj[i].0,
                &gate_proj[i].1,
                &gate_proj[i].2,
                gs,
                bits,
            )
            .unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let all_inds = ops::argpartition_axis(&gates, -top_k, -1).unwrap();
            let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let scores = raw_scores
                .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                .unwrap();
            let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &sw_gate[i].0,
                &sw_gate[i].1,
                &sw_gate[i].2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp,
                &sw_up[i].0,
                &sw_up[i].1,
                &sw_up[i].2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &sw_down[i].0,
                &sw_down[i].1,
                &sw_down[i].2,
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
                .multiply(scores.expand_dims(-1).unwrap())
                .unwrap()
                .sum_axes(&[-2], false)
                .unwrap();
            keep.extend([
                gate_out,
                gates,
                all_inds,
                top_inds.clone(),
                raw_scores,
                scores,
                x_exp,
                g_out,
                u_out,
                activated,
                d_out,
                expert_sum.clone(),
            ]);
            h = h.add(expert_sum).unwrap();
        }
        (h, keep)
    };

    for _ in 0..5 {
        let mut ss = states.clone();
        let (r, _keep) = forward_keep_alive(&x, &mut ss);
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        mlx_rs::transforms::eval(t).unwrap();
    }
    total = 0;
    for _ in 0..n {
        let mut ss = states.clone();
        let (r, _keep) = forward_keep_alive(&x, &mut ss);
        let t0 = std::time::Instant::now();
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        mlx_rs::transforms::eval(t).unwrap();
        total += t0.elapsed().as_nanos();
    }
    println!(
        "Interleaved keep-alive (48 layers): {:.2}ms",
        total as f64 / n as f64 / 1e6
    );

    // Test 3b: same but eval only h (not states)
    for _ in 0..5 {
        let mut ss = states.clone();
        let r = forward_interleaved(&x, &mut ss);
        mlx_rs::transforms::eval([&r]).unwrap();
    }
    total = 0;
    for _ in 0..n {
        let mut ss = states.clone();
        let r = forward_interleaved(&x, &mut ss);
        let t0 = std::time::Instant::now();
        mlx_rs::transforms::eval([&r]).unwrap();
        total += t0.elapsed().as_nanos();
    }
    println!(
        "Interleaved eval h only (48 layers): {:.2}ms",
        total as f64 / n as f64 / 1e6
    );

    // Test 4: interleaved state + quantized_matmul only (no gather_qmm)
    let simple_w: Vec<_> = (0..n_layers).map(|_| make_qw(d, d)).collect();
    let mut sw: Vec<Array> = Vec::new();
    for i in 0..n_layers {
        sw.extend([
            simple_w[i].0.clone(),
            simple_w[i].1.clone(),
            simple_w[i].2.clone(),
        ]);
    }
    mlx_rs::transforms::eval(sw.iter().collect::<Vec<_>>()).unwrap();

    let forward_interleaved_qmm = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
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
            // Simple quantized_matmul chain (no gather_qmm FFI)
            let out =
                quantized_forward(&h, &simple_w[i].0, &simple_w[i].1, &simple_w[i].2, gs, bits)
                    .unwrap();
            h = h.add(out).unwrap();
        }
        h
    };

    for _ in 0..5 {
        let mut ss = states.clone();
        let r = forward_interleaved_qmm(&x, &mut ss);
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        mlx_rs::transforms::eval(t).unwrap();
    }
    total = 0;
    for _ in 0..n {
        let mut ss = states.clone();
        let r = forward_interleaved_qmm(&x, &mut ss);
        let t0 = std::time::Instant::now();
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        mlx_rs::transforms::eval(t).unwrap();
        total += t0.elapsed().as_nanos();
    }
    println!(
        "Interleaved state + quantized_matmul (no gather_qmm): {:.2}ms",
        total as f64 / n as f64 / 1e6
    );

    // Test 5: interleaved state + MoE using gather_qmm
    let forward_interleaved_ops = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
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

            let gate_out = quantized_forward(
                &h,
                &gate_proj[i].0,
                &gate_proj[i].1,
                &gate_proj[i].2,
                gs,
                bits,
            )
            .unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let all_inds = ops::argpartition_axis(&gates, -top_k, -1).unwrap();
            let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let scores = raw_scores
                .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                .unwrap();
            let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &sw_gate[i].0,
                &sw_gate[i].1,
                &sw_gate[i].2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp,
                &sw_up[i].0,
                &sw_up[i].1,
                &sw_up[i].2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &sw_down[i].0,
                &sw_down[i].1,
                &sw_down[i].2,
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
                .multiply(scores.expand_dims(-1).unwrap())
                .unwrap()
                .sum_axes(&[-2], false)
                .unwrap();
            h = h.add(expert_sum).unwrap();
        }
        h
    };

    for _ in 0..5 {
        let mut ss = states.clone();
        let r = forward_interleaved_ops(&x, &mut ss);
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        mlx_rs::transforms::eval(t).unwrap();
    }
    total = 0;
    for _ in 0..n {
        let mut ss = states.clone();
        let r = forward_interleaved_ops(&x, &mut ss);
        let t0 = std::time::Instant::now();
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        mlx_rs::transforms::eval(t).unwrap();
        total += t0.elapsed().as_nanos();
    }
    println!(
        "Interleaved state + gather_qmm: {:.2}ms",
        total as f64 / n as f64 / 1e6
    );
}

#[test]
#[ignore = "requires GPU"]
#[cfg(any())]
fn bench_cxx_bypass() {
    use mlx_rs::Dtype;
    let n_layers = 48i32;
    let n_gdn = 36i32;
    let hv = 32i32;
    let dv = 128i32;
    let dk = 128i32;
    let d = 2048i32;
    let gs = 64i32;
    let bits = 4i32;
    let n_experts = 512i32;
    let d_inter = 512i32;
    let top_k = 10i32;
    let n = 20;

    // Self-contained C++ benchmark (no prior Rust MLX operations)
    #[allow(unsafe_code)]
    let self_contained_us = unsafe {
        mlx_sys::mlx_bench_self_contained(
            n_layers, n_gdn, d, n_experts, d_inter, top_k, gs, bits, hv, dv, dk, 5, n,
        )
    };
    println!(
        "C++ self-contained BEFORE any Rust ops: {:.2}ms",
        self_contained_us / 1000.0
    );

    // Now do a tiny eval to see if ANY eval causes the slowdown
    {
        let tiny = Array::ones::<f32>(&[1, 1, 1]).unwrap();
        tiny.eval().unwrap();
    }
    #[allow(unsafe_code)]
    let after_tiny_us = unsafe {
        mlx_sys::mlx_bench_self_contained(
            n_layers, n_gdn, d, n_experts, d_inter, top_k, gs, bits, hv, dv, dk, 5, n,
        )
    };
    println!(
        "C++ self-contained AFTER tiny eval: {:.2}ms",
        after_tiny_us / 1000.0
    );

    // Now create and eval ONE large weight to test memory impact
    {
        let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_inter, d], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let (w, s, b) = crate::test_support::quantize_affine(&raw, gs, bits).unwrap();
        mlx_rs::transforms::eval(vec![&w, &s, &b]).unwrap();
        // raw, w, s, b will be dropped here
    }
    #[allow(unsafe_code)]
    let after_big_us = unsafe {
        mlx_sys::mlx_bench_self_contained(
            n_layers, n_gdn, d, n_experts, d_inter, top_k, gs, bits, hv, dv, dk, 5, n,
        )
    };
    println!(
        "C++ self-contained AFTER one big quantize: {:.2}ms",
        after_big_us / 1000.0
    );

    let make_sw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
        let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        crate::test_support::quantize_affine(&raw, gs, bits).unwrap()
    };
    let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
        let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        crate::test_support::quantize_affine(&raw, gs, bits).unwrap()
    };

    let sw_gate: Vec<_> = (0..n_layers).map(|_| make_sw(d, d_inter)).collect();
    let sw_up: Vec<_> = (0..n_layers).map(|_| make_sw(d, d_inter)).collect();
    let sw_down: Vec<_> = (0..n_layers).map(|_| make_sw(d_inter, d)).collect();
    let gate_proj: Vec<_> = (0..n_layers).map(|_| make_qw(d, n_experts)).collect();
    let mut all_w: Vec<Array> = Vec::new();
    for i in 0..n_layers as usize {
        all_w.extend([
            sw_gate[i].0.clone(),
            sw_gate[i].1.clone(),
            sw_gate[i].2.clone(),
        ]);
        all_w.extend([sw_up[i].0.clone(), sw_up[i].1.clone(), sw_up[i].2.clone()]);
        all_w.extend([
            sw_down[i].0.clone(),
            sw_down[i].1.clone(),
            sw_down[i].2.clone(),
        ]);
        all_w.extend([
            gate_proj[i].0.clone(),
            gate_proj[i].1.clone(),
            gate_proj[i].2.clone(),
        ]);
    }
    mlx_rs::transforms::eval(all_w.iter().collect::<Vec<_>>()).unwrap();

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

    // Prepare raw pointer arrays for FFI
    let gate_w: Vec<_> = sw_gate.iter().map(|t| t.0.as_ptr()).collect();
    let gate_s: Vec<_> = sw_gate.iter().map(|t| t.1.as_ptr()).collect();
    let gate_b: Vec<_> = sw_gate.iter().map(|t| t.2.as_ptr()).collect();
    let up_w: Vec<_> = sw_up.iter().map(|t| t.0.as_ptr()).collect();
    let up_s: Vec<_> = sw_up.iter().map(|t| t.1.as_ptr()).collect();
    let up_b: Vec<_> = sw_up.iter().map(|t| t.2.as_ptr()).collect();
    let down_w: Vec<_> = sw_down.iter().map(|t| t.0.as_ptr()).collect();
    let down_s: Vec<_> = sw_down.iter().map(|t| t.1.as_ptr()).collect();
    let down_b: Vec<_> = sw_down.iter().map(|t| t.2.as_ptr()).collect();
    let gp_w: Vec<_> = gate_proj.iter().map(|t| t.0.as_ptr()).collect();
    let gp_s: Vec<_> = gate_proj.iter().map(|t| t.1.as_ptr()).collect();
    let gp_b: Vec<_> = gate_proj.iter().map(|t| t.2.as_ptr()).collect();

    let state_ptrs_for_cxx: Vec<_> = states.iter().map(|s| s.as_ptr()).collect();

    let n = 20;
    let stream = Stream::new();

    // Warmup
    for _ in 0..5 {
        let state_ptrs: Vec<_> = states.iter().map(|s| s.as_ptr()).collect();
        #[allow(unsafe_code)]
        let (result, state_outs) = unsafe {
            let mut result = mlx_sys::mlx_array_new();
            let mut state_outs: Vec<mlx_sys::mlx_array> =
                (0..n_gdn).map(|_| mlx_sys::mlx_array_new()).collect();
            let status = mlx_sys::mlx_bench_interleaved_cxx(
                &raw mut result,
                state_outs.as_mut_ptr(),
                x.as_ptr(),
                state_ptrs.as_ptr(),
                gate_w.as_ptr(),
                gate_s.as_ptr(),
                gate_b.as_ptr(),
                up_w.as_ptr(),
                up_s.as_ptr(),
                up_b.as_ptr(),
                down_w.as_ptr(),
                down_s.as_ptr(),
                down_b.as_ptr(),
                gp_w.as_ptr(),
                gp_s.as_ptr(),
                gp_b.as_ptr(),
                n_layers,
                n_gdn,
                d,
                n_experts,
                top_k,
                gs,
                bits,
                stream.as_ptr(),
            );
            assert_eq!(status, 0, "C++ shim failed");
            let r = Array::from_ptr(result);
            let so: Vec<Array> = state_outs.into_iter().map(|p| Array::from_ptr(p)).collect();
            (r, so)
        };
        let mut t: Vec<&Array> = vec![&result];
        t.extend(state_outs.iter());
        mlx_rs::transforms::eval(t).unwrap();
    }

    // Benchmark
    let mut total = 0u128;
    for _ in 0..n {
        let state_ptrs: Vec<_> = states.iter().map(|s| s.as_ptr()).collect();
        #[allow(unsafe_code)]
        let (result, state_outs) = unsafe {
            let mut result = mlx_sys::mlx_array_new();
            let mut state_outs: Vec<mlx_sys::mlx_array> =
                (0..n_gdn).map(|_| mlx_sys::mlx_array_new()).collect();
            let status = mlx_sys::mlx_bench_interleaved_cxx(
                &raw mut result,
                state_outs.as_mut_ptr(),
                x.as_ptr(),
                state_ptrs.as_ptr(),
                gate_w.as_ptr(),
                gate_s.as_ptr(),
                gate_b.as_ptr(),
                up_w.as_ptr(),
                up_s.as_ptr(),
                up_b.as_ptr(),
                down_w.as_ptr(),
                down_s.as_ptr(),
                down_b.as_ptr(),
                gp_w.as_ptr(),
                gp_s.as_ptr(),
                gp_b.as_ptr(),
                n_layers,
                n_gdn,
                d,
                n_experts,
                top_k,
                gs,
                bits,
                stream.as_ptr(),
            );
            assert_eq!(status, 0, "C++ shim failed");
            let r = Array::from_ptr(result);
            let so: Vec<Array> = state_outs.into_iter().map(|p| Array::from_ptr(p)).collect();
            (r, so)
        };
        let mut t: Vec<&Array> = vec![&result];
        t.extend(state_outs.iter());
        let t0 = std::time::Instant::now();
        mlx_rs::transforms::eval(t).unwrap();
        total += t0.elapsed().as_nanos();
    }
    println!(
        "C++ bypass interleaved (48 layers): {:.2}ms",
        total as f64 / n as f64 / 1e6
    );

    // Test: build + eval entirely in C++ (no Rust involvement in eval)
    #[allow(unsafe_code)]
    let avg_us = unsafe {
        mlx_sys::mlx_bench_interleaved_cxx_with_eval(
            x.as_ptr(),
            state_ptrs_for_cxx.as_ptr(),
            gate_w.as_ptr(),
            gate_s.as_ptr(),
            gate_b.as_ptr(),
            up_w.as_ptr(),
            up_s.as_ptr(),
            up_b.as_ptr(),
            down_w.as_ptr(),
            down_s.as_ptr(),
            down_b.as_ptr(),
            gp_w.as_ptr(),
            gp_s.as_ptr(),
            gp_b.as_ptr(),
            n_layers,
            n_gdn,
            d,
            n_experts,
            top_k,
            gs,
            bits,
            5,
            n,
        )
    };
    println!("C++ build+eval (48 layers): {:.2}ms", avg_us / 1000.0);

    // Test: state ops only (no MoE)
    #[allow(unsafe_code)]
    let state_only_us = unsafe {
        mlx_sys::mlx_bench_state_ops_only(x.as_ptr(), state_ptrs_for_cxx.as_ptr(), n_gdn, d, 5, n)
    };
    println!(
        "C++ state ops only (36 layers): {:.2}ms",
        state_only_us / 1000.0
    );

    // Test: interleaved but eval h only (no states in eval list)
    #[allow(unsafe_code)]
    let h_only_us = unsafe {
        mlx_sys::mlx_bench_interleaved_h_only_eval(
            x.as_ptr(),
            state_ptrs_for_cxx.as_ptr(),
            gate_w.as_ptr(),
            gate_s.as_ptr(),
            gate_b.as_ptr(),
            up_w.as_ptr(),
            up_s.as_ptr(),
            up_b.as_ptr(),
            down_w.as_ptr(),
            down_s.as_ptr(),
            down_b.as_ptr(),
            gp_w.as_ptr(),
            gp_s.as_ptr(),
            gp_b.as_ptr(),
            n_layers,
            n_gdn,
            d,
            n_experts,
            top_k,
            gs,
            bits,
            5,
            n,
        )
    };
    println!(
        "C++ interleaved h-only eval (48 layers): {:.2}ms",
        h_only_us / 1000.0
    );

    // For comparison: the standard Rust interleaved version
    let forward_interleaved = |h_in: &Array, ss: &mut Vec<Array>| -> Array {
        let mut h = h_in.clone();
        let mut gdn_idx = 0usize;
        for i in 0..n_layers as usize {
            if gdn_idx < n_gdn as usize && (i + 1) % 4 != 0 {
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
            let gate_out = quantized_forward(
                &h,
                &gate_proj[i].0,
                &gate_proj[i].1,
                &gate_proj[i].2,
                gs,
                bits,
            )
            .unwrap();
            let gates_v = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let all_inds = ops::argpartition_axis(&gates_v, -top_k, -1).unwrap();
            let top_inds = all_inds.index((.., .., (n_experts - top_k)..));
            let raw_scores = gates_v.take_along_axis(&top_inds, -1).unwrap();
            let scores = raw_scores
                .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                .unwrap();
            let x_exp = h.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &sw_gate[i].0,
                &sw_gate[i].1,
                &sw_gate[i].2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp,
                &sw_up[i].0,
                &sw_up[i].1,
                &sw_up[i].2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &sw_down[i].0,
                &sw_down[i].1,
                &sw_down[i].2,
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
                .multiply(scores.expand_dims(-1).unwrap())
                .unwrap()
                .sum_axes(&[-2], false)
                .unwrap();
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
        "Rust C API interleaved (48 layers): {:.2}ms",
        total as f64 / n as f64 / 1e6
    );
}

/// Test eval scaling with graph size using quantized_matmul + rms_norm
#[test]
#[ignore = "benchmark, requires GPU"]
fn bench_eval_scaling() {
    let b: i32 = 1;
    let d: i32 = 2048;
    let gs: i32 = 64;
    let bits: i32 = 4;
    let n_layers: i32 = 48;

    let x = Array::from_slice(&vec![0.1f32; (b * d) as usize], &[b, 1, d]);

    fn make_qw2d(rows: i32, cols: i32, gs: i32, bits: i32) -> (Array, Array, Array) {
        let raw = Array::from_slice(&vec![0.01f32; (rows * cols) as usize], &[rows, cols]);
        let (w, s, b_arr) = crate::test_support::quantize_affine(&raw, gs, bits).unwrap();
        mlx_rs::transforms::eval([&w, &s, &b_arr]).unwrap();
        (w, s, b_arr)
    }

    let weights: Vec<_> = (0..n_layers).map(|_| make_qw2d(d, d, gs, bits)).collect();
    let norm_ws: Vec<_> = (0..n_layers)
        .map(|_| {
            let w = Array::ones::<f32>(&[d]).unwrap();
            mlx_rs::transforms::eval([&w]).unwrap();
            w
        })
        .collect();

    for n_extras in &[0, 2, 5, 8, 12] {
        let total_ops = n_layers * (1 + n_extras + 1);
        let build = |h_in: &Array| -> Array {
            let mut h = h_in.clone();
            for i in 0..n_layers as usize {
                h = quantized_forward(&h, &weights[i].0, &weights[i].1, &weights[i].2, gs, bits)
                    .unwrap();
                for j in 0..*n_extras as usize {
                    let idx = (i + j + 1) % n_layers as usize;
                    let extra = quantized_forward(
                        &h,
                        &weights[idx].0,
                        &weights[idx].1,
                        &weights[idx].2,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let scale = Array::from_slice(&[0.01f32], &[1]);
                    h = h.add(extra.multiply(&scale).unwrap()).unwrap();
                }
                h = fast::rms_norm(&h, Some(&norm_ws[i]), 1e-6).unwrap();
            }
            h
        };
        for _ in 0..3 {
            let r = build(&x);
            mlx_rs::transforms::eval([&r]).unwrap();
        }
        let n = 10;
        let mut total_ns = 0u128;
        for _ in 0..n {
            let r = build(&x);
            let t0 = std::time::Instant::now();
            mlx_rs::transforms::eval([&r]).unwrap();
            total_ns += t0.elapsed().as_nanos();
        }
        let avg_ms = total_ns as f64 / n as f64 / 1e6;
        let us_per_op = avg_ms * 1000.0 / total_ops as f64;
        println!("extras={n_extras:2} ops~={total_ops:4} eval={avg_ms:.2}ms ({us_per_op:.1}us/op)");
    }
}

/// Measure async_eval pipelining: does GPU overlap with CPU graph building?
///
/// cargo test -p higgs-models --release -- bench_async_pipeline --nocapture --ignored
#[test]
#[ignore = "benchmark helper"]
fn bench_async_pipeline() {
    use mlx_rs::random::normal;
    use mlx_rs::transforms::{async_eval, eval};

    let d: &[i32] = &[2048, 2048];
    let w = normal::<f32>(d, None, None, None).unwrap();
    eval([&w].into_iter()).unwrap();

    let build_graph = |x: &Array| -> Array {
        let mut h = x.clone();
        for _ in 0..40 {
            let mm = h.matmul(&w).unwrap();
            h = mm.add(&h).unwrap();
        }
        h
    };

    let x = normal::<f32>(&[1, 1, 2048], None, None, None).unwrap();
    eval([&x].into_iter()).unwrap();

    // Sequential
    let n = 20usize;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let y = build_graph(&x);
        eval([&y].into_iter()).unwrap();
    }
    let seq_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

    // Pipelined
    let t0 = std::time::Instant::now();
    let mut y = build_graph(&x);
    async_eval([&y].into_iter()).unwrap();
    for _ in 0..n {
        let next_y = build_graph(&y);
        async_eval([&next_y].into_iter()).unwrap();
        eval([&y].into_iter()).unwrap();
        y = next_y;
    }
    let pipe_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

    eprintln!("Rust mlx-rs sequential:  {seq_ms:.2}ms/step");
    eprintln!("Rust mlx-rs pipelined:   {pipe_ms:.2}ms/step");
    eprintln!("Speedup: {:.2}x", seq_ms / pipe_ms);
}

/// Measure pure FFI graph-building overhead: no eval, just op dispatch.
///
/// cargo test -p higgs-models --release -- bench_ffi_overhead --nocapture --ignored
#[test]
#[ignore = "benchmark helper"]
fn bench_ffi_overhead() {
    use mlx_rs::transforms::eval;

    let a = Array::ones::<f32>(&[1, 1, 2048]).unwrap();
    let b = Array::ones::<f32>(&[1, 1, 2048]).unwrap();
    eval([&a, &b].into_iter()).unwrap();

    let n = 2000usize;

    // Graph build only (no eval)
    let t0 = std::time::Instant::now();
    let mut x = a.clone();
    for _ in 0..n {
        x = x.add(&b).unwrap();
    }
    let build_us = t0.elapsed().as_micros();
    eprintln!(
        "Rust mlx-rs: {n} adds graph-build = {build_us}us ({:.1}us/op)",
        build_us as f64 / n as f64
    );

    // Graph build + eval
    let t0 = std::time::Instant::now();
    let mut x = a.clone();
    for _ in 0..n {
        x = x.add(&b).unwrap();
    }
    eval([&x].into_iter()).unwrap();
    let total_us = t0.elapsed().as_micros();
    eprintln!(
        "Rust mlx-rs: {n} adds + eval = {total_us}us ({:.1}us/op)",
        total_us as f64 / n as f64
    );

    // With task-local stream set
    let stream = Stream::new();
    mlx_rs::with_stream(&stream, || {
        let t0 = std::time::Instant::now();
        let mut x = a.clone();
        for _ in 0..n {
            x = x.add(&b).unwrap();
        }
        let build_us = t0.elapsed().as_micros();
        eprintln!(
            "Rust mlx-rs (task-local stream): {n} adds graph-build = {build_us}us ({:.1}us/op)",
            build_us as f64 / n as f64
        );
    });
}
