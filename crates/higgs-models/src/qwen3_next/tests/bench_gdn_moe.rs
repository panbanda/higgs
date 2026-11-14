//! Ignored benchmarks for `GatedDeltaNet` and MoE interaction.

use super::gdn::gated_delta_step_ref;
use crate::qwen3_next::{
    activations::swiglu, ffi::gather_qmm, gated_delta_kernel::gated_delta_kernel_ffi,
    gated_delta_net::compute_g_compiled, linear::quantized_forward,
};
use mlx_rs::{
    Array, fast, nn,
    ops::{self, indexing::IndexOp},
};

/// Benchmark 36 GDN layers using bare Arrays.
/// Isolates GDN ops from the model framework to compare direct GPU time.
#[test]
#[ignore = "requires GPU"]
fn bench_gdn_layers() {
    use mlx_rs::Dtype;

    let d = 2048i32;
    let hk = 16i32;
    let hv = 32i32;
    let dk = 128i32;
    let dv = 128i32;
    let gs = 64i32;
    let bits = 4i32;
    let key_dim = hk * dk;
    let value_dim = hv * dv;
    let conv_dim = key_dim * 2 + value_dim;
    let qkvz_out = key_dim * 2 + value_dim * 2;
    let ba_out = hv * 2;
    let n_layers = 36;

    let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
        let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let (w, s, b) = crate::test_support::quantize_affine(&raw, gs, bits).unwrap();
        (w, s, b)
    };

    struct GDNWeights {
        in_proj_qkvz: (Array, Array, Array),
        in_proj_ba: (Array, Array, Array),
        out_proj: (Array, Array, Array),
        conv_w: Array,
        a_log: Array,
        dt_bias: Array,
        norm_w: Array,
    }

    let mut layers = Vec::new();
    let mut all_w: Vec<&Array> = Vec::new();
    for _ in 0..n_layers {
        layers.push(GDNWeights {
            in_proj_qkvz: make_qw(d, qkvz_out),
            in_proj_ba: make_qw(d, ba_out),
            out_proj: make_qw(value_dim, d),
            conv_w: mlx_rs::random::normal::<f32>(&[conv_dim, 4, 1], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap(),
            a_log: mlx_rs::random::normal::<f32>(&[hv], None, None, None).unwrap(),
            dt_bias: mlx_rs::random::normal::<f32>(&[hv], None, None, None).unwrap(),
            norm_w: Array::ones::<f32>(&[dv])
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap(),
        });
    }
    for l in &layers {
        all_w.extend([&l.in_proj_qkvz.0, &l.in_proj_qkvz.1, &l.in_proj_qkvz.2]);
        all_w.extend([&l.in_proj_ba.0, &l.in_proj_ba.1, &l.in_proj_ba.2]);
        all_w.extend([&l.out_proj.0, &l.out_proj.1, &l.out_proj.2]);
        all_w.extend([&l.conv_w, &l.a_log, &l.dt_bias, &l.norm_w]);
    }
    mlx_rs::transforms::eval(all_w).unwrap();

    let x = Array::ones::<f32>(&[1, 1, d])
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();
    let qk_norm_w = Array::ones::<f32>(&[dk]).unwrap();
    let inv_scale = Array::from_f32((dk as f32).sqrt().recip());
    let inv_scale_sq = {
        let s = (dk as f32).sqrt().recip();
        Array::from_f32(s * s)
    };
    let states: Vec<Array> = (0..n_layers)
        .map(|_| {
            Array::zeros::<f32>(&[1, hv, dv, dk])
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap()
        })
        .collect();
    let conv_states: Vec<Array> = (0..n_layers)
        .map(|_| {
            Array::zeros::<f32>(&[1, 3, conv_dim])
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap()
        })
        .collect();
    x.eval().unwrap();
    for s in &states {
        s.eval().unwrap();
    }
    for c in &conv_states {
        c.eval().unwrap();
    }

    let gdn_forward =
        |h: &Array, l: &GDNWeights, state: &Array, conv_state: &Array| -> (Array, Array, Array) {
            let qkvz = quantized_forward(
                h,
                &l.in_proj_qkvz.0,
                &l.in_proj_qkvz.1,
                &l.in_proj_qkvz.2,
                gs,
                bits,
            )
            .unwrap();
            let ba = quantized_forward(
                h,
                &l.in_proj_ba.0,
                &l.in_proj_ba.1,
                &l.in_proj_ba.2,
                gs,
                bits,
            )
            .unwrap();

            let q = qkvz
                .index((.., .., ..key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let k = qkvz
                .index((.., .., key_dim..2 * key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let v = qkvz
                .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                .reshape(&[1, 1, hv, dv])
                .unwrap();
            let z = qkvz.index((.., .., 2 * key_dim + value_dim..));

            let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
            let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();

            // Conv1d
            let q_flat = q.reshape(&[1, 1, -1]).unwrap();
            let k_flat = k.reshape(&[1, 1, -1]).unwrap();
            let v_flat = v.reshape(&[1, 1, -1]).unwrap();
            let mixed = ops::concatenate(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
            let conv_in = ops::concatenate(&[conv_state, &mixed], 1).unwrap();
            let new_conv_state = conv_in.index((.., -3.., ..));

            let conv_out =
                nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap()).unwrap();

            let conv_q = conv_out
                .index((.., .., ..key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let conv_k = conv_out
                .index((.., .., key_dim..2 * key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let conv_v = conv_out
                .index((.., .., 2 * key_dim..))
                .reshape(&[1, 1, hv, dv])
                .unwrap();

            // RMS norm
            let norm_q = fast::rms_norm(&conv_q, Some(&qk_norm_w), 1e-6)
                .unwrap()
                .multiply(&inv_scale_sq)
                .unwrap();
            let norm_k = fast::rms_norm(&conv_k, Some(&qk_norm_w), 1e-6)
                .unwrap()
                .multiply(&inv_scale)
                .unwrap();

            // Metal kernel (computes g and beta internally)
            let (y, new_state) = gated_delta_kernel_ffi(
                &norm_q, &norm_k, &conv_v, &l.a_log, &a, &l.dt_bias, &b, state, 1, 1, hk, dk, hv,
                dv,
            )
            .unwrap();

            // Gated RMSNorm + swiglu
            let normed = fast::rms_norm(&y, Some(&l.norm_w), 1e-6).unwrap();
            let z_shaped = z
                .index((.., .., ..value_dim))
                .reshape(&[1, 1, hv, dv])
                .unwrap();
            let gated = swiglu(&z_shaped, &normed).unwrap();

            // Output proj
            let out = quantized_forward(
                &gated.reshape(&[1, 1, -1]).unwrap(),
                &l.out_proj.0,
                &l.out_proj.1,
                &l.out_proj.2,
                gs,
                bits,
            )
            .unwrap();
            (out, new_state, new_conv_state)
        };

    // Warmup
    for _ in 0..5 {
        let mut h = x.clone();
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        for (j, l) in layers.iter().enumerate() {
            let (out, ns, nc) = gdn_forward(&h, l, &ss[j], &cs[j]);
            h = out;
            ss[j] = ns;
            cs[j] = nc;
        }
        let mut eval_targets: Vec<&Array> = vec![&h];
        eval_targets.extend(ss.iter());
        eval_targets.extend(cs.iter());
        mlx_rs::transforms::eval(eval_targets).unwrap();
    }

    // Benchmark
    let n = 20;
    let mut total = 0u128;
    for _ in 0..n {
        let mut h = x.clone();
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        for (j, l) in layers.iter().enumerate() {
            let (out, ns, nc) = gdn_forward(&h, l, &ss[j], &cs[j]);
            h = out;
            ss[j] = ns;
            cs[j] = nc;
        }
        let t0 = std::time::Instant::now();
        let mut eval_targets: Vec<&Array> = vec![&h];
        eval_targets.extend(ss.iter());
        eval_targets.extend(cs.iter());
        mlx_rs::transforms::eval(eval_targets).unwrap();
        total += t0.elapsed().as_nanos();
    }

    let avg_ms = total as f64 / n as f64 / 1e6;
    println!("Rust 36 GDN layers (bare arrays): {avg_ms:.2}ms");
    println!("Per layer: {:.3}ms", avg_ms / 36.0);
}

/// Benchmark 48 layers of interleaved GDN + MoE (matching real model structure).
/// GDN layers: 0,1,2, 4,5,6, 8,9,10, ...  (every layer except multiples of 4 minus 1)
/// FA layers: 3,7,11,... (every 4th layer, 0-indexed)
/// All layers have MoE.
#[test]
#[ignore = "requires GPU"]
fn bench_combined_gdn_moe() {
    use mlx_rs::Dtype;

    let d = 2048i32;
    let hk = 16i32;
    let hv = 32i32;
    let dk = 128i32;
    let dv = 128i32;
    let gs = 64i32;
    let bits = 4i32;
    let key_dim = hk * dk;
    let value_dim = hv * dv;
    let conv_dim = key_dim * 2 + value_dim;
    let qkvz_out = key_dim * 2 + value_dim * 2;
    let ba_out = hv * 2;
    let n_layers = 48;
    let full_attn_interval = 4;
    let d_inter = 512i32;
    let n_experts = 512i32;
    let top_k = 10i32;
    let shared_inter = 512i32;

    let make_qw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
        let raw = mlx_rs::random::normal::<f32>(&[d_out, d_in], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let (w, s, b) = crate::test_support::quantize_affine(&raw, gs, bits).unwrap();
        (w, s, b)
    };
    let make_sw = |d_in: i32, d_out: i32| -> (Array, Array, Array) {
        let raw = mlx_rs::random::normal::<f32>(&[n_experts, d_out, d_in], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let (w, s, b) = crate::test_support::quantize_affine(&raw, gs, bits).unwrap();
        (w, s, b)
    };

    struct GDNWeights {
        in_proj_qkvz: (Array, Array, Array),
        in_proj_ba: (Array, Array, Array),
        out_proj: (Array, Array, Array),
        conv_w: Array,
        a_log: Array,
        dt_bias: Array,
        norm_w: Array,
    }
    struct MoEWeights {
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
    struct AttnWeights {
        q_proj: (Array, Array, Array),
        k_proj: (Array, Array, Array),
        v_proj: (Array, Array, Array),
        o_proj: (Array, Array, Array),
    }

    let mut gdn_layers: Vec<Option<GDNWeights>> = Vec::new();
    let mut attn_layers: Vec<Option<AttnWeights>> = Vec::new();
    let mut moe_layers: Vec<MoEWeights> = Vec::new();
    let mut all_w: Vec<Array> = Vec::new();

    for i in 0..n_layers {
        let is_linear = (i + 1) % full_attn_interval != 0;
        if is_linear {
            let gdn = GDNWeights {
                in_proj_qkvz: make_qw(d, qkvz_out),
                in_proj_ba: make_qw(d, ba_out),
                out_proj: make_qw(value_dim, d),
                conv_w: mlx_rs::random::normal::<f32>(&[conv_dim, 4, 1], None, None, None)
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap(),
                a_log: mlx_rs::random::normal::<f32>(&[hv], None, None, None).unwrap(),
                dt_bias: mlx_rs::random::normal::<f32>(&[hv], None, None, None).unwrap(),
                norm_w: Array::ones::<f32>(&[dv])
                    .unwrap()
                    .as_dtype(Dtype::Float16)
                    .unwrap(),
            };
            all_w.extend([
                gdn.in_proj_qkvz.0.clone(),
                gdn.in_proj_qkvz.1.clone(),
                gdn.in_proj_qkvz.2.clone(),
            ]);
            all_w.extend([
                gdn.in_proj_ba.0.clone(),
                gdn.in_proj_ba.1.clone(),
                gdn.in_proj_ba.2.clone(),
            ]);
            all_w.extend([
                gdn.out_proj.0.clone(),
                gdn.out_proj.1.clone(),
                gdn.out_proj.2.clone(),
            ]);
            all_w.extend([
                gdn.conv_w.clone(),
                gdn.a_log.clone(),
                gdn.dt_bias.clone(),
                gdn.norm_w.clone(),
            ]);
            gdn_layers.push(Some(gdn));
            attn_layers.push(None);
        } else {
            let attn = AttnWeights {
                q_proj: make_qw(d, d),
                k_proj: make_qw(d, d),
                v_proj: make_qw(d, d),
                o_proj: make_qw(d, d),
            };
            all_w.extend([
                attn.q_proj.0.clone(),
                attn.q_proj.1.clone(),
                attn.q_proj.2.clone(),
            ]);
            all_w.extend([
                attn.k_proj.0.clone(),
                attn.k_proj.1.clone(),
                attn.k_proj.2.clone(),
            ]);
            all_w.extend([
                attn.v_proj.0.clone(),
                attn.v_proj.1.clone(),
                attn.v_proj.2.clone(),
            ]);
            all_w.extend([
                attn.o_proj.0.clone(),
                attn.o_proj.1.clone(),
                attn.o_proj.2.clone(),
            ]);
            gdn_layers.push(None);
            attn_layers.push(Some(attn));
        }
        let moe = MoEWeights {
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
        };
        all_w.extend([moe.gate.0.clone(), moe.gate.1.clone(), moe.gate.2.clone()]);
        all_w.extend([
            moe.sw_gate.0.clone(),
            moe.sw_gate.1.clone(),
            moe.sw_gate.2.clone(),
        ]);
        all_w.extend([
            moe.sw_up.0.clone(),
            moe.sw_up.1.clone(),
            moe.sw_up.2.clone(),
        ]);
        all_w.extend([
            moe.sw_down.0.clone(),
            moe.sw_down.1.clone(),
            moe.sw_down.2.clone(),
        ]);
        all_w.extend([
            moe.se_gate.0.clone(),
            moe.se_gate.1.clone(),
            moe.se_gate.2.clone(),
        ]);
        all_w.extend([
            moe.se_up.0.clone(),
            moe.se_up.1.clone(),
            moe.se_up.2.clone(),
        ]);
        all_w.extend([
            moe.se_down.0.clone(),
            moe.se_down.1.clone(),
            moe.se_down.2.clone(),
        ]);
        all_w.extend([
            moe.se_gate_proj.0.clone(),
            moe.se_gate_proj.1.clone(),
            moe.se_gate_proj.2.clone(),
        ]);
        all_w.push(moe.norm_w.clone());
        moe_layers.push(moe);
    }
    let refs: Vec<&Array> = all_w.iter().collect();
    mlx_rs::transforms::eval(refs).unwrap();

    let x = Array::ones::<f32>(&[1, 1, d])
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();
    let qk_norm_w = Array::ones::<f32>(&[dk]).unwrap();
    let inv_scale = Array::from_f32((dk as f32).sqrt().recip());
    let inv_scale_sq = {
        let s = (dk as f32).sqrt().recip();
        Array::from_f32(s * s)
    };
    let states: Vec<Array> = (0..36)
        .map(|_| {
            Array::zeros::<f32>(&[1, hv, dv, dk])
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap()
        })
        .collect();
    let conv_states: Vec<Array> = (0..36)
        .map(|_| {
            Array::zeros::<f32>(&[1, 3, conv_dim])
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap()
        })
        .collect();
    x.eval().unwrap();
    for s in &states {
        s.eval().unwrap();
    }
    for c in &conv_states {
        c.eval().unwrap();
    }

    let forward = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
        let mut h = h_in.clone();
        let mut gdn_idx = 0usize;

        for i in 0..n_layers as usize {
            let normed = fast::rms_norm(&h, Some(&moe_layers[i].norm_w), 1e-6).unwrap();

            // Attention
            let r = if gdn_layers[i].is_some() {
                let l = gdn_layers[i].as_ref().unwrap();
                let qkvz = quantized_forward(
                    &normed,
                    &l.in_proj_qkvz.0,
                    &l.in_proj_qkvz.1,
                    &l.in_proj_qkvz.2,
                    gs,
                    bits,
                )
                .unwrap();
                let ba = quantized_forward(
                    &normed,
                    &l.in_proj_ba.0,
                    &l.in_proj_ba.1,
                    &l.in_proj_ba.2,
                    gs,
                    bits,
                )
                .unwrap();
                let q = qkvz
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let k = qkvz
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let v = qkvz
                    .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();

                let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                let mixed = ops::concatenate(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                let conv_in = ops::concatenate(&[&cs[gdn_idx], &mixed], 1).unwrap();
                cs[gdn_idx] = conv_in.index((.., -3.., ..));
                let conv_out =
                    nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap()).unwrap();
                let conv_q = conv_out
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_k = conv_out
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_v = conv_out
                    .index((.., .., 2 * key_dim..))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();

                let norm_q = fast::rms_norm(&conv_q, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale_sq)
                    .unwrap();
                let norm_k = fast::rms_norm(&conv_k, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale)
                    .unwrap();

                let (y, new_state) = gated_delta_kernel_ffi(
                    &norm_q,
                    &norm_k,
                    &conv_v,
                    &l.a_log,
                    &a,
                    &l.dt_bias,
                    &b,
                    &ss[gdn_idx],
                    1,
                    1,
                    hk,
                    dk,
                    hv,
                    dv,
                )
                .unwrap();
                ss[gdn_idx] = new_state;
                gdn_idx += 1;

                let normed_y = fast::rms_norm(&y, Some(&l.norm_w), 1e-6).unwrap();
                let z_shaped = z
                    .index((.., .., ..value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let gated = swiglu(&z_shaped, &normed_y).unwrap();
                quantized_forward(
                    &gated.reshape(&[1, 1, -1]).unwrap(),
                    &l.out_proj.0,
                    &l.out_proj.1,
                    &l.out_proj.2,
                    gs,
                    bits,
                )
                .unwrap()
            } else {
                // Simplified attention: just qkvo matmuls
                let al = attn_layers[i].as_ref().unwrap();
                let q =
                    quantized_forward(&normed, &al.q_proj.0, &al.q_proj.1, &al.q_proj.2, gs, bits)
                        .unwrap();
                let _k =
                    quantized_forward(&normed, &al.k_proj.0, &al.k_proj.1, &al.k_proj.2, gs, bits)
                        .unwrap();
                let v =
                    quantized_forward(&normed, &al.v_proj.0, &al.v_proj.1, &al.v_proj.2, gs, bits)
                        .unwrap();
                let proxy = v
                    .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                    .unwrap();
                quantized_forward(&proxy, &al.o_proj.0, &al.o_proj.1, &al.o_proj.2, gs, bits)
                    .unwrap()
            };

            let h2 = h.add(r).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&moe_layers[i].norm_w), 1e-6).unwrap();

            // MoE
            let m = &moe_layers[i];
            let gate_out =
                quantized_forward(&normed2, &m.gate.0, &m.gate.1, &m.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let scores = raw_scores
                .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                .unwrap();

            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &m.sw_gate.0,
                &m.sw_gate.1,
                &m.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &m.sw_down.0,
                &m.sw_down.1,
                &m.sw_down.2,
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

            let sh_g =
                quantized_forward(&normed2, &m.se_gate.0, &m.se_gate.1, &m.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &m.se_down.0, &m.se_down.1, &m.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                quantized_forward(
                    &normed2,
                    &m.se_gate_proj.0,
                    &m.se_gate_proj.1,
                    &m.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(sh_gate_val).unwrap();

            h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
        }
        h
    };

    // Warmup
    for _ in 0..5 {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let result = forward(&x, &mut ss, &mut cs);
        let mut eval_targets: Vec<&Array> = vec![&result];
        eval_targets.extend(ss.iter());
        eval_targets.extend(cs.iter());
        mlx_rs::transforms::eval(eval_targets).unwrap();
    }

    // Benchmark
    let n = 20;
    let mut total_forward = 0u128;
    let mut total_eval = 0u128;
    for _ in 0..n {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let t0 = std::time::Instant::now();
        let result = forward(&x, &mut ss, &mut cs);
        let t1 = std::time::Instant::now();
        let mut eval_targets: Vec<&Array> = vec![&result];
        eval_targets.extend(ss.iter());
        eval_targets.extend(cs.iter());
        mlx_rs::transforms::eval(eval_targets).unwrap();
        let t2 = std::time::Instant::now();
        total_forward += (t1 - t0).as_nanos();
        total_eval += (t2 - t1).as_nanos();
    }

    let fwd_ms = total_forward as f64 / n as f64 / 1e6;
    let eval_ms = total_eval as f64 / n as f64 / 1e6;
    println!(
        "Rust 48 combined: forward={fwd_ms:.2}ms eval={eval_ms:.2}ms total={:.2}ms",
        fwd_ms + eval_ms
    );

    // Test: eval only the final result (not states) to see if eval target count matters
    let mut total_eval_one = 0u128;
    for _ in 0..n {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let result = forward(&x, &mut ss, &mut cs);
        let t0 = std::time::Instant::now();
        mlx_rs::transforms::eval([&result]).unwrap();
        total_eval_one += t0.elapsed().as_nanos();
    }
    let eval_one_ms = total_eval_one as f64 / n as f64 / 1e6;
    println!("Rust 48 combined (eval result only): {eval_one_ms:.2}ms");

    // Variant: GDN only (skip MoE, replace with passthrough)
    let forward_gdn_only = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
        let mut h = h_in.clone();
        let mut gdn_idx = 0usize;
        for i in 0..n_layers as usize {
            let is_gdn = gdn_layers[i].is_some();
            if !is_gdn {
                continue;
            }
            let normed = fast::rms_norm(&h, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let l = gdn_layers[i].as_ref().unwrap();
            let qkvz = quantized_forward(
                &normed,
                &l.in_proj_qkvz.0,
                &l.in_proj_qkvz.1,
                &l.in_proj_qkvz.2,
                gs,
                bits,
            )
            .unwrap();
            let ba = quantized_forward(
                &normed,
                &l.in_proj_ba.0,
                &l.in_proj_ba.1,
                &l.in_proj_ba.2,
                gs,
                bits,
            )
            .unwrap();
            let q = qkvz
                .index((.., .., ..key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let k = qkvz
                .index((.., .., key_dim..2 * key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let v = qkvz
                .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                .reshape(&[1, 1, hv, dv])
                .unwrap();
            let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
            let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
            let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
            let q_flat = q.reshape(&[1, 1, -1]).unwrap();
            let k_flat = k.reshape(&[1, 1, -1]).unwrap();
            let v_flat = v.reshape(&[1, 1, -1]).unwrap();
            let mixed = ops::concatenate(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
            let conv_in = ops::concatenate(&[&cs[gdn_idx], &mixed], 1).unwrap();
            cs[gdn_idx] = conv_in.index((.., -3.., ..));
            let conv_out =
                nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap()).unwrap();
            let conv_q = conv_out
                .index((.., .., ..key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let conv_k = conv_out
                .index((.., .., key_dim..2 * key_dim))
                .reshape(&[1, 1, hk, dk])
                .unwrap();
            let conv_v = conv_out
                .index((.., .., 2 * key_dim..))
                .reshape(&[1, 1, hv, dv])
                .unwrap();
            let norm_q = fast::rms_norm(&conv_q, Some(&qk_norm_w), 1e-6)
                .unwrap()
                .multiply(&inv_scale_sq)
                .unwrap();
            let norm_k = fast::rms_norm(&conv_k, Some(&qk_norm_w), 1e-6)
                .unwrap()
                .multiply(&inv_scale)
                .unwrap();
            let (y, new_state) = gated_delta_kernel_ffi(
                &norm_q,
                &norm_k,
                &conv_v,
                &l.a_log,
                &a,
                &l.dt_bias,
                &b,
                &ss[gdn_idx],
                1,
                1,
                hk,
                dk,
                hv,
                dv,
            )
            .unwrap();
            ss[gdn_idx] = new_state;
            gdn_idx += 1;
            let normed_y = fast::rms_norm(&y, Some(&l.norm_w), 1e-6).unwrap();
            let z_shaped = z
                .index((.., .., ..value_dim))
                .reshape(&[1, 1, hv, dv])
                .unwrap();
            let gated = swiglu(&z_shaped, &normed_y).unwrap();
            let r = quantized_forward(
                &gated.reshape(&[1, 1, -1]).unwrap(),
                &l.out_proj.0,
                &l.out_proj.1,
                &l.out_proj.2,
                gs,
                bits,
            )
            .unwrap();
            h = h.add(r).unwrap();
        }
        h
    };

    // Variant: MoE only (skip GDN)
    let forward_moe_only = |h_in: &Array| -> Array {
        let mut h = h_in.clone();
        for i in 0..n_layers as usize {
            let normed = fast::rms_norm(&h, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            // Simple attn proxy
            let attn_out = quantized_forward(
                &normed,
                &moe_layers[i].gate.0,
                &moe_layers[i].gate.1,
                &moe_layers[i].gate.2,
                gs,
                bits,
            )
            .unwrap();
            let h2 = h.add(attn_out.sum_axes(&[-1], true).unwrap()).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let m = &moe_layers[i];
            let gate_out =
                quantized_forward(&normed2, &m.gate.0, &m.gate.1, &m.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let scores = raw_scores
                .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                .unwrap();
            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &m.sw_gate.0,
                &m.sw_gate.1,
                &m.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &m.sw_down.0,
                &m.sw_down.1,
                &m.sw_down.2,
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
            let sh_g =
                quantized_forward(&normed2, &m.se_gate.0, &m.se_gate.1, &m.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &m.se_down.0, &m.se_down.1, &m.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                quantized_forward(
                    &normed2,
                    &m.se_gate_proj.0,
                    &m.se_gate_proj.1,
                    &m.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(sh_gate_val).unwrap();
            h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
        }
        h
    };

    // Warmup GDN-only
    for _ in 0..5 {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let r = forward_gdn_only(&x, &mut ss, &mut cs);
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        t.extend(cs.iter());
        mlx_rs::transforms::eval(t).unwrap();
    }
    let mut total_gdn = 0u128;
    for _ in 0..n {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let r = forward_gdn_only(&x, &mut ss, &mut cs);
        let t0 = std::time::Instant::now();
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        t.extend(cs.iter());
        mlx_rs::transforms::eval(t).unwrap();
        total_gdn += t0.elapsed().as_nanos();
    }
    println!(
        "Rust GDN-only (36 layers, combined weights): {:.2}ms",
        total_gdn as f64 / n as f64 / 1e6
    );

    // Warmup MoE-only
    for _ in 0..5 {
        let r = forward_moe_only(&x);
        mlx_rs::transforms::eval([&r]).unwrap();
    }
    let mut total_moe = 0u128;
    for _ in 0..n {
        let r = forward_moe_only(&x);
        let t0 = std::time::Instant::now();
        mlx_rs::transforms::eval([&r]).unwrap();
        total_moe += t0.elapsed().as_nanos();
    }
    println!(
        "Rust MoE-only (48 layers, combined weights): {:.2}ms",
        total_moe as f64 / n as f64 / 1e6
    );

    // Combined but with kernel replaced by zeros_like
    let forward_no_kernel = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
        let mut h = h_in.clone();
        let mut gdn_idx = 0usize;
        for i in 0..n_layers as usize {
            let normed = fast::rms_norm(&h, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let r = if gdn_layers[i].is_some() {
                let l = gdn_layers[i].as_ref().unwrap();
                let qkvz = quantized_forward(
                    &normed,
                    &l.in_proj_qkvz.0,
                    &l.in_proj_qkvz.1,
                    &l.in_proj_qkvz.2,
                    gs,
                    bits,
                )
                .unwrap();
                let ba = quantized_forward(
                    &normed,
                    &l.in_proj_ba.0,
                    &l.in_proj_ba.1,
                    &l.in_proj_ba.2,
                    gs,
                    bits,
                )
                .unwrap();
                let q = qkvz
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let k = qkvz
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let v = qkvz
                    .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                let mixed = ops::concatenate(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                let conv_in = ops::concatenate(&[&cs[gdn_idx], &mixed], 1).unwrap();
                cs[gdn_idx] = conv_in.index((.., -3.., ..));
                let conv_out =
                    nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap()).unwrap();
                let conv_q = conv_out
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_k = conv_out
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let _conv_v = conv_out
                    .index((.., .., 2 * key_dim..))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let _norm_q = fast::rms_norm(&conv_q, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale_sq)
                    .unwrap();
                let _norm_k = fast::rms_norm(&conv_k, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale)
                    .unwrap();
                let _g = compute_g_compiled((&l.a_log, &a, &l.dt_bias)).unwrap();
                let _beta = nn::sigmoid(&b).unwrap();

                // SKIP kernel: use zeros instead
                let y = Array::zeros::<f32>(&[1, 1, hv, dv])
                    .unwrap()
                    .as_dtype(mlx_rs::Dtype::Float16)
                    .unwrap();
                ss[gdn_idx] = Array::zeros::<f32>(&[1, hv, dv, dk])
                    .unwrap()
                    .as_dtype(mlx_rs::Dtype::Float16)
                    .unwrap();
                gdn_idx += 1;

                let normed_y = fast::rms_norm(&y, Some(&l.norm_w), 1e-6).unwrap();
                let z_shaped = z
                    .index((.., .., ..value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let gated = swiglu(&z_shaped, &normed_y).unwrap();
                quantized_forward(
                    &gated.reshape(&[1, 1, -1]).unwrap(),
                    &l.out_proj.0,
                    &l.out_proj.1,
                    &l.out_proj.2,
                    gs,
                    bits,
                )
                .unwrap()
            } else {
                let al = attn_layers[i].as_ref().unwrap();
                let q =
                    quantized_forward(&normed, &al.q_proj.0, &al.q_proj.1, &al.q_proj.2, gs, bits)
                        .unwrap();
                let _k =
                    quantized_forward(&normed, &al.k_proj.0, &al.k_proj.1, &al.k_proj.2, gs, bits)
                        .unwrap();
                let v =
                    quantized_forward(&normed, &al.v_proj.0, &al.v_proj.1, &al.v_proj.2, gs, bits)
                        .unwrap();
                let proxy = v
                    .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                    .unwrap();
                quantized_forward(&proxy, &al.o_proj.0, &al.o_proj.1, &al.o_proj.2, gs, bits)
                    .unwrap()
            };
            let h2 = h.add(r).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let m = &moe_layers[i];
            let gate_out =
                quantized_forward(&normed2, &m.gate.0, &m.gate.1, &m.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let scores = raw_scores
                .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                .unwrap();
            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &m.sw_gate.0,
                &m.sw_gate.1,
                &m.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &m.sw_down.0,
                &m.sw_down.1,
                &m.sw_down.2,
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
            let sh_g =
                quantized_forward(&normed2, &m.se_gate.0, &m.se_gate.1, &m.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &m.se_down.0, &m.se_down.1, &m.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                quantized_forward(
                    &normed2,
                    &m.se_gate_proj.0,
                    &m.se_gate_proj.1,
                    &m.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(sh_gate_val).unwrap();
            h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
        }
        h
    };

    for _ in 0..5 {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let r = forward_no_kernel(&x, &mut ss, &mut cs);
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        t.extend(cs.iter());
        mlx_rs::transforms::eval(t).unwrap();
    }
    let mut total_nk = 0u128;
    for _ in 0..n {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let r = forward_no_kernel(&x, &mut ss, &mut cs);
        let t0 = std::time::Instant::now();
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        t.extend(cs.iter());
        mlx_rs::transforms::eval(t).unwrap();
        total_nk += t0.elapsed().as_nanos();
    }
    println!(
        "Rust combined NO KERNEL (GDN ops + MoE): {:.2}ms",
        total_nk as f64 / n as f64 / 1e6
    );

    // Variant: ops-based GDN recurrence (no Metal kernel) interleaved with MoE
    let gqa_repeat = hv / hk;
    let forward_ops_gdn = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
        let mut h = h_in.clone();
        let mut gdn_idx = 0usize;
        for i in 0..n_layers as usize {
            let normed = fast::rms_norm(&h, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let r = if gdn_layers[i].is_some() {
                let l = gdn_layers[i].as_ref().unwrap();
                let qkvz = quantized_forward(
                    &normed,
                    &l.in_proj_qkvz.0,
                    &l.in_proj_qkvz.1,
                    &l.in_proj_qkvz.2,
                    gs,
                    bits,
                )
                .unwrap();
                let ba = quantized_forward(
                    &normed,
                    &l.in_proj_ba.0,
                    &l.in_proj_ba.1,
                    &l.in_proj_ba.2,
                    gs,
                    bits,
                )
                .unwrap();
                let q = qkvz
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let k = qkvz
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let v = qkvz
                    .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                let mixed = ops::concatenate(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                let conv_in = ops::concatenate(&[&cs[gdn_idx], &mixed], 1).unwrap();
                cs[gdn_idx] = conv_in.index((.., -3.., ..));
                let conv_out =
                    nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap()).unwrap();
                let conv_q = conv_out
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_k = conv_out
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_v = conv_out
                    .index((.., .., 2 * key_dim..))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let norm_q = fast::rms_norm(&conv_q, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale_sq)
                    .unwrap();
                let norm_k = fast::rms_norm(&conv_k, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale)
                    .unwrap();
                let g = compute_g_compiled((&l.a_log, &a, &l.dt_bias)).unwrap();
                let beta = nn::sigmoid(&b).unwrap();

                // Ops-based recurrence: repeat q,k for GQA then run step
                let q_rep = ops::broadcast_to(
                    norm_q.reshape(&[1, hk, 1, dk]).unwrap(),
                    &[1, hk, gqa_repeat, dk],
                )
                .unwrap()
                .reshape(&[1, hv, dk])
                .unwrap();
                let k_rep = ops::broadcast_to(
                    norm_k.reshape(&[1, hk, 1, dk]).unwrap(),
                    &[1, hk, gqa_repeat, dk],
                )
                .unwrap()
                .reshape(&[1, hv, dk])
                .unwrap();
                let v_sq = conv_v.squeeze_axes(&[1]).unwrap();
                let g_sq = g.squeeze_axes(&[0, 1]).unwrap();
                let beta_sq = beta.squeeze_axes(&[0, 1]).unwrap();
                let (y, new_state) =
                    gated_delta_step_ref(&q_rep, &k_rep, &v_sq, &g_sq, &beta_sq, &ss[gdn_idx]);
                ss[gdn_idx] = new_state;
                gdn_idx += 1;

                let y_4d = y.expand_dims(0).unwrap().expand_dims(0).unwrap();
                let normed_y = fast::rms_norm(&y_4d, Some(&l.norm_w), 1e-6).unwrap();
                let z_shaped = z
                    .index((.., .., ..value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let gated = swiglu(&z_shaped, &normed_y).unwrap();
                quantized_forward(
                    &gated.reshape(&[1, 1, -1]).unwrap(),
                    &l.out_proj.0,
                    &l.out_proj.1,
                    &l.out_proj.2,
                    gs,
                    bits,
                )
                .unwrap()
            } else {
                let al = attn_layers[i].as_ref().unwrap();
                let q =
                    quantized_forward(&normed, &al.q_proj.0, &al.q_proj.1, &al.q_proj.2, gs, bits)
                        .unwrap();
                let _k =
                    quantized_forward(&normed, &al.k_proj.0, &al.k_proj.1, &al.k_proj.2, gs, bits)
                        .unwrap();
                let v =
                    quantized_forward(&normed, &al.v_proj.0, &al.v_proj.1, &al.v_proj.2, gs, bits)
                        .unwrap();
                let proxy = v
                    .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                    .unwrap();
                quantized_forward(&proxy, &al.o_proj.0, &al.o_proj.1, &al.o_proj.2, gs, bits)
                    .unwrap()
            };
            let h2 = h.add(r).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let m = &moe_layers[i];
            let gate_out =
                quantized_forward(&normed2, &m.gate.0, &m.gate.1, &m.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let scores = raw_scores
                .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                .unwrap();
            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &m.sw_gate.0,
                &m.sw_gate.1,
                &m.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &m.sw_down.0,
                &m.sw_down.1,
                &m.sw_down.2,
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
            let sh_g =
                quantized_forward(&normed2, &m.se_gate.0, &m.se_gate.1, &m.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &m.se_down.0, &m.se_down.1, &m.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                quantized_forward(
                    &normed2,
                    &m.se_gate_proj.0,
                    &m.se_gate_proj.1,
                    &m.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(sh_gate_val).unwrap();
            h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
        }
        h
    };

    for _ in 0..5 {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let r = forward_ops_gdn(&x, &mut ss, &mut cs);
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        t.extend(cs.iter());
        mlx_rs::transforms::eval(t).unwrap();
    }
    let mut total_ops = 0u128;
    for _ in 0..n {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let r = forward_ops_gdn(&x, &mut ss, &mut cs);
        let t0 = std::time::Instant::now();
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        t.extend(cs.iter());
        mlx_rs::transforms::eval(t).unwrap();
        total_ops += t0.elapsed().as_nanos();
    }
    println!(
        "Rust combined OPS GDN (no Metal kernel): {:.2}ms",
        total_ops as f64 / n as f64 / 1e6
    );

    // Variant: Metal kernel with per-layer eval barriers
    let forward_eval_barrier = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
        let mut h = h_in.clone();
        let mut gdn_idx = 0usize;
        for i in 0..n_layers as usize {
            let normed = fast::rms_norm(&h, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let r = if gdn_layers[i].is_some() {
                let l = gdn_layers[i].as_ref().unwrap();
                let qkvz = quantized_forward(
                    &normed,
                    &l.in_proj_qkvz.0,
                    &l.in_proj_qkvz.1,
                    &l.in_proj_qkvz.2,
                    gs,
                    bits,
                )
                .unwrap();
                let ba = quantized_forward(
                    &normed,
                    &l.in_proj_ba.0,
                    &l.in_proj_ba.1,
                    &l.in_proj_ba.2,
                    gs,
                    bits,
                )
                .unwrap();
                let q = qkvz
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let k = qkvz
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let v = qkvz
                    .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                let mixed = ops::concatenate(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                let conv_in = ops::concatenate(&[&cs[gdn_idx], &mixed], 1).unwrap();
                cs[gdn_idx] = conv_in.index((.., -3.., ..));
                let conv_out =
                    nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap()).unwrap();
                let conv_q = conv_out
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_k = conv_out
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_v = conv_out
                    .index((.., .., 2 * key_dim..))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let norm_q = fast::rms_norm(&conv_q, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale_sq)
                    .unwrap();
                let norm_k = fast::rms_norm(&conv_k, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale)
                    .unwrap();
                let (y, new_state) = gated_delta_kernel_ffi(
                    &norm_q,
                    &norm_k,
                    &conv_v,
                    &l.a_log,
                    &a,
                    &l.dt_bias,
                    &b,
                    &ss[gdn_idx],
                    1,
                    1,
                    hk,
                    dk,
                    hv,
                    dv,
                )
                .unwrap();
                ss[gdn_idx] = new_state;
                gdn_idx += 1;
                let normed_y = fast::rms_norm(&y, Some(&l.norm_w), 1e-6).unwrap();
                let z_shaped = z
                    .index((.., .., ..value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let gated = swiglu(&z_shaped, &normed_y).unwrap();
                quantized_forward(
                    &gated.reshape(&[1, 1, -1]).unwrap(),
                    &l.out_proj.0,
                    &l.out_proj.1,
                    &l.out_proj.2,
                    gs,
                    bits,
                )
                .unwrap()
            } else {
                let al = attn_layers[i].as_ref().unwrap();
                let q =
                    quantized_forward(&normed, &al.q_proj.0, &al.q_proj.1, &al.q_proj.2, gs, bits)
                        .unwrap();
                let _k =
                    quantized_forward(&normed, &al.k_proj.0, &al.k_proj.1, &al.k_proj.2, gs, bits)
                        .unwrap();
                let v =
                    quantized_forward(&normed, &al.v_proj.0, &al.v_proj.1, &al.v_proj.2, gs, bits)
                        .unwrap();
                let proxy = v
                    .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                    .unwrap();
                quantized_forward(&proxy, &al.o_proj.0, &al.o_proj.1, &al.o_proj.2, gs, bits)
                    .unwrap()
            };
            let h2 = h.add(r).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let m = &moe_layers[i];
            let gate_out =
                quantized_forward(&normed2, &m.gate.0, &m.gate.1, &m.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let scores = raw_scores
                .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                .unwrap();
            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &m.sw_gate.0,
                &m.sw_gate.1,
                &m.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &m.sw_down.0,
                &m.sw_down.1,
                &m.sw_down.2,
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
            let sh_g =
                quantized_forward(&normed2, &m.se_gate.0, &m.se_gate.1, &m.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &m.se_down.0, &m.se_down.1, &m.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                quantized_forward(
                    &normed2,
                    &m.se_gate_proj.0,
                    &m.se_gate_proj.1,
                    &m.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(sh_gate_val).unwrap();
            h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();

            // Eval barrier: force layer-by-layer evaluation
            h.eval().unwrap();
            ss.iter().for_each(|s| s.eval().unwrap());
            cs.iter().for_each(|c| c.eval().unwrap());
        }
        h
    };

    for _ in 0..3 {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let r = forward_eval_barrier(&x, &mut ss, &mut cs);
        r.eval().unwrap();
    }
    let mut total_eb = 0u128;
    for _ in 0..n {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let t0 = std::time::Instant::now();
        let r = forward_eval_barrier(&x, &mut ss, &mut cs);
        r.eval().unwrap();
        total_eb += t0.elapsed().as_nanos();
    }
    println!(
        "Rust combined EVAL BARRIER (per-layer eval): {:.2}ms",
        total_eb as f64 / n as f64 / 1e6
    );

    // Variant: async_eval after each layer (non-blocking pipeline hint)
    let forward_async = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
        let mut h = h_in.clone();
        let mut gdn_idx = 0usize;
        for i in 0..n_layers as usize {
            let normed = fast::rms_norm(&h, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let r = if gdn_layers[i].is_some() {
                let l = gdn_layers[i].as_ref().unwrap();
                let qkvz = quantized_forward(
                    &normed,
                    &l.in_proj_qkvz.0,
                    &l.in_proj_qkvz.1,
                    &l.in_proj_qkvz.2,
                    gs,
                    bits,
                )
                .unwrap();
                let ba = quantized_forward(
                    &normed,
                    &l.in_proj_ba.0,
                    &l.in_proj_ba.1,
                    &l.in_proj_ba.2,
                    gs,
                    bits,
                )
                .unwrap();
                let q = qkvz
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let k = qkvz
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let v = qkvz
                    .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                let mixed = ops::concatenate(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                let conv_in = ops::concatenate(&[&cs[gdn_idx], &mixed], 1).unwrap();
                cs[gdn_idx] = conv_in.index((.., -3.., ..));
                let conv_out =
                    nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap()).unwrap();
                let conv_q = conv_out
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_k = conv_out
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_v = conv_out
                    .index((.., .., 2 * key_dim..))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let norm_q = fast::rms_norm(&conv_q, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale_sq)
                    .unwrap();
                let norm_k = fast::rms_norm(&conv_k, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale)
                    .unwrap();
                let (y, new_state) = gated_delta_kernel_ffi(
                    &norm_q,
                    &norm_k,
                    &conv_v,
                    &l.a_log,
                    &a,
                    &l.dt_bias,
                    &b,
                    &ss[gdn_idx],
                    1,
                    1,
                    hk,
                    dk,
                    hv,
                    dv,
                )
                .unwrap();
                ss[gdn_idx] = new_state;
                gdn_idx += 1;
                let normed_y = fast::rms_norm(&y, Some(&l.norm_w), 1e-6).unwrap();
                let z_shaped = z
                    .index((.., .., ..value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let gated = swiglu(&z_shaped, &normed_y).unwrap();
                quantized_forward(
                    &gated.reshape(&[1, 1, -1]).unwrap(),
                    &l.out_proj.0,
                    &l.out_proj.1,
                    &l.out_proj.2,
                    gs,
                    bits,
                )
                .unwrap()
            } else {
                let al = attn_layers[i].as_ref().unwrap();
                let q =
                    quantized_forward(&normed, &al.q_proj.0, &al.q_proj.1, &al.q_proj.2, gs, bits)
                        .unwrap();
                let _k =
                    quantized_forward(&normed, &al.k_proj.0, &al.k_proj.1, &al.k_proj.2, gs, bits)
                        .unwrap();
                let v =
                    quantized_forward(&normed, &al.v_proj.0, &al.v_proj.1, &al.v_proj.2, gs, bits)
                        .unwrap();
                let proxy = v
                    .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                    .unwrap();
                quantized_forward(&proxy, &al.o_proj.0, &al.o_proj.1, &al.o_proj.2, gs, bits)
                    .unwrap()
            };
            let h2 = h.add(r).unwrap();

            // Async eval hint: start processing GDN computation while building MoE graph
            mlx_rs::transforms::async_eval([&h2]).unwrap();

            let normed2 = fast::rms_norm(&h2, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let m = &moe_layers[i];
            let gate_out =
                quantized_forward(&normed2, &m.gate.0, &m.gate.1, &m.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let scores = raw_scores
                .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                .unwrap();
            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &m.sw_gate.0,
                &m.sw_gate.1,
                &m.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &m.sw_down.0,
                &m.sw_down.1,
                &m.sw_down.2,
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
            let sh_g =
                quantized_forward(&normed2, &m.se_gate.0, &m.se_gate.1, &m.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &m.se_down.0, &m.se_down.1, &m.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                quantized_forward(
                    &normed2,
                    &m.se_gate_proj.0,
                    &m.se_gate_proj.1,
                    &m.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(sh_gate_val).unwrap();
            h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
        }
        h
    };

    for _ in 0..3 {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let r = forward_async(&x, &mut ss, &mut cs);
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        t.extend(cs.iter());
        mlx_rs::transforms::eval(t).unwrap();
    }
    let mut total_async = 0u128;
    for _ in 0..n {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let t0 = std::time::Instant::now();
        let r = forward_async(&x, &mut ss, &mut cs);
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        t.extend(cs.iter());
        mlx_rs::transforms::eval(t).unwrap();
        total_async += t0.elapsed().as_nanos();
    }
    println!(
        "Rust combined ASYNC EVAL (per-layer hint): {:.2}ms",
        total_async as f64 / n as f64 / 1e6
    );

    // Variant: eval kernel outputs (y + state) immediately after each GDN layer
    let forward_eval_kernel = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
        let mut h = h_in.clone();
        let mut gdn_idx = 0usize;
        for i in 0..n_layers as usize {
            let normed = fast::rms_norm(&h, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let r = if gdn_layers[i].is_some() {
                let l = gdn_layers[i].as_ref().unwrap();
                let qkvz = quantized_forward(
                    &normed,
                    &l.in_proj_qkvz.0,
                    &l.in_proj_qkvz.1,
                    &l.in_proj_qkvz.2,
                    gs,
                    bits,
                )
                .unwrap();
                let ba = quantized_forward(
                    &normed,
                    &l.in_proj_ba.0,
                    &l.in_proj_ba.1,
                    &l.in_proj_ba.2,
                    gs,
                    bits,
                )
                .unwrap();
                let q = qkvz
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let k = qkvz
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let v = qkvz
                    .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                let mixed = ops::concatenate(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                let conv_in = ops::concatenate(&[&cs[gdn_idx], &mixed], 1).unwrap();
                cs[gdn_idx] = conv_in.index((.., -3.., ..));
                let conv_out =
                    nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap()).unwrap();
                let conv_q = conv_out
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_k = conv_out
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_v = conv_out
                    .index((.., .., 2 * key_dim..))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let norm_q = fast::rms_norm(&conv_q, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale_sq)
                    .unwrap();
                let norm_k = fast::rms_norm(&conv_k, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale)
                    .unwrap();
                let (y, new_state) = gated_delta_kernel_ffi(
                    &norm_q,
                    &norm_k,
                    &conv_v,
                    &l.a_log,
                    &a,
                    &l.dt_bias,
                    &b,
                    &ss[gdn_idx],
                    1,
                    1,
                    hk,
                    dk,
                    hv,
                    dv,
                )
                .unwrap();

                // Targeted eval: resolve kernel outputs to break graph
                mlx_rs::transforms::eval([&y, &new_state, &cs[gdn_idx]]).unwrap();

                ss[gdn_idx] = new_state;
                gdn_idx += 1;
                let normed_y = fast::rms_norm(&y, Some(&l.norm_w), 1e-6).unwrap();
                let z_shaped = z
                    .index((.., .., ..value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let gated = swiglu(&z_shaped, &normed_y).unwrap();
                quantized_forward(
                    &gated.reshape(&[1, 1, -1]).unwrap(),
                    &l.out_proj.0,
                    &l.out_proj.1,
                    &l.out_proj.2,
                    gs,
                    bits,
                )
                .unwrap()
            } else {
                let al = attn_layers[i].as_ref().unwrap();
                let q =
                    quantized_forward(&normed, &al.q_proj.0, &al.q_proj.1, &al.q_proj.2, gs, bits)
                        .unwrap();
                let _k =
                    quantized_forward(&normed, &al.k_proj.0, &al.k_proj.1, &al.k_proj.2, gs, bits)
                        .unwrap();
                let v =
                    quantized_forward(&normed, &al.v_proj.0, &al.v_proj.1, &al.v_proj.2, gs, bits)
                        .unwrap();
                let proxy = v
                    .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                    .unwrap();
                quantized_forward(&proxy, &al.o_proj.0, &al.o_proj.1, &al.o_proj.2, gs, bits)
                    .unwrap()
            };
            let h2 = h.add(r).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let m = &moe_layers[i];
            let gate_out =
                quantized_forward(&normed2, &m.gate.0, &m.gate.1, &m.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let scores = raw_scores
                .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                .unwrap();
            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &m.sw_gate.0,
                &m.sw_gate.1,
                &m.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &m.sw_down.0,
                &m.sw_down.1,
                &m.sw_down.2,
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
            let sh_g =
                quantized_forward(&normed2, &m.se_gate.0, &m.se_gate.1, &m.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &m.se_down.0, &m.se_down.1, &m.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                quantized_forward(
                    &normed2,
                    &m.se_gate_proj.0,
                    &m.se_gate_proj.1,
                    &m.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(sh_gate_val).unwrap();
            h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
        }
        h
    };

    for _ in 0..3 {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let r = forward_eval_kernel(&x, &mut ss, &mut cs);
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        t.extend(cs.iter());
        mlx_rs::transforms::eval(t).unwrap();
    }
    let mut total_ek = 0u128;
    for _ in 0..n {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let t0 = std::time::Instant::now();
        let r = forward_eval_kernel(&x, &mut ss, &mut cs);
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        t.extend(cs.iter());
        mlx_rs::transforms::eval(t).unwrap();
        total_ek += t0.elapsed().as_nanos();
    }
    println!(
        "Rust combined EVAL KERNEL OUTPUTS: {:.2}ms",
        total_ek as f64 / n as f64 / 1e6
    );

    // Layer scaling test: run with 1, 4, 12, 24, 48 layers to check non-linearity
    // Test: tiny state (replace [1,32,128,128] with [1,1,1,1]) to check memory hypothesis
    let tiny_states: Vec<Array> = (0..36)
        .map(|_| {
            Array::zeros::<f32>(&[1, 1, 1, 1])
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap()
        })
        .collect();
    for s in &tiny_states {
        s.eval().unwrap();
    }

    let forward_tiny_state = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
        let mut h = h_in.clone();
        let mut gdn_idx = 0usize;
        for i in 0..n_layers as usize {
            let normed = fast::rms_norm(&h, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let r = if gdn_layers[i].is_some() {
                let l = gdn_layers[i].as_ref().unwrap();
                let qkvz = quantized_forward(
                    &normed,
                    &l.in_proj_qkvz.0,
                    &l.in_proj_qkvz.1,
                    &l.in_proj_qkvz.2,
                    gs,
                    bits,
                )
                .unwrap();
                let ba = quantized_forward(
                    &normed,
                    &l.in_proj_ba.0,
                    &l.in_proj_ba.1,
                    &l.in_proj_ba.2,
                    gs,
                    bits,
                )
                .unwrap();
                let q = qkvz
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let k = qkvz
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let v = qkvz
                    .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                let mixed = ops::concatenate(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                let conv_in = ops::concatenate(&[&cs[gdn_idx], &mixed], 1).unwrap();
                cs[gdn_idx] = conv_in.index((.., -3.., ..));
                let conv_out =
                    nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap()).unwrap();
                let conv_q = conv_out
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_k = conv_out
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_v = conv_out
                    .index((.., .., 2 * key_dim..))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let _norm_q = fast::rms_norm(&conv_q, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale_sq)
                    .unwrap();
                let _norm_k = fast::rms_norm(&conv_k, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale)
                    .unwrap();
                let g = compute_g_compiled((&l.a_log, &a, &l.dt_bias)).unwrap();
                let beta = nn::sigmoid(&b).unwrap();

                // Tiny state: just multiply by a scalar instead of full state ops
                let g_scalar = g.sum_axes(&[-1], true).unwrap();
                let tiny_decayed = ss[gdn_idx].multiply(g_scalar).unwrap();
                ss[gdn_idx] = tiny_decayed.add(Array::from_f32(0.1)).unwrap();

                // Use conv_v directly as y (same shape [1,1,Hv,Dv])
                let y = conv_v
                    .multiply(beta.reshape(&[1, 1, hv, 1]).unwrap())
                    .unwrap();

                gdn_idx += 1;
                let normed_y = fast::rms_norm(&y, Some(&l.norm_w), 1e-6).unwrap();
                let z_shaped = z
                    .index((.., .., ..value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let gated = swiglu(&z_shaped, &normed_y).unwrap();
                quantized_forward(
                    &gated.reshape(&[1, 1, -1]).unwrap(),
                    &l.out_proj.0,
                    &l.out_proj.1,
                    &l.out_proj.2,
                    gs,
                    bits,
                )
                .unwrap()
            } else {
                let al = attn_layers[i].as_ref().unwrap();
                let q =
                    quantized_forward(&normed, &al.q_proj.0, &al.q_proj.1, &al.q_proj.2, gs, bits)
                        .unwrap();
                let _k =
                    quantized_forward(&normed, &al.k_proj.0, &al.k_proj.1, &al.k_proj.2, gs, bits)
                        .unwrap();
                let v =
                    quantized_forward(&normed, &al.v_proj.0, &al.v_proj.1, &al.v_proj.2, gs, bits)
                        .unwrap();
                let proxy = v
                    .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                    .unwrap();
                quantized_forward(&proxy, &al.o_proj.0, &al.o_proj.1, &al.o_proj.2, gs, bits)
                    .unwrap()
            };
            let h2 = h.add(r).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let m = &moe_layers[i];
            let gate_out =
                quantized_forward(&normed2, &m.gate.0, &m.gate.1, &m.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let scores = raw_scores
                .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                .unwrap();
            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &m.sw_gate.0,
                &m.sw_gate.1,
                &m.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &m.sw_down.0,
                &m.sw_down.1,
                &m.sw_down.2,
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
            let sh_g =
                quantized_forward(&normed2, &m.se_gate.0, &m.se_gate.1, &m.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &m.se_down.0, &m.se_down.1, &m.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                quantized_forward(
                    &normed2,
                    &m.se_gate_proj.0,
                    &m.se_gate_proj.1,
                    &m.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(sh_gate_val).unwrap();
            h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
        }
        h
    };

    for _ in 0..5 {
        let mut ss = tiny_states.clone();
        let mut cs = conv_states.clone();
        let r = forward_tiny_state(&x, &mut ss, &mut cs);
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        t.extend(cs.iter());
        mlx_rs::transforms::eval(t).unwrap();
    }
    let mut total_ts = 0u128;
    for _ in 0..n {
        let mut ss = tiny_states.clone();
        let mut cs = conv_states.clone();
        let r = forward_tiny_state(&x, &mut ss, &mut cs);
        let t0 = std::time::Instant::now();
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        t.extend(cs.iter());
        mlx_rs::transforms::eval(t).unwrap();
        total_ts += t0.elapsed().as_nanos();
    }
    println!(
        "Rust combined TINY STATE (all ops, no large state): {:.2}ms",
        total_ts as f64 / n as f64 / 1e6
    );

    for test_layers in [1i32, 4, 12, 24, 48] {
        let test_layers_u = test_layers as usize;
        let n_gdn = (0..test_layers_u)
            .filter(|i| gdn_layers.get(*i).map_or(false, |g| g.is_some()))
            .count();
        let forward_n = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
            let mut h = h_in.clone();
            let mut gdn_idx = 0usize;
            for i in 0..test_layers_u {
                let normed = fast::rms_norm(&h, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
                let r = if gdn_layers[i].is_some() {
                    let l = gdn_layers[i].as_ref().unwrap();
                    let qkvz = quantized_forward(
                        &normed,
                        &l.in_proj_qkvz.0,
                        &l.in_proj_qkvz.1,
                        &l.in_proj_qkvz.2,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let ba = quantized_forward(
                        &normed,
                        &l.in_proj_ba.0,
                        &l.in_proj_ba.1,
                        &l.in_proj_ba.2,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let q = qkvz
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let k = qkvz
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let v = qkvz
                        .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                    let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                    let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                    let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                    let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                    let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                    let mixed = ops::concatenate(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                    let conv_in = ops::concatenate(&[&cs[gdn_idx], &mixed], 1).unwrap();
                    cs[gdn_idx] = conv_in.index((.., -3.., ..));
                    let conv_out =
                        nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap())
                            .unwrap();
                    let conv_q = conv_out
                        .index((.., .., ..key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_k = conv_out
                        .index((.., .., key_dim..2 * key_dim))
                        .reshape(&[1, 1, hk, dk])
                        .unwrap();
                    let conv_v = conv_out
                        .index((.., .., 2 * key_dim..))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let norm_q = fast::rms_norm(&conv_q, Some(&qk_norm_w), 1e-6)
                        .unwrap()
                        .multiply(&inv_scale_sq)
                        .unwrap();
                    let norm_k = fast::rms_norm(&conv_k, Some(&qk_norm_w), 1e-6)
                        .unwrap()
                        .multiply(&inv_scale)
                        .unwrap();
                    let (y, new_state) = gated_delta_kernel_ffi(
                        &norm_q,
                        &norm_k,
                        &conv_v,
                        &l.a_log,
                        &a,
                        &l.dt_bias,
                        &b,
                        &ss[gdn_idx],
                        1,
                        1,
                        hk,
                        dk,
                        hv,
                        dv,
                    )
                    .unwrap();
                    ss[gdn_idx] = new_state;
                    gdn_idx += 1;
                    let normed_y = fast::rms_norm(&y, Some(&l.norm_w), 1e-6).unwrap();
                    let z_shaped = z
                        .index((.., .., ..value_dim))
                        .reshape(&[1, 1, hv, dv])
                        .unwrap();
                    let gated = swiglu(&z_shaped, &normed_y).unwrap();
                    quantized_forward(
                        &gated.reshape(&[1, 1, -1]).unwrap(),
                        &l.out_proj.0,
                        &l.out_proj.1,
                        &l.out_proj.2,
                        gs,
                        bits,
                    )
                    .unwrap()
                } else {
                    let al = attn_layers[i].as_ref().unwrap();
                    let q = quantized_forward(
                        &normed,
                        &al.q_proj.0,
                        &al.q_proj.1,
                        &al.q_proj.2,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let _k = quantized_forward(
                        &normed,
                        &al.k_proj.0,
                        &al.k_proj.1,
                        &al.k_proj.2,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let v = quantized_forward(
                        &normed,
                        &al.v_proj.0,
                        &al.v_proj.1,
                        &al.v_proj.2,
                        gs,
                        bits,
                    )
                    .unwrap();
                    let proxy = v
                        .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                        .unwrap();
                    quantized_forward(&proxy, &al.o_proj.0, &al.o_proj.1, &al.o_proj.2, gs, bits)
                        .unwrap()
                };
                let h2 = h.add(r).unwrap();
                let normed2 = fast::rms_norm(&h2, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
                let m = &moe_layers[i];
                let gate_out =
                    quantized_forward(&normed2, &m.gate.0, &m.gate.1, &m.gate.2, gs, bits).unwrap();
                let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
                let neg_k = -top_k;
                let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
                let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
                let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
                let scores = raw_scores
                    .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                    .unwrap();
                let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
                let g_out = gather_qmm(
                    &x_exp,
                    &m.sw_gate.0,
                    &m.sw_gate.1,
                    &m.sw_gate.2,
                    &top_inds,
                    true,
                    gs,
                    bits,
                    false,
                )
                .unwrap();
                let u_out = gather_qmm(
                    &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
                )
                .unwrap();
                let activated = swiglu(&g_out, &u_out).unwrap();
                let d_out = gather_qmm(
                    &activated,
                    &m.sw_down.0,
                    &m.sw_down.1,
                    &m.sw_down.2,
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
                let sh_g =
                    quantized_forward(&normed2, &m.se_gate.0, &m.se_gate.1, &m.se_gate.2, gs, bits)
                        .unwrap();
                let sh_u =
                    quantized_forward(&normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, gs, bits)
                        .unwrap();
                let sh_act = swiglu(&sh_g, &sh_u).unwrap();
                let sh_d =
                    quantized_forward(&sh_act, &m.se_down.0, &m.se_down.1, &m.se_down.2, gs, bits)
                        .unwrap();
                let sh_gate_val = nn::sigmoid(
                    quantized_forward(
                        &normed2,
                        &m.se_gate_proj.0,
                        &m.se_gate_proj.1,
                        &m.se_gate_proj.2,
                        gs,
                        bits,
                    )
                    .unwrap(),
                )
                .unwrap();
                let shared_out = sh_d.multiply(sh_gate_val).unwrap();
                h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
            }
            h
        };
        for _ in 0..3 {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_n(&x, &mut ss, &mut cs);
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
        }
        let mut total_n = 0u128;
        for _ in 0..n {
            let mut ss = states.clone();
            let mut cs = conv_states.clone();
            let r = forward_n(&x, &mut ss, &mut cs);
            let t0 = std::time::Instant::now();
            let mut t: Vec<&Array> = vec![&r];
            t.extend(ss.iter());
            t.extend(cs.iter());
            mlx_rs::transforms::eval(t).unwrap();
            total_n += t0.elapsed().as_nanos();
        }
        let ms = total_n as f64 / n as f64 / 1e6;
        println!(
            "Layer scaling: {test_layers} layers ({n_gdn} GDN): {ms:.2}ms ({:.2}ms/layer)",
            ms / test_layers as f64
        );
    }

    // Variant: replace recurrence with a single matmul (same data flow, fewer ops)
    let forward_matmul_gdn = |h_in: &Array, ss: &mut Vec<Array>, cs: &mut Vec<Array>| -> Array {
        let mut h = h_in.clone();
        let mut gdn_idx = 0usize;
        for i in 0..n_layers as usize {
            let normed = fast::rms_norm(&h, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let r = if gdn_layers[i].is_some() {
                let l = gdn_layers[i].as_ref().unwrap();
                let qkvz = quantized_forward(
                    &normed,
                    &l.in_proj_qkvz.0,
                    &l.in_proj_qkvz.1,
                    &l.in_proj_qkvz.2,
                    gs,
                    bits,
                )
                .unwrap();
                let ba = quantized_forward(
                    &normed,
                    &l.in_proj_ba.0,
                    &l.in_proj_ba.1,
                    &l.in_proj_ba.2,
                    gs,
                    bits,
                )
                .unwrap();
                let q = qkvz
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let k = qkvz
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let v = qkvz
                    .index((.., .., 2 * key_dim..2 * key_dim + value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let z = qkvz.index((.., .., 2 * key_dim + value_dim..));
                let b = ba.index((.., .., ..hv)).reshape(&[1, 1, hv]).unwrap();
                let a = ba.index((.., .., hv..)).reshape(&[1, 1, hv]).unwrap();
                let q_flat = q.reshape(&[1, 1, -1]).unwrap();
                let k_flat = k.reshape(&[1, 1, -1]).unwrap();
                let v_flat = v.reshape(&[1, 1, -1]).unwrap();
                let mixed = ops::concatenate(&[&q_flat, &k_flat, &v_flat], -1).unwrap();
                let conv_in = ops::concatenate(&[&cs[gdn_idx], &mixed], 1).unwrap();
                cs[gdn_idx] = conv_in.index((.., -3.., ..));
                let conv_out =
                    nn::silu(ops::conv1d(&conv_in, &l.conv_w, 1, 0, 1, conv_dim).unwrap()).unwrap();
                let conv_q = conv_out
                    .index((.., .., ..key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_k = conv_out
                    .index((.., .., key_dim..2 * key_dim))
                    .reshape(&[1, 1, hk, dk])
                    .unwrap();
                let conv_v = conv_out
                    .index((.., .., 2 * key_dim..))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let _norm_q = fast::rms_norm(&conv_q, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale_sq)
                    .unwrap();
                let _norm_k = fast::rms_norm(&conv_k, Some(&qk_norm_w), 1e-6)
                    .unwrap()
                    .multiply(&inv_scale)
                    .unwrap();
                let g = compute_g_compiled((&l.a_log, &a, &l.dt_bias)).unwrap();
                let _beta = nn::sigmoid(&b).unwrap();

                // Variant A: no reduction, just multiply + add on state
                let g_exp = g.reshape(&[1, hv, 1, 1]).unwrap();
                let decayed = ss[gdn_idx].multiply(g_exp).unwrap();
                let v_exp = conv_v.reshape(&[1, hv, dv, 1]).unwrap();
                ss[gdn_idx] = decayed.add(v_exp).unwrap();
                // y = just take a slice of state (no reduction)
                let y_proxy = ss[gdn_idx]
                    .index((.., .., .., 0..1))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                gdn_idx += 1;

                let normed_y = fast::rms_norm(&y_proxy, Some(&l.norm_w), 1e-6).unwrap();
                let z_shaped = z
                    .index((.., .., ..value_dim))
                    .reshape(&[1, 1, hv, dv])
                    .unwrap();
                let gated = swiglu(&z_shaped, &normed_y).unwrap();
                quantized_forward(
                    &gated.reshape(&[1, 1, -1]).unwrap(),
                    &l.out_proj.0,
                    &l.out_proj.1,
                    &l.out_proj.2,
                    gs,
                    bits,
                )
                .unwrap()
            } else {
                let al = attn_layers[i].as_ref().unwrap();
                let q =
                    quantized_forward(&normed, &al.q_proj.0, &al.q_proj.1, &al.q_proj.2, gs, bits)
                        .unwrap();
                let _k =
                    quantized_forward(&normed, &al.k_proj.0, &al.k_proj.1, &al.k_proj.2, gs, bits)
                        .unwrap();
                let v =
                    quantized_forward(&normed, &al.v_proj.0, &al.v_proj.1, &al.v_proj.2, gs, bits)
                        .unwrap();
                let proxy = v
                    .multiply(nn::sigmoid(&q.sum_axes(&[-1], true).unwrap()).unwrap())
                    .unwrap();
                quantized_forward(&proxy, &al.o_proj.0, &al.o_proj.1, &al.o_proj.2, gs, bits)
                    .unwrap()
            };
            let h2 = h.add(r).unwrap();
            let normed2 = fast::rms_norm(&h2, Some(&moe_layers[i].norm_w), 1e-6).unwrap();
            let m = &moe_layers[i];
            let gate_out =
                quantized_forward(&normed2, &m.gate.0, &m.gate.1, &m.gate.2, gs, bits).unwrap();
            let gates = ops::softmax_axis(&gate_out, -1, true).unwrap();
            let neg_k = -top_k;
            let all_inds = ops::argpartition_axis(&gates, neg_k, -1).unwrap();
            let top_inds = all_inds.index((.., .., (n_experts + neg_k)..));
            let raw_scores = gates.take_along_axis(&top_inds, -1).unwrap();
            let scores = raw_scores
                .divide(raw_scores.sum_axes(&[-1], true).unwrap())
                .unwrap();
            let x_exp = normed2.expand_dims(-2).unwrap().expand_dims(-2).unwrap();
            let g_out = gather_qmm(
                &x_exp,
                &m.sw_gate.0,
                &m.sw_gate.1,
                &m.sw_gate.2,
                &top_inds,
                true,
                gs,
                bits,
                false,
            )
            .unwrap();
            let u_out = gather_qmm(
                &x_exp, &m.sw_up.0, &m.sw_up.1, &m.sw_up.2, &top_inds, true, gs, bits, false,
            )
            .unwrap();
            let activated = swiglu(&g_out, &u_out).unwrap();
            let d_out = gather_qmm(
                &activated,
                &m.sw_down.0,
                &m.sw_down.1,
                &m.sw_down.2,
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
            let sh_g =
                quantized_forward(&normed2, &m.se_gate.0, &m.se_gate.1, &m.se_gate.2, gs, bits)
                    .unwrap();
            let sh_u =
                quantized_forward(&normed2, &m.se_up.0, &m.se_up.1, &m.se_up.2, gs, bits).unwrap();
            let sh_act = swiglu(&sh_g, &sh_u).unwrap();
            let sh_d =
                quantized_forward(&sh_act, &m.se_down.0, &m.se_down.1, &m.se_down.2, gs, bits)
                    .unwrap();
            let sh_gate_val = nn::sigmoid(
                quantized_forward(
                    &normed2,
                    &m.se_gate_proj.0,
                    &m.se_gate_proj.1,
                    &m.se_gate_proj.2,
                    gs,
                    bits,
                )
                .unwrap(),
            )
            .unwrap();
            let shared_out = sh_d.multiply(sh_gate_val).unwrap();
            h = h2.add(expert_sum).unwrap().add(shared_out).unwrap();
        }
        h
    };

    for _ in 0..5 {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let r = forward_matmul_gdn(&x, &mut ss, &mut cs);
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        t.extend(cs.iter());
        mlx_rs::transforms::eval(t).unwrap();
    }
    let mut total_mm = 0u128;
    for _ in 0..n {
        let mut ss = states.clone();
        let mut cs = conv_states.clone();
        let r = forward_matmul_gdn(&x, &mut ss, &mut cs);
        let t0 = std::time::Instant::now();
        let mut t: Vec<&Array> = vec![&r];
        t.extend(ss.iter());
        t.extend(cs.iter());
        mlx_rs::transforms::eval(t).unwrap();
        total_mm += t0.elapsed().as_nanos();
    }
    println!(
        "Rust combined MATMUL GDN (proxy recurrence): {:.2}ms",
        total_mm as f64 / n as f64 / 1e6
    );
}
