//! Smoke test: MQA SDPA with `qk_dim=576`, `v_dim=512` on Metal (MLA absorbed shapes).
#![allow(clippy::unwrap_used, clippy::print_stdout)]
use mlx_rs::{Array, fast, ops, random::normal, transforms::eval};

fn main() {
    // Decode shape: q [1, H=16, 1, 576], k [1, 1, S, 576], v [1, 1, S, 512]
    for s in [128, 4096] {
        let q = normal::<f32>(&[1, 16, 1, 576], None, None, None).unwrap();
        let k = normal::<f32>(&[1, 1, s, 576], None, None, None).unwrap();
        let v = normal::<f32>(&[1, 1, s, 512], None, None, None).unwrap();
        let scale = 1.0 / (576.0f32).sqrt();
        match fast::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            scale,
            None::<fast::ScaledDotProductAttentionMask>,
            None::<&Array>,
        ) {
            Ok(out) => {
                if let Err(e) = eval([&out]) {
                    println!("S={s}: EVAL FAILED: {e}");
                    continue;
                }
                println!("S={s}: OK shape={:?}", out.shape());
                // Cross-check against explicit softmax path.
                let attn = ops::softmax_axis(
                    q.matmul(k.transpose_axes(&[0, 1, 3, 2]).unwrap())
                        .unwrap()
                        .multiply(Array::from_f32(scale))
                        .unwrap(),
                    -1,
                    true,
                )
                .unwrap();
                let reference = attn.matmul(&v).unwrap();
                let diff = ops::abs(out.subtract(&reference).unwrap())
                    .unwrap()
                    .max(None)
                    .unwrap();
                eval([&diff]).unwrap();
                println!("S={s}: max|sdpa-explicit|={}", diff.item::<f32>());
            }
            Err(e) => println!("S={s}: SDPA REJECTED: {e}"),
        }
    }
    // Prefill shape with causal mask: q [1, 16, T, 576]
    let t = 64;
    let q = normal::<f32>(&[1, 16, t, 576], None, None, None).unwrap();
    let k = normal::<f32>(&[1, 1, t, 576], None, None, None).unwrap();
    let v = normal::<f32>(&[1, 1, t, 512], None, None, None).unwrap();
    let scale = 1.0 / (576.0f32).sqrt();
    match fast::scaled_dot_product_attention(
        &q,
        &k,
        &v,
        scale,
        Some(fast::ScaledDotProductAttentionMask::Causal),
        None::<&Array>,
    ) {
        Ok(out) => {
            eval([&out]).unwrap();
            println!("prefill T={t} causal: OK shape={:?}", out.shape());
        }
        Err(e) => println!("prefill causal: SDPA REJECTED: {e}"),
    }
}
