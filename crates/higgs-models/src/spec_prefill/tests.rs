//! Tests for the model-free PFlash selection half (steps 4-6).
//!
//! These prove the selection logic without loading any model — they cannot
//! OOM and need no GPU. The scorer half (step 1-3) gets its own gated tests
//! once implemented (DESIGN §5.4 asserts the ~75 MB memory bound).

#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::shadow_unrelated,
    clippy::shadow_reuse
)]

use super::*;
use mlx_rs::{Dtype, random};

fn needle_importance(s: usize, needle_pos: usize) -> Vec<f32> {
    // Flat low-importance background; one clearly salient needle block.
    let mut v = vec![0.01_f32; s];
    for i in needle_pos..(needle_pos + 16).min(s) {
        v[i] = 1.0;
    }
    v
}

#[test]
fn smooth_importance_is_length_preserving_and_peaks_at_needle() {
    let imp = needle_importance(512, 256);
    let sm = smooth_importance(&imp, 13).unwrap();
    assert_eq!(sm.len(), imp.len());
    // The smoothed peak is still at / next to the needle center.
    let peak = sm.iter().copied().fold(0.0_f32, f32::max);
    let peak_idx = sm.iter().position(|x| (*x - peak).abs() < 1e-6).unwrap();
    assert!(
        (240..=272).contains(&peak_idx),
        "peak at {peak_idx}, expected near needle (256)"
    );
}

#[test]
fn smooth_importance_rejects_even_kernel() {
    let imp = vec![0.0_f32; 8];
    assert!(smooth_importance(&imp, 12).is_err());
    assert!(smooth_importance(&imp, 0).is_err());
    assert!(smooth_importance(&imp, 1).is_ok());
}

#[test]
fn select_survivors_keeps_needle_block_at_aggressive_keep_ratio() {
    // The whole point of SpecPrefill-Full-LAH: a salient needle must be in the
    // survival mask even at keep_ratio = 0.10. (Our naive per-token scorer
    // failed exactly this — see RESEARCH §5.3. The block-max selection here is
    // what makes the needle survive once the scorer ranks it above background.)
    let s: usize = 4096;
    let chunk = 32;
    let needle_pos: usize = 2048; // mid-prompt
    let tokens: Vec<u32> = (0..s).map(|i| i as u32 % 1000).collect();
    let imp = needle_importance(s, needle_pos);
    let cfg = PrefillScoreConfig {
        keep_ratio: 0.10,
        chunk,
        avgpool: 13,
        lookahead: 8,
    };
    let plan = select_survivors(&tokens, &imp, &cfg).unwrap();
    let kept_positions: std::collections::HashSet<i32> =
        plan.original_positions.iter().copied().collect();
    // Every needle token survives.
    for i in needle_pos..(needle_pos + 16) {
        assert!(
            kept_positions.contains(&(i as i32)),
            "needle token {i} dropped at keep=0.10"
        );
    }
    // ~10% keep ratio: plan length is near keep_ratio * s (plus the two forced blocks).
    let expected = (0.10 * s as f32) as usize;
    assert!(
        plan.len() <= expected + 2 * chunk,
        "plan kept {} tokens, expected ~{expected} (+2 forced blocks)",
        plan.len()
    );
}

#[test]
fn select_survivors_always_keeps_sink_and_final_token_blocks() {
    let s: usize = 1024;
    let tokens: Vec<u32> = (0..s).map(|i| i as u32).collect();
    // Importance concentrated in the middle — sink and tail would otherwise lose.
    let mut imp = vec![0.0_f32; s];
    for i in 400..500 {
        imp[i] = 1.0;
    }
    let cfg = PrefillScoreConfig {
        keep_ratio: 0.10,
        chunk: 32,
        avgpool: 13,
        lookahead: 8,
    };
    let plan = select_survivors(&tokens, &imp, &cfg).unwrap();
    let kept: std::collections::HashSet<i32> = plan.original_positions.iter().copied().collect();
    // First token (BOS / system-prompt anchor) and last token (sampled logits).
    assert!(kept.contains(&0), "sink token 0 dropped");
    assert!(kept.contains(&((s - 1) as i32)), "final token dropped");
}

#[test]
fn select_survivors_preserves_original_order_and_positions() {
    let s: usize = 256;
    let tokens: Vec<u32> = (1000..(1000 + s)).map(|x| x as u32).collect();
    let imp = needle_importance(s, 128);
    let plan = select_survivors(&tokens, &imp, &PrefillScoreConfig::default()).unwrap();
    // Positions strictly increasing; token ids match tokens[position].
    assert!(plan.original_positions.windows(2).all(|w| w[0] < w[1]));
    for (tok, pos) in plan.token_ids.iter().zip(plan.original_positions.iter()) {
        assert_eq!(*tok, tokens[*pos as usize]);
    }
}

#[test]
fn select_survivors_rejects_bad_inputs() {
    let t = vec![0_u32; 4];
    let i = vec![0.0_f32; 4];
    // length mismatch
    assert!(select_survivors(&t, &i[..3], &PrefillScoreConfig::default()).is_err());
    // keep_ratio out of range
    let bad = PrefillScoreConfig {
        keep_ratio: 0.90,
        ..PrefillScoreConfig::default()
    };
    assert!(select_survivors(&t, &i, &bad).is_err());
    // chunk = 0
    let bad_chunk = PrefillScoreConfig {
        chunk: 0,
        ..PrefillScoreConfig::default()
    };
    assert!(select_survivors(&t, &i, &bad_chunk).is_err());
}

#[test]
fn layer_importance_shape_and_range() {
    // Memory-safety smoke: the scorer must produce [S] from [H, lah, d] x
    // [Hkv, S, d] without materializing [H, S, S]. S=2048 is enough to OOM the
    // naive form; this runs in microseconds.
    let n_heads = 16;
    let n_kv_heads = 8;
    let head_dim = 128;
    let lah = 9; // lookahead + final prompt token
    let s = 2048;
    let q = random::uniform::<f32, f32>(0.0, 1.0, &[n_heads, lah, head_dim], None).unwrap();
    let k = random::uniform::<f32, f32>(0.0, 1.0, &[n_kv_heads, s, head_dim], None).unwrap();
    let imp = super::layer_importance(&q, &k, n_heads, n_kv_heads, head_dim, 0.0884).unwrap();
    assert_eq!(
        imp.shape(),
        &[s],
        "importance must be [S], got {:?}",
        imp.shape()
    );
    // softmax outputs are in [0, 1]; the mean-over-lah of max-over-heads stays so.
    let vals = imp.as_dtype(Dtype::Float32).unwrap();
    mlx_rs::transforms::eval([&vals]).unwrap();
    let slice = vals.as_slice::<f32>();
    assert!(
        slice.iter().all(|x| *x >= 0.0 && *x <= 1.0),
        "importance out of [0,1]"
    );
}
