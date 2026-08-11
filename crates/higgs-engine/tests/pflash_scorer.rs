//! PFlash scorer validation against the real Qwen3-0.6B drafter.
//!
//! Decides whether the SpecPrefill-Full-LAH scorer (`pflash_importance`) closes
//! the naive 50%→10% keep gap: a real needle inserted in a long prompt must
//! rank high enough to survive `select_survivors` at keep_ratio = 0.10.
//!
//! Ignored by default (needs the drafter on disk):
//!   HIGGS_PREFLASH_DRAFTER=/path/to/Qwen3-0.6B-4bit \
//!     cargo test -p higgs-engine --test pflash_scorer -- --nocapture --ignored

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::collections::HashSet;

use higgs_engine::model_loader::{load_model, load_tokenizer};
use higgs_models::cache::SteppingKeyValueCache;
use higgs_models::spec_prefill::{PrefillScoreConfig, select_survivors};
use mlx_rs::{Array, Dtype};

const DRAFTER_DEFAULT: &str = "/Users/peppi/AI-Models/shared/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8";

const HAYSTACK: &str = "The archival logs from the northern climate station describe routine \
    instrument calibration, battery cycling, wind-vane realignment, and the weekly replacement of \
    desiccant cartridges in the humidity enclosure. Operators note that condensation readings drift \
    predictably with the diurnal temperature swing and that the backup telemetry uplink engages \
    whenever the primary link drops below four decibels of signal margin. ";

fn build_niah_prompt(
    tok: &tokenizers::Tokenizer,
    target_tokens: usize,
    needle: &str,
    question: &str,
) -> (Vec<u32>, usize) {
    let mut text = String::new();
    while tok.encode(text.clone(), true).unwrap().get_ids().len() < target_tokens {
        text.push_str(HAYSTACK);
        text.push(' ');
    }
    let pre = text.clone();
    let full = format!("{pre}\n\n{needle}\n\n{text}\n\n{question}");
    // Token offset of the needle: tokenize the prefix before it.
    let prefix = format!("{pre}\n\n");
    let needle_offset = tok.encode(prefix, true).unwrap().get_ids().len();
    let ids = tok.encode(full, true).unwrap().get_ids().to_vec();
    (ids, needle_offset)
}

#[test]
#[ignore = "requires the Qwen3-0.6B drafter on disk (set HIGGS_PREFLASH_DRAFTER)"]
fn pflash_scorer_ranks_needle_high() {
    let dir =
        std::env::var("HIGGS_PREFLASH_DRAFTER").unwrap_or_else(|_| DRAFTER_DEFAULT.to_owned());
    assert!(
        std::path::Path::new(&dir).exists(),
        "drafter not found at {dir}"
    );

    let mut model = load_model(&dir).expect("load drafter");
    let tok = load_tokenizer(&dir).expect("load tokenizer");
    let n_layers = model.num_layers();
    assert!(n_layers >= 8, "drafter has too few layers ({n_layers})");
    let score_layers: Vec<usize> = ((n_layers - 8)..n_layers).collect();

    let needle = "The special authorization code for the generator is FORGE-TANGENT-4471.";
    let question = "What is the special authorization code for the generator?";
    let (ids, needle_off) = build_niah_prompt(&tok, 4096, needle, question);
    let s = ids.len();
    let inputs = Array::from_slice(&ids, &[1, s as i32]);

    let mut cache: Vec<Option<SteppingKeyValueCache>> = Vec::new();
    let imp = model
        .pflash_importance(&inputs, &score_layers, 8, &mut cache)
        .expect("pflash_importance");
    let imp_f32 = imp.as_dtype(Dtype::Float32).unwrap();
    mlx_rs::transforms::eval([&imp_f32]).unwrap();
    let importance = imp_f32.as_slice::<f32>().to_vec();
    assert_eq!(
        importance.len(),
        s,
        "importance length must match prompt length"
    );

    // Does the needle survive compression at keep_ratio = 0.10?
    let cfg = PrefillScoreConfig::default(); // keep 0.10, chunk 32, avgpool 13, lah 8
    let tokens_u32: Vec<u32> = ids.iter().map(|&t| t as u32).collect();
    let plan = select_survivors(&tokens_u32, &importance, &cfg).unwrap();
    let kept: HashSet<i32> = plan.original_positions.iter().copied().collect();
    let needle_end = (needle_off + 16).min(s);
    let survived = (needle_off..needle_end)
        .filter(|p| kept.contains(&(*p as i32)))
        .count();
    let survival_rate = survived as f32 / 16.0;

    // Block-level ranking diagnostic.
    let chunk = cfg.chunk;
    let n_blocks = s.div_ceil(chunk);
    let block_score: Vec<f32> = (0..n_blocks)
        .map(|b| {
            let lo = b * chunk;
            let hi = lo + chunk.min(s - lo);
            importance[lo..hi].iter().sum::<f32>() / (hi - lo) as f32
        })
        .collect();
    let mut ranked: Vec<usize> = (0..n_blocks).collect();
    ranked.sort_by(|&a, &b| {
        block_score[b]
            .partial_cmp(&block_score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let needle_block = needle_off / chunk;
    let needle_block_rank = ranked.iter().position(|&b| b == needle_block).unwrap() + 1;

    println!(
        "prompt={s} tokens, needle_block={needle_block}/{n_blocks} rank={needle_block_rank} (top {:.1}%), \
         needle survival at keep=0.10: {survived}/16 ({:.0}%)",
        100.0 * needle_block_rank as f32 / n_blocks as f32,
        100.0 * survival_rate,
    );

    // The SpecPrefill-Full-LAH claim: needle should survive at keep 0.10.
    // If it doesn't, the scorer diverges from the paper (RESEARCH §5.3) — do not
    // silently relax this; investigate (position-id, qk-norm, LAH Q capture).
    assert!(
        survival_rate >= 0.5,
        "needle survival {survived}/16 at keep=0.10 is too low — scorer diverges from SpecPrefill"
    );
}
