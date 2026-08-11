//! Cross-turn radix reuse for hybrid (GDN/SSM) caches — divergence-rate harness.
//!
//! The radix prefix cache stores a hybrid cache as a whole dense CLONE (GDN
//! sequential state can't be block-paged). For cross-turn reuse to fire, the
//! stored clone must end at the conversation boundary, not at the generation-
//! prompt suffix (`<|im_start|>assistant\n…`), which diverges cross-turn.
//!
//! Correctness limit this quantifies: a hybrid clone's body KV is recomputed by
//! a separate `forward(body)` (the cache can't be sliced like dense KV, and the
//! SSM state can't be trimmed). The SSM half is bit-identical to a full prefill
//! (sequential, no look-ahead), but the attention KV is recomputed at a
//! different sequence length → different kernel tiling → a hair of drift that
//! occasionally flips greedy argmax at branch points. This is the same
//! "best-effort, not exact replay" class as TurboQuant session retention.
//!
//! So instead of asserting strict warm==cold, this harness MEASURES the
//! divergence rate over many turns and asserts it stays inside a sane bound —
//! catching catastrophic regressions while tolerating the inherent branch-point
//! flips. It also asserts the real goal: reuse fires and saves the prefix.
//!
//! Ignored by default (loads a real model):
//!   HIGGS_PRUNE_MODEL=/path/to/qwen3.5-9b-mlx-4bit \
//!     cargo test -p higgs-engine --test cross_turn_hybrid_reuse -- --ignored --nocapture
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::tests_outside_test_module,
    clippy::too_many_lines,
    clippy::indexing_slicing,
    clippy::doc_markdown,
    clippy::items_after_statements
)]

use std::path::Path;

use higgs_engine::chat_template::ChatMessage;
use higgs_engine::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
use higgs_engine::simple::SimpleEngine;
use higgs_models::SamplingParams;
use higgs_models::turboquant::KvCacheConfig;

fn user(content: &str) -> ChatMessage {
    ChatMessage {
        role: "user".to_owned(),
        content: content.to_owned(),
        tool_calls: None,
    }
}

fn assistant(content: String) -> ChatMessage {
    ChatMessage {
        role: "assistant".to_owned(),
        content,
        tool_calls: None,
    }
}

/// Character-level Levenshtein distance (small strings: greedy generations).
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur: Vec<usize> = vec![0; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

/// Length of the longest common leading token-span of two greedy outputs, as a
/// fraction of the longer output — 1.0 = identical prefix through the end.
fn common_prefix_frac(a: &str, b: &str) -> f64 {
    let max = a.len().max(b.len());
    if max == 0 {
        return 1.0;
    }
    let common = a
        .char_indices()
        .zip(b.chars())
        .take_while(|((_, x), y)| x == y)
        .count();
    // `common` is in chars of `a`; normalize by char-length of the longer.
    let max_chars = a.chars().count().max(b.chars().count()).max(1);
    common as f64 / max_chars as f64
}

/// Normalized edit distance in [0,1]: 0 = identical, 1 = fully disjoint.
fn norm_edit(a: &str, b: &str) -> f64 {
    let max = a.len().max(b.len()).max(1);
    levenshtein(a, b) as f64 / max as f64
}

#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL to a hybrid Qwen3.5 model dir"]
fn cross_turn_hybrid_reuse_is_approximate_within_bound() {
    let dir = std::env::var("HIGGS_PRUNE_MODEL").expect("set HIGGS_PRUNE_MODEL");
    let model_dir = Path::new(&dir);
    let tuning = MlxRuntimeTuning::from_model_dir(model_dir, RequestedMlxProfile::Auto);
    let engine =
        SimpleEngine::load(model_dir, KvCacheConfig::default(), tuning, false).expect("load model");

    // Greedy: deterministic given identical logits, so any warm-vs-cold gap is
    // the reuse approximation, not sampling noise.
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    const MAX_NEW: u32 = 40;
    const TURNS: usize = 10;

    // Deliberately long seed so turn 1's stored prefix spans many blocks (a
    // short prompt saves only the template wrapper, vacuous for reuse).
    let user_turns = [
        "Explain photosynthesis in thorough detail. Cover both the light-dependent \
         reactions and the Calvin cycle, the role of chlorophyll and accessory pigments, \
         where in the chloroplast each stage occurs, the inputs and outputs of each \
         stage, and how the overall process connects to the global carbon cycle. Be \
         precise and use correct biological terminology throughout.",
        "Now name its two main stages.",
        "Which stage needs light directly?",
        "Where in the cell does that happen?",
        "Give one input and one output of it.",
        "How does this connect to the carbon cycle?",
        "Summarize all of the above in two sentences.",
        "What is the role of water in this process?",
        "Why are leaves green?",
        "Name a gas released as a byproduct.",
    ];

    let mut convo: Vec<ChatMessage> = Vec::new();
    let mut warm_texts: Vec<String> = Vec::with_capacity(TURNS);
    let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(TURNS);

    // ---- WARM pass: radix cache accumulates; each turn reuses the shared prefix. ----
    engine.clear_prefix_cache();
    let warm_start = std::time::Instant::now();
    for turn in 0..TURNS {
        convo.push(user(user_turns[turn % user_turns.len()]));
        let prompt = engine
            .prepare_chat_prompt_with_thinking(&convo, None, false)
            .expect("render prompt");
        let out = engine
            .generate_with_thinking(
                &prompt,
                MAX_NEW,
                &params,
                &[],
                false,
                None,
                false,
                None,
                None,
                None,
            )
            .expect("warm generate");
        prompts.push(prompt);
        warm_texts.push(out.text.clone());
        convo.push(assistant(out.text));
    }
    let warm_elapsed = warm_start.elapsed();
    let warm_stats = engine.cache_stats();

    // ---- THE primary goal: reuse must fire and save real prefix tokens. ----
    assert!(
        warm_stats.radix_hits >= TURNS as u64 - 1,
        "expected a cache hit on nearly every turn after the first ({} hits in {TURNS} turns)",
        warm_stats.radix_hits
    );
    assert!(
        warm_stats.prefill_saved_tokens > 200,
        "reuse must save the shared conversation prefixes, not just the template wrapper (saved {})",
        warm_stats.prefill_saved_tokens
    );

    // ---- COLD pass: clear before each turn => dense full prefill, no reuse. ----
    let cold_start = std::time::Instant::now();
    let mut exact = 0usize;
    let mut prefix_fracs: Vec<f64> = Vec::with_capacity(TURNS);
    let mut edits: Vec<f64> = Vec::with_capacity(TURNS);
    println!("\n=== cross-turn hybrid reuse (approximate) — per-turn divergence ===");
    println!("turn | prompt_toks | prefix_frac | norm_edit | exact");
    println!("-----+-------------+-------------+-----------+------");
    for (turn, prompt) in prompts.iter().enumerate() {
        engine.clear_prefix_cache();
        let cold = engine
            .generate_with_thinking(
                prompt,
                MAX_NEW,
                &params,
                &[],
                false,
                None,
                false,
                None,
                None,
                None,
            )
            .expect("cold generate");
        let warm = &warm_texts[turn];
        let pf = common_prefix_frac(warm, &cold.text);
        let ne = norm_edit(warm, &cold.text);
        let is_exact = warm == &cold.text;
        if is_exact {
            exact += 1;
        }
        prefix_fracs.push(pf);
        edits.push(ne);
        println!(
            "{:>4} | {:>11} | {:>11.3} | {:>9.3} | {}",
            turn,
            prompt.len(),
            pf,
            ne,
            if is_exact { "yes" } else { "no" }
        );
    }
    let cold_elapsed = cold_start.elapsed();

    let mean_edit: f64 = edits.iter().sum::<f64>() / TURNS as f64;
    let max_edit: f64 = edits.iter().cloned().fold(0. / 0., f64::max);
    let mean_prefix: f64 = prefix_fracs.iter().sum::<f64>() / TURNS as f64;

    println!("\n=== aggregate ===");
    println!("exact matches:      {exact}/{TURNS}");
    println!("mean common prefix: {mean_prefix:.3}");
    println!("mean normalized edit distance: {mean_edit:.3}");
    println!("max  normalized edit distance: {max_edit:.3}");
    println!(
        "wall: warm {:.2?} (reuse) vs cold {:.2?} (full prefill each turn)",
        warm_elapsed, cold_elapsed
    );
    println!(
        "cache: hits={}  saved_tokens={}  entries={}",
        warm_stats.radix_hits, warm_stats.prefill_saved_tokens, warm_stats.radix_entries
    );

    // ---- THE bound: reuse is approximate, not exact. We tolerate the inherent
    // branch-point argmax flips but reject catastrophic divergence. The bound is
    // a KNOB, not a constant: dense (slice-based) reuse is bit-exact and should
    // set the strict defaults via env; hybrid (recomputed-body) reuse carries a
    // small drift and uses the lenient defaults. Two gates:
    //   1. Outputs stay close in edit distance (reuse never rewrites wholesale).
    //   2. The shared prefix stays long (reuse preserves the early tokens that
    //      carry the answer structure; a flip only happens at a branch point).
    //
    // Override per-deployment / per-model:
    //   HIGGS_REUSE_MAX_MEAN_EDIT  (default 0.5; dense: 0.0)
    //   HIGGS_REUSE_MIN_MEAN_PREFIX(default 0.4; dense: 1.0)
    let max_mean_edit: f64 = std::env::var("HIGGS_REUSE_MAX_MEAN_EDIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5);
    let min_mean_prefix: f64 = std::env::var("HIGGS_REUSE_MIN_MEAN_PREFIX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.4);
    assert!(
        mean_edit <= max_mean_edit,
        "mean normalized edit distance {mean_edit:.3} exceeds {max_mean_edit} — reuse is \
         diverging beyond the configured bound (tighten HIGGS_REUSE_MAX_MEAN_EDIT to force a \
         bit-exact mechanism)"
    );
    assert!(
        mean_prefix >= min_mean_prefix,
        "mean common-prefix fraction {mean_prefix:.3} below {min_mean_prefix} — reuse corrupts \
         the answer's early structure (raise HIGGS_REUSE_MIN_MEAN_PREFIX to force bit-exact)"
    );
    // And reuse must actually be faster (the whole point of the cache).
    assert!(
        warm_elapsed < cold_elapsed,
        "warm reuse ({warm_elapsed:.2?}) must be faster than cold full prefill ({cold_elapsed:.2?})"
    );

    println!(
        "\nPASS (bound: mean_edit≤{max_mean_edit}, mean_prefix≥{min_mean_prefix}): reuse saved {} \
         prefix tokens across {TURNS} turns; mean edit dist {mean_edit:.3}, mean common prefix \
         {mean_prefix:.3}, warm faster than cold.",
        warm_stats.prefill_saved_tokens
    );
}
