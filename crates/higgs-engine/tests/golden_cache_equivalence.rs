//! Tier-1 trust oracle: the radix prefix cache reuses Hybrid prefixes across
//! turns, stays within a bounded deterministic drift envelope, and is faster than
//! cold full-prefill of the same prompts on a real hybrid model.
//!
//! # Which cache this proves (read before trusting the name)
//!
//! `higgs` has TWO between-turn KV reuse paths, and they are NOT the same:
//!
//! 1. **Radix prefix cache** (`SimpleEngine::generate*`, automatic): stores the
//!    post-prefill cache **densely** — dense KV blocks for attention layers, a
//!    dense *clone* for hybrid (GDN/SSM) caches (hybrid is deliberately never
//!    block-paged; see `paged_prefix_cache::slice_into_blocks`). Dense KV reuse
//!    is bit-identical. Hybrid reuse is bounded-drift because a cached segmented
//!    prefill is not numerically identical to a cold one-shot full prefill on MLX.
//!    **This test proves the bounded Hybrid contract.**
//!
//! 2. **Per-session retention** (`SimpleEngine::generate_continued`, keyed by
//!    `session_id`): `quantize_for_retention` TurboQuant-compresses the retained
//!    KV (2-bit keys / 3-bit values) for any power-of-two `head_dim` — and
//!    Qwen3.5-9B has `head_dim = 256`. That path is therefore **lossy by design**
//!    and is NOT expected to be bit-identical to a dense cold prefill. It is the
//!    subject of Tier 4, not this oracle.
//!
//! The hybrid model exercises the dense-clone reuse path (the operation the wider
//! MLX ecosystem reports as broken for hybrid models, mlx-lm #980). Proving
//! bounded reuse here is the strongest practical correctness evidence for it.
//!
//! Ignored by default (loads a real model):
//!   HIGGS_PRUNE_MODEL=/path/to/qwen3.5-9b-mlx-4bit \
//!     cargo test -p higgs-engine --test golden_cache_equivalence -- --ignored --nocapture
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
use std::time::Instant;

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
    let mut cur = vec![0; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}

fn norm_edit(a: &str, b: &str) -> f64 {
    let max_chars = a.chars().count().max(b.chars().count()).max(1);
    levenshtein(a, b) as f64 / max_chars as f64
}

#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL to a hybrid Qwen3.5 model dir"]
fn golden_radix_cache_reuse_stays_within_bound() {
    let dir = std::env::var("HIGGS_PRUNE_MODEL").expect("set HIGGS_PRUNE_MODEL");
    let model_dir = Path::new(&dir);
    let tuning = MlxRuntimeTuning::from_model_dir(model_dir, RequestedMlxProfile::Auto);
    let engine =
        SimpleEngine::load(model_dir, KvCacheConfig::default(), tuning, false).expect("load model");

    // Greedy decode: deterministic argmax, the precondition for token identity.
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    const TURNS: usize = 20;
    const MAX_NEW: u32 = 32;

    let user_turns = [
        "Explain photosynthesis in one sentence.",
        "Now name its two main stages.",
        "Which stage needs light directly?",
        "Where in the cell does that happen?",
        "Give one input and one output of it.",
        "How does this connect to the carbon cycle?",
    ];

    // Build the conversation turn by turn, appending the model's own greedy reply
    // so each turn's prompt is a true token-extension of the prior prefix. Record
    // the EXACT prompt tokens so the cold pass replays identical inputs.
    let mut convo: Vec<ChatMessage> = Vec::new();
    let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(TURNS);
    let mut warm_texts: Vec<String> = Vec::with_capacity(TURNS);

    // ---- WARM pass: radix cache accumulates; each turn reuses the shared prefix.
    engine.clear_prefix_cache();
    let warm_start = Instant::now();
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
    let warm_entries = engine.prefix_cache_len();
    assert!(
        warm_entries > 0,
        "warm pass must have populated the radix prefix cache (got 0 entries)"
    );

    // ---- COLD pass: clear before each turn => genuine dense full prefill, no reuse.
    let cold_start = Instant::now();
    let mut mismatches: Vec<usize> = Vec::new();
    let mut cold_completion: Vec<u32> = Vec::with_capacity(TURNS);
    let mut edits: Vec<f64> = Vec::with_capacity(TURNS);
    for (turn, prompt) in prompts.iter().enumerate() {
        engine.clear_prefix_cache();
        assert_eq!(
            engine.prefix_cache_len(),
            0,
            "cache must be empty for the cold baseline at turn {turn}"
        );
        let out = engine
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
        cold_completion.push(out.completion_tokens);
        edits.push(norm_edit(&warm_texts[turn], &out.text));
        if out.text != warm_texts[turn] {
            mismatches.push(turn);
            eprintln!(
                "DIVERGENCE turn {turn}:\n  warm: {:?}\n  cold: {:?}",
                warm_texts[turn], out.text
            );
        }
    }
    let cold_elapsed = cold_start.elapsed();
    let exact = TURNS - mismatches.len();
    let exact_frac = exact as f64 / TURNS as f64;
    let mean_edit = edits.iter().sum::<f64>() / TURNS as f64;
    let max_edit = edits.iter().copied().fold(0.0_f64, f64::max);

    println!("\n=== golden radix bounded reuse (hybrid, {TURNS} turns, greedy) ===");
    println!("turn | prompt_toks | warm_gen | cold_gen | norm_edit | identical");
    println!("-----+-------------+----------+----------+-----------+----------");
    for turn in 0..TURNS {
        println!(
            "{:>4} | {:>11} | {:>8} | {:>8} | {:>9.3} | {}",
            turn,
            prompts[turn].len(),
            warm_texts[turn].len(),
            cold_completion[turn],
            edits[turn],
            if mismatches.contains(&turn) {
                "NO"
            } else {
                "yes"
            }
        );
    }
    println!(
        "\nexact matches: {exact}/{TURNS}\nmean normalized edit distance: {mean_edit:.3}\nmax  normalized edit distance: {max_edit:.3}\nradix entries after warm pass: {warm_entries}\nwall: warm {warm_elapsed:?} (reuse) vs cold {cold_elapsed:?} (full prefill each turn)"
    );

    let max_mean_edit: f64 = std::env::var("HIGGS_REUSE_MAX_MEAN_EDIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.10);
    let min_exact_frac: f64 = std::env::var("HIGGS_REUSE_MIN_EXACT_FRAC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.80);
    assert!(
        mean_edit <= max_mean_edit,
        "mean normalized edit distance {mean_edit:.3} exceeds {max_mean_edit}; mismatches at turns {mismatches:?}"
    );
    assert!(
        exact_frac >= min_exact_frac,
        "exact fraction {exact_frac:.3} below {min_exact_frac}; mismatches at turns {mismatches:?}"
    );
    assert!(
        warm_elapsed < cold_elapsed,
        "warm reuse ({warm_elapsed:.2?}) must be faster than cold full prefill ({cold_elapsed:.2?})"
    );
    println!("ORACLE PASS: bounded Hybrid reuse stayed within drift and speed limits.");
}

/// Control that isolates WHY the oracle above diverges: is it MLX greedy
/// nondeterminism (two cold runs already differ → token-identity is unachievable
/// and the divergence is not a reuse bug), or does prefix reuse *deterministically*
/// change the output (a real reuse effect)?
///
/// Asserts only the thing that MUST hold for the oracle to be meaningful — cold
/// determinism. It reports (does not assert) whether warm reuse matches cold, so
/// this test passes while documenting the measured reuse effect.
#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL to a hybrid Qwen3.5 model dir"]
fn radix_reuse_vs_cold_determinism_control() {
    let dir = std::env::var("HIGGS_PRUNE_MODEL").expect("set HIGGS_PRUNE_MODEL");
    let model_dir = Path::new(&dir);
    let tuning = MlxRuntimeTuning::from_model_dir(model_dir, RequestedMlxProfile::Auto);
    let engine =
        SimpleEngine::load(model_dir, KvCacheConfig::default(), tuning, false).expect("load model");
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let run = |p: &[u32]| {
        engine
            .generate_with_thinking(p, 48, &params, &[], false, None, false, None, None, None)
            .expect("generate")
            .text
    };

    // A two-turn conversation so turn-1's stored prefix is a genuine prefix of p2.
    // Deliberately long so the block-aligned stored prefix is several blocks even
    // under a coarse tokenizer (a short prompt may fall below one block and store
    // nothing, making the reuse comparison vacuous).
    let t1 = vec![user(
        "Explain photosynthesis in thorough detail. Cover both the light-dependent reactions \
         and the Calvin cycle, the role of chlorophyll and accessory pigments, where in the \
         chloroplast each stage occurs, the inputs and outputs of each stage, how ATP and NADPH \
         are produced and consumed, and how the overall process connects to cellular respiration \
         and the global carbon cycle. Be precise and use correct biological terminology throughout.",
    )];
    let p1 = engine
        .prepare_chat_prompt_with_thinking(&t1, None, false)
        .expect("render p1");
    engine.clear_prefix_cache();
    let a1 = run(&p1);
    let mut convo = t1;
    convo.push(assistant(a1));
    convo.push(user("Now summarize that in one sentence."));
    let p2 = engine
        .prepare_chat_prompt_with_thinking(&convo, None, false)
        .expect("render p2");

    // (A) Cold determinism: two cache-cleared full prefills of the SAME prompt.
    engine.clear_prefix_cache();
    let cold_a = run(&p2);
    engine.clear_prefix_cache();
    let cold_b = run(&p2);

    // (B) Warm: populate radix with p1's prefix (a prefix of p2), then run p2 reusing it.
    engine.clear_prefix_cache();
    let _ = run(&p1);
    let reused_entries = engine.prefix_cache_len();
    let warm = run(&p2);

    let cold_deterministic = cold_a == cold_b;
    let warm_matches_cold = warm == cold_a;
    println!("\n=== radix reuse vs cold determinism control ===");
    println!("cold full-prefill deterministic (run A == run B): {cold_deterministic}");
    println!("radix entries before warm p2: {reused_entries}");
    println!("warm (reuse) == cold: {warm_matches_cold}");
    println!("COLD: {cold_a:?}");
    println!("WARM: {warm:?}");

    // Cold greedy decode must be deterministic for token-identity to be a
    // meaningful claim at all.
    assert_eq!(
        cold_a, cold_b,
        "cold full-prefill must be deterministic for token-identity to be a meaningful claim"
    );
    assert!(
        reused_entries > 0,
        "warm path must have a stored prefix in the radix"
    );
    // The cache must never change the computed output. Pure-KV reuses the stored
    // prefix block-for-block (exact); hybrid keys at full offset so a partial
    // match never fires (full prefill). Either way, warm == cold. Before the
    // hybrid full-offset fix, this diverged by a token on the hybrid model.
    assert_eq!(
        warm, cold_a,
        "prefix-cache reuse must be byte-identical to cold full-prefill"
    );
}

/// Tier-4 contract: per-session continuation (`generate_continued` with a
/// `session_id`) is a BEST-EFFORT latency optimization, NOT exact replay. The
/// retained KV is TurboQuant-compressed, so its output is not guaranteed
/// bit-identical to a stateless full prefill. This test pins that contract:
///   - the continued path WORKS (non-empty output every turn) and SAVES prefill
///     (the win) — asserted;
///   - where it diverges from cold is measured and reported, NOT asserted (the
///     dense-KV radix reuse is the bit-identical option; Hybrid radix reuse is a
///     bounded-drift optimization too — see the bounded oracle above).
#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL"]
fn per_session_continuation_is_best_effort() {
    let dir = std::env::var("HIGGS_PRUNE_MODEL").expect("set HIGGS_PRUNE_MODEL");
    let model_dir = Path::new(&dir);
    let tuning = MlxRuntimeTuning::from_model_dir(model_dir, RequestedMlxProfile::Auto);
    let engine =
        SimpleEngine::load(model_dir, KvCacheConfig::default(), tuning, false).expect("load model");
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    const SID: u64 = 1234;
    const COLD_BASE: u64 = 9000;
    const TURNS: usize = 12;
    const MAX_NEW: u32 = 24;
    let followups = [
        "Explain photosynthesis in one sentence.",
        "Now name its two main stages.",
        "Which stage needs light directly?",
        "Where in the cell does it happen?",
        "Name one input and one output.",
    ];

    let encode = |text: &str| -> Vec<u32> {
        engine
            .tokenizer()
            .encode(text, false)
            .expect("encode")
            .get_ids()
            .to_vec()
    };

    let first = user(followups[0]);
    let mut seq = engine
        .prepare_chat_prompt_with_thinking(std::slice::from_ref(&first), None, false)
        .expect("render");

    let mut first_divergence: Option<usize> = None;
    let mut total_prompt: u64 = 0;
    let mut total_prefilled: u64 = 0;
    for turn in 0..TURNS {
        let seq_in = seq.len();

        // cache-resident: reuses SID's (TurboQuant-compressed) retained KV
        let cached = engine
            .generate_continued(SID, &seq, MAX_NEW, &params)
            .expect("cached generate");
        let retained = engine
            .retained_session_tokens(SID)
            .expect("retained after cached");
        // May be empty if the model emits EOS immediately on a turn — a valid
        // outcome, not a failure of the best-effort path.
        let gen_cached = retained[seq_in..].to_vec();

        // cold: a fresh session full-prefills the same tokens (stateless baseline)
        let cold_sid = COLD_BASE + turn as u64;
        engine
            .generate_continued(cold_sid, &seq, MAX_NEW, &params)
            .expect("cold generate");
        let gen_cold = engine
            .retained_session_tokens(cold_sid)
            .expect("retained after cold")[seq_in..]
            .to_vec();
        engine.drop_retained_session(cold_sid);

        if gen_cached != gen_cold && first_divergence.is_none() {
            first_divergence = Some(turn);
        }
        if turn > 0 {
            assert!(
                cached.continued,
                "turn {turn}: per-session path should reuse the retained cache"
            );
            total_prompt += u64::from(cached.prompt_tokens);
            total_prefilled += u64::from(cached.prefilled_tokens);
        }

        // grow the conversation from the live (cached) retained tokens
        seq = retained;
        seq.extend_from_slice(&encode(followups[(turn + 1) % followups.len()]));
    }

    let saved = total_prompt.saturating_sub(total_prefilled);
    println!("\n=== per-session continuation: best-effort characterization ===");
    println!(
        "first divergence from stateless cold at turn: {first_divergence:?} (None = matched all {TURNS} turns)"
    );
    println!(
        "prefill saved on continued turns: {saved} of {total_prompt} prompt tokens ({:.0}% saved)",
        (saved as f64) / (total_prompt.max(1) as f64) * 100.0
    );

    // The win: continued turns prefill far fewer tokens than the full prompt.
    assert!(
        total_prefilled < total_prompt,
        "continuation must save prefill ({total_prefilled} >= {total_prompt})"
    );
    // Best-effort contract: we deliberately do NOT assert gen_cached == gen_cold.
}
