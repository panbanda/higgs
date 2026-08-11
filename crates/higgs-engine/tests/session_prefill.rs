//! Cache-resident multi-turn prefill savings, proven end-to-end on a real model.
//!
//! The claim under test: with a per-`session_id` live KV cache,
//! [`SimpleEngine::generate_continued`] re-prefills ONLY the new suffix on a
//! follow-up turn whose prompt is a true extension of the retained tokens — it
//! does NOT re-prefill the whole conversation history.
//!
//! We build turn 2 by CONCATENATION so the turn-1 tokens are a genuine prefix of
//! the turn-2 prompt. The retained tokens the engine holds after turn 1 are
//! `toks1 ++ generated_ids`; we take those EXACT tokens (via
//! `retained_session_tokens`) and append a suffix:
//!   `toks2 = retained ++ suffix_ids`.
//! Rebuilding the prefix from `out1.text` instead would round-trip the generated
//! IDs through detok→retok, which is not stable for Qwen BPE and silently
//! diverges — we report that reconstruction for contrast but do not rely on it.
//!
//! Ignored by default (loads a real model):
//!   HIGGS_PRUNE_MODEL=/path/to/model \
//!     cargo test -p higgs-engine --test session_prefill -- --ignored --nocapture
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::tests_outside_test_module,
    clippy::wildcard_enum_match_arm,
    clippy::indexing_slicing,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::too_many_lines,
    clippy::panic,
    clippy::items_after_statements,
    clippy::redundant_clone,
    clippy::cast_possible_wrap
)]

use std::path::Path;
use std::time::Instant;

use higgs_engine::chat_template::ChatMessage;
use higgs_engine::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
use higgs_engine::simple::SimpleEngine;
use higgs_models::SamplingParams;
use higgs_models::turboquant::KvCacheConfig;

fn encode_ids(engine: &SimpleEngine, text: &str) -> Vec<u32> {
    engine
        .tokenizer()
        .encode(text, false)
        .expect("encode")
        .get_ids()
        .to_vec()
}

/// First index at which `a` and `b` differ over their shared length, or `None`
/// if one is a prefix of the other.
fn first_divergence(a: &[u32], b: &[u32]) -> Option<usize> {
    a.iter().zip(b.iter()).position(|(x, y)| x != y)
}

#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL"]
fn session_continuation_saves_prefill() {
    let dir = std::env::var("HIGGS_PRUNE_MODEL").expect("set HIGGS_PRUNE_MODEL");
    let model_dir = Path::new(&dir);
    let tuning = MlxRuntimeTuning::from_model_dir(model_dir, RequestedMlxProfile::Auto);
    let engine =
        SimpleEngine::load(model_dir, KvCacheConfig::default(), tuning, false).expect("load");
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    const SID: u64 = 7;

    // ---- Turn 1: full prefill, retains live cache + the tokens it now holds.
    let turn1_msg = ChatMessage {
        role: "user".to_owned(),
        content: "Explain photosynthesis in one sentence.".to_owned(),
        tool_calls: None,
    };
    let toks1 = engine
        .prepare_chat_prompt_with_thinking(std::slice::from_ref(&turn1_msg), None, false)
        .expect("render turn 1");
    let out1 = engine
        .generate_continued(SID, &toks1, 64, &params)
        .expect("turn 1 generate");

    assert!(
        !out1.continued,
        "first turn must be a clean full prefill, got continued=true"
    );
    assert_eq!(
        out1.prefilled_tokens as usize,
        toks1.len(),
        "first turn must prefill the entire prompt"
    );

    // Ground truth: the exact tokens the live cache now covers (prompt + gen).
    let retained = engine
        .retained_session_tokens(SID)
        .expect("turn 1 must retain a live cache");
    assert!(
        retained.starts_with(&toks1),
        "retained tokens must begin with the turn-1 prompt"
    );

    // For contrast: the lossy reconstruction the naive approach would use.
    let mut retok = toks1.clone();
    retok.extend_from_slice(&encode_ids(&engine, &out1.text));
    let roundtrip_stable = retok == retained;

    // ---- Build turn 2 as a TRUE token-extension of the retained tokens.
    let suffix_ids = encode_ids(&engine, "\nNow summarize that in three words.");
    let mut toks2 = retained.clone();
    toks2.extend_from_slice(&suffix_ids);

    let out2 = engine
        .generate_continued(SID, &toks2, 32, &params)
        .expect("turn 2 generate");

    // ---- Report.
    println!("\n=== session-resident prefill ===");
    println!(
        "detok->retok round-trip stable for this model: {roundtrip_stable} \
         (retained={} tokens, naive reconstruction={} tokens)",
        retained.len(),
        retok.len()
    );
    println!("turn | prompt_tokens | prefilled_tokens | continued | saved");
    println!("-----+---------------+------------------+-----------+------");
    println!(
        "  1  | {:>13} | {:>16} | {:>9} | {:>5}",
        out1.prompt_tokens, out1.prefilled_tokens, out1.continued, 0
    );

    if out2.continued {
        let saved = out2.prompt_tokens.saturating_sub(out2.prefilled_tokens);
        println!(
            "  2  | {:>13} | {:>16} | {:>9} | {:>5}",
            out2.prompt_tokens, out2.prefilled_tokens, out2.continued, saved
        );
        println!(
            "\nturn-2 reused the live cache: prefilled only the {}-token suffix \
             instead of {} tokens; saved {} prefill tokens.",
            out2.prefilled_tokens, out2.prompt_tokens, saved
        );
        assert!(
            out2.prefilled_tokens < out2.prompt_tokens,
            "continuation must prefill fewer tokens than the full prompt \
             (prefilled={}, prompt={})",
            out2.prefilled_tokens,
            out2.prompt_tokens
        );
        // The suffix we appended is exactly what should have been prefilled.
        assert_eq!(
            out2.prefilled_tokens as usize,
            suffix_ids.len(),
            "continuation should prefill exactly the appended suffix"
        );
    } else {
        // Did not continue — surface WHY (prefix mismatch), but do not panic.
        println!(
            "  2  | {:>13} | {:>16} | {:>9} | {:>5}",
            out2.prompt_tokens, out2.prefilled_tokens, out2.continued, "n/a"
        );
        match first_divergence(&retained, &toks2) {
            Some(idx) => println!("fell back: prefix mismatch at index {idx}"),
            None => println!(
                "fell back: prefix mismatch (retained IS a prefix of toks2 by construction; \
                 the engine's guard rejected it unexpectedly)"
            ),
        }
        println!(
            "retained.len()={}, toks2.len()={}, suffix.len()={}",
            retained.len(),
            toks2.len(),
            suffix_ids.len()
        );
        // toks2 is retained ++ suffix by construction, so the guard MUST accept
        // it. A fallback here is a real regression in the continuation path.
        panic!(
            "turn 2 fell back to full prefill despite toks2 being retained++suffix \
             — continuation guard regression"
        );
    }
}

// On-demand decode-throughput probe: sequential decode tok/s at ~4k context.
// Run with HIGGS_DIAG_SESSION_TIMING=1 and read the decode= bucket
// (2026-07-02 baseline on Qwen3.6-35B-A3B-4bit, debug build: ~24 tok/s).
#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL"]
fn decode_throughput_probe() {
    let dir = std::env::var("HIGGS_PRUNE_MODEL").expect("set HIGGS_PRUNE_MODEL");
    let model_dir = Path::new(&dir);
    let tuning = MlxRuntimeTuning::from_model_dir(model_dir, RequestedMlxProfile::Auto);
    let engine =
        SimpleEngine::load(model_dir, KvCacheConfig::default(), tuning, false).expect("load");
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let mut content = String::new();
    while engine
        .prepare_chat_prompt_with_thinking(
            &[ChatMessage {
                role: "user".to_owned(),
                content: content.clone(),
                tool_calls: None,
            }],
            None,
            false,
        )
        .expect("render")
        .len()
        < std::env::var("HIGGS_SESSION_BENCH_TARGET_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4096)
    {
        content.push_str(
            " Explain photosynthesis, chloroplast structure, electron transport, ATP synthesis, NADPH production, Calvin-cycle carbon fixation, and ecological carbon flow in precise terms.",
        );
    }
    let toks = engine
        .prepare_chat_prompt_with_thinking(
            &[ChatMessage {
                role: "user".to_owned(),
                content,
                tool_calls: None,
            }],
            None,
            false,
        )
        .expect("render");
    let out = engine
        .generate_continued(777, &toks, 192, &params)
        .expect("generate");
    println!(
        "decode probe: prompt={} completion={}",
        toks.len(),
        out.completion_tokens
    );
}

#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL"]
fn long_context_session_continuation_beats_cold_prefill() {
    let dir = std::env::var("HIGGS_PRUNE_MODEL").expect("set HIGGS_PRUNE_MODEL");
    let model_dir = Path::new(&dir);
    let tuning = MlxRuntimeTuning::from_model_dir(model_dir, RequestedMlxProfile::Auto);
    let engine =
        SimpleEngine::load(model_dir, KvCacheConfig::default(), tuning, false).expect("load");
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    let target_tokens = std::env::var("HIGGS_SESSION_BENCH_TARGET_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1536);
    let mut content = String::new();
    while engine
        .prepare_chat_prompt_with_thinking(
            &[ChatMessage {
                role: "user".to_owned(),
                content: content.clone(),
                tool_calls: None,
            }],
            None,
            false,
        )
        .expect("render growing prompt")
        .len()
        < target_tokens
    {
        content.push_str(
            " Explain photosynthesis, chloroplast structure, electron transport, ATP synthesis, NADPH production, Calvin-cycle carbon fixation, and ecological carbon flow in precise terms.",
        );
    }

    let turn1 = ChatMessage {
        role: "user".to_owned(),
        content,
        tool_calls: None,
    };
    let toks1 = engine
        .prepare_chat_prompt_with_thinking(std::slice::from_ref(&turn1), None, false)
        .expect("render long turn 1");

    const SID: u64 = 4242;
    let turn1_start = Instant::now();
    let out1 = engine
        .generate_continued(SID, &toks1, 1, &params)
        .expect("turn 1 generate");
    let turn1_elapsed = turn1_start.elapsed();
    assert!(!out1.continued);

    let retained = engine
        .retained_session_tokens(SID)
        .expect("turn 1 must retain a live cache");
    let suffix_ids = encode_ids(&engine, "\nNow answer in exactly five words.");
    let mut turn2 = retained.clone();
    turn2.extend_from_slice(&suffix_ids);

    let continued_start = Instant::now();
    let continued = engine
        .generate_continued(SID, &turn2, 1, &params)
        .expect("continued generate");
    let continued_elapsed = continued_start.elapsed();

    engine.drop_retained_session(SID + 1);
    engine.clear_prefix_cache();
    let cold_start = Instant::now();
    let cold = engine
        .generate_continued(SID + 1, &turn2, 1, &params)
        .expect("cold generate");
    let cold_elapsed = cold_start.elapsed();
    engine.drop_retained_session(SID + 1);

    let saved = continued
        .prompt_tokens
        .saturating_sub(continued.prefilled_tokens);
    let saved_pct = saved as f64 / continued.prompt_tokens.max(1) as f64;

    println!("\n=== long-context session continuation ===");
    println!(
        "turn1 prompt={} prefilled={} wall={turn1_elapsed:.2?}",
        out1.prompt_tokens, out1.prefilled_tokens
    );
    println!(
        "turn2 continued: prompt={} prefilled={} saved={} ({:.1}%) wall={continued_elapsed:.2?}",
        continued.prompt_tokens,
        continued.prefilled_tokens,
        saved,
        saved_pct * 100.0
    );
    println!(
        "turn2 cold:      prompt={} prefilled={} wall={cold_elapsed:.2?}",
        cold.prompt_tokens, cold.prefilled_tokens
    );

    assert!(
        continued.continued,
        "turn 2 must use the retained session cache"
    );
    assert_eq!(
        continued.prefilled_tokens as usize,
        suffix_ids.len(),
        "continued turn must prefill only the appended suffix"
    );
    assert!(
        saved_pct >= 0.90,
        "continued path saved only {:.1}%",
        saved_pct * 100.0
    );
    assert!(
        continued_elapsed < cold_elapsed,
        "continued turn ({continued_elapsed:.2?}) must beat cold full prefill ({cold_elapsed:.2?})"
    );
}
