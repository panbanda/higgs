//! Real-model release gate for session-paired dSpark caching.
//!
//! Manual:
//! ```text
//! HIGGS_DFLASH_TARGET_DIR=/path/to/Bonsai-27B-mlx-1bit \
//! HIGGS_DFLASH_DRAFTER_DIR=/path/to/dSpark-MLX \
//! cargo test -p higgs-engine --test dspark_session_cache -- --ignored --nocapture
//! ```

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::tests_outside_test_module,
    clippy::unwrap_used
)]

mod support;

use std::path::Path;

use higgs_engine::{
    chat_template::ChatMessage,
    mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile},
    simple::SimpleEngine,
};
use higgs_models::{SamplingParams, Speculation, turboquant::KvCacheConfig};
use support::{
    ReferenceDsparkEnv, assert_acceptance_within, assert_bonsai_27b_full_q4,
    assert_decode_tps_within, dflash_acceptance, dflash_decode_tps, dflash_prefill_seconds,
};

fn greedy(speculation: Speculation) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        speculation,
        ..SamplingParams::default()
    }
}

fn append_suffix(engine: &SimpleEngine, prefix: &[u32], text: &str) -> (Vec<u32>, usize) {
    let suffix = engine
        .tokenizer()
        .encode(text, false)
        .expect("encode suffix")
        .get_ids()
        .to_vec();
    let suffix_len = suffix.len();
    let mut extended = prefix.to_vec();
    extended.extend_from_slice(&suffix);
    (extended, suffix_len)
}

#[test]
#[ignore = "loads real Bonsai target + dSpark drafter; set HIGGS_DFLASH_TARGET_DIR + HIGGS_DFLASH_DRAFTER_DIR"]
fn bonsai_session_pair_resumes_suffix_only_and_demotes_atomically() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_test_writer()
        .try_init();
    let _reference_dspark = ReferenceDsparkEnv::install();
    let target = std::env::var("HIGGS_DFLASH_TARGET_DIR")
        .expect("set HIGGS_DFLASH_TARGET_DIR to the Bonsai target model");
    let drafter = std::env::var("HIGGS_DFLASH_DRAFTER_DIR")
        .expect("set HIGGS_DFLASH_DRAFTER_DIR to the MLX dSpark drafter");
    assert_bonsai_27b_full_q4(Path::new(&target), Path::new(&drafter));
    eprintln!("dspark-session checkpoint: loading target + drafter");
    let tuning = MlxRuntimeTuning::from_model_dir(Path::new(&target), RequestedMlxProfile::Auto);
    let engine = SimpleEngine::load_with_dflash(
        &target,
        KvCacheConfig {
            max_retained_sessions: 2,
            ..KvCacheConfig::default()
        },
        tuning,
        false,
        Some(Path::new(&drafter)),
        None,
    )
    .expect("load paired dSpark engine");
    eprintln!("dspark-session checkpoint: engine loaded");

    let prompt = engine
        .prepare_chat_prompt_with_thinking(
            &[ChatMessage {
                role: "user".to_owned(),
                content: "Print the integers from 1 upward as comma-separated values. \
                          Output only the sequence and continue for many terms."
                    .to_owned(),
                tool_calls: None,
            }],
            None,
            false,
        )
        .expect("render no-thinking prompt");
    eprintln!(
        "dspark-session checkpoint: turn1 prompt rendered ({} tokens)",
        prompt.len()
    );

    const SID: u64 = 0xD5A4_0001;
    const COLD_SID: u64 = SID + 1;
    const WARMUP_SID: u64 = SID + 2;
    const MTP_SELECTOR_SID: u64 = SID + 4;

    // This engine intentionally has a dSpark drafter but no MTP checkpoint.
    // An explicit MTP selector must therefore remain sidecar-free AR; it must
    // never silently route to the available dSpark capability.
    let mtp_selected = engine
        .generate_continued_with_thinking(
            MTP_SELECTOR_SID,
            &prompt,
            2,
            &greedy(Speculation::Mtp),
            false,
        )
        .expect("explicit MTP selector without MTP weights");
    assert!(!mtp_selected.continued);
    assert!(
        engine.last_dflash_accepts().is_empty(),
        "explicit MTP without MTP weights must not silently invoke dSpark"
    );
    assert_eq!(
        engine.cache_stats().retained_paired_sessions,
        0,
        "an MTP-selected sidecar-free turn may retain only target state"
    );
    engine.drop_retained_session(MTP_SELECTOR_SID);

    let warmup = engine
        .generate_continued_with_thinking(
            WARMUP_SID,
            &prompt,
            8,
            &greedy(Speculation::DFlash),
            false,
        )
        .expect("session dSpark kernel warmup");
    assert_eq!(
        warmup.completion_tokens, 8,
        "the session performance warmup must enter dSpark decode"
    );
    assert!(
        !engine.last_dflash_accepts().is_empty(),
        "the session performance warmup must execute speculative rounds"
    );
    engine.drop_retained_session(WARMUP_SID);

    let first = engine
        .generate_continued_with_thinking(SID, &prompt, 1, &greedy(Speculation::DFlash), false)
        .expect("one-token paired turn");
    eprintln!("dspark-session checkpoint: turn1 complete");
    assert!(!first.continued);
    assert_eq!(first.prefilled_tokens as usize, prompt.len());
    assert_eq!(first.completion_tokens, 1);
    assert!(
        engine.last_dflash_accepts().is_empty(),
        "max_tokens=1 must perform no speculative rounds"
    );
    assert_eq!(
        engine.cache_stats().retained_paired_sessions,
        1,
        "the one-token cache-only forward must publish one complete pair"
    );

    let retained = engine
        .retained_session_tokens(SID)
        .expect("one-token turn must seal a retained pair");
    assert_eq!(
        retained.len(),
        prompt.len() + 1,
        "max_tokens=1 must cache-forward the visible non-EOS token before sealing"
    );

    let (second_prompt, second_suffix_len) =
        append_suffix(&engine, &retained, ", 2, 3, 4, 5, 6, 7, 8, 9, ");
    let second_started = std::time::Instant::now();
    let second = engine
        .generate_continued_with_thinking(
            SID,
            &second_prompt,
            32,
            &greedy(Speculation::DFlash),
            false,
        )
        .expect("resume paired dSpark session");
    let second_wall = second_started.elapsed();
    assert_eq!(
        second.completion_tokens, 32,
        "the session decode gate requires the complete tg32 workload"
    );
    let second_accepts = engine.last_dflash_accepts();
    let second_acceptance = dflash_acceptance(&engine, "warm paired session");
    let second_decode_tps = dflash_decode_tps(&engine, "warm paired session");
    let second_prefill_seconds = dflash_prefill_seconds(&engine, "warm paired session");
    eprintln!(
        "dspark-session checkpoint: turn2 generated={} decode={second_decode_tps:.2} tok/s \
         accepts={second_accepts:?}",
        second.completion_tokens,
    );
    assert!(
        second.continued,
        "turn two must move-reuse both cache halves"
    );
    assert_eq!(
        second.prefilled_tokens as usize, second_suffix_len,
        "paired continuation must prefill only the appended suffix"
    );
    assert!(
        !second_accepts.is_empty(),
        "direct session dSpark must run without an MTP checkpoint"
    );
    assert_eq!(
        engine.cache_stats().retained_paired_sessions,
        1,
        "a resumed dSpark turn must replace the session with one sealed pair"
    );

    // Decode-only release gate against the identical session decoder with no
    // retained pair. `drop_retained_session` makes the comparator cold while
    // preserving the same prompt, sampling domain, and dSpark implementation.
    engine.drop_retained_session(COLD_SID);
    let cold_started = std::time::Instant::now();
    let cold = engine
        .generate_continued_with_thinking(
            COLD_SID,
            &second_prompt,
            32,
            &greedy(Speculation::DFlash),
            false,
        )
        .expect("cold session dSpark reference");
    let cold_wall = cold_started.elapsed();
    let cold_acceptance = dflash_acceptance(&engine, "uncached session baseline");
    let cold_decode_tps = dflash_decode_tps(&engine, "uncached session baseline");
    let cold_prefill_seconds = dflash_prefill_seconds(&engine, "uncached session baseline");
    assert!(
        !cold.continued,
        "the session performance baseline must begin without retained state"
    );
    assert_eq!(
        cold.prefilled_tokens, cold.prompt_tokens,
        "the session performance baseline must prefill its complete prompt"
    );
    assert_eq!(
        second.text, cold.text,
        "paired session reuse must preserve the cold greedy output"
    );
    assert_eq!(second.completion_tokens, cold.completion_tokens);
    eprintln!(
        "dspark-session release gate: warm_decode={second_decode_tps:.2} \
         uncached_decode={cold_decode_tps:.2} tok/s warm_acceptance={:.2}% ({}/{}) \
         uncached_acceptance={:.2}% ({}/{})",
        second_acceptance.rate() * 100.0,
        second_acceptance.matched,
        second_acceptance.drafted,
        cold_acceptance.rate() * 100.0,
        cold_acceptance.matched,
        cold_acceptance.drafted,
    );
    assert_acceptance_within("warm paired session", second_acceptance, cold_acceptance);
    assert_decode_tps_within("warm paired session", second_decode_tps, cold_decode_tps);
    assert!(
        second_prefill_seconds < cold_prefill_seconds,
        "paired session reuse must remove target+dSpark prefill work: \
         warm={second_prefill_seconds:.3}s cold={cold_prefill_seconds:.3}s"
    );
    assert!(
        second_wall.as_secs_f64() <= cold_wall.as_secs_f64() * 1.03,
        "paired session wall time may vary within 3% but must not erase the saved prefill: \
         warm={second_wall:.2?} cold={cold_wall:.2?}"
    );
    engine.drop_retained_session(COLD_SID);

    let paired_tokens = engine
        .retained_session_tokens(SID)
        .expect("second turn must retain its pair");
    let (none_prompt, none_suffix_len) =
        append_suffix(&engine, &paired_tokens, "\nNow answer with one integer.");
    let none = engine
        .generate_continued_with_thinking(SID, &none_prompt, 1, &greedy(Speculation::None), false)
        .expect("explicit autoregressive continuation");
    assert!(none.continued, "none may reuse the target half by demotion");
    assert_eq!(none.prefilled_tokens as usize, none_suffix_len);
    assert_eq!(
        engine.cache_stats().retained_paired_sessions,
        0,
        "an execution path without taps must atomically discard the dSpark sidecar"
    );
    assert_eq!(
        engine.cache_stats().retained_sessions,
        1,
        "target-only continuity must remain retained after sidecar demotion"
    );

    let target_only_tokens = engine
        .retained_session_tokens(SID)
        .expect("autoregressive turn must retain target-only state");
    let (third_prompt, _) = append_suffix(
        &engine,
        &target_only_tokens,
        "\nContinue the sequence again.",
    );
    let third = engine
        .generate_continued_with_thinking(
            SID,
            &third_prompt,
            2,
            &greedy(Speculation::DFlash),
            false,
        )
        .expect("dSpark after target-only demotion");
    assert!(
        !third.continued,
        "a target-only cache cannot be combined with an independently reconstructed drafter"
    );
    assert_eq!(
        third.prefilled_tokens, third.prompt_tokens,
        "dSpark must cold-prefill after the sidecar was discarded"
    );
    assert_eq!(
        engine.cache_stats().retained_paired_sessions,
        1,
        "the cold dSpark retry must restore one complete retained pair"
    );

    // Two callers may race one exact session extension, but the per-session
    // lock makes the retained pair move-only. Whichever worker enters first
    // resumes the seed pair. When the queued worker enters, the retained state
    // already includes the first worker's completion and is therefore longer
    // than this shared request prompt; it must discard that pair and cold-prefill
    // instead of reusing either half under the shorter key.
    engine.drop_retained_session(SID);
    assert_eq!(engine.cache_stats().retained_sessions, 0);
    const CONCURRENT_SID: u64 = SID + 3;
    let seed = engine
        .generate_continued_with_thinking(
            CONCURRENT_SID,
            &prompt,
            1,
            &greedy(Speculation::DFlash),
            false,
        )
        .expect("seed one retained pair for the concurrent extension");
    assert!(!seed.continued);
    assert_eq!(engine.cache_stats().retained_paired_sessions, 1);
    let concurrent_seed = engine
        .retained_session_tokens(CONCURRENT_SID)
        .expect("concurrency seed must retain one complete pair");
    let (concurrent_prompt, concurrent_suffix_len) =
        append_suffix(&engine, &concurrent_seed, ", 10, 11, 12, ");
    let start = std::sync::Barrier::new(3);
    let concurrent = std::thread::scope(|scope| {
        let left = scope.spawn(|| {
            start.wait();
            engine.generate_continued_with_thinking(
                CONCURRENT_SID,
                &concurrent_prompt,
                4,
                &greedy(Speculation::DFlash),
                false,
            )
        });
        let right = scope.spawn(|| {
            start.wait();
            engine.generate_continued_with_thinking(
                CONCURRENT_SID,
                &concurrent_prompt,
                4,
                &greedy(Speculation::DFlash),
                false,
            )
        });
        start.wait();
        [left, right].map(|worker| {
            worker
                .join()
                .expect("paired session worker must not panic")
                .expect("paired session worker generation")
        })
    });

    assert_eq!(
        concurrent.iter().filter(|result| result.continued).count(),
        1,
        "the seed pair is move-owned, so exactly one racing worker may resume it"
    );
    let resumed = concurrent
        .iter()
        .find(|result| result.continued)
        .expect("one worker must resume the seed pair");
    let cold_after_queue = concurrent
        .iter()
        .find(|result| !result.continued)
        .expect("the queued worker must safely cold-prefill");
    assert_eq!(
        resumed.prefilled_tokens as usize, concurrent_suffix_len,
        "the winning worker must prefill only the strict extension"
    );
    assert_eq!(
        cold_after_queue.prefilled_tokens, cold_after_queue.prompt_tokens,
        "the queued worker must not relabel the first worker's longer retained pair"
    );
    assert_eq!(
        resumed.text, cold_after_queue.text,
        "serialized paired reuse and the cold fallback must preserve greedy output"
    );
    assert_eq!(
        resumed.completion_tokens, cold_after_queue.completion_tokens,
        "both workers must complete the same bounded workload"
    );
    let concurrent_final = engine
        .retained_session_tokens(CONCURRENT_SID)
        .expect("the queued worker must replace the SID with one complete pair");
    assert_eq!(
        concurrent_final.get(..concurrent_prompt.len()),
        Some(concurrent_prompt.as_slice()),
        "the final retained pair must be keyed by the shared request prompt"
    );
    let concurrent_stats = engine.cache_stats();
    assert_eq!(concurrent_stats.retained_sessions, 1);
    assert_eq!(
        concurrent_stats.retained_paired_sessions, 1,
        "the racing SID must finish with one inseparable target+dSpark pair"
    );

    // A genuine token-prefix divergence on that same SID must consume and drop
    // the old pair, then publish only a newly cold-prefilled pair for the new
    // conversation.
    let divergent_prompt = engine
        .prepare_chat_prompt_with_thinking(
            &[ChatMessage {
                role: "user".to_owned(),
                content: "Reply with only the word seven.".to_owned(),
                tool_calls: None,
            }],
            None,
            false,
        )
        .expect("render divergent no-thinking prompt");
    assert_ne!(
        concurrent_final.get(..divergent_prompt.len()),
        Some(divergent_prompt.as_slice()),
        "the divergence fixture must not match the retained conversation"
    );
    let divergent = engine
        .generate_continued_with_thinking(
            CONCURRENT_SID,
            &divergent_prompt,
            1,
            &greedy(Speculation::DFlash),
            false,
        )
        .expect("cold dSpark generation after paired session divergence");
    assert!(
        !divergent.continued,
        "a divergent prompt must not reuse either half of the retained pair"
    );
    assert_eq!(
        divergent.prefilled_tokens, divergent.prompt_tokens,
        "paired divergence must restore a complete cold prefill"
    );
    let divergent_final = engine
        .retained_session_tokens(CONCURRENT_SID)
        .expect("divergent request must publish one newly proven pair");
    assert_eq!(
        divergent_final.get(..divergent_prompt.len()),
        Some(divergent_prompt.as_slice())
    );
    let divergent_stats = engine.cache_stats();
    assert_eq!(divergent_stats.retained_sessions, 1);
    assert_eq!(
        divergent_stats.retained_paired_sessions, 1,
        "paired divergence must replace, never split or duplicate, the retained entry"
    );

    // Real-model cap+1 memory gate. Equal prompts produce equal-sized target
    // and dSpark snapshots, so inserting a third session into a cap of two must
    // replace one whole pair without increasing retained byte accounting.
    engine.drop_retained_session(CONCURRENT_SID);
    assert_eq!(engine.cache_stats().retained_sessions, 0);
    const PLATEAU_SID: u64 = SID + 10;
    for offset in 0..2 {
        let result = engine
            .generate_continued_with_thinking(
                PLATEAU_SID + offset,
                &prompt,
                1,
                &greedy(Speculation::DFlash),
                false,
            )
            .expect("populate paired-session memory cap");
        assert!(!result.continued);
    }
    let at_cap = engine.cache_stats();
    assert_eq!(at_cap.retained_paired_sessions, 2);
    assert!(at_cap.retained_paired_target_bytes > 0);
    assert!(at_cap.retained_paired_dflash_bytes > 0);

    let over_cap = engine
        .generate_continued_with_thinking(
            PLATEAU_SID + 2,
            &prompt,
            1,
            &greedy(Speculation::DFlash),
            false,
        )
        .expect("insert paired session at cap plus one");
    assert!(!over_cap.continued);
    let plateau = engine.cache_stats();
    assert_eq!(
        plateau.retained_paired_sessions, 2,
        "the configured count cap must retain exactly two complete pairs"
    );
    assert_eq!(
        plateau.retained_paired_target_bytes, at_cap.retained_paired_target_bytes,
        "target bytes must plateau at the paired-session cap"
    );
    assert_eq!(
        plateau.retained_paired_dflash_bytes, at_cap.retained_paired_dflash_bytes,
        "dSpark bytes must plateau at the paired-session cap"
    );
    assert!(
        engine.retained_session_tokens(PLATEAU_SID).is_none(),
        "the least-recently-used session must lose the whole pair"
    );
    assert!(
        engine.retained_session_tokens(PLATEAU_SID + 1).is_some()
            && engine.retained_session_tokens(PLATEAU_SID + 2).is_some(),
        "the two newest sessions must retain complete pairs"
    );

    assert_eq!(
        engine.evict_idle_retained(std::time::Duration::ZERO),
        2,
        "real-model TTL expiry must evict both retained ownership entries"
    );
    let after_ttl = engine.cache_stats();
    assert_eq!(after_ttl.retained_sessions, 0);
    assert_eq!(after_ttl.retained_paired_sessions, 0);
    assert_eq!(after_ttl.retained_paired_target_bytes, 0);
    assert_eq!(after_ttl.retained_paired_dflash_bytes, 0);
}
