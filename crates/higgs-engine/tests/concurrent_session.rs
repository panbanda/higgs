//! Tier-2: per-session serialization holds under real concurrency.
//!
//! Many threads fire `SimpleEngine::generate_continued` for the SAME `session_id`
//! at once. Per-session serialization (a lock held for the whole call, acquired
//! before the model lock) must make this safe and correct:
//!   - no panic / native SIGABRT (the MLX gate + session lock serialize all GPU
//!     work; two requests for one conversation never interleave take/stash);
//!   - the same prompt from every thread yields the SAME greedy output as a
//!     sequential single-shot baseline — i.e. no cache poisoning / corruption;
//!   - a colliding DIFFERENT prompt on the same id still produces its own correct
//!     output (the strict prefix guard rejects the mismatched cache → full
//!     prefill);
//!   - the engine remains usable afterwards.
//!
//! Ignored by default (loads a real model):
//!   HIGGS_PRUNE_MODEL=/path/to/model \
//!     cargo test -p higgs-engine --test concurrent_session -- --ignored --nocapture
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::tests_outside_test_module,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::items_after_statements,
    clippy::shadow_reuse,
    clippy::shadow_same,
    // We must collect the JoinHandles so every thread is spawned BEFORE any join;
    // a lazy iterator would spawn-and-join serially, defeating the concurrency test.
    clippy::needless_collect
)]

use std::path::Path;
use std::sync::Arc;
use std::thread;

use higgs_engine::chat_template::ChatMessage;
use higgs_engine::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
use higgs_engine::simple::SimpleEngine;
use higgs_models::SamplingParams;
use higgs_models::turboquant::KvCacheConfig;

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        ..Default::default()
    }
}

#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL"]
fn concurrent_same_session_is_serialized_and_correct() {
    let dir = std::env::var("HIGGS_PRUNE_MODEL").expect("set HIGGS_PRUNE_MODEL");
    let model_dir = Path::new(&dir);
    let tuning = MlxRuntimeTuning::from_model_dir(model_dir, RequestedMlxProfile::Auto);
    let engine = Arc::new(
        SimpleEngine::load(model_dir, KvCacheConfig::default(), tuning, false).expect("load model"),
    );

    let render = |content: &str| -> Vec<u32> {
        let msg = ChatMessage {
            role: "user".to_owned(),
            content: content.to_owned(),
            tool_calls: None,
        };
        engine
            .prepare_chat_prompt_with_thinking(std::slice::from_ref(&msg), None, false)
            .expect("render")
    };

    let prompt_a = render("List the planets of the solar system in order from the sun.");

    // Baseline: one clean generation of prompt_a (greedy ⇒ deterministic).
    let baseline = engine
        .generate_continued(1, &prompt_a, 48, &greedy())
        .expect("baseline")
        .text;
    assert!(!baseline.is_empty(), "baseline must produce output");

    // Stress: N threads all fire prompt_a on the SAME session id at once.
    const N: usize = 8;
    const SID: u64 = 777;
    let handles: Vec<_> = (0..N)
        .map(|_| {
            let engine = Arc::clone(&engine);
            let prompt = prompt_a.clone();
            thread::spawn(move || {
                engine
                    .generate_continued(SID, &prompt, 48, &greedy())
                    .map(|g| g.text)
            })
        })
        .collect();
    let results: Vec<String> = handles
        .into_iter()
        .map(|h| {
            h.join()
                .expect("worker thread must not panic/abort")
                .expect("generate_continued must succeed")
        })
        .collect();

    // Serialization + deterministic greedy ⇒ every concurrent same-prompt result
    // equals the baseline. Any divergence would be cache poisoning/corruption.
    for (i, r) in results.iter().enumerate() {
        assert_eq!(
            r, &baseline,
            "thread {i} diverged from the baseline — concurrent same-session corruption"
        );
    }

    // A colliding DIFFERENT prompt on the same id must still be correct: the
    // strict prefix guard rejects the mismatched retained cache and full-prefills.
    let prompt_b = render("What is the capital of France? Answer in one word.");
    let baseline_b = engine
        .generate_continued(2, &prompt_b, 16, &greedy())
        .expect("baseline_b")
        .text;
    let collide = engine
        .generate_continued(SID, &prompt_b, 16, &greedy())
        .expect("collide")
        .text;
    assert_eq!(
        collide, baseline_b,
        "colliding session_id must still produce the correct independent output"
    );

    // Engine still usable after the stress.
    let post = engine
        .generate_continued(3, &prompt_a, 16, &greedy())
        .expect("post-stress")
        .text;
    assert!(
        !post.is_empty(),
        "engine must remain usable after concurrent stress"
    );

    println!("\n=== concurrent same-session: PASS ===");
    println!(
        "{N} concurrent same-id requests all matched the baseline; collision handled; engine healthy."
    );
}
