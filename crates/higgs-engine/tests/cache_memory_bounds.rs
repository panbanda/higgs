//! Tier-3: cache-resident KV retention limits bound the live cache, end-to-end.
//!
//! Proves the config → engine → enforcement wiring on a real model: with
//! `max_retained_sessions = 2`, driving four distinct conversations leaves at
//! most two retained live KV caches (LRU-evicted), so resident KV stays bounded.
//! The cap/TTL/token-cap *logic* is unit-tested in `simple.rs`; this checks the
//! wire from `KvCacheConfig` through `generate_continued` to the retained map.
//!
//! Ignored by default (loads a real model):
//!   HIGGS_PRUNE_MODEL=/path/to/model \
//!     cargo test -p higgs-engine --test cache_memory_bounds -- --ignored --nocapture
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::tests_outside_test_module,
    clippy::doc_markdown
)]

use std::path::Path;

use higgs_engine::chat_template::ChatMessage;
use higgs_engine::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
use higgs_engine::simple::SimpleEngine;
use higgs_models::SamplingParams;
use higgs_models::turboquant::KvCacheConfig;

#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL"]
fn retained_count_cap_bounds_live_sessions() {
    let dir = std::env::var("HIGGS_PRUNE_MODEL").expect("set HIGGS_PRUNE_MODEL");
    let model_dir = Path::new(&dir);
    let tuning = MlxRuntimeTuning::from_model_dir(model_dir, RequestedMlxProfile::Auto);

    // Cap retained conversations at 2 (everything else default).
    let cfg = KvCacheConfig {
        max_retained_sessions: 2,
        ..Default::default()
    };
    let engine = SimpleEngine::load(model_dir, cfg, tuning, false).expect("load model");
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    // Four distinct conversations, each retained then LRU-evicted past the cap.
    for sid in 1u64..=4 {
        let msg = ChatMessage {
            role: "user".to_owned(),
            content: format!("Reply with just the word ok (request {sid})."),
            tool_calls: None,
        };
        let toks = engine
            .prepare_chat_prompt_with_thinking(std::slice::from_ref(&msg), None, false)
            .expect("render");
        engine
            .generate_continued(sid, &toks, 8, &params)
            .expect("generate");
        let n = engine.retained_session_count();
        assert!(n <= 2, "count cap exceeded after session {sid}: {n} > 2");
    }

    assert_eq!(
        engine.retained_session_count(),
        2,
        "exactly the configured cap of retained sessions survives four conversations"
    );
    println!(
        "retained count cap holds: {} live sessions after 4 distinct conversations",
        engine.retained_session_count()
    );

    // Observability (Tier-6): the stats snapshot reflects the same reality.
    let stats = engine.cache_stats();
    assert_eq!(
        stats.retained_sessions, 2,
        "cache_stats.retained_sessions matches the live count"
    );
    assert!(
        stats.sessions_evicted >= 2,
        "cache_stats reports the LRU evictions (4 sessions, cap 2): got {}",
        stats.sessions_evicted
    );
    println!(
        "cache_stats: retained={} evicted={} continuations={} radix_hits={}/{}",
        stats.retained_sessions,
        stats.sessions_evicted,
        stats.continuations,
        stats.radix_hits,
        stats.radix_lookups
    );
}
