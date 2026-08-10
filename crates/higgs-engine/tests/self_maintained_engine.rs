//! Validates the *generic, engine-driven* self-maintenance loop: the whole task
//! is handed over as one prompt, and `generate_self_maintained` decides when to
//! checkpoint (segment budget) and asks the model to summarize its own progress
//! and position to resume. This is the harder hypothesis than the pre-chunked
//! harness — the orchestrator does NOT track position; the model must.
//!
//!   HIGGS_PRUNE_MODEL=/path/to/model \
//!     cargo test -p higgs-engine --test self_maintained_engine -- --ignored --nocapture
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

use higgs_engine::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
use higgs_engine::prune_eval::extract_number;
use higgs_engine::simple::{SelfMaintainCfg, SimpleEngine};
use higgs_models::SamplingParams;
use higgs_models::cache::RopeShift;
use higgs_models::turboquant::KvCacheConfig;

const START: i64 = 50;
const CYCLE: [i64; 6] = [3, -5, 4, -2, 6, -3];

fn find_num(v: &serde_json::Value, key: &str) -> Option<f64> {
    match v {
        serde_json::Value::Object(map) => map
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .or_else(|| map.values().find_map(|s| find_num(s, key))),
        serde_json::Value::Array(items) => items.iter().find_map(|s| find_num(s, key)),
        _ => None,
    }
}

#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL"]
fn self_maintained_engine() {
    let dir = std::env::var("HIGGS_PRUNE_MODEL").expect("set HIGGS_PRUNE_MODEL");
    let model_dir = Path::new(&dir);
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).unwrap())
            .unwrap();
    let rope = RopeShift {
        base: find_num(&config, "rope_theta").expect("rope_theta") as f32,
        dims: find_num(&config, "head_dim").expect("head_dim") as i32,
        scale: 1.0,
        traditional: false,
    };
    let tuning = MlxRuntimeTuning::from_model_dir(model_dir, RequestedMlxProfile::Auto);
    let engine =
        SimpleEngine::load(model_dir, KvCacheConfig::default(), tuning, false).expect("load");
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let enable_thinking = std::env::var("HIGGS_THINK").ok().is_some_and(|v| v != "0");

    let k = 36_usize;
    let ds: Vec<i64> = (0..k).map(|i| CYCLE[i % CYCLE.len()]).collect();
    let gold = START + ds.iter().sum::<i64>();
    let ops = ds
        .iter()
        .map(|&d| {
            if d >= 0 {
                format!("- add {d}")
            } else {
                format!("- subtract {}", -d)
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let task = format!(
        "A counter starts at {START}. Apply each of the following {k} operations in order to the running total, one at a time:\n{ops}\nWhat is the final value of the counter?"
    );

    // Small segment budget forces the engine to checkpoint mid-task.
    let cfg = SelfMaintainCfg {
        seg_max_tokens: if enable_thinking { 600 } else { 170 },
        summary_max_tokens: if enable_thinking { 400 } else { 160 },
        max_segments: 12,
        enable_thinking,
    };

    let out = engine
        .generate_self_maintained(&task, &params, rope, &cfg)
        .expect("generate");
    let got = extract_number(&out.text);
    let ok = got == Some(gold as f64);

    println!("thinking={enable_thinking} gold={gold}");
    println!(
        "result: {} (got {:?}) | segments={} peakKV={} total_tokens={}",
        if ok { "OK" } else { "WRONG" },
        got,
        out.segments,
        out.peak_resident_kv,
        out.total_tokens
    );
    for (i, s) in out.summaries.iter().enumerate() {
        println!(
            "  summary {}: {}",
            i + 1,
            s.replace('\n', " ").chars().take(160).collect::<String>()
        );
    }

    // The point of the test is to observe whether engine-driven checkpointing
    // resumes correctly; assert only that it ran and stayed bounded.
    assert!(out.segments >= 1);
    assert!(out.peak_resident_kv > 0);
}
