//! End-to-end smoke test for the KV-prune decode runner.
//!
//! Ignored by default — it loads a real model. Run with:
//!   HIGGS_PRUNE_MODEL=/path/to/qwen3.6-35b-a3b-4bit \
//!     cargo test -p higgs-engine --test prune_smoke -- --ignored --nocapture
//!
//! It generates the same prompt at several `keep_frac` levels and prints text +
//! metrics so you can eyeball whether a stock model stays coherent under
//! pruning. It asserts only that generation runs and produces tokens — grading
//! is the sweep harness's job.
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

use higgs_engine::chat_template::ChatMessage;
use higgs_engine::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
use higgs_engine::prune::PrunePolicy;
use higgs_engine::simple::SimpleEngine;
use higgs_models::SamplingParams;
use higgs_models::cache::RopeShift;
use higgs_models::turboquant::KvCacheConfig;

/// First numeric value for `key` anywhere in the (possibly nested) JSON.
fn find_num(v: &serde_json::Value, key: &str) -> Option<f64> {
    match v {
        serde_json::Value::Object(map) => {
            if let Some(found) = map.get(key).and_then(serde_json::Value::as_f64) {
                return Some(found);
            }
            map.values().find_map(|sub| find_num(sub, key))
        }
        serde_json::Value::Array(items) => items.iter().find_map(|sub| find_num(sub, key)),
        _ => None,
    }
}

#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL to the model dir"]
fn prune_smoke() {
    let dir = std::env::var("HIGGS_PRUNE_MODEL")
        .expect("set HIGGS_PRUNE_MODEL to a Qwen3 model directory");
    let model_dir = Path::new(&dir);

    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).unwrap())
            .unwrap();
    let rope = RopeShift {
        base: find_num(&config, "rope_theta").expect("rope_theta in config") as f32,
        dims: find_num(&config, "head_dim").expect("head_dim in config") as i32,
        scale: 1.0,
        traditional: false,
    };
    println!("rope: base={} dims={}", rope.base, rope.dims);

    let tuning = MlxRuntimeTuning::from_model_dir(model_dir, RequestedMlxProfile::Auto);
    let engine =
        SimpleEngine::load(model_dir, KvCacheConfig::default(), tuning, false).expect("load model");

    let msg = ChatMessage {
        role: "user".to_owned(),
        content: "Natalia sold clips to 48 of her friends in April, then sold half as many \
                  clips in May. How many clips did she sell altogether in April and May? \
                  Reason step by step, then end with a line 'Answer: <number>'."
            .to_owned(),
        tool_calls: None,
    };
    let prompt_tokens = engine
        .prepare_chat_prompt_with_thinking(std::slice::from_ref(&msg), None, false)
        .expect("render prompt");
    println!("prompt tokens: {}", prompt_tokens.len());

    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };
    let max_tokens: u32 = 320;

    // Knee sweep: age-based prune at increasing aggressiveness to find where
    // stock reasoning breaks (expected answer = 72).
    let scenarios = [
        (1.0_f32, false, "0% (baseline)"),
        (0.85, false, "~15%"),
        (0.7, false, "~30%"),
        (0.6, false, "~40%"),
        (0.5, false, "~50%"),
        (0.4, false, "~60%"),
    ];
    for (keep_frac, protect_facts, label) in scenarios {
        let policy = PrunePolicy {
            sink: 4,
            min_window: 64,
            keep_frac,
            protect_facts,
        };
        let out = engine
            .generate_with_prune(&prompt_tokens, max_tokens, &params, &policy, rope)
            .expect("generate");
        let tok_per_s = if out.decode_seconds > 0.0 {
            out.completion_tokens as f32 / out.decode_seconds
        } else {
            0.0
        };
        println!("\n===== {label} =====");
        println!(
            "tokens={} peak_resident_kv={} pruned_steps={} tok/s={tok_per_s:.1}",
            out.completion_tokens, out.peak_resident_kv, out.pruned_steps
        );
        println!("text: {}", out.text.trim());
        assert!(out.completion_tokens > 0, "no tokens generated");
    }
}
