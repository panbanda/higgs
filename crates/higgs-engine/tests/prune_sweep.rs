//! Aggregate KV-prune accuracy sweep over the 50-problem set.
//!
//! Ignored by default — loads a real model. Example:
//!   HIGGS_PRUNE_MODEL=/path/to/qwen3.6-35b-a3b-4bit \
//!     cargo test -p higgs-engine --test prune_sweep -- --ignored --nocapture
//!
//! Env knobs:
//!   HIGGS_PRUNE_N      limit number of problems (default: all 50)
//!   HIGGS_PRUNE_RATES  comma-separated keep_fracs (default: 1.0,0.7,0.55,0.4)
//!   HIGGS_PRUNE_MAXTOK max tokens per problem (default: 384)
//!   HIGGS_PRUNE_STRUCT 1 = structural (protect facts) instead of age-based
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr,
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
use higgs_engine::prune_eval::{SweepRow, grade, problem_set, prompt_for, render_table};
use higgs_engine::simple::SimpleEngine;
use higgs_models::SamplingParams;
use higgs_models::cache::RopeShift;
use higgs_models::turboquant::KvCacheConfig;

fn find_num(v: &serde_json::Value, key: &str) -> Option<f64> {
    match v {
        serde_json::Value::Object(map) => map
            .get(key)
            .and_then(serde_json::Value::as_f64)
            .or_else(|| map.values().find_map(|sub| find_num(sub, key))),
        serde_json::Value::Array(items) => items.iter().find_map(|sub| find_num(sub, key)),
        _ => None,
    }
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL to the model dir"]
fn prune_sweep() {
    let dir = std::env::var("HIGGS_PRUNE_MODEL")
        .expect("set HIGGS_PRUNE_MODEL to a Qwen3 model directory");
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

    let max_tokens: u32 = env_or("HIGGS_PRUNE_MAXTOK", 384);
    let protect_facts = env_or::<u32>("HIGGS_PRUNE_STRUCT", 0) != 0;
    let rates: Vec<f32> = std::env::var("HIGGS_PRUNE_RATES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1.0, 0.7, 0.55, 0.4]);

    let mut problems = problem_set();
    if let Ok(raw) = std::env::var("HIGGS_PRUNE_N") {
        if let Ok(limit) = raw.parse::<usize>() {
            problems.truncate(limit);
        }
    }

    println!(
        "model={dir}\nproblems={} rates={rates:?} max_tokens={max_tokens} protect_facts={protect_facts}",
        problems.len()
    );

    let tuning = MlxRuntimeTuning::from_model_dir(model_dir, RequestedMlxProfile::Auto);
    let engine =
        SimpleEngine::load(model_dir, KvCacheConfig::default(), tuning, false).expect("load model");
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    let mut rows: Vec<SweepRow> = Vec::new();
    for &keep_frac in &rates {
        let policy = PrunePolicy {
            sink: 4,
            min_window: 64,
            keep_frac,
            protect_facts,
        };
        let prune_pct = ((1.0 - keep_frac) * 100.0).round() as u32;

        let (mut correct, mut sum_peak, mut sum_tps) = (0_u32, 0.0_f32, 0.0_f32);
        for (i, problem) in problems.iter().enumerate() {
            let msg = ChatMessage {
                role: "user".to_owned(),
                content: prompt_for(problem),
                tool_calls: None,
            };
            let toks = engine
                .prepare_chat_prompt_with_thinking(std::slice::from_ref(&msg), None, false)
                .expect("render prompt");
            let out = engine
                .generate_with_prune(&toks, max_tokens, &params, &policy, rope)
                .expect("generate");
            let ok = grade(&out.text, problem.answer);
            correct += u32::from(ok);
            sum_peak += out.peak_resident_kv as f32;
            if out.decode_seconds > 0.0 {
                sum_tps += out.completion_tokens as f32 / out.decode_seconds;
            }
            eprintln!(
                "[{prune_pct:>2}% prune {:>2}/{}] {:<14} gold={:<10} {}",
                i + 1,
                problems.len(),
                problem.category,
                problem.answer,
                if ok { "OK" } else { "WRONG" }
            );
        }
        let n = problems.len() as u32;
        rows.push(SweepRow {
            prune_pct,
            accuracy: correct as f32 / n.max(1) as f32,
            mean_peak_kv: sum_peak / n.max(1) as f32,
            mean_tok_per_s: sum_tps / n.max(1) as f32,
            n,
        });
    }

    println!("\n{}", render_table(&rows, 0.05));
    assert!(!rows.is_empty());
}
