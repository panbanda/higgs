//! Self-maintenance vs pruning, head-to-head on a long running-state task.
//!
//! The task: a counter starting at 50, with K signed operations applied in
//! order. The final value depends on carrying the running total across the whole
//! (growing) reasoning trace — exactly the context that pruning destroys.
//!
//! Three modes at the *same* memory pressure:
//!   full       — one shot, no management (native self-maintenance)
//!   prune      — one shot with KV-prune (keep_frac 0.5)
//!   checkpoint — self-summary: the model re-states the running total every C
//!                ops and continues from it, bounding resident context itself
//!
//! Ignored by default (loads a real model):
//!   HIGGS_PRUNE_MODEL=/path/to/model \
//!     cargo test -p higgs-engine --test self_maintenance -- --ignored --nocapture
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
use higgs_engine::prune_eval::extract_number;
use higgs_engine::simple::{PrunedGeneration, SimpleEngine};
use higgs_models::SamplingParams;
use higgs_models::cache::RopeShift;
use higgs_models::turboquant::KvCacheConfig;

const START: i64 = 50;
const CHUNK: usize = 6;
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

fn deltas(k: usize) -> Vec<i64> {
    (0..k).map(|i| CYCLE[i % CYCLE.len()]).collect()
}

fn fmt_ops(ops: &[i64]) -> String {
    ops.iter()
        .map(|&d| {
            if d >= 0 {
                format!("- add {d}")
            } else {
                format!("- subtract {}", -d)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn run(
    engine: &SimpleEngine,
    params: &SamplingParams,
    rope: RopeShift,
    prompt: String,
    max_tokens: u32,
    policy: &PrunePolicy,
    enable_thinking: bool,
) -> PrunedGeneration {
    let msg = ChatMessage {
        role: "user".to_owned(),
        content: prompt,
        tool_calls: None,
    };
    let toks = engine
        .prepare_chat_prompt_with_thinking(std::slice::from_ref(&msg), None, enable_thinking)
        .expect("render");
    engine
        .generate_with_prune(&toks, max_tokens, params, policy, rope)
        .expect("generate")
}

#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL"]
fn self_maintenance_vs_pruning() {
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
    let disabled = PrunePolicy::disabled();
    let pruned = PrunePolicy {
        sink: 4,
        min_window: 64,
        keep_frac: 0.5,
        protect_facts: false,
    };

    // HIGGS_THINK=1 enables thinking mode; default is non-thinking.
    let enable_thinking = std::env::var("HIGGS_THINK").ok().is_some_and(|v| v != "0");
    let tfac: u32 = if enable_thinking { 4 } else { 1 };
    println!("thinking mode = {enable_thinking} (token budgets x{tfac} when thinking)");

    for &k in &[18_usize, 36] {
        let ds = deltas(k);
        let gold = START + ds.iter().sum::<i64>();
        let big_tokens = (512 + (k as u32) * 12) * tfac;
        println!("\n########## K={k} ops, gold final = {gold} ##########");

        // full (no management)
        let full_prompt = format!(
            "A counter starts at {START}. Apply each of the following operations in order to the running total:\n{}\nShow the running total after each step, then end with a line 'Answer: <final total>'.",
            fmt_ops(&ds)
        );
        let full = run(
            &engine,
            &params,
            rope,
            full_prompt.clone(),
            big_tokens,
            &disabled,
            enable_thinking,
        );
        let full_val = extract_number(&full.text);
        let full_ok = full_val == Some(gold as f64);

        // prune (keep_frac 0.5)
        let pr = run(
            &engine,
            &params,
            rope,
            full_prompt,
            big_tokens,
            &pruned,
            enable_thinking,
        );
        let pr_val = extract_number(&pr.text);
        let pr_ok = pr_val == Some(gold as f64);

        // checkpoint (self-summary): re-state running total every CHUNK ops
        let (mut state, mut peak, mut toks, mut calls) = (START as f64, 0_u32, 0_u32, 0_u32);
        for chunk in ds.chunks(CHUNK) {
            let prompt = format!(
                "A counter currently equals {}. Apply each of the following operations in order to the running total:\n{}\nShow the running total after each step, then end with a line 'Answer: <final total>'.",
                state as i64,
                fmt_ops(chunk)
            );
            let out = run(
                &engine,
                &params,
                rope,
                prompt,
                256 * tfac,
                &disabled,
                enable_thinking,
            );
            if let Some(v) = extract_number(&out.text) {
                state = v;
            }
            peak = peak.max(out.peak_resident_kv);
            toks += out.completion_tokens;
            calls += 1;
        }
        let cp_ok = (state - gold as f64).abs() < 0.5;

        println!("mode        | correct | final     | peakKV | tokens | calls");
        println!("------------+---------+-----------+--------+--------+------");
        println!(
            "full        | {:>7} | {:>9} | {:>6} | {:>6} | {:>4}",
            yn(full_ok),
            show(full_val),
            full.peak_resident_kv,
            full.completion_tokens,
            1
        );
        println!(
            "prune 50%   | {:>7} | {:>9} | {:>6} | {:>6} | {:>4}",
            yn(pr_ok),
            show(pr_val),
            pr.peak_resident_kv,
            pr.completion_tokens,
            1
        );
        println!(
            "checkpoint  | {:>7} | {:>9} | {:>6} | {:>6} | {:>4}",
            yn(cp_ok),
            format!("{}", state as i64),
            peak,
            toks,
            calls
        );
    }
}

fn yn(b: bool) -> &'static str {
    if b { "OK" } else { "WRONG" }
}

fn show(v: Option<f64>) -> String {
    v.map_or_else(|| "—".to_owned(), |n| format!("{}", n as i64))
}
