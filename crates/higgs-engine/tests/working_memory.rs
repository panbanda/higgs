//! The "third way" test: model-authored running notes (anchors) vs one-shot, on
//! a NON-patterned sequential task the model can't shortcut.
//!
//! Task: three registers A,B,C with K ops (add / subtract / copy / set). Tracking
//! the final A requires genuine step-by-step state — no closed form.
//!
//! `working_memory_rate` is the decisive test: N distinct tasks, accuracy *rate*
//! for one-shot vs self-notes, plus how often the model actually emits NOTES.
//! `working_memory_detail` shows one K with the summarize-loop as a cost ref.
//!
//!   HIGGS_PRUNE_MODEL=/path/to/model \
//!     cargo test -p higgs-engine --test working_memory -- --ignored --nocapture
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
use higgs_engine::simple::SimpleEngine;
use higgs_models::SamplingParams;
use higgs_models::cache::RopeShift;
use higgs_models::turboquant::KvCacheConfig;

const NAMES: [char; 3] = ['A', 'B', 'C'];
const START: [i64; 3] = [10, 20, 30];

#[derive(Clone, Copy)]
enum Op {
    Add(usize, i64),
    Sub(usize, i64),
    Copy(usize, usize),
    SetN(usize, i64),
}

fn gen_ops(k: usize, seed: usize) -> Vec<Op> {
    (0..k)
        .map(|i| {
            let j = i + seed * 13;
            let var = (j * 5 + 1) % 3;
            let operand = ((j * 3) % 9 + 1) as i64;
            let src = (j * 2 + 2) % 3;
            match (j * 7 + 3) % 4 {
                0 => Op::Add(var, operand),
                1 => Op::Sub(var, operand),
                2 => Op::Copy(var, src),
                _ => Op::SetN(var, operand),
            }
        })
        .collect()
}

fn simulate(ops: &[Op]) -> i64 {
    let mut v = START;
    for op in ops {
        match *op {
            Op::Add(x, n) => v[x] += n,
            Op::Sub(x, n) => v[x] -= n,
            Op::Copy(x, s) => v[x] = v[s],
            Op::SetN(x, n) => v[x] = n,
        }
    }
    v[0]
}

fn fmt_ops(ops: &[Op]) -> String {
    ops.iter()
        .enumerate()
        .map(|(i, op)| {
            let body = match *op {
                Op::Add(x, n) => format!("add {n} to {}", NAMES[x]),
                Op::Sub(x, n) => format!("subtract {n} from {}", NAMES[x]),
                Op::Copy(x, s) => {
                    format!("set {} to the current value of {}", NAMES[x], NAMES[s])
                }
                Op::SetN(x, n) => format!("set {} to {n}", NAMES[x]),
            };
            format!("{}. {body}", i + 1)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

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

fn setup() -> (SimpleEngine, RopeShift, SamplingParams) {
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
    (engine, rope, params)
}

/// One greedy generation; returns (text, completion_tokens, peak_resident_kv).
fn run_one(
    engine: &SimpleEngine,
    params: &SamplingParams,
    rope: RopeShift,
    prompt: String,
    max_tokens: u32,
) -> (String, u32, u32) {
    let msg = ChatMessage {
        role: "user".to_owned(),
        content: prompt,
        tool_calls: None,
    };
    let toks = engine
        .prepare_chat_prompt_with_thinking(std::slice::from_ref(&msg), None, false)
        .expect("render");
    let out = engine
        .generate_with_prune(&toks, max_tokens, params, &PrunePolicy::disabled(), rope)
        .expect("gen");
    (out.text, out.completion_tokens, out.peak_resident_kv)
}

fn header_for(ops: &[Op], k: usize) -> String {
    format!(
        "Three integer variables start at A=10, B=20, C=30. Apply these {k} operations in order:\n{}\n",
        fmt_ops(ops)
    )
}

fn prompt_oneshot(header: &str) -> String {
    format!(
        "{header}Work through the operations one at a time, tracking A, B, and C. End with a line 'Answer: <number>' giving only the final value of A."
    )
}

fn prompt_notes(header: &str) -> String {
    format!(
        "{header}Process the operations in order. After every 6 operations, write one checkpoint line exactly like 'NOTES: A=_, B=_, C=_' with the current values, then keep going. End with a line 'Answer: <number>' giving only the final value of A."
    )
}

#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL"]
fn working_memory_rate() {
    let (engine, rope, params) = setup();
    let k = 36_usize;
    let n = 8_usize;
    let budget = 700 + (k as u32) * 18;

    let (mut a_ok, mut c_ok, mut notes_seen) = (0_u32, 0_u32, 0_u32);
    let (mut a_tok_sum, mut c_tok_sum) = (0_u32, 0_u32);
    println!("K={k}, {n} distinct tasks, one-shot vs self-notes\n");
    println!("seed | gold | one-shot | self-notes | notes?");
    println!("-----+------+----------+------------+-------");
    for seed in 0..n {
        let ops = gen_ops(k, seed);
        let gold = simulate(&ops);
        let header = header_for(&ops, k);

        let (a_text, a_tok, _) = run_one(&engine, &params, rope, prompt_oneshot(&header), budget);
        let (c_text, c_tok, _) =
            run_one(&engine, &params, rope, prompt_notes(&header), budget + 200);
        let a_v = extract_number(&a_text);
        let c_v = extract_number(&c_text);
        let a_hit = a_v == Some(gold as f64);
        let c_hit = c_v == Some(gold as f64);
        let emitted = c_text.contains("NOTES:");
        a_ok += u32::from(a_hit);
        c_ok += u32::from(c_hit);
        notes_seen += u32::from(emitted);
        a_tok_sum += a_tok;
        c_tok_sum += c_tok;

        // Dump the first task's self-notes output so we can SEE the anchoring.
        if seed == 0 {
            let snippet: String = c_text.chars().take(600).collect();
            println!("--- seed 0 self-notes output (first 600 chars) ---\n{snippet}\n---");
        }
        println!(
            "{seed:>4} | {gold:>4} | {:>8} | {:>10} | {}",
            if a_hit { "OK" } else { "x" },
            if c_hit { "OK" } else { "x" },
            if emitted { "yes" } else { "NO" },
        );
    }

    println!("\n=== RATE (n={n}, K={k}) ===");
    println!(
        "one-shot   accuracy {a_ok}/{n}  avg_tokens {}",
        a_tok_sum / n as u32
    );
    println!(
        "self-notes accuracy {c_ok}/{n}  avg_tokens {}  NOTES emitted {notes_seen}/{n}",
        c_tok_sum / n as u32
    );
    assert!(notes_seen <= n as u32);
}

/// Drift curve: one-shot vs self-notes accuracy as chain length K grows, all at
/// full context (no pruning). Resident KV is reported to show it stays tiny —
/// ruling out capacity, isolating reasoning drift. If self-notes' lead widens
/// with K while KV stays small, that's the finding.
#[test]
#[ignore = "loads a real model; set HIGGS_PRUNE_MODEL"]
fn drift_curve() {
    let (engine, rope, params) = setup();
    let seeds = 5_usize;
    println!("Drift curve (n={seeds} tasks/K): one-shot vs self-notes accuracy as K grows.\n");
    println!("  K | one-shot | self-notes | max_resident_KV");
    println!("----+----------+------------+----------------");
    for &k in &[24_usize, 42, 60, 80] {
        let budget = 600 + (k as u32) * 22;
        let (mut a_ok, mut c_ok, mut kv_max) = (0_u32, 0_u32, 0_u32);
        for seed in 0..seeds {
            let ops = gen_ops(k, seed);
            let gold = simulate(&ops);
            let header = header_for(&ops, k);
            let (a_text, _, a_kv) =
                run_one(&engine, &params, rope, prompt_oneshot(&header), budget);
            let (c_text, _, c_kv) =
                run_one(&engine, &params, rope, prompt_notes(&header), budget + 250);
            a_ok += u32::from(extract_number(&a_text) == Some(gold as f64));
            c_ok += u32::from(extract_number(&c_text) == Some(gold as f64));
            kv_max = kv_max.max(a_kv).max(c_kv);
        }
        println!("{k:>3} |   {a_ok}/{seeds}    |    {c_ok}/{seeds}     | {kv_max:>14}");
    }
}
