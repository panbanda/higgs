#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::shadow_unrelated,
    clippy::shadow_reuse,
    clippy::shadow_same,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::useless_vec
)]
//! `bench_mlx_tuning` — port of `benchmarks/bench_mlx_tuning.py`.
//!
//! Sweeps five MLX tuning profiles (baseline / latency / balanced /
//! throughput / throughput+TQ) and reports a composite score combining
//! TTFT, decode tok/s, short-QA accuracy, long-context retrieval,
//! structured-output correctness, and prefix-cache speedup. The composite
//! formula is implemented in `higgs_bench::score` and exercised by the
//! unit tests at the bottom of this file (ports of
//! `benchmarks/test_bench_mlx_tuning.py`).

use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use higgs_bench::http::{StreamChatOptions, StreamMetrics};
use higgs_bench::score::{
    AccuracyInputs, CacheInputs, SpeedInputs, compute_bests, compute_iteration_score,
    normalize_text,
};
use higgs_bench::stats::median;
use higgs_bench::{
    BenchOutput, ModelInfo, OutputFormat, RunMetadata, format_json, format_markdown, http,
    persist_result, process,
};
use serde::Serialize;

const PORT: u16 = 8099;

const SHORT_PROMPT: &str = "What is 17 + 25? Reply with digits only.";
const MEDIUM_PROMPT: &str = "Explain how KV cache reuse affects time-to-first-token and decode throughput for autoregressive transformer inference on Apple Silicon. Keep the answer technical and concise.";

const LONG_CONTEXT_NEEDLE: &str = "CAPE-TOWN-7419";

#[derive(Debug, Parser)]
#[command(
    name = "bench_mlx_tuning",
    about = "Sweep MLX tuning profiles and emit a composite score (port of bench_mlx_tuning.py)",
    version
)]
struct Args {
    /// Model path (local directory or HF cache directory) — same shape as
    /// the Python script.
    model_path: String,

    /// Repeats per prompt in the prompt sweep.
    #[arg(long, default_value_t = 2)]
    repeats: u32,

    /// Port the spawned higgs server should listen on.
    #[arg(long, default_value_t = PORT)]
    port: u16,

    /// Skip spawning a server; assume one is already running on `--port`.
    /// In this mode only the first iteration runs (no profile sweep).
    #[arg(long)]
    no_spawn: bool,

    /// Server startup timeout (seconds).
    #[arg(long, default_value_t = 120)]
    server_timeout_s: u64,

    /// Output format (json, markdown).
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
}

#[derive(Debug, Clone)]
struct Iteration {
    slug: &'static str,
    label: &'static str,
    profile: &'static str,
    args: &'static [&'static str],
    notes: &'static str,
}

const ITERATIONS: &[Iteration] = &[
    Iteration {
        slug: "baseline",
        label: "1. Baseline",
        profile: "baseline",
        args: &[],
        notes: "Current conservative defaults",
    },
    Iteration {
        slug: "latency",
        label: "2. Latency Profile",
        profile: "latency",
        args: &[],
        notes: "Favor single-pass prefill and speculative decode",
    },
    Iteration {
        slug: "balanced",
        label: "3. Balanced Profile",
        profile: "balanced",
        args: &[],
        notes: "Model-aware chunking plus larger paged KV budget",
    },
    Iteration {
        slug: "throughput",
        label: "4. Throughput Profile",
        profile: "throughput",
        args: &[],
        notes: "Bigger decode-oriented chunks and paged KV budget",
    },
    Iteration {
        slug: "throughput_turboquant",
        label: "5. Throughput + Safe TurboQuant",
        profile: "throughput",
        args: &[
            "--kv-cache",
            "turboquant",
            "--kv-bits",
            "3",
            "--kv-key-bits",
            "2",
            "--kv-value-bits",
            "3",
            "--kv-adaptive-dense-layers",
            "8",
        ],
        notes: "Adds quality-preserving KV quantization after MLX runtime tuning",
    },
];

const QA_CASES: &[(&str, &str)] = &[
    ("Reply with digits only. What is 37 * 19?", "703"),
    (
        "Reply with one lowercase word only. Which word appears twice in 'alpha beta gamma beta delta'?",
        "beta",
    ),
    (
        "Reply with digits only. How many vowels are in the word instrumentation?",
        "6",
    ),
    (
        "Reply with lowercase letters only. Reverse the string stressed.",
        "desserts",
    ),
    (
        "Reply with comma-separated digits only. Sort 7,1,9,1 ascending.",
        "1,1,7,9",
    ),
];

fn long_prompt() -> String {
    use std::fmt::Write as _;
    let mut s = String::from(
        "Write a technical note about optimizing LLM inference on unified-memory Apple Silicon systems. Cover TTFT, decode throughput, prompt length sensitivity, prefix cache reuse, chunked prefill, speculative decode, and quantized KV caches. Include concrete engineering tradeoffs and failure modes.\n\n",
    );
    for idx in 1..=90 {
        let _ = writeln!(
            s,
            "Section {idx}: Repeated background detail about scheduler fairness, prompt staging, and kernel launch overhead on MLX devices."
        );
    }
    s
}

fn long_context_filler() -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    for idx in 1..=180 {
        let _ = writeln!(
            s,
            "Paragraph {idx}: Cape Town cluster notes about unified memory pressure, prefill staging, and context reuse across serving workloads."
        );
    }
    s.pop();
    s
}

fn prefix_cache_doc() -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    for idx in 1..=160 {
        let _ = writeln!(
            s,
            "Policy {idx}: Route latency data through the prefill pipeline, keep the region failover target as cape town, and retain prefix blocks for reuse."
        );
    }
    s.pop();
    s
}

fn structured_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": "mlx_report",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "model_family": {"type": "string"},
                    "iteration": {"type": "integer"},
                    "prefill_focus": {"type": "boolean"},
                    "kv_bits": {"type": "integer"},
                    "primary_goal": {"type": "string"},
                },
                "required": [
                    "model_family",
                    "iteration",
                    "prefill_focus",
                    "kv_bits",
                    "primary_goal",
                ],
                "additionalProperties": false,
            },
        },
    })
}

#[derive(Debug, Serialize)]
struct Params {
    model_path: String,
    port: u16,
    repeats: u32,
    iterations: Vec<String>,
}

#[derive(Debug, Serialize, Clone)]
struct PromptMetric {
    ttft_ms: f64,
    decode_tps: f64,
    prompt_tokens: f64,
    completion_tokens: f64,
}

#[derive(Debug, Serialize, Clone)]
struct PromptSweep {
    short: PromptMetric,
    medium: PromptMetric,
    long: PromptMetric,
    weighted_ttft_ms: f64,
    weighted_decode_tps: f64,
}

#[derive(Debug, Serialize, Clone)]
struct QaCaseResult {
    prompt: String,
    expected: String,
    output: String,
    passed: bool,
}

#[derive(Debug, Serialize, Clone)]
struct QaResults {
    passed: u32,
    total: u32,
    accuracy: f64,
    cases: Vec<QaCaseResult>,
}

#[derive(Debug, Serialize, Clone)]
struct LongContextResult {
    expected: String,
    output: String,
    passed: bool,
    accuracy: f64,
    prompt_tokens: u32,
}

#[derive(Debug, Serialize, Clone)]
struct StructuredResult {
    passed: bool,
    accuracy: f64,
    raw: Option<String>,
    parsed: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Clone)]
struct PrefixCacheResult {
    cold_ttft_ms: f64,
    warm_ttft_ms: f64,
    speedup: f64,
    answer: String,
    passed: bool,
    accuracy: f64,
}

#[derive(Debug, Serialize, Clone)]
struct ScoreOut {
    accuracy: f64,
    speed: f64,
    cache: f64,
    composite: f64,
}

#[derive(Debug, Serialize, Clone)]
struct IterationResult {
    iteration: String,
    label: String,
    notes: String,
    args: Vec<String>,
    prompt_sweep: PromptSweep,
    qa: QaResults,
    long_context: LongContextResult,
    structured_output: StructuredResult,
    prefix_cache: PrefixCacheResult,
    score: Option<ScoreOut>,
}

#[derive(Debug, Serialize)]
struct Results {
    model_path: String,
    iterations: Vec<IterationResult>,
    winner: Option<String>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: failed to start tokio runtime: {e}");
            return ExitCode::from(2);
        }
    };
    match runtime.block_on(run(args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

async fn run(args: Args) -> Result<()> {
    let mut metadata = RunMetadata::capture("bench_mlx_tuning");
    let started = Instant::now();
    metadata.model = Some(ModelInfo {
        key: args.model_path.clone(),
        path: args.model_path.clone(),
        quantization: "unknown".into(),
        approx_size_gb: 0.0,
    });

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let base_url = format!("http://127.0.0.1:{}", args.port);

    let iterations: Vec<Iteration> = if args.no_spawn {
        ITERATIONS.iter().take(1).cloned().collect()
    } else {
        ITERATIONS.to_vec()
    };

    let mut iter_results = Vec::new();
    for (idx, iteration) in iterations.iter().enumerate() {
        let result = benchmark_iteration(&client, &base_url, &args, idx + 1, iteration)
            .await
            .with_context(|| format!("iteration '{}'", iteration.slug))?;
        iter_results.push(result);
    }

    score_results(&mut iter_results);

    let winner = iter_results
        .iter()
        .max_by(|a, b| {
            let aa = a.score.as_ref().map_or(0.0, |s| s.composite);
            let bb = b.score.as_ref().map_or(0.0, |s| s.composite);
            aa.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.label.clone());

    print_summary(&iter_results, winner.as_deref());

    metadata.duration_ms = started.elapsed().as_millis() as u64;
    let params = Params {
        model_path: args.model_path.clone(),
        port: args.port,
        repeats: args.repeats,
        iterations: iterations.iter().map(|i| i.slug.to_owned()).collect(),
    };
    let results = Results {
        model_path: args.model_path.clone(),
        iterations: iter_results,
        winner,
    };
    let output = BenchOutput {
        metadata,
        params,
        results,
    };
    let path = persist_result(&output)?;
    eprintln!("[persisted] {}", path.display());
    let rendered = match args.format {
        OutputFormat::Json => format_json(&output)?,
        OutputFormat::Markdown => format_markdown(&output)?,
    };
    println!("{rendered}");
    Ok(())
}

async fn benchmark_iteration(
    client: &reqwest::Client,
    base_url: &str,
    args: &Args,
    iteration_index: usize,
    iteration: &Iteration,
) -> Result<IterationResult> {
    eprintln!("\n{}", "=".repeat(80));
    eprintln!("{}", iteration.label);
    eprintln!("Notes: {}", iteration.notes);
    if !iteration.args.is_empty() {
        eprintln!("Args: {}", iteration.args.join(" "));
    }
    eprintln!("{}", "=".repeat(80));

    let child = if args.no_spawn {
        None
    } else {
        let extra_args: Vec<String> = iteration.args.iter().map(|s| (*s).to_owned()).collect();
        let extra_env = vec![("HIGGS_MLX_PROFILE".to_owned(), iteration.profile.to_owned())];
        Some(
            process::start_higgs_server_with_env(
                &args.model_path,
                args.port,
                &extra_args,
                &extra_env,
                Duration::from_secs(args.server_timeout_s),
            )
            .await?,
        )
    };

    let body = run_iteration_body(client, base_url, args, iteration_index, iteration).await;

    if let Some(c) = child {
        if let Err(e) = process::stop_server(c).await {
            eprintln!("warning: stop_server: {e:#}");
        }
        // 2s settle time, mirror the Python `time.sleep(2)` after kill.
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    body
}

async fn run_iteration_body(
    client: &reqwest::Client,
    base_url: &str,
    args: &Args,
    iteration_index: usize,
    iteration: &Iteration,
) -> Result<IterationResult> {
    let model = http::first_model_id(client, base_url).await?;
    eprintln!("Model: {model}");

    eprintln!("Warmup...");
    let warmup_msgs = serde_json::json!([
        {"role": "user", "content": "Say ready."},
    ]);
    let _ = stream(client, base_url, &model, &warmup_msgs, 4, None).await?;

    let sweep = prompt_sweep(client, base_url, &model, args.repeats).await?;
    let qa = run_qa(client, base_url, &model).await?;
    let long_ctx = run_long_context(client, base_url, &model).await?;
    let structured = run_structured(client, base_url, &model, iteration_index).await?;
    let prefix_cache = run_prefix_cache(client, base_url, &model).await?;

    eprintln!(
        "Weighted prompt metrics: TTFT={:.0} ms  decode={:.1} tok/s",
        sweep.weighted_ttft_ms, sweep.weighted_decode_tps
    );
    eprintln!(
        "Accuracy checks: qa={}/{}  needle={}  json={}  cache={}",
        qa.passed,
        qa.total,
        if long_ctx.passed { "pass" } else { "fail" },
        if structured.passed { "pass" } else { "fail" },
        if prefix_cache.passed { "pass" } else { "fail" },
    );
    eprintln!(
        "Prefix cache: cold={:.0} ms  warm={:.0} ms  speedup={:.2}x",
        prefix_cache.cold_ttft_ms, prefix_cache.warm_ttft_ms, prefix_cache.speedup
    );

    Ok(IterationResult {
        iteration: iteration.slug.to_owned(),
        label: iteration.label.to_owned(),
        notes: iteration.notes.to_owned(),
        args: iteration.args.iter().map(|s| (*s).to_owned()).collect(),
        prompt_sweep: sweep,
        qa,
        long_context: long_ctx,
        structured_output: structured,
        prefix_cache,
        score: None,
    })
}

async fn stream(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    messages: &serde_json::Value,
    max_tokens: u32,
    response_format: Option<serde_json::Value>,
) -> Result<StreamMetrics> {
    let opts = StreamChatOptions {
        max_tokens,
        temperature: 0.0,
        api_key: None,
        response_format,
        estimate_prompt_tokens: false,
        // bench_mlx_tuning only targets a local Higgs server.
        include_usage: true,
    };
    http::stream_chat_metrics(client, base_url, model, messages, &opts).await
}

async fn prompt_sweep(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    repeats: u32,
) -> Result<PromptSweep> {
    let long = long_prompt();
    let prompts: [(&str, &str); 3] = [
        ("short", SHORT_PROMPT),
        ("medium", MEDIUM_PROMPT),
        ("long", &long),
    ];
    let weights: [(&str, f64); 3] = [("short", 0.2), ("medium", 0.3), ("long", 0.5)];
    let mut metrics: Vec<(&str, PromptMetric)> = Vec::new();

    for (label, prompt) in prompts {
        let mut runs: Vec<StreamMetrics> = Vec::new();
        for attempt in 0..repeats {
            if attempt == 0 {
                let warmup = serde_json::json!([
                    {"role": "user", "content": format!("[warmup {label}] {prompt}")},
                ]);
                let _ = stream(client, base_url, model, &warmup, 8, None).await?;
            }
            let msgs = serde_json::json!([{"role": "user", "content": prompt}]);
            let r = stream(client, base_url, model, &msgs, 64, None).await?;
            runs.push(r);
        }
        let m = PromptMetric {
            ttft_ms: median(&runs.iter().map(|r| r.ttft_ms).collect::<Vec<_>>()),
            decode_tps: median(&runs.iter().map(|r| r.decode_tps).collect::<Vec<_>>()),
            prompt_tokens: median(
                &runs
                    .iter()
                    .map(|r| f64::from(r.prompt_tokens))
                    .collect::<Vec<_>>(),
            ),
            completion_tokens: median(
                &runs
                    .iter()
                    .map(|r| f64::from(r.completion_tokens))
                    .collect::<Vec<_>>(),
            ),
        };
        metrics.push((label, m));
    }

    let mut weighted_ttft = 0.0;
    let mut weighted_decode = 0.0;
    for (label, weight) in weights {
        let m = metrics
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, m)| m)
            .expect("prompt label present in metrics");
        weighted_ttft += m.ttft_ms * weight;
        weighted_decode += m.decode_tps * weight;
    }

    let pick = |name: &str| -> PromptMetric {
        metrics
            .iter()
            .find(|(l, _)| *l == name)
            .map(|(_, m)| m.clone())
            .expect("prompt label present")
    };

    Ok(PromptSweep {
        short: pick("short"),
        medium: pick("medium"),
        long: pick("long"),
        weighted_ttft_ms: weighted_ttft,
        weighted_decode_tps: weighted_decode,
    })
}

async fn run_qa(client: &reqwest::Client, base_url: &str, model: &str) -> Result<QaResults> {
    let mut cases = Vec::new();
    let mut passed = 0_u32;
    for (prompt, expected) in QA_CASES {
        let msgs = serde_json::json!([{"role": "user", "content": prompt}]);
        let resp = http::chat_with_options(client, base_url, model, &msgs, 16, 0.0, None).await?;
        let normalized = normalize_text(&resp.content);
        let success = normalized == *expected;
        if success {
            passed += 1;
        }
        cases.push(QaCaseResult {
            prompt: (*prompt).to_owned(),
            expected: (*expected).to_owned(),
            output: normalized,
            passed: success,
        });
    }
    let total = QA_CASES.len() as u32;
    Ok(QaResults {
        passed,
        total,
        accuracy: f64::from(passed) / f64::from(total),
        cases,
    })
}

async fn run_long_context(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
) -> Result<LongContextResult> {
    let filler = long_context_filler();
    let prompt = format!(
        "Read the following deployment notes carefully.\n\n{filler}\n\nImportant hidden code: {LONG_CONTEXT_NEEDLE}\n\n{filler}\n\nQuestion: reply with the deployment code only."
    );
    let msgs = serde_json::json!([{"role": "user", "content": prompt}]);
    let resp = http::chat_with_options(client, base_url, model, &msgs, 8, 0.0, None).await?;
    let output = normalize_text(&resp.content).replace(' ', "");
    let expected = LONG_CONTEXT_NEEDLE.to_lowercase();
    let passed = output.contains(&expected);
    Ok(LongContextResult {
        expected,
        output,
        passed,
        accuracy: if passed { 1.0 } else { 0.0 },
        prompt_tokens: resp.prompt_tokens,
    })
}

async fn run_structured(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    iteration_index: usize,
) -> Result<StructuredResult> {
    let prompt = format!(
        "Return structured JSON only. Facts: model_family=qwen, iteration={iteration_index}, prefill_focus=true, kv_bits=3, primary_goal=latency."
    );
    let schema = structured_schema();
    let msgs = serde_json::json!([{"role": "user", "content": prompt}]);
    let resp =
        http::chat_with_options(client, base_url, model, &msgs, 64, 0.0, Some(&schema)).await?;
    let parsed: serde_json::Value = match serde_json::from_str(&resp.content) {
        Ok(v) => v,
        Err(_) => {
            return Ok(StructuredResult {
                passed: false,
                accuracy: 0.0,
                raw: Some(resp.content),
                parsed: None,
            });
        }
    };
    let expected = serde_json::json!({
        "model_family": "qwen",
        "iteration": iteration_index,
        "prefill_focus": true,
        "kv_bits": 3,
        "primary_goal": "latency",
    });
    let passed = parsed == expected;
    Ok(StructuredResult {
        passed,
        accuracy: if passed { 1.0 } else { 0.0 },
        raw: None,
        parsed: Some(parsed),
    })
}

async fn run_prefix_cache(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
) -> Result<PrefixCacheResult> {
    let doc = prefix_cache_doc();
    let prefix = serde_json::json!([
        {"role": "system", "content": format!(
            "You are reviewing an operations handbook. Study the material and reply READY only.\n\n{doc}",
        )},
        {"role": "user", "content": "Read the handbook and reply READY only."},
    ]);
    let followup = serde_json::json!([
        prefix[0],
        prefix[1],
        {"role": "assistant", "content": "READY"},
        {"role": "user", "content": "What is the failover region? Reply with two words only."},
    ]);

    let cold = stream(client, base_url, model, &followup, 8, None).await?;
    let _ = stream(client, base_url, model, &prefix, 4, None).await?;
    let warm = stream(client, base_url, model, &followup, 8, None).await?;

    let warm_output = normalize_text(&warm.output);
    let passed = warm_output.contains("cape town");
    let speedup = if warm.ttft_ms > 0.0 {
        cold.ttft_ms / warm.ttft_ms
    } else {
        0.0
    };
    Ok(PrefixCacheResult {
        cold_ttft_ms: cold.ttft_ms,
        warm_ttft_ms: warm.ttft_ms,
        speedup,
        answer: warm.output,
        passed,
        accuracy: if passed { 1.0 } else { 0.0 },
    })
}

fn score_results(results: &mut [IterationResult]) {
    let speeds: Vec<SpeedInputs> = results
        .iter()
        .map(|r| SpeedInputs {
            weighted_ttft_ms: r.prompt_sweep.weighted_ttft_ms,
            weighted_decode_tps: r.prompt_sweep.weighted_decode_tps,
        })
        .collect();
    let caches: Vec<CacheInputs> = results
        .iter()
        .map(|r| CacheInputs {
            passed: r.prefix_cache.passed,
            speedup: r.prefix_cache.speedup,
        })
        .collect();
    let (best_ttft, best_decode, best_cache) = compute_bests(&speeds, &caches);

    for r in results.iter_mut() {
        let acc = AccuracyInputs {
            qa: r.qa.accuracy,
            long_context: r.long_context.accuracy,
            structured_output: r.structured_output.accuracy,
            prefix_cache: r.prefix_cache.accuracy,
        };
        let speed = SpeedInputs {
            weighted_ttft_ms: r.prompt_sweep.weighted_ttft_ms,
            weighted_decode_tps: r.prompt_sweep.weighted_decode_tps,
        };
        let cache = CacheInputs {
            passed: r.prefix_cache.passed,
            speedup: r.prefix_cache.speedup,
        };
        let s = compute_iteration_score(acc, speed, cache, best_ttft, best_decode, best_cache);
        r.score = Some(ScoreOut {
            accuracy: s.accuracy,
            speed: s.speed,
            cache: s.cache,
            composite: s.composite,
        });
    }
}

fn print_summary(results: &[IterationResult], winner: Option<&str>) {
    let mut ordered: Vec<&IterationResult> = results.iter().collect();
    ordered.sort_by(|a, b| {
        let aa = a.score.as_ref().map_or(0.0, |s| s.composite);
        let bb = b.score.as_ref().map_or(0.0, |s| s.composite);
        bb.partial_cmp(&aa).unwrap_or(std::cmp::Ordering::Equal)
    });

    eprintln!("\n{}", "#".repeat(80));
    eprintln!("FINAL SUMMARY");
    eprintln!("{}", "#".repeat(80));
    eprintln!(
        "{:32} {:>8} {:>10} {:>10} {:>6} {:>8} {:>6} {:>8}",
        "Iteration", "Score", "TTFT", "Decode", "QA", "Needle", "JSON", "Cache"
    );
    eprintln!("{}", "-".repeat(96));
    for r in &ordered {
        let label = if r.label.len() > 32 {
            &r.label[..32]
        } else {
            &r.label
        };
        let composite = r.score.as_ref().map_or(0.0, |s| s.composite);
        eprintln!(
            "{:32} {:>7.1} {:>9.0} {:>9.1} {:>2}/{:<3} {:>8} {:>6} {:>7.2}x",
            label,
            composite,
            r.prompt_sweep.weighted_ttft_ms,
            r.prompt_sweep.weighted_decode_tps,
            r.qa.passed,
            r.qa.total,
            if r.long_context.passed {
                "pass"
            } else {
                "fail"
            },
            if r.structured_output.passed {
                "pass"
            } else {
                "fail"
            },
            r.prefix_cache.speedup,
        );
    }
    if let Some(w) = winner {
        eprintln!("\nWinner: {w}");
    }
}

#[cfg(test)]
mod tests {
    //! Ports of `benchmarks/test_bench_mlx_tuning.py`.
    use super::*;
    use higgs_bench::score::{
        AccuracyInputs, CacheInputs, SpeedInputs, clamp_cache_speedup, compute_iteration_score,
        normalize_text,
    };

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "lhs={a} rhs={b}");
    }

    #[test]
    fn test_normalize_text() {
        let normalized = normalize_text("  Leading\tand\ntrailing  MIXED  Case ");
        assert_eq!(normalized, "leading and trailing mixed case");
    }

    #[test]
    fn test_cache_speedup_is_capped() {
        approx(clamp_cache_speedup(96.0), 32.0);
        approx(clamp_cache_speedup(12.0), 12.0);
    }

    #[test]
    fn test_compute_iteration_score_applies_formula() {
        let acc = AccuracyInputs {
            qa: 1.0,
            long_context: 0.5,
            structured_output: 0.0,
            prefix_cache: 1.0,
        };
        let speed = SpeedInputs {
            weighted_ttft_ms: 100.0,
            weighted_decode_tps: 100.0,
        };
        let cache = CacheInputs {
            passed: true,
            speedup: 96.0,
        };
        let s = compute_iteration_score(acc, speed, cache, 100.0, 50.0, 16.0);

        let expected_accuracy = (1.0_f64 * 0.45) + (0.5 * 0.25) + (0.0 * 0.15) + (1.0 * 0.15);
        let expected_speed = (100.0_f64 / 100.0) * 0.55 + (100.0 / 50.0) * 0.45;
        let expected_cache = 32.0_f64 / 16.0;
        let expected_composite = 100.0
            * ((expected_accuracy * 0.45) + (expected_speed * 0.45) + (expected_cache * 0.10));

        approx(s.accuracy, expected_accuracy);
        approx(s.speed, expected_speed);
        approx(s.cache, expected_cache);
        approx(s.composite, expected_composite);
    }

    fn dummy_iteration_result(
        weighted_ttft_ms: f64,
        weighted_decode_tps: f64,
        qa_accuracy: f64,
        long_accuracy: f64,
        structured_accuracy: f64,
        cache_passed: bool,
        cache_speedup: f64,
    ) -> IterationResult {
        IterationResult {
            iteration: "dummy".into(),
            label: "dummy".into(),
            notes: String::new(),
            args: vec![],
            prompt_sweep: PromptSweep {
                short: PromptMetric {
                    ttft_ms: 0.0,
                    decode_tps: 0.0,
                    prompt_tokens: 0.0,
                    completion_tokens: 0.0,
                },
                medium: PromptMetric {
                    ttft_ms: 0.0,
                    decode_tps: 0.0,
                    prompt_tokens: 0.0,
                    completion_tokens: 0.0,
                },
                long: PromptMetric {
                    ttft_ms: 0.0,
                    decode_tps: 0.0,
                    prompt_tokens: 0.0,
                    completion_tokens: 0.0,
                },
                weighted_ttft_ms,
                weighted_decode_tps,
            },
            qa: QaResults {
                passed: 0,
                total: 5,
                accuracy: qa_accuracy,
                cases: vec![],
            },
            long_context: LongContextResult {
                expected: String::new(),
                output: String::new(),
                passed: long_accuracy >= 1.0,
                accuracy: long_accuracy,
                prompt_tokens: 0,
            },
            structured_output: StructuredResult {
                passed: structured_accuracy >= 1.0,
                accuracy: structured_accuracy,
                raw: None,
                parsed: None,
            },
            prefix_cache: PrefixCacheResult {
                cold_ttft_ms: 0.0,
                warm_ttft_ms: 0.0,
                speedup: cache_speedup,
                answer: String::new(),
                passed: cache_passed,
                accuracy: if cache_passed { 1.0 } else { 0.0 },
            },
            score: None,
        }
    }

    #[test]
    fn test_score_results_marks_all_results() {
        let mut results = vec![
            dummy_iteration_result(200.0, 80.0, 0.4, 0.0, 1.0, false, 12.0),
            dummy_iteration_result(100.0, 160.0, 0.8, 1.0, 1.0, true, 96.0),
        ];
        score_results(&mut results);
        assert!(results[0].score.is_some());
        assert!(results[1].score.is_some());
        assert!(
            (results[0].score.as_ref().unwrap().composite
                - results[1].score.as_ref().unwrap().composite)
                .abs()
                > 1e-9
        );
    }

    #[test]
    fn test_rank_results_by_score() {
        let mut results = vec![
            {
                let mut r = dummy_iteration_result(0.0, 0.0, 0.0, 0.0, 0.0, false, 0.0);
                r.score = Some(ScoreOut {
                    accuracy: 0.0,
                    speed: 0.0,
                    cache: 0.0,
                    composite: 33.0,
                });
                r
            },
            {
                let mut r = dummy_iteration_result(0.0, 0.0, 0.0, 0.0, 0.0, false, 0.0);
                r.score = Some(ScoreOut {
                    accuracy: 0.0,
                    speed: 0.0,
                    cache: 0.0,
                    composite: 72.0,
                });
                r
            },
            {
                let mut r = dummy_iteration_result(0.0, 0.0, 0.0, 0.0, 0.0, false, 0.0);
                r.score = Some(ScoreOut {
                    accuracy: 0.0,
                    speed: 0.0,
                    cache: 0.0,
                    composite: 51.0,
                });
                r
            },
        ];
        results.sort_by(|a, b| {
            b.score
                .as_ref()
                .unwrap()
                .composite
                .partial_cmp(&a.score.as_ref().unwrap().composite)
                .unwrap()
        });
        approx(results[0].score.as_ref().unwrap().composite, 72.0);
    }
}
