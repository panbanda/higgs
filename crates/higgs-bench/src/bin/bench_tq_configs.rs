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
    clippy::too_many_lines
)]
//! `bench_tq_configs` — sweep TurboQuant KV-cache configs.
//!
//! Port of `benchmarks/bench_tq_configs.py`. For each KV-cache config:
//! launch a higgs server, measure prefill TTFT + decode tok/s at three
//! context sizes, then generate 10 short answers for a quality (Jaccard)
//! comparison against the baseline config.

use std::collections::HashSet;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use higgs_bench::{
    BenchOutput, ModelInfo, OutputFormat, RunMetadata, default_manifest_path, format_json,
    format_markdown, http, models, persist_result, process,
};
use serde::Serialize;

const WORD: &str = "the quick brown fox jumps over the lazy dog ";

const QUALITY_PROMPTS: [&str; 10] = [
    "Explain how a hash table works step by step.",
    "What causes the seasons on Earth?",
    "Write a Python function to find the longest common subsequence.",
    "Describe the process of photosynthesis in detail.",
    "What is the difference between TCP and UDP?",
    "Explain the theory of general relativity in simple terms.",
    "How does a neural network learn through backpropagation?",
    "What are the main causes of climate change?",
    "Describe the water cycle and its importance.",
    "How does public key cryptography work?",
];

#[derive(Debug, Parser)]
#[command(
    name = "bench_tq_configs",
    about = "Sweep TurboQuant KV-cache configs and report decode tok/s + Jaccard quality",
    version
)]
struct Args {
    #[arg(long)]
    model: String,

    #[arg(long)]
    manifest: Option<std::path::PathBuf>,

    #[arg(long, default_value_t = 8097)]
    port: u16,

    #[arg(long, default_value_t = 64)]
    gen_tokens: u32,

    #[arg(long, default_value_t = 120)]
    server_timeout_s: u64,

    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
}

#[derive(Debug, Serialize)]
struct Params {
    port: u16,
    model_key: String,
    model_path: String,
    gen_tokens: u32,
    context_sizes: Vec<u32>,
    configs: Vec<ConfigSpec>,
}

#[derive(Debug, Serialize, Clone)]
struct ConfigSpec {
    label: String,
    args: Vec<String>,
}

#[derive(Debug, Serialize, Clone)]
struct SweepPoint {
    prompt_tokens: u32,
    completion_tokens: u32,
    ttft_ms: f64,
    decode_tokps: f64,
    total_ms: f64,
    error: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct ConfigResult {
    label: String,
    args: Vec<String>,
    sweep: Vec<SweepPoint>,
    outputs: Vec<String>,
    quality_jaccard_avg: Option<f64>,
    quality_jaccard_min: Option<f64>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct Results {
    configs: Vec<ConfigResult>,
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

fn build_configs() -> Vec<ConfigSpec> {
    let to_owned = |xs: &[&str]| xs.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    vec![
        ConfigSpec {
            label: "baseline (no TQ)".into(),
            args: vec![],
        },
        ConfigSpec {
            label: "TQ default (bits=3, norm ON)".into(),
            args: to_owned(&["--kv-cache", "turboquant", "--kv-bits", "3"]),
        },
        ConfigSpec {
            label: "TQ no-norm-correction".into(),
            args: to_owned(&[
                "--kv-cache",
                "turboquant",
                "--kv-bits",
                "3",
                "--kv-no-norm-correction",
            ]),
        },
        ConfigSpec {
            label: "TQ asymmetric (K=4, V=3)".into(),
            args: to_owned(&[
                "--kv-cache",
                "turboquant",
                "--kv-bits",
                "3",
                "--kv-key-bits",
                "4",
                "--kv-value-bits",
                "3",
            ]),
        },
        ConfigSpec {
            label: "TQ layer-adaptive (8 dense)".into(),
            args: to_owned(&[
                "--kv-cache",
                "turboquant",
                "--kv-bits",
                "3",
                "--kv-adaptive-dense-layers",
                "8",
            ]),
        },
    ]
}

async fn run(args: Args) -> Result<()> {
    let mut metadata = RunMetadata::capture("bench_tq_configs");
    let started = Instant::now();

    let manifest_path = args.manifest.clone().unwrap_or_else(default_manifest_path);
    let model = models::find_by_key(&manifest_path, &args.model)?;
    metadata.model = Some(ModelInfo {
        key: model.key.clone(),
        path: model.path.clone(),
        quantization: model.quantization.clone(),
        approx_size_gb: model.approx_size_gb,
    });

    let configs = build_configs();
    let context_sizes: Vec<u32> = vec![100, 1000, 4000];

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let base_url = format!("http://127.0.0.1:{}", args.port);

    let mut config_results: Vec<ConfigResult> = Vec::new();
    for cfg in &configs {
        eprintln!("=== config: {} ===", cfg.label);
        let cr = match run_one_config(&client, &base_url, &model, cfg, &context_sizes, &args).await
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!("config '{}' failed: {e:#}", cfg.label);
                ConfigResult {
                    label: cfg.label.clone(),
                    args: cfg.args.clone(),
                    sweep: Vec::new(),
                    outputs: Vec::new(),
                    quality_jaccard_avg: None,
                    quality_jaccard_min: None,
                    error: Some(format!("{e:#}")),
                }
            }
        };
        config_results.push(cr);
    }

    // Compare quality vs baseline (first config).
    if let Some(baseline_outputs) = config_results
        .first()
        .map(|c| c.outputs.clone())
        .filter(|v| !v.is_empty())
    {
        for cr in config_results.iter_mut().skip(1) {
            if cr.outputs.len() == baseline_outputs.len() && !cr.outputs.is_empty() {
                let scores: Vec<f64> = cr
                    .outputs
                    .iter()
                    .zip(baseline_outputs.iter())
                    .map(|(a, b)| jaccard(a, b))
                    .collect();
                let avg = scores.iter().sum::<f64>() / scores.len() as f64;
                let min = scores.iter().copied().fold(1.0_f64, f64::min);
                cr.quality_jaccard_avg = Some(avg);
                cr.quality_jaccard_min = Some(min);
            }
        }
    }

    metadata.duration_ms = started.elapsed().as_millis() as u64;
    let params = Params {
        port: args.port,
        model_key: model.key.clone(),
        model_path: model.path.clone(),
        gen_tokens: args.gen_tokens,
        context_sizes,
        configs,
    };
    let results = Results {
        configs: config_results,
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

async fn run_one_config(
    client: &reqwest::Client,
    base_url: &str,
    model: &models::Model,
    cfg: &ConfigSpec,
    context_sizes: &[u32],
    args: &Args,
) -> Result<ConfigResult> {
    let child = process::start_higgs_server(
        &model.path,
        args.port,
        &cfg.args,
        Duration::from_secs(args.server_timeout_s),
    )
    .await?;

    let inner = run_one_config_inner(client, base_url, cfg, context_sizes, args).await;

    if let Err(e) = process::stop_server(child).await {
        eprintln!("warning: stop_server: {e:#}");
    }
    // small grace period before next server starts
    tokio::time::sleep(Duration::from_secs(2)).await;
    inner
}

async fn run_one_config_inner(
    client: &reqwest::Client,
    base_url: &str,
    cfg: &ConfigSpec,
    context_sizes: &[u32],
    args: &Args,
) -> Result<ConfigResult> {
    let model_id = http::first_model_id(client, base_url).await?;
    eprintln!("  server ready: {model_id}");

    // Warmup
    let warmup = serde_json::json!([{"role": "user", "content": "hi"}]);
    let _ = http::chat(client, base_url, &model_id, &warmup, 2, 0.0).await?;

    let mut sweep = Vec::new();
    for &ctx in context_sizes {
        match measure_decode(client, base_url, &model_id, ctx, args.gen_tokens).await {
            Ok(p) => {
                eprintln!(
                    "  ctx≈{} ptoks={} ttft={:.0}ms tps={:.1}",
                    ctx, p.prompt_tokens, p.ttft_ms, p.decode_tokps
                );
                sweep.push(p);
            }
            Err(e) => {
                eprintln!("  ctx≈{ctx} FAILED: {e:#}");
                sweep.push(SweepPoint {
                    prompt_tokens: ctx,
                    completion_tokens: 0,
                    ttft_ms: 0.0,
                    decode_tokps: 0.0,
                    total_ms: 0.0,
                    error: Some(format!("{e:#}")),
                });
            }
        }
    }

    let mut outputs = Vec::new();
    for prompt in QUALITY_PROMPTS {
        let messages = serde_json::json!([{"role": "user", "content": prompt}]);
        let r = http::chat(client, base_url, &model_id, &messages, 64, 0.0).await?;
        outputs.push(r.content);
    }
    eprintln!(
        "  generated {} quality outputs (avg {} chars)",
        outputs.len(),
        if outputs.is_empty() {
            0
        } else {
            outputs.iter().map(String::len).sum::<usize>() / outputs.len()
        }
    );

    Ok(ConfigResult {
        label: cfg.label.clone(),
        args: cfg.args.clone(),
        sweep,
        outputs,
        quality_jaccard_avg: None,
        quality_jaccard_min: None,
        error: None,
    })
}

async fn measure_decode(
    client: &reqwest::Client,
    base_url: &str,
    model_id: &str,
    target_ctx_tokens: u32,
    gen_tokens: u32,
) -> Result<SweepPoint> {
    let repeats = std::cmp::max(1, (target_ctx_tokens / 10) as usize);
    let prompt = WORD.repeat(repeats);

    // TTFT measurement: max_tokens=1
    let pfx = format!("[ttft{}] ", chrono::Utc::now().timestamp_micros());
    let ttft_messages = serde_json::json!([
        {"role": "user", "content": format!("{pfx}{prompt}")},
    ]);
    let ttft = http::chat(client, base_url, model_id, &ttft_messages, 1, 0.0)
        .await
        .context("ttft probe")?;

    // Full generation
    let pfx2 = format!("[dec{}] Summarize: ", chrono::Utc::now().timestamp_micros());
    let full_messages = serde_json::json!([
        {"role": "user", "content": format!("{pfx2}{prompt}")},
    ]);
    let full = http::chat(client, base_url, model_id, &full_messages, gen_tokens, 0.0)
        .await
        .context("decode probe")?;

    let decode_tokps = if full.completion_tokens > 1 {
        let decode_secs = ((full.elapsed_ms - ttft.elapsed_ms).max(10.0)) / 1000.0;
        f64::from(full.completion_tokens.saturating_sub(1)) / decode_secs
    } else {
        0.0
    };

    Ok(SweepPoint {
        prompt_tokens: ttft.prompt_tokens,
        completion_tokens: full.completion_tokens,
        ttft_ms: ttft.elapsed_ms,
        decode_tokps,
        total_ms: full.elapsed_ms,
        error: None,
    })
}

fn jaccard(a: &str, b: &str) -> f64 {
    let wa: HashSet<String> = a
        .to_lowercase()
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let wb: HashSet<String> = b
        .to_lowercase()
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    if wa.is_empty() && wb.is_empty() {
        return 0.0;
    }
    let inter = wa.intersection(&wb).count();
    let union = wa.union(&wb).count();
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}
