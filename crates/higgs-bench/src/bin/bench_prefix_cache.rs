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
//! `bench_prefix_cache` — TTFT on cache miss vs cache hit.
//!
//! Port of `benchmarks/bench_prefix_cache.py`. For each model, spawns a
//! higgs server, sends a long shared system prompt with three different
//! short user prompts, and reports the speedup from prefix-cache hits on
//! the second and third request.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use higgs_bench::{
    BenchOutput, ModelInfo, OutputFormat, RunMetadata, default_manifest_path, format_json,
    format_markdown, http, models, persist_result, process, server,
};
use serde::Serialize;

const SYSTEM_PROMPT: &str = include_str!("../../assets/prefix_cache_system_prompt.txt");

const USER_PROMPTS: [&str; 3] = [
    "Explain the CAP theorem in one paragraph.",
    "What is eventual consistency? Keep it brief.",
    "Compare Redis and Memcached in three sentences.",
];

#[derive(Debug, Parser)]
#[command(
    name = "bench_prefix_cache",
    about = "Measure prefix-cache TTFT speedup (miss vs hit) for one or more models",
    version
)]
struct Args {
    /// Comma-separated list of model keys to benchmark. Defaults to all
    /// models tagged `large` in the manifest.
    #[arg(long)]
    models: Option<String>,

    /// Filter by tag instead of explicit keys.
    #[arg(long)]
    tag: Option<String>,

    #[arg(long)]
    manifest: Option<std::path::PathBuf>,

    /// Port to bind the spawned higgs server on (or to use as the
    /// external server when `--no-spawn` is set).
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Skip spawning a server; assume one is already running on
    /// `--port`. Implies a single model is being benchmarked.
    #[arg(long)]
    no_spawn: bool,

    /// Max tokens to generate per request.
    #[arg(long, default_value_t = 80)]
    max_tokens: u32,

    /// Server startup timeout in seconds.
    #[arg(long, default_value_t = 120)]
    server_timeout_s: u64,

    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
}

#[derive(Debug, Serialize)]
struct Params {
    port: u16,
    max_tokens: u32,
    no_spawn: bool,
    user_prompts: Vec<String>,
    system_prompt_words: usize,
    model_keys: Vec<String>,
}

#[derive(Debug, Serialize, Clone)]
struct PerRequest {
    tag: String,
    user: String,
    ttft_ms: f64,
    total_ms: f64,
    num_tokens: u32,
}

#[derive(Debug, Serialize, Clone)]
struct ModelSummary {
    model: String,
    miss_ttft_ms: f64,
    avg_hit_ttft_ms: f64,
    speedup: f64,
    requests: Vec<PerRequest>,
}

#[derive(Debug, Serialize)]
struct Results {
    per_model: Vec<ModelSummary>,
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
    let mut metadata = RunMetadata::capture("bench_prefix_cache");
    let started = Instant::now();

    let manifest_path = args.manifest.clone().unwrap_or_else(default_manifest_path);
    let manifest = models::load_manifest(&manifest_path)?;

    let selected: Vec<models::Model> = if let Some(list) = &args.models {
        list.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|key| {
                manifest
                    .find_by_key(key)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("model key '{key}' not in manifest"))
            })
            .collect::<Result<Vec<_>>>()?
    } else if let Some(tag) = &args.tag {
        manifest.find_by_tag(tag).into_iter().cloned().collect()
    } else {
        manifest.find_by_tag("large").into_iter().cloned().collect()
    };
    if selected.is_empty() {
        anyhow::bail!("no models selected; pass --models or --tag");
    }
    if args.no_spawn && selected.len() != 1 {
        anyhow::bail!("--no-spawn requires exactly one model");
    }

    if let Some(first) = selected.first() {
        metadata.model = Some(ModelInfo {
            key: first.key.clone(),
            path: first.path.clone(),
            quantization: first.quantization.clone(),
            approx_size_gb: first.approx_size_gb,
        });
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let base_url = format!("http://127.0.0.1:{}", args.port);

    let mut per_model = Vec::new();
    for model in &selected {
        let summary = bench_one_model(&client, &base_url, model, &args)
            .await
            .with_context(|| format!("benchmarking model '{}'", model.key))?;
        per_model.push(summary);
    }

    metadata.duration_ms = started.elapsed().as_millis() as u64;
    let params = Params {
        port: args.port,
        max_tokens: args.max_tokens,
        no_spawn: args.no_spawn,
        user_prompts: USER_PROMPTS.iter().map(|s| (*s).to_owned()).collect(),
        system_prompt_words: SYSTEM_PROMPT.split_whitespace().count(),
        model_keys: selected.iter().map(|m| m.key.clone()).collect(),
    };
    let results = Results { per_model };
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

async fn bench_one_model(
    client: &reqwest::Client,
    base_url: &str,
    model: &models::Model,
    args: &Args,
) -> Result<ModelSummary> {
    eprintln!("=== model: {} ({}) ===", model.label, model.path);

    let child = if args.no_spawn {
        None
    } else {
        Some(
            process::start_higgs_server(
                &model.path,
                args.port,
                &[],
                Duration::from_secs(args.server_timeout_s),
            )
            .await?,
        )
    };

    let summary = run_requests(client, base_url, model, args).await;

    if let Some(c) = child {
        if let Err(e) = process::stop_server(c).await {
            eprintln!("warning: stop_server: {e:#}");
        }
    }

    summary
}

async fn run_requests(
    client: &reqwest::Client,
    base_url: &str,
    model: &models::Model,
    args: &Args,
) -> Result<ModelSummary> {
    if args.no_spawn {
        server::wait_until_ready(base_url, Duration::from_secs(30)).await?;
    }
    let model_id = http::first_model_id(client, base_url).await?;

    // Warmup
    let warmup = serde_json::json!([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Say hello."}
    ]);
    let _ = http::stream_chat(client, base_url, &model_id, &warmup, 10, 0.0).await?;

    let mut requests = Vec::new();
    for (i, user) in USER_PROMPTS.iter().enumerate() {
        let messages = serde_json::json!([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]);
        let r =
            http::stream_chat(client, base_url, &model_id, &messages, args.max_tokens, 0.0).await?;
        let tag = if i == 0 { "MISS" } else { "HIT" };
        eprintln!(
            "  [{tag}] ttft={:.1}ms total={:.1}ms toks={}",
            r.ttft_ms, r.total_ms, r.num_tokens
        );
        requests.push(PerRequest {
            tag: tag.to_owned(),
            user: (*user).to_owned(),
            ttft_ms: r.ttft_ms,
            total_ms: r.total_ms,
            num_tokens: r.num_tokens,
        });
    }

    let miss_ttft = requests
        .first()
        .map(|r| r.ttft_ms)
        .ok_or_else(|| anyhow::anyhow!("no requests recorded"))?;
    let hits: Vec<f64> = requests.iter().skip(1).map(|r| r.ttft_ms).collect();
    let avg_hit = if hits.is_empty() {
        0.0
    } else {
        hits.iter().sum::<f64>() / hits.len() as f64
    };
    let speedup = if avg_hit > 0.0 {
        miss_ttft / avg_hit
    } else {
        0.0
    };

    Ok(ModelSummary {
        model: model.label.clone(),
        miss_ttft_ms: miss_ttft,
        avg_hit_ttft_ms: avg_hit,
        speedup,
        requests,
    })
}
