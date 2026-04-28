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
    clippy::shadow_same
)]
//! `bench_ttft_quick` — quick TTFT measurement at varying prompt sizes.
//!
//! Port of `benchmarks/bench_ttft_quick.py`. Sends a non-streaming
//! `max_tokens=1` chat completion at four prompt sizes (short, medium,
//! long, very_long), uses a unique prefix per request to defeat the
//! prefix cache, and reports the median wall time per size.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use higgs_bench::{
    BenchOutput, ModelInfo, OutputFormat, RunMetadata, default_manifest_path, format_json,
    format_markdown, http, models, persist_result, server, stats,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "bench_ttft_quick",
    about = "Quick TTFT benchmark at four prompt sizes against a running higgs server",
    version
)]
struct Args {
    #[arg(long, default_value_t = 9999)]
    port: u16,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Model key from `benchmarks/models.toml`.
    #[arg(long)]
    model: String,

    #[arg(long)]
    manifest: Option<std::path::PathBuf>,

    #[arg(long, default_value_t = 1)]
    warmup: u32,

    #[arg(long, default_value_t = 3)]
    iters: u32,

    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,

    #[arg(long)]
    no_wait: bool,
}

#[derive(Debug, Serialize)]
struct Params {
    host: String,
    port: u16,
    model_key: String,
    model_path: String,
    warmup: u32,
    iters: u32,
}

#[derive(Debug, Serialize, Clone)]
struct SizeResult {
    label: String,
    prompt_tokens: Option<u32>,
    median_ttft_ms: f64,
    times_ms: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct Results {
    sizes: Vec<SizeResult>,
}

const WORD: &str = "the quick brown fox jumps over the lazy dog ";

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
    let mut metadata = RunMetadata::capture("bench_ttft_quick");
    let started = Instant::now();

    let manifest_path = args.manifest.clone().unwrap_or_else(default_manifest_path);
    let model = models::find_by_key(&manifest_path, &args.model)?;
    metadata.model = Some(ModelInfo {
        key: model.key.clone(),
        path: model.path.clone(),
        quantization: model.quantization.clone(),
        approx_size_gb: model.approx_size_gb,
    });

    let base_url = format!("http://{}:{}", args.host, args.port);
    if !args.no_wait {
        server::wait_until_ready(&base_url, Duration::from_secs(30))
            .await
            .with_context(|| format!("higgs server not reachable at {base_url}"))?;
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;

    let sizes_def: [(&str, usize); 4] = [
        ("short", 3),
        ("medium", 50),
        ("long", 200),
        ("very_long", 500),
    ];

    let mut size_results = Vec::new();
    for (label, repeat) in sizes_def {
        let prompt = WORD.repeat(repeat);

        for i in 0..args.warmup {
            let p = format!("[warmup {i}] {prompt}");
            let _ = ttft_call(&client, &base_url, &model.path, &p).await?;
        }

        let mut times = Vec::with_capacity(args.iters as usize);
        let mut ptoks: Option<u32> = None;
        for i in 0..args.iters {
            let stamp = chrono::Utc::now().timestamp_micros();
            let p = format!("[run {i} {stamp}] {prompt}");
            let (elapsed_ms, pt) = ttft_call(&client, &base_url, &model.path, &p).await?;
            times.push(elapsed_ms);
            ptoks = pt.or(ptoks);
        }
        let median = stats::median(&times);
        eprintln!("[{label:>10}] tokens={ptoks:?} median_ttft_ms={median:.1}");
        size_results.push(SizeResult {
            label: label.to_owned(),
            prompt_tokens: ptoks,
            median_ttft_ms: median,
            times_ms: times,
        });
    }

    metadata.duration_ms = started.elapsed().as_millis() as u64;
    let params = Params {
        host: args.host.clone(),
        port: args.port,
        model_key: model.key.clone(),
        model_path: model.path.clone(),
        warmup: args.warmup,
        iters: args.iters,
    };
    let results = Results {
        sizes: size_results,
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

async fn ttft_call(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    prompt: &str,
) -> Result<(f64, Option<u32>)> {
    let messages = serde_json::json!([{"role": "user", "content": prompt}]);
    let res = http::chat(client, base_url, model, &messages, 1, 0.0).await?;
    Ok((res.elapsed_ms, Some(res.prompt_tokens)))
}
