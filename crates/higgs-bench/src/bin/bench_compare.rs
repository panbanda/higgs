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
//! `bench_compare` measures any OpenAI-compatible chat-completions endpoint
//! purely from the client side, so the identical measurement code can be
//! applied to higgs, oMLX, or any other server.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use higgs_bench::client::{Endpoint, EndpointSettings};
use higgs_bench::compare::{parse_extra_body, redact_secret_args, sanitize_model_key};
use higgs_bench::rss::RssSampler;
use higgs_bench::runner::{Plan, Results, Scenario, run_scenarios};
use higgs_bench::{
    BENCH_SCHEMA_VERSION, BenchOutput, ModelInfo, OutputFormat, RunMetadata, format_json,
    format_markdown, path_for_output, persist_result, public_model_ref,
};
use serde::Serialize;
use serde_json::{Map, Value};

const PREFIX_CACHE_CONTEXT_TOKENS: usize = 4096;
const CONCURRENCY_ROUNDS: usize = 2;

#[derive(Debug, Parser)]
#[command(
    name = "bench_compare",
    about = "Measure any OpenAI-compatible chat-completions server from the client side",
    version
)]
struct Args {
    /// Base URL of the server, for example `http://127.0.0.1:8899`.
    #[arg(long)]
    base_url: String,

    /// Model name sent as the request `model`.
    #[arg(long)]
    model: String,

    /// Server name recorded in output (higgs, omlx, ...).
    #[arg(long)]
    label: String,

    /// Public model identifier for metadata and the result filename.
    #[arg(long)]
    model_key: Option<String>,

    /// Bearer token sent with every request.
    #[arg(long)]
    api_key: Option<String>,

    /// JSON object merged into every request body; its keys override defaults.
    #[arg(long)]
    extra_body: Option<String>,

    /// Server process id; enables RSS sampling.
    #[arg(long)]
    pid: Option<u32>,

    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values_t = [Scenario::Decode, Scenario::Prefill, Scenario::PrefixCache, Scenario::Concurrency]
    )]
    scenarios: Vec<Scenario>,

    #[arg(long, default_value_t = 5)]
    trials: u32,

    #[arg(long, default_value_t = 1)]
    warmup: u32,

    #[arg(long, default_value_t = 256)]
    max_tokens: u32,

    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Approximate prompt sizes (tokens) for the prefill scenario.
    #[arg(long, value_delimiter = ',', default_values_t = [512, 2048, 8192])]
    prefill_tokens: Vec<usize>,

    /// Simultaneous request counts for the concurrency scenario.
    #[arg(long, value_delimiter = ',', default_values_t = [1, 2, 4, 8])]
    concurrency: Vec<usize>,

    #[arg(long, default_value_t = 900)]
    timeout_secs: u64,

    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
}

#[derive(Debug, Serialize)]
struct Params {
    label: String,
    base_url: String,
    request_model: String,
    model_key: String,
    api_key_set: bool,
    extra_body: Map<String, Value>,
    pid: Option<u32>,
    scenarios: Vec<Scenario>,
    trials: u32,
    warmup: u32,
    max_tokens: u32,
    temperature: f32,
    prefill_tokens: Vec<usize>,
    concurrency: Vec<usize>,
    timeout_secs: u64,
}

fn plan_from(args: &Args) -> Plan {
    Plan {
        scenarios: args.scenarios.clone(),
        trials: args.trials,
        warmup: args.warmup,
        max_tokens: args.max_tokens,
        prefill_tokens: args.prefill_tokens.clone(),
        prefix_cache_tokens: PREFIX_CACHE_CONTEXT_TOKENS,
        concurrency: args.concurrency.clone(),
        concurrency_rounds: CONCURRENCY_ROUNDS,
    }
}

async fn run(args: Args) -> Result<()> {
    let plan = plan_from(&args);
    plan.validate()?;

    let mut metadata = RunMetadata::capture("bench_compare");
    metadata.args = redact_secret_args(&metadata.args);
    let started = Instant::now();

    let model_key = args
        .model_key
        .clone()
        .unwrap_or_else(|| sanitize_model_key(&public_model_ref(&args.model, "")));
    metadata.model = Some(ModelInfo {
        key: format!("{}__{model_key}", sanitize_model_key(&args.label)),
        path: model_key.clone(),
        quantization: "n/a".to_owned(),
        approx_size_gb: 0.0,
    });

    let extra_body = match &args.extra_body {
        Some(raw) => parse_extra_body(raw)?,
        None => Map::new(),
    };
    let endpoint = Endpoint::new(EndpointSettings {
        base_url: args.base_url.clone(),
        model: args.model.clone(),
        api_key: args.api_key.clone(),
        temperature: args.temperature,
        extra_body: extra_body.clone(),
        timeout: Duration::from_secs(args.timeout_secs),
    })?;
    let sampler = args
        .pid
        .map(RssSampler::start)
        .transpose()
        .map_err(|error| error.context("--pid"))?;

    let results = run_scenarios(&endpoint, &plan, sampler, &|line| eprintln!("{line}")).await?;

    metadata.duration_ms = started.elapsed().as_millis() as u64;
    let params = Params {
        label: args.label.clone(),
        base_url: args.base_url.clone(),
        request_model: public_model_ref(&args.model, ""),
        model_key,
        api_key_set: args.api_key.is_some(),
        extra_body,
        pid: args.pid,
        scenarios: args.scenarios.clone(),
        trials: args.trials,
        warmup: args.warmup,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        prefill_tokens: args.prefill_tokens.clone(),
        concurrency: args.concurrency.clone(),
        timeout_secs: args.timeout_secs,
    };
    let output: BenchOutput<Params, Results> = BenchOutput {
        schema_version: BENCH_SCHEMA_VERSION,
        metadata,
        params,
        results,
    };

    let path = persist_result(&output)?;
    eprintln!("[persisted] {}", path_for_output(&path));
    let rendered = match args.format {
        OutputFormat::Json => format_json(&output)?,
        OutputFormat::Markdown => format_markdown(&output)?,
    };
    println!("{rendered}");
    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("error: failed to start tokio runtime: {error}");
            return ExitCode::from(2);
        }
    };
    match runtime.block_on(run(args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(1)
        }
    }
}
