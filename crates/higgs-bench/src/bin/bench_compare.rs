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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use futures::StreamExt;
use futures::future::try_join_all;
use higgs_bench::compare::{
    StreamAccumulator, StreamOutcome, Summary, TrialMetrics, build_long_output_prompt,
    build_prefill_prompt, build_prompt_body, build_request_body, decode_prompt, fresh_nonce,
    parse_extra_body, prefill_tokps, redact_secret_args, sanitize_model_key,
};
use higgs_bench::{
    BENCH_SCHEMA_VERSION, BenchOutput, ModelInfo, OutputFormat, RunMetadata, format_json,
    format_markdown, path_for_output, persist_result, public_model_ref, stats,
};
use serde::Serialize;
use serde_json::{Map, Value, json};
use sysinfo::{Pid, ProcessesToUpdate, System};

const PREFIX_CACHE_CONTEXT_TOKENS: usize = 4096;
const PREFIX_CACHE_MAX_TOKENS: u32 = 32;
const PREFIX_CACHE_FOLLOW_UP: &str = "Now state the main point of the text in one short sentence.";
const PREFIX_CACHE_FALLBACK_REPLY: &str = "Understood.";
const PREFILL_MAX_TOKENS: u32 = 8;
const CONCURRENCY_ROUNDS: usize = 2;
const RSS_SAMPLE_INTERVAL: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Scenario {
    #[value(name = "decode")]
    Decode,
    #[value(name = "prefill")]
    Prefill,
    #[value(name = "prefix_cache")]
    PrefixCache,
    #[value(name = "concurrency")]
    Concurrency,
}

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

#[derive(Debug, Serialize)]
struct DecodeResults {
    trials: Vec<TrialMetrics>,
    ttft_ms: Summary,
    decode_tokps: Summary,
    e2e_tokps: Summary,
    completion_tokens: Summary,
    trials_below_max_tokens: usize,
    trials_with_usage_missing: usize,
}

#[derive(Debug, Serialize)]
struct PrefillTrial {
    metrics: TrialMetrics,
    prefill_tokps: Option<f64>,
}

#[derive(Debug, Serialize)]
struct PrefillSizeResults {
    target_tokens: usize,
    trials: Vec<PrefillTrial>,
    prompt_tokens_median: f64,
    ttft_ms_median: f64,
    prefill_tokps_median: f64,
    trials_with_cached_tokens: usize,
}

#[derive(Debug, Serialize)]
struct PrefillResults {
    sizes: Vec<PrefillSizeResults>,
}

#[derive(Debug, Serialize)]
struct PrefixCacheTrial {
    turn1: TrialMetrics,
    turn2: TrialMetrics,
    ttft_ratio_turn2_over_turn1: f64,
}

#[derive(Debug, Serialize)]
struct PrefixCacheResults {
    trials: Vec<PrefixCacheTrial>,
    turn1_ttft_ms_median: f64,
    turn2_ttft_ms_median: f64,
    ttft_ratio_median: f64,
    turn2_cached_tokens_median: Option<f64>,
    turn2_prompt_tokens_median: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ConcurrencyRound {
    aggregate_tokps: f64,
    mean_ttft_ms: f64,
    total_completion_tokens: u64,
    wall_ms: f64,
    requests: Vec<TrialMetrics>,
}

#[derive(Debug, Serialize)]
struct ConcurrencyLevelResults {
    concurrency: usize,
    rounds: Vec<ConcurrencyRound>,
    aggregate_tokps_median: f64,
    mean_ttft_ms_median: f64,
}

#[derive(Debug, Serialize)]
struct ConcurrencyResults {
    levels: Vec<ConcurrencyLevelResults>,
}

#[derive(Debug, Default, Serialize)]
struct Results {
    #[serde(skip_serializing_if = "Option::is_none")]
    decode: Option<DecodeResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill: Option<PrefillResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefix_cache: Option<PrefixCacheResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    concurrency: Option<ConcurrencyResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rss_mb_peak: Option<f64>,
}

/// Samples a process's resident set size on a background thread so transient
/// peaks between explicit samples are still captured.
struct RssSampler {
    pid: Pid,
    peak_bytes: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl RssSampler {
    fn start(raw_pid: u32) -> Result<Self> {
        let pid = Pid::from_u32(raw_pid);
        let initial = read_rss_bytes(&mut System::new(), pid)
            .with_context(|| format!("--pid {raw_pid}: process not found"))?;
        let peak_bytes = Arc::new(AtomicU64::new(initial));
        let stop = Arc::new(AtomicBool::new(false));
        let worker = {
            let peak_bytes = Arc::clone(&peak_bytes);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut system = System::new();
                while !stop.load(Ordering::Relaxed) {
                    if let Some(bytes) = read_rss_bytes(&mut system, pid) {
                        peak_bytes.fetch_max(bytes, Ordering::Relaxed);
                    }
                    std::thread::sleep(RSS_SAMPLE_INTERVAL);
                }
            })
        };
        Ok(Self {
            pid,
            peak_bytes,
            stop,
            worker: Some(worker),
        })
    }

    fn sample_now(&self) {
        if let Some(bytes) = read_rss_bytes(&mut System::new(), self.pid) {
            self.peak_bytes.fetch_max(bytes, Ordering::Relaxed);
        }
    }

    fn finish(mut self) -> f64 {
        self.sample_now();
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.peak_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
    }
}

fn read_rss_bytes(system: &mut System, pid: Pid) -> Option<u64> {
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    system.process(pid).map(sysinfo::Process::memory)
}

struct Endpoint {
    client: reqwest::Client,
    url: String,
    api_key: Option<String>,
    model: String,
    temperature: f32,
    extra_body: Map<String, Value>,
}

impl Endpoint {
    async fn stream_chat(&self, messages: &Value, max_tokens: u32) -> Result<StreamOutcome> {
        let body = build_request_body(
            &self.model,
            messages,
            max_tokens,
            self.temperature,
            &self.extra_body,
        );
        let mut request = self.client.post(&self.url).json(&body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }

        let started = Instant::now();
        let response = request
            .send()
            .await
            .with_context(|| format!("POST {}", self.url))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("{} returned HTTP {status}: {text}", self.url);
        }

        let mut accumulator = StreamAccumulator::default();
        let mut stream = response.bytes_stream();
        let mut pending: Vec<u8> = Vec::new();
        while let Some(chunk) = stream.next().await {
            pending.extend_from_slice(&chunk.context("read SSE chunk")?);
            while let Some(newline) = pending.iter().position(|&byte| byte == b'\n') {
                let line: Vec<u8> = pending.drain(..=newline).collect();
                accumulator.ingest_line(&line, started.elapsed());
            }
        }
        if !pending.is_empty() {
            accumulator.ingest_line(&pending, started.elapsed());
        }
        accumulator.finish(started.elapsed())
    }

    async fn stream_user_prompt(&self, prompt: &str, max_tokens: u32) -> Result<TrialMetrics> {
        let messages = json!([{ "role": "user", "content": prompt }]);
        Ok(self.stream_chat(&messages, max_tokens).await?.metrics)
    }
}

fn present(values: impl Iterator<Item = Option<f64>>) -> Vec<f64> {
    values.flatten().collect()
}

fn progress(scenario: &str, phase: &str, index: u32, total: u32) {
    eprintln!("[{scenario} {phase} {}/{total}]", index + 1);
}

async fn run_decode(endpoint: &Endpoint, args: &Args) -> Result<DecodeResults> {
    let prompt = decode_prompt();
    for index in 0..args.warmup {
        progress("decode", "warmup", index, args.warmup);
        endpoint.stream_user_prompt(prompt, args.max_tokens).await?;
    }
    let mut trials = Vec::new();
    for index in 0..args.trials {
        progress("decode", "trial", index, args.trials);
        trials.push(endpoint.stream_user_prompt(prompt, args.max_tokens).await?);
    }

    let ttft: Vec<f64> = trials.iter().map(|t| t.ttft_ms).collect();
    let decode = present(trials.iter().map(|t| t.decode_tokps));
    let e2e: Vec<f64> = trials.iter().map(|t| t.e2e_tokps).collect();
    let completion: Vec<f64> = trials
        .iter()
        .map(|t| f64::from(t.completion_tokens))
        .collect();
    Ok(DecodeResults {
        ttft_ms: Summary::of(&ttft),
        decode_tokps: Summary::of(&decode),
        e2e_tokps: Summary::of(&e2e),
        completion_tokens: Summary::of(&completion),
        trials_below_max_tokens: trials
            .iter()
            .filter(|t| t.completion_tokens < args.max_tokens)
            .count(),
        trials_with_usage_missing: trials.iter().filter(|t| t.usage_missing).count(),
        trials,
    })
}

async fn run_prefill_trial(endpoint: &Endpoint, body: &str) -> Result<PrefillTrial> {
    let prompt = build_prefill_prompt(&fresh_nonce(), body);
    let metrics = endpoint
        .stream_user_prompt(&prompt, PREFILL_MAX_TOKENS)
        .await?;
    let prefill = metrics
        .prompt_tokens
        .and_then(|tokens| prefill_tokps(tokens, metrics.ttft_ms));
    Ok(PrefillTrial {
        metrics,
        prefill_tokps: prefill,
    })
}

async fn run_prefill(endpoint: &Endpoint, args: &Args) -> Result<PrefillResults> {
    let mut sizes = Vec::new();
    for &target_tokens in &args.prefill_tokens {
        let body = build_prompt_body(target_tokens);
        for index in 0..args.warmup {
            progress(
                &format!("prefill {target_tokens}"),
                "warmup",
                index,
                args.warmup,
            );
            run_prefill_trial(endpoint, &body).await?;
        }
        let mut trials = Vec::new();
        for index in 0..args.trials {
            progress(
                &format!("prefill {target_tokens}"),
                "trial",
                index,
                args.trials,
            );
            trials.push(run_prefill_trial(endpoint, &body).await?);
        }

        let ttft: Vec<f64> = trials.iter().map(|t| t.metrics.ttft_ms).collect();
        let prompt_tokens = present(
            trials
                .iter()
                .map(|t| t.metrics.prompt_tokens.map(f64::from)),
        );
        let rates = present(trials.iter().map(|t| t.prefill_tokps));
        sizes.push(PrefillSizeResults {
            target_tokens,
            prompt_tokens_median: stats::median(&prompt_tokens),
            ttft_ms_median: stats::median(&ttft),
            prefill_tokps_median: stats::median(&rates),
            trials_with_cached_tokens: trials
                .iter()
                .filter(|t| t.metrics.cached_tokens.is_some_and(|cached| cached > 0))
                .count(),
            trials,
        });
    }
    Ok(PrefillResults { sizes })
}

async fn run_prefix_cache_trial(endpoint: &Endpoint, context: &str) -> Result<PrefixCacheTrial> {
    let first_user = nonce_context(context);
    let turn1_messages = json!([{ "role": "user", "content": first_user }]);
    let turn1 = endpoint
        .stream_chat(&turn1_messages, PREFIX_CACHE_MAX_TOKENS)
        .await?;

    let reply = if turn1.text.trim().is_empty() {
        PREFIX_CACHE_FALLBACK_REPLY
    } else {
        turn1.text.as_str()
    };
    let turn2_messages = json!([
        { "role": "user", "content": first_user },
        { "role": "assistant", "content": reply },
        { "role": "user", "content": PREFIX_CACHE_FOLLOW_UP },
    ]);
    let turn2 = endpoint
        .stream_chat(&turn2_messages, PREFIX_CACHE_MAX_TOKENS)
        .await?;

    Ok(PrefixCacheTrial {
        ttft_ratio_turn2_over_turn1: turn2.metrics.ttft_ms / turn1.metrics.ttft_ms,
        turn1: turn1.metrics,
        turn2: turn2.metrics,
    })
}

fn nonce_context(body: &str) -> String {
    build_prefill_prompt(&fresh_nonce(), body)
}

async fn run_prefix_cache(endpoint: &Endpoint, args: &Args) -> Result<PrefixCacheResults> {
    let context = build_prompt_body(PREFIX_CACHE_CONTEXT_TOKENS);
    for index in 0..args.warmup {
        progress("prefix_cache", "warmup", index, args.warmup);
        run_prefix_cache_trial(endpoint, &context).await?;
    }
    let mut trials = Vec::new();
    for index in 0..args.trials {
        progress("prefix_cache", "trial", index, args.trials);
        trials.push(run_prefix_cache_trial(endpoint, &context).await?);
    }

    let turn1_ttft: Vec<f64> = trials.iter().map(|t| t.turn1.ttft_ms).collect();
    let turn2_ttft: Vec<f64> = trials.iter().map(|t| t.turn2.ttft_ms).collect();
    let ratios: Vec<f64> = trials
        .iter()
        .map(|t| t.ttft_ratio_turn2_over_turn1)
        .collect();
    let median_of = |select: fn(&TrialMetrics) -> Option<u32>| {
        let values = present(trials.iter().map(|t| select(&t.turn2).map(f64::from)));
        (!values.is_empty()).then(|| stats::median(&values))
    };
    Ok(PrefixCacheResults {
        turn1_ttft_ms_median: stats::median(&turn1_ttft),
        turn2_ttft_ms_median: stats::median(&turn2_ttft),
        ttft_ratio_median: stats::median(&ratios),
        turn2_cached_tokens_median: median_of(|m| m.cached_tokens),
        turn2_prompt_tokens_median: median_of(|m| m.prompt_tokens),
        trials,
    })
}

async fn run_concurrency_round(
    endpoint: &Endpoint,
    level: usize,
    max_tokens: u32,
) -> Result<ConcurrencyRound> {
    let started = Instant::now();
    let requests = try_join_all((0..level).map(|_| {
        let prompt = build_long_output_prompt(&fresh_nonce());
        async move { endpoint.stream_user_prompt(&prompt, max_tokens).await }
    }))
    .await?;
    let wall = started.elapsed();

    let total_completion_tokens: u64 = requests
        .iter()
        .map(|r| u64::from(r.completion_tokens))
        .sum();
    let ttft: Vec<f64> = requests.iter().map(|r| r.ttft_ms).collect();
    Ok(ConcurrencyRound {
        aggregate_tokps: total_completion_tokens as f64 / wall.as_secs_f64(),
        mean_ttft_ms: stats::mean(&ttft),
        total_completion_tokens,
        wall_ms: wall.as_secs_f64() * 1000.0,
        requests,
    })
}

async fn run_concurrency(endpoint: &Endpoint, args: &Args) -> Result<ConcurrencyResults> {
    let mut levels = Vec::new();
    for &level in &args.concurrency {
        let label = format!("concurrency {level}");
        for index in 0..args.warmup {
            progress(&label, "warmup", index, args.warmup);
            run_concurrency_round(endpoint, level, args.max_tokens).await?;
        }
        let mut rounds = Vec::new();
        for index in 0..CONCURRENCY_ROUNDS {
            progress(&label, "round", index as u32, CONCURRENCY_ROUNDS as u32);
            rounds.push(run_concurrency_round(endpoint, level, args.max_tokens).await?);
        }
        let aggregate: Vec<f64> = rounds.iter().map(|r| r.aggregate_tokps).collect();
        let ttft: Vec<f64> = rounds.iter().map(|r| r.mean_ttft_ms).collect();
        levels.push(ConcurrencyLevelResults {
            concurrency: level,
            aggregate_tokps_median: stats::median(&aggregate),
            mean_ttft_ms_median: stats::median(&ttft),
            rounds,
        });
    }
    Ok(ConcurrencyResults { levels })
}

fn validate(args: &Args) -> Result<()> {
    if args.trials == 0 {
        anyhow::bail!("--trials must be >= 1");
    }
    if args.scenarios.is_empty() {
        anyhow::bail!("--scenarios must name at least one scenario");
    }
    if args.concurrency.contains(&0) || args.prefill_tokens.contains(&0) {
        anyhow::bail!("--concurrency and --prefill-tokens values must be >= 1");
    }
    Ok(())
}

async fn run(args: Args) -> Result<()> {
    validate(&args)?;

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
    let endpoint = Endpoint {
        client: reqwest::Client::builder()
            .timeout(Duration::from_secs(args.timeout_secs))
            .build()?,
        url: format!(
            "{}/v1/chat/completions",
            args.base_url.trim_end_matches('/')
        ),
        api_key: args.api_key.clone(),
        model: args.model.clone(),
        temperature: args.temperature,
        extra_body: extra_body.clone(),
    };
    let sampler = args.pid.map(RssSampler::start).transpose()?;

    let mut results = Results::default();
    for scenario in &args.scenarios {
        eprintln!("[scenario] {scenario:?}");
        match scenario {
            Scenario::Decode => results.decode = Some(run_decode(&endpoint, &args).await?),
            Scenario::Prefill => results.prefill = Some(run_prefill(&endpoint, &args).await?),
            Scenario::PrefixCache => {
                results.prefix_cache = Some(run_prefix_cache(&endpoint, &args).await?);
            }
            Scenario::Concurrency => {
                results.concurrency = Some(run_concurrency(&endpoint, &args).await?);
            }
        }
        if let Some(sampler) = &sampler {
            sampler.sample_now();
        }
    }
    results.rss_mb_peak = sampler.map(RssSampler::finish);

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
    let output = BenchOutput {
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
