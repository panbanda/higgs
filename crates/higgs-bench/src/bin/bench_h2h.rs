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
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::needless_range_loop
)]
//! `bench_h2h` — Higgs vs oMLX head-to-head TTFT and decode tok/s.
//!
//! Port of `bench_h2h.py`. Spins up a higgs server then an
//! oMLX subprocess (path: `/Applications/oMLX.app/Contents/MacOS/omlx-cli`,
//! overridable via `OMLX_CLI`) for each model, sends the same set of
//! prompts to both, and prints a side-by-side comparison.
//!
//! Models are selected by key. Defaults to all models tagged `h2h` in
//! `benchmarks/models.toml`; the original Python uses `--models 35B|27B|DSV2`.
//! Pass `--models qwen3.5-35B-a3b-3bit` etc. to mirror that behavior.

use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use higgs_bench::http::{StreamChatOptions, StreamMetrics};
use higgs_bench::{
    BenchOutput, ModelInfo, OutputFormat, RunMetadata, default_manifest_path, format_json,
    format_markdown, http, models, persist_result, process,
};
use serde::Serialize;
use tokio::process::Child;

const HIGGS_PORT: u16 = 8899;
const OMLX_PORT: u16 = 8000;

const MAX_TOKENS: u32 = 100;
const COOLDOWN_S: u64 = 5;
const WARMUP_TOKENS: u32 = 10;

const SHORT_PROMPT: &str = "What is 2+2? Answer in one word.";
const MEDIUM_PROMPT: &str = "Write a detailed technical explanation of how transformer architectures work, covering attention mechanisms, positional encoding, layer normalization, feed-forward networks, and the differences between encoder and decoder architectures. Include discussion of multi-head attention, scaled dot-product attention, and how these components work together.";
const LONG_PROMPT: &str = "Write an extremely comprehensive and detailed technical guide covering the following topics in depth:\n\n1. COMPILER DESIGN: Explain lexical analysis, parsing (LL, LR, LALR), abstract syntax trees, semantic analysis, intermediate representations (SSA form, three-address code), optimization passes (constant folding, dead code elimination, loop unrolling, register allocation via graph coloring), and code generation for modern CPU architectures.\n\n2. OPERATING SYSTEMS: Cover process scheduling algorithms (CFS, MLFQ, lottery scheduling), virtual memory management (page tables, TLB, huge pages, NUMA), file systems (ext4, btrfs, ZFS internals), I/O scheduling, interrupt handling, system calls.\n\n3. DISTRIBUTED SYSTEMS: Explain consensus protocols (Paxos, Raft, PBFT), distributed hash tables, vector clocks, CRDTs, the CAP theorem, leader election algorithms, distributed transactions (2PC, 3PC, saga pattern).\n\n4. CRYPTOGRAPHY: Cover symmetric encryption (AES internals, modes of operation), asymmetric encryption (RSA, elliptic curves, key exchange), hash functions (SHA-256 internals), digital signatures, zero-knowledge proofs.\n\n5. DATABASE INTERNALS: Explain B-tree and LSM-tree storage engines, write-ahead logging, MVCC, query optimization, join algorithms, buffer pool management.\n\nBe thorough and technical throughout.";

const SYSTEM_PROMPT: &str = "You are a highly skilled software architect with deep expertise in distributed systems, database design, and cloud-native applications. You provide thorough, well-reasoned technical advice with step-by-step reasoning and concrete examples.";

const TURN_QUESTIONS: &[&str] = &[
    "Explain the CAP theorem and its practical implications for system design.",
    "How would you design a rate limiter for a distributed API gateway?",
    "Compare event sourcing with traditional CRUD. When would you pick each?",
    "What are the key differences between Raft and Paxos consensus protocols?",
    "Design a notification system that handles 1M users with real-time delivery.",
    "How does MVCC work in PostgreSQL? Walk me through a concurrent update scenario.",
    "What strategies would you use to migrate a monolith to microservices safely?",
];

#[derive(Debug, Parser)]
#[command(
    name = "bench_h2h",
    about = "Head-to-head benchmark: Higgs vs oMLX (port of bench_h2h.py)",
    version
)]
struct Args {
    /// Comma-separated list of model keys from `benchmarks/models.toml`.
    /// Defaults to all models tagged `h2h`.
    #[arg(long)]
    models: Option<String>,

    /// Number of multi-turn conversation turns.
    #[arg(long, default_value_t = 5)]
    turns: usize,

    /// Skip multi-turn tests.
    #[arg(long)]
    skip_multiturn: bool,

    /// Run only the higgs side.
    #[arg(long)]
    higgs_only: bool,

    /// Run only the oMLX side.
    #[arg(long)]
    omlx_only: bool,

    /// Override the manifest path.
    #[arg(long)]
    manifest: Option<std::path::PathBuf>,

    /// Server startup timeout (seconds).
    #[arg(long, default_value_t = 180)]
    server_timeout_s: u64,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
}

#[derive(Debug, Serialize)]
struct Params {
    max_tokens: u32,
    turns: usize,
    cooldown_s: u64,
    skip_multiturn: bool,
    higgs_only: bool,
    omlx_only: bool,
    model_keys: Vec<String>,
}

#[derive(Debug, Serialize, Clone)]
struct PromptResult {
    label: String,
    ttft_ms: f64,
    decode_tps: f64,
    prompt_tokens: u32,
    completion_tokens: u32,
    completion_tokens_estimated: bool,
    total_ms: f64,
}

#[derive(Debug, Serialize, Clone)]
struct TurnResult {
    turn: usize,
    ttft_ms: f64,
    decode_tps: f64,
    prompt_tokens: u32,
    completion_tokens: u32,
    total_ms: f64,
}

#[derive(Debug, Serialize, Clone)]
struct BackendResults {
    backend: String,
    model: String,
    single_turn: Vec<PromptResult>,
    multi_turn: Vec<TurnResult>,
}

#[derive(Debug, Serialize, Clone)]
struct ModelComparison {
    model_key: String,
    label: String,
    higgs: Option<BackendResults>,
    omlx: Option<BackendResults>,
}

#[derive(Debug, Serialize)]
struct Results {
    comparisons: Vec<ModelComparison>,
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
    let mut metadata = RunMetadata::capture("bench_h2h");
    let started = Instant::now();

    let manifest_path = args.manifest.clone().unwrap_or_else(default_manifest_path);
    let manifest = models::load_manifest(&manifest_path)?;
    let selected = select_models(&manifest, &args)?;
    if selected.is_empty() {
        anyhow::bail!("no models selected; pass --models <key,...> or tag a model with `h2h`");
    }
    eprintln!("{}", "=".repeat(80));
    eprintln!("HEAD-TO-HEAD: Higgs vs oMLX");
    eprintln!(
        "Max tokens: {MAX_TOKENS}  Turns: {}  Cooldown: {}s",
        args.turns, COOLDOWN_S
    );
    eprintln!(
        "Models: {}",
        selected
            .iter()
            .map(|m| m.key.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!("{}", "=".repeat(80));

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;

    // Persist one BenchOutput per model. `persist_result` derives the
    // filename from `metadata.model.key` and `bench_summarize` groups by
    // it; a single combined record would only record the first model's
    // key, hiding every other model's data inside `results.comparisons`.
    // Splitting per-model keeps each model independently summarizable.
    let mut all_comparisons: Vec<ModelComparison> = Vec::new();
    for model in &selected {
        eprintln!("\n{}", "#".repeat(80));
        eprintln!("# MODEL: {}", model.label);
        eprintln!("# Path:  {}", model.path);
        eprintln!("{}", "#".repeat(80));

        let mut comparison = ModelComparison {
            model_key: model.key.clone(),
            label: model.label.clone(),
            higgs: None,
            omlx: None,
        };

        if !args.omlx_only {
            match run_for_higgs(&client, model, &args).await {
                Ok(r) => comparison.higgs = Some(r),
                Err(e) => eprintln!("  Higgs error: {e:#}"),
            }
            tokio::time::sleep(Duration::from_secs(COOLDOWN_S)).await;
        }
        if !args.higgs_only {
            match run_for_omlx(&client, model, &args).await {
                Ok(r) => comparison.omlx = Some(r),
                Err(e) => eprintln!("  oMLX error: {e:#}"),
            }
            tokio::time::sleep(Duration::from_secs(COOLDOWN_S)).await;
        }

        if let (Some(h), Some(o)) = (&comparison.higgs, &comparison.omlx) {
            print_comparison(&model.label, h, o);
        }

        let mut per_model_meta = metadata.clone();
        per_model_meta.model = Some(ModelInfo {
            key: model.key.clone(),
            path: model.path.clone(),
            quantization: model.quantization.clone(),
            approx_size_gb: model.approx_size_gb,
        });
        per_model_meta.duration_ms = started.elapsed().as_millis() as u64;
        let per_params = Params {
            max_tokens: MAX_TOKENS,
            turns: args.turns,
            cooldown_s: COOLDOWN_S,
            skip_multiturn: args.skip_multiturn,
            higgs_only: args.higgs_only,
            omlx_only: args.omlx_only,
            model_keys: vec![model.key.clone()],
        };
        let per_results = Results {
            comparisons: vec![comparison.clone()],
        };
        let per_model_output = BenchOutput {
            metadata: per_model_meta,
            params: per_params,
            results: per_results,
        };
        let path = persist_result(&per_model_output)?;
        eprintln!("[persisted] {}", path.display());
        all_comparisons.push(comparison);
    }

    // Render a combined view to stdout for the human watching the run.
    metadata.duration_ms = started.elapsed().as_millis() as u64;
    if let Some(first) = selected.first() {
        metadata.model = Some(ModelInfo {
            key: first.key.clone(),
            path: first.path.clone(),
            quantization: first.quantization.clone(),
            approx_size_gb: first.approx_size_gb,
        });
    }
    let params = Params {
        max_tokens: MAX_TOKENS,
        turns: args.turns,
        cooldown_s: COOLDOWN_S,
        skip_multiturn: args.skip_multiturn,
        higgs_only: args.higgs_only,
        omlx_only: args.omlx_only,
        model_keys: selected.iter().map(|m| m.key.clone()).collect(),
    };
    let combined = BenchOutput {
        metadata,
        params,
        results: Results {
            comparisons: all_comparisons,
        },
    };
    let rendered = match args.format {
        OutputFormat::Json => format_json(&combined)?,
        OutputFormat::Markdown => format_markdown(&combined)?,
    };
    println!("{rendered}");
    Ok(())
}

fn select_models(manifest: &models::Manifest, args: &Args) -> Result<Vec<models::Model>> {
    if let Some(list) = &args.models {
        list.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|key| {
                manifest
                    .find_by_key(key)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("model key '{key}' not in manifest"))
            })
            .collect()
    } else {
        Ok(manifest.find_by_tag("h2h").into_iter().cloned().collect())
    }
}

async fn run_for_higgs(
    client: &reqwest::Client,
    model: &models::Model,
    args: &Args,
) -> Result<BackendResults> {
    eprintln!("\n  --- higgs ---");
    eprintln!("  Starting Higgs on :{HIGGS_PORT} ...");
    let child = process::start_higgs_server(
        &model.path,
        HIGGS_PORT,
        &[],
        Duration::from_secs(args.server_timeout_s),
    )
    .await?;
    let base_url = format!("http://127.0.0.1:{HIGGS_PORT}");
    let result = run_backend(client, &base_url, "higgs", model, None, args).await;
    if let Err(e) = process::stop_server(child).await {
        eprintln!("warning: stop_server: {e:#}");
    }
    result
}

async fn run_for_omlx(
    client: &reqwest::Client,
    model: &models::Model,
    args: &Args,
) -> Result<BackendResults> {
    eprintln!("\n  --- omlx ---");
    eprintln!("  Starting oMLX on :{OMLX_PORT} (--no-cache) ...");
    // oMLX's `--model-dir` expects a local parent directory containing
    // the model one level below — not a HuggingFace repo id like
    // `mlx-community/Foo`. The default manifest uses HF-style
    // `model.path` strings so that Higgs can resolve them via its own
    // cache; here we require an explicit `omlx_model_dir` cache prefix
    // (typically `~/.cache/lm-studio/models`) and fail loudly if missing.
    let cache_prefix = model.resolved_omlx_model_dir().ok_or_else(|| {
        anyhow::anyhow!(
            "model '{}' has no `omlx_model_dir`; H2H requires a local cache path. \
                 Add `omlx_model_dir = \"~/.cache/lm-studio/models/...\"` to its manifest entry.",
            model.key
        )
    })?;
    // oMLX walks one level deep from --model-dir, so `omlx_model_dir`
    // must be the parent of the model's own directory. The model's
    // directory name is the trailing path component of `model.path`.
    let model_subdir = Path::new(&model.path);
    let model_full_path = cache_prefix.join(model_subdir);
    let parent = model_full_path
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .ok_or_else(|| {
            anyhow::anyhow!("can't resolve parent dir of {}", model_full_path.display())
        })?;
    let child: Child = process::start_omlx_server(
        &parent,
        OMLX_PORT,
        Duration::from_secs(args.server_timeout_s),
    )
    .await?;
    let base_url = format!("http://127.0.0.1:{OMLX_PORT}");
    // oMLX discovers models by directory basename — use that as the model id.
    let basename = model_full_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| model.path.clone());
    let result = run_backend(
        client,
        &base_url,
        "omlx",
        model,
        Some(("omlx", basename.as_str())),
        args,
    )
    .await;
    if let Err(e) = process::stop_server(child).await {
        eprintln!("warning: stop_server: {e:#}");
    }
    result
}

async fn run_backend(
    client: &reqwest::Client,
    base_url: &str,
    backend: &str,
    model: &models::Model,
    omlx_auth: Option<(&str, &str)>,
    args: &Args,
) -> Result<BackendResults> {
    let model_id = if let Some((api_key, override_id)) = omlx_auth {
        // Verify oMLX is responding, then use the override (basename) as id.
        let _ = http::first_model_id_with_auth(client, base_url, Some(api_key)).await?;
        override_id.to_owned()
    } else {
        http::first_model_id(client, base_url).await?
    };
    let api_key = omlx_auth.map(|(k, _)| k);

    eprintln!("  Server ready: model={model_id}");
    eprintln!("  Warmup...");
    let warmup_msgs = serde_json::json!([{"role": "user", "content": "Say hi."}]);
    let _ = stream(
        client,
        base_url,
        &model_id,
        &warmup_msgs,
        WARMUP_TOKENS,
        api_key,
    )
    .await?;
    eprintln!("  Warmup done.");

    eprintln!("\n  [Single-turn]");
    let mut single_turn = Vec::new();
    for (label, prompt) in [
        ("short", SHORT_PROMPT),
        ("medium", MEDIUM_PROMPT),
        ("long", LONG_PROMPT),
    ] {
        let msgs = serde_json::json!([{"role": "user", "content": prompt}]);
        let r = stream(client, base_url, &model_id, &msgs, MAX_TOKENS, api_key).await?;
        eprintln!(
            "    {label:8}: TTFT={:>7.0}ms  decode={:>5.1}tok/s  prompt={:>4}tok  gen={:>3}tok  total={:>5.1}s",
            r.ttft_ms,
            r.decode_tps,
            r.prompt_tokens,
            r.completion_tokens,
            r.total_ms / 1000.0
        );
        single_turn.push(PromptResult {
            label: label.to_owned(),
            ttft_ms: r.ttft_ms,
            decode_tps: r.decode_tps,
            prompt_tokens: r.prompt_tokens,
            completion_tokens: r.completion_tokens,
            completion_tokens_estimated: r.completion_tokens_estimated,
            total_ms: r.total_ms,
        });
    }

    let mut multi_turn = Vec::new();
    if !args.skip_multiturn {
        eprintln!("\n  [Multi-turn, {} turns]", args.turns);
        let mut messages: Vec<serde_json::Value> =
            vec![serde_json::json!({"role": "system", "content": SYSTEM_PROMPT})];
        let n = args.turns.min(TURN_QUESTIONS.len());
        for i in 0..n {
            messages.push(serde_json::json!({"role": "user", "content": TURN_QUESTIONS[i]}));
            let value = serde_json::Value::Array(messages.clone());
            let r = stream(client, base_url, &model_id, &value, 80, api_key).await?;
            eprintln!(
                "    turn {}: TTFT={:>7.0}ms  decode={:>5.1}tok/s  ctx~{}",
                i + 1,
                r.ttft_ms,
                r.decode_tps,
                r.prompt_tokens
            );
            multi_turn.push(TurnResult {
                turn: i + 1,
                ttft_ms: r.ttft_ms,
                decode_tps: r.decode_tps,
                prompt_tokens: r.prompt_tokens,
                completion_tokens: r.completion_tokens,
                total_ms: r.total_ms,
            });
            messages.push(serde_json::json!({"role": "assistant", "content": r.output}));
        }
    }

    Ok(BackendResults {
        backend: backend.to_owned(),
        model: model.label.clone(),
        single_turn,
        multi_turn,
    })
}

async fn stream(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    messages: &serde_json::Value,
    max_tokens: u32,
    api_key: Option<&str>,
) -> Result<StreamMetrics> {
    let is_omlx = api_key.is_some();
    let opts = StreamChatOptions {
        max_tokens,
        temperature: 0.0,
        api_key,
        response_format: None,
        // oMLX doesn't report prompt tokens in SSE; estimate.
        estimate_prompt_tokens: is_omlx,
        // Higgs honors stream_options.include_usage; oMLX rejects unknown
        // body keys, so only request it on the Higgs side. The oMLX path
        // falls back to the character-count completion-token estimate.
        include_usage: !is_omlx,
    };
    http::stream_chat_metrics(client, base_url, model, messages, &opts)
        .await
        .with_context(|| format!("stream {base_url}/v1/chat/completions"))
}

fn print_comparison(label: &str, higgs: &BackendResults, omlx: &BackendResults) {
    eprintln!("\n{}", "=".repeat(80));
    eprintln!("COMPARISON: {label}");
    eprintln!("{}", "=".repeat(80));

    eprintln!(
        "\n  {:12} {:>20}   {:>20}",
        "", "TTFT (ms)", "Decode (tok/s)"
    );
    eprintln!(
        "  {:12} {:>9} {:>9}   {:>9} {:>9}  {:>7} {:>7}",
        "Prompt", "Higgs", "oMLX", "Higgs", "oMLX", "TTFT", "Decode"
    );
    eprintln!("  {}", "-".repeat(74));

    for label in ["short", "medium", "long"] {
        let h = higgs.single_turn.iter().find(|r| r.label == label);
        let o = omlx.single_turn.iter().find(|r| r.label == label);
        let (Some(h), Some(o)) = (h, o) else {
            continue;
        };
        let ttft_ratio = if h.ttft_ms > 0.0 {
            format!("{:.2}x", o.ttft_ms / h.ttft_ms)
        } else {
            "—".to_owned()
        };
        let dec_ratio = if o.decode_tps > 0.0 {
            format!("{:.2}x", h.decode_tps / o.decode_tps)
        } else {
            "—".to_owned()
        };
        eprintln!(
            "  {label:12} {:>8.0}  {:>8.0}   {:>8.1}  {:>8.1}  {ttft_ratio:>7} {dec_ratio:>7}",
            h.ttft_ms, o.ttft_ms, h.decode_tps, o.decode_tps,
        );
    }

    if !higgs.multi_turn.is_empty() && !omlx.multi_turn.is_empty() {
        eprintln!("\n  Multi-turn TTFT progression:");
        eprintln!(
            "  {:>5}  {:>11}  {:>11}  {:>10}  {:>10}",
            "Turn", "Higgs TTFT", "oMLX TTFT", "Higgs dec", "oMLX dec"
        );
        eprintln!("  {}", "-".repeat(55));
        let n = higgs.multi_turn.len().min(omlx.multi_turn.len());
        for i in 0..n {
            let h = &higgs.multi_turn[i];
            let o = &omlx.multi_turn[i];
            eprintln!(
                "  {:>5}  {:>9.0}ms  {:>9.0}ms  {:>8.1}/s  {:>8.1}/s",
                i + 1,
                h.ttft_ms,
                o.ttft_ms,
                h.decode_tps,
                o.decode_tps,
            );
        }
    }
}
