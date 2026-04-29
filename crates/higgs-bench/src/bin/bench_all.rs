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
//! `bench_all` — sweep TTFT/prefill/decode across many local models.
//!
//! Port of `bench_all.py`. For each model in the manifest
//! (default: tag `all`), launches a higgs server, runs three prompts of
//! varying lengths in streaming mode, and reports per-prompt TTFT,
//! prefill tok/s, decode tok/s, and total wall time.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use higgs_bench::{
    BenchOutput, ModelInfo, OutputFormat, RunMetadata, default_manifest_path, format_json,
    format_markdown, http, models, persist_result, process,
};
use serde::Serialize;

const PROMPT_SHORT: &str = "Explain what a neural network is in one paragraph.";
const PROMPT_MEDIUM: &str = "Write a detailed technical explanation of how transformer architectures work, covering attention mechanisms, positional encoding, layer normalization, feed-forward networks, and the differences between encoder and decoder architectures. Include discussion of multi-head attention, scaled dot-product attention, and how these components work together. Also explain the training process including backpropagation through the attention mechanism. Discuss the key innovations that made transformers superior to RNNs and LSTMs for sequence modeling tasks. Cover the evolution from the original Attention Is All You Need paper through modern variants like GPT, BERT, and their derivatives. Explain how context windows work and the computational complexity of self-attention.";
const PROMPT_LONG: &str = include_str!("../../assets/bench_all_long_prompt.txt");

#[derive(Debug, Parser)]
#[command(
    name = "bench_all",
    about = "TTFT/prefill/decode sweep across all models in the manifest",
    version
)]
struct Args {
    /// Tag to filter the manifest by (defaults to `all`).
    #[arg(long, default_value = "all")]
    tag: String,

    /// Comma-separated list of model keys (overrides --tag).
    #[arg(long)]
    models: Option<String>,

    #[arg(long)]
    manifest: Option<std::path::PathBuf>,

    #[arg(long, default_value_t = 8899)]
    port: u16,

    #[arg(long, default_value_t = 200)]
    max_tokens: u32,

    #[arg(long, default_value_t = 120)]
    server_timeout_s: u64,

    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
}

#[derive(Debug, Serialize)]
struct Params {
    port: u16,
    max_tokens: u32,
    tag: String,
    model_keys: Vec<String>,
}

#[derive(Debug, Serialize, Clone)]
struct PromptResult {
    label: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    ttft_ms: f64,
    prefill_tokps: f64,
    decode_tokps: f64,
    total_ms: f64,
    error: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
struct ModelResult {
    model_key: String,
    model_path: String,
    prompts: Vec<PromptResult>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct Results {
    per_model: Vec<ModelResult>,
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
    let mut metadata = RunMetadata::capture("bench_all");
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
    } else {
        manifest
            .find_by_tag(&args.tag)
            .into_iter()
            .cloned()
            .collect()
    };
    if selected.is_empty() {
        anyhow::bail!("no models match tag/keys; populate benchmarks/models.toml");
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let base_url = format!("http://127.0.0.1:{}", args.port);

    let prompts: [(&str, &str); 3] = [
        ("short", PROMPT_SHORT),
        ("medium", PROMPT_MEDIUM),
        ("long", PROMPT_LONG),
    ];

    let mut per_model = Vec::new();
    for model in &selected {
        eprintln!("--- {} ---", model.path);
        let result = match process::start_higgs_server(
            &model.path,
            args.port,
            &[],
            Duration::from_secs(args.server_timeout_s),
        )
        .await
        {
            Ok(child) => {
                let prompts_result =
                    run_prompts(&client, &base_url, &prompts, args.max_tokens).await;
                if let Err(e) = process::stop_server(child).await {
                    eprintln!("warning: stop_server: {e:#}");
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
                match prompts_result {
                    Ok(p) => ModelResult {
                        model_key: model.key.clone(),
                        model_path: model.path.clone(),
                        prompts: p,
                        error: None,
                    },
                    Err(e) => ModelResult {
                        model_key: model.key.clone(),
                        model_path: model.path.clone(),
                        prompts: Vec::new(),
                        error: Some(format!("{e:#}")),
                    },
                }
            }
            Err(e) => {
                eprintln!("  failed to start server: {e:#}");
                ModelResult {
                    model_key: model.key.clone(),
                    model_path: model.path.clone(),
                    prompts: Vec::new(),
                    error: Some(format!("{e:#}")),
                }
            }
        };
        per_model.push(result);
    }

    metadata.duration_ms = started.elapsed().as_millis() as u64;
    let params = Params {
        port: args.port,
        max_tokens: args.max_tokens,
        tag: args.tag.clone(),
        model_keys: selected.iter().map(|m| m.key.clone()).collect(),
    };

    // Persist one JSON file per model so `bench_summarize` (which
    // groups by `metadata.model.key`) sees every comparison instead of
    // only the first model's run. The aggregate `Results { per_model }`
    // is still rendered to stdout for the human-facing report.
    let by_key: std::collections::HashMap<String, &models::Model> =
        selected.iter().map(|m| (m.key.clone(), m)).collect();
    for model_result in &per_model {
        let Some(model) = by_key.get(&model_result.model_key) else {
            continue;
        };
        let mut model_meta = metadata.clone();
        model_meta.model = Some(ModelInfo {
            key: model.key.clone(),
            path: model.path.clone(),
            quantization: model.quantization.clone(),
            approx_size_gb: model.approx_size_gb,
        });
        let single = BenchOutput {
            metadata: model_meta,
            params: &params,
            results: model_result,
        };
        match persist_result(&single) {
            Ok(path) => eprintln!("[persisted] {}", path.display()),
            Err(e) => eprintln!("warning: persist {}: {e:#}", model.key),
        }
    }

    let results = Results { per_model };
    let output = BenchOutput {
        metadata,
        params,
        results,
    };
    let rendered = match args.format {
        OutputFormat::Json => format_json(&output)?,
        OutputFormat::Markdown => format_markdown(&output)?,
    };
    println!("{rendered}");
    Ok(())
}

async fn run_prompts(
    client: &reqwest::Client,
    base_url: &str,
    prompts: &[(&str, &str)],
    max_tokens: u32,
) -> Result<Vec<PromptResult>> {
    let model_id = http::first_model_id(client, base_url)
        .await
        .context("discover model id")?;
    let mut out = Vec::new();
    for (label, prompt) in prompts {
        let messages = serde_json::json!([{"role": "user", "content": prompt}]);
        match http::stream_chat(client, base_url, &model_id, &messages, max_tokens, 0.0).await {
            Ok(r) => {
                let prompt_tokens = r.prompt_tokens.unwrap_or(0);
                let completion_tokens = r.completion_tokens.unwrap_or(r.num_tokens);
                let prefill_tokps = if r.ttft_ms > 0.0 && prompt_tokens > 0 {
                    f64::from(prompt_tokens) / (r.ttft_ms / 1000.0)
                } else {
                    0.0
                };
                let decode_secs = ((r.total_ms - r.ttft_ms).max(1.0)) / 1000.0;
                let decode_tokps = if completion_tokens > 1 {
                    f64::from(completion_tokens.saturating_sub(1)) / decode_secs
                } else {
                    0.0
                };
                eprintln!(
                    "  {label:>8}: prompt={prompt_tokens}t TTFT={:.0}ms prefill={prefill_tokps:.1}t/s decode={decode_tokps:.1}t/s gen={completion_tokens}t",
                    r.ttft_ms
                );
                out.push(PromptResult {
                    label: (*label).to_owned(),
                    prompt_tokens,
                    completion_tokens,
                    ttft_ms: r.ttft_ms,
                    prefill_tokps,
                    decode_tokps,
                    total_ms: r.total_ms,
                    error: None,
                });
            }
            Err(e) => {
                eprintln!("  {label:>8}: ERROR {e:#}");
                out.push(PromptResult {
                    label: (*label).to_owned(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    ttft_ms: 0.0,
                    prefill_tokps: 0.0,
                    decode_tokps: 0.0,
                    total_ms: 0.0,
                    error: Some(format!("{e:#}")),
                });
            }
        }
    }
    Ok(out)
}
