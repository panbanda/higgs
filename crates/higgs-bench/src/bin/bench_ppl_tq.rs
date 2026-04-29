#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::indexing_slicing,
    clippy::shadow_unrelated,
    clippy::shadow_reuse,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::needless_pass_by_value,
    clippy::needless_bool_assign,
    clippy::if_not_else,
    clippy::uninlined_format_args
)]
//! `bench_ppl_tq` -- port of `bench_ppl_tq.py`.
//!
//! Compares a higgs server running with a baseline KV cache against the same
//! server running with TurboQuant KV-cache compression. Reports:
//!
//! 1. Decode tok/s vs context length (the headline TQ tradeoff)
//! 2. Output-quality Jaccard similarity between baseline and TQ generations
//! 3. A perplexity proxy computed from generated-token logprobs (PPL =
//!    exp(-mean log_prob)). The prompts are deterministic, temperature is 0.
//!
//! Why this is a "proxy" PPL: the original Python script's Part 1 used
//! `mlx-lm` to load a model in-process and forward over a held-out wikitext
//! window. higgs has no public API for in-process forced-decoding logits and
//! we must not modify the engine in this PR, so we instead drive the server
//! via `/v1/chat/completions` with `logprobs=true` and average the logprobs
//! of the (greedy) generated tokens. The number is comparable across
//! baseline and TQ runs of the same server build, which is exactly the
//! comparison the original script motivated. See PR description for the
//! caveat.

use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// Per-request nonce for cache-busting prefixes. An atomic counter is used
// rather than wall-clock time so each prefix is guaranteed distinct within a
// single bench run, which keeps the prefix cache cold across context-length
// probes.
static PROMPT_NONCE: AtomicU64 = AtomicU64::new(0);

fn next_nonce() -> u64 {
    PROMPT_NONCE.fetch_add(1, Ordering::Relaxed)
}

use anyhow::{Context, Result};
use clap::Parser;
use higgs_bench::http::{self, ChatResult};
use higgs_bench::{
    BenchOutput, ModelInfo, OutputFormat, RunMetadata, format_json, format_markdown,
    persist_result, process,
};
use serde::Serialize;

const PORT: u16 = 8097;
const WORD: &str = "the quick brown fox jumps over the lazy dog ";

const QUALITY_PROMPTS: &[&str] = &[
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

/// Short, neutral prompts whose continuations are scored for the proxy PPL.
/// Deliberately diverse so model state doesn't dominate the per-token logprob.
const PPL_PROMPTS: &[&str] = &[
    "Continue the sentence: The capital of France is",
    "Continue the sentence: Water boils at one hundred",
    "Continue the sentence: Two plus two equals",
    "Continue the sentence: The largest planet in our solar system is",
    "Continue the sentence: A triangle has three",
];

const DEFAULT_BASELINE_CTX: &[u32] = &[100, 1000, 4000, 8000, 16000];
const DEFAULT_TQ_CTX: &[u32] = &[100, 1000, 4000, 8000, 16000, 24000, 32000];

#[derive(Debug, Parser)]
#[command(
    name = "bench_ppl_tq",
    about = "Compare baseline vs TurboQuant KV cache: decode speed, output quality, proxy PPL",
    version
)]
struct Args {
    /// Model path (HF id or local directory).
    model_path: String,

    /// TurboQuant bit width.
    #[arg(long, default_value_t = 3)]
    bits: u32,

    /// Server port (the binary spins up two servers sequentially).
    #[arg(long, default_value_t = PORT)]
    port: u16,

    /// Server-startup timeout (seconds).
    #[arg(long, default_value_t = 120)]
    server_timeout_s: u64,

    /// Skip server tests entirely (returns metadata-only output).
    #[arg(long)]
    skip_server: bool,

    /// Skip the proxy-PPL section.
    #[arg(long)]
    skip_ppl: bool,

    /// Output format (json, markdown).
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
}

#[derive(Debug, Serialize)]
struct DecodeRow {
    ctx_tokens: u32,
    decode_tps: f64,
    ttft_ms: f64,
    completion_tokens: u32,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct QualityRow {
    prompt: String,
    jaccard: f64,
    verdict: String,
}

#[derive(Debug, Serialize)]
struct PplRow {
    prompt: String,
    avg_neg_logprob: f64,
    proxy_ppl: f64,
    n_tokens: u32,
}

#[derive(Debug, Serialize)]
struct ServerSweep {
    decode: Vec<DecodeRow>,
    proxy_ppl: Vec<PplRow>,
}

#[derive(Debug, Serialize)]
struct Params {
    model_path: String,
    bits: u32,
    port: u16,
    skip_ppl: bool,
}

#[derive(Debug, Serialize)]
struct Results {
    baseline: Option<ServerSweep>,
    turboquant: Option<ServerSweep>,
    quality: Vec<QualityRow>,
    quality_avg_jaccard: f64,
    decode_speedup: Vec<DecodeSpeedup>,
}

#[derive(Debug, Serialize)]
struct DecodeSpeedup {
    ctx_tokens: u32,
    baseline_tps: f64,
    tq_tps: f64,
    speedup: f64,
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
    let mut metadata = RunMetadata::capture("bench_ppl_tq");
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

    let mut results = Results {
        baseline: None,
        turboquant: None,
        quality: Vec::new(),
        quality_avg_jaccard: 0.0,
        decode_speedup: Vec::new(),
    };

    if !args.skip_server {
        // --- Baseline server ---
        eprintln!("\n=== Baseline server (no TQ) ===");
        let baseline = run_against_server(
            &client,
            &base_url,
            &args,
            &[],
            DEFAULT_BASELINE_CTX,
            "baseline",
        )
        .await
        .context("baseline server sweep")?;
        let base_quality_outputs = baseline.quality_outputs;
        results.baseline = Some(baseline.sweep);

        tokio::time::sleep(Duration::from_secs(2)).await;

        // --- TQ server ---
        eprintln!("\n=== TurboQuant server ({}-bit) ===", args.bits);
        let tq_extra: Vec<String> = vec![
            "--kv-cache".into(),
            "turboquant".into(),
            "--kv-bits".into(),
            args.bits.to_string(),
            "--kv-seed".into(),
            "0".into(),
        ];
        let tq = run_against_server(
            &client,
            &base_url,
            &args,
            &tq_extra,
            DEFAULT_TQ_CTX,
            &format!("turboquant-{}bit", args.bits),
        )
        .await
        .context("TQ server sweep")?;
        let tq_quality_outputs = tq.quality_outputs;
        results.turboquant = Some(tq.sweep);

        // Quality comparison: baseline vs TQ jaccard.
        let mut total_jaccard = 0.0;
        for ((prompt, base_out), tq_out) in QUALITY_PROMPTS
            .iter()
            .zip(base_quality_outputs.iter())
            .zip(tq_quality_outputs.iter())
        {
            let j = jaccard(base_out, tq_out);
            let verdict = if j > 0.5 { "MATCH" } else { "DIVERGED" };
            total_jaccard += j;
            results.quality.push(QualityRow {
                prompt: (*prompt).to_owned(),
                jaccard: j,
                verdict: verdict.to_owned(),
            });
        }
        if !results.quality.is_empty() {
            results.quality_avg_jaccard = total_jaccard / (results.quality.len() as f64);
        }

        // Decode speedup table.
        if let (Some(b), Some(t)) = (results.baseline.as_ref(), results.turboquant.as_ref()) {
            for tq_row in &t.decode {
                if let Some(base_row) = b.decode.iter().find(|r| r.ctx_tokens == tq_row.ctx_tokens)
                {
                    let speedup = if base_row.decode_tps > 0.0 {
                        tq_row.decode_tps / base_row.decode_tps
                    } else {
                        0.0
                    };
                    results.decode_speedup.push(DecodeSpeedup {
                        ctx_tokens: tq_row.ctx_tokens,
                        baseline_tps: base_row.decode_tps,
                        tq_tps: tq_row.decode_tps,
                        speedup,
                    });
                }
            }
        }
    } else {
        eprintln!("[--skip-server] — emitting metadata-only output");
    }

    metadata.duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let params = Params {
        model_path: args.model_path.clone(),
        bits: args.bits,
        port: args.port,
        skip_ppl: args.skip_ppl,
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

struct ServerRun {
    sweep: ServerSweep,
    quality_outputs: Vec<String>,
}

async fn run_against_server(
    client: &reqwest::Client,
    base_url: &str,
    args: &Args,
    extra_args: &[String],
    ctx_lengths: &[u32],
    label: &str,
) -> Result<ServerRun> {
    let child = process::start_higgs_server(
        &args.model_path,
        args.port,
        extra_args,
        Duration::from_secs(args.server_timeout_s),
    )
    .await?;
    let model = match http::first_model_id(client, base_url).await {
        Ok(m) => m,
        Err(e) => {
            let _ = process::stop_server(child).await;
            return Err(e);
        }
    };
    eprintln!("Ready: {model}");

    let body = run_server_body(client, base_url, &model, args, ctx_lengths, label).await;
    if let Err(e) = process::stop_server(child).await {
        eprintln!("warning: stop_server: {e:#}");
    }
    body
}

async fn run_server_body(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    args: &Args,
    ctx_lengths: &[u32],
    label: &str,
) -> Result<ServerRun> {
    let decode = decode_sweep(client, base_url, model, ctx_lengths, label).await?;

    let proxy_ppl = if args.skip_ppl {
        Vec::new()
    } else {
        ppl_sweep(client, base_url, model).await?
    };

    let quality_outputs = collect_quality_outputs(client, base_url, model).await?;

    Ok(ServerRun {
        sweep: ServerSweep { decode, proxy_ppl },
        quality_outputs,
    })
}

async fn decode_sweep(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    ctx_lengths: &[u32],
    label: &str,
) -> Result<Vec<DecodeRow>> {
    eprintln!("Decode sweep -- {label}");
    eprintln!(
        "  {:>10} | {:>14} | {:>10} | {:>10}",
        "Context", "Decode tok/s", "TTFT ms", "Gen toks"
    );
    let mut rows = Vec::new();
    for &ctx_tokens in ctx_lengths {
        let repeat = (ctx_tokens / 10).max(1) as usize;
        let prompt = WORD.repeat(repeat);

        // Two requests per context: max_tokens=1 isolates TTFT, max_tokens=128
        // gives a decode time after subtracting TTFT.
        let mut p1 = format!("[a{}] Summarize: ", next_nonce());
        p1.push_str(&prompt);
        let r1 = match chat_with_prompt(client, base_url, model, &p1, 1).await {
            Ok(r) => r,
            Err(e) => {
                rows.push(DecodeRow {
                    ctx_tokens,
                    decode_tps: 0.0,
                    ttft_ms: 0.0,
                    completion_tokens: 0,
                    error: Some(format!("{e:#}").chars().take(120).collect()),
                });
                continue;
            }
        };
        let ttft_ms = r1.elapsed_ms;
        let ptoks = r1.prompt_tokens;

        let mut p2 = format!("[b{}] Summarize: ", next_nonce());
        p2.push_str(&prompt);
        let r2 = match chat_with_prompt(client, base_url, model, &p2, 128).await {
            Ok(r) => r,
            Err(e) => {
                rows.push(DecodeRow {
                    ctx_tokens: ptoks,
                    decode_tps: 0.0,
                    ttft_ms,
                    completion_tokens: 0,
                    error: Some(format!("{e:#}").chars().take(120).collect()),
                });
                continue;
            }
        };
        let ctoks = r2.completion_tokens;
        let row = if ctoks > 1 {
            let decode_s = ((r2.elapsed_ms - ttft_ms) / 1000.0).max(0.01);
            let tps = (f64::from(ctoks) - 1.0) / decode_s;
            eprintln!(
                "  {:>10} | {:>11.1} t/s | {:>7.0} ms | {:>10}",
                ptoks, tps, ttft_ms, ctoks
            );
            DecodeRow {
                ctx_tokens: ptoks,
                decode_tps: tps,
                ttft_ms,
                completion_tokens: ctoks,
                error: None,
            }
        } else {
            eprintln!(
                "  {:>10} | {:>14} | {:>7.0} ms | {:>10}",
                ptoks, "NO OUTPUT", ttft_ms, ctoks
            );
            DecodeRow {
                ctx_tokens: ptoks,
                decode_tps: 0.0,
                ttft_ms,
                completion_tokens: ctoks,
                error: None,
            }
        };
        rows.push(row);
    }
    Ok(rows)
}

async fn ppl_sweep(client: &reqwest::Client, base_url: &str, model: &str) -> Result<Vec<PplRow>> {
    eprintln!("Proxy-PPL sweep");
    let mut out = Vec::new();
    for &prompt in PPL_PROMPTS {
        let messages = serde_json::json!([{"role": "user", "content": prompt}]);
        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": 32,
            "temperature": 0,
            "logprobs": true,
        });
        let url = format!("{base_url}/v1/chat/completions");
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;
        if !resp.status().is_success() {
            anyhow::bail!("ppl_sweep: HTTP {}", resp.status());
        }
        let value: serde_json::Value = resp.json().await?;
        let lps = value
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("logprobs"))
            .and_then(|l| l.get("content"))
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut sum_neg = 0.0_f64;
        let mut n = 0_u32;
        for entry in &lps {
            if let Some(lp) = entry.get("logprob").and_then(serde_json::Value::as_f64) {
                if lp.is_finite() {
                    sum_neg += -lp;
                    n += 1;
                }
            }
        }
        let avg_neg = if n > 0 { sum_neg / f64::from(n) } else { 0.0 };
        let ppl = if n > 0 { avg_neg.exp() } else { f64::NAN };
        eprintln!(
            "  prompt='{}...' n={n} ppl_proxy={ppl:.4}",
            prompt.chars().take(40).collect::<String>()
        );
        out.push(PplRow {
            prompt: prompt.to_owned(),
            avg_neg_logprob: avg_neg,
            proxy_ppl: ppl,
            n_tokens: n,
        });
    }
    Ok(out)
}

async fn collect_quality_outputs(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
) -> Result<Vec<String>> {
    let mut outs = Vec::new();
    for prompt in QUALITY_PROMPTS {
        let r = chat_with_prompt(client, base_url, model, prompt, 64).await?;
        outs.push(r.content);
    }
    Ok(outs)
}

async fn chat_with_prompt(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<ChatResult> {
    let messages = serde_json::json!([{"role": "user", "content": prompt}]);
    http::chat_with_options(client, base_url, model, &messages, max_tokens, 0.0, None).await
}

fn jaccard(a: &str, b: &str) -> f64 {
    use std::collections::HashSet;
    let aw: HashSet<String> = a
        .to_lowercase()
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let bw: HashSet<String> = b
        .to_lowercase()
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    if aw.is_empty() && bw.is_empty() {
        return 0.0;
    }
    let inter = aw.intersection(&bw).count() as f64;
    let uni = aw.union(&bw).count() as f64;
    if uni == 0.0 { 0.0 } else { inter / uni }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaccard_identical_one() {
        let a = "the quick brown fox";
        assert!((jaccard(a, a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jaccard_disjoint_zero() {
        let a = "alpha beta gamma";
        let b = "delta epsilon";
        assert!(jaccard(a, b).abs() < 1e-9);
    }

    #[test]
    fn jaccard_partial() {
        let a = "the quick brown fox";
        let b = "the slow brown dog";
        // {the, brown} ∩, {the, quick, brown, fox, slow, dog} ∪ -> 2/6
        let j = jaccard(a, b);
        assert!((j - (2.0_f64 / 6.0)).abs() < 1e-9);
    }

    #[test]
    fn proxy_ppl_math() {
        // PPL proxy: exp(mean(neg_logprob)). Two tokens with logprob -1.0
        // and -2.0 => mean = 1.5 => ppl = e^1.5.
        let logprobs = [-1.0_f64, -2.0_f64];
        let sum: f64 = logprobs.iter().map(|x| -x).sum();
        let avg = sum / (logprobs.len() as f64);
        let ppl = avg.exp();
        let expected = 1.5_f64.exp();
        assert!((ppl - expected).abs() < 1e-9);
    }
}
