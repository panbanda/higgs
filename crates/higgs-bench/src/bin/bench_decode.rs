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
//! `bench_decode` drives a running higgs server over HTTP and reports
//! per-trial decode tok/s and TTFT (time to first token).

use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use futures::StreamExt;
use higgs_bench::{
    BenchOutput, ModelInfo, OutputFormat, RunMetadata, default_manifest_path, format_json,
    format_markdown, models, path_for_output, persist_result, public_model_ref, server, stats,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "bench_decode",
    about = "Measure decode tok/s and TTFT against a running higgs server",
    version
)]
struct Args {
    /// Port the higgs server is listening on.
    #[arg(long, default_value_t = 8899)]
    port: u16,

    /// Host the higgs server is listening on.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Model key from `benchmarks/models.toml`.
    #[arg(long)]
    model: String,

    /// Override the manifest path.
    #[arg(long)]
    manifest: Option<std::path::PathBuf>,

    /// Maximum tokens to generate per trial.
    #[arg(long, default_value_t = 200)]
    max_tokens: u32,

    /// Number of warmup trials (not measured).
    #[arg(long, default_value_t = 1)]
    warmup: u32,

    /// Number of measured trials.
    #[arg(long, default_value_t = 5)]
    trials: u32,

    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    #[arg(long)]
    top_k: Option<u32>,

    #[arg(long)]
    top_p: Option<f32>,

    #[arg(long)]
    repetition_penalty: Option<f32>,

    #[arg(long)]
    frequency_penalty: Option<f32>,

    /// Prompt to send. Defaults to a short fixed prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Output format (json | markdown).
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,

    /// Skip the readiness probe and start measuring immediately.
    #[arg(long)]
    no_wait: bool,
}

#[derive(Debug, Serialize)]
struct Params {
    host: String,
    port: u16,
    model_key: String,
    model_path: String,
    max_tokens: u32,
    warmup: u32,
    trials: u32,
    temperature: f32,
    top_k: Option<u32>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    prompt: String,
}

#[derive(Debug, Serialize, Clone)]
struct TrialResult {
    ttft_ms: f64,
    decode_tokps: f64,
    tokens_after_first: u32,
    total_tokens: u32,
    wall_ms: f64,
}

#[derive(Debug, Serialize)]
struct Results {
    trials: Vec<TrialResult>,
    ttft_ms_mean: f64,
    ttft_ms_median: f64,
    ttft_ms_p95: f64,
    ttft_ms_stdev: f64,
    decode_tokps_mean: f64,
    decode_tokps_median: f64,
    decode_tokps_p95: f64,
    decode_tokps_stdev: f64,
}

const DEFAULT_PROMPT: &str = "What is 2+2? Answer in one word.";

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
    if args.trials == 0 {
        anyhow::bail!("--trials must be >= 1");
    }

    let mut metadata = RunMetadata::capture("bench_decode");
    let started = Instant::now();

    let manifest_path = args.manifest.clone().unwrap_or_else(default_manifest_path);
    let model = models::find_by_key(&manifest_path, &args.model)?;
    let public_model_path = public_model_ref(&model.path, "");
    metadata.model = Some(ModelInfo {
        key: model.key.clone(),
        path: public_model_path.clone(),
        quantization: model.quantization.clone(),
        approx_size_gb: model.approx_size_gb,
    });

    let base_url = format!("http://{}:{}", args.host, args.port);
    if !args.no_wait {
        server::wait_until_ready(&base_url, Duration::from_secs(30))
            .await
            .with_context(|| {
                format!(
                    "higgs server not reachable at {base_url}; start it with `higgs serve --port {}`",
                    args.port
                )
            })?;
    }

    let prompt = args
        .prompt
        .clone()
        .unwrap_or_else(|| DEFAULT_PROMPT.to_owned());
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;

    for i in 0..args.warmup {
        eprintln!("[warmup {}/{}]", i + 1, args.warmup);
        let _ = run_trial(&client, &base_url, &model.path, &prompt, &args).await?;
    }

    let mut trials = Vec::with_capacity(args.trials as usize);
    for i in 0..args.trials {
        eprintln!("[trial  {}/{}]", i + 1, args.trials);
        let trial = run_trial(&client, &base_url, &model.path, &prompt, &args).await?;
        trials.push(trial);
    }

    let ttft: Vec<f64> = trials.iter().map(|t| t.ttft_ms).collect();
    let dec: Vec<f64> = trials.iter().map(|t| t.decode_tokps).collect();
    let results = Results {
        trials: trials.clone(),
        ttft_ms_mean: stats::mean(&ttft),
        ttft_ms_median: stats::median(&ttft),
        ttft_ms_p95: stats::p95(&ttft),
        ttft_ms_stdev: stats::stdev(&ttft),
        decode_tokps_mean: stats::mean(&dec),
        decode_tokps_median: stats::median(&dec),
        decode_tokps_p95: stats::p95(&dec),
        decode_tokps_stdev: stats::stdev(&dec),
    };

    metadata.duration_ms = started.elapsed().as_millis() as u64;

    let params = Params {
        host: args.host.clone(),
        port: args.port,
        model_key: model.key.clone(),
        model_path: public_model_path,
        max_tokens: args.max_tokens,
        warmup: args.warmup,
        trials: args.trials,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        frequency_penalty: args.frequency_penalty,
        prompt,
    };

    let output = BenchOutput {
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

#[allow(clippy::too_many_lines)]
async fn run_trial(
    client: &reqwest::Client,
    base_url: &str,
    model_path: &str,
    prompt: &str,
    args: &Args,
) -> Result<TrialResult> {
    // Request the terminal usage chunk so we measure throughput against
    // server-reported token counts rather than SSE chunk count (which can
    // diverge once the server emits buffered visible text).
    //
    // Disable reasoning so the bench measures time-to-first-generated-token,
    // not time-to-first-visible-answer. Qwen3.5+ models default to thinking
    // mode in higgs and would otherwise emit `reasoning_content` deltas the
    // current code skips, biasing TTFT high.
    let mut body = serde_json::json!({
        "model": model_path,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
        "stream_options": { "include_usage": true },
        "reasoning": { "effort": "none" },
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    });
    if let Some(p) = args.top_p {
        body["top_p"] = serde_json::json!(p);
    }
    if let Some(k) = args.top_k {
        body["top_k"] = serde_json::json!(k);
    }
    if let Some(rp) = args.repetition_penalty {
        body["repetition_penalty"] = serde_json::json!(rp);
    }
    if let Some(fp) = args.frequency_penalty {
        body["frequency_penalty"] = serde_json::json!(fp);
    }

    let url = format!("{base_url}/v1/chat/completions");
    let started = Instant::now();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("{url} returned HTTP {status}: {text}");
    }

    let mut stream = resp.bytes_stream();
    let mut first_token_at: Option<Instant> = None;
    // SSE-chunk-based estimate. Used only as a fallback when the server
    // doesn't send a terminal usage chunk.
    let mut chunks_after_first: u32 = 0;
    // Server-reported token counts from the terminal `usage` chunk (preferred).
    let mut server_completion_tokens: Option<u32> = None;
    let mut server_prompt_tokens: Option<u32> = None;
    // Byte buffer so multibyte UTF-8 sequences split across network chunks
    // aren't corrupted by a per-chunk lossy decode.
    let mut buf: Vec<u8> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read SSE chunk")?;
        buf.extend_from_slice(&chunk);

        while let Some(idx) = buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = buf.drain(..=idx).collect();
            handle_sse_line(
                &line_bytes,
                &mut first_token_at,
                &mut chunks_after_first,
                &mut server_completion_tokens,
                &mut server_prompt_tokens,
            );
        }
    }

    // Some servers close the stream without a trailing newline on the final
    // `data:` line. Flush whatever's left so we don't drop the terminal usage
    // chunk silently.
    if !buf.is_empty() {
        handle_sse_line(
            &buf,
            &mut first_token_at,
            &mut chunks_after_first,
            &mut server_completion_tokens,
            &mut server_prompt_tokens,
        );
    }

    let total_elapsed = started.elapsed();
    let first_at =
        first_token_at.ok_or_else(|| anyhow::anyhow!("no streamed tokens received from server"))?;
    let ttft_ms = first_at.duration_since(started).as_secs_f64() * 1000.0;
    let after_first = total_elapsed.saturating_sub(first_at.duration_since(started));

    // Prefer server-reported completion_tokens. Subtract one for the first
    // visible token (already counted as TTFT, not part of decode wall). Fall
    // back to SSE-chunk count only when the server omits the usage chunk
    // (legacy / non-higgs backends that don't honor `stream_options.include_usage`).
    let tokens_after_first =
        server_completion_tokens.map_or(chunks_after_first, |total| total.saturating_sub(1));
    let decode_tokps = if after_first.as_secs_f64() > 0.0 && tokens_after_first > 0 {
        f64::from(tokens_after_first) / after_first.as_secs_f64()
    } else {
        0.0
    };

    let total_tokens = server_completion_tokens.unwrap_or(chunks_after_first + 1);
    let _ = server_prompt_tokens; // currently unused; surface in future result schema.

    Ok(TrialResult {
        ttft_ms,
        decode_tokps,
        tokens_after_first,
        total_tokens,
        wall_ms: total_elapsed.as_secs_f64() * 1000.0,
    })
}

/// Parse one raw SSE line (without trailing newline) and update the rolling
/// counters. Tolerates non-UTF-8 prefixes, blank lines, and `[DONE]`.
fn handle_sse_line(
    line_bytes: &[u8],
    first_token_at: &mut Option<Instant>,
    chunks_after_first: &mut u32,
    server_completion_tokens: &mut Option<u32>,
    server_prompt_tokens: &mut Option<u32>,
) {
    let line = String::from_utf8_lossy(line_bytes);
    let line = line.trim();
    if line.is_empty() || !line.starts_with("data:") {
        return;
    }
    let data = line[5..].trim();
    if data == "[DONE]" {
        return;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
        return;
    };
    if let Some(usage) = value.get("usage").filter(|u| u.is_object()) {
        if let Some(ct) = usage
            .get("completion_tokens")
            .and_then(serde_json::Value::as_u64)
        {
            *server_completion_tokens = Some(u32::try_from(ct).unwrap_or(u32::MAX));
        }
        if let Some(pt) = usage
            .get("prompt_tokens")
            .and_then(serde_json::Value::as_u64)
        {
            *server_prompt_tokens = Some(u32::try_from(pt).unwrap_or(u32::MAX));
        }
    }
    let delta = value
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"))
        .and_then(|d| d.get("content"))
        .and_then(|s| s.as_str());
    if let Some(s) = delta {
        if !s.is_empty() {
            if first_token_at.is_none() {
                *first_token_at = Some(Instant::now());
            } else {
                *chunks_after_first += 1;
            }
        }
    }
}
