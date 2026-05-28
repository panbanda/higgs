#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! `bench_speculative` starts a fresh higgs server per trial and compares
//! greedy decode against MTP and prompt-lookup speculative modes.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use higgs_bench::{
    BenchOutput, ModelInfo, OutputFormat, RunMetadata, default_manifest_path, format_json,
    format_markdown, models, path_for_output, persist_result, public_model_ref, results_dir,
    server, speculative, stats,
};
use serde::Serialize;

const DEFAULT_PROMPT: &str = "Write a concise technical explanation of speculative decoding for local LLM inference. Include acceptance rate, verification cost, and why greedy decode is the easiest correctness target.";

#[derive(Debug, Parser)]
#[command(
    name = "bench_speculative",
    about = "Sweep Higgs speculative decode modes with a fresh server per trial",
    version
)]
struct Args {
    /// Model key from `benchmarks/models.toml`.
    #[arg(long, conflicts_with = "model_path")]
    model: Option<String>,

    /// Direct model path or Hugging Face repo ID passed to `higgs serve --model`.
    #[arg(long, conflicts_with = "model")]
    model_path: Option<String>,

    /// Request model name. Defaults to the HF repo ID or local path basename.
    #[arg(long)]
    model_name: Option<String>,

    /// Override the manifest path.
    #[arg(long)]
    manifest: Option<PathBuf>,

    /// Higgs binary to launch for each trial.
    #[arg(long, default_value = "./target/release/higgs")]
    higgs_bin: PathBuf,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 8098)]
    port: u16,

    /// Maximum completion tokens per request.
    #[arg(long, default_value_t = 192)]
    max_tokens: u32,

    /// Number of measured repeats per trial mode.
    #[arg(long, default_value_t = 1)]
    repeats: u32,

    /// Comma-separated trial modes: `baseline`, `mtp_default`,
    /// `prompt_lookup`, `prompt_lookup_unchecked`, or numeric MTP draft
    /// depths such as `1,2,3`.
    #[arg(long, default_value = "baseline,1,2,3")]
    trials: String,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long, default_value_t = 300)]
    startup_timeout_secs: u64,

    #[arg(long, default_value_t = 600)]
    request_timeout_secs: u64,

    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
}

#[derive(Debug, Serialize)]
struct Params {
    host: String,
    port: u16,
    model_key: Option<String>,
    model_ref: String,
    request_model: String,
    max_tokens: u32,
    repeats: u32,
    trials: Vec<speculative::TrialSpec>,
    prompt: String,
    startup_timeout_secs: u64,
    request_timeout_secs: u64,
}

#[derive(Debug, Serialize, Clone)]
struct TrialRun {
    elapsed_s: f64,
    completion_tokens: u32,
    tok_s: f64,
    content_prefix: String,
    telemetry: String,
}

#[derive(Debug, Serialize)]
struct TrialSummary {
    label: String,
    runs: Vec<TrialRun>,
    elapsed_s_mean: f64,
    tok_s_mean: f64,
    tok_s_median: f64,
    tok_s_p95: f64,
    tok_s_stdev: f64,
    speedup_vs_baseline: Option<f64>,
}

#[derive(Debug, Serialize)]
struct Results {
    trials: Vec<TrialSummary>,
}

struct ResolvedModel {
    key: Option<String>,
    serve_path: String,
    public_ref: String,
    request_model: String,
    metadata: ModelInfo,
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
    if args.repeats == 0 {
        anyhow::bail!("--repeats must be >= 1");
    }

    let mut metadata = RunMetadata::capture("bench_speculative");
    let started = Instant::now();

    let manifest_path = args.manifest.clone().unwrap_or_else(default_manifest_path);
    let model = resolve_model(&args, &manifest_path)?;
    metadata.model = Some(model.metadata.clone());

    let trial_specs = speculative::parse_trial_specs(&args.trials)?;
    let prompt = args
        .prompt
        .clone()
        .unwrap_or_else(|| DEFAULT_PROMPT.to_owned());
    let base_url = format!("http://{}:{}", args.host, args.port);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(args.request_timeout_secs))
        .build()
        .context("build request client")?;

    let mut summaries = Vec::with_capacity(trial_specs.len());
    let mut baseline_tok_s: Option<f64> = None;

    for spec in &trial_specs {
        let mut runs = Vec::with_capacity(args.repeats as usize);
        for repeat_idx in 0..args.repeats {
            eprintln!(
                "[{} repeat {}/{}]",
                spec.label,
                repeat_idx + 1,
                args.repeats
            );
            let run =
                run_trial(&args, spec, repeat_idx, &client, &base_url, &model, &prompt).await?;
            runs.push(run);
        }

        let tok_s: Vec<f64> = runs.iter().map(|run| run.tok_s).collect();
        let elapsed: Vec<f64> = runs.iter().map(|run| run.elapsed_s).collect();
        let tok_s_mean = stats::mean(&tok_s);
        if spec.label == "baseline_mtp_off" {
            baseline_tok_s = Some(tok_s_mean);
        }
        let speedup_vs_baseline = baseline_tok_s
            .filter(|baseline| *baseline > 0.0)
            .map(|baseline| tok_s_mean / baseline);

        summaries.push(TrialSummary {
            label: spec.label.clone(),
            runs,
            elapsed_s_mean: stats::mean(&elapsed),
            tok_s_mean,
            tok_s_median: stats::median(&tok_s),
            tok_s_p95: stats::p95(&tok_s),
            tok_s_stdev: stats::stdev(&tok_s),
            speedup_vs_baseline,
        });
    }

    metadata.duration_ms = started.elapsed().as_millis() as u64;

    let params = Params {
        host: args.host,
        port: args.port,
        model_key: model.key,
        model_ref: model.public_ref,
        request_model: model.request_model,
        max_tokens: args.max_tokens,
        repeats: args.repeats,
        trials: trial_specs,
        prompt,
        startup_timeout_secs: args.startup_timeout_secs,
        request_timeout_secs: args.request_timeout_secs,
    };

    let output = BenchOutput {
        metadata,
        params,
        results: Results { trials: summaries },
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

#[allow(clippy::too_many_arguments)]
async fn run_trial(
    args: &Args,
    spec: &speculative::TrialSpec,
    repeat_idx: u32,
    client: &reqwest::Client,
    base_url: &str,
    model: &ResolvedModel,
    prompt: &str,
) -> Result<TrialRun> {
    let log_path = log_path(&spec.label, repeat_idx)?;
    let mut child = start_higgs_server(args, spec, &model.serve_path, &log_path)
        .with_context(|| format!("start higgs server for trial {}", spec.label))?;

    let result = async {
        server::wait_until_ready(base_url, Duration::from_secs(args.startup_timeout_secs))
            .await
            .with_context(|| format!("higgs server not ready at {base_url}"))?;
        request_completion(
            client,
            base_url,
            &model.request_model,
            prompt,
            args.max_tokens,
        )
        .await
    }
    .await;

    stop_child(&mut child);
    let telemetry = read_filtered_telemetry(&log_path);
    let payload = result?;
    let elapsed_s = payload.elapsed.as_secs_f64();
    let tok_s = if elapsed_s > 0.0 {
        f64::from(payload.completion_tokens) / elapsed_s
    } else {
        0.0
    };

    Ok(TrialRun {
        elapsed_s,
        completion_tokens: payload.completion_tokens,
        tok_s,
        content_prefix: payload.content.chars().take(120).collect(),
        telemetry,
    })
}

struct CompletionPayload {
    elapsed: Duration,
    completion_tokens: u32,
    content: String,
}

async fn request_completion(
    client: &reqwest::Client,
    base_url: &str,
    model_name: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<CompletionPayload> {
    let body = serde_json::json!({
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "reasoning": { "effort": "none" },
        "max_tokens": max_tokens,
        "stream": false,
    });

    let url = format!("{base_url}/v1/chat/completions");
    let started = Instant::now();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;
    let elapsed = started.elapsed();

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("{url} returned HTTP {status}: {text}");
    }

    let payload: serde_json::Value = resp
        .json()
        .await
        .context("parse completion response JSON")?;
    let completion_tokens = payload
        .get("usage")
        .and_then(|usage| usage.get("completion_tokens"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|tokens| u32::try_from(tokens).ok())
        .unwrap_or(0);
    let content = payload
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned();

    Ok(CompletionPayload {
        elapsed,
        completion_tokens,
        content,
    })
}

fn resolve_model(args: &Args, manifest_path: &Path) -> Result<ResolvedModel> {
    match (&args.model, &args.model_path) {
        (Some(key), None) => {
            let model = models::find_by_key(manifest_path, key)?;
            let request_model = args
                .model_name
                .clone()
                .unwrap_or_else(|| speculative::derive_model_name(&model.path));
            let public_ref = public_model_ref(&model.path, &request_model);
            Ok(ResolvedModel {
                key: Some(model.key.clone()),
                serve_path: model.path.clone(),
                public_ref: public_ref.clone(),
                request_model,
                metadata: ModelInfo {
                    key: model.key,
                    path: public_ref,
                    quantization: model.quantization,
                    approx_size_gb: model.approx_size_gb,
                },
            })
        }
        (None, Some(path)) => {
            let request_model = args
                .model_name
                .clone()
                .unwrap_or_else(|| speculative::derive_model_name(path));
            let public_ref = public_model_ref(path, &request_model);
            Ok(ResolvedModel {
                key: None,
                serve_path: path.clone(),
                public_ref: public_ref.clone(),
                request_model,
                metadata: ModelInfo {
                    key: "direct".to_owned(),
                    path: public_ref,
                    quantization: "unknown".to_owned(),
                    approx_size_gb: 0.0,
                },
            })
        }
        (None, None) => anyhow::bail!("pass either --model <key> or --model-path <path>"),
        (Some(_), Some(_)) => anyhow::bail!("pass only one of --model or --model-path"),
    }
}

fn start_higgs_server(
    args: &Args,
    spec: &speculative::TrialSpec,
    model_path: &str,
    log_path: &Path,
) -> Result<Child> {
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create log dir {}", parent.display()))?;
    }
    let log = fs::File::create(log_path)
        .with_context(|| format!("create server log {}", log_path.display()))?;
    let log_stderr = log
        .try_clone()
        .with_context(|| format!("clone server log {}", log_path.display()))?;

    let mut cmd = Command::new(&args.higgs_bin);
    cmd.arg("serve")
        .arg("--model")
        .arg(model_path)
        .arg("--host")
        .arg(&args.host)
        .arg("--port")
        .arg(args.port.to_string())
        .arg("--mlx-profile")
        .arg("throughput")
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_stderr));

    clear_speculative_env(&mut cmd);
    for (key, value) in &spec.env {
        cmd.env(key, value);
    }

    cmd.spawn()
        .with_context(|| format!("spawn {}", args.higgs_bin.display()))
}

fn stop_child(child: &mut Child) {
    if matches!(child.try_wait(), Ok(Some(_))) {
        return;
    }
    if let Err(e) = child.kill() {
        eprintln!("warning: failed to stop higgs server: {e}");
    }
    if let Err(e) = child.wait() {
        eprintln!("warning: failed to wait for higgs server exit: {e}");
    }
}

fn log_path(label: &str, repeat_idx: u32) -> Result<PathBuf> {
    let logs_dir = results_dir().join("bench_speculative").join("logs");
    fs::create_dir_all(&logs_dir)
        .with_context(|| format!("create log dir {}", logs_dir.display()))?;
    let ts = chrono::Utc::now().format("%Y%m%dT%H%M%S%.3fZ");
    Ok(logs_dir.join(format!(
        "{}__{}__{}.log",
        sanitize_filename(label),
        repeat_idx,
        ts
    )))
}

fn sanitize_filename(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn clear_speculative_env(cmd: &mut Command) {
    for key in SPECULATIVE_ENV_KEYS {
        cmd.env_remove(key);
    }
}

const SPECULATIVE_ENV_KEYS: &[&str] = &[
    "HIGGS_MTP",
    "HIGGS_MTP_DRAFT_N_MAX",
    "HIGGS_PROMPT_LOOKUP",
    "HIGGS_PROMPT_LOOKUP_UNCHECKED",
    "HIGGS_MTP_PRIME_PREFILL",
    "HIGGS_MTP_MIRROR_VERIFY",
];

fn read_filtered_telemetry(path: &Path) -> String {
    let Ok(body) = fs::read_to_string(path) else {
        return String::new();
    };
    body.lines()
        .filter(|line| {
            line.contains("MTP decode complete") || line.contains("Prompt-lookup decode complete")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::{SPECULATIVE_ENV_KEYS, clear_speculative_env};
    use std::process::Command;

    #[test]
    fn clear_speculative_env_marks_all_flags_for_removal() {
        let mut cmd = Command::new("higgs");

        clear_speculative_env(&mut cmd);

        for key in SPECULATIVE_ENV_KEYS {
            assert!(
                cmd.get_envs()
                    .any(|(name, value)| name == *key && value.is_none()),
                "expected {key} to be explicitly removed"
            );
        }
    }
}
