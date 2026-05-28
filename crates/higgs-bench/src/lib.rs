#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Shared infrastructure for higgs end-to-end benches.
//!
//! Every bench binary in this crate produces output that conforms to the
//! `BenchOutput<P, R>` schema: a `metadata` block describing the run host
//! and git state, a `params` block describing the bench inputs, and a
//! `results` block with the measurements. See `docs/benchmarking.md`.

pub mod models;
pub mod server;
pub mod speculative;
pub mod stats;

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sysinfo::System;

#[allow(clippy::needless_raw_string_hashes, clippy::doc_markdown)]
mod built_info {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

/// Bench-crate version (from `CARGO_PKG_VERSION` at compile time).
pub const BENCH_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the short git commit hash captured at compile time.
#[must_use]
pub fn git_commit_short() -> String {
    built_info::GIT_COMMIT_HASH_SHORT.map_or_else(|| "unknown".into(), str::to_owned)
}

/// Returns the full git commit hash captured at compile time.
#[must_use]
pub fn git_commit() -> String {
    built_info::GIT_COMMIT_HASH.map_or_else(|| "unknown".into(), str::to_owned)
}

/// Returns whether the working tree was dirty at compile time.
#[must_use]
pub fn git_dirty() -> bool {
    built_info::GIT_DIRTY.unwrap_or(false)
}

/// Information about the host machine where the benchmark ran.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostInfo {
    /// Machine hostname.
    pub hostname: String,
    /// Operating system name, version, and kernel.
    pub os: String,
    /// CPU brand string.
    pub cpu: String,
    /// Total RAM in gigabytes.
    pub ram_gb: f64,
    /// GPU identifier, if detected.
    pub gpu: Option<String>,
}

impl HostInfo {
    #[must_use]
    pub fn capture() -> Self {
        let mut sys = System::new();
        sys.refresh_memory();
        sys.refresh_cpu_all();

        let hostname = hostname_for_output(
            System::host_name().unwrap_or_else(|| "unknown".into()),
            include_hostname(),
        );
        let os_name = System::name().unwrap_or_else(|| "unknown".into());
        let os_version = System::os_version().unwrap_or_else(|| "?".into());
        let kernel = System::kernel_version().unwrap_or_else(|| "?".into());
        let os = format!("{os_name} {os_version} ({kernel})");

        let cpu = sys
            .cpus()
            .first()
            .map_or_else(|| "unknown".into(), |c| c.brand().trim().to_owned());

        // sysinfo's `total_memory()` returns bytes since 0.30.
        let total_bytes = sys.total_memory();
        let ram_gb = (total_bytes as f64) / 1_073_741_824.0_f64;

        Self {
            hostname,
            os,
            cpu,
            ram_gb: round2(ram_gb),
            gpu: detect_gpu(),
        }
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn detect_gpu() -> Option<String> {
    // sysinfo doesn't expose GPU info portably; only label macOS+aarch64 as
    // Apple Silicon. Intel Macs return None rather than getting mislabeled.
    (cfg!(target_os = "macos") && std::env::consts::ARCH == "aarch64")
        .then(|| "Apple Silicon (MLX)".into())
}

fn include_hostname() -> bool {
    matches!(
        std::env::var("HIGGS_BENCH_INCLUDE_HOSTNAME")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn hostname_for_output(hostname: String, include_hostname: bool) -> String {
    if include_hostname {
        hostname
    } else {
        "redacted".to_owned()
    }
}

fn redact_arg_for_output(arg: &str) -> String {
    if let Some((prefix, value)) = arg.split_once('=') {
        if is_local_path_like(value) {
            return format!("{prefix}={}", redacted_local_path_value(value));
        }
    }

    if is_local_path_like(arg) {
        redacted_local_path_value(arg)
    } else {
        arg.to_owned()
    }
}

fn is_local_path_like(value: &str) -> bool {
    let path = Path::new(value);
    path.is_absolute()
        || value.starts_with("./")
        || value.starts_with("../")
        || value.starts_with("~/")
        || value.get(1..3) == Some(":\\")
}

fn redacted_local_path_value(value: &str) -> String {
    if looks_like_hf_cache_path(value) {
        let model_name = speculative::derive_model_name(value);
        if model_name != value && !is_local_path_like(&model_name) {
            return model_name;
        }
    }

    let trimmed = value.trim_end_matches(['/', '\\']);
    let basename = Path::new(trimmed)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("path");
    format!("<local-path:{basename}>")
}

fn looks_like_hf_cache_path(value: &str) -> bool {
    (value.contains("/models--") || value.contains("\\models--"))
        && (value.contains("/snapshots/") || value.contains("\\snapshots\\"))
}

/// Returns a model reference suitable for benchmark output.
///
/// Hugging Face repo IDs are preserved. Local paths are reduced to either the
/// explicit request model name or a derived basename/cache repo so benchmark
/// JSON and markdown do not expose a developer's filesystem layout.
#[must_use]
pub fn public_model_ref(path: &str, request_model: &str) -> String {
    if is_local_path_like(path) {
        if request_model.is_empty() {
            speculative::derive_model_name(path)
        } else {
            request_model.to_owned()
        }
    } else {
        path.to_owned()
    }
}

/// Formats generated artifact paths without exposing the absolute workspace.
#[must_use]
pub fn path_for_output(path: &Path) -> String {
    if let Some(root) = workspace_root() {
        if let Ok(relative) = path.strip_prefix(root) {
            return relative.display().to_string();
        }
    }
    redact_arg_for_output(&path.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::{hostname_for_output, public_model_ref, redact_arg_for_output};

    #[test]
    fn benchmark_metadata_redacts_hostname_by_default() {
        assert_eq!(
            hostname_for_output("developer-laptop".to_owned(), false),
            "redacted"
        );
    }

    #[test]
    fn benchmark_metadata_redacts_absolute_path_args() {
        assert_eq!(
            redact_arg_for_output("/Users/alice/models/Qwen3.6-27B-mtp"),
            "<local-path:Qwen3.6-27B-mtp>"
        );
        assert_eq!(
            redact_arg_for_output("--manifest=/Users/alice/dev/higgs/benchmarks/models.toml"),
            "--manifest=<local-path:models.toml>"
        );
    }

    #[test]
    fn model_refs_hide_local_absolute_paths() {
        assert_eq!(
            public_model_ref(
                "/Users/alice/.cache/huggingface/hub/models--org--Qwen3.6-27B-mtp/snapshots/abcdef",
                ""
            ),
            "org/Qwen3.6-27B-mtp"
        );
        assert_eq!(
            public_model_ref("/Users/alice/models/private-qwen", "local-qwen"),
            "local-qwen"
        );
    }
}

/// The model under test, captured into bench output for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Manifest key (e.g., `qwen3-1.7B-4bit`).
    pub key: String,
    /// Model path used (`HuggingFace` repo ID or local path).
    pub path: String,
    /// Quantization format (e.g., `4bit`).
    pub quantization: String,
    /// Approximate model size in gigabytes.
    pub approx_size_gb: f64,
}

/// Reproducibility metadata recorded with every bench run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    /// Bench binary name (e.g., `bench_decode`).
    pub bench_name: String,
    /// `higgs-bench` crate version.
    pub bench_version: String,
    /// `higgs` server version under test, when known.
    pub higgs_version: Option<String>,
    /// Full git commit captured at compile time.
    pub git_commit: String,
    /// Short git commit (first 7 chars).
    pub git_commit_short: String,
    /// Whether the working tree was dirty at compile time.
    pub git_dirty: bool,
    /// RFC3339 UTC timestamp when the bench run started.
    pub started_at: DateTime<Utc>,
    /// Wall-clock time of the entire bench run, in milliseconds.
    pub duration_ms: u64,
    /// Host machine information (CPU, RAM, OS, GPU).
    pub host: HostInfo,
    /// MLX runtime version when known.
    pub mlx_version: Option<String>,
    /// Model under test; `None` for benches that don't pin to a model.
    pub model: Option<ModelInfo>,
    /// Full argv that produced this run (used for the "How to reproduce"
    /// block in `--format markdown`).
    pub args: Vec<String>,
}

impl RunMetadata {
    /// Snapshots host + git + argv at bench startup. Caller fills in
    /// `duration_ms`, `model`, and `higgs_version` once they're known.
    #[must_use]
    pub fn capture<S: Into<String>>(bench_name: S) -> Self {
        Self {
            bench_name: bench_name.into(),
            bench_version: BENCH_VERSION.into(),
            higgs_version: None,
            git_commit: git_commit(),
            git_commit_short: git_commit_short(),
            git_dirty: git_dirty(),
            started_at: Utc::now(),
            duration_ms: 0,
            host: HostInfo::capture(),
            mlx_version: None,
            model: None,
            args: std::env::args()
                .map(|arg| redact_arg_for_output(&arg))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Standardized output envelope produced by every bench binary.
pub struct BenchOutput<P, R>
where
    P: Serialize,
    R: Serialize,
{
    /// Reproducibility metadata (host, model, git, argv, timing).
    pub metadata: RunMetadata,
    /// Bench-specific parameters (CLI flags, prompt, etc.).
    pub params: P,
    /// Bench-specific results (numbers, derived stats).
    pub results: R,
}

/// Serializes the output as pretty JSON.
pub fn format_json<P, R>(output: &BenchOutput<P, R>) -> Result<String>
where
    P: Serialize,
    R: Serialize,
{
    serde_json::to_string_pretty(output).context("serialize bench output to JSON")
}

/// Renders a human-readable Markdown report.
///
/// The results are rendered as a one-row-per-key table from a JSON
/// flattening of `results` — benches that want richer tables should
/// produce their own `Markdown` and return it here.
pub fn format_markdown<P, R>(output: &BenchOutput<P, R>) -> Result<String>
where
    P: Serialize,
    R: Serialize,
{
    use std::fmt::Write as _;
    let meta = &output.metadata;
    let mut s = String::new();
    writeln!(s, "# {} run\n", meta.bench_name)?;

    s.push_str("## How to reproduce\n\n");
    s.push_str("```bash\n");
    s.push_str(&shell_quote_argv(&meta.args));
    s.push_str("\n```\n\n");

    s.push_str("## Environment\n\n");
    s.push_str("| Field | Value |\n|---|---|\n");
    writeln!(s, "| host | {} |", meta.host.hostname)?;
    writeln!(s, "| os | {} |", meta.host.os)?;
    writeln!(s, "| cpu | {} |", meta.host.cpu)?;
    writeln!(s, "| ram_gb | {} |", meta.host.ram_gb)?;
    if let Some(gpu) = &meta.host.gpu {
        writeln!(s, "| gpu | {gpu} |")?;
    }
    writeln!(
        s,
        "| git_commit | {}{} |",
        meta.git_commit_short,
        if meta.git_dirty { " (dirty)" } else { "" }
    )?;
    writeln!(s, "| started_at | {} |", meta.started_at.to_rfc3339())?;
    writeln!(s, "| duration_ms | {} |", meta.duration_ms)?;
    if let Some(model) = &meta.model {
        writeln!(
            s,
            "| model | {} ({}, ~{} GB) |",
            model.key, model.quantization, model.approx_size_gb
        )?;
    }
    s.push('\n');

    s.push_str("## Params\n\n");
    s.push_str("```json\n");
    s.push_str(&serde_json::to_string_pretty(&output.params)?);
    s.push_str("\n```\n\n");

    s.push_str("## Results\n\n");
    let results_json = serde_json::to_value(&output.results)?;
    if let Some(map) = results_json.as_object() {
        s.push_str("| Metric | Value |\n|---|---|\n");
        for (k, v) in map {
            writeln!(s, "| {k} | {} |", render_json_value(v))?;
        }
    } else {
        s.push_str("```json\n");
        s.push_str(&serde_json::to_string_pretty(&results_json)?);
        s.push_str("\n```\n");
    }

    Ok(s)
}

fn render_json_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".into(),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => v.to_string(),
    }
}

fn shell_quote_argv(args: &[String]) -> String {
    args.iter()
        .map(|a| {
            if a.chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_./=".contains(c))
            {
                a.clone()
            } else {
                format!("'{}'", a.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Persists the bench output to
/// `target/bench-results/<bench_name>/<commit>__<model>__<ts>.json`.
/// Returns the path written.
pub fn persist_result<P, R>(output: &BenchOutput<P, R>) -> Result<PathBuf>
where
    P: Serialize,
    R: Serialize,
{
    let dir = results_dir().join(&output.metadata.bench_name);
    fs::create_dir_all(&dir).with_context(|| format!("create results dir {}", dir.display()))?;

    // Sanitize the model key before using it in the path: `--manifest`
    // could feed in a key containing `/` or `..` and the filename gets
    // joined into `target/bench-results/<bench>/`. Map anything outside
    // [A-Za-z0-9._-] to `_`.
    let model_key = output.metadata.model.as_ref().map_or_else(
        || "no-model".to_owned(),
        |m| {
            m.key
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                        c
                    } else {
                        '_'
                    }
                })
                .collect()
        },
    );
    // Include subsecond resolution so same-second reruns of the same
    // (commit, model, bench) don't collide and overwrite each other.
    let ts = format!(
        "{}-{:03}",
        output.metadata.started_at.format("%Y%m%dT%H%M%SZ"),
        output.metadata.started_at.timestamp_subsec_millis()
    );
    let filename = format!(
        "{}__{}__{}.json",
        output.metadata.git_commit_short, model_key, ts
    );
    let path = dir.join(filename);

    let json = serde_json::to_string_pretty(output)?;
    fs::write(&path, json).with_context(|| format!("write result file {}", path.display()))?;
    Ok(path)
}

/// Returns the absolute path to `<workspace>/target/bench-results/`. Falls
/// back to `target/bench-results/` relative to the current directory if
/// the workspace root cannot be located.
#[must_use]
pub fn results_dir() -> PathBuf {
    workspace_root()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("target")
        .join("bench-results")
}

fn workspace_root() -> Option<PathBuf> {
    // CARGO_MANIFEST_DIR points at this crate; the workspace root is two
    // levels up (crates/higgs-bench → ..).
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate = manifest_dir.parent()?.parent()?;
    candidate
        .join("Cargo.toml")
        .exists()
        .then(|| candidate.to_owned())
}

/// Looks up a model entry from the workspace `benchmarks/models.toml`.
/// Convenience wrapper used by binary entrypoints.
pub fn load_default_manifest() -> Result<models::Manifest> {
    let path = workspace_root()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("benchmarks")
        .join("models.toml");
    models::load_manifest(&path)
}

/// Returns the path to the workspace `benchmarks/models.toml`.
#[must_use]
pub fn default_manifest_path() -> PathBuf {
    workspace_root()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("benchmarks")
        .join("models.toml")
}

/// Convenience: write `JSON` or `markdown` to stdout depending on `format`.
pub fn render<P, R>(output: &BenchOutput<P, R>, format: OutputFormat) -> Result<String>
where
    P: Serialize,
    R: Serialize,
{
    match format {
        OutputFormat::Json => format_json(output),
        OutputFormat::Markdown => format_markdown(output),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    Json,
    Markdown,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json => f.write_str("json"),
            Self::Markdown => f.write_str("markdown"),
        }
    }
}

/// Walks `target/bench-results/` and returns one entry per `.json` file.
pub fn collect_results(root: &Path) -> Result<Vec<StoredResult>> {
    let mut out = Vec::new();
    if !root.exists() {
        return Ok(out);
    }
    for bench_entry_res in fs::read_dir(root)? {
        let Ok(bench_entry) = bench_entry_res else {
            continue;
        };
        if !bench_entry.file_type()?.is_dir() {
            continue;
        }
        for file_entry_res in fs::read_dir(bench_entry.path())? {
            let Ok(file_entry) = file_entry_res else {
                continue;
            };
            let path = file_entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let body = fs::read_to_string(&path)?;
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) {
                out.push(StoredResult { path, value });
            }
        }
    }
    Ok(out)
}

/// One row of a persisted bench result (raw JSON).
#[derive(Debug, Clone)]
/// One persisted JSON result, returned by `collect_results`.
pub struct StoredResult {
    /// Path to the JSON file under `target/bench-results/<bench>/...`.
    pub path: PathBuf,
    /// Parsed JSON content.
    pub value: serde_json::Value,
}
