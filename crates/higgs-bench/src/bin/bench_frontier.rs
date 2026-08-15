#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::shadow_reuse,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
//! In-process context-frontier benchmark for decode throughput and KV memory.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use higgs_bench::{
    BENCH_SCHEMA_VERSION, BenchOutput, ModelInfo, OutputFormat, RunMetadata, format_json,
    format_markdown, persist_result, public_model_ref,
};
use higgs_engine::{
    model_loader::{load_model, load_tokenizer},
    tokenizers,
};
use higgs_models::{AnyCache, turboquant::KvCacheConfig};
use mlx_rs::{Array, argmax_axis, transforms::eval};
use serde::Serialize;

const DEFAULT_FRONTIERS: &str = "2048,4096,8192,16384";
const PREFILL_CHUNK_SIZE: i32 = 512;
const CORPUS: &str = "The benchmark corpus is deliberately ordinary prose. It creates stable tokenized context while exercising attention cache growth. Local inference performance depends on both cache layout and the cost of reading it during decode. ";

#[derive(Debug, Parser)]
#[command(
    name = "bench_frontier",
    about = "Measure decode and KV memory at context frontiers",
    version
)]
struct Args {
    /// Local model directory (not a benchmark manifest key).
    #[arg(long)]
    model_dir: PathBuf,

    /// Ascending comma-separated context lengths in tokens.
    #[arg(long, default_value = DEFAULT_FRONTIERS)]
    frontiers: String,

    /// Greedy decode tokens sampled at each frontier (32-128).
    #[arg(long, default_value_t = 64)]
    probe_tokens: usize,

    /// Number of complete frontier sweeps.
    #[arg(long, default_value_t = 1)]
    runs: usize,

    /// Verify the first KV frontier against an expected bytes-per-token value.
    #[arg(long)]
    verify_kv_analytic: bool,

    /// Expected dense-KV bytes per token; required with --verify-kv-analytic.
    #[arg(long)]
    expect_kv_bytes_per_token: Option<f64>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
}

#[derive(Debug, Serialize)]
struct Params {
    model_dir: String,
    frontiers: Vec<usize>,
    probe_tokens: usize,
    runs: usize,
    verify_kv_analytic: bool,
    expect_kv_bytes_per_token: Option<f64>,
}

#[derive(Debug, Serialize, Clone)]
struct FrontierRow {
    run: usize,
    frontier: usize,
    incremental_prefill_ms: f64,
    prefill_tokps: f64,
    probe_decode_tokps: f64,
    kv_bytes: usize,
    kv_bytes_per_token: f64,
}

#[derive(Debug, Serialize)]
struct Results {
    rows: Vec<FrontierRow>,
    analytic_verified: Option<bool>,
}

fn main() -> ExitCode {
    match run(&Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(1)
        }
    }
}

fn run(args: &Args) -> Result<()> {
    if !(32..=128).contains(&args.probe_tokens) {
        anyhow::bail!("--probe-tokens must be in 32..=128");
    }
    if args.runs == 0 {
        anyhow::bail!("--runs must be >= 1");
    }
    let frontiers = parse_frontiers(&args.frontiers)?;
    let expected_bpt = match (args.verify_kv_analytic, args.expect_kv_bytes_per_token) {
        (true, Some(value)) if value > 0.0 => Some(value),
        (true, _) => {
            anyhow::bail!("--verify-kv-analytic requires positive --expect-kv-bytes-per-token")
        }
        (false, _) => None,
    };

    let mut metadata = RunMetadata::capture("bench_frontier");
    let started = Instant::now();
    let model_path = args.model_dir.to_string_lossy().into_owned();
    metadata.model = Some(ModelInfo {
        key: args
            .model_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("local-model")
            .to_owned(),
        path: public_model_ref(&model_path, ""),
        quantization: "local".to_owned(),
        approx_size_gb: 0.0,
    });

    let mut model = load_model(&args.model_dir).context("load model")?;
    let tokenizer = load_tokenizer(&args.model_dir).context("load tokenizer")?;
    let prompt_tokens =
        synthetic_prompt(&tokenizer, *frontiers.last().expect("frontiers validated"))?;
    let mut rows = Vec::with_capacity(args.runs * frontiers.len());

    for run_index in 0..args.runs {
        let mut cache = model
            .make_cache_with_config(KvCacheConfig::default())
            .context("create KV cache")?;
        let mut previous = 0;
        for &frontier in &frontiers {
            let incremental_tokens = frontier - previous;
            eval(cache.eval_targets()).context("evaluate cache before prefill timing")?;
            let prefill_started = Instant::now();
            let logits = model
                .forward_chunked(
                    &token_array(&prompt_tokens[previous..frontier])?,
                    &mut cache,
                    PREFILL_CHUNK_SIZE,
                )
                .context("incremental prefill")?;
            eval([&logits]).context("evaluate prefill logits")?;
            let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1_000.0;
            previous = frontier;

            let kv_bytes = cache.eval_targets().into_iter().map(Array::nbytes).sum();
            let checkpoint = cache.checkpoint_for_rollback();
            let probe_tokps = decode_probe(&mut model, &mut cache, logits, args.probe_tokens)?;
            cache.rollback(checkpoint, args.probe_tokens);

            rows.push(FrontierRow {
                run: run_index + 1,
                frontier,
                incremental_prefill_ms: prefill_ms,
                prefill_tokps: incremental_tokens as f64 / (prefill_ms / 1_000.0),
                probe_decode_tokps: probe_tokps,
                kv_bytes,
                kv_bytes_per_token: kv_bytes as f64 / frontier as f64,
            });
        }
    }

    let first_row = rows.first().context("frontier sweep produced no rows")?;
    let analytic_verified = expected_bpt
        .map(|expected| verify_kv_bytes(first_row.kv_bytes, expected, first_row.frontier))
        .transpose()?
        .map(|()| true);
    metadata.duration_ms = started.elapsed().as_millis() as u64;
    let output = BenchOutput {
        schema_version: BENCH_SCHEMA_VERSION,
        metadata,
        params: Params {
            model_dir: public_model_ref(&model_path, ""),
            frontiers,
            probe_tokens: args.probe_tokens,
            runs: args.runs,
            verify_kv_analytic: args.verify_kv_analytic,
            expect_kv_bytes_per_token: args.expect_kv_bytes_per_token,
        },
        results: Results {
            rows,
            analytic_verified,
        },
    };
    let path = persist_result(&output)?;
    eprintln!("[persisted] {}", higgs_bench::path_for_output(&path));
    println!(
        "{}",
        match args.format {
            OutputFormat::Json => format_json(&output)?,
            OutputFormat::Markdown => format_frontier_markdown(&output)?,
        }
    );
    Ok(())
}

fn synthetic_prompt(tokenizer: &tokenizers::Tokenizer, maximum: usize) -> Result<Vec<u32>> {
    let corpus = tokenizer
        .encode(CORPUS, false)
        .map_err(|error| anyhow::anyhow!("tokenize synthetic corpus: {error}"))?;
    let ids = corpus.get_ids();
    if ids.is_empty() {
        anyhow::bail!("synthetic corpus tokenized to zero tokens");
    }
    Ok(ids.iter().copied().cycle().take(maximum).collect())
}

fn token_array(tokens: &[u32]) -> Result<Array> {
    let tokens: Vec<i32> = tokens
        .iter()
        .copied()
        .map(|id| i32::try_from(id).context("token id exceeds i32 range"))
        .collect::<Result<_>>()?;
    let length = i32::try_from(tokens.len()).context("token array exceeds i32 dimensions")?;
    Ok(Array::from_slice(&tokens, &[1, length]))
}

fn decode_probe(
    model: &mut higgs_models::AnyModel,
    cache: &mut AnyCache,
    mut logits: Array,
    probe_tokens: usize,
) -> Result<f64> {
    let mut elapsed = 0.0;
    for step in 0..probe_tokens {
        let token = argmax_axis!(&logits, -1).context("argmax greedy token")?;
        eval([&token]).context("evaluate greedy token")?;
        let input = token_array(&[token.item::<u32>()])?;
        let started = Instant::now();
        logits = model
            .forward_last_token(&input, None, cache)
            .context("decode probe step")?;
        eval([&logits]).context("evaluate decode logits")?;
        if step > 0 {
            elapsed += started.elapsed().as_secs_f64();
        }
    }
    Ok((probe_tokens - 1) as f64 / elapsed)
}

fn parse_frontiers(raw: &str) -> Result<Vec<usize>> {
    let values: Vec<usize> = raw
        .split(',')
        .map(str::trim)
        .map(|value| {
            value
                .parse::<usize>()
                .with_context(|| format!("invalid frontier {value:?}"))
        })
        .collect::<Result<_>>()?;
    if values.is_empty() || values.contains(&0) {
        anyhow::bail!("--frontiers must contain positive lengths");
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        anyhow::bail!("--frontiers must be strictly ascending");
    }
    Ok(values)
}

fn verify_kv_bytes(measured: usize, bytes_per_token: f64, frontier: usize) -> Result<()> {
    let expected = bytes_per_token * frontier as f64;
    let relative_error = (measured as f64 - expected).abs() / expected;
    if relative_error > 0.10 {
        anyhow::bail!(
            "KV analytic check failed: measured {measured} bytes, expected {expected:.0} bytes ({:.1}% error)",
            relative_error * 100.0
        );
    }
    Ok(())
}

fn format_frontier_markdown(output: &BenchOutput<Params, Results>) -> Result<String> {
    let mut markdown = format_markdown(output)?;
    markdown.push_str("\n\n## Frontier rows\n\n| Run | Frontier | Incremental prefill ms | Prefill tok/s | Probe decode tok/s | KV bytes | KV bytes/token |\n|---:|---:|---:|---:|---:|---:|---:|\n");
    for row in &output.results.rows {
        writeln!(
            markdown,
            "| {} | {} | {:.3} | {:.2} | {:.2} | {} | {:.2} |",
            row.run,
            row.frontier,
            row.incremental_prefill_ms,
            row.prefill_tokps,
            row.probe_decode_tokps,
            row.kv_bytes,
            row.kv_bytes_per_token
        )?;
    }
    Ok(markdown)
}

#[cfg(test)]
mod tests {
    use super::{parse_frontiers, verify_kv_bytes};

    #[test]
    fn parses_sorted_unique_frontiers() {
        assert_eq!(
            parse_frontiers("2048,4096,8192").unwrap(),
            vec![2048, 4096, 8192]
        );
        assert!(parse_frontiers("4096,2048").is_err());
        assert!(parse_frontiers("2048,2048").is_err());
    }

    #[test]
    fn accepts_kv_measurement_within_ten_percent() {
        assert!(verify_kv_bytes(109, 1.0, 100).is_ok());
        assert!(verify_kv_bytes(111, 1.0, 100).is_err());
    }
}
