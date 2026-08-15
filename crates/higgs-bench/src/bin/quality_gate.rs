#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::unwrap_used
)]
//! Deterministic model-output regression gate for local MLX checkpoints.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use higgs_engine::model_loader::{load_model, load_tokenizer};
use higgs_models::{AnyModel, turboquant::KvCacheConfig};
use mlx_rs::{Array, Dtype, ops::indexing::IndexOp, transforms::eval};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: u32 = 1;
const PREFILL_CHUNK_SIZE: i32 = 512;

#[derive(Debug, Parser)]
#[command(
    name = "quality_gate",
    about = "Record and check greedy model quality fixtures"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Record {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        prompts: PathBuf,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        #[arg(long)]
        out: PathBuf,
    },
    Check {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long, default_value_t = 0.0)]
        #[arg(
            long_help = "Maximum absolute logprob delta. At 0.0 (strict/default), each prompt passes only when tokens are exact and the delta is exactly 0.0; use this for same-numerics A/A checks, where MLX greedy replay is deterministic on one machine. Above 0.0, a prompt passes when tokens are exact or the delta is within tolerance."
        )]
        logprob_tolerance: f32,
        #[arg(long)]
        perturb_logits: Option<f32>,
        #[arg(long)]
        allow_model_mismatch: bool,
    },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum PromptFile {
    List(Vec<String>),
    Object { prompts: Vec<String> },
}
impl PromptFile {
    fn into_prompts(self) -> Vec<String> {
        match self {
            Self::List(p) | Self::Object { prompts: p } => p,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Fixture {
    schema_version: u32,
    model_basename: String,
    config_json_sha256: String,
    prompts: Vec<PromptRecord>,
}
#[derive(Debug, Serialize, Deserialize)]
struct PromptRecord {
    prompt: String,
    token_ids: Vec<u32>,
    logprobs: Vec<f32>,
}
#[derive(Debug, Serialize)]
struct CheckSummary {
    schema_version: u32,
    model_basename: String,
    config_json_sha256: String,
    logprob_tolerance: f32,
    passed: bool,
    prompts: Vec<PromptSummary>,
}
#[derive(Debug, Serialize)]
struct PromptSummary {
    prompt_index: usize,
    token_exact: bool,
    max_abs_logprob_delta: f32,
    passed: bool,
}
struct Scores {
    argmax: u32,
    values: Vec<f32>,
    normalizer: f32,
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}
fn run(args: Args) -> Result<bool> {
    match args.command {
        Command::Record {
            model_dir,
            prompts,
            max_tokens,
            out,
        } => record(&model_dir, &prompts, max_tokens, &out).map(|()| true),
        Command::Check {
            model_dir,
            fixture,
            logprob_tolerance,
            perturb_logits,
            allow_model_mismatch,
        } => check(
            &model_dir,
            &fixture,
            logprob_tolerance,
            perturb_logits,
            allow_model_mismatch,
        ),
    }
}

fn record(model_dir: &Path, prompts_path: &Path, max_tokens: usize, out: &Path) -> Result<()> {
    let prompt_file: PromptFile = serde_json::from_slice(
        &std::fs::read(prompts_path).with_context(|| format!("read {}", prompts_path.display()))?,
    )
    .with_context(|| format!("parse {}", prompts_path.display()))?;
    let prompts = prompt_file.into_prompts();
    if prompts.is_empty() {
        bail!("prompt file contains no prompts");
    }
    let tokenizer = load_tokenizer(model_dir)?;
    let mut model = load_model(model_dir)?;
    let mut records = Vec::with_capacity(prompts.len());
    for (index, prompt) in prompts.iter().enumerate() {
        eprintln!("[record {}/{}]", index + 1, prompts.len());
        let (token_ids, logprobs) =
            greedy_continuation(&mut model, &tokenize(&tokenizer, prompt)?, max_tokens)?;
        records.push(PromptRecord {
            prompt: prompt.clone(),
            token_ids,
            logprobs,
        });
    }
    let fixture = Fixture {
        schema_version: SCHEMA_VERSION,
        model_basename: model_basename(model_dir)?,
        config_json_sha256: config_hash(model_dir)?,
        prompts: records,
    };
    std::fs::write(out, serde_json::to_vec_pretty(&fixture)?)
        .with_context(|| format!("write {}", out.display()))?;
    eprintln!("[recorded] {}", out.display());
    Ok(())
}

fn check(
    model_dir: &Path,
    fixture_path: &Path,
    tolerance: f32,
    perturb: Option<f32>,
    allow_model_mismatch: bool,
) -> Result<bool> {
    if tolerance < 0.0 {
        bail!("--logprob-tolerance must be non-negative");
    }
    if perturb.is_some_and(|epsilon| !epsilon.is_finite() || epsilon <= 0.0) {
        bail!("--perturb-logits must be finite and positive");
    }
    let fixture: Fixture = serde_json::from_slice(
        &std::fs::read(fixture_path).with_context(|| format!("read {}", fixture_path.display()))?,
    )
    .with_context(|| format!("parse {}", fixture_path.display()))?;
    if fixture.schema_version != SCHEMA_VERSION {
        bail!(
            "unsupported fixture schema version {}",
            fixture.schema_version
        );
    }
    let model_config_hash = config_hash(model_dir)?;
    ensure_fixture_matches_model_config(
        &fixture.config_json_sha256,
        &model_config_hash,
        allow_model_mismatch,
    )?;
    let tokenizer = load_tokenizer(model_dir)?;
    let mut model = load_model(model_dir)?;
    let mut summaries = Vec::with_capacity(fixture.prompts.len());
    for (index, record) in fixture.prompts.iter().enumerate() {
        let (token_exact, max_delta) = teacher_forced(
            &mut model,
            &tokenize(&tokenizer, &record.prompt)?,
            record,
            perturb,
        )?;
        let passed = prompt_passed(token_exact, max_delta, tolerance);
        eprintln!(
            "[check {}/{}] token_exact={} max_abs_logprob_delta={:.8} passed={}",
            index + 1,
            fixture.prompts.len(),
            token_exact,
            max_delta,
            passed
        );
        summaries.push(PromptSummary {
            prompt_index: index,
            token_exact,
            max_abs_logprob_delta: max_delta,
            passed,
        });
    }
    let passed = summaries.iter().all(|summary| summary.passed);
    println!(
        "{}",
        serde_json::to_string(&CheckSummary {
            schema_version: SCHEMA_VERSION,
            model_basename: model_basename(model_dir)?,
            config_json_sha256: model_config_hash,
            logprob_tolerance: tolerance,
            passed,
            prompts: summaries
        })?
    );
    Ok(passed)
}

fn tokenize(tokenizer: &tokenizers::Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    let tokens = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!("tokenize prompt: {e}"))?
        .get_ids()
        .to_vec();
    if tokens.is_empty() {
        bail!("prompt tokenized to an empty sequence");
    }
    Ok(tokens)
}
fn greedy_continuation(
    model: &mut AnyModel,
    prompt: &[u32],
    count: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let mut cache = model.make_cache_with_config(KvCacheConfig::default())?;
    let input = token_array(prompt)?;
    let mut logits = model.forward_chunked(&input, &mut cache, PREFILL_CHUNK_SIZE)?;
    let mut token_ids = Vec::with_capacity(count);
    let mut logprobs = Vec::with_capacity(count);
    for _ in 0..count {
        let scores = score(&logits, None)?;
        token_ids.push(scores.argmax);
        logprobs.push(scores.logprob(scores.argmax)?);
        logits = model.forward_last_token(&token_array(&[scores.argmax])?, None, &mut cache)?;
    }
    Ok((token_ids, logprobs))
}
fn teacher_forced(
    model: &mut AnyModel,
    prompt: &[u32],
    record: &PromptRecord,
    perturb: Option<f32>,
) -> Result<(bool, f32)> {
    if record.token_ids.len() != record.logprobs.len() {
        bail!("fixture prompt has mismatched token_ids and logprobs lengths");
    }
    let mut cache = model.make_cache_with_config(KvCacheConfig::default())?;
    let mut logits =
        model.forward_chunked(&token_array(prompt)?, &mut cache, PREFILL_CHUNK_SIZE)?;
    let mut exact = true;
    let mut max_delta = 0.0_f32;
    for (token, reference) in record.token_ids.iter().zip(&record.logprobs) {
        let scores = score(&logits, perturb)?;
        exact &= scores.argmax == *token;
        max_delta = max_delta.max((scores.logprob(*token)? - *reference).abs());
        logits = model.forward_last_token(&token_array(&[*token])?, None, &mut cache)?;
    }
    Ok((exact, max_delta))
}
fn score(logits: &Array, perturb: Option<f32>) -> Result<Scores> {
    let last = logits.index((.., -1, ..)).as_dtype(Dtype::Float32)?;
    eval([&last])?;
    let mut values = last.as_slice::<f32>().to_vec();
    if values.is_empty() {
        bail!("model returned empty logits");
    }
    if let Some(epsilon) = perturb {
        let (argmax_index, _) = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .context("model returned empty logits")?;
        values[argmax_index] -= epsilon;
    }
    let (index, _) = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .context("model returned empty logits")?;
    let argmax = u32::try_from(index)?;
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let normalizer = values
        .iter()
        .map(|value| (value - max).exp())
        .sum::<f32>()
        .ln()
        + max;
    Ok(Scores {
        argmax,
        values,
        normalizer,
    })
}
impl Scores {
    fn logprob(&self, token: u32) -> Result<f32> {
        let index = usize::try_from(token)?;
        self.values
            .get(index)
            .map(|value| *value - self.normalizer)
            .context("fixture token id is outside model vocabulary")
    }
}
fn token_array(tokens: &[u32]) -> Result<Array> {
    let values: Vec<i32> = tokens
        .iter()
        .copied()
        .map(i32::try_from)
        .collect::<std::result::Result<_, _>>()?;
    let length = i32::try_from(values.len())?;
    Ok(Array::from_slice(&values, &[1, length]))
}
fn model_basename(model_dir: &Path) -> Result<String> {
    model_dir
        .file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .context("--model-dir must end in a valid UTF-8 directory name")
}
fn config_hash(model_dir: &Path) -> Result<String> {
    let path = model_dir.join("config.json");
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn ensure_fixture_matches_model_config(
    fixture_config_hash: &str,
    model_config_hash: &str,
    allow_model_mismatch: bool,
) -> Result<()> {
    if fixture_config_hash != model_config_hash && !allow_model_mismatch {
        bail!(
            "fixture config_json_sha256 {fixture_config_hash} does not match model config_json_sha256 {model_config_hash}; pass --allow-model-mismatch to override"
        );
    }
    Ok(())
}

fn prompt_passed(token_exact: bool, max_abs_logprob_delta: f32, tolerance: f32) -> bool {
    if tolerance == 0.0 {
        token_exact && max_abs_logprob_delta == 0.0
    } else {
        token_exact || max_abs_logprob_delta <= tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perturbation_reduces_the_current_argmax_logit() {
        let logits = Array::from_slice(&[1.0_f32, 1.4, 0.5], &[1, 1, 3]);
        let unperturbed = score(&logits, None).unwrap();

        let scores = score(&logits, Some(0.5)).unwrap();

        assert_eq!(scores.argmax, 0);
        assert!((scores.values[1] - 0.9).abs() < f32::EPSILON);
        assert!((scores.logprob(1).unwrap() - unperturbed.logprob(1).unwrap()).abs() > 0.0);
    }

    #[test]
    fn strict_tolerance_requires_exact_tokens_and_logprobs() {
        assert!(prompt_passed(true, 0.0, 0.0));
        assert!(!prompt_passed(false, 0.0, 0.0));
        assert!(!prompt_passed(true, f32::EPSILON, 0.0));
    }

    #[test]
    fn non_strict_tolerance_accepts_exact_tokens_or_bounded_logprob_delta() {
        assert!(prompt_passed(true, 1.0, 0.1));
        assert!(prompt_passed(false, 0.1, 0.1));
        assert!(!prompt_passed(false, 0.2, 0.1));
    }

    #[test]
    fn config_mismatch_requires_explicit_override() {
        let error = ensure_fixture_matches_model_config("fixture-hash", "model-hash", false)
            .unwrap_err()
            .to_string();

        assert!(error.contains("fixture-hash"));
        assert!(error.contains("model-hash"));
        assert!(ensure_fixture_matches_model_config("fixture-hash", "model-hash", true).is_ok());
    }
}
