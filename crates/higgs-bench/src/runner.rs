//! The client-side benchmark scenarios shared by `bench_compare` and `higgs bench`.
//!
//! Each scenario drives an [`Endpoint`] with synthetic prompts and records time to first token,
//! decode and prefill rates, prefix-cache reuse and aggregate throughput under concurrency.

use std::fmt::Write as _;
use std::time::Instant;

use anyhow::{Result, bail};
use clap::ValueEnum;
use futures::future::try_join_all;
use serde::Serialize;
use serde_json::json;

use crate::client::Endpoint;
use crate::compare::{
    StreamOutcome, Summary, TrialMetrics, build_long_output_prompt, build_prefill_prompt,
    build_prompt_body, decode_prompt, fresh_nonce, prefill_tokps,
};
use crate::rss::RssSampler;
use crate::stats;

const PREFIX_CACHE_MAX_TOKENS: u32 = 32;
const PREFIX_CACHE_FOLLOW_UP: &str = "Now state the main point of the text in one short sentence.";
const PREFIX_CACHE_FALLBACK_REPLY: &str = "Understood.";
const PREFILL_MAX_TOKENS: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Scenario {
    #[value(name = "decode")]
    Decode,
    #[value(name = "prefill")]
    Prefill,
    #[value(name = "prefix_cache")]
    PrefixCache,
    #[value(name = "concurrency")]
    Concurrency,
}

impl Scenario {
    pub const ALL: [Self; 4] = [
        Self::Decode,
        Self::Prefill,
        Self::PrefixCache,
        Self::Concurrency,
    ];
}

/// What to measure and how much of it.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub scenarios: Vec<Scenario>,
    pub trials: u32,
    pub warmup: u32,
    pub max_tokens: u32,
    /// Approximate prompt sizes (tokens) for the prefill scenario.
    pub prefill_tokens: Vec<usize>,
    /// Approximate size (tokens) of the first message of a prefix-cache trial.
    pub prefix_cache_tokens: usize,
    /// Simultaneous request counts for the concurrency scenario.
    pub concurrency: Vec<usize>,
    /// Measured rounds per concurrency level (after the warm-up rounds).
    pub concurrency_rounds: usize,
}

impl Plan {
    /// Sized to finish in about a minute on a 1-2B model.
    pub fn quick() -> Self {
        Self {
            scenarios: Scenario::ALL.to_vec(),
            trials: 3,
            warmup: 1,
            max_tokens: 128,
            prefill_tokens: vec![512, 2048],
            prefix_cache_tokens: 2048,
            concurrency: vec![1, 2, 4, 8],
            concurrency_rounds: 2,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.trials == 0 {
            bail!("trials must be >= 1");
        }
        if self.scenarios.is_empty() {
            bail!("at least one scenario is required");
        }
        if self.max_tokens < 2 {
            bail!("max tokens must be >= 2 so a decode rate can be measured");
        }
        if self.concurrency.contains(&0) || self.prefill_tokens.contains(&0) {
            bail!("concurrency and prefill token counts must be >= 1");
        }
        if self.prefix_cache_tokens == 0 {
            bail!("prefix-cache context tokens must be >= 1");
        }
        if self.concurrency_rounds == 0 {
            bail!("concurrency rounds must be >= 1");
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct DecodeResults {
    pub trials: Vec<TrialMetrics>,
    pub ttft_ms: Summary,
    pub decode_tokps: Summary,
    pub e2e_tokps: Summary,
    pub completion_tokens: Summary,
    pub trials_below_max_tokens: usize,
    pub trials_with_usage_missing: usize,
}

#[derive(Debug, Serialize)]
pub struct PrefillTrial {
    pub metrics: TrialMetrics,
    pub prefill_tokps: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct PrefillSizeResults {
    pub target_tokens: usize,
    pub trials: Vec<PrefillTrial>,
    pub prompt_tokens_median: f64,
    pub ttft_ms_median: f64,
    pub prefill_tokps_median: f64,
    pub trials_with_cached_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct PrefillResults {
    pub sizes: Vec<PrefillSizeResults>,
}

#[derive(Debug, Serialize)]
pub struct PrefixCacheTrial {
    pub turn1: TrialMetrics,
    pub turn2: TrialMetrics,
    pub ttft_ratio_turn2_over_turn1: f64,
}

#[derive(Debug, Serialize)]
pub struct PrefixCacheResults {
    pub trials: Vec<PrefixCacheTrial>,
    pub turn1_ttft_ms_median: f64,
    pub turn2_ttft_ms_median: f64,
    pub ttft_ratio_median: f64,
    pub turn2_cached_tokens_median: Option<f64>,
    pub turn2_prompt_tokens_median: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ConcurrencyRound {
    pub aggregate_tokps: f64,
    pub mean_ttft_ms: f64,
    pub total_completion_tokens: u64,
    pub wall_ms: f64,
    pub requests: Vec<TrialMetrics>,
}

#[derive(Debug, Serialize)]
pub struct ConcurrencyLevelResults {
    pub concurrency: usize,
    pub rounds: Vec<ConcurrencyRound>,
    pub aggregate_tokps_median: f64,
    pub mean_ttft_ms_median: f64,
}

#[derive(Debug, Serialize)]
pub struct ConcurrencyResults {
    pub levels: Vec<ConcurrencyLevelResults>,
}

#[derive(Debug, Default, Serialize)]
pub struct Results {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode: Option<DecodeResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill: Option<PrefillResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache: Option<PrefixCacheResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<ConcurrencyResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rss_mb_peak: Option<f64>,
}

fn present(values: impl Iterator<Item = Option<f64>>) -> Vec<f64> {
    values.flatten().collect()
}

/// Receives one short line per step, for the caller to show or drop.
pub type Progress<'a> = &'a (dyn Fn(&str) + Sync);

fn step(progress: Progress<'_>, scenario: &str, phase: &str, index: usize, total: usize) {
    progress(&format!("[{scenario} {phase} {}/{total}]", index + 1));
}

async fn run_decode(
    endpoint: &Endpoint,
    plan: &Plan,
    progress: Progress<'_>,
) -> Result<DecodeResults> {
    let prompt = decode_prompt();
    for index in 0..plan.warmup {
        step(
            progress,
            "decode",
            "warmup",
            index as usize,
            plan.warmup as usize,
        );
        endpoint.stream_user_prompt(prompt, plan.max_tokens).await?;
    }
    let mut trials = Vec::new();
    for index in 0..plan.trials {
        step(
            progress,
            "decode",
            "trial",
            index as usize,
            plan.trials as usize,
        );
        trials.push(endpoint.stream_user_prompt(prompt, plan.max_tokens).await?);
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
            .filter(|t| t.completion_tokens < plan.max_tokens)
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

async fn run_prefill(
    endpoint: &Endpoint,
    plan: &Plan,
    progress: Progress<'_>,
) -> Result<PrefillResults> {
    let mut sizes = Vec::new();
    for &target_tokens in &plan.prefill_tokens {
        let body = build_prompt_body(target_tokens);
        let label = format!("prefill {target_tokens}");
        for index in 0..plan.warmup {
            step(
                progress,
                &label,
                "warmup",
                index as usize,
                plan.warmup as usize,
            );
            run_prefill_trial(endpoint, &body).await?;
        }
        let mut trials = Vec::new();
        for index in 0..plan.trials {
            step(
                progress,
                &label,
                "trial",
                index as usize,
                plan.trials as usize,
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
    let first_user = build_prefill_prompt(&fresh_nonce(), context);
    let turn1_messages = json!([{ "role": "user", "content": first_user }]);
    let turn1: StreamOutcome = endpoint
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

async fn run_prefix_cache(
    endpoint: &Endpoint,
    plan: &Plan,
    progress: Progress<'_>,
) -> Result<PrefixCacheResults> {
    let context = build_prompt_body(plan.prefix_cache_tokens);
    for index in 0..plan.warmup {
        step(
            progress,
            "prefix_cache",
            "warmup",
            index as usize,
            plan.warmup as usize,
        );
        run_prefix_cache_trial(endpoint, &context).await?;
    }
    let mut trials = Vec::new();
    for index in 0..plan.trials {
        step(
            progress,
            "prefix_cache",
            "trial",
            index as usize,
            plan.trials as usize,
        );
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

async fn run_concurrency(
    endpoint: &Endpoint,
    plan: &Plan,
    progress: Progress<'_>,
) -> Result<ConcurrencyResults> {
    let mut levels = Vec::new();
    for &level in &plan.concurrency {
        let label = format!("concurrency {level}");
        for index in 0..plan.warmup {
            step(
                progress,
                &label,
                "warmup",
                index as usize,
                plan.warmup as usize,
            );
            run_concurrency_round(endpoint, level, plan.max_tokens).await?;
        }
        let mut rounds = Vec::new();
        for index in 0..plan.concurrency_rounds {
            step(progress, &label, "round", index, plan.concurrency_rounds);
            rounds.push(run_concurrency_round(endpoint, level, plan.max_tokens).await?);
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

/// Runs the planned scenarios in order. `rss` is sampled after each one and, when present,
/// provides the peak resident set size of the server.
pub async fn run_scenarios(
    endpoint: &Endpoint,
    plan: &Plan,
    rss: Option<RssSampler>,
    progress: Progress<'_>,
) -> Result<Results> {
    plan.validate()?;
    let mut results = Results::default();
    for &scenario in &plan.scenarios {
        progress(&format!("[scenario] {scenario:?}"));
        match scenario {
            Scenario::Decode => results.decode = Some(run_decode(endpoint, plan, progress).await?),
            Scenario::Prefill => {
                results.prefill = Some(run_prefill(endpoint, plan, progress).await?);
            }
            Scenario::PrefixCache => {
                results.prefix_cache = Some(run_prefix_cache(endpoint, plan, progress).await?);
            }
            Scenario::Concurrency => {
                results.concurrency = Some(run_concurrency(endpoint, plan, progress).await?);
            }
        }
        if let Some(sampler) = &rss {
            sampler.sample_now();
        }
    }
    results.rss_mb_peak = rss.map(RssSampler::finish);
    Ok(results)
}

/// The measured scenarios as aligned text rows: scenario, measure, value.
pub fn render_table(results: &Results) -> String {
    let mut rows: Vec<[String; 3]> = Vec::new();
    let mut row = |scenario: &str, measure: String, value: String| {
        rows.push([scenario.to_owned(), measure, value]);
    };

    if let Some(decode) = &results.decode {
        row(
            "decode",
            "decode tok/s (median, mean, p95)".to_owned(),
            format!(
                "{:.1}  {:.1}  {:.1}",
                decode.decode_tokps.median, decode.decode_tokps.mean, decode.decode_tokps.p95
            ),
        );
        row(
            "decode",
            "time to first token (ms, median)".to_owned(),
            format!("{:.0}", decode.ttft_ms.median),
        );
        row(
            "decode",
            "tokens per reply (mean)".to_owned(),
            format!("{:.0}", decode.completion_tokens.mean),
        );
    }
    if let Some(prefill) = &results.prefill {
        for size in &prefill.sizes {
            row(
                "prefill",
                format!(
                    "{:.0} prompt tokens: tok/s, TTFT ms",
                    size.prompt_tokens_median
                ),
                format!(
                    "{:.0}  {:.0}",
                    size.prefill_tokps_median, size.ttft_ms_median
                ),
            );
        }
    }
    if let Some(cache) = &results.prefix_cache {
        row(
            "prefix cache",
            "follow-up TTFT ms: first turn, second turn".to_owned(),
            format!(
                "{:.0}  {:.0}  ({:.2}x)",
                cache.turn1_ttft_ms_median, cache.turn2_ttft_ms_median, cache.ttft_ratio_median
            ),
        );
        row(
            "prefix cache",
            "second turn tokens: cached / prompt".to_owned(),
            match (
                cache.turn2_cached_tokens_median,
                cache.turn2_prompt_tokens_median,
            ) {
                (Some(cached), Some(prompt)) => format!("{cached:.0} / {prompt:.0}"),
                (Some(_) | None, Some(_) | None) => "not reported".to_owned(),
            },
        );
    }
    if let Some(concurrency) = &results.concurrency {
        for level in &concurrency.levels {
            row(
                "concurrency",
                format!(
                    "{} at once: aggregate tok/s, mean TTFT ms",
                    level.concurrency
                ),
                format!(
                    "{:.1}  {:.0}",
                    level.aggregate_tokps_median, level.mean_ttft_ms_median
                ),
            );
        }
    }
    if let Some(rss) = results.rss_mb_peak {
        row(
            "memory",
            "server peak RSS (MB)".to_owned(),
            format!("{rss:.0}"),
        );
    }

    let widths = [0, 1, 2].map(|column| {
        rows.iter()
            .filter_map(|cells| cells.get(column))
            .map(String::len)
            .max()
            .unwrap_or(0)
    });
    let mut table = String::new();
    for [scenario, measure, value] in &rows {
        let [first, second, _] = widths;
        let _ = writeln!(table, "{scenario:<first$}  {measure:<second$}  {value}");
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_quick_plan_is_valid_and_runs_every_scenario() {
        let plan = Plan::quick();
        assert!(plan.validate().is_ok());
        for scenario in Scenario::ALL {
            assert!(plan.scenarios.contains(&scenario));
        }
    }

    #[test]
    fn invalid_plans_are_rejected() {
        let invalid = [
            Plan {
                trials: 0,
                ..Plan::quick()
            },
            Plan {
                scenarios: Vec::new(),
                ..Plan::quick()
            },
            Plan {
                max_tokens: 1,
                ..Plan::quick()
            },
            Plan {
                concurrency: vec![1, 0],
                ..Plan::quick()
            },
            Plan {
                prefill_tokens: vec![0],
                ..Plan::quick()
            },
            Plan {
                prefix_cache_tokens: 0,
                ..Plan::quick()
            },
            Plan {
                concurrency_rounds: 0,
                ..Plan::quick()
            },
        ];
        for plan in invalid {
            assert!(plan.validate().is_err(), "{plan:?}");
        }
    }

    fn summary(value: f64) -> Summary {
        Summary {
            mean: value,
            median: value,
            p95: value,
            stdev: 0.0,
        }
    }

    #[test]
    fn the_table_lists_each_measured_scenario_in_aligned_columns() {
        let results = Results {
            decode: Some(DecodeResults {
                trials: Vec::new(),
                ttft_ms: summary(41.0),
                decode_tokps: summary(212.5),
                e2e_tokps: summary(200.0),
                completion_tokens: summary(128.0),
                trials_below_max_tokens: 0,
                trials_with_usage_missing: 0,
            }),
            prefix_cache: Some(PrefixCacheResults {
                trials: Vec::new(),
                turn1_ttft_ms_median: 480.0,
                turn2_ttft_ms_median: 60.0,
                ttft_ratio_median: 0.125,
                turn2_cached_tokens_median: Some(2040.0),
                turn2_prompt_tokens_median: Some(2090.0),
            }),
            concurrency: Some(ConcurrencyResults {
                levels: vec![ConcurrencyLevelResults {
                    concurrency: 4,
                    rounds: Vec::new(),
                    aggregate_tokps_median: 640.0,
                    mean_ttft_ms_median: 90.0,
                }],
            }),
            rss_mb_peak: Some(2345.4),
            ..Results::default()
        };
        let table = render_table(&results);
        assert!(table.contains("212.5  212.5  212.5"), "{table}");
        assert!(table.contains("2040 / 2090"), "{table}");
        assert!(
            table.contains("(0.12x)") || table.contains("(0.13x)"),
            "{table}"
        );
        assert!(table.contains("4 at once"), "{table}");
        assert!(table.contains("2345"), "{table}");
        assert!(!table.contains("prefill"), "{table}");
    }

    #[test]
    fn missing_cache_counts_are_reported_as_not_reported() {
        let results = Results {
            prefix_cache: Some(PrefixCacheResults {
                trials: Vec::new(),
                turn1_ttft_ms_median: 1.0,
                turn2_ttft_ms_median: 1.0,
                ttft_ratio_median: 1.0,
                turn2_cached_tokens_median: None,
                turn2_prompt_tokens_median: None,
            }),
            ..Results::default()
        };
        assert!(render_table(&results).contains("not reported"));
    }
}
