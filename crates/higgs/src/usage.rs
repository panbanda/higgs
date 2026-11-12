//! Builds the `usage` object attached to completion responses.
//!
//! Everything here derives from what the engine actually observed
//! ([`GenerationStats`]); a figure that was not observed stays `None` and is
//! left out of the response instead of being reported as zero.

use std::time::{Duration, Instant};

use higgs_engine::engine::GenerationStats;

use crate::types::openai::{
    CompletionTokensDetails, CompletionUsage, PromptTokensDetails, UsageTimings,
};

/// Token counts as reported by the engine for one finished request.
#[derive(Debug, Clone, Copy)]
pub struct TokenCounts {
    pub prompt: u32,
    pub completion: u32,
    /// Tokens spent inside a reasoning block, when the response had one.
    pub reasoning: Option<u32>,
}

/// Assemble the full usage object for a request that began at `received_at`.
pub fn build_usage(
    counts: TokenCounts,
    stats: &GenerationStats,
    received_at: Instant,
) -> CompletionUsage {
    let mut usage = CompletionUsage::new(counts.prompt, counts.completion);
    usage.prompt_tokens_details = stats
        .cached_tokens
        .map(|cached_tokens| PromptTokensDetails { cached_tokens });
    usage.completion_tokens_details = counts
        .reasoning
        .map(|reasoning_tokens| CompletionTokensDetails { reasoning_tokens });
    usage.timings = build_timings(counts, stats, received_at);
    usage
}

fn build_timings(
    counts: TokenCounts,
    stats: &GenerationStats,
    received_at: Instant,
) -> UsageTimings {
    let time_to_first_token = stats
        .first_token_at
        .map(|first| first.saturating_duration_since(received_at));
    let prompt_eval = elapsed_between(stats.prefill_started_at, stats.first_token_at);
    let generation = elapsed_between(stats.first_token_at, stats.finished_at);
    let total = stats
        .finished_at
        .map(|finished| finished.saturating_duration_since(received_at));

    let prefilled_tokens = stats
        .cached_tokens
        .map(|cached| counts.prompt.saturating_sub(cached));
    // The first completion token comes out of prefill, so only the tokens
    // after it were produced during the measured generation span.
    let decoded_tokens = counts.completion.saturating_sub(1);

    UsageTimings {
        time_to_first_token: time_to_first_token.map(|d| d.as_secs_f64()),
        prompt_eval_duration: prompt_eval.map(|d| d.as_secs_f64()),
        generation_duration: generation
            .filter(|_| decoded_tokens > 0)
            .map(|d| d.as_secs_f64()),
        total_time: total.map(|d| d.as_secs_f64()),
        prompt_tokens_per_second: rate(prefilled_tokens, prompt_eval),
        generation_tokens_per_second: rate(Some(decoded_tokens), generation),
    }
}

fn elapsed_between(start: Option<Instant>, end: Option<Instant>) -> Option<Duration> {
    Some(end?.saturating_duration_since(start?))
}

fn rate(tokens: Option<u32>, duration: Option<Duration>) -> Option<f64> {
    let positive_tokens = tokens.filter(|count| *count > 0)?;
    let seconds = duration?.as_secs_f64();
    (seconds > 0.0).then(|| f64::from(positive_tokens) / seconds)
}

/// Number of tokens `text` occupies when encoded by `tokenizer`.
///
/// Used for reasoning text, which the engine does not count separately: this
/// is a re-encoding of the extracted reasoning, so it can differ by a token
/// at the boundaries from what the model sampled. Empty text yields `None`.
pub fn count_text_tokens(
    tokenizer: &higgs_engine::tokenizers::Tokenizer,
    text: &str,
) -> Option<u32> {
    if text.is_empty() {
        return None;
    }
    let encoding = tokenizer.encode(text, false).ok()?;
    u32::try_from(encoding.get_ids().len()).ok()
}

#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;

    fn counts(prompt: u32, completion: u32) -> TokenCounts {
        TokenCounts {
            prompt,
            completion,
            reasoning: None,
        }
    }

    fn full_stats(received_at: Instant) -> GenerationStats {
        GenerationStats {
            cached_tokens: Some(400),
            prefill_started_at: Some(received_at + Duration::from_millis(100)),
            first_token_at: Some(received_at + Duration::from_millis(600)),
            finished_at: Some(received_at + Duration::from_millis(2600)),
        }
    }

    #[test]
    fn empty_stats_report_only_token_counts() {
        let usage = build_usage(counts(10, 5), &GenerationStats::default(), Instant::now());
        assert_eq!(usage, CompletionUsage::new(10, 5));
        let json = serde_json::to_value(&usage).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        );
    }

    #[test]
    fn full_stats_fill_every_field() {
        let received_at = Instant::now();
        let usage = build_usage(counts(1000, 101), &full_stats(received_at), received_at);

        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, 400);
        let timings = usage.timings;
        assert!((timings.time_to_first_token.unwrap() - 0.6).abs() < 1e-9);
        assert!((timings.prompt_eval_duration.unwrap() - 0.5).abs() < 1e-9);
        assert!((timings.generation_duration.unwrap() - 2.0).abs() < 1e-9);
        assert!((timings.total_time.unwrap() - 2.6).abs() < 1e-9);
        // 600 uncached prompt tokens over 0.5 s.
        assert!((timings.prompt_tokens_per_second.unwrap() - 1200.0).abs() < 1e-6);
        // 100 decoded tokens (101 minus the prefill token) over 2 s.
        assert!((timings.generation_tokens_per_second.unwrap() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn unknown_cache_hit_omits_details_and_prompt_rate() {
        let received_at = Instant::now();
        let stats = GenerationStats {
            cached_tokens: None,
            ..full_stats(received_at)
        };
        let usage = build_usage(counts(1000, 11), &stats, received_at);
        assert!(usage.prompt_tokens_details.is_none());
        assert!(usage.timings.prompt_tokens_per_second.is_none());
        assert!(usage.timings.generation_tokens_per_second.is_some());
        let json = serde_json::to_value(&usage).unwrap();
        assert!(json.get("prompt_tokens_details").is_none());
        assert!(json.get("prompt_tokens_per_second").is_none());
    }

    #[test]
    fn zero_cached_tokens_is_reported_when_measured() {
        let received_at = Instant::now();
        let stats = GenerationStats {
            cached_tokens: Some(0),
            ..full_stats(received_at)
        };
        let usage = build_usage(counts(1000, 11), &stats, received_at);
        let json = serde_json::to_value(&usage).unwrap();
        assert_eq!(json["prompt_tokens_details"]["cached_tokens"], 0);
    }

    #[test]
    fn fully_cached_prompt_has_no_prompt_rate() {
        let received_at = Instant::now();
        let stats = GenerationStats {
            cached_tokens: Some(1000),
            ..full_stats(received_at)
        };
        let usage = build_usage(counts(1000, 11), &stats, received_at);
        assert!(usage.timings.prompt_tokens_per_second.is_none());
        assert!(usage.timings.prompt_eval_duration.is_some());
    }

    #[test]
    fn single_token_completion_has_no_generation_figures() {
        let received_at = Instant::now();
        let usage = build_usage(counts(100, 1), &full_stats(received_at), received_at);
        assert!(usage.timings.generation_duration.is_none());
        assert!(usage.timings.generation_tokens_per_second.is_none());
        assert!(usage.timings.time_to_first_token.is_some());
    }

    #[test]
    fn instantaneous_spans_never_produce_infinite_rates() {
        let received_at = Instant::now();
        let stats = GenerationStats {
            cached_tokens: Some(0),
            prefill_started_at: Some(received_at),
            first_token_at: Some(received_at),
            finished_at: Some(received_at),
        };
        let usage = build_usage(counts(100, 10), &stats, received_at);
        assert!(usage.timings.prompt_tokens_per_second.is_none());
        assert!(usage.timings.generation_tokens_per_second.is_none());
    }

    #[test]
    fn missing_prefill_start_omits_prompt_eval_only() {
        let received_at = Instant::now();
        let stats = GenerationStats {
            prefill_started_at: None,
            ..full_stats(received_at)
        };
        let usage = build_usage(counts(100, 10), &stats, received_at);
        assert!(usage.timings.prompt_eval_duration.is_none());
        assert!(usage.timings.prompt_tokens_per_second.is_none());
        assert!(usage.timings.time_to_first_token.is_some());
        assert!(usage.timings.total_time.is_some());
    }

    #[test]
    fn reasoning_tokens_appear_only_when_known() {
        let received_at = Instant::now();
        let with_reasoning = TokenCounts {
            reasoning: Some(7),
            ..counts(10, 20)
        };
        let reasoned = build_usage(with_reasoning, &GenerationStats::default(), received_at);
        let reasoned_json = serde_json::to_value(&reasoned).unwrap();
        assert_eq!(
            reasoned_json["completion_tokens_details"]["reasoning_tokens"],
            7
        );

        let plain = build_usage(counts(10, 20), &GenerationStats::default(), received_at);
        let plain_json = serde_json::to_value(&plain).unwrap();
        assert!(plain_json.get("completion_tokens_details").is_none());
    }

    #[test]
    fn timing_field_names_match_omlx() {
        let received_at = Instant::now();
        let usage = build_usage(counts(1000, 101), &full_stats(received_at), received_at);
        let json = serde_json::to_value(&usage).unwrap();
        for name in [
            "time_to_first_token",
            "prompt_eval_duration",
            "generation_duration",
            "total_time",
            "prompt_tokens_per_second",
            "generation_tokens_per_second",
        ] {
            assert!(
                json.get(name).is_some_and(serde_json::Value::is_f64),
                "{name}"
            );
        }
    }
}
