use std::time::Instant;

pub use higgs_models::{TokenLogprobInfo, TopLogprobEntry};

/// Reasoning tokens allowed before the engine forces `</think>`, unless the
/// model or request configures another budget.
pub const DEFAULT_THINKING_BUDGET: u32 = 256;

/// Whether a request runs with a reasoning block, and how long it may be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Off,
    On {
        /// Per-request budget; `None` defers to the model's configured default.
        budget: Option<u32>,
    },
}

impl ThinkingMode {
    pub const fn new(enabled: bool, budget: Option<u32>) -> Self {
        if enabled {
            Self::On { budget }
        } else {
            Self::Off
        }
    }

    pub const fn is_on(self) -> bool {
        matches!(self, Self::On { .. })
    }

    /// The request's budget, or `default` when it did not set one.
    pub const fn budget_or(self, default: u32) -> u32 {
        match self {
            Self::On {
                budget: Some(budget),
            } => budget,
            Self::On { budget: None } | Self::Off => default,
        }
    }
}

/// What the engine measured while serving one request.
///
/// Every field is `None` until the engine actually observed it, so callers
/// can omit a measurement instead of reporting a placeholder.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GenerationStats {
    /// Prompt tokens served from a prefix cache instead of being prefilled.
    pub cached_tokens: Option<u32>,
    /// When the engine began working on the prompt, after any queueing.
    pub prefill_started_at: Option<Instant>,
    /// When the first completion token was produced.
    pub first_token_at: Option<Instant>,
    /// When the last completion token was produced.
    pub finished_at: Option<Instant>,
}

impl GenerationStats {
    /// Fold a streaming event into the stats, stamping times with `now`.
    ///
    /// Prefill-progress events mark the start of prompt evaluation and carry
    /// the prefix-cache hit; the first token event marks the end of it.
    pub fn observe(&mut self, output: &StreamingOutput, now: Instant) {
        if let Some(progress) = output.prefill_progress {
            self.cached_tokens = Some(progress.cached);
            self.prefill_started_at.get_or_insert(now);
            return;
        }
        self.first_token_at.get_or_insert(now);
        self.finished_at = Some(now);
    }
}

/// Output from a generation request.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub text: String,
    pub finish_reason: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub token_logprobs: Option<Vec<TokenLogprobInfo>>,
    pub stats: GenerationStats,
}

/// Prefill progress for one streaming request, in absolute prompt tokens.
///
/// `processed` counts cached + prefilled tokens, so `processed / total` is
/// directly displayable; `cached` exposes the prefix-cache hit separately.
/// Matches the semantics of llama.cpp's `prompt_progress` SSE field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillProgress {
    pub processed: u32,
    pub cached: u32,
    pub total: u32,
}

/// Output from a streaming generation step.
#[derive(Debug, Clone)]
pub struct StreamingOutput {
    pub new_text: String,
    pub finished: bool,
    pub finish_reason: Option<String>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub token_logprob: Option<TokenLogprobInfo>,
    /// Set on progress-only events emitted during chunked prefill
    /// (`new_text` is empty, `completion_tokens` is 0). `None` on token
    /// outputs.
    pub prefill_progress: Option<PrefillProgress>,
}

#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_output_construction_and_field_access() {
        let output = GenerationOutput {
            text: "Hello world".to_owned(),
            finish_reason: "stop".to_owned(),
            prompt_tokens: 10,
            completion_tokens: 5,
            token_logprobs: None,
            stats: GenerationStats::default(),
        };
        assert_eq!(output.text, "Hello world");
        assert_eq!(output.finish_reason, "stop");
        assert_eq!(output.prompt_tokens, 10);
        assert_eq!(output.completion_tokens, 5);
    }

    #[test]
    fn generation_output_empty_defaults() {
        let output = GenerationOutput {
            text: String::new(),
            finish_reason: "length".to_owned(),
            prompt_tokens: 0,
            completion_tokens: 0,
            token_logprobs: None,
            stats: GenerationStats::default(),
        };
        assert_eq!(output.text, "");
        assert_eq!(output.prompt_tokens, 0);
        assert_eq!(output.completion_tokens, 0);
    }

    #[test]
    fn streaming_output_finished_true() {
        let output = StreamingOutput {
            new_text: "done".to_owned(),
            finished: true,
            finish_reason: Some("stop".to_owned()),
            prompt_tokens: 20,
            completion_tokens: 15,
            token_logprob: None,
            prefill_progress: None,
        };
        assert!(output.finished);
        assert_eq!(output.finish_reason.as_deref(), Some("stop"));
        assert_eq!(output.new_text, "done");
    }

    #[test]
    fn streaming_output_finished_false() {
        let output = StreamingOutput {
            new_text: "partial".to_owned(),
            finished: false,
            finish_reason: None,
            prompt_tokens: 20,
            completion_tokens: 3,
            token_logprob: None,
            prefill_progress: None,
        };
        assert!(!output.finished);
        assert!(output.finish_reason.is_none());
    }

    #[test]
    fn streaming_output_empty_text_zero_tokens() {
        let output = StreamingOutput {
            new_text: String::new(),
            finished: true,
            finish_reason: Some("length".to_owned()),
            prompt_tokens: 0,
            completion_tokens: 0,
            token_logprob: None,
            prefill_progress: None,
        };
        assert_eq!(output.new_text, "");
        assert_eq!(output.prompt_tokens, 0);
        assert_eq!(output.completion_tokens, 0);
    }

    #[test]
    fn generation_output_clone() {
        let output = GenerationOutput {
            text: "test".to_owned(),
            finish_reason: "stop".to_owned(),
            prompt_tokens: 5,
            completion_tokens: 3,
            token_logprobs: None,
            stats: GenerationStats::default(),
        };
        let cloned = output.clone();
        assert_eq!(cloned.text, output.text);
        assert_eq!(cloned.finish_reason, output.finish_reason);
    }

    #[test]
    fn streaming_output_clone() {
        let output = StreamingOutput {
            new_text: "stream".to_owned(),
            finished: false,
            finish_reason: None,
            prompt_tokens: 10,
            completion_tokens: 2,
            token_logprob: None,
            prefill_progress: None,
        };
        let cloned = output.clone();
        assert_eq!(cloned.new_text, output.new_text);
        assert_eq!(cloned.finished, output.finished);
        assert_eq!(cloned.finish_reason, output.finish_reason);
    }

    #[test]
    fn generation_output_debug_format() {
        let output = GenerationOutput {
            text: "hi".to_owned(),
            finish_reason: "stop".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 1,
            token_logprobs: None,
            stats: GenerationStats::default(),
        };
        let debug_str = format!("{output:?}");
        assert!(debug_str.contains("GenerationOutput"));
        assert!(debug_str.contains("hi"));
    }

    #[test]
    fn streaming_output_debug_format() {
        let output = StreamingOutput {
            new_text: "token".to_owned(),
            finished: true,
            finish_reason: Some("stop".to_owned()),
            prompt_tokens: 5,
            completion_tokens: 10,
            token_logprob: None,
            prefill_progress: None,
        };
        let debug_str = format!("{output:?}");
        assert!(debug_str.contains("StreamingOutput"));
        assert!(debug_str.contains("token"));
    }

    #[test]
    fn generation_output_with_logprobs() {
        let output = GenerationOutput {
            text: "hello".to_owned(),
            finish_reason: "stop".to_owned(),
            prompt_tokens: 5,
            completion_tokens: 1,
            token_logprobs: Some(vec![TokenLogprobInfo {
                token_id: 42,
                logprob: -0.5,
                top_logprobs: vec![
                    TopLogprobEntry {
                        token_id: 42,
                        logprob: -0.5,
                    },
                    TopLogprobEntry {
                        token_id: 99,
                        logprob: -1.2,
                    },
                ],
            }]),
            stats: GenerationStats::default(),
        };
        let lps = output.token_logprobs.unwrap();
        assert_eq!(lps.len(), 1);
        assert_eq!(lps[0].token_id, 42);
        assert_eq!(lps[0].top_logprobs.len(), 2);
    }

    fn progress_output(cached: u32) -> StreamingOutput {
        StreamingOutput {
            new_text: String::new(),
            finished: false,
            finish_reason: None,
            prompt_tokens: 10,
            completion_tokens: 0,
            token_logprob: None,
            prefill_progress: Some(PrefillProgress {
                processed: cached,
                cached,
                total: 10,
            }),
        }
    }

    fn token_output(completion_tokens: u32) -> StreamingOutput {
        StreamingOutput {
            new_text: "x".to_owned(),
            finished: false,
            finish_reason: None,
            prompt_tokens: 10,
            completion_tokens,
            token_logprob: None,
            prefill_progress: None,
        }
    }

    #[test]
    fn thinking_mode_off_ignores_budget() {
        let mode = ThinkingMode::new(false, Some(10));
        assert_eq!(mode, ThinkingMode::Off);
        assert!(!mode.is_on());
        assert_eq!(
            mode.budget_or(DEFAULT_THINKING_BUDGET),
            DEFAULT_THINKING_BUDGET
        );
    }

    #[test]
    fn thinking_mode_request_budget_beats_default() {
        let mode = ThinkingMode::new(true, Some(32));
        assert!(mode.is_on());
        assert_eq!(mode.budget_or(512), 32);
    }

    #[test]
    fn thinking_mode_without_budget_uses_default() {
        let mode = ThinkingMode::new(true, None);
        assert_eq!(mode.budget_or(512), 512);
        assert_eq!(DEFAULT_THINKING_BUDGET, 256);
    }

    #[test]
    fn stats_observe_tracks_prefill_and_token_times() {
        let start = Instant::now();
        let mut stats = GenerationStats::default();
        stats.observe(&progress_output(4), start);
        let first = start + std::time::Duration::from_millis(5);
        let last = start + std::time::Duration::from_millis(9);
        stats.observe(&token_output(1), first);
        stats.observe(&token_output(2), last);

        assert_eq!(stats.cached_tokens, Some(4));
        assert_eq!(stats.prefill_started_at, Some(start));
        assert_eq!(stats.first_token_at, Some(first));
        assert_eq!(stats.finished_at, Some(last));
    }

    #[test]
    fn stats_observe_without_progress_leaves_cache_unknown() {
        let mut stats = GenerationStats::default();
        stats.observe(&token_output(1), Instant::now());
        assert_eq!(stats.cached_tokens, None);
        assert_eq!(stats.prefill_started_at, None);
        assert!(stats.first_token_at.is_some());
    }

    #[test]
    fn stats_later_progress_updates_cache_but_keeps_first_start() {
        let start = Instant::now();
        let mut stats = GenerationStats::default();
        stats.observe(&progress_output(0), start);
        stats.observe(
            &progress_output(6),
            start + std::time::Duration::from_millis(2),
        );
        assert_eq!(stats.cached_tokens, Some(6));
        assert_eq!(stats.prefill_started_at, Some(start));
        assert_eq!(stats.first_token_at, None);
    }
}
