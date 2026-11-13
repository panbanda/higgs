//! Speculative decode strategies for a request that has the engine to itself.
//!
//! Both strategies draft several tokens, verify them with the backbone in one
//! pass, and keep the accepted prefix. They advance one sequence at a time, so
//! the worker runs them only while a single request is active and hands the
//! request over to ordinary decode as soon as another arrives.
//!
//! A speculating request is always at a cycle boundary between worker turns:
//! its KV cache holds the prompt and every emitted token, rejected drafts are
//! already rolled back, and [`Speculation::confirmed`] is the next token the
//! model would produce. That token has been neither emitted nor fed to the
//! cache, which is exactly the state ordinary decode starts from after it
//! emits the token, so the handover is just emitting [`Speculation::confirmed`].

use higgs_models::{AnyCache, AnyModel, MtpCache};
use mlx_rs::{
    Array,
    ops::indexing::{IndexOp, argmax_axis},
    transforms::eval,
};

use crate::{
    error::EngineError,
    mlx_tuning::MlxRuntimeTuning,
    mtp::{self, AdaptiveDraftDepth, MtpStats, PromptLookupConfig},
};

/// Which speculative strategy a request may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculationKind {
    /// Drafts copied from text the request already produced; works on any
    /// architecture.
    PromptLookup,
    /// Drafts from the checkpoint's multi-token-prediction head.
    Mtp,
}

/// The engine-wide switches that decide whether a request may speculate.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct SpeculationSettings {
    pub prompt_lookup: bool,
    pub mtp: bool,
    pub prime_mtp_from_prefill: bool,
    mtp_adaptive_draft: bool,
    mtp_prompt_lookup: bool,
    mtp_draft_n_max: usize,
    unchecked_prompt_lookup: bool,
    prompt_lookup_config: PromptLookupConfig,
}

impl SpeculationSettings {
    pub fn from_tuning(tuning: &MlxRuntimeTuning) -> Self {
        let defaults = PromptLookupConfig::default();
        Self {
            prompt_lookup: tuning.prompt_lookup(),
            mtp: tuning.enable_mtp(),
            prime_mtp_from_prefill: flag("HIGGS_MTP_PRIME_PREFILL").unwrap_or(true),
            mtp_adaptive_draft: tuning.mtp_adaptive_draft(),
            mtp_prompt_lookup: tuning.mtp_prompt_lookup(),
            mtp_draft_n_max: tuning.mtp_draft_n_max(),
            unchecked_prompt_lookup: flag("HIGGS_PROMPT_LOOKUP_UNCHECKED").unwrap_or(false),
            prompt_lookup_config: PromptLookupConfig {
                max_drafts: env_usize("HIGGS_PROMPT_LOOKUP_DRAFT_N_MAX", defaults.max_drafts),
                max_ngram: env_usize("HIGGS_PROMPT_LOOKUP_NGRAM_MAX", defaults.max_ngram),
                max_window: env_usize("HIGGS_PROMPT_LOOKUP_WINDOW", defaults.max_window),
            },
        }
    }
}

fn flag(name: &str) -> Option<bool> {
    crate::simple::parse_enabled_flag(std::env::var(name).ok().as_deref())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

/// What about a request decides whether it may speculate.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
pub struct SpeculationRequest {
    /// Greedy sampling with no penalties: the verifier compares argmax tokens,
    /// so any other sampling would change the output.
    pub plain_greedy: bool,
    pub constrained: bool,
    pub logprobs: bool,
    pub has_images: bool,
}

/// The strategy a request may speculate with, if any. Prompt lookup wins when
/// both are available, as it always has.
pub const fn choose_speculation(
    settings: &SpeculationSettings,
    model_has_mtp: bool,
    request: SpeculationRequest,
) -> Option<SpeculationKind> {
    if !request.plain_greedy || request.constrained || request.logprobs || request.has_images {
        return None;
    }
    if settings.prompt_lookup {
        Some(SpeculationKind::PromptLookup)
    } else if settings.mtp && model_has_mtp {
        Some(SpeculationKind::Mtp)
    } else {
        None
    }
}

/// How many tokens one cycle may add to the completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CycleBudget {
    /// Tokens left before `max_tokens`.
    pub remaining: usize,
    /// Tokens that may be emitted before a `</think>` has to be forced;
    /// `None` when no reasoning budget applies.
    pub thinking_room: Option<usize>,
}

impl CycleBudget {
    /// Most tokens the cycle may emit: the confirmed token plus its drafts.
    fn emittable(self) -> usize {
        self.thinking_room
            .map_or(self.remaining, |room| room.min(self.remaining))
    }
}

/// A request's speculative decode state.
pub enum Speculation {
    PromptLookup(PromptLookupDecode),
    Mtp(Box<MtpDecode>),
}

impl Speculation {
    /// Enter prompt-lookup decode for a request whose cache holds everything
    /// emitted so far except its last token `last_token`.
    pub fn start_prompt_lookup(
        model: &mut AnyModel,
        cache: &mut AnyCache,
        settings: &SpeculationSettings,
        last_token: u32,
    ) -> Result<Self, EngineError> {
        PromptLookupDecode::start(model, cache, settings, last_token).map(Self::PromptLookup)
    }

    /// Enter MTP decode right after prefill.
    pub fn start_mtp(
        model: &mut AnyModel,
        cache: &mut AnyCache,
        settings: &SpeculationSettings,
        prefilled: &[u32],
        prefill_hidden: Option<&Array>,
        first_token: u32,
    ) -> Result<Self, EngineError> {
        MtpDecode::start(
            model,
            cache,
            settings,
            prefilled,
            prefill_hidden,
            first_token,
        )
        .map(|decode| Self::Mtp(Box::new(decode)))
    }

    /// A prompt-lookup state that has settled on `confirmed`, for tests of the
    /// worker's handover that have no model to start one with.
    #[cfg(test)]
    pub fn prompt_lookup_confirming(confirmed: u32) -> Self {
        Self::PromptLookup(PromptLookupDecode {
            confirmed,
            config: PromptLookupConfig::default(),
            unchecked: false,
            stats: MtpStats::default(),
            started: std::time::Instant::now(),
        })
    }

    pub const fn kind(&self) -> SpeculationKind {
        match self {
            Self::PromptLookup(_) => SpeculationKind::PromptLookup,
            Self::Mtp(_) => SpeculationKind::Mtp,
        }
    }

    /// The next token the model would produce: neither emitted nor in the cache.
    pub const fn confirmed(&self) -> u32 {
        match self {
            Self::PromptLookup(decode) => decode.confirmed,
            Self::Mtp(decode) => decode.confirmed,
        }
    }

    /// Run one cycle and return the tokens to emit, in order, starting with
    /// [`Self::confirmed`]. `None` when the budget leaves no room to draft.
    pub fn cycle(
        &mut self,
        model: &mut AnyModel,
        cache: &mut AnyCache,
        history: &[u32],
        budget: CycleBudget,
    ) -> Result<Option<Vec<u32>>, EngineError> {
        match self {
            Self::PromptLookup(decode) => decode.cycle(model, cache, history, budget),
            Self::Mtp(decode) => decode.cycle(model, cache, history, budget),
        }
    }

    pub fn log_summary(&self, reason: &str) {
        let (stats, started, label) = match self {
            Self::PromptLookup(decode) => (&decode.stats, decode.started, "Prompt-lookup"),
            Self::Mtp(decode) => (&decode.stats, decode.started, "MTP"),
        };
        let elapsed = started.elapsed().as_secs_f64().max(f64::EPSILON);
        tracing::info!(
            reason,
            cycles = stats.cycles(),
            drafted = stats.drafted(),
            accepted_drafts = stats.accepted_drafts(),
            emitted = stats.emitted(),
            accept_rate = format!("{:.1}%", stats.acceptance_rate_percent()),
            tok_per_s = format!("{:.1}", f64::from(stats.emitted()) / elapsed),
            "{label} decode complete"
        );
    }
}

pub struct PromptLookupDecode {
    confirmed: u32,
    config: PromptLookupConfig,
    unchecked: bool,
    stats: MtpStats,
    started: std::time::Instant,
}

impl PromptLookupDecode {
    fn start(
        model: &mut AnyModel,
        cache: &mut AnyCache,
        settings: &SpeculationSettings,
        last_token: u32,
    ) -> Result<Self, EngineError> {
        let input = token_input(last_token)?;
        let logits = model
            .forward_all_logits(&input, None, cache)
            .map_err(EngineError::Mlx)?;
        let next = argmax_axis(logits.index((.., -1, ..)), -1, None).map_err(EngineError::Mlx)?;
        eval([&next]).map_err(EngineError::Mlx)?;
        Ok(Self {
            confirmed: next.item_cast(),
            config: settings.prompt_lookup_config,
            unchecked: settings.unchecked_prompt_lookup,
            stats: MtpStats::default(),
            started: std::time::Instant::now(),
        })
    }

    fn cycle(
        &mut self,
        model: &mut AnyModel,
        cache: &mut AnyCache,
        history: &[u32],
        budget: CycleBudget,
    ) -> Result<Option<Vec<u32>>, EngineError> {
        let Some(max_drafts) = budget.emittable().checked_sub(1) else {
            return Ok(None);
        };
        let config = PromptLookupConfig {
            max_drafts: self.config.max_drafts.min(max_drafts),
            ..self.config
        };
        let result = if self.unchecked {
            mtp::unchecked_prompt_lookup_cycle(model, cache, history, self.confirmed, config)?
        } else {
            mtp::prompt_lookup_cycle(model, cache, history, self.confirmed, config)?
        };
        self.stats
            .record_cycle(result.drafted, result.tokens.len(), result.accepted_drafts);
        self.confirmed = result.next_token_id;
        Ok(Some(result.tokens))
    }
}

pub struct MtpDecode {
    cache: MtpCache,
    hidden: Array,
    confirmed: u32,
    adaptive_depth: Option<AdaptiveDraftDepth>,
    draft_n_max: usize,
    prompt_lookup: Option<PromptLookupConfig>,
    stats: MtpStats,
    started: std::time::Instant,
}

impl MtpDecode {
    fn start(
        model: &mut AnyModel,
        cache: &mut AnyCache,
        settings: &SpeculationSettings,
        prefilled: &[u32],
        prefill_hidden: Option<&Array>,
        first_token: u32,
    ) -> Result<Self, EngineError> {
        let mut mtp_cache = model
            .make_mtp_cache()
            .ok_or_else(|| EngineError::Generation("MTP cache creation failed".into()))?;
        if let Some(hidden) = prefill_hidden {
            mtp::prime_mtp_cache(model, &mut mtp_cache, prefilled, hidden)?;
        }

        let first_input = token_input(first_token)?;
        let (hidden, logits) = model
            .forward_with_hidden(&first_input, None, cache)
            .map_err(EngineError::Mlx)?;
        let next = argmax_axis(logits.index((.., -1, ..)), -1, None).map_err(EngineError::Mlx)?;
        let last_hidden = hidden.index((.., -1.., ..));
        if let (Some(prefill), Some(last_prefilled)) =
            (prefill_hidden, prefilled.len().checked_sub(1))
        {
            let row = i32::try_from(last_prefilled)
                .map_err(|_| EngineError::Generation("hidden row index too large".to_owned()))?;
            let previous_hidden = prefill.index((.., row..row + 1, ..));
            mtp::mirror_mtp_token(model, &mut mtp_cache, &previous_hidden, first_token)?;
        }
        eval([&next, &last_hidden]).map_err(EngineError::Mlx)?;

        Ok(Self {
            cache: mtp_cache,
            hidden: last_hidden,
            confirmed: next.item_cast(),
            adaptive_depth: settings.mtp_adaptive_draft.then(|| {
                AdaptiveDraftDepth::new(settings.mtp_draft_n_max, settings.mtp_draft_n_max)
            }),
            draft_n_max: settings.mtp_draft_n_max,
            prompt_lookup: settings
                .mtp_prompt_lookup
                .then_some(settings.prompt_lookup_config),
            stats: MtpStats::default(),
            started: std::time::Instant::now(),
        })
    }

    fn cycle(
        &mut self,
        model: &mut AnyModel,
        cache: &mut AnyCache,
        history: &[u32],
        budget: CycleBudget,
    ) -> Result<Option<Vec<u32>>, EngineError> {
        // The MTP head always drafts at least one token.
        let Some(max_drafts) = budget
            .emittable()
            .checked_sub(1)
            .filter(|&drafts| drafts > 0)
        else {
            return Ok(None);
        };
        let depth = self
            .adaptive_depth
            .as_ref()
            .map_or(self.draft_n_max, AdaptiveDraftDepth::current)
            .min(max_drafts);
        let lookup = self
            .prompt_lookup
            .map(|config| PromptLookupConfig {
                max_drafts: config.max_drafts.min(max_drafts),
                ..config
            })
            .filter(|config| config.max_drafts > 0);
        let looked_up = match lookup {
            Some(config) => mtp::mtp_prompt_lookup_cycle(
                model,
                cache,
                &mut self.cache,
                &self.hidden,
                history,
                self.confirmed,
                config,
            )?,
            None => None,
        };
        let result = match looked_up {
            Some(result) => result,
            None => mtp::mtp_cycle(
                model,
                cache,
                &mut self.cache,
                &self.hidden,
                self.confirmed,
                depth,
            )?,
        };
        self.stats
            .record_cycle(result.drafted, result.tokens.len(), result.accepted_drafts);
        if let Some(adaptive) = &mut self.adaptive_depth {
            adaptive.observe(result.accepted_drafts, result.drafted);
        }
        self.hidden = result.hidden;
        self.confirmed = result.next_token_id;
        Ok(Some(result.tokens))
    }
}

fn token_input(token: u32) -> Result<Array, EngineError> {
    let id = i32::try_from(token)
        .map_err(|_| EngineError::Generation("token id exceeds i32 range".to_owned()))?;
    Ok(Array::from_slice(&[id], &[1, 1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(prompt_lookup: bool, mtp: bool) -> SpeculationSettings {
        SpeculationSettings {
            prompt_lookup,
            mtp,
            prime_mtp_from_prefill: true,
            mtp_adaptive_draft: false,
            mtp_prompt_lookup: false,
            mtp_draft_n_max: 2,
            unchecked_prompt_lookup: false,
            prompt_lookup_config: PromptLookupConfig::default(),
        }
    }

    const GREEDY: SpeculationRequest = SpeculationRequest {
        plain_greedy: true,
        constrained: false,
        logprobs: false,
        has_images: false,
    };

    #[test]
    fn nothing_speculates_unless_switched_on() {
        assert_eq!(
            choose_speculation(&settings(false, false), true, GREEDY),
            None
        );
    }

    #[test]
    fn prompt_lookup_wins_over_mtp() {
        assert_eq!(
            choose_speculation(&settings(true, true), true, GREEDY),
            Some(SpeculationKind::PromptLookup)
        );
    }

    #[test]
    fn mtp_needs_a_head_in_the_checkpoint() {
        assert_eq!(
            choose_speculation(&settings(false, true), true, GREEDY),
            Some(SpeculationKind::Mtp)
        );
        assert_eq!(
            choose_speculation(&settings(false, true), false, GREEDY),
            None
        );
    }

    #[test]
    fn only_plain_greedy_requests_speculate() {
        let on = settings(true, true);
        for request in [
            SpeculationRequest {
                plain_greedy: false,
                ..GREEDY
            },
            SpeculationRequest {
                constrained: true,
                ..GREEDY
            },
            SpeculationRequest {
                logprobs: true,
                ..GREEDY
            },
            SpeculationRequest {
                has_images: true,
                ..GREEDY
            },
        ] {
            assert_eq!(choose_speculation(&on, true, request), None);
        }
    }

    #[test]
    fn a_cycle_may_not_outrun_the_completion_or_the_reasoning_budget() {
        let budget = |remaining, thinking_room| CycleBudget {
            remaining,
            thinking_room,
        };
        assert_eq!(budget(10, None).emittable(), 10);
        assert_eq!(budget(10, Some(4)).emittable(), 4);
        assert_eq!(budget(3, Some(4)).emittable(), 3);
        assert_eq!(budget(0, None).emittable(), 0);
    }
}
