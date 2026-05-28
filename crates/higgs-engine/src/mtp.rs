//! MTP (Multi-Token Prediction) speculative decode.
//!
//! Uses the model's built-in MTP head to draft tokens, then verifies them by
//! processing the verifier window through the backbone in one batch and rolling
//! back to the committed prefix on rejection.
//!
//! Expected speedup: ~1.5x on dense models at ~80% acceptance rate.

use higgs_models::{AnyCache, AnyModel, MtpCache};
use mlx_rs::{
    Array, argmax_axis,
    ops::{self, concatenate_axis, indexing::IndexOp},
    transforms::eval,
};

use crate::error::EngineError;

const fn draft_matches_target(draft_token_id: u32, target_id: u32) -> bool {
    draft_token_id == target_id
}

/// Aggregate MTP decode counters.
///
/// Tracks per-cycle telemetry for MTP speculative decoding.
#[derive(Debug, Default, Clone)]
pub struct MtpStats {
    /// Number of speculative decode cycles executed.
    cycles: u32,
    /// Total speculative tokens drafted by the MTP head.
    drafted: u32,
    /// Drafted tokens that matched the backbone verifier.
    accepted_drafts: u32,
    /// Tokens emitted by MTP cycles, including confirmed tokens and accepted drafts.
    emitted: u32,
}

impl MtpStats {
    pub fn record_cycle(&mut self, drafted_count: usize, emitted_count: usize) {
        let drafted = u32::try_from(drafted_count).unwrap_or(u32::MAX);
        let emitted = u32::try_from(emitted_count).unwrap_or(u32::MAX);
        self.cycles = self.cycles.saturating_add(1);
        self.drafted = self.drafted.saturating_add(drafted);
        self.emitted = self.emitted.saturating_add(emitted);
        self.accepted_drafts = self
            .accepted_drafts
            .saturating_add(emitted.saturating_sub(1).min(drafted));
    }

    pub const fn cycles(&self) -> u32 {
        self.cycles
    }

    pub const fn drafted(&self) -> u32 {
        self.drafted
    }

    pub const fn accepted_drafts(&self) -> u32 {
        self.accepted_drafts
    }

    pub const fn emitted(&self) -> u32 {
        self.emitted
    }

    #[allow(clippy::cast_precision_loss)]
    pub fn acceptance_rate_percent(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            f64::from(self.accepted_drafts) * 100.0 / f64::from(self.drafted)
        }
    }
}

/// Result of a single MTP speculative decode cycle.
pub struct MtpCycleResult {
    /// Token IDs accepted this cycle (the confirmed token plus accepted drafts).
    pub tokens: Vec<u32>,
    /// Hidden state at the last accepted position (for next MTP draft).
    pub hidden: Array,
    /// The next confirmed token to process in the following cycle.
    pub next_token_id: u32,
    /// Number of speculative draft tokens produced this cycle.
    pub drafted: usize,
    /// Number of speculative draft tokens accepted this cycle.
    pub accepted_drafts: usize,
}

/// Prompt-lookup speculative decode settings.
#[derive(Debug, Clone, Copy)]
pub struct PromptLookupConfig {
    pub max_drafts: usize,
    pub max_ngram: usize,
    pub max_window: usize,
}

impl Default for PromptLookupConfig {
    fn default() -> Self {
        Self {
            max_drafts: 6,
            max_ngram: 8,
            max_window: 2048,
        }
    }
}

/// Result of one architecture-neutral prompt-lookup speculative cycle.
pub struct PromptLookupCycleResult {
    /// Token IDs accepted this cycle (the confirmed token plus accepted drafts).
    pub tokens: Vec<u32>,
    /// The next confirmed token to process in the following cycle.
    pub next_token_id: u32,
    /// Number of prompt-lookup draft tokens proposed this cycle.
    pub drafted: usize,
    /// Number of prompt-lookup draft tokens accepted this cycle.
    pub accepted_drafts: usize,
}

fn greedy_token_id(logits: &Array) -> Result<u32, EngineError> {
    let token_arr = argmax_axis!(&logits.index((.., -1, ..)), -1).map_err(EngineError::Mlx)?;
    eval([&token_arr]).map_err(EngineError::Mlx)?;
    Ok(token_arr.item())
}

fn greedy_token_ids(logits: &Array) -> Result<Vec<u32>, EngineError> {
    let token_arr = argmax_axis!(logits, -1).map_err(EngineError::Mlx)?;
    eval([&token_arr]).map_err(EngineError::Mlx)?;
    Ok(token_arr.as_slice::<u32>().to_vec())
}

fn parse_enabled_flag(raw: Option<&str>) -> Option<bool> {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("1" | "true" | "on" | "yes") => Some(true),
        Some("0" | "false" | "off" | "no") => Some(false),
        _ => None,
    }
}

fn mtp_mirror_verify_enabled() -> bool {
    parse_enabled_flag(std::env::var("HIGGS_MTP_MIRROR_VERIFY").ok().as_deref()).unwrap_or(false)
}

fn accepted_draft_prefix_len(drafts: &[u32], verifier_targets: &[u32]) -> usize {
    drafts
        .iter()
        .zip(verifier_targets.iter())
        .take_while(|(draft, target)| draft_matches_target(**draft, **target))
        .count()
}

fn emitted_tokens(confirmed_token_id: u32, drafts: &[u32], accepted_drafts: usize) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(accepted_drafts.saturating_add(1));
    tokens.push(confirmed_token_id);
    tokens.extend(drafts.iter().take(accepted_drafts).copied());
    tokens
}

pub fn prompt_lookup_draft(
    context: &[u32],
    max_drafts: usize,
    max_ngram: usize,
    max_window: usize,
) -> Vec<u32> {
    if context.is_empty() || max_drafts == 0 || max_ngram == 0 {
        return Vec::new();
    }

    let end = context.len();
    let capped_ngram = max_ngram.min(end);
    let search_start = end.saturating_sub(max_window.max(1));

    for ngram in (1..=capped_ngram).rev() {
        let Some(suffix) = context.get(end - ngram..end) else {
            continue;
        };
        let search_end = end.saturating_sub(ngram);

        for pos in (search_start..search_end).rev() {
            let match_end = pos + ngram;
            if context.get(pos..match_end) != Some(suffix) {
                continue;
            }

            let draft_start = match_end;
            if draft_start >= end {
                continue;
            }
            let draft_end = draft_start.saturating_add(max_drafts).min(end);
            if let Some(draft) = context.get(draft_start..draft_end) {
                return draft.to_vec();
            }
        }
    }

    Vec::new()
}

/// Run one prompt-lookup speculative decode cycle.
///
/// This is architecture-neutral: the draft provider only copies tokens from
/// prior prompt/history, and the model verifies `[confirmed + drafts]` in one
/// forward pass using all-position logits.
pub fn prompt_lookup_cycle(
    model: &mut AnyModel,
    cache: &mut AnyCache,
    history_before_confirmed: &[u32],
    confirmed_token_id: u32,
    config: PromptLookupConfig,
) -> Result<PromptLookupCycleResult, EngineError> {
    let mut lookup_context = Vec::with_capacity(history_before_confirmed.len().saturating_add(1));
    lookup_context.extend_from_slice(history_before_confirmed);
    lookup_context.push(confirmed_token_id);
    let drafts = prompt_lookup_draft(
        &lookup_context,
        config.max_drafts,
        config.max_ngram,
        config.max_window,
    );

    let base_cache = cache.clone();
    let mut verify_tokens = Vec::with_capacity(drafts.len().saturating_add(1));
    verify_tokens.push(confirmed_token_id);
    verify_tokens.extend(drafts.iter().copied());

    let logits = model
        .forward_all_logits(&token_input(&verify_tokens)?, None, cache)
        .map_err(EngineError::Mlx)?;
    let verifier_targets = greedy_token_ids(&logits)?;
    if verifier_targets.len() < verify_tokens.len() {
        return Err(EngineError::Generation(format!(
            "prompt-lookup verifier returned {} target ids for {} input tokens",
            verifier_targets.len(),
            verify_tokens.len()
        )));
    }

    let accepted_drafts = accepted_draft_prefix_len(&drafts, &verifier_targets);
    let tokens = emitted_tokens(confirmed_token_id, &drafts, accepted_drafts);

    let next_token_id = if accepted_drafts == drafts.len() {
        *verifier_targets.get(accepted_drafts).ok_or_else(|| {
            EngineError::Generation(format!(
                "prompt-lookup verifier missing target at accepted index {accepted_drafts}"
            ))
        })?
    } else {
        *cache = base_cache;
        let replay_logits = model
            .forward_all_logits(&token_input(&tokens)?, None, cache)
            .map_err(EngineError::Mlx)?;
        let replay_targets = greedy_token_ids(&replay_logits)?;
        *replay_targets.get(accepted_drafts).ok_or_else(|| {
            EngineError::Generation(format!(
                "prompt-lookup replay returned {} target ids for accepted index {}",
                replay_targets.len(),
                accepted_drafts
            ))
        })?
    };

    Ok(PromptLookupCycleResult {
        tokens,
        next_token_id,
        drafted: drafts.len(),
        accepted_drafts,
    })
}

/// Run one unchecked prompt-lookup cycle.
///
/// This path copies draft tokens from prompt/history without per-token verifier
/// logits. It still advances the target model cache over the emitted span and
/// samples the next token from the final position, but it is not guaranteed to
/// reproduce greedy decode if the copied tokens would have been rejected.
pub fn unchecked_prompt_lookup_cycle(
    model: &mut AnyModel,
    cache: &mut AnyCache,
    history_before_confirmed: &[u32],
    confirmed_token_id: u32,
    config: PromptLookupConfig,
) -> Result<PromptLookupCycleResult, EngineError> {
    let mut lookup_context = Vec::with_capacity(history_before_confirmed.len().saturating_add(1));
    lookup_context.extend_from_slice(history_before_confirmed);
    lookup_context.push(confirmed_token_id);
    let drafts = prompt_lookup_draft(
        &lookup_context,
        config.max_drafts,
        config.max_ngram,
        config.max_window,
    );

    let mut tokens = Vec::with_capacity(drafts.len().saturating_add(1));
    tokens.push(confirmed_token_id);
    tokens.extend(drafts.iter().copied());

    let logits = model
        .forward_last_token(&token_input(&tokens)?, None, cache)
        .map_err(EngineError::Mlx)?;
    let next_token_id = greedy_token_id(&logits)?;

    Ok(PromptLookupCycleResult {
        tokens,
        next_token_id,
        drafted: drafts.len(),
        accepted_drafts: drafts.len(),
    })
}

fn token_input(tokens: &[u32]) -> Result<Array, EngineError> {
    let mut input = Vec::with_capacity(tokens.len());
    for &token in tokens {
        input.push(
            i32::try_from(token)
                .map_err(|_| EngineError::Generation("token id exceeds i32 range".to_owned()))?,
        );
    }
    let len = i32::try_from(input.len())
        .map_err(|_| EngineError::Generation("token batch too large".to_owned()))?;
    Ok(Array::from_slice(&input, &[1, len]))
}

fn hidden_row(hidden: &Array, row: usize) -> Result<Array, EngineError> {
    let row_i32 = i32::try_from(row)
        .map_err(|_| EngineError::Generation("hidden row index too large".to_owned()))?;
    Ok(hidden.index((.., row_i32..row_i32 + 1, ..)))
}

fn hidden_rows(hidden: &Array, start: usize, end: usize) -> Result<Array, EngineError> {
    let start_i32 = i32::try_from(start)
        .map_err(|_| EngineError::Generation("hidden row start index too large".to_owned()))?;
    let end_i32 = i32::try_from(end)
        .map_err(|_| EngineError::Generation("hidden row end index too large".to_owned()))?;
    Ok(hidden.index((.., start_i32..end_i32, ..)))
}

fn zero_hidden_row_like(hidden: &Array) -> Result<Array, EngineError> {
    let shape = hidden.shape();
    let batch = *shape
        .first()
        .ok_or_else(|| EngineError::Generation("hidden tensor missing batch dim".to_owned()))?;
    let hidden_dim = *shape
        .get(2)
        .ok_or_else(|| EngineError::Generation("hidden tensor missing hidden dim".to_owned()))?;
    ops::zeros_dtype(&[batch, 1, hidden_dim], hidden.dtype()).map_err(EngineError::Mlx)
}

fn shifted_hidden_rows(
    initial_hidden: &Array,
    hidden: &Array,
    count: usize,
) -> Result<Array, EngineError> {
    if count == 0 {
        return Err(EngineError::Generation(
            "cannot build shifted hidden rows for empty token batch".to_owned(),
        ));
    }
    if count == 1 {
        return Ok(initial_hidden.clone());
    }

    let tail = hidden_rows(hidden, 0, count - 1)?;
    concatenate_axis(&[initial_hidden, &tail], 1).map_err(EngineError::Mlx)
}

/// Prime an MTP cache from a backbone hidden sequence.
///
/// `hidden` must contain the raw backbone hidden states for `tokens`.
/// The first MTP row uses a zero previous-hidden row, matching llama.cpp's
/// draft-mtp prompt mirroring behavior.
pub fn prime_mtp_cache(
    model: &mut AnyModel,
    mtp_cache: &mut MtpCache,
    tokens: &[u32],
    hidden: &Array,
) -> Result<(), EngineError> {
    if tokens.is_empty() {
        return Ok(());
    }

    let zero = zero_hidden_row_like(hidden)?;
    let shifted = shifted_hidden_rows(&zero, hidden, tokens.len())?;
    model
        .mtp_advance_many(&shifted, tokens, mtp_cache)
        .map_err(EngineError::Mlx)
}

/// Mirror one accepted backbone token into an already-primed MTP cache.
pub fn mirror_mtp_token(
    model: &mut AnyModel,
    mtp_cache: &mut MtpCache,
    previous_hidden: &Array,
    token: u32,
) -> Result<(), EngineError> {
    model
        .mtp_advance_many(previous_hidden, &[token], mtp_cache)
        .map_err(EngineError::Mlx)
}

fn backbone_verify_batch(
    model: &mut AnyModel,
    cache: &mut AnyCache,
    tokens: &[u32],
) -> Result<(Array, Vec<u32>), EngineError> {
    let input = token_input(tokens)?;
    let (hidden, logits) = model
        .forward_with_hidden(&input, None, cache)
        .map_err(EngineError::Mlx)?;
    let target_ids = greedy_token_ids(&logits)?;
    Ok((hidden, target_ids))
}

fn commit_mtp_cache(
    model: &mut AnyModel,
    mtp_cache: &mut MtpCache,
    confirmed_mtp_cache: MtpCache,
    accepted_hidden_rows: &Array,
    drafts: &[u32],
    accepted_drafts: usize,
) -> Result<(), EngineError> {
    *mtp_cache = confirmed_mtp_cache;

    if accepted_drafts > 0 {
        let accepted = drafts.get(..accepted_drafts).ok_or_else(|| {
            EngineError::Generation(format!(
                "MTP cache commit missing accepted draft prefix len {accepted_drafts}"
            ))
        })?;
        let hidden_before = hidden_rows(accepted_hidden_rows, 0, accepted_drafts)?;
        model
            .mtp_advance_many(&hidden_before, accepted, mtp_cache)
            .map_err(EngineError::Mlx)?;
    }

    Ok(())
}

fn trim_mtp_cache_by(mtp_cache: &mut MtpCache, rejected: usize) {
    if rejected == 0 {
        return;
    }

    for layer in mtp_cache {
        layer.trim_by(rejected);
    }
}

fn mirror_verified_mtp_cache(
    model: &mut AnyModel,
    mtp_cache: &mut MtpCache,
    base_mtp_cache: MtpCache,
    previous_hidden: &Array,
    verify_hidden: &Array,
    verify_tokens: &[u32],
    accepted_token_count: usize,
) -> Result<(), EngineError> {
    let mut mirrored = base_mtp_cache;
    let shifted = shifted_hidden_rows(previous_hidden, verify_hidden, verify_tokens.len())?;
    model
        .mtp_advance_many(&shifted, verify_tokens, &mut mirrored)
        .map_err(EngineError::Mlx)?;

    let rejected = verify_tokens.len().saturating_sub(accepted_token_count);
    trim_mtp_cache_by(&mut mirrored, rejected);
    *mtp_cache = mirrored;

    Ok(())
}

/// Run one MTP speculative decode cycle.
///
/// Given the backbone's hidden state at position t and the confirmed token t+1:
/// 1. MTP drafts up to `draft_n_max` future tokens.
/// 2. The backbone verifies the confirmed token plus all drafts in one batch.
/// 3. The caches are kept on full acceptance or rebuilt from the accepted prefix
///    after a rejection.
#[allow(clippy::too_many_lines)]
pub fn mtp_cycle(
    model: &mut AnyModel,
    cache: &mut AnyCache,
    mtp_cache: &mut MtpCache,
    hidden: &Array,
    confirmed_token_id: u32,
    draft_n_max: usize,
) -> Result<MtpCycleResult, EngineError> {
    let draft_limit = draft_n_max.max(1);
    let base_cache = cache.clone();
    let base_mtp_cache = mtp_cache.clone();
    let mut speculative_mtp_cache = mtp_cache.clone();
    let mut confirmed_mtp_cache: Option<MtpCache> = None;
    let mut speculative_hidden = hidden.clone();
    let mut speculative_token = confirmed_token_id;
    let mut drafts = Vec::with_capacity(draft_limit);

    for draft_idx in 0..draft_limit {
        let (next_hidden, draft_logits) = model
            .mtp_draft_with_hidden(
                &speculative_hidden,
                speculative_token,
                &mut speculative_mtp_cache,
            )
            .map_err(EngineError::Mlx)?;
        let draft_token_id = greedy_token_id(&draft_logits)?;
        drafts.push(draft_token_id);
        speculative_hidden = next_hidden;
        speculative_token = draft_token_id;
        if draft_idx == 0 {
            confirmed_mtp_cache = Some(speculative_mtp_cache.clone());
        }
    }

    let first_draft = *drafts
        .first()
        .ok_or_else(|| EngineError::Generation("MTP produced no draft tokens".to_owned()))?;

    let mut verify_tokens = Vec::with_capacity(drafts.len().saturating_add(1));
    verify_tokens.push(confirmed_token_id);
    verify_tokens.extend(drafts.iter().copied());

    let (verify_hidden, verifier_targets) = backbone_verify_batch(model, cache, &verify_tokens)?;
    let verify_hidden_for_mtp = verify_hidden.clone();
    if verifier_targets.len() < verify_tokens.len() {
        return Err(EngineError::Generation(format!(
            "batched MTP verifier returned {} target ids for {} input tokens",
            verifier_targets.len(),
            verify_tokens.len()
        )));
    }

    let first_target = *verifier_targets
        .first()
        .ok_or_else(|| EngineError::Generation("MTP verifier returned no targets".to_owned()))?;
    let accepted_drafts = if draft_matches_target(first_draft, first_target) {
        accepted_draft_prefix_len(&drafts, &verifier_targets)
    } else {
        0
    };
    let tokens = emitted_tokens(confirmed_token_id, &drafts, accepted_drafts);

    let (accepted_hidden_rows, next_token_id) = if accepted_drafts == drafts.len() {
        let next = *verifier_targets.get(accepted_drafts).ok_or_else(|| {
            EngineError::Generation(format!(
                "MTP verifier missing target at accepted index {accepted_drafts}"
            ))
        })?;
        (verify_hidden, next)
    } else {
        *cache = base_cache;
        let (replay_hidden, replay_targets) = backbone_verify_batch(model, cache, &tokens)?;
        let next = *replay_targets.get(accepted_drafts).ok_or_else(|| {
            EngineError::Generation(format!(
                "MTP replay returned {} target ids for accepted index {}",
                replay_targets.len(),
                accepted_drafts
            ))
        })?;
        (replay_hidden, next)
    };

    let h_last = hidden_row(&accepted_hidden_rows, accepted_drafts)?;
    if mtp_mirror_verify_enabled() {
        mirror_verified_mtp_cache(
            model,
            mtp_cache,
            base_mtp_cache,
            hidden,
            &verify_hidden_for_mtp,
            &verify_tokens,
            tokens.len(),
        )?;
    } else {
        commit_mtp_cache(
            model,
            mtp_cache,
            confirmed_mtp_cache.ok_or_else(|| {
                EngineError::Generation("MTP produced no cache checkpoint".to_owned())
            })?,
            &accepted_hidden_rows,
            &drafts,
            accepted_drafts,
        )?;
    }

    if accepted_drafts < drafts.len() && tokens.is_empty() {
        return Err(EngineError::Generation(
            "MTP accepted no committed tokens".to_owned(),
        ));
    }

    Ok(MtpCycleResult {
        tokens,
        hidden: h_last,
        next_token_id,
        drafted: drafts.len(),
        accepted_drafts,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        MtpStats, accepted_draft_prefix_len, draft_matches_target, emitted_tokens,
        prompt_lookup_draft,
    };

    #[test]
    fn draft_match_helper_accepts_identical_tokens() {
        assert!(draft_matches_target(17, 17));
    }

    #[test]
    fn draft_match_helper_rejects_different_tokens() {
        assert!(!draft_matches_target(17, 18));
    }

    #[test]
    fn mtp_stats_tracks_drafted_and_bonus_acceptance_rate() {
        let mut stats = MtpStats::default();
        stats.record_cycle(3, 4);
        stats.record_cycle(2, 1);

        assert_eq!(stats.cycles(), 2);
        assert_eq!(stats.drafted(), 5);
        assert_eq!(stats.emitted(), 5);
        assert_eq!(stats.accepted_drafts(), 3);
        assert!((stats.acceptance_rate_percent() - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn accepted_draft_prefix_len_stops_at_first_mismatch() {
        let drafts = [10, 20, 30];
        let verifier_targets = [10, 21, 30, 40];

        assert_eq!(accepted_draft_prefix_len(&drafts, &verifier_targets), 1);
    }

    #[test]
    fn accepted_draft_prefix_len_accepts_full_prefix() {
        let drafts = [10, 20, 30];
        let verifier_targets = [10, 20, 30, 40];

        assert_eq!(accepted_draft_prefix_len(&drafts, &verifier_targets), 3);
    }

    #[test]
    fn emitted_tokens_includes_confirmed_and_accepted_drafts() {
        let drafts = [10, 20, 30];

        assert_eq!(emitted_tokens(7, &drafts, 2), vec![7, 10, 20]);
    }

    #[test]
    fn prompt_lookup_drafts_from_longest_prior_suffix_match() {
        let context = [1, 2, 3, 4, 5, 1, 2];

        assert_eq!(prompt_lookup_draft(&context, 3, 4, 64), vec![3, 4, 5]);
    }

    #[test]
    fn prompt_lookup_caps_drafts() {
        let context = [9, 8, 7, 6, 9, 8];

        assert_eq!(prompt_lookup_draft(&context, 1, 3, 64), vec![7]);
    }

    #[test]
    fn prompt_lookup_ignores_current_tail_self_match() {
        let context = [1, 2, 3, 4];

        assert!(prompt_lookup_draft(&context, 3, 4, 64).is_empty());
    }

    #[test]
    fn prompt_lookup_respects_search_window() {
        let context = [1, 2, 3, 4, 5, 1, 2];

        assert!(prompt_lookup_draft(&context, 3, 4, 3).is_empty());
    }
}
