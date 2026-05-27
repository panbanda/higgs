//! MTP (Multi-Token Prediction) speculative decode.
//!
//! Uses the model's built-in MTP head to draft tokens, then verifies them by
//! processing the verifier window through the backbone in one batch and rolling
//! back to the committed prefix on rejection.
//!
//! Expected speedup: ~1.5x on dense models at ~80% acceptance rate.

use higgs_models::{AnyCache, AnyModel, MtpCache};
use mlx_rs::{Array, argmax_axis, ops::indexing::IndexOp, transforms::eval};

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

    for (idx, &token) in drafts.iter().take(accepted_drafts).enumerate() {
        let hidden_before = hidden_row(accepted_hidden_rows, idx)?;
        model
            .mtp_advance(&hidden_before, token, mtp_cache)
            .map_err(EngineError::Mlx)?;
    }

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
    use super::{MtpStats, accepted_draft_prefix_len, draft_matches_target, emitted_tokens};

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
}
