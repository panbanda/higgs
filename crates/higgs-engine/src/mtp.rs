//! MTP (Multi-Token Prediction) speculative decode.
//!
//! Uses the model's built-in MTP head to draft tokens, then verifies them by
//! processing only committed tokens through the backbone.
//!
//! Expected speedup: ~1.5x on dense models at ~80% acceptance rate.

use higgs_models::{AnyCache, AnyModel, MtpCache};
use mlx_rs::{Array, argmax_axis, ops::indexing::IndexOp, transforms::eval};

use crate::error::EngineError;

const fn draft_matches_target(draft_token_id: u32, target_id: u32) -> bool {
    draft_token_id == target_id
}

/// Aggregate MTP decode counters.
#[derive(Debug, Default, Clone)]
pub struct MtpStats {
    cycles: u32,
    drafted: u32,
    accepted_drafts: u32,
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

fn backbone_step(
    model: &mut AnyModel,
    cache: &mut AnyCache,
    token_id: u32,
) -> Result<(Array, u32), EngineError> {
    let input = Array::from_slice(&[token_id.cast_signed()], &[1, 1]);
    let (hidden, logits) = model
        .forward_with_hidden(&input, None, cache)
        .map_err(EngineError::Mlx)?;
    let target_arr = argmax_axis!(&logits.index((.., -1, ..)), -1).map_err(EngineError::Mlx)?;
    let h_last = hidden.index((.., -1.., ..));
    eval([&target_arr, &h_last]).map_err(EngineError::Mlx)?;
    Ok((h_last, target_arr.item()))
}

/// Run one MTP speculative decode cycle.
///
/// Given the backbone's hidden state at position t and the confirmed token t+1:
/// 1. MTP drafts up to `draft_n_max` future tokens.
/// 2. The backbone verifies drafts one by one.
/// 3. Only accepted tokens are committed to the backbone and MTP caches.
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
    let base_mtp_cache = mtp_cache.clone();
    let mut speculative_mtp_cache = base_mtp_cache.clone();
    let mut speculative_hidden = hidden.clone();
    let mut speculative_token = confirmed_token_id;
    let mut drafts = Vec::with_capacity(draft_limit);

    for _ in 0..draft_limit {
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
    }

    let mut tokens = vec![confirmed_token_id];
    let mut replay_steps: Vec<(Array, u32)> = vec![(hidden.clone(), confirmed_token_id)];
    let first_draft = *drafts
        .first()
        .ok_or_else(|| EngineError::Generation("MTP produced no draft tokens".to_owned()))?;

    let (mut hidden_after, mut target_id) = backbone_step(model, cache, confirmed_token_id)?;
    if !draft_matches_target(first_draft, target_id) {
        if drafts.len() == 1 {
            *mtp_cache = speculative_mtp_cache;
            return Ok(MtpCycleResult {
                tokens,
                hidden: hidden_after,
                next_token_id: target_id,
                drafted: drafts.len(),
                accepted_drafts: 0,
            });
        }

        *mtp_cache = base_mtp_cache;
        for (step_hidden, step_token) in replay_steps {
            model
                .mtp_advance(&step_hidden, step_token, mtp_cache)
                .map_err(EngineError::Mlx)?;
        }
        return Ok(MtpCycleResult {
            tokens,
            hidden: hidden_after,
            next_token_id: target_id,
            drafted: drafts.len(),
            accepted_drafts: 0,
        });
    }

    tokens.push(first_draft);
    let mut accepted_drafts = 1usize;
    let mut token_to_process = first_draft;
    let mut hidden_before_token = hidden_after;

    for &draft in drafts.iter().skip(1) {
        replay_steps.push((hidden_before_token.clone(), token_to_process));
        (hidden_after, target_id) = backbone_step(model, cache, token_to_process)?;
        if !draft_matches_target(draft, target_id) {
            *mtp_cache = base_mtp_cache;
            for (step_hidden, step_token) in replay_steps {
                model
                    .mtp_advance(&step_hidden, step_token, mtp_cache)
                    .map_err(EngineError::Mlx)?;
            }
            return Ok(MtpCycleResult {
                tokens,
                hidden: hidden_after,
                next_token_id: target_id,
                drafted: drafts.len(),
                accepted_drafts,
            });
        }

        tokens.push(draft);
        accepted_drafts += 1;
        token_to_process = draft;
        hidden_before_token = hidden_after;
    }

    if drafts.len() == 1 {
        let (h_last, bonus_id) = backbone_step(model, cache, token_to_process)?;
        *mtp_cache = speculative_mtp_cache;
        model
            .mtp_advance(&hidden_before_token, token_to_process, mtp_cache)
            .map_err(EngineError::Mlx)?;
        return Ok(MtpCycleResult {
            tokens,
            hidden: h_last,
            next_token_id: bonus_id,
            drafted: drafts.len(),
            accepted_drafts,
        });
    }

    replay_steps.push((hidden_before_token, token_to_process));
    let (h_last, bonus_id) = backbone_step(model, cache, token_to_process)?;

    *mtp_cache = base_mtp_cache;
    for (step_hidden, step_token) in replay_steps {
        model
            .mtp_advance(&step_hidden, step_token, mtp_cache)
            .map_err(EngineError::Mlx)?;
    }

    Ok(MtpCycleResult {
        tokens,
        hidden: h_last,
        next_token_id: bonus_id,
        drafted: drafts.len(),
        accepted_drafts,
    })
}

#[cfg(test)]
mod tests {
    use super::{MtpStats, draft_matches_target};

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
        assert_eq!(stats.acceptance_rate_percent(), 60.0);
    }
}
