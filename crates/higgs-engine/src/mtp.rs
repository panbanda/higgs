//! MTP (Multi-Token Prediction) speculative decode.
//!
//! Uses the model's built-in MTP head to draft one extra token per cycle,
//! then verifies by processing the confirmed token through the backbone.
//!
//! Expected speedup: ~1.5x on dense models at ~80% acceptance rate.

use higgs_models::{AnyCache, AnyModel, MtpCache};
use mlx_rs::{Array, argmax_axis, ops::indexing::IndexOp, transforms::eval};

use crate::error::EngineError;

/// Result of a single MTP speculative decode cycle.
pub struct MtpCycleResult {
    /// Token IDs accepted this cycle (1 or 2).
    pub tokens: Vec<u32>,
    /// Hidden state at the last accepted position (for next MTP draft).
    pub hidden: Array,
    /// The next confirmed token to process in the following cycle.
    pub next_token_id: u32,
}

/// Run one MTP speculative decode cycle.
///
/// Given the backbone's hidden state at position t and the confirmed token t+1:
/// 1. MTP drafts token t+2
/// 2. Process confirmed token through backbone → get prediction + hidden
/// 3. Check if backbone's prediction matches the draft
/// 4. If accepted: process draft through backbone too, return both tokens
/// 5. If rejected: return just confirmed (draft was never processed, no rollback needed)
#[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
pub fn mtp_cycle(
    model: &mut AnyModel,
    cache: &mut AnyCache,
    mtp_cache: &mut MtpCache,
    hidden: &Array,
    confirmed_token_id: u32,
) -> Result<MtpCycleResult, EngineError> {
    // Step 1: MTP draft — predict token at position t+2
    let draft_logits = model
        .mtp_draft(hidden, confirmed_token_id, mtp_cache)
        .map_err(EngineError::Mlx)?;
    let draft_token_arr =
        argmax_axis!(&draft_logits.index((.., -1, ..)), -1).map_err(EngineError::Mlx)?;
    eval([&draft_token_arr]).map_err(EngineError::Mlx)?;
    let draft_token_id: u32 = draft_token_arr.item();

    // Step 2: Process confirmed token through backbone
    let confirmed_input = Array::from_slice(&[confirmed_token_id as i32], &[1, 1]);
    let (confirmed_hidden, confirmed_logits) = model
        .forward_with_hidden(&confirmed_input, None, cache)
        .map_err(EngineError::Mlx)?;

    let target_arr =
        argmax_axis!(&confirmed_logits.index((.., -1, ..)), -1).map_err(EngineError::Mlx)?;
    let h_confirmed = confirmed_hidden.index((.., -1.., ..));
    eval([&target_arr, &h_confirmed]).map_err(EngineError::Mlx)?;
    let target_id: u32 = target_arr.item();

    if target_id == draft_token_id {
        // ACCEPT: draft matches backbone prediction.
        // Keep the MTP cache aligned with the newly accepted draft token too.
        model
            .mtp_advance(&h_confirmed, draft_token_id, mtp_cache)
            .map_err(EngineError::Mlx)?;

        // Process draft token to advance cache and get bonus prediction.
        let draft_input = Array::from_slice(&[draft_token_id as i32], &[1, 1]);
        let (draft_hidden, draft_logits) = model
            .forward_with_hidden(&draft_input, None, cache)
            .map_err(EngineError::Mlx)?;

        let bonus_token =
            argmax_axis!(&draft_logits.index((.., -1, ..)), -1).map_err(EngineError::Mlx)?;
        let h_last = draft_hidden.index((.., -1.., ..));
        eval([&bonus_token, &h_last]).map_err(EngineError::Mlx)?;
        let bonus_id: u32 = bonus_token.item();

        Ok(MtpCycleResult {
            tokens: vec![confirmed_token_id, draft_token_id],
            hidden: h_last,
            next_token_id: bonus_id,
        })
    } else {
        // REJECT: keep the MTP cache entry for the confirmed token.
        // `mtp_draft()` advanced the speculative head using the confirmed token,
        // not the rejected draft token, so rolling it back would drop real history.

        Ok(MtpCycleResult {
            tokens: vec![confirmed_token_id],
            hidden: h_confirmed,
            next_token_id: target_id,
        })
    }
}
