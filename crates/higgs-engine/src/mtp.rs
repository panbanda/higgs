//! MTP (Multi-Token Prediction) speculative decode.
//!
//! Uses the model's built-in MTP head to draft tokens, then verifies them by
//! processing the verifier window through the backbone in one batch and rolling
//! back to the committed prefix on rejection.
//!
//! Expected speedup: ~1.5x on dense models at ~80% acceptance rate.

use higgs_models::mlx_exec::eval;
use higgs_models::{
    AnyCache, AnyModel, MtpCache, SamplingParams, apply_penalties, deep_clone_mtp_cache, sample,
};
use mlx_rs::{
    Array, argmax_axis,
    ops::{self, concatenate_axis, indexing::IndexOp},
};

use crate::error::EngineError;

const fn draft_matches_target(draft_token_id: u32, target_id: u32) -> bool {
    draft_token_id == target_id
}

/// Capture a backbone-cache rollback point before a speculative verify.
///
/// KV caches are rolled back by *trimming the offset* (see [`rollback_backbone`]),
/// so we deliberately return `None` and never clone them. Cloning a KV cache and
/// later restoring it (`*cache = base`) makes the checkpoint share the live
/// cache's underlying MLX buffers; the in-place `slice_update` writes during
/// verify then let MLX donate a buffer that the checkpoint still references,
/// corrupting it and double-freeing on drop (the `malloc: pointer being freed
/// was not allocated` abort). Hybrid SSM/recurrent state cannot be offset-
/// trimmed, so those still need a full clone-restore.
fn capture_backbone_checkpoint(cache: &AnyCache) -> Option<AnyCache> {
    match cache {
        AnyCache::KV(_) => None,
        AnyCache::Hybrid(_) => Some(cache.deep_clone()),
    }
}

/// Roll the backbone cache back after a rejected speculative verify.
///
/// `verify_len` is the number of tokens the verify batch advanced the cache by.
/// KV caches rewind by `trim_by(verify_len)` (no clone, no buffer aliasing);
/// hybrid caches restore the clone captured by [`capture_backbone_checkpoint`].
fn rollback_backbone(cache: &mut AnyCache, checkpoint: Option<AnyCache>, verify_len: usize) {
    match checkpoint {
        Some(base) => *cache = base,
        None => cache.trim_by(verify_len),
    }
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
    pub fn record_cycle(
        &mut self,
        drafted_count: usize,
        emitted_count: usize,
        accepted_drafts_count: usize,
    ) {
        let drafted = u32::try_from(drafted_count).unwrap_or(u32::MAX);
        let emitted = u32::try_from(emitted_count).unwrap_or(u32::MAX);
        let accepted_drafts = u32::try_from(accepted_drafts_count)
            .unwrap_or(u32::MAX)
            .min(drafted);
        self.cycles = self.cycles.saturating_add(1);
        self.drafted = self.drafted.saturating_add(drafted);
        self.emitted = self.emitted.saturating_add(emitted);
        self.accepted_drafts = self.accepted_drafts.saturating_add(accepted_drafts);
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

/// Small adaptive controller for choosing the next MTP draft depth.
#[derive(Debug, Clone)]
pub struct AdaptiveDraftDepth {
    current: usize,
    min: usize,
    max: usize,
}

impl AdaptiveDraftDepth {
    #[must_use]
    pub fn new(initial: usize, max_depth: usize) -> Self {
        let capped_max = max_depth.max(1);
        Self {
            current: initial.clamp(1, capped_max),
            min: 1,
            max: capped_max,
        }
    }

    #[must_use]
    pub const fn current(&self) -> usize {
        self.current
    }

    pub const fn observe(&mut self, accepted_drafts: usize, drafted: usize) {
        if drafted == 0 {
            self.current = self.min;
            return;
        }

        if accepted_drafts == drafted && self.current < self.max {
            self.current += 1;
        } else if accepted_drafts.saturating_mul(4) <= drafted && self.current > self.min {
            self.current -= 1;
        } else if accepted_drafts.saturating_mul(4) >= drafted.saturating_mul(3)
            && self.current < self.max
        {
            self.current += 1;
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

/// Run one prompt-lookup draft inside an MTP decode loop.
///
/// This verifies copied prompt/history tokens with the backbone and mirrors the
/// accepted verifier span into the MTP cache, so the next cycle can continue
/// with either prompt lookup or the model's MTP head.
pub fn mtp_prompt_lookup_cycle(
    model: &mut AnyModel,
    cache: &mut AnyCache,
    mtp_cache: &mut MtpCache,
    previous_hidden: &Array,
    history_before_confirmed: &[u32],
    confirmed_token_id: u32,
    config: PromptLookupConfig,
) -> Result<Option<MtpCycleResult>, EngineError> {
    let mut lookup_context = Vec::with_capacity(history_before_confirmed.len().saturating_add(1));
    lookup_context.extend_from_slice(history_before_confirmed);
    lookup_context.push(confirmed_token_id);
    let drafts = prompt_lookup_draft(
        &lookup_context,
        config.max_drafts,
        config.max_ngram,
        config.max_window,
    );
    if drafts.is_empty() {
        return Ok(None);
    }

    let base_cache = capture_backbone_checkpoint(cache);
    let base_mtp_cache = deep_clone_mtp_cache(mtp_cache);
    let mut verify_tokens = Vec::with_capacity(drafts.len().saturating_add(1));
    verify_tokens.push(confirmed_token_id);
    verify_tokens.extend(drafts.iter().copied());

    let (verify_hidden, verifier_targets) = backbone_verify_batch(model, cache, &verify_tokens)?;
    let verify_hidden_for_mtp = verify_hidden.clone();
    if verifier_targets.len() < verify_tokens.len() {
        return Err(EngineError::Generation(format!(
            "hybrid prompt-lookup verifier returned {} target ids for {} input tokens",
            verifier_targets.len(),
            verify_tokens.len()
        )));
    }

    let accepted_drafts = accepted_draft_prefix_len(&drafts, &verifier_targets);
    let tokens = emitted_tokens(confirmed_token_id, &drafts, accepted_drafts);

    let (accepted_hidden_rows, next_token_id) = if accepted_drafts == drafts.len() {
        let next = *verifier_targets.get(accepted_drafts).ok_or_else(|| {
            EngineError::Generation(format!(
                "hybrid prompt-lookup verifier missing target at accepted index {accepted_drafts}"
            ))
        })?;
        (verify_hidden, next)
    } else {
        rollback_backbone(cache, base_cache, verify_tokens.len());
        let (replay_hidden, replay_targets) = backbone_verify_batch(model, cache, &tokens)?;
        let next = *replay_targets.get(accepted_drafts).ok_or_else(|| {
            EngineError::Generation(format!(
                "hybrid prompt-lookup replay returned {} target ids for accepted index {}",
                replay_targets.len(),
                accepted_drafts
            ))
        })?;
        (replay_hidden, next)
    };

    let h_last = hidden_row(&accepted_hidden_rows, accepted_drafts)?;
    mirror_verified_mtp_cache(
        model,
        mtp_cache,
        base_mtp_cache,
        previous_hidden,
        &verify_hidden_for_mtp,
        &verify_tokens,
        tokens.len(),
    )?;

    Ok(Some(MtpCycleResult {
        tokens,
        hidden: h_last,
        next_token_id,
        drafted: drafts.len(),
        accepted_drafts,
    }))
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

    let base_cache = capture_backbone_checkpoint(cache);
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
        rollback_backbone(cache, base_cache, verify_tokens.len());
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

/// `backbone_verify_batch` that optionally also collects DFlash tap-layer
/// hiddens, so MTP cycles run as the floor of a DFlash gate can keep the
/// drafter's context cache fed (tap clones are lazy — measured ~free).
fn backbone_verify_batch_tapped(
    model: &mut AnyModel,
    cache: &mut AnyCache,
    tokens: &[u32],
    tap_layers: Option<&[usize]>,
    sampling: Option<&SamplingParams>,
    history: Option<&[u32]>,
) -> Result<(Array, Vec<u32>, Option<Vec<Array>>), EngineError> {
    let input = token_input(tokens)?;
    let (hidden, logits, taps) = match tap_layers {
        Some(layers) => {
            let (hidden, logits, taps) = model
                .forward_with_hidden_taps(&input, None, cache, layers)
                .map_err(EngineError::Mlx)?;
            (hidden, logits, Some(taps))
        }
        None => {
            let (hidden, logits) = model
                .forward_with_hidden(&input, None, cache)
                .map_err(EngineError::Mlx)?;
            (hidden, logits, None)
        }
    };
    let target_ids = verify_targets(&logits, sampling, history)?;
    Ok((hidden, target_ids, taps))
}

/// Per-position verify targets: greedy argmax by default, or sampled at the
/// request's temperature when `sampling` is hot. Sampled targets keep the
/// output distribution exact under speculation — the emitted tokens are
/// ALWAYS the targets themselves; drafts only decide how many positions
/// commit per cycle.
pub(crate) fn verify_targets(
    logits: &Array,
    sampling: Option<&SamplingParams>,
    history: Option<&[u32]>,
) -> Result<Vec<u32>, EngineError> {
    // No history and greedy: the fast batched argmax (normal-path behavior).
    let Some(params) = sampling else {
        return greedy_token_ids(logits);
    };
    // Match `higgs_models::sample` exactly. Treating tiny non-zero
    // temperatures as greedy changes request semantics inside a verifier.
    #[allow(clippy::float_cmp)]
    let sampled = params.temperature != 0.0;
    let penalized = history.is_some()
        && (params.repetition_penalty.is_some()
            || params.frequency_penalty.is_some()
            || params.presence_penalty.is_some());
    if !sampled && !penalized {
        return greedy_token_ids(logits);
    }
    // Per-position walk. Penalties are applied over the caller's generated
    // history PLUS the targets chosen earlier in this chain — the sequential
    // semantics `decode_step` provides (dropping them here reintroduced the
    // sampling loops nanobot's repeat_penalty exists to break).
    let positions = usize::try_from(*logits.shape().get(1).unwrap_or(&0))
        .map_err(|_| EngineError::Generation("verify logits shape".to_owned()))?;
    let mut chain: Vec<u32> = history.map(<[u32]>::to_vec).unwrap_or_default();
    let mut out = Vec::with_capacity(positions);
    for i in 0..positions {
        let i_idx = i32::try_from(i)
            .map_err(|_| EngineError::Generation("verify position overflow".to_owned()))?;
        let raw_row = logits.index((.., i_idx, ..));
        let row = if penalized {
            apply_penalties(&raw_row, &chain, params).map_err(EngineError::Mlx)?
        } else {
            raw_row
        };
        let tok: u32 = if sampled {
            let t = sample(&row, params).map_err(EngineError::Mlx)?;
            eval([&t]).map_err(EngineError::Mlx)?;
            t.item()
        } else {
            let t = argmax_axis!(&row, -1).map_err(EngineError::Mlx)?;
            eval([&t]).map_err(EngineError::Mlx)?;
            t.item()
        };
        if history.is_some() {
            chain.push(tok);
        }
        out.push(tok);
    }
    Ok(out)
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
    Ok(mtp_cycle_inner(
        model,
        cache,
        mtp_cache,
        hidden,
        confirmed_token_id,
        draft_n_max,
        None,
        None,
        None,
        None,
    )?
    .0)
}

/// `mtp_cycle` for callers that retain the post-decode cache (session
/// continuation): accepted drafts are truncated BEFORE any `stop_ids` token so
/// the backbone never advances past the stop — see the truncation note in
/// `mtp_cycle_inner`. The stop token, when reached, arrives as
/// `next_token_id` (pending, unforwarded).
pub fn mtp_cycle_bounded(
    model: &mut AnyModel,
    cache: &mut AnyCache,
    mtp_cache: &mut MtpCache,
    hidden: &Array,
    confirmed_token_id: u32,
    draft_n_max: usize,
    stop_ids: &[u32],
    sampling: Option<&SamplingParams>,
    history: &[u32],
) -> Result<MtpCycleResult, EngineError> {
    Ok(mtp_cycle_inner(
        model,
        cache,
        mtp_cache,
        hidden,
        confirmed_token_id,
        draft_n_max,
        None,
        Some(stop_ids),
        sampling,
        Some(history),
    )?
    .0)
}

/// `mtp_cycle_bounded` that also returns DFlash tap-layer hiddens for the
/// emitted tokens, so session-path MTP floor cycles can keep a drafter
/// context backlog contiguous (same rationale as `mtp_cycle_tapped`).
#[allow(clippy::too_many_arguments)]
pub fn mtp_cycle_session(
    model: &mut AnyModel,
    cache: &mut AnyCache,
    mtp_cache: &mut MtpCache,
    hidden: &Array,
    confirmed_token_id: u32,
    draft_n_max: usize,
    stop_ids: &[u32],
    sampling: Option<&SamplingParams>,
    tap_layers: &[usize],
    history: &[u32],
) -> Result<(MtpCycleResult, Vec<Array>), EngineError> {
    let (result, taps) = mtp_cycle_inner(
        model,
        cache,
        mtp_cache,
        hidden,
        confirmed_token_id,
        draft_n_max,
        Some(tap_layers),
        Some(stop_ids),
        sampling,
        Some(history),
    )?;
    Ok((result, taps.unwrap_or_default()))
}

/// `mtp_cycle` that also returns DFlash tap-layer hiddens for the emitted
/// tokens, so MTP cycles run as the floor of a DFlash gate keep the drafter's
/// context cache contiguous across floored stretches (a context hole blinds
/// the drafter at the next spec probe — measured accept collapse from ~6 to
/// ~1.2 on deterministic text).
pub fn mtp_cycle_tapped(
    model: &mut AnyModel,
    cache: &mut AnyCache,
    mtp_cache: &mut MtpCache,
    hidden: &Array,
    confirmed_token_id: u32,
    draft_n_max: usize,
    tap_layers: &[usize],
) -> Result<(MtpCycleResult, Vec<Array>), EngineError> {
    let (result, taps) = mtp_cycle_inner(
        model,
        cache,
        mtp_cache,
        hidden,
        confirmed_token_id,
        draft_n_max,
        Some(tap_layers),
        None,
        None,
        None,
    )?;
    Ok((result, taps.unwrap_or_default()))
}

#[allow(clippy::too_many_lines)]
fn mtp_cycle_inner(
    model: &mut AnyModel,
    cache: &mut AnyCache,
    mtp_cache: &mut MtpCache,
    hidden: &Array,
    confirmed_token_id: u32,
    draft_n_max: usize,
    tap_layers: Option<&[usize]>,
    stop_ids: Option<&[u32]>,
    sampling: Option<&SamplingParams>,
    history: Option<&[u32]>,
) -> Result<(MtpCycleResult, Option<Vec<Array>>), EngineError> {
    let draft_limit = draft_n_max.max(1);
    let base_cache = capture_backbone_checkpoint(cache);
    let base_mtp_cache = deep_clone_mtp_cache(mtp_cache);
    let mut speculative_mtp_cache = deep_clone_mtp_cache(mtp_cache);
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
            confirmed_mtp_cache = Some(deep_clone_mtp_cache(&speculative_mtp_cache));
        }
    }

    let first_draft = *drafts
        .first()
        .ok_or_else(|| EngineError::Generation("MTP produced no draft tokens".to_owned()))?;

    let mut verify_tokens = Vec::with_capacity(drafts.len().saturating_add(1));
    verify_tokens.push(confirmed_token_id);
    verify_tokens.extend(drafts.iter().copied());

    let chain_history: Option<Vec<u32>> = history.map(|h| {
        let mut v = Vec::with_capacity(h.len() + 1);
        v.extend_from_slice(h);
        v.push(confirmed_token_id);
        v
    });
    let (verify_hidden, verifier_targets, verify_taps) = backbone_verify_batch_tapped(
        model,
        cache,
        &verify_tokens,
        tap_layers,
        sampling,
        chain_history.as_deref(),
    )?;
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
    let mut accepted_drafts = if draft_matches_target(first_draft, first_target) {
        accepted_draft_prefix_len(&drafts, &verifier_targets)
    } else {
        0
    };
    // Stop-aware truncation for callers that RETAIN the post-decode cache
    // (session continuation): an accepted stop token would leave its KV/SSM
    // advance in the cache while the retained token list pops it, and Hybrid
    // caches cannot be trimmed after the fact. Truncating acceptance BEFORE
    // the stop token routes through the partial-accept rollback+replay, so
    // the stop token surfaces only as `next_token_id` (pending, unforwarded)
    // — the caller emits it and stops, exactly like sequential decode.
    if let Some(stops) = stop_ids {
        if let Some(i) = drafts
            .get(..accepted_drafts)
            .unwrap_or(&[])
            .iter()
            .position(|t| stops.contains(t))
        {
            accepted_drafts = i;
        }
    }
    let tokens = emitted_tokens(confirmed_token_id, &drafts, accepted_drafts);

    let (accepted_hidden_rows, next_token_id, accepted_taps) = if accepted_drafts == drafts.len() {
        let next = *verifier_targets.get(accepted_drafts).ok_or_else(|| {
            EngineError::Generation(format!(
                "MTP verifier missing target at accepted index {accepted_drafts}"
            ))
        })?;
        // Taps cover the verify batch == the emitted tokens on full accept.
        (verify_hidden, next, verify_taps)
    } else {
        rollback_backbone(cache, base_cache, verify_tokens.len());
        let (replay_hidden, replay_targets, replay_taps) = backbone_verify_batch_tapped(
            model,
            cache,
            &tokens,
            tap_layers,
            sampling,
            chain_history.as_deref(),
        )?;
        let next = *replay_targets.get(accepted_drafts).ok_or_else(|| {
            EngineError::Generation(format!(
                "MTP replay returned {} target ids for accepted index {}",
                replay_targets.len(),
                accepted_drafts
            ))
        })?;
        // The replay batch is exactly the emitted tokens, so its taps align 1:1.
        (replay_hidden, next, replay_taps)
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

    debug_assert!(
        !tokens.is_empty(),
        "MTP must always emit the confirmed token"
    );

    Ok((
        MtpCycleResult {
            tokens,
            hidden: h_last,
            next_token_id,
            drafted: drafts.len(),
            accepted_drafts,
        },
        accepted_taps,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        AdaptiveDraftDepth, MtpStats, accepted_draft_prefix_len, draft_matches_target,
        emitted_tokens, prompt_lookup_draft,
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
    fn mtp_stats_tracks_explicit_accepted_draft_count() {
        let mut stats = MtpStats::default();
        stats.record_cycle(3, 2, 2);
        stats.record_cycle(2, 1, 0);

        assert_eq!(stats.cycles(), 2);
        assert_eq!(stats.drafted(), 5);
        assert_eq!(stats.emitted(), 3);
        assert_eq!(stats.accepted_drafts(), 2);
        assert!((stats.acceptance_rate_percent() - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn adaptive_draft_depth_grows_on_full_acceptance_and_backs_off_on_rejection() {
        let mut depth = AdaptiveDraftDepth::new(2, 4);

        depth.observe(2, 2);
        assert_eq!(depth.current(), 3);

        depth.observe(3, 3);
        assert_eq!(depth.current(), 4);

        depth.observe(0, 4);
        assert_eq!(depth.current(), 3);
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
