//! Speculative / compressive prefill (PFlash) for higgs.
//!
//! Implements the *selection* half of SpecPrefill (Liu et al., arXiv:2502.02789)
//! — the algorithm Lucebox's PFlash uses verbatim. A small drafter (Qwen3-0.6B)
//! scores prompt-token importance; the heavy target then prefills only the kept
//! fraction, cutting target prefill FLOPs by ~1/keep_ratio.
//!
//! See `.planning/DESIGN-pflash-higgs.md` for the full design and
//! `docs/RESEARCH-pflash-prior-art.md` for prior art. Summary of the
//! SpecPrefill-Full-LAH recipe:
//!   1. Drafter forward + `lookahead` greedy decode; capture per-layer Q.
//!   2. Block-wise attention scoring (NEVER materialize `[H, S, S]` — see
//!      `SAFETY` below).
//!   3. `importance = mean_over_lookahead( max_over_(layers, heads)(attn) )`.
//!   4. 1D avgpool smoothing (`avgpool`, default 13).
//!   5. Chunk-top-K selection (`chunk`, default 32; `keep_ratio`, default 0.10).
//!   6. Restore original prompt positions on survivors (critical for NIAH).
//!
//! # SAFETY — the lesson from the probe crash
//!
//! A prior Python probe computed `Q @ K.T` as a full `[H, S, S]` tensor while
//! the Bonsai target was resident. At S=32K that is ~32 GB; the allocator OOM'd
//! and crashed the server.
//!
//! The scorer half (steps 1-3, NOT yet implemented here) MUST compute attention
//! block-pair by block-pair: one K-block of 128 at a time, producing a transient
//! `[lookahead, n_kv_heads, 128]` (~16 KB) and accumulating into a per-layer
//! `[lookahead, n_heads, S]` (~75 MB at S=128K), streamed to a running max so
//! peak is **~75 MB regardless of S**. S grows the accumulator linearly, never
//! quadratically. The regression test `scorer_never_materializes_full_attention`
//! in the design (§5.4) asserts this bound once the scorer lands.
//!
//! This module currently ships the model-free selection half (steps 4-6), which
//! is pure arithmetic over `Vec<f32>` and cannot OOM. The scorer half is
//! scaffolded below as `score_prompt` and is the next implementation step.

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]
mod tests;

/// Knobs for SpecPrefill selection. Defaults mirror the published recipe
/// (Cross-Family Appendix A.1: chunk=32, avgpool=13, lookahead=8) and the
/// "highest tradable point" keep_ratio=0.10 (RESEARCH §3.5).
#[derive(Debug, Clone)]
pub struct PrefillScoreConfig {
    /// Fraction of source tokens kept after compression.
    pub keep_ratio: f32,
    /// Prompt-token block size for survivor selection.
    pub chunk: usize,
    /// 1D avgpool smoothing kernel width.
    pub avgpool: usize,
    /// Lookahead decoded tokens used for importance aggregation.
    pub lookahead: usize,
}

impl Default for PrefillScoreConfig {
    fn default() -> Self {
        Self {
            keep_ratio: 0.10,
            chunk: 32,
            avgpool: 13,
            lookahead: 8,
        }
    }
}

/// A survivor plan: the kept token ids in their original order, plus the
/// **original prompt position** of each survivor (for RoPE position-id restore —
/// SpecPrefill §3.2.4; critical for NIAH and counting tasks).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurvivalPlan {
    pub token_ids: Vec<u32>,
    pub original_positions: Vec<i32>,
}

impl SurvivalPlan {
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

/// 1D average-pool smoothing of a per-token importance vector (step 4).
///
/// Uses a **shrinking window at the edges**: samples near the start/end average
/// over fewer than `kernel` neighbors (no reflect/replicate padding). This is
/// intentional — the sink block (token 0) is force-kept by `select_survivors`
/// regardless of its smoothed score, so edge under-weighting does not lose it.
/// `kernel` must be odd and >= 1 so the window is symmetric around each sample
/// (the published kernel=13 is odd).
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
pub fn smooth_importance(importance: &[f32], kernel: usize) -> Result<Vec<f32>, String> {
    if importance.is_empty() {
        return Ok(Vec::new());
    }
    if kernel == 0 || kernel % 2 == 0 {
        return Err(format!(
            "smooth_importance: kernel must be odd and >= 1, got {kernel}"
        ));
    }
    let half = kernel / 2;
    let n = importance.len();
    let mut out = vec![0.0_f32; n];
    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let cnt = (hi - lo) as f32;
        let sum: f32 = importance[lo..hi].iter().sum();
        out[i] = sum / cnt;
    }
    Ok(out)
}

/// Select survivors from a smoothed importance vector (steps 5-6).
///
/// Chunks the prompt into `chunk`-sized blocks, scores each block by the max
/// smoothed importance it contains, and keeps the top `keep_ratio` fraction of
/// blocks. The first block (sink / system-prompt anchor) and the block holding
/// the final prompt token (whose logits the target samples from) are always
/// kept — SpecPrefill's sink convention.
///
/// Returns the kept token ids (a subsequence of `tokens`, original order) and
/// each survivor's original prompt position. Token-level positions are restored
/// verbatim (Option A in DESIGN §2.6); the target applies RoPE at these original
/// positions even though its KV cache stores only `M = survivors.len()` rows.
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
pub fn select_survivors(
    tokens: &[u32],
    importance: &[f32],
    cfg: &PrefillScoreConfig,
) -> Result<SurvivalPlan, String> {
    if tokens.len() != importance.len() {
        return Err(format!(
            "select_survivors: tokens ({}) and importance ({}) length mismatch",
            tokens.len(),
            importance.len()
        ));
    }
    if tokens.is_empty() {
        return Ok(SurvivalPlan {
            token_ids: Vec::new(),
            original_positions: Vec::new(),
        });
    }
    if !(0.02..=0.50).contains(&cfg.keep_ratio) {
        return Err(format!(
            "select_survivors: keep_ratio {} out of range [0.02, 0.50]",
            cfg.keep_ratio
        ));
    }
    if cfg.chunk == 0 {
        return Err("select_survivors: chunk must be >= 1".to_string());
    }
    let smoothed = smooth_importance(importance, cfg.avgpool)?;

    let s = tokens.len();
    let n_blocks = s.div_ceil(cfg.chunk);
    // Block score = mean smoothed importance over the block (SpecPrefill §3.2.3:
    // "average importance within each chunk"). A multi-token salient span (a real
    // needle sentence) elevates the whole block, so mean ranks it correctly while
    // damping single-token noise spikes that `max` would over-promote.
    let mut block_score: Vec<(usize, f32)> = (0..n_blocks)
        .map(|b| {
            let lo = b * cfg.chunk;
            let hi = lo + cfg.chunk.min(s - lo);
            let sum: f32 = smoothed[lo..hi].iter().sum();
            (b, sum / (hi - lo) as f32)
        })
        .collect();
    // Stable descending sort by score: ties keep lower block index (earlier text).
    block_score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let keep = (cfg.keep_ratio * n_blocks as f32).round() as usize;
    let keep = keep.clamp(1, n_blocks);
    let last_block = n_blocks - 1;
    let mut kept: Vec<bool> = vec![false; n_blocks];
    for &(b, _) in block_score.iter().take(keep) {
        kept[b] = true;
    }
    // Sink (block 0) + final-token block are mandatory.
    kept[0] = true;
    kept[last_block] = true;

    let mut token_ids = Vec::new();
    let mut original_positions = Vec::new();
    for b in 0..n_blocks {
        if !kept[b] {
            continue;
        }
        let lo = b * cfg.chunk;
        let hi = lo + cfg.chunk.min(s - lo);
        for i in lo..hi {
            token_ids.push(tokens[i]);
            original_positions.push(i as i32);
        }
    }
    Ok(SurvivalPlan {
        token_ids,
        original_positions,
    })
}

// ---------------------------------------------------------------------------
// Scorer half (steps 1-3, the mlx-rs-heavy part).
// ---------------------------------------------------------------------------

use mlx_rs::ops;
use mlx_rs::{Array, error::Exception};

/// Per-layer importance contribution from the lookahead queries (step 2-3).
///
/// `q_lah` is `[n_heads, lah, head_dim]` — the post-RoPE/norm queries at the
/// `lah` lookahead positions for one drafter layer. `k` is
/// `[n_kv_heads, S, head_dim]` — the post-RoPE/norm keys over the prompt.
///
/// Returns `importance: [S] = mean_over_lah( max_over_heads( softmax(Q·K^T) ) )`.
///
/// # Memory safety (the lesson from the probe crash)
///
/// The attention tensor here is `[n_heads, lah, S]` — **S-linear, not S²**,
/// because queries are only the `lah=8` lookahead positions, not all prompt
/// positions. At `S = 128K`, `n_heads = 16`, `lah = 8`, f32: ~64 MB. The crash
/// came from running uncached full forwards (unbounded lazy graph) and from
/// treating all prompt tokens as queries; this function does neither. No
/// `[H, S, S]` is ever materialized.
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
pub fn layer_importance(
    q_lah: &Array,
    k: &Array,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    scale: f32,
) -> Result<Array, Exception> {
    let k_shape = k.shape();
    let s = *k_shape.get(1).ok_or_else(|| {
        Exception::custom("layer_importance: k must be [n_kv_heads, S, head_dim]")
    })?;
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(Exception::custom(format!(
            "layer_importance: n_heads {n_heads} not divisible by n_kv_heads {n_kv_heads}"
        )));
    }
    let group = n_heads / n_kv_heads;

    // Scale the queries (equivalent to scaling the scores; broadcasts a scalar
    // array — mlx-rs has no Mul<f32> overload, so go through ops::multiply).
    let q_scaled = ops::multiply(q_lah, &Array::from_f32(scale))?;

    // GQA: expand keys from [n_kv_heads, S, d] to [n_heads, S, d] by repeating
    // each kv head `group` times (matches the Qwen3 attention's head mapping).
    let k_expanded = if group == 1 {
        k.clone()
    } else {
        ops::broadcast_to(
            &k.reshape(&[n_kv_heads, 1, s, head_dim])?,
            &[n_kv_heads, group, s, head_dim],
        )?
        .reshape(&[n_heads, s, head_dim])?
    };
    let k_t = k_expanded.transpose_axes(&[0, 2, 1])?; // [n_heads, head_dim, S]

    // scores = q_scaled @ k_t -> [n_heads, lah, S]
    let scores = q_scaled.matmul(&k_t)?;
    let attn = ops::softmax_axis(&scores, -1, true)?;

    // importance = max over heads, then mean over lah -> [S].
    let max_h = ops::max_axis(&attn, 0, None)?;
    let importance = ops::mean_axis(&max_h, 0, None)?;
    Ok(importance)
}

// Internally (full signature once the drafter-forward capture lands):
//   pub fn score_prompt(
//       drafter: &AnyModel,
//       tokens: &[u32],
//       cfg: &PrefillScoreConfig,
//   ) -> Result<Vec<f32>, Exception> { ... }
//
// Qwen3-0.6B attention access (confirmed):
//   transformer::Attention exposes q_proj / k_proj (MaybeQuantized<nn::Linear>),
//   q_norm / k_norm (Option<RmsNorm>), rope (nn::Rope) — see
//   crates/higgs-models/src/transformer.rs:163-186.
