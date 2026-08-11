//! Token-age KV-prune policy for the prune-rate sweep.
//!
//! This is the experiment harness that measures how aggressively a stock
//! Qwen3.6 MoE can have its KV pruned before reasoning accuracy degrades. It
//! prunes by token *age* (oldest non-sink tokens first) — there is no Thread-2
//! schema yet, deliberately, so the curve isolates the mechanism.
#![allow(clippy::doc_markdown, clippy::too_long_first_doc_paragraph)]

use higgs_models::AnyCache;
use higgs_models::cache::RopeShift;

use crate::error::EngineError;

/// Token-age prune policy.
///
/// Keep the first `sink` tokens (attention sinks) and at least `min_window`
/// recent tokens resident; cap the resident length at `keep_frac` of the logical
/// (never-pruned) length by evicting the oldest non-sink tokens.
#[derive(Debug, Clone, Copy)]
pub struct PrunePolicy {
    /// Always-resident prefix tokens (StreamingLLM-style attention sinks).
    pub sink: i32,
    /// Floor on resident length above the sinks, so recent context survives.
    pub min_window: i32,
    /// Target resident fraction of the logical length. `>= 1.0` disables pruning.
    pub keep_frac: f32,
    /// When true, never evict tokens flagged as fact-bearing (conclusions):
    /// prune only contiguous runs of unprotected scratch. When false, prune by
    /// pure token age.
    pub protect_facts: bool,
}

impl PrunePolicy {
    /// A no-op policy (the prune-rate = 0 baseline row).
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            sink: 4,
            min_window: 64,
            keep_frac: 1.0,
            protect_facts: false,
        }
    }
}

/// Resident-length target: `max(sink + min_window, floor(keep_frac * full_len))`.
/// `keep_frac >= 1.0` yields `full_len` (no prune).
#[allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#[must_use]
pub fn budget(full_len: i32, policy: &PrunePolicy) -> i32 {
    if policy.keep_frac >= 1.0 {
        return full_len;
    }
    let frac_budget = (policy.keep_frac.max(0.0) * full_len as f32).floor() as i32;
    (policy.sink + policy.min_window).max(frac_budget)
}

/// Plan a prune given the current resident `offset` and the logical `full_len`
/// (length as if nothing had ever been pruned). Returns the half-open span
/// `[a, b)` of oldest non-sink tokens to evict, or `None` if no prune is due.
///
/// After pruning the returned span the resident length becomes exactly the
/// budget `max(sink + min_window, floor(keep_frac * full_len))`, so the realized
/// prune rate tracks `1 - keep_frac`.
#[must_use]
pub fn plan_prune(offset: i32, full_len: i32, policy: &PrunePolicy) -> Option<(i32, i32)> {
    let budget = budget(full_len, policy);
    if offset <= budget {
        return None;
    }
    let drop = offset - budget;
    let a = policy.sink.max(0);
    Some((a, a + drop))
}

/// Plan a structural prune: evict the oldest contiguous runs of *unprotected*
/// tokens within `[sink, resident - min_window)` until resident reaches
/// `target`. `protected[i]` marks token `i` (a sink, recent, or fact-bearing
/// token) as never-evictable. Returns non-overlapping spans in ascending order;
/// the caller applies them in descending order so earlier indices stay valid
/// across sequential single-span prunes.
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::indexing_slicing
)]
#[must_use]
pub fn plan_structural_prune(
    protected: &[bool],
    sink: i32,
    min_window: i32,
    target: i32,
) -> Vec<(i32, i32)> {
    let resident = protected.len() as i32;
    let mut need = resident - target;
    if need <= 0 {
        return Vec::new();
    }
    let lo = sink.max(0);
    let hi = (resident - min_window.max(0)).max(lo);

    let mut spans = Vec::new();
    let mut i = lo;
    while i < hi && need > 0 {
        if protected[i as usize] {
            i += 1;
            continue;
        }
        let run_start = i;
        while i < hi && !protected[i as usize] {
            i += 1;
        }
        let take = (i - run_start).min(need);
        spans.push((run_start, run_start + take));
        need -= take;
    }
    spans
}

/// Apply [`plan_prune`] to a live cache. Returns `true` if a prune happened.
pub fn apply_prune(
    cache: &mut AnyCache,
    full_len: i32,
    policy: &PrunePolicy,
    rope: RopeShift,
) -> Result<bool, EngineError> {
    if let Some((a, b)) = plan_prune(cache.resident_len(), full_len, policy) {
        cache.prune_span(a, b, rope)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Apply a structural prune, evicting unprotected scratch runs and draining the
/// matching entries from `protected` so it stays aligned with the cache. Returns
/// the number of spans pruned. Spans are applied in descending order so each
/// single-span [`AnyCache::prune_span`] sees valid indices.
#[allow(clippy::as_conversions, clippy::cast_sign_loss)]
pub fn apply_structural_prune(
    cache: &mut AnyCache,
    protected: &mut Vec<bool>,
    full_len: i32,
    policy: &PrunePolicy,
    rope: RopeShift,
) -> Result<u32, EngineError> {
    let target = budget(full_len, policy);
    let spans = plan_structural_prune(protected, policy.sink, policy.min_window, target);
    let mut pruned = 0_u32;
    for &(a, b) in spans.iter().rev() {
        cache.prune_span(a, b, rope)?;
        protected.drain(a as usize..b as usize);
        pruned += 1;
    }
    Ok(pruned)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn disabled_never_prunes() {
        let p = PrunePolicy::disabled();
        assert_eq!(plan_prune(10_000, 10_000, &p), None);
    }

    #[test]
    fn below_budget_is_noop() {
        let p = PrunePolicy {
            sink: 4,
            min_window: 64,
            keep_frac: 0.5,
            protect_facts: false,
        };
        // full_len 100 → frac budget 50, floor 68 (sink+window) → budget 68.
        assert_eq!(plan_prune(50, 100, &p), None);
    }

    #[test]
    fn prunes_oldest_nonsink_to_budget() {
        let p = PrunePolicy {
            sink: 4,
            min_window: 8,
            keep_frac: 0.5,
            protect_facts: false,
        };
        // full_len 1000 → frac budget 500; sink+window = 12; budget = 500.
        // offset 800 → drop 300, span [4, 304).
        let (a, b) = plan_prune(800, 1000, &p).unwrap();
        assert_eq!((a, b), (4, 304));
        // Post-prune resident == budget, and the realized keep ratio == keep_frac.
        let resident_after = 800 - (b - a);
        assert_eq!(resident_after, 500);
    }

    #[test]
    fn respects_min_window_floor() {
        let p = PrunePolicy {
            sink: 4,
            min_window: 64,
            keep_frac: 0.01,
            protect_facts: false,
        };
        // Tiny keep_frac would imply ~0 budget, but the floor holds it at 68.
        let (a, b) = plan_prune(200, 1000, &p).unwrap();
        assert_eq!(a, 4);
        assert_eq!(200 - (b - a), 68);
    }

    #[test]
    fn structural_skips_protected_facts() {
        // len 20: sinks {0,1}, window protects {17,18,19}, a fact at 10.
        // Everything else in [2,17) is scratch. target 12 → drop 8.
        let mut protected = vec![false; 20];
        protected[0] = true;
        protected[1] = true;
        protected[10] = true; // the fact (e.g. the "24")
        let spans = plan_structural_prune(&protected, 2, 3, 12);
        // Oldest scratch run [2,10) is exactly 8 long → one span, fact untouched.
        assert_eq!(spans, vec![(2, 10)]);
    }

    #[test]
    fn structural_spills_across_protected_runs() {
        // Same layout, but a much smaller target forces dropping past the fact.
        let mut protected = vec![false; 20];
        protected[0] = true;
        protected[1] = true;
        protected[10] = true;
        // target 4 → need 16; prunable region [2,17) has runs [2,10)=8 and
        // [11,17)=6 (14 total, fact at 10 skipped). Drops both, can't reach 4.
        let spans = plan_structural_prune(&protected, 2, 3, 4);
        assert_eq!(spans, vec![(2, 10), (11, 17)]);
    }

    #[test]
    fn structural_noop_when_under_target() {
        let protected = vec![false; 10];
        assert!(plan_structural_prune(&protected, 2, 3, 50).is_empty());
    }
}
