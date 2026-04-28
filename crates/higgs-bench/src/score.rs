// `clippy::suboptimal_flops` would rewrite `a*b + c*d` as `c.mul_add(d,
// a*b)`, which fuses to a single rounding step and produces results
// numerically distinct from the Python reference. The acceptance
// criterion requires exact equivalence with Python, so the literal
// add/multiply form is the correct one.
#![allow(
    clippy::module_name_repetitions,
    clippy::suboptimal_flops,
    clippy::if_not_else
)]
//! Composite-score math for `bench_mlx_tuning`.
//!
//! Pulled into the library crate (rather than the bin file) so the unit
//! tests ported from `test_bench_mlx_tuning.py` can exercise
//! the formulas directly. The formulas mirror the Python verbatim —
//! `bench_mlx_tuning.py`'s acceptance criterion is "composite score
//! matches Python given the same inputs", so any drift here is a bug.

/// Cap on the prefix-cache speedup factored into the composite score.
/// Mirrors `CACHE_SPEEDUP_CAP = 32.0` in `bench_mlx_tuning.py`.
pub const CACHE_SPEEDUP_CAP: f64 = 32.0;

/// Trims, lowercases, and collapses runs of whitespace.
///
/// Matches `bench_mlx_tuning.py::normalize_text`.
#[must_use]
pub fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Caps `speedup` at `CACHE_SPEEDUP_CAP`.
#[must_use]
pub const fn clamp_cache_speedup(speedup: f64) -> f64 {
    if speedup < CACHE_SPEEDUP_CAP {
        speedup
    } else {
        CACHE_SPEEDUP_CAP
    }
}

/// Sub-bench accuracy inputs (one per iteration).
#[derive(Debug, Clone, Copy)]
pub struct AccuracyInputs {
    pub qa: f64,
    pub long_context: f64,
    pub structured_output: f64,
    pub prefix_cache: f64,
}

/// Speed inputs: weighted TTFT (ms) and weighted decode tok/s.
#[derive(Debug, Clone, Copy)]
pub struct SpeedInputs {
    pub weighted_ttft_ms: f64,
    pub weighted_decode_tps: f64,
}

/// Prefix cache inputs.
#[derive(Debug, Clone, Copy)]
pub struct CacheInputs {
    pub passed: bool,
    pub speedup: f64,
}

/// Composite components: each in `[0, 1+]`, weighted into `composite`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    pub accuracy: f64,
    pub speed: f64,
    pub cache: f64,
    pub composite: f64,
}

/// Mirrors `compute_accuracy_score` in `bench_mlx_tuning.py`.
#[must_use]
pub fn compute_accuracy_score(acc: AccuracyInputs) -> f64 {
    (acc.qa * 0.45)
        + (acc.long_context * 0.25)
        + (acc.structured_output * 0.15)
        + (acc.prefix_cache * 0.15)
}

/// Mirrors `compute_speed_score` in `bench_mlx_tuning.py`.
#[must_use]
pub fn compute_speed_score(speed: SpeedInputs, best_ttft: f64, best_decode: f64) -> f64 {
    let ttft_score = if best_ttft != 0.0 && speed.weighted_ttft_ms != 0.0 {
        best_ttft / speed.weighted_ttft_ms
    } else {
        0.0
    };
    let decode_score = if best_decode != 0.0 {
        speed.weighted_decode_tps / best_decode
    } else {
        0.0
    };
    (ttft_score * 0.55) + (decode_score * 0.45)
}

/// Mirrors `compute_iteration_score` in `bench_mlx_tuning.py`.
#[must_use]
pub fn compute_iteration_score(
    acc: AccuracyInputs,
    speed: SpeedInputs,
    cache: CacheInputs,
    best_ttft: f64,
    best_decode: f64,
    best_cache: f64,
) -> Score {
    let accuracy = compute_accuracy_score(acc);
    let speed_score = compute_speed_score(speed, best_ttft, best_decode);
    let cache_speedup = if cache.passed {
        clamp_cache_speedup(cache.speedup)
    } else {
        0.0
    };
    let cache_score = if best_cache != 0.0 {
        cache_speedup / best_cache
    } else {
        0.0
    };
    let composite = 100.0 * ((accuracy * 0.45) + (speed_score * 0.45) + (cache_score * 0.10));
    Score {
        accuracy,
        speed: speed_score,
        cache: cache_score,
        composite,
    }
}

/// Computes per-iteration `best_*` aggregates from a slice of (speed,
/// cache) inputs. Mirrors the `score_results` body in
/// `bench_mlx_tuning.py`.
#[must_use]
pub fn compute_bests(speeds: &[SpeedInputs], caches: &[CacheInputs]) -> (f64, f64, f64) {
    let best_ttft = speeds
        .iter()
        .map(|s| s.weighted_ttft_ms)
        .fold(f64::INFINITY, f64::min);
    let best_decode = speeds
        .iter()
        .map(|s| s.weighted_decode_tps)
        .fold(f64::NEG_INFINITY, f64::max);
    let best_cache = caches
        .iter()
        .filter(|c| c.passed)
        .map(|c| clamp_cache_speedup(c.speedup))
        .fold(f64::NEG_INFINITY, f64::max);
    let best_cache_finalized = if best_cache.is_finite() && best_cache > 0.0 {
        best_cache
    } else {
        1.0
    };
    (best_ttft, best_decode, best_cache_finalized)
}
