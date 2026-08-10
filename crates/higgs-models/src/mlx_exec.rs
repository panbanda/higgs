//! Single sanctioned gate for MLX graph evaluation.
//!
//! # Why this exists
//!
//! MLX submits GPU work through a **process-global** Metal command buffer.
//! Two threads calling `mlx_rs::transforms::eval` (or `async_eval`) at the same
//! time race inside `concatenate_gpu` / `get_command_encoder` and the process
//! dies with a native `SIGABRT` — no Rust panic, no unwinding, no diagnostic.
//!
//! Every HTTP request runs in its own `tokio::task::spawn_blocking` worker, so
//! consecutive turns and concurrent requests land on *different* OS threads.
//! The engine therefore serializes all MLX evaluation. Historically that was a
//! *convention*: "hold the model `Mutex` around eval". A convention is invisible
//! — a future `eval` added off-lock silently reintroduces the abort.
//!
//! This module turns the convention into structure:
//!
//! * [`acquire`] returns an [`MlxExecToken`] guarding a dedicated process-global
//!   `Mutex`. Holding it is the de-jure "the Metal command buffer is mine right
//!   now" token. It really serializes eval (it is a real `Mutex`), independent
//!   of any other lock.
//! * [`eval`] and [`async_eval`] are the only sanctioned entry points. Both
//!   `debug_assert!` that the calling thread holds the token before delegating
//!   to `mlx_rs`. An off-lock eval thus **panics loudly** in any debug/test
//!   build the instant it runs, instead of aborting natively under load.
//!
//! The assert is compiled out in `--release`, but the gate `Mutex` is not: even
//! a release build still serializes correctly as long as callers acquire the
//! token. The assert + the test suite (run in debug) exist to make a *future*
//! off-gate eval impossible to merge silently.
//!
//! # Release-mode policy
//!
//! In `--release` the `debug_assert` is gone, so safety rests on two pillars:
//! 1. the gate `Mutex`, which serializes evaluation in *every* build; and
//! 2. a workspace `clippy.toml` `disallowed-methods` ban on raw
//!    `mlx_rs::transforms::eval` / `async_eval` (clippy runs with `-Dwarnings`),
//!    so a *new* off-gate eval fails the build instead of slipping in.
//!
//! That compile-time ban is strictly stronger than a runtime release check would
//! be — it stops the bug before it ships — so we deliberately do NOT add a
//! release-only assertion feature. If one is ever wanted, gate the
//! `debug_assert` behind a `mlx-exec-checks` cfg and run it in a release CI job.

use std::cell::Cell;
use std::sync::{Mutex, MutexGuard};

use mlx_rs::Array;
use mlx_rs::error::Exception;

/// The one process-global MLX-execution gate. Serializes every sanctioned
/// `eval` / `async_eval` so the shared Metal command buffer has a single
/// writer at a time.
static MLX_GATE: Mutex<()> = Mutex::new(());

thread_local! {
    /// True while *this* thread holds a live [`MlxExecToken`].
    static HELD: Cell<bool> = const { Cell::new(false) };
}

/// RAII proof that the current thread owns the MLX-execution gate.
///
/// While alive, [`held`] returns `true` on the owning thread and the sanctioned
/// [`eval`] / [`async_eval`] wrappers will accept calls. On `Drop` the gate is
/// released and the thread-local flag cleared.
#[must_use = "the MLX gate is released as soon as the token is dropped"]
pub struct MlxExecToken {
    _guard: MutexGuard<'static, ()>,
}

impl Drop for MlxExecToken {
    fn drop(&mut self) {
        HELD.with(|h| h.set(false));
    }
}

/// Acquire the MLX-execution gate, blocking until it is free.
///
/// Hold the returned token across the entire prefill + decode + stash scope of
/// a generation (or the lifetime of the dedicated batch-engine worker thread).
/// Poison is recovered: the guarded data is `()`, so a thread that panicked
/// while holding the gate left nothing to corrupt.
pub fn acquire() -> MlxExecToken {
    let guard = MLX_GATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    HELD.with(|h| h.set(true));
    MlxExecToken { _guard: guard }
}

/// Whether the current thread holds a live [`MlxExecToken`].
#[must_use]
pub fn held() -> bool {
    HELD.with(Cell::get)
}

/// Sanctioned `mlx_rs::transforms::eval`. Panics in debug/test builds if the
/// calling thread does not hold the gate.
///
/// # Panics
/// In debug builds, if called without a live [`MlxExecToken`] on this thread.
#[allow(clippy::disallowed_methods)] // this IS the sanctioned wrapper; it must call the raw transform
pub fn eval<'a, T>(outputs: T) -> Result<(), Exception>
where
    T: IntoIterator<Item = &'a Array>,
{
    debug_assert!(
        held(),
        "MLX eval off the exec gate — the process-global Metal command buffer will SIGABRT under concurrency. Acquire mlx_exec::acquire() first."
    );
    mlx_rs::transforms::eval(outputs)
}

/// Sanctioned `mlx_rs::transforms::async_eval`. Panics in debug/test builds if
/// the calling thread does not hold the gate.
///
/// # Panics
/// In debug builds, if called without a live [`MlxExecToken`] on this thread.
#[allow(clippy::disallowed_methods)] // this IS the sanctioned wrapper; it must call the raw transform
pub fn async_eval<'a, T>(outputs: T) -> Result<(), Exception>
where
    T: IntoIterator<Item = &'a Array>,
{
    debug_assert!(
        held(),
        "MLX async_eval off the exec gate — the process-global Metal command buffer will SIGABRT under concurrency. Acquire mlx_exec::acquire() first."
    );
    mlx_rs::transforms::async_eval(outputs)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::shadow_reuse,
    clippy::disallowed_methods
)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use mlx_rs::Array;

    use super::{acquire, async_eval, eval, held};

    #[test]
    fn held_is_false_without_token() {
        assert!(!held(), "no token acquired yet on this thread");
    }

    #[test]
    fn token_sets_and_clears_held() {
        assert!(!held());
        {
            let _t = acquire();
            assert!(held(), "held() must be true while the token is alive");
        }
        assert!(!held(), "held() must be false after the token drops");
    }

    #[test]
    fn eval_succeeds_under_the_gate() {
        let _t = acquire();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        eval([&a]).expect("eval under the gate must succeed");
    }

    /// Proves the guard actually fires: calling `eval` with no token panics in
    /// debug builds (where `debug_assert` is live). This is the whole point of
    /// the enforcement — an off-lock eval is loud, not a silent native abort.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "off the exec gate")]
    fn eval_without_token_panics_in_debug() {
        assert!(!held());
        let a = Array::from_slice(&[1.0f32], &[1]);
        let _ = eval([&a]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "off the exec gate")]
    fn async_eval_without_token_panics_in_debug() {
        assert!(!held());
        let a = Array::from_slice(&[1.0f32], &[1]);
        let _ = async_eval([&a]);
    }

    /// High-fan-out concurrency stress: many threads each acquire the gate and
    /// hammer `eval` on real Arrays. The gate must serialize them so the shared
    /// Metal command buffer never races (pre-fix this aborts the process), and
    /// `held()` must be observed true inside the critical section and false
    /// outside it.
    #[test]
    fn mlx_eval_gate_serializes_high_fanout() {
        const THREADS: usize = 16;
        const ITERS: usize = 200;

        let errors = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::with_capacity(THREADS);

        for t in 0..THREADS {
            let errors = Arc::clone(&errors);
            handles.push(thread::spawn(move || {
                // Outside the gate, this thread must not be marked as holding it.
                if held() {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
                for i in 0..ITERS {
                    let token = acquire();
                    if !held() {
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                    let base = (t * ITERS + i) as f32;
                    let a = Array::from_slice(&[base, base + 1.0, base + 2.0], &[3]);
                    let b = Array::from_slice(&[1.0f32, 1.0, 1.0], &[3]);
                    // A concatenate + add graph mirrors the real prefill pattern
                    // that historically raced inside concatenate_gpu.
                    let c = mlx_rs::ops::concatenate_axis(&[&a, &b], 0).unwrap();
                    let d = c.add(&c).unwrap();
                    if eval([&d]).is_err() {
                        errors.fetch_add(1, Ordering::Relaxed);
                    }
                    drop(token);
                }
            }));
        }

        for h in handles {
            h.join().expect("worker thread must not panic/abort");
        }
        assert_eq!(
            errors.load(Ordering::Relaxed),
            0,
            "no eval errors and held() observed correctly across all threads"
        );
    }
}
