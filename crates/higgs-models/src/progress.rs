//! Prefill progress reporting via a thread-scoped sink.
//!
//! The chunked-prefill loops live deep in the model crate with no access to
//! the engine's streaming channel; threading a callback through every
//! `forward_chunked` signature (the `AnyModel` zoo plus per-model overrides
//! and their test callers) would churn a dozen call sites for one optional
//! observer. Engines run generation on a dedicated blocking thread, so a
//! thread-local sink installed for the duration of one prefill is exact and
//! invisible to every other code path.

use std::cell::RefCell;

type Sink = Box<dyn FnMut(i32, i32)>;

thread_local! {
    static PREFILL_SINK: RefCell<Option<Sink>> = const { RefCell::new(None) };
}

/// RAII guard that removes the thread's prefill sink on drop, keeping
/// installs scoped to a single prefill call.
pub struct PrefillSinkGuard;

impl Drop for PrefillSinkGuard {
    fn drop(&mut self) {
        PREFILL_SINK.with(|s| *s.borrow_mut() = None);
    }
}

/// Install a prefill-progress sink for the current thread.
///
/// The sink receives `(processed, total)` after each completed prefill chunk.
/// `processed` is the cumulative number of tokens forwarded so far in *this*
/// prefill — i.e. relative to the suffix that survived prefix-cache reuse, not
/// a per-chunk delta and not an absolute prompt offset. Callers that want an
/// absolute prompt position add the cached-prefix length themselves. Hold the
/// returned guard for the duration of the prefill; dropping it uninstalls the
/// sink.
///
/// The sink must not re-enter the progress machinery: calling
/// [`report_prefill_progress`] or installing another sink from inside the sink
/// callback panics, because the thread-local is `borrow_mut`-held while the
/// sink runs.
pub fn install_prefill_progress_sink(sink: Sink) -> PrefillSinkGuard {
    PREFILL_SINK.with(|s| *s.borrow_mut() = Some(sink));
    PrefillSinkGuard
}

/// Report chunked-prefill progress: `processed` of `total` tokens done.
/// No-op when no sink is installed (the common path: one `thread_local`
/// lookup + `Option` check per ~1024-token chunk).
///
/// The sink is invoked while the thread-local is `borrow_mut`-held, so the
/// sink must not call back into this function or reinstall the sink.
pub fn report_prefill_progress(processed: i32, total: i32) {
    PREFILL_SINK.with(|s| {
        if let Some(f) = s.borrow_mut().as_mut() {
            f(processed, total);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Reports reach the sink only while the guard is alive; the no-sink
    /// path is a silent no-op (the invariant every model forward relies on).
    #[test]
    fn test_sink_scoped_by_guard() {
        let seen: Rc<RefCell<Vec<(i32, i32)>>> = Rc::new(RefCell::new(Vec::new()));

        // No sink installed — must not panic, must not record.
        report_prefill_progress(512, 4096);
        assert!(seen.borrow().is_empty());

        let sink_seen = Rc::clone(&seen);
        let guard = install_prefill_progress_sink(Box::new(move |p, t| {
            sink_seen.borrow_mut().push((p, t));
        }));
        report_prefill_progress(1024, 4096);
        report_prefill_progress(2048, 4096);
        drop(guard);

        // After the guard drops, reports are no-ops again.
        report_prefill_progress(3072, 4096);
        assert_eq!(*seen.borrow(), vec![(1024, 4096), (2048, 4096)]);
    }
}
