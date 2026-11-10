//! A dedicated thread that owns MLX state.
//!
//! MLX arrays must be built, evaluated and dropped on one thread: MLX keeps no lock on lazy graph
//! state, so `mlx_rs::Array` is not `Send`. [`EngineThread`] builds its core on a worker thread and
//! runs every job there, so callers on any thread only exchange plain data with it.

use crate::error::EngineError;

/// Handle to a worker thread that owns a `T`.
pub struct EngineThread<T> {
    jobs: std::sync::mpsc::Sender<Job<T>>,
}

type Job<T> = Box<dyn FnOnce(&mut T) + Send>;

impl<T: 'static> EngineThread<T> {
    /// Start the thread and build its core there with `init`.
    ///
    /// Returns once `init` has finished, with its error if it failed.
    pub fn spawn<Init>(name: &str, init: Init) -> Result<Self, EngineError>
    where
        Init: FnOnce() -> Result<T, EngineError> + Send + 'static,
    {
        let (jobs, job_rx) = std::sync::mpsc::channel::<Job<T>>();
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<(), EngineError>>(1);

        std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                let mut core = match init() {
                    Ok(core) => core,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                if ready_tx.send(Ok(())).is_err() {
                    return;
                }
                for job in job_rx {
                    // A panicking job fails only its own caller, whose reply channel drops with
                    // it; the core keeps serving, as the poison-recovering mutex it replaces did.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        job(&mut core);
                    }));
                }
            })
            .map_err(|error| EngineError::Generation(format!("failed to spawn {name}: {error}")))?;

        ready_rx
            .recv()
            .map_err(|_| EngineError::Generation(format!("{name} exited while starting")))??;
        Ok(Self { jobs })
    }

    /// Run `job` against the core on the owning thread and wait for its result.
    pub fn run<R, F>(&self, job: F) -> Result<R, EngineError>
    where
        R: Send + 'static,
        F: FnOnce(&mut T) -> R + Send + 'static,
    {
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
        self.jobs
            .send(Box::new(move |core: &mut T| {
                let _ = reply_tx.send(job(core));
            }))
            .map_err(|_| EngineError::Generation("engine thread is not running".to_owned()))?;
        reply_rx
            .recv()
            .map_err(|_| EngineError::Generation("engine job panicked".to_owned()))
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;
    use std::thread::ThreadId;

    use super::*;

    fn assert_send_sync<S: Send + Sync>() {}

    #[test]
    fn handle_is_send_and_sync_for_a_core_that_is_not() {
        assert_send_sync::<EngineThread<Rc<Cell<u32>>>>();
    }

    #[test]
    fn core_is_built_and_used_on_the_worker_thread() {
        let engine =
            EngineThread::spawn("test-engine", || Ok(Rc::new(std::thread::current().id())))
                .unwrap();

        let (built_on, ran_on) = engine
            .run(|built_on: &mut Rc<ThreadId>| (**built_on, std::thread::current().id()))
            .unwrap();

        assert_eq!(built_on, ran_on);
        assert_ne!(built_on, std::thread::current().id());
    }

    #[test]
    fn spawn_returns_the_init_error() {
        let result = EngineThread::<Rc<u32>>::spawn("test-engine", || {
            Err(EngineError::Generation("model failed to load".to_owned()))
        });

        match result {
            Err(EngineError::Generation(message)) => assert_eq!(message, "model failed to load"),
            Err(other) => panic!("unexpected error: {other}"),
            Ok(_) => panic!("spawn must fail when init fails"),
        }
    }

    #[test]
    fn concurrent_callers_are_serialized_on_the_core() {
        let engine = std::sync::Arc::new(
            EngineThread::spawn("test-engine", || Ok(Rc::new(Cell::new(0_u32)))).unwrap(),
        );

        let callers: Vec<_> = (0..4)
            .map(|_| {
                let caller_engine = std::sync::Arc::clone(&engine);
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        caller_engine
                            .run(|count: &mut Rc<Cell<u32>>| count.set(count.get() + 1))
                            .unwrap();
                    }
                })
            })
            .collect();
        for caller in callers {
            caller.join().unwrap();
        }

        assert_eq!(engine.run(|count| count.get()).unwrap(), 400);
    }

    #[test]
    fn a_panicking_job_fails_only_its_caller() {
        let engine = EngineThread::spawn("test-engine", || Ok(Rc::new(Cell::new(7_u32)))).unwrap();

        let failed = engine.run(|_: &mut Rc<Cell<u32>>| -> u32 { panic!("job failed") });

        assert!(failed.is_err());
        assert_eq!(engine.run(|value| value.get()).unwrap(), 7);
    }

    #[test]
    fn dropping_the_handle_drops_the_core_on_the_worker_thread() {
        struct ReportsDrop {
            dropped_on: std::sync::mpsc::Sender<ThreadId>,
            _not_send: Rc<()>,
        }
        impl Drop for ReportsDrop {
            fn drop(&mut self) {
                let _ = self.dropped_on.send(std::thread::current().id());
            }
        }

        let (dropped_tx, dropped_rx) = std::sync::mpsc::channel();
        let engine = EngineThread::spawn("test-engine", move || {
            Ok(ReportsDrop {
                dropped_on: dropped_tx,
                _not_send: Rc::new(()),
            })
        })
        .unwrap();
        let worker = engine
            .run(|_: &mut ReportsDrop| std::thread::current().id())
            .unwrap();

        drop(engine);

        let dropped_on = dropped_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert_eq!(dropped_on, worker);
    }
}
