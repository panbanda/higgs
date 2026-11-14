//! Resident-set-size sampling of a server process.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use sysinfo::{Pid, ProcessesToUpdate, System};

const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);

/// Samples a process's resident set size on a background thread so transient
/// peaks between explicit samples are still captured.
pub struct RssSampler {
    pid: Pid,
    peak_bytes: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl RssSampler {
    pub fn start(raw_pid: u32) -> Result<Self> {
        let pid = Pid::from_u32(raw_pid);
        let initial = read_rss_bytes(&mut System::new(), pid)
            .with_context(|| format!("process {raw_pid} not found"))?;
        let peak_bytes = Arc::new(AtomicU64::new(initial));
        let stop = Arc::new(AtomicBool::new(false));
        let worker = {
            let worker_peak = Arc::clone(&peak_bytes);
            let worker_stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut system = System::new();
                while !worker_stop.load(Ordering::Relaxed) {
                    if let Some(bytes) = read_rss_bytes(&mut system, pid) {
                        worker_peak.fetch_max(bytes, Ordering::Relaxed);
                    }
                    std::thread::sleep(SAMPLE_INTERVAL);
                }
            })
        };
        Ok(Self {
            pid,
            peak_bytes,
            stop,
            worker: Some(worker),
        })
    }

    pub fn sample_now(&self) {
        if let Some(bytes) = read_rss_bytes(&mut System::new(), self.pid) {
            self.peak_bytes.fetch_max(bytes, Ordering::Relaxed);
        }
    }

    /// Stops sampling and returns the peak in mebibytes.
    pub fn finish(mut self) -> f64 {
        self.sample_now();
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.peak_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
    }
}

fn read_rss_bytes(system: &mut System, pid: Pid) -> Option<u64> {
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    system.process(pid).map(sysinfo::Process::memory)
}
