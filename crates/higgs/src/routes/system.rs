//! `GET /v1/system`: process, memory, and loaded-model facts for dashboards.

use std::process::Command;
use std::sync::LazyLock;
use std::time::Instant;

use axum::{Json, extract::State};
use serde::Serialize;

use crate::state::{Engine, SharedState};

/// Process start, captured on first touch. `build_router` touches it so
/// uptime counts from server construction rather than the first request.
pub static STARTED_AT: LazyLock<Instant> = LazyLock::new(Instant::now);

#[derive(Debug, Serialize)]
pub struct SystemResponse {
    pub version: &'static str,
    pub pid: u32,
    pub uptime_secs: u64,
    pub memory: MemoryInfo,
    pub models: Vec<LoadedModel>,
    pub metrics_enabled: bool,
}

#[derive(Debug, Default, Serialize)]
pub struct MemoryInfo {
    /// Unified memory installed on the machine.
    pub physical_total_bytes: Option<u64>,
    /// Resident set size of this process.
    pub process_rss_bytes: Option<u64>,
    /// Bytes currently held by live MLX arrays (weights plus KV caches).
    pub mlx_active_bytes: Option<u64>,
    /// High-water mark of MLX active memory since start.
    pub mlx_peak_bytes: Option<u64>,
    /// Bytes MLX keeps in its allocator cache for reuse.
    pub mlx_cache_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct LoadedModel {
    pub name: String,
    /// Configured checkpoint path or Hugging Face repo id.
    pub path: Option<String>,
    /// `simple` (serialized) or `batch` (interleaved) engine.
    pub engine: &'static str,
    pub mlx_profile: Option<String>,
    pub kv_cache: Option<String>,
}

pub async fn system(State(state): State<SharedState>) -> Json<SystemResponse> {
    let mut models: Vec<LoadedModel> = state
        .router
        .local_engines()
        .iter()
        .map(|(name, engine)| {
            let configured = state
                .config
                .models
                .iter()
                .find(|m| m.name.as_deref() == Some(name.as_str()) || m.path == *name);
            LoadedModel {
                name: name.clone(),
                path: configured.map(|m| m.path.clone()),
                engine: engine_kind(engine),
                mlx_profile: configured.and_then(|m| serialized_name(&m.mlx_profile)),
                kv_cache: configured.and_then(|m| serialized_name(&m.kv_cache)),
            }
        })
        .collect();
    models.sort_by(|a, b| a.name.cmp(&b.name));

    Json(SystemResponse {
        version: env!("CARGO_PKG_VERSION"),
        pid: std::process::id(),
        uptime_secs: STARTED_AT.elapsed().as_secs(),
        memory: memory_info(),
        models,
        metrics_enabled: state.metrics.is_some(),
    })
}

/// The config-file spelling of an enum field (its serde representation).
fn serialized_name<T: Serialize>(value: &T) -> Option<String> {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
}

const fn engine_kind(engine: &Engine) -> &'static str {
    match engine {
        Engine::Simple(_) => "simple",
        Engine::Batch(_) => "batch",
        #[cfg(test)]
        Engine::Stub(_) => "stub",
    }
}

fn memory_info() -> MemoryInfo {
    MemoryInfo {
        physical_total_bytes: *PHYSICAL_MEMORY,
        process_rss_bytes: process_rss_bytes(),
        mlx_active_bytes: mlx_rs::memory::active_memory()
            .ok()
            .and_then(|b| u64::try_from(b).ok()),
        mlx_peak_bytes: mlx_rs::memory::peak_memory()
            .ok()
            .and_then(|b| u64::try_from(b).ok()),
        mlx_cache_bytes: mlx_rs::memory::cache_memory()
            .ok()
            .and_then(|b| u64::try_from(b).ok()),
    }
}

// The workspace forbids unsafe code, so libc sysctl/proc_pidinfo are out;
// these shell out to the macOS tools instead.
static PHYSICAL_MEMORY: LazyLock<Option<u64>> = LazyLock::new(|| {
    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
});

fn process_rss_bytes() -> Option<u64> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    let kib: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    Some(kib.saturating_mul(1024))
}

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn physical_memory_is_reported_on_macos() {
        assert!(PHYSICAL_MEMORY.is_some_and(|bytes| bytes > 0));
    }

    #[test]
    fn rss_is_positive() {
        assert!(process_rss_bytes().is_some_and(|bytes| bytes > 0));
    }
}
