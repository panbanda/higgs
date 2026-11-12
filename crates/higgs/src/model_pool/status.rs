use std::fmt;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::config::LoadMode;

/// Where a model is in its load/unload lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelState {
    Unloaded,
    Loading,
    Loaded,
    Unloading,
    /// The last load attempt failed; the next request retries it.
    Failed,
}

impl fmt::Display for ModelState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Unloaded => "unloaded",
            Self::Loading => "loading",
            Self::Loaded => "loaded",
            Self::Unloading => "unloading",
            Self::Failed => "failed",
        })
    }
}

/// Point-in-time view of one configured local model, as served by `/v1/models/status`.
#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    pub name: String,
    pub state: ModelState,
    /// Why the last load failed. Set only while `state` is `failed`.
    pub last_error: Option<String>,
    pub pinned: bool,
    pub load: LoadMode,
    pub in_flight: usize,
    pub last_used_at: Option<DateTime<Utc>>,
    /// How long the most recent load took; kept while the model stays loaded.
    pub load_duration_ms: Option<u64>,
    /// Footprint predicted from the weight files on disk before loading.
    pub estimated_bytes: Option<u64>,
    /// Growth in MLX active memory observed while loading. Approximate when other
    /// models are generating at the same time.
    pub measured_bytes: Option<u64>,
    /// `simple` (serialized) or `batch` (interleaved).
    pub engine: &'static str,
    pub idle_ttl_seconds: Option<u64>,
}
