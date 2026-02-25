use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError, RwLock};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};

use crate::metrics_log::MetricsLogger;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingMethod {
    Pattern,
    Auto,
    Default,
    Higgs,
}

impl From<crate::router::RoutingMethod> for RoutingMethod {
    fn from(m: crate::router::RoutingMethod) -> Self {
        match m {
            crate::router::RoutingMethod::Direct => Self::Higgs,
            crate::router::RoutingMethod::Pattern => Self::Pattern,
            crate::router::RoutingMethod::Auto => Self::Auto,
            crate::router::RoutingMethod::Default => Self::Default,
        }
    }
}

impl std::fmt::Display for RoutingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pattern => write!(f, "pattern"),
            Self::Auto => write!(f, "auto"),
            Self::Default => write!(f, "default"),
            Self::Higgs => write!(f, "higgs"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RequestRecord {
    pub id: u64,
    pub timestamp: Instant,
    pub wallclock: DateTime<Utc>,
    pub model: String,
    pub provider: String,
    pub routing_method: RoutingMethod,
    pub status: u16,
    pub duration: Duration,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub error_body: Option<String>,
}

pub struct MetricsStore {
    records: RwLock<Vec<RequestRecord>>,
    id_index: RwLock<HashMap<u64, usize>>,
    window: Duration,
    logger: Option<Mutex<MetricsLogger>>,
    next_id: AtomicU64,
}

impl MetricsStore {
    pub fn new(window: Duration) -> Self {
        Self {
            records: RwLock::new(Vec::new()),
            id_index: RwLock::new(HashMap::new()),
            window,
            logger: None,
            next_id: AtomicU64::new(1),
        }
    }

    pub fn with_logger(window: Duration, logger: MetricsLogger) -> Self {
        Self {
            records: RwLock::new(Vec::new()),
            id_index: RwLock::new(HashMap::new()),
            window,
            logger: Some(Mutex::new(logger)),
            next_id: AtomicU64::new(1),
        }
    }

    #[allow(clippy::significant_drop_tightening)]
    pub fn record(&self, mut record: RequestRecord) {
        record.id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.log_record(&record);
        let mut records = self.records.write().unwrap_or_else(PoisonError::into_inner);
        let idx = records.len();
        let id = record.id;
        records.push(record);
        self.id_index
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id, idx);
    }

    /// Record a pending entry and return its stable ID for later finalization.
    #[allow(clippy::significant_drop_tightening)]
    pub fn record_pending(&self, mut record: RequestRecord) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        record.id = id;
        let mut records = self.records.write().unwrap_or_else(PoisonError::into_inner);
        let idx = records.len();
        records.push(record);
        self.id_index
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id, idx);
        id
    }

    /// Update `output_tokens` and duration for a previously recorded entry by ID.
    pub fn finalize_stream(&self, id: u64, output_tokens: u64, duration: Duration) {
        let completed = {
            let mut records = self.records.write().unwrap_or_else(PoisonError::into_inner);
            let index = self.id_index.read().unwrap_or_else(PoisonError::into_inner);
            if let Some(&idx) = index.get(&id) {
                if let Some(record) = records.get_mut(idx) {
                    record.output_tokens = output_tokens;
                    record.duration = duration;
                    Some(record.clone())
                } else {
                    None
                }
            } else {
                None
            }
        };
        if let Some(record) = completed {
            self.log_record(&record);
        }
    }

    pub fn snapshot(&self) -> Vec<RequestRecord> {
        #[allow(clippy::unchecked_time_subtraction)]
        let cutoff = Instant::now() - self.window;
        self.records
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|r| r.timestamp >= cutoff)
            .cloned()
            .collect()
    }

    pub const fn window(&self) -> Duration {
        self.window
    }

    pub const fn window_minutes(&self) -> u64 {
        self.window.as_secs() / 60
    }

    pub fn evict_expired(&self) {
        #[allow(clippy::unchecked_time_subtraction)]
        let cutoff = Instant::now() - self.window;
        let mut records = self.records.write().unwrap_or_else(PoisonError::into_inner);
        records.retain(|r| r.timestamp >= cutoff);

        let mut index = self
            .id_index
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        index.clear();
        for (i, record) in records.iter().enumerate() {
            index.insert(record.id, i);
        }
    }

    fn log_record(&self, record: &RequestRecord) {
        let Some(ref logger) = self.logger else {
            return;
        };
        let duration_ms = u64::try_from(record.duration.as_millis()).unwrap_or(u64::MAX);
        let entry = serde_json::json!({
            "timestamp": record.wallclock.to_rfc3339(),
            "model": &record.model,
            "provider": &record.provider,
            "routing_method": record.routing_method.to_string(),
            "status": record.status,
            "duration_ms": duration_ms,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "error": &record.error_body,
        });
        if let Ok(line) = serde_json::to_string(&entry)
            && let Ok(mut l) = logger.lock()
            && let Err(e) = l.write_line(&line)
        {
            tracing::warn!("failed to write metrics log: {e}");
        }
    }

    pub fn group_by<F, K>(records: &[RequestRecord], key_fn: F) -> HashMap<K, Vec<&RequestRecord>>
    where
        F: Fn(&RequestRecord) -> K,
        K: Eq + std::hash::Hash,
    {
        let mut groups: HashMap<K, Vec<&RequestRecord>> = HashMap::new();
        for record in records {
            groups.entry(key_fn(record)).or_default().push(record);
        }
        groups
    }

    pub fn duration_percentile(durations: &[Duration], p: u8) -> Duration {
        if durations.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted = durations.to_vec();
        sorted.sort();
        let len_minus_one = sorted.len().saturating_sub(1);
        // p is u8 (0..=255), len_minus_one fits in u64 for any realistic Vec.
        // Multiply in u64 to avoid overflow, then integer-round the division by 100.
        let numerator = u64::from(p) * u64::try_from(len_minus_one).unwrap_or(u64::MAX);
        // Round half-up: (numerator * 2 + 100) / 200 == round(numerator / 100)
        let index_u64 = numerator.saturating_mul(2).saturating_add(100) / 200;
        let index = usize::try_from(index_u64)
            .unwrap_or(len_minus_one)
            .min(len_minus_one);
        sorted.get(index).copied().unwrap_or(Duration::ZERO)
    }

    pub fn status_counts(records: &[RequestRecord]) -> HashMap<u16, u64> {
        let mut counts: HashMap<u16, u64> = HashMap::new();
        for record in records {
            *counts.entry(record.status).or_default() += 1;
        }
        counts
    }

    fn per_minute_buckets(
        records: &[RequestRecord],
        num_buckets: usize,
        value_fn: impl Fn(&RequestRecord) -> u64,
    ) -> Vec<u64> {
        let now = Instant::now();
        let mut buckets = vec![0u64; num_buckets];
        for record in records {
            if let Some(elapsed) = now.checked_duration_since(record.timestamp) {
                let bucket_secs = elapsed.as_secs() / 60;
                if let Ok(bucket_index) = usize::try_from(bucket_secs) {
                    if bucket_index < num_buckets {
                        let target = num_buckets.saturating_sub(1).saturating_sub(bucket_index);
                        if let Some(slot) = buckets.get_mut(target) {
                            *slot += value_fn(record);
                        }
                    }
                }
            }
        }
        buckets
    }

    pub fn tokens_per_minute(records: &[RequestRecord], num_buckets: usize) -> Vec<u64> {
        Self::per_minute_buckets(records, num_buckets, |r| r.input_tokens + r.output_tokens)
    }

    pub fn requests_per_minute(records: &[RequestRecord], num_buckets: usize) -> Vec<u64> {
        Self::per_minute_buckets(records, num_buckets, |_| 1)
    }
}

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::expect_used,
    clippy::unchecked_time_subtraction
)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn sample_record() -> RequestRecord {
        RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: Utc::now(),
            model: "claude-opus-4-6".to_owned(),
            provider: "anthropic".to_owned(),
            routing_method: RoutingMethod::Default,
            status: 200,
            duration: Duration::from_millis(500),
            input_tokens: 100,
            output_tokens: 200,
            error_body: None,
        }
    }

    #[test]
    fn window_returns_configured_duration() {
        let store = MetricsStore::new(Duration::from_secs(3600));
        assert_eq!(store.window(), Duration::from_secs(3600));
    }

    #[test]
    fn records_and_retrieves() {
        let store = MetricsStore::new(Duration::from_secs(60));
        store.record(sample_record());
        let snap = store.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].model, "claude-opus-4-6");
    }

    #[test]
    fn snapshot_excludes_expired() {
        let store = MetricsStore::new(Duration::from_millis(50));
        let mut old = sample_record();
        old.timestamp = Instant::now() - Duration::from_millis(100);
        store.record(old);
        store.record(sample_record());
        let snap = store.snapshot();
        assert_eq!(snap.len(), 1);
    }

    #[test]
    fn evict_removes_old_records() {
        let store = MetricsStore::new(Duration::from_millis(50));
        let mut old = sample_record();
        old.timestamp = Instant::now() - Duration::from_millis(100);
        store.record(old);
        store.record(sample_record());
        store.evict_expired();
        assert_eq!(store.records.read().unwrap().len(), 1);
    }

    #[test]
    fn snapshot_returns_owned_data() {
        let store = MetricsStore::new(Duration::from_secs(60));
        store.record(sample_record());
        let snap = store.snapshot();
        drop(snap);
        assert_eq!(store.snapshot().len(), 1);
    }

    #[test]
    fn group_by_model() {
        let store = MetricsStore::new(Duration::from_secs(60));
        for _ in 0..3 {
            store.record(sample_record());
        }
        let mut sonnet = sample_record();
        sonnet.model = "claude-sonnet-4-5-20250929".to_owned();
        store.record(sonnet);

        let snap = store.snapshot();
        let groups = MetricsStore::group_by(&snap, |r| r.model.clone());
        assert_eq!(groups.len(), 2);
        assert_eq!(groups["claude-opus-4-6"].len(), 3);
        assert_eq!(groups["claude-sonnet-4-5-20250929"].len(), 1);
    }

    #[test]
    fn status_counts_all_codes() {
        let store = MetricsStore::new(Duration::from_secs(60));
        for status in [200, 200, 429, 429, 429, 500] {
            let mut r = sample_record();
            r.status = status;
            store.record(r);
        }
        let snap = store.snapshot();
        let counts = MetricsStore::status_counts(&snap);
        assert_eq!(counts.len(), 3);
        assert_eq!(counts[&200], 2);
        assert_eq!(counts[&429], 3);
        assert_eq!(counts[&500], 1);
    }

    #[test]
    fn tokens_per_minute_buckets() {
        let store = MetricsStore::new(Duration::from_secs(300));
        for _ in 0..3 {
            let mut r = sample_record();
            r.input_tokens = 100;
            r.output_tokens = 50;
            store.record(r);
        }
        let snap = store.snapshot();
        let buckets = MetricsStore::tokens_per_minute(&snap, 5);
        assert_eq!(buckets.len(), 5);
        assert_eq!(*buckets.last().unwrap(), 450);
    }

    #[test]
    fn requests_per_minute_buckets() {
        let store = MetricsStore::new(Duration::from_secs(300));
        for _ in 0..5 {
            store.record(sample_record());
        }
        let snap = store.snapshot();
        let buckets = MetricsStore::requests_per_minute(&snap, 5);
        assert_eq!(buckets.len(), 5);
        assert_eq!(*buckets.last().unwrap(), 5);
    }

    #[test]
    fn record_pending_returns_unique_ids() {
        let store = MetricsStore::new(Duration::from_secs(60));
        let id0 = store.record_pending(sample_record());
        let id1 = store.record_pending(sample_record());
        assert_ne!(id0, id1);
        assert!(id0 > 0);
        assert!(id1 > 0);
    }

    #[test]
    fn finalize_stream_updates_record_by_id() {
        let store = MetricsStore::new(Duration::from_secs(60));
        let mut rec = sample_record();
        rec.output_tokens = 0;
        rec.duration = Duration::ZERO;
        let id = store.record_pending(rec);

        store.finalize_stream(id, 500, Duration::from_secs(3));

        let snap = store.snapshot();
        let record = snap.iter().find(|r| r.id == id).expect("record not found");
        assert_eq!(record.output_tokens, 500);
        assert_eq!(record.duration, Duration::from_secs(3));
    }

    #[test]
    fn finalize_stream_ignores_unknown_id() {
        let store = MetricsStore::new(Duration::from_secs(60));
        store.record(sample_record());
        store.finalize_stream(999_999, 100, Duration::from_secs(1));
        assert_eq!(store.snapshot().len(), 1);
    }

    #[test]
    fn finalize_stable_after_eviction() {
        let store = MetricsStore::new(Duration::from_millis(50));
        let mut old = sample_record();
        old.timestamp = Instant::now() - Duration::from_millis(100);
        store.record(old);

        let mut rec = sample_record();
        rec.output_tokens = 0;
        let id = store.record_pending(rec);

        store.evict_expired();

        store.finalize_stream(id, 999, Duration::from_secs(5));
        let snap = store.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].output_tokens, 999);
    }

    fn store_with_logger(dir: &std::path::Path) -> MetricsStore {
        let config = crate::config::MetricsLogConfig {
            enabled: true,
            path: dir.join("metrics.jsonl").to_string_lossy().to_string(),
            max_size_mb: 50,
            max_files: 5,
        };
        let logger = crate::metrics_log::MetricsLogger::new(&config).unwrap();
        MetricsStore::with_logger(Duration::from_secs(60), logger)
    }

    #[test]
    fn record_writes_to_logger() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_with_logger(dir.path());

        store.record(sample_record());

        let content = std::fs::read_to_string(dir.path().join("metrics.jsonl")).unwrap();
        let entry: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(entry["model"], "claude-opus-4-6");
        assert_eq!(entry["status"], 200);
        assert_eq!(entry["provider"], "anthropic");
    }

    #[test]
    fn finalize_stream_writes_to_logger() {
        let dir = tempfile::tempdir().unwrap();
        let store = store_with_logger(dir.path());

        let mut rec = sample_record();
        rec.output_tokens = 0;
        rec.duration = Duration::ZERO;
        let id = store.record_pending(rec);

        let content = std::fs::read_to_string(dir.path().join("metrics.jsonl")).unwrap();
        assert!(content.is_empty(), "record_pending should not log");

        store.finalize_stream(id, 500, Duration::from_secs(3));

        let log_content = std::fs::read_to_string(dir.path().join("metrics.jsonl")).unwrap();
        let entry: serde_json::Value = serde_json::from_str(log_content.trim()).unwrap();
        assert_eq!(entry["output_tokens"], 500);
        assert_eq!(entry["duration_ms"], 3000);
    }

    #[test]
    fn percentile_duration() {
        let store = MetricsStore::new(Duration::from_secs(60));
        for ms in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] {
            let mut r = sample_record();
            r.duration = Duration::from_millis(ms);
            store.record(r);
        }
        let snap = store.snapshot();
        let durations: Vec<Duration> = snap.iter().map(|r| r.duration).collect();
        let p50 = MetricsStore::duration_percentile(&durations, 50);
        let p95 = MetricsStore::duration_percentile(&durations, 95);
        assert!(p50.as_millis() >= 500 && p50.as_millis() <= 600);
        assert!(p95.as_millis() >= 900 && p95.as_millis() <= 1000);
    }

    #[test]
    fn percentile_empty_returns_zero() {
        let durations: Vec<Duration> = vec![];
        assert_eq!(
            MetricsStore::duration_percentile(&durations, 50),
            Duration::ZERO
        );
    }

    #[test]
    fn percentile_single_value() {
        let durations = vec![Duration::from_millis(42)];
        assert_eq!(
            MetricsStore::duration_percentile(&durations, 50),
            Duration::from_millis(42)
        );
        assert_eq!(
            MetricsStore::duration_percentile(&durations, 99),
            Duration::from_millis(42)
        );
    }

    #[test]
    fn routing_method_display() {
        assert_eq!(RoutingMethod::Pattern.to_string(), "pattern");
        assert_eq!(RoutingMethod::Auto.to_string(), "auto");
        assert_eq!(RoutingMethod::Default.to_string(), "default");
        assert_eq!(RoutingMethod::Higgs.to_string(), "higgs");
    }

    #[test]
    fn higgs_routing_method_in_record() {
        let store = MetricsStore::new(Duration::from_secs(60));
        let mut rec = sample_record();
        rec.routing_method = RoutingMethod::Higgs;
        store.record(rec);
        let snap = store.snapshot();
        assert_eq!(snap[0].routing_method, RoutingMethod::Higgs);
    }

    #[test]
    fn concurrent_pending_finalize_and_eviction() {
        let store = Arc::new(MetricsStore::new(Duration::from_secs(60)));
        let iterations: u64 = 200;

        let writer_store = Arc::clone(&store);
        let writer = std::thread::spawn(move || {
            for i in 0..iterations {
                let mut rec = sample_record();
                rec.output_tokens = 0;
                let id = writer_store.record_pending(rec);
                writer_store.finalize_stream(id, i + 1, Duration::from_millis(i + 1));
            }
        });

        let evictor_store = Arc::clone(&store);
        let evictor = std::thread::spawn(move || {
            for _ in 0..iterations {
                evictor_store.evict_expired();
            }
        });

        writer.join().unwrap();
        evictor.join().unwrap();

        let snap = store.snapshot();
        assert_eq!(snap.len(), usize::try_from(iterations).unwrap());
        for record in &snap {
            assert!(
                record.output_tokens > 0,
                "record {} was never finalized",
                record.id
            );
        }
    }
}
