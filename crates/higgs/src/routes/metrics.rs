use std::collections::HashMap;

use axum::{Json, extract::State, http::StatusCode};
use higgs_engine::simple::CacheStats;
use serde::Serialize;

use crate::{metrics::MetricsStore, state::SharedState};

#[derive(Debug, Serialize)]
pub struct MetricsResponse {
    pub window_minutes: u64,
    pub totals: MetricsTotals,
    pub status_counts: HashMap<u16, u64>,
    pub requests_per_minute: Vec<u64>,
    pub tokens_per_minute: Vec<u64>,
    pub models: Vec<MetricsGroup>,
    pub providers: Vec<MetricsGroup>,
    pub cache: CacheMetricsView,
}

/// Cache-resident KV effectiveness, aggregated across all local engines.
#[derive(Debug, Default, Serialize)]
pub struct CacheMetricsView {
    /// Radix prefix-cache lookups on the normal generate path.
    pub radix_lookups: u64,
    /// Radix prefix-cache hits (a stored prefix was reused).
    pub radix_hits: u64,
    /// Memory-only paired target/dSpark radix lookups.
    pub paired_radix_lookups: u64,
    /// Paired radix hits that materialized and forked both cache halves.
    pub paired_radix_hits: u64,
    /// Prompt tokens NOT re-prefilled thanks to reuse (radix + continuation).
    pub prefill_saved_tokens: u64,
    /// Per-session continuations (best-effort retained-cache reuse).
    pub continuations: u64,
    /// Retained sessions evicted (count cap + idle TTL).
    pub sessions_evicted: u64,
    /// Currently retained per-session caches.
    pub retained_sessions: u64,
    /// Currently retained sessions that own an inseparable target/dSpark pair.
    pub retained_paired_sessions: u64,
    /// Conservative target bytes retained by paired sessions.
    pub retained_paired_target_bytes: u64,
    /// Conservative dSpark bytes retained by paired sessions.
    pub retained_paired_dflash_bytes: u64,
    /// Currently stored radix prefixes.
    pub radix_entries: u64,
    /// Currently stored paired target/dSpark radix endpoints.
    pub paired_radix_entries: u64,
    /// Conservative target bytes retained by paired radix endpoints.
    pub paired_radix_target_bytes: u64,
    /// Conservative dSpark bytes retained by paired radix endpoints.
    pub paired_radix_dflash_bytes: u64,
}

impl CacheMetricsView {
    fn add(&mut self, stats: CacheStats) {
        self.radix_lookups = self.radix_lookups.saturating_add(stats.radix_lookups);
        self.radix_hits = self.radix_hits.saturating_add(stats.radix_hits);
        self.paired_radix_lookups = self
            .paired_radix_lookups
            .saturating_add(stats.paired_radix_lookups);
        self.paired_radix_hits = self
            .paired_radix_hits
            .saturating_add(stats.paired_radix_hits);
        self.prefill_saved_tokens = self
            .prefill_saved_tokens
            .saturating_add(stats.prefill_saved_tokens);
        self.continuations = self.continuations.saturating_add(stats.continuations);
        self.sessions_evicted = self.sessions_evicted.saturating_add(stats.sessions_evicted);
        self.retained_sessions = self
            .retained_sessions
            .saturating_add(u64::try_from(stats.retained_sessions).unwrap_or(u64::MAX));
        self.retained_paired_sessions = self
            .retained_paired_sessions
            .saturating_add(u64::try_from(stats.retained_paired_sessions).unwrap_or(u64::MAX));
        self.retained_paired_target_bytes = self
            .retained_paired_target_bytes
            .saturating_add(u64::try_from(stats.retained_paired_target_bytes).unwrap_or(u64::MAX));
        self.retained_paired_dflash_bytes = self
            .retained_paired_dflash_bytes
            .saturating_add(u64::try_from(stats.retained_paired_dflash_bytes).unwrap_or(u64::MAX));
        self.radix_entries = self
            .radix_entries
            .saturating_add(u64::try_from(stats.radix_entries).unwrap_or(u64::MAX));
        self.paired_radix_entries = self
            .paired_radix_entries
            .saturating_add(u64::try_from(stats.paired_radix_entries).unwrap_or(u64::MAX));
        self.paired_radix_target_bytes = self
            .paired_radix_target_bytes
            .saturating_add(u64::try_from(stats.paired_radix_target_bytes).unwrap_or(u64::MAX));
        self.paired_radix_dflash_bytes = self
            .paired_radix_dflash_bytes
            .saturating_add(u64::try_from(stats.paired_radix_dflash_bytes).unwrap_or(u64::MAX));
    }
}

#[derive(Debug, Serialize)]
pub struct MetricsTotals {
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub errors: u64,
}

#[derive(Debug, Serialize)]
pub struct MetricsGroup {
    pub name: String,
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub p50_ms: u64,
    pub p95_ms: u64,
    pub errors: u64,
}

pub async fn metrics(
    State(state): State<SharedState>,
) -> Result<Json<MetricsResponse>, StatusCode> {
    let Some(metrics) = state.metrics.as_ref() else {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };
    let mut response = build_metrics_response(metrics);
    response.cache = aggregate_cache(&state.router);
    Ok(Json(response))
}

fn build_metrics_response(metrics: &MetricsStore) -> MetricsResponse {
    let snapshot = metrics.snapshot();
    let input_tokens: u64 = snapshot.iter().map(|r| r.input_tokens).sum();
    let output_tokens: u64 = snapshot.iter().map(|r| r.output_tokens).sum();
    let errors = u64::try_from(snapshot.iter().filter(|r| r.status >= 400).count()).unwrap_or(0);
    let num_buckets = usize::try_from(metrics.window_minutes().max(1)).unwrap_or(1);

    MetricsResponse {
        window_minutes: metrics.window_minutes(),
        totals: MetricsTotals {
            requests: u64::try_from(snapshot.len()).unwrap_or(u64::MAX),
            input_tokens,
            output_tokens,
            errors,
        },
        status_counts: MetricsStore::status_counts(&snapshot),
        requests_per_minute: MetricsStore::requests_per_minute(&snapshot, num_buckets),
        tokens_per_minute: MetricsStore::tokens_per_minute(&snapshot, num_buckets),
        models: build_groups(MetricsStore::group_by(&snapshot, |r| r.model.clone())),
        providers: build_groups(MetricsStore::group_by(&snapshot, |r| r.provider.clone())),
        cache: CacheMetricsView::default(),
    }
}

/// Aggregate cache-effectiveness counters across all local engines.
fn aggregate_cache(router: &crate::router::Router) -> CacheMetricsView {
    let mut v = CacheMetricsView::default();
    for engine in router.local_engines().values() {
        if let Some(s) = engine.cache_stats() {
            v.add(s);
        }
    }
    v
}

fn build_groups(groups: HashMap<String, Vec<&crate::metrics::RequestRecord>>) -> Vec<MetricsGroup> {
    let mut out: Vec<MetricsGroup> = groups
        .into_iter()
        .map(|(name, records)| {
            let requests = u64::try_from(records.len()).unwrap_or(u64::MAX);
            let input_tokens: u64 = records.iter().map(|r| r.input_tokens).sum();
            let output_tokens: u64 = records.iter().map(|r| r.output_tokens).sum();
            let durations: Vec<_> = records.iter().map(|r| r.duration).collect();
            let errors =
                u64::try_from(records.iter().filter(|r| r.status >= 400).count()).unwrap_or(0);

            MetricsGroup {
                name,
                requests,
                input_tokens,
                output_tokens,
                p50_ms: u64::try_from(
                    MetricsStore::duration_percentile(&durations, 50).as_millis(),
                )
                .unwrap_or(u64::MAX),
                p95_ms: u64::try_from(
                    MetricsStore::duration_percentile(&durations, 95).as_millis(),
                )
                .unwrap_or(u64::MAX),
                errors,
            }
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use std::time::{Duration, Instant};

    use chrono::Utc;

    use super::*;
    use crate::metrics::{MetricsStore, RequestRecord, RoutingMethod};

    fn cache_stats_fixture() -> CacheStats {
        CacheStats {
            radix_lookups: 1,
            radix_hits: 2,
            paired_radix_lookups: 3,
            paired_radix_hits: 4,
            prefill_saved_tokens: 5,
            continuations: 6,
            sessions_evicted: 7,
            retained_sessions: 8,
            retained_paired_sessions: 9,
            retained_paired_target_bytes: 10,
            retained_paired_dflash_bytes: 11,
            radix_entries: 12,
            paired_radix_entries: 13,
            paired_radix_target_bytes: 14,
            paired_radix_dflash_bytes: 15,
        }
    }

    fn sample_record(model: &str, provider: &str, status: u16) -> RequestRecord {
        RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: Utc::now(),
            model: model.to_owned(),
            provider: provider.to_owned(),
            routing_method: RoutingMethod::Higgs,
            status,
            duration: Duration::from_millis(120),
            input_tokens: 10,
            output_tokens: 20,
            error_body: None,
        }
    }

    #[test]
    #[allow(unknown_lints)]
    #[allow(clippy::duration_suboptimal_units)]
    fn response_aggregates_snapshot() {
        let metrics = MetricsStore::new(Duration::from_secs(60));
        metrics.record(sample_record("model-a", "higgs", 200));
        metrics.record(sample_record("model-a", "higgs", 500));
        metrics.record(sample_record("model-b", "openai", 200));

        let response = build_metrics_response(&metrics);
        assert_eq!(response.totals.requests, 3);
        assert_eq!(response.totals.input_tokens, 30);
        assert_eq!(response.totals.output_tokens, 60);
        assert_eq!(response.totals.errors, 1);
        assert_eq!(response.models.len(), 2);
        assert_eq!(response.providers.len(), 2);
        assert_eq!(response.status_counts.get(&500), Some(&1));
    }

    #[test]
    fn cache_metrics_aggregate_every_paired_stat() {
        let mut view = CacheMetricsView::default();
        view.add(cache_stats_fixture());
        view.add(cache_stats_fixture());

        assert_eq!(view.paired_radix_lookups, 6);
        assert_eq!(view.paired_radix_hits, 8);
        assert_eq!(view.retained_paired_sessions, 18);
        assert_eq!(view.retained_paired_target_bytes, 20);
        assert_eq!(view.retained_paired_dflash_bytes, 22);
        assert_eq!(view.paired_radix_entries, 26);
        assert_eq!(view.paired_radix_target_bytes, 28);
        assert_eq!(view.paired_radix_dflash_bytes, 30);
    }

    #[test]
    fn cache_metrics_render_paired_fields_with_stable_names() {
        let mut view = CacheMetricsView::default();
        view.add(cache_stats_fixture());
        let rendered = serde_json::to_value(view).unwrap();

        for (name, expected) in [
            ("paired_radix_lookups", 3),
            ("paired_radix_hits", 4),
            ("retained_paired_sessions", 9),
            ("retained_paired_target_bytes", 10),
            ("retained_paired_dflash_bytes", 11),
            ("paired_radix_entries", 13),
            ("paired_radix_target_bytes", 14),
            ("paired_radix_dflash_bytes", 15),
        ] {
            assert_eq!(
                rendered.get(name).and_then(serde_json::Value::as_u64),
                Some(expected),
                "missing or renamed paired cache metric {name}"
            );
        }
    }
}
