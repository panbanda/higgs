use std::collections::HashMap;

use axum::{Json, extract::State, http::StatusCode};
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
    Ok(Json(build_metrics_response(metrics)))
}

fn build_metrics_response(metrics: &MetricsStore) -> MetricsResponse {
    let snapshot = metrics.snapshot();
    let input_tokens: u64 = snapshot.iter().map(|r| r.input_tokens).sum();
    let output_tokens: u64 = snapshot.iter().map(|r| r.output_tokens).sum();
    let errors = u64::try_from(snapshot.iter().filter(|r| r.is_error()).count()).unwrap_or(0);
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
    }
}

fn build_groups(groups: HashMap<String, Vec<&crate::metrics::RequestRecord>>) -> Vec<MetricsGroup> {
    let mut out: Vec<MetricsGroup> = groups
        .into_iter()
        .map(|(name, records)| {
            let requests = u64::try_from(records.len()).unwrap_or(u64::MAX);
            let input_tokens: u64 = records.iter().map(|r| r.input_tokens).sum();
            let output_tokens: u64 = records.iter().map(|r| r.output_tokens).sum();
            // Failed requests contribute error counts, but not model/provider
            // latency percentiles because no inference latency was observed.
            let durations: Vec<_> = records
                .iter()
                .filter(|r| !r.is_error())
                .map(|r| r.duration)
                .collect();
            let errors =
                u64::try_from(records.iter().filter(|r| r.is_error()).count()).unwrap_or(0);

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

#[allow(clippy::panic, clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use chrono::Utc;

    use super::*;
    use crate::metrics::{MetricsStore, RequestRecord, RoutingMethod};

    fn sample_record(model: &str, provider: &str, status: u16) -> RequestRecord {
        RequestRecord {
            id: 0,
            timestamp: Instant::now(),
            wallclock: Utc::now(),
            model: Some(model.to_owned()),
            provider: Some(provider.to_owned()),
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
}
