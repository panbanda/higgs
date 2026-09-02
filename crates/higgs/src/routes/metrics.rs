use std::collections::HashMap;
use std::time::Duration;

use axum::{Json, extract::State, http::StatusCode};
use serde::Serialize;

use crate::{metrics::MetricsStore, state::SharedState};

#[derive(Debug, Serialize)]
pub struct MetricsResponse {
    pub window_minutes: u64,
    pub totals: MetricsTotals,
    /// Request duration over successful requests in the window.
    pub latency: LatencySummary,
    /// Time to first token over local streaming requests in the window.
    pub ttft: LatencySummary,
    /// Output tokens per second of decode time, aggregated over requests
    /// with a known TTFT.
    pub tokens_per_second: Option<f64>,
    pub status_counts: HashMap<u16, u64>,
    pub requests_per_minute: Vec<u64>,
    pub tokens_per_minute: Vec<u64>,
    pub models: Vec<MetricsGroup>,
    pub providers: Vec<MetricsGroup>,
}

#[derive(Debug, Default, Serialize)]
pub struct LatencySummary {
    /// Number of requests the percentiles were computed from.
    pub samples: u64,
    pub avg_ms: u64,
    pub p50_ms: u64,
    pub p95_ms: u64,
    pub p99_ms: u64,
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
    pub avg_ms: u64,
    pub p50_ms: u64,
    pub p95_ms: u64,
    pub errors: u64,
    pub ttft_p50_ms: Option<u64>,
    pub ttft_p95_ms: Option<u64>,
    pub tokens_per_second: Option<f64>,
    pub cached_tokens: u64,
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

    let successful: Vec<&crate::metrics::RequestRecord> =
        snapshot.iter().filter(|r| !r.is_error()).collect();
    MetricsResponse {
        window_minutes: metrics.window_minutes(),
        totals: MetricsTotals {
            requests: u64::try_from(snapshot.len()).unwrap_or(u64::MAX),
            input_tokens,
            output_tokens,
            errors,
        },
        latency: latency_summary(successful.iter().map(|r| r.duration)),
        ttft: latency_summary(
            successful
                .iter()
                .filter_map(|r| r.timing.ttft_ms.map(Duration::from_millis)),
        ),
        tokens_per_second: aggregate_tokens_per_second(&successful),
        status_counts: MetricsStore::status_counts(&snapshot),
        requests_per_minute: MetricsStore::requests_per_minute(&snapshot, num_buckets),
        tokens_per_minute: MetricsStore::tokens_per_minute(&snapshot, num_buckets),
        models: build_groups(MetricsStore::group_by(&snapshot, |r| r.model.clone())),
        providers: build_groups(MetricsStore::group_by(&snapshot, |r| r.provider.clone())),
    }
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn latency_summary(samples: impl Iterator<Item = Duration>) -> LatencySummary {
    let durations: Vec<Duration> = samples.collect();
    if durations.is_empty() {
        return LatencySummary::default();
    }
    let total: Duration = durations.iter().sum();
    let count = u32::try_from(durations.len()).unwrap_or(u32::MAX);
    LatencySummary {
        samples: u64::from(count),
        avg_ms: millis(total / count),
        p50_ms: millis(MetricsStore::duration_percentile(&durations, 50)),
        p95_ms: millis(MetricsStore::duration_percentile(&durations, 95)),
        p99_ms: millis(MetricsStore::duration_percentile(&durations, 99)),
    }
}

/// Total output tokens over total decode time, so long generations weigh
/// more than short ones instead of averaging per-request rates.
fn aggregate_tokens_per_second(records: &[&crate::metrics::RequestRecord]) -> Option<f64> {
    let mut tokens: u64 = 0;
    let mut decode = Duration::ZERO;
    for record in records {
        if let Some(span) = record.decode_duration() {
            if record.output_tokens > 0 && !span.is_zero() {
                tokens = tokens.saturating_add(record.output_tokens);
                decode = decode.saturating_add(span);
            }
        }
    }
    if tokens == 0 || decode.is_zero() {
        return None;
    }
    Some(u32::try_from(tokens).map_or(f64::MAX, f64::from) / decode.as_secs_f64())
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
            let successful: Vec<&crate::metrics::RequestRecord> =
                records.iter().copied().filter(|r| !r.is_error()).collect();
            let latency = latency_summary(successful.iter().map(|r| r.duration));
            let ttft = latency_summary(
                successful
                    .iter()
                    .filter_map(|r| r.timing.ttft_ms.map(Duration::from_millis)),
            );
            let errors =
                u64::try_from(records.iter().filter(|r| r.is_error()).count()).unwrap_or(0);

            MetricsGroup {
                name,
                requests,
                input_tokens,
                output_tokens,
                avg_ms: latency.avg_ms,
                p50_ms: latency.p50_ms,
                p95_ms: latency.p95_ms,
                errors,
                ttft_p50_ms: (ttft.samples > 0).then_some(ttft.p50_ms),
                ttft_p95_ms: (ttft.samples > 0).then_some(ttft.p95_ms),
                tokens_per_second: aggregate_tokens_per_second(&successful),
                cached_tokens: records
                    .iter()
                    .map(|r| r.timing.cached_tokens.unwrap_or(0))
                    .sum(),
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
            timing: crate::metrics::RequestTiming::default(),
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
        // Latency only counts the two successful requests.
        assert_eq!(response.latency.samples, 2);
        assert_eq!(response.latency.p50_ms, 120);
        // Nothing recorded a first token, so TTFT and throughput are absent.
        assert_eq!(response.ttft.samples, 0);
        assert!(response.tokens_per_second.is_none());
        let model_a = response
            .models
            .iter()
            .find(|g| g.name == "model-a")
            .unwrap();
        assert_eq!(model_a.avg_ms, 120);
        assert!(model_a.ttft_p50_ms.is_none());
        assert_eq!(model_a.cached_tokens, 0);
    }

    #[test]
    fn streaming_timing_feeds_ttft_and_throughput() {
        let metrics = MetricsStore::new(Duration::from_secs(60));
        let mut record = sample_record("model-a", "higgs", 200);
        // 20 tokens over a 2s request with a 1s TTFT: 20 tok/s of decode.
        record.duration = Duration::from_secs(2);
        record.timing = crate::metrics::RequestTiming {
            ttft_ms: Some(1000),
            cached_tokens: Some(7),
        };
        metrics.record(record);
        metrics.record(sample_record("model-a", "higgs", 200));

        let response = build_metrics_response(&metrics);
        assert_eq!(response.ttft.samples, 1);
        assert_eq!(response.ttft.p50_ms, 1000);
        let throughput = response.tokens_per_second.unwrap();
        assert!((throughput - 20.0).abs() < f64::EPSILON, "{throughput}");
        let model_a = response
            .models
            .iter()
            .find(|g| g.name == "model-a")
            .unwrap();
        assert_eq!(model_a.ttft_p50_ms, Some(1000));
        assert_eq!(model_a.cached_tokens, 7);
        assert!((model_a.tokens_per_second.unwrap() - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn zero_decode_span_does_not_contribute_to_throughput() {
        let metrics = MetricsStore::new(Duration::from_secs(60));
        let mut record = sample_record("model-a", "higgs", 200);
        // TTFT at or past the whole request duration leaves no decode time,
        // even though output tokens were recorded.
        record.duration = Duration::from_secs(1);
        record.timing = crate::metrics::RequestTiming {
            ttft_ms: Some(1000),
            cached_tokens: None,
        };
        metrics.record(record);

        let response = build_metrics_response(&metrics);
        assert!(response.tokens_per_second.is_none());
    }
}
