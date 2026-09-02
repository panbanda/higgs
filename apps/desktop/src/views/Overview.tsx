import { Sparkline } from "../components/charts/Sparkline";
import { BarChart, type BarDatum } from "../components/charts/BarChart";
import { DataTable, type Column } from "../components/DataTable";
import {
  aggregateTokensPerSecond,
  latencySummary,
  perMinute,
  routingLabel,
  statusClass,
  statusCounts,
  statusLabel,
  ttftSummary,
  withinWindow,
} from "../lib/dashboard";
import { formatAge, formatMs, formatRate } from "../lib/format";
import type { RequestRecord } from "../lib/types";
import type { ServerData } from "../hooks/useServerData";
import type { Settings } from "../lib/types";
import "./dashboard.css";

export interface OverviewProps {
  settings: Settings;
  data: ServerData;
  onOpenRequests: () => void;
}

const TTFT_THRESHOLD_MS = 1500;
const TTFT_SCALE_MAX = TTFT_THRESHOLD_MS / 0.75;
const LATENCY_WARN_MS = 2000;

function minuteLabels(count: number): string[] {
  return Array.from({ length: count }, (_, i) => {
    const offset = count - 1 - i;
    return offset === 0 ? "now" : `-${offset}m`;
  });
}

/** Buckets records into fixed-width per-minute windows, newest last, keeping full records for per-bucket aggregation. */
function minuteRecordBuckets(records: RequestRecord[], minutes: number, now = Date.now()): RequestRecord[][] {
  const end = Math.floor(now / 60_000) * 60_000;
  const first = end - (minutes - 1) * 60_000;
  const buckets: RequestRecord[][] = Array.from({ length: minutes }, () => []);
  for (const record of records) {
    const at = Date.parse(record.timestamp);
    if (!Number.isFinite(at) || at < first) continue;
    const index = Math.min(minutes - 1, Math.floor((at - first) / 60_000));
    buckets[index].push(record);
  }
  return buckets;
}

function decodeRate(record: RequestRecord): number | null {
  if (record.status >= 400 || typeof record.ttft_ms !== "number" || record.output_tokens === 0) return null;
  const span = record.duration_ms - record.ttft_ms;
  return span > 0 ? record.output_tokens / (span / 1000) : null;
}

/** Maps statusClass's ok/client/server to the square-marker color modifiers. */
function statusTone(code: number): "ok" | "warn" | "bad" {
  const cls = statusClass(code);
  if (cls === "ok") return "ok";
  if (cls === "client") return "warn";
  return "bad";
}

interface StatusRow {
  status: number;
  count: number;
}

export function Overview({ settings, data, onOpenRequests }: OverviewProps) {
  const windowMinutes = Math.max(1, settings.windowMinutes);
  const now = Date.now();
  const usingLiveLog = data.records.length > 0;
  const windowed = usingLiveLog ? withinWindow(data.records, windowMinutes) : [];

  const ttft = usingLiveLog
    ? ttftSummary(windowed)
    : data.metrics
      ? { samples: data.metrics.ttft.samples, p50: data.metrics.ttft.p50_ms, p95: data.metrics.ttft.p95_ms }
      : { samples: 0, p50: 0, p95: 0 };

  const durationSummary = usingLiveLog
    ? (() => {
        const s = latencySummary(windowed);
        return { avg_ms: s.avg, p50_ms: s.p50, p95_ms: s.p95, p99_ms: s.p99 };
      })()
    : (data.metrics?.latency ?? { avg_ms: NaN, p50_ms: NaN, p95_ms: NaN, p99_ms: NaN });

  const tokensPerSecond = usingLiveLog ? aggregateTokensPerSecond(windowed) : (data.metrics?.tokens_per_second ?? null);

  const prevWindowed = usingLiveLog
    ? data.records.filter((r) => {
        const at = Date.parse(r.timestamp);
        return at >= now - 2 * windowMinutes * 60_000 && at < now - windowMinutes * 60_000;
      })
    : [];
  const prevTokensPerSecond = usingLiveLog ? aggregateTokensPerSecond(prevWindowed) : null;
  const deltaPct =
    tokensPerSecond != null && prevTokensPerSecond != null && prevTokensPerSecond > 0
      ? ((tokensPerSecond - prevTokensPerSecond) / prevTokensPerSecond) * 100
      : null;

  const decodeSeries = usingLiveLog ? minuteRecordBuckets(windowed, windowMinutes, now).map((bucket) => aggregateTokensPerSecond(bucket) ?? 0) : [];

  const requestBuckets = usingLiveLog ? perMinute(windowed, windowMinutes, now) : null;
  const requestsPerMinute: BarDatum[] = requestBuckets
    ? requestBuckets.map((bucket, i) => ({ label: minuteLabels(requestBuckets.length)[i], value: bucket.requests, accent: bucket.errors || undefined }))
    : (data.metrics?.requests_per_minute ?? []).map((value, i, arr) => ({ label: minuteLabels(arr.length)[i], value }));

  const statuses: StatusRow[] = usingLiveLog
    ? statusCounts(windowed)
    : Object.entries(data.metrics?.status_counts ?? {})
        .map(([status, count]) => ({ status: Number(status), count }))
        .sort((a, b) => a.status - b.status);

  const liveLogRows = usingLiveLog
    ? [...windowed].sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp)).slice(0, 25)
    : [];

  const maxRequestsPerMinute = Math.max(2, ...requestsPerMinute.map((bucket) => bucket.value));

  const ttftBarPct = Math.min(100, (ttft.p95 / TTFT_SCALE_MAX) * 100);
  const ttftP50Pct = Math.min(100, (ttft.p50 / TTFT_SCALE_MAX) * 100);
  const ttftThresholdPct = (TTFT_THRESHOLD_MS / TTFT_SCALE_MAX) * 100;
  const ttftWithinThreshold = ttft.samples > 0 && ttft.p95 <= TTFT_THRESHOLD_MS;

  const latencyRows = [
    { label: "avg", value: durationSummary.avg_ms },
    { label: "p50", value: durationSummary.p50_ms },
    { label: "p95", value: durationSummary.p95_ms },
    { label: "p99", value: durationSummary.p99_ms },
  ];
  const latencyMax = Math.max(1, ...latencyRows.map((r) => r.value || 0));

  const logColumns: Column<RequestRecord>[] = [
    { key: "age", header: "Age", render: (row) => <span className="num">{formatAge(row.timestamp)}</span> },
    { key: "model", header: "Model", render: (row) => row.model ?? "–" },
    { key: "route", header: "Route", render: (row) => <span className="label">{routingLabel(row.routing_method)}</span> },
    {
      key: "status",
      header: "Status",
      render: (row) => (
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className={`status-square ${statusTone(row.status)}`} />
          <span className="mono" style={{ fontSize: 12 }}>
            {row.status}
          </span>
        </span>
      ),
    },
    { key: "ttft", header: "TTFT", align: "right", render: (row) => <span className="num">{typeof row.ttft_ms === "number" ? formatMs(row.ttft_ms) : "–"}</span> },
    {
      key: "decode",
      header: "Decode",
      align: "right",
      render: (row) => {
        const rate = decodeRate(row);
        return <span className="num">{rate != null ? formatRate(rate) : "–"}</span>;
      },
    },
    { key: "duration", header: "Duration", align: "right", render: (row) => <span className="num">{formatMs(row.duration_ms)}</span> },
    {
      key: "cache",
      header: "Cache",
      align: "right",
      render: (row) => {
        if (!row.input_tokens || row.cached_tokens == null) return <span className="num muted">–</span>;
        const pct = (row.cached_tokens / row.input_tokens) * 100;
        return (
          <span className={`num ${pct > 0 ? "cache-hit" : ""}`} style={{ color: pct > 0 ? "var(--ok)" : undefined }}>
            {Math.round(pct)}%
          </span>
        );
      },
    },
  ];

  return (
    <div className="view">
      <h1>Overview</h1>
      <p className="view-subtitle">
        Last {windowMinutes} minutes · Updated {Math.max(0, Math.round((now - data.lastRefresh) / 1000))}s ago
      </p>

      {!usingLiveLog && !data.localAvailable && (
        <div className="notice">The live request log needs the desktop app. Showing server-reported metrics instead.</div>
      )}
      {data.logError && <div className="notice bad">{data.logError}</div>}

      <div className="hero-row">
        <div className="panel hero-panel">
          <div className="hero-header">
            <span className="meta">Time to first token · p95</span>
            <span className="mono label">threshold {(TTFT_THRESHOLD_MS / 1000).toFixed(1)} s</span>
          </div>
          <div className="hero-value-row">
            <span className="key hero-value">
              {ttft.samples > 0 ? formatMs(ttft.p95) : "–"}
            </span>
            {ttft.samples > 0 && (
              <span className={`mono hero-note ${ttftWithinThreshold ? "ok" : "bad"}`}>
                {ttftWithinThreshold ? "within threshold" : "over threshold"}
              </span>
            )}
          </div>
          <div className="threshold-bar">
            <div className="threshold-fill" style={{ width: `${ttftBarPct}%` }} />
            <div className="threshold-marker warn" style={{ left: `${ttftThresholdPct}%` }} />
            <div className="threshold-marker ok" style={{ left: `${ttftP50Pct}%` }} />
          </div>
          <div className="mono hero-footer">
            <span>p50 {ttft.samples > 0 ? formatMs(ttft.p50) : "–"}</span>
            <span>p95 {ttft.samples > 0 ? formatMs(ttft.p95) : "–"}</span>
            <span>limit {(TTFT_THRESHOLD_MS / 1000).toFixed(1)} s</span>
          </div>
        </div>

        <div className="panel hero-panel">
          <div className="hero-header">
            <span className="meta">Decode speed</span>
            <span className="mono label">{windowMinutes}m window</span>
          </div>
          <div className="hero-value-row">
            <span className="key hero-value">{tokensPerSecond != null ? formatRate(tokensPerSecond).replace(" tok/s", "") : "–"}</span>
            <span className="label" style={{ fontSize: 14 }}>tok/s</span>
            {deltaPct != null && (
              <span className="mono hero-note neutral">
                {deltaPct >= 0 ? "+" : ""}
                {deltaPct.toFixed(0)}% vs previous window
              </span>
            )}
          </div>
          {decodeSeries.length > 1 ? (
            <Sparkline
              values={decodeSeries}
              labels={minuteLabels(decodeSeries.length)}
              formatValue={(v) => formatRate(v)}
              title="Decode speed"
              tone="neutral"
              showFooter={false}
            />
          ) : (
            <div className="muted small pad">Not enough data yet</div>
          )}
          <div className="mono hero-footer">
            <span>-{windowMinutes}m</span>
            <span>peak {formatRate(Math.max(0, ...decodeSeries))}</span>
            <span>now</span>
          </div>
        </div>

        <div className="panel hero-panel">
          <span className="meta">Latency · successful requests</span>
          <div className="latency-rows">
            {latencyRows.map((row) => (
              <div key={row.label} className="latency-row">
                <span className="mono label">{row.label}</span>
                <div className="latency-bar">
                  <div
                    className={`latency-bar-fill ${row.value >= LATENCY_WARN_MS ? "warn" : "neutral"}`}
                    style={{ width: `${Math.min(100, (row.value / latencyMax) * 100)}%` }}
                  />
                </div>
                <span className="mono latency-value">{Number.isFinite(row.value) ? formatMs(row.value) : "–"}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="live-row">
        <div className="panel live-log-panel">
          <div className="live-log-header">
            <span className="meta">Live log</span>
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <span className="mono label">click a row to inspect</span>
              <button type="button" className="btn small" onClick={onOpenRequests}>
                All requests
              </button>
            </div>
          </div>
          {usingLiveLog ? (
            <DataTable columns={logColumns} rows={liveLogRows} rowKey={(row) => `${row.timestamp}-${row.model ?? ""}`} dense empty="No requests yet" onRowClick={onOpenRequests} />
          ) : (
            <div className="muted small pad">Live log needs the desktop app.</div>
          )}
        </div>

        <div className="panel side-col">
          <div className="side-col-header">
            <span className="meta">Requests per minute</span>
            <span className="mono label">errors in red</span>
          </div>
          <BarChart data={requestsPerMinute} title="Requests per minute" showFooter={false} />
          <div className="mono side-col-footnote">
            <span>-{windowMinutes}m</span>
            <span>max {maxRequestsPerMinute}</span>
            <span>now</span>
          </div>
          <div className="rule" />
          <span className="meta">Status codes</span>
          <div className="status-list">
            {statuses.length === 0 && <div className="muted small">No traffic yet</div>}
            {statuses.map((row) => (
              <div key={row.status} className="status-row">
                <span className={`status-square ${statusTone(row.status)}`} />
                <span>{statusLabel(row.status)}</span>
                <span className="mono status-count">{row.count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
