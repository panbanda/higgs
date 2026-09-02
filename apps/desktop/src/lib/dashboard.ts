import type { RequestRecord } from "./types";

/** Aggregations over metrics-log records, ported from crates/higgs/src/metrics.rs. */

export function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const index = Math.min(sorted.length - 1, Math.floor((sorted.length * p) / 100));
  return sorted[index];
}

export interface LatencySummary {
  avg: number;
  p50: number;
  p95: number;
  p99: number;
}

export function latencySummary(records: RequestRecord[]): LatencySummary {
  const durations = records
    .filter((record) => !isError(record))
    .map((record) => record.duration_ms)
    .sort((a, b) => a - b);
  const avg = durations.length === 0 ? 0 : durations.reduce((sum, d) => sum + d, 0) / durations.length;
  return { avg, p50: percentile(durations, 50), p95: percentile(durations, 95), p99: percentile(durations, 99) };
}

/** TTFT percentiles over records that recorded one (local streaming only). */
export function ttftSummary(records: RequestRecord[]): LatencySummary & { samples: number } {
  const values = records
    .filter((record) => !isError(record) && typeof record.ttft_ms === "number")
    .map((record) => record.ttft_ms as number)
    .sort((a, b) => a - b);
  const avg = values.length === 0 ? 0 : values.reduce((sum, v) => sum + v, 0) / values.length;
  return { samples: values.length, avg, p50: percentile(values, 50), p95: percentile(values, 95), p99: percentile(values, 99) };
}

/** Output tokens per second of decode time, weighted by generation length. */
export function aggregateTokensPerSecond(records: RequestRecord[]): number | null {
  let tokens = 0;
  let decodeMs = 0;
  for (const record of records) {
    if (isError(record) || typeof record.ttft_ms !== "number" || record.output_tokens === 0) continue;
    const span = record.duration_ms - record.ttft_ms;
    if (span <= 0) continue;
    tokens += record.output_tokens;
    decodeMs += span;
  }
  return tokens > 0 && decodeMs > 0 ? tokens / (decodeMs / 1000) : null;
}

export function cachedTokens(records: RequestRecord[]): number {
  return records.reduce((sum, record) => sum + (record.cached_tokens ?? 0), 0);
}

/**
 * Cache-hit rate over only the records that reported `cached_tokens`,
 * dividing their cached tokens by their input tokens. Returns null when no
 * record reported a cache-hit count.
 */
export function cacheHitRate(records: RequestRecord[]): number | null {
  const reporting = records.filter((record) => record.cached_tokens != null);
  if (reporting.length === 0) return null;
  const cached = reporting.reduce((sum, record) => sum + (record.cached_tokens ?? 0), 0);
  const input = reporting.reduce((sum, record) => sum + record.input_tokens, 0);
  return input > 0 ? (cached / input) * 100 : 0;
}

export function isError(record: RequestRecord): boolean {
  return record.status >= 400;
}

export function statusCounts(records: RequestRecord[]): Array<{ status: number; count: number }> {
  const counts = new Map<number, number>();
  for (const record of records) counts.set(record.status, (counts.get(record.status) ?? 0) + 1);
  return [...counts.entries()].map(([status, count]) => ({ status, count })).sort((a, b) => a.status - b.status);
}

export function statusLabel(code: number): string {
  const labels: Record<number, string> = {
    200: "OK",
    201: "Created",
    204: "No Content",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    429: "Rate Limited",
    500: "Internal Error",
    502: "Bad Gateway",
    503: "Unavailable",
    504: "Gateway Timeout",
    529: "Overloaded",
  };
  return labels[code] ? `${code} ${labels[code]}` : String(code);
}

export type StatusClass = "ok" | "client" | "server";

export function statusClass(code: number): StatusClass {
  if (code < 400) return "ok";
  if (code < 500) return "client";
  return "server";
}

export interface GroupStats {
  name: string;
  requests: number;
  input_tokens: number;
  output_tokens: number;
  avg_ms: number;
  p50_ms: number;
  p95_ms: number;
  errors: number;
  ttft_p50_ms: number | null;
  ttft_p95_ms: number | null;
  tokens_per_second: number | null;
  cached_tokens: number;
}

export function groupBy(records: RequestRecord[], key: (record: RequestRecord) => string | null): GroupStats[] {
  const groups = new Map<string, RequestRecord[]>();
  for (const record of records) {
    const name = key(record) ?? "(none)";
    const list = groups.get(name);
    if (list) list.push(record);
    else groups.set(name, [record]);
  }
  return [...groups.entries()]
    .map(([name, list]) => {
      const latency = latencySummary(list);
      const ttft = ttftSummary(list);
      return {
        name,
        requests: list.length,
        input_tokens: list.reduce((sum, r) => sum + r.input_tokens, 0),
        output_tokens: list.reduce((sum, r) => sum + r.output_tokens, 0),
        avg_ms: latency.avg,
        p50_ms: latency.p50,
        p95_ms: latency.p95,
        errors: list.filter(isError).length,
        ttft_p50_ms: ttft.samples > 0 ? ttft.p50 : null,
        ttft_p95_ms: ttft.samples > 0 ? ttft.p95 : null,
        tokens_per_second: aggregateTokensPerSecond(list),
        cached_tokens: cachedTokens(list),
      };
    })
    .sort((a, b) => a.name.localeCompare(b.name));
}

export interface MinuteBucket {
  /** Epoch ms at the start of the minute. */
  start: number;
  requests: number;
  tokens: number;
  errors: number;
}

/** Fixed-width per-minute buckets ending now, oldest first. */
export function perMinute(records: RequestRecord[], minutes: number, now = Date.now()): MinuteBucket[] {
  const end = Math.floor(now / 60_000) * 60_000;
  const buckets: MinuteBucket[] = [];
  for (let i = minutes - 1; i >= 0; i -= 1) {
    buckets.push({ start: end - i * 60_000, requests: 0, tokens: 0, errors: 0 });
  }
  const first = buckets[0]?.start ?? end;
  for (const record of records) {
    const at = Date.parse(record.timestamp);
    if (!Number.isFinite(at) || at < first) continue;
    const index = Math.min(buckets.length - 1, Math.floor((at - first) / 60_000));
    const bucket = buckets[index];
    bucket.requests += 1;
    bucket.tokens += record.input_tokens + record.output_tokens;
    if (isError(record)) bucket.errors += 1;
  }
  return buckets;
}

export function withinWindow(records: RequestRecord[], minutes: number, now = Date.now()): RequestRecord[] {
  const cutoff = now - minutes * 60_000;
  return records.filter((record) => Date.parse(record.timestamp) >= cutoff);
}

export function routingLabel(method: string | null): string {
  switch (method) {
    case "higgs":
      return "local";
    case "pattern":
      return "route";
    case "auto":
      return "auto";
    case "default":
      return "default";
    default:
      return method ?? "–";
  }
}
