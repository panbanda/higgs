import { cacheHitRate, ttftSummary, withinWindow } from "../lib/dashboard";
import { formatBytes, formatMs } from "../lib/format";
import type { ServerData } from "../hooks/useServerData";
import type { Settings } from "../lib/types";

interface Props {
  settings: Settings;
  data: ServerData;
}

type Tone = "ok" | "warn" | "bad" | "neutral";

interface SignalDatum {
  label: string;
  value: string;
  unit: string;
  state: string;
  tone: Tone;
  pct: number;
}

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86_400);
  const hours = Math.floor((seconds % 86_400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const parts: string[] = [];
  if (days > 0) parts.push(`${days}d`);
  if (days > 0 || hours > 0) parts.push(`${hours}h`);
  parts.push(`${minutes}m`);
  return parts.join(" ");
}

function clampPct(value: number): number {
  return Math.max(0, Math.min(100, value));
}

/** Six-tile telemetry row mounted under the header on every section. */
export function SignalStrip({ settings, data }: Props) {
  const windowMinutes = Math.max(1, settings.windowMinutes);
  const usingLiveLog = data.records.length > 0;
  const windowed = usingLiveLog ? withinWindow(data.records, windowMinutes) : [];

  const requestCount = usingLiveLog ? windowed.length : (data.metrics?.totals.requests ?? 0);
  const errorCount = usingLiveLog ? windowed.filter((r) => r.status >= 400).length : (data.metrics?.totals.errors ?? 0);
  const ttftP95 = usingLiveLog ? ttftSummary(windowed).p95 : (data.metrics?.ttft.p95_ms ?? 0);
  const ttftSamples = usingLiveLog ? ttftSummary(windowed).samples : (data.metrics?.ttft.samples ?? 0);

  const requestsPerMin = requestCount / windowMinutes;
  const errorRate = requestCount > 0 ? (errorCount / requestCount) * 100 : 0;
  const serverInputTokens = data.metrics?.totals.input_tokens ?? 0;
  const serverCachedTokens = (data.metrics?.models ?? []).reduce((sum, m) => sum + m.cached_tokens, 0);
  const cacheHitPct = usingLiveLog
    ? cacheHitRate(windowed)
    : serverInputTokens > 0
      ? (serverCachedTokens / serverInputTokens) * 100
      : null;

  const mem = data.system?.memory;
  const memPct =
    mem?.mlx_active_bytes != null && mem.physical_total_bytes ? (mem.mlx_active_bytes / mem.physical_total_bytes) * 100 : null;

  const uptime = data.system?.uptime_secs ?? null;

  const signals: SignalDatum[] = [
    {
      label: "Requests/min",
      value: requestsPerMin.toFixed(requestsPerMin < 10 ? 1 : 0),
      unit: "req/min",
      state: requestCount > 0 ? "active" : "idle",
      tone: requestCount > 0 ? "ok" : "neutral",
      pct: clampPct((requestsPerMin / 30) * 100),
    },
    {
      label: "Errors",
      value: errorRate.toFixed(1),
      unit: "%",
      state: errorRate < 1 ? "ok" : errorRate < 5 ? "watch" : "high",
      tone: errorRate < 1 ? "ok" : errorRate < 5 ? "warn" : "bad",
      pct: clampPct(errorRate * 5),
    },
    {
      label: "Memory",
      value: mem?.mlx_active_bytes != null ? formatBytes(mem.mlx_active_bytes).replace(" ", "") : "–",
      unit: mem?.physical_total_bytes != null ? `of ${formatBytes(mem.physical_total_bytes)}` : "",
      state: memPct == null ? "–" : `${Math.round(memPct)}%`,
      tone: memPct == null ? "neutral" : memPct < 70 ? "ok" : memPct < 90 ? "warn" : "bad",
      pct: clampPct(memPct ?? 0),
    },
    {
      label: "Cache hit",
      value: cacheHitPct != null ? cacheHitPct.toFixed(0) : "–",
      unit: cacheHitPct != null ? "%" : "",
      state: cacheHitPct == null ? "no data" : cacheHitPct >= 50 ? "warm" : cacheHitPct > 0 ? "partial" : "cold",
      tone: cacheHitPct == null ? "neutral" : cacheHitPct >= 50 ? "ok" : cacheHitPct > 0 ? "warn" : "neutral",
      pct: clampPct(cacheHitPct ?? 0),
    },
    {
      label: "TTFT p95",
      value: ttftSamples > 0 ? formatMs(ttftP95) : "–",
      unit: "",
      state: ttftSamples === 0 ? "–" : ttftP95 <= 1500 ? "ok" : "over threshold",
      tone: ttftSamples === 0 ? "neutral" : ttftP95 <= 1500 ? "ok" : "bad",
      pct: clampPct((ttftP95 / 2000) * 100),
    },
    {
      label: "Uptime",
      value: uptime != null ? formatUptime(uptime) : "–",
      unit: "",
      state: uptime != null ? "up" : "–",
      tone: "neutral",
      pct: uptime != null ? clampPct((uptime / 604_800) * 100) : 0,
    },
  ];
  return (
    <div className="signal-strip">
      {signals.map((signal) => (
        <div key={signal.label} className="signal">
          <div className="signal-top">
            <span className="meta">{signal.label}</span>
            <span className={`signal-state ${signal.tone}`}>{signal.state}</span>
          </div>
          <span className="signal-value">
            {signal.value}
            {signal.unit && <span className="signal-unit">{signal.unit}</span>}
          </span>
          <div className="signal-bar">
            <div className={`signal-bar-fill ${signal.tone}`} style={{ width: `${signal.pct}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}
