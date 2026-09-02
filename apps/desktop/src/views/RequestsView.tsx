import { useEffect, useMemo, useState } from "react";
import { isError, routingLabel, statusClass, withinWindow } from "../lib/dashboard";
import { formatCompact, formatMs, formatNumber, formatRate } from "../lib/format";
import type { ServerData } from "../hooks/useServerData";
import type { RequestRecord, Settings } from "../lib/types";
import "./views.css";

export interface RequestsViewProps {
  settings: Settings;
  data: ServerData;
}

type RangeOption = "5" | "15" | "60" | "all";
type StatusOption = "all" | "2xx" | "4xx" | "5xx";

const STATUS_COLOR: Record<"ok" | "client" | "server", string> = { ok: "var(--ok)", client: "var(--warn)", server: "var(--bad)" };

function matchesStatusOption(status: number, option: StatusOption): boolean {
  if (option === "all") return true;
  if (option === "2xx") return status >= 200 && status < 300;
  if (option === "4xx") return status >= 400 && status < 500;
  return status >= 500;
}

function recordKey(record: RequestRecord): string {
  return `${record.timestamp}-${record.model ?? ""}-${record.duration_ms}-${record.input_tokens}-${record.output_tokens}`;
}

/** Decode throughput for a single record: output tokens over the post-TTFT span. */
function recordTokensPerSecond(record: RequestRecord): number | null {
  if (typeof record.ttft_ms !== "number" || record.output_tokens === 0) return null;
  const span = record.duration_ms - record.ttft_ms;
  if (span <= 0) return null;
  return record.output_tokens / (span / 1000);
}

/** Share of input tokens served from cache, as a whole percentage. */
function recordCachePercent(record: RequestRecord): number | null {
  if (typeof record.cached_tokens !== "number" || record.input_tokens <= 0) return null;
  return Math.round((record.cached_tokens / record.input_tokens) * 100);
}

export function RequestsView({ data }: RequestsViewProps) {
  const [model, setModel] = useState("all");
  const [provider, setProvider] = useState("all");
  const [statusOption, setStatusOption] = useState<StatusOption>("all");
  const [range, setRange] = useState<RangeOption>("all");
  const [search, setSearch] = useState("");
  const [errorsOnly, setErrorsOnly] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [includeNonInference, setIncludeNonInference] = useState(false);

  const source = includeNonInference ? data.allRecords : data.records;

  const models = useMemo(() => [...new Set(source.map((r) => r.model ?? "(none)"))].sort(), [source]);
  const providers = useMemo(() => [...new Set(source.map((r) => r.provider ?? "(none)"))].sort(), [source]);

  const ranged = range === "all" ? source : withinWindow(source, Number(range));

  const filtered = ranged.filter((record) => {
    if (model !== "all" && (record.model ?? "(none)") !== model) return false;
    if (provider !== "all" && (record.provider ?? "(none)") !== provider) return false;
    if (!matchesStatusOption(record.status, statusOption)) return false;
    if (errorsOnly && !isError(record)) return false;
    if (search.trim()) {
      const needle = search.trim().toLowerCase();
      const haystack = `${record.model ?? ""} ${record.provider ?? ""} ${record.error ?? ""}`.toLowerCase();
      if (!haystack.includes(needle)) return false;
    }
    return true;
  });

  const sorted = [...filtered].sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp));

  const errorRate = filtered.length === 0 ? 0 : (filtered.filter(isError).length / filtered.length) * 100;
  const avgDuration = filtered.length === 0 ? 0 : filtered.reduce((sum, r) => sum + r.duration_ms, 0) / filtered.length;

  const selectedRecord = sorted.find((record) => recordKey(record) === selected) ?? null;

  useEffect(() => {
    if (selected && !selectedRecord) setSelected(null);
  }, [selected, selectedRecord]);

  return (
    <div className="view">
      <div className="view-head">
        <h1>Requests</h1>
        <div style={{ display: "flex", gap: 8 }}>
          <input
            type="search"
            className="requests-search"
            placeholder="Search model, provider, error…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <button type="button" className={`btn ${errorsOnly ? "hot" : ""}`} onClick={() => setErrorsOnly((current) => !current)}>
            Errors only
          </button>
        </div>
      </div>

      {source.length === 0 && (
        <div className="notice">{data.logError ?? (!data.localAvailable ? "Live request logs need the desktop app." : "No requests logged yet.")}</div>
      )}

      <div className="requests-toolbar">
        <label className="field inline small">
          <input type="checkbox" checked={includeNonInference} onChange={(e) => setIncludeNonInference(e.target.checked)} />
          Include non-inference requests
        </label>
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="all">All models</option>
          {models.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <select value={provider} onChange={(e) => setProvider(e.target.value)}>
          <option value="all">All providers</option>
          {providers.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <select value={statusOption} onChange={(e) => setStatusOption(e.target.value as StatusOption)}>
          <option value="all">All statuses</option>
          <option value="2xx">2xx</option>
          <option value="4xx">4xx</option>
          <option value="5xx">5xx</option>
        </select>
        <select value={range} onChange={(e) => setRange(e.target.value as RangeOption)}>
          <option value="5">Last 5m</option>
          <option value="15">Last 15m</option>
          <option value="60">Last 60m</option>
          <option value="all">All time</option>
        </select>
      </div>

      <p className="requests-summary">
        <span className="num">{formatNumber(filtered.length)}</span> requests · error rate <span className="num">{errorRate.toFixed(1)}%</span> · avg
        duration <span className="num">{formatMs(avgDuration)}</span>
      </p>

      <div className="requests-layout">
        <div className="panel">
          <div className="vrow head requests-grid">
            <span className="meta">Time</span>
            <span className="meta">Model</span>
            <span className="meta">Route</span>
            <span className="meta">Status</span>
            <span className="meta" style={{ textAlign: "right" }}>
              TTFT
            </span>
            <span className="meta" style={{ textAlign: "right" }}>
              Decode
            </span>
            <span className="meta" style={{ textAlign: "right" }}>
              Duration
            </span>
            <span className="meta" style={{ textAlign: "right" }}>
              Cache
            </span>
            <span className="meta" style={{ textAlign: "right" }}>
              In / out
            </span>
          </div>
          {sorted.length === 0 && <div className="config-list-empty">No requests match these filters</div>}
          {sorted.map((record) => {
            const key = recordKey(record);
            const cls = statusClass(record.status);
            const cachePercent = recordCachePercent(record);
            const tps = recordTokensPerSecond(record);
            return (
              <div
                className={`vrow clickable requests-grid ${selected === key ? "selected" : ""}`}
                key={key}
                role="button"
                tabIndex={0}
                onClick={() => setSelected(key)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setSelected(key);
                  }
                }}
              >
                <span className="mono muted">{new Date(record.timestamp).toLocaleTimeString()}</span>
                <span>{record.model ?? "–"}</span>
                <span className="muted">{routingLabel(record.routing_method)}</span>
                <div className="state-cell">
                  <span className="swatch" style={{ background: STATUS_COLOR[cls] }} />
                  <span className="mono">{record.status}</span>
                </div>
                <span className="num" style={{ textAlign: "right" }}>
                  {typeof record.ttft_ms === "number" ? formatMs(record.ttft_ms) : "–"}
                </span>
                <span className="num" style={{ textAlign: "right" }}>
                  {tps != null ? formatRate(tps) : "–"}
                </span>
                <span className="num" style={{ textAlign: "right" }}>
                  {formatMs(record.duration_ms)}
                </span>
                <span className="num" style={{ textAlign: "right", color: cachePercent != null ? "var(--ok)" : undefined }}>
                  {cachePercent != null ? `${cachePercent}%` : "–"}
                </span>
                <span className="mono muted" style={{ textAlign: "right" }}>
                  {formatCompact(record.input_tokens)} / {formatCompact(record.output_tokens)}
                </span>
              </div>
            );
          })}
        </div>

        <div className="panel detail-panel">
          <span className="meta">Detail</span>
          {selectedRecord ? (
            <>
              <dl className="detail-dl">
                <dt className="label">Timestamp</dt>
                <dd className="mono">{new Date(selectedRecord.timestamp).toLocaleString()}</dd>
                <dt className="label">Model</dt>
                <dd>{selectedRecord.model ?? "–"}</dd>
                <dt className="label">Provider</dt>
                <dd>{selectedRecord.provider ?? "–"}</dd>
                <dt className="label">Route</dt>
                <dd>{routingLabel(selectedRecord.routing_method)}</dd>
                <dt className="label">Status</dt>
                <dd>
                  <span className="state-cell" style={{ justifyContent: "flex-end" }}>
                    <span className="swatch" style={{ background: STATUS_COLOR[statusClass(selectedRecord.status)] }} />
                    <span className="mono">{selectedRecord.status}</span>
                  </span>
                </dd>
                <dt className="label">Duration</dt>
                <dd className="num">{formatMs(selectedRecord.duration_ms)}</dd>
                <dt className="label">Input tokens</dt>
                <dd className="num">{formatNumber(selectedRecord.input_tokens)}</dd>
                <dt className="label">Output tokens</dt>
                <dd className="num">{formatNumber(selectedRecord.output_tokens)}</dd>
                <dt className="label">TTFT</dt>
                <dd className="num">{typeof selectedRecord.ttft_ms === "number" ? formatMs(selectedRecord.ttft_ms) : "–"}</dd>
                <dt className="label">Decode tok/s</dt>
                <dd className="num">
                  {(() => {
                    const rate = recordTokensPerSecond(selectedRecord);
                    return rate != null ? formatRate(rate) : "–";
                  })()}
                </dd>
                <dt className="label">Cached tokens</dt>
                <dd className="num">{typeof selectedRecord.cached_tokens === "number" ? formatNumber(selectedRecord.cached_tokens) : "–"}</dd>
              </dl>
              {selectedRecord.error && <pre className="detail-error">{selectedRecord.error}</pre>}
            </>
          ) : (
            <div className="detail-empty">Select a row to see full details.</div>
          )}
        </div>
      </div>
    </div>
  );
}
