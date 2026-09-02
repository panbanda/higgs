import { groupBy, type GroupStats } from "../lib/dashboard";
import { formatMs, formatNumber, formatRate } from "../lib/format";
import type { ServerData } from "../hooks/useServerData";
import type { Settings } from "../lib/types";
import "./views.css";

export interface RoutingViewProps {
  settings: Settings;
  data: ServerData;
  onEditConfig: () => void;
}

interface ProviderRow {
  name: string;
  url: string;
  format: string;
  stripAuth: boolean;
  apiKeySet: boolean;
  traffic: GroupStats | null;
}

export function RoutingView({ data, onEditConfig }: RoutingViewProps) {
  const config = data.config;
  const hasConfig = Object.keys(config).length > 0;
  const traffic = groupBy(data.records, (r) => r.provider);
  const providerEntries = Object.entries(config.provider ?? {});

  const providerRows: ProviderRow[] = [
    {
      name: "higgs",
      url: "(local)",
      format: "openai",
      stripAuth: false,
      apiKeySet: false,
      traffic: traffic.find((t) => t.name === "higgs") ?? null,
    },
    ...providerEntries.map(([name, provider]) => ({
      name,
      url: provider.url ?? "–",
      format: provider.format ?? "openai",
      stripAuth: Boolean(provider.strip_auth),
      apiKeySet: Boolean(provider.api_key),
      traffic: traffic.find((t) => t.name === name) ?? null,
    })),
  ];

  const routes = config.routes ?? [];
  const auto = config.auto_router ?? {};

  return (
    <div className="view">
      <div className="view-head">
        <h1>Providers &amp; Routing</h1>
        <button type="button" className="btn" onClick={onEditConfig}>
          Edit config
        </button>
      </div>

      {!hasConfig && !data.localAvailable && <div className="notice">Config is only readable in the desktop app.</div>}

      <div className="panel">
        <div className="vrow head providers-grid">
          <span className="meta">Provider</span>
          <span className="meta">URL</span>
          <span className="meta">Format</span>
          <span className="meta">Auth</span>
          <span className="meta" style={{ textAlign: "right" }}>
            Requests
          </span>
          <span className="meta" style={{ textAlign: "right" }}>
            P50
          </span>
          <span className="meta" style={{ textAlign: "right" }}>
            TTFT p50
          </span>
          <span className="meta" style={{ textAlign: "right" }}>
            Tok/s
          </span>
        </div>
        {providerRows.map((row) => (
          <div className="vrow providers-grid" key={row.name}>
            <span style={{ fontWeight: 700 }}>{row.name}</span>
            <span className="mono muted">{row.url}</span>
            <span>{row.format}</span>
            <span className="muted">{row.stripAuth ? "stripped" : row.apiKeySet ? "key set" : "none"}</span>
            <span className="num" style={{ textAlign: "right" }}>
              {row.traffic ? formatNumber(row.traffic.requests) : "–"}
            </span>
            <span className="num" style={{ textAlign: "right" }}>
              {row.traffic ? formatMs(row.traffic.p50_ms) : "–"}
            </span>
            <span className="num" style={{ textAlign: "right" }}>
              {row.traffic?.ttft_p50_ms != null ? formatMs(row.traffic.ttft_p50_ms) : "–"}
            </span>
            <span className="num" style={{ textAlign: "right" }}>
              {row.traffic?.tokens_per_second != null ? formatRate(row.traffic.tokens_per_second) : "–"}
            </span>
          </div>
        ))}
      </div>

      <h2 className="section-label">Routes</h2>
      <div className="panel">
        <div className="vrow head routes-grid">
          <span className="meta">Pattern</span>
          <span className="meta">Provider</span>
          <span className="meta">Model rewrite</span>
          <span className="meta">Name</span>
          <span className="meta">Description</span>
        </div>
        {routes.length === 0 && <div className="config-list-empty">No routes configured</div>}
        {routes.map((route, index) => (
          <div className="vrow routes-grid" key={`${index}-${route.pattern ?? ""}`}>
            <span className="mono">{route.pattern ?? "–"}</span>
            <span>{route.provider}</span>
            <span className="mono muted">{route.model ?? "–"}</span>
            <span>{route.name ?? "–"}</span>
            <span className="muted">{route.description ?? "–"}</span>
          </div>
        ))}
        <p className="routing-note">Exact local model names beat regex routes.</p>
      </div>

      <div className="grid-2">
        <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 8 }}>
          <span className="meta">Default route</span>
          <span className="label">
            Provider: <span className="mono">{config.default?.provider ?? "higgs"}</span>
          </span>
        </div>
        <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 8 }}>
          <span className="meta">Auto router</span>
          <span className="label">Enabled: {auto.enabled ? "yes" : "no"}</span>
          <span className="label">Force: {auto.force ? "yes" : "no"}</span>
          <span className="label">Model: {auto.model ?? "–"}</span>
          <span className="label">Timeout: {auto.timeout_ms ? formatMs(auto.timeout_ms) : "–"}</span>
        </div>
      </div>
    </div>
  );
}
