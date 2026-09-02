import { useEffect, useState } from "react";
import { modelCacheInfo } from "../lib/api";
import { type GroupStats, groupBy } from "../lib/dashboard";
import { displayModelName, type ModelEntry } from "../lib/config";
import { formatBytes, formatMs, formatRate } from "../lib/format";
import type { ServerData } from "../hooks/useServerData";
import type { ModelCacheInfo, Settings } from "../lib/types";
import "./views.css";

export interface ModelsViewProps {
  settings: Settings;
  data: ServerData;
  onChatWith: (modelId: string) => void;
  onEditConfig: () => void;
}

interface ConfiguredModelRow {
  entry: ModelEntry;
  displayName: string;
  served: boolean;
  servedId: string | null;
  cache: ModelCacheInfo | null;
  traffic: GroupStats | null;
  engine: string | null;
}

type ModelState = "serving" | "cached" | "not downloaded";

const STATE_COLOR: Record<ModelState, string> = { serving: "var(--ok)", cached: "var(--text-muted)", "not downloaded": "var(--warn)" };

function modelState(row: ConfiguredModelRow): ModelState {
  if (row.served) return "serving";
  if (row.cache?.cached) return "cached";
  return "not downloaded";
}

export function ModelsView({ data, onChatWith, onEditConfig }: ModelsViewProps) {
  const configuredModels = data.config.models ?? [];
  const [cacheByPath, setCacheByPath] = useState<Record<string, ModelCacheInfo>>({});

  useEffect(() => {
    if (!data.localAvailable) return;
    let cancelled = false;
    for (const model of configuredModels) {
      modelCacheInfo(model.path)
        .then((info) => {
          if (!cancelled) setCacheByPath((current) => ({ ...current, [model.path]: info }));
        })
        .catch(() => {
          /* not fatal: leave cache status unknown for this model */
        });
    }
    return () => {
      cancelled = true;
    };
    // Re-run when the set of configured model paths changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.localAvailable, configuredModels.map((m) => m.path).join(",")]);

  const trafficByModel = groupBy(data.records, (r) => r.model);
  const trafficFallback = data.metrics?.models ?? [];

  const findTraffic = (entry: ModelEntry, displayName: string): GroupStats | null => {
    const live = trafficByModel.find((group) => group.name === displayName || group.name === entry.path);
    if (live) return live;
    return trafficFallback.find((group) => group.name === displayName || group.name === entry.path) ?? null;
  };

  const systemModels = data.system?.models ?? [];
  const findSystemModel = (entry: ModelEntry, displayName: string) =>
    systemModels.find((model) => model.name === displayName || model.name === entry.path || model.path === entry.path);

  const rows: ConfiguredModelRow[] = configuredModels.map((entry) => {
    const displayName = displayModelName(entry);
    const served = data.models.find((model) => model.id === displayName || model.id === entry.path);
    const systemModel = findSystemModel(entry, displayName);
    return {
      entry,
      displayName,
      served: Boolean(served),
      servedId: served?.id ?? null,
      cache: cacheByPath[entry.path] ?? null,
      traffic: findTraffic(entry, displayName),
      engine: systemModel?.engine ?? null,
    };
  });

  const configuredNames = new Set(configuredModels.map((entry) => displayModelName(entry)));
  const configuredPaths = new Set(configuredModels.map((entry) => entry.path));
  const remoteModels = data.models.filter((model) => !configuredNames.has(model.id) && !configuredPaths.has(model.id));

  return (
    <div className="view">
      <div className="view-head">
        <h1>Models</h1>
        <button type="button" className="btn" onClick={onEditConfig}>
          Edit config
        </button>
      </div>

      <div className="panel">
        <div className="vrow head models-grid">
          <span className="meta">Model</span>
          <span className="meta">State</span>
          <span className="meta">Engine</span>
          <span className="meta" style={{ textAlign: "right" }}>
            On disk
          </span>
          <span className="meta" style={{ textAlign: "right" }}>
            Requests
          </span>
          <span className="meta" style={{ textAlign: "right" }}>
            TTFT p50
          </span>
          <span className="meta" style={{ textAlign: "right" }}>
            Decode
          </span>
          <span className="meta" />
        </div>
        {rows.length === 0 && <div className="config-list-empty">No models configured</div>}
        {rows.map((row) => {
          const state = modelState(row);
          return (
            <div className="vrow models-grid" style={{ minHeight: 56 }} key={row.entry.path}>
              <div className="row-name">
                <span>{row.displayName}</span>
                <span className="mono muted path">{row.entry.path}</span>
              </div>
              <div className="state-cell">
                <span className="swatch" style={{ background: STATE_COLOR[state] }} />
                <span>{state}</span>
              </div>
              <span className="mono">{row.engine ?? "–"}</span>
              <span className="num" style={{ textAlign: "right" }}>
                {row.cache?.cached ? formatBytes(row.cache.size_bytes) : "–"}
              </span>
              <span className="num" style={{ textAlign: "right" }}>
                {row.traffic ? row.traffic.requests : "–"}
              </span>
              <span className="num" style={{ textAlign: "right" }}>
                {row.traffic?.ttft_p50_ms != null ? formatMs(row.traffic.ttft_p50_ms) : "–"}
              </span>
              <span className="num" style={{ textAlign: "right" }}>
                {row.traffic?.tokens_per_second != null ? formatRate(row.traffic.tokens_per_second) : "–"}
              </span>
              <div className="row-actions">
                {row.served && row.servedId && (
                  <button type="button" className="btn hot" onClick={() => onChatWith(row.servedId!)}>
                    Chat
                  </button>
                )}
                <button type="button" className="btn" onClick={onEditConfig}>
                  Config
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {remoteModels.length > 0 && (
        <>
          <h2 className="section-label">Served (not in config)</h2>
          <div className="panel">
            <div className="vrow head models-remote-grid">
              <span className="meta">Model</span>
              <span className="meta" />
            </div>
            {remoteModels.map((model) => (
              <div className="vrow models-remote-grid" key={model.id}>
                <span className="mono">{model.id}</span>
                <div className="row-actions">
                  <button type="button" className="btn hot" onClick={() => onChatWith(model.id)}>
                    Chat
                  </button>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
