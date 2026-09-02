import { useEffect, useState } from "react";
import { openUrl } from "@tauri-apps/plugin-opener";
import {
  hubCancel,
  hubDelete,
  hubDownloadStart,
  hubDownloadStatus,
  hubModel,
  hubSearch,
  inTauri,
  modelCacheInfo,
  readConfig,
  writeConfigStructured,
} from "../lib/api";
import { parseConfig } from "../lib/config";
import { formatAge, formatBytes, formatCompact } from "../lib/format";
import type { ServerData } from "../hooks/useServerData";
import type { HubDownloadStatus, HubModelDetail, HubModelSummary, ModelCacheInfo, Settings } from "../lib/types";
import "./views.css";

export interface HubViewProps {
  settings: Settings;
  data: ServerData;
}

type HubState = "downloaded" | "gated" | "not downloaded";

const STATE_COLOR: Record<HubState, string> = { downloaded: "var(--ok)", gated: "var(--warn)", "not downloaded": "var(--text-muted)" };

const TEXT_GENERATION_PIPELINE = "text-generation";

const QUANT_PATTERNS: Array<[string, string]> = [
  ["4-bit", "4-bit"],
  ["4bit", "4-bit"],
  ["8-bit", "8-bit"],
  ["8bit", "8-bit"],
  ["6-bit", "6-bit"],
  ["6bit", "6-bit"],
  ["3-bit", "3-bit"],
  ["3bit", "3-bit"],
  ["2-bit", "2-bit"],
  ["2bit", "2-bit"],
  ["bf16", "bf16"],
  ["fp16", "fp16"],
  ["fp32", "fp32"],
];

function quantFromId(id: string): string | null {
  const lower = id.toLowerCase();
  const match = QUANT_PATTERNS.find(([needle]) => lower.includes(needle));
  return match ? match[1] : null;
}

function hubState(model: HubModelSummary, cache: ModelCacheInfo | null): HubState {
  if (cache?.cached) return "downloaded";
  if (model.gated) return "gated";
  return "not downloaded";
}

function openOnHub(id: string) {
  const url = `https://huggingface.co/${id}`;
  if (inTauri) void openUrl(url);
  else window.open(url, "_blank", "noopener,noreferrer");
}

function progressPercent(status: HubDownloadStatus): number {
  if (status.total_bytes <= 0) return 0;
  return Math.min(100, Math.round((status.total_done / status.total_bytes) * 100));
}

export function HubView({ settings, data }: HubViewProps) {
  const [queryInput, setQueryInput] = useState("");
  const [query, setQuery] = useState("");
  const [author, setAuthor] = useState("mlx-community");
  const [allPipelines, setAllPipelines] = useState(false);
  const [results, setResults] = useState<HubModelSummary[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [cacheByRepo, setCacheByRepo] = useState<Record<string, ModelCacheInfo>>({});

  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<HubModelDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [addedNotice, setAddedNotice] = useState<string | null>(null);
  const [addError, setAddError] = useState<string | null>(null);

  const [downloadingRepo, setDownloadingRepo] = useState<string | null>(null);
  const [downloadStatus, setDownloadStatus] = useState<HubDownloadStatus | null>(null);

  const token = settings.hfToken || null;

  useEffect(() => {
    const timer = setTimeout(() => setQuery(queryInput), 300);
    return () => clearTimeout(timer);
  }, [queryInput]);

  useEffect(() => {
    if (!data.localAvailable) return;
    let cancelled = false;
    setSearching(true);
    setSearchError(null);
    hubSearch(query, author || null, allPipelines ? null : TEXT_GENERATION_PIPELINE, token, 30)
      .then((list) => {
        if (cancelled) return;
        setResults(list);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setResults([]);
        setSearchError(String(error));
      })
      .finally(() => {
        if (!cancelled) setSearching(false);
      });
    return () => {
      cancelled = true;
    };
  }, [data.localAvailable, query, author, allPipelines, token]);

  useEffect(() => {
    if (!data.localAvailable) return;
    let cancelled = false;
    for (const model of results) {
      modelCacheInfo(model.id)
        .then((info) => {
          if (!cancelled) setCacheByRepo((current) => ({ ...current, [model.id]: info }));
        })
        .catch(() => {
          /* not fatal: leave cache status unknown for this model */
        });
    }
    return () => {
      cancelled = true;
    };
    // Re-run only when the visible result set changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.localAvailable, results.map((r) => r.id).join(",")]);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    setDetail(null);
    setDetailError(null);
    setLoadingDetail(true);
    hubModel(selected, token)
      .then((result) => {
        if (!cancelled) setDetail(result);
      })
      .catch((error: unknown) => {
        if (!cancelled) setDetailError(String(error));
      })
      .finally(() => {
        if (!cancelled) setLoadingDetail(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selected, token]);

  useEffect(() => {
    if (!downloadingRepo) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await hubDownloadStatus(downloadingRepo);
        if (cancelled) return;
        setDownloadStatus(status);
        if (status.state !== "running") {
          const finished = downloadingRepo;
          // Refresh the cache state before clearing the job: clearing it
          // re-runs this effect's cleanup, which would drop the update.
          if (status.state === "done") {
            const info = await modelCacheInfo(finished).catch(() => null);
            if (info) setCacheByRepo((current) => ({ ...current, [finished]: info }));
          }
          setDownloadingRepo(null);
        }
      } catch (error) {
        if (cancelled) return;
        setDownloadStatus({
          state: "error",
          file: null,
          file_index: 0,
          file_count: 0,
          bytes_done: 0,
          bytes_total: 0,
          total_done: 0,
          total_bytes: 0,
          message: String(error),
          path: null,
        });
        setDownloadingRepo(null);
      }
    };
    void poll();
    const timer = setInterval(() => void poll(), 500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [downloadingRepo]);

  const selectRepo = (id: string) => {
    setSelected(id);
    setConfirmDelete(false);
    setAddedNotice(null);
    setAddError(null);
  };

  const startDownload = async () => {
    if (!selected) return;
    setDownloadStatus({
      state: "running",
      file: null,
      file_index: 0,
      file_count: 0,
      bytes_done: 0,
      bytes_total: 0,
      total_done: 0,
      total_bytes: 0,
      message: null,
      path: null,
    });
    setDownloadingRepo(selected);
    await hubDownloadStart(selected, token);
  };

  const cancelDownload = async () => {
    if (!downloadingRepo) return;
    await hubCancel(downloadingRepo);
  };

  const alreadyConfigured = (data.config.models ?? []).some((entry) => entry.path === detail?.id);

  const addToConfig = async () => {
    if (!detail || !data.configFile) return;
    setAddError(null);
    try {
      const fresh = await readConfig(data.configFile.path);
      const parsed = parseConfig(fresh.parsed);
      const models = [...(parsed.models ?? [])];
      if (!models.some((entry) => entry.path === detail.id)) {
        const shortName = detail.id.split("/").pop() ?? detail.id;
        models.push({ path: detail.id, name: shortName });
        await writeConfigStructured(fresh.path, { ...parsed, models });
        await data.reloadConfig();
      }
      setAddedNotice(`Added ${detail.id} to config`);
    } catch (error) {
      setAddError(String(error));
    }
  };

  const deleteFromCache = async () => {
    if (!detail) return;
    await hubDelete(detail.id);
    setConfirmDelete(false);
    const info = await modelCacheInfo(detail.id).catch(() => null);
    if (info) setCacheByRepo((current) => ({ ...current, [detail.id]: info }));
  };

  const selectedCache = detail ? (cacheByRepo[detail.id] ?? null) : null;
  const isDownloading = downloadingRepo === selected && downloadStatus !== null;

  return (
    <div className="view">
      <div className="view-head">
        <h1>Hub</h1>
      </div>

      <div className="view-toolbar">
        <input
          type="text"
          className="requests-search"
          placeholder="Search models…"
          value={queryInput}
          onChange={(event) => setQueryInput(event.target.value)}
        />
        <select value={author} onChange={(event) => setAuthor(event.target.value)}>
          <option value="mlx-community">mlx-community</option>
          <option value="">Any author</option>
        </select>
        <label className="field inline">
          <input type="checkbox" checked={allPipelines} onChange={(event) => setAllPipelines(event.target.checked)} />
          All pipelines
        </label>
      </div>
      <p className="routing-note">
        Higgs runs MLX checkpoints; see <span className="mono">docs/models.md</span> for supported families.
      </p>

      {!data.localAvailable && <div className="notice">Hub browsing needs the desktop app (not available in browser mode).</div>}
      {searchError && <div className="notice bad">{searchError}</div>}

      <div className="requests-layout">
        <div className="panel">
          <div className="vrow head hub-grid">
            <span className="meta">Model</span>
            <span className="meta">Quant</span>
            <span className="meta" style={{ textAlign: "right" }}>
              Downloads
            </span>
            <span className="meta" style={{ textAlign: "right" }}>
              Likes
            </span>
            <span className="meta">Updated</span>
            <span className="meta">State</span>
          </div>
          {searching && results.length === 0 && <div className="config-list-empty">Searching…</div>}
          {!searching && results.length === 0 && <div className="config-list-empty">No models found</div>}
          {results.map((model) => {
            const state = hubState(model, cacheByRepo[model.id] ?? null);
            return (
              <div
                key={model.id}
                className={`vrow clickable hub-grid ${selected === model.id ? "selected" : ""}`}
                onClick={() => selectRepo(model.id)}
              >
                <span className="mono">{model.id}</span>
                <span className="mono muted">{quantFromId(model.id) ?? "–"}</span>
                <span className="num" style={{ textAlign: "right" }}>
                  {formatCompact(model.downloads)}
                </span>
                <span className="num" style={{ textAlign: "right" }}>
                  {formatCompact(model.likes)}
                </span>
                <span className="mono muted" style={{ textAlign: "right" }}>
                  {model.last_modified ? formatAge(model.last_modified) : "–"}
                </span>
                <div className="state-cell">
                  <span className="swatch" style={{ background: STATE_COLOR[state] }} />
                  <span>{state}</span>
                </div>
              </div>
            );
          })}
        </div>

        <div className="panel detail-panel">
          {!selected && <div className="detail-empty">Select a model to see details</div>}
          {selected && loadingDetail && <div className="detail-empty">Loading…</div>}
          {selected && detailError && <div className="detail-error">{detailError}</div>}
          {selected && detail && !loadingDetail && (
            <>
              <span className="meta">{detail.id}</span>
              <dl className="detail-dl">
                <dt className="label">Size</dt>
                <dd className="num">{formatBytes(detail.total_bytes)}</dd>
                <dt className="label">Files</dt>
                <dd className="num">{detail.siblings.length}</dd>
                <dt className="label">Model type</dt>
                <dd className="mono">{detail.config_model_type ?? "–"}</dd>
                <dt className="label">Quantization</dt>
                <dd className="mono">{detail.quantization ?? "–"}</dd>
                <dt className="label">On disk</dt>
                <dd>{selectedCache?.cached ? formatBytes(selectedCache.size_bytes) : "not downloaded"}</dd>
              </dl>

              {detail.tags.length > 0 && (
                <div className="hub-tags">
                  {detail.tags.map((tag) => (
                    <span key={tag} className="pill">
                      {tag}
                    </span>
                  ))}
                </div>
              )}

              {isDownloading && downloadStatus && (
                <div className="hub-progress">
                  <div className="hub-progress-bar">
                    <div className="hub-progress-fill" style={{ width: `${progressPercent(downloadStatus)}%` }} />
                  </div>
                  <span className="stats">
                    {downloadStatus.file ?? "starting…"} ({downloadStatus.file_index}/{downloadStatus.file_count}) ·{" "}
                    {formatBytes(downloadStatus.total_done)} / {formatBytes(downloadStatus.total_bytes)}
                  </span>
                </div>
              )}
              {downloadStatus?.state === "error" && !isDownloading && <div className="notice bad">{downloadStatus.message}</div>}
              {downloadStatus?.state === "cancelled" && !isDownloading && <div className="notice">Download cancelled</div>}

              {addedNotice && <div className="notice ok">{addedNotice}</div>}
              {addError && <div className="notice bad">{addError}</div>}

              <div className="row-actions" style={{ justifyContent: "flex-start", flexWrap: "wrap" }}>
                {isDownloading ? (
                  <button type="button" className="btn stop" onClick={() => void cancelDownload()}>
                    Cancel
                  </button>
                ) : selectedCache?.cached ? (
                  <button type="button" className="btn" disabled>
                    Downloaded
                  </button>
                ) : (
                  <button type="button" className="btn primary" disabled={!data.localAvailable} onClick={() => void startDownload()}>
                    Download
                  </button>
                )}
                {selectedCache?.cached && !confirmDelete && (
                  <button
                    type="button"
                    className="btn"
                    disabled={alreadyConfigured}
                    onClick={() => void addToConfig()}
                  >
                    {alreadyConfigured ? "Already in config" : "Add to config"}
                  </button>
                )}
                {selectedCache?.cached && !confirmDelete && (
                  <button type="button" className="btn danger" onClick={() => setConfirmDelete(true)}>
                    Delete from cache
                  </button>
                )}
                {confirmDelete && (
                  <>
                    <span className="label">Delete cached files?</span>
                    <button type="button" className="btn danger" onClick={() => void deleteFromCache()}>
                      Confirm delete
                    </button>
                    <button type="button" className="btn" onClick={() => setConfirmDelete(false)}>
                      Cancel
                    </button>
                  </>
                )}
                <button type="button" className="btn ghost" onClick={() => openOnHub(detail.id)}>
                  Open on Hugging Face
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
