import { invoke, Channel } from "@tauri-apps/api/core";
import type {
  CommandOutput,
  ConfigFile,
  DaemonStatus,
  HealthStatus,
  HiggsBinaryInfo,
  HubDownloadStatus,
  HubModelDetail,
  HubModelSummary,
  Metrics,
  MetricsLog,
  ModelCacheInfo,
  ModelInfo,
  ProfileList,
  Settings,
  SystemInfo,
} from "./types";

export interface Connection {
  base_url: string;
  api_key: string | null;
}

export function connectionFrom(settings: Settings): Connection {
  return { base_url: settings.baseUrl, api_key: settings.apiKey || null };
}

export type StreamEvent =
  | { type: "chunk"; data: ChatChunk }
  | { type: "done" }
  | { type: "cancelled" }
  | { type: "error"; message: string };

export interface ToolCallDelta {
  index: number;
  id?: string;
  type?: string;
  function?: { name?: string; arguments?: string };
}

export interface ChatChunk {
  id?: string;
  model?: string;
  choices?: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string | null;
      reasoning_content?: string | null;
      tool_calls?: ToolCallDelta[];
    };
    finish_reason?: string | null;
  }>;
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number } | null;
  prompt_progress?: { total: number; cache: number; processed: number; time_ms: number };
}

/**
 * Outside the Tauri shell (plain `pnpm dev` in a browser) HTTP requests go
 * straight to the server with `fetch`, which needs `server.cors_origins`
 * configured on the Higgs side. Local-machine commands (config files, the
 * metrics log, the daemon) go through the Vite dev middleware in
 * dev/local-bridge.ts, which only exists under `pnpm dev`; a production
 * browser build has no local access and rejects with `LOCAL_UNAVAILABLE`.
 */
export const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export const devBridge = !inTauri && import.meta.env.DEV;

/** True when config, log, and daemon commands can run at all. */
export const localAvailable = inTauri || devBridge;

export const LOCAL_UNAVAILABLE = "Local machine access needs the desktop app (not available in browser mode)";

async function local<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (inTauri) return invoke<T>(command, args);
  if (!devBridge) throw new Error(LOCAL_UNAVAILABLE);
  const response = await fetch(`/__local/${command}`, {
    method: "POST",
    // The bridge only accepts requests carrying this per-session token (see
    // dev/local-bridge.ts and vite.config.ts's `define`), so another
    // localhost-bound page can't call it even if it also passes the
    // loopback check.
    headers: { "Content-Type": "application/json", "X-Higgs-Bridge": __HIGGS_BRIDGE_TOKEN__ },
    body: JSON.stringify(args ?? {}),
  });
  const payload = (await response.json()) as { ok: boolean; result?: T; error?: string };
  if (!payload.ok) throw new Error(payload.error ?? `local command failed: ${command}`);
  return payload.result as T;
}

// HTTP API

export function checkHealth(connection: Connection): Promise<HealthStatus> {
  if (!inTauri) return browserHealth(connection);
  return invoke("check_health", { connection });
}

export function listModels(connection: Connection): Promise<ModelInfo[]> {
  if (!inTauri) return browserJson<{ data: ModelInfo[] }>(connection, "/v1/models").then((body) => body.data);
  return invoke("list_models", { connection });
}

export function fetchMetrics(connection: Connection): Promise<Metrics> {
  if (!inTauri) return browserJson<Metrics>(connection, "/metrics");
  return invoke("fetch_metrics", { connection });
}

export function fetchSystem(connection: Connection): Promise<SystemInfo> {
  if (!inTauri) return browserJson<SystemInfo>(connection, "/v1/system");
  return invoke("fetch_system", { connection });
}

export function streamChat(
  requestId: string,
  connection: Connection,
  body: unknown,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  if (!inTauri) return browserStream(requestId, connection, body, onEvent);
  const channel = new Channel<StreamEvent>();
  channel.onmessage = onEvent;
  return invoke("stream_chat", { requestId, connection, body, onEvent: channel });
}

export function cancelChat(requestId: string): Promise<void> {
  if (!inTauri) {
    browserAborts.get(requestId)?.abort();
    return Promise.resolve();
  }
  return invoke("cancel_chat", { requestId });
}

// Local machine (desktop app only)

export function listProfiles(): Promise<ProfileList> {
  return local("list_profiles");
}

export function readConfig(path: string): Promise<ConfigFile> {
  return local("read_config", { path });
}

export function writeConfigRaw(path: string, raw: string): Promise<void> {
  return local("write_config_raw", { path, raw });
}

/** Returns the TOML that was written. Comments in the old file are dropped. */
export function writeConfigStructured(path: string, config: unknown): Promise<string> {
  return local("write_config_structured", { path, config });
}

export function readMetricsLog(path: string, maxRecords: number, sinceOffset: number | null): Promise<MetricsLog> {
  return local("read_metrics_log", { path, maxRecords, sinceOffset });
}

export function daemonStatus(profile: string | null): Promise<DaemonStatus> {
  return local("daemon_status", { profile });
}

export function readTextTail(path: string, maxBytes: number): Promise<string> {
  return local("read_text_tail", { path, maxBytes });
}

/** Runs an allowlisted `higgs` subcommand: doctor, start, stop, config, --version. */
export function runHiggs(binary: string, args: string[]): Promise<CommandOutput> {
  return local("run_higgs", { binary: binary || null, args });
}

/**
 * Resolves the `higgs` binary the same way `runHiggs` would (an explicit
 * path from Settings, then `PATH`, then the copy bundled with the desktop
 * app) and reports where it came from and its `--version` output.
 */
export function higgsBinaryInfo(binary: string): Promise<HiggsBinaryInfo> {
  return local("higgs_binary_info", { binary: binary || null });
}

export function modelCacheInfo(path: string): Promise<ModelCacheInfo> {
  return local("model_cache_info", { path });
}

// Hugging Face hub browsing (desktop app only)

export function hubSearch(
  query: string,
  author: string | null,
  pipelineTag: string | null,
  token: string | null,
  limit: number,
): Promise<HubModelSummary[]> {
  return local("hub_search", { query, author, pipelineTag, token, limit });
}

export function hubModel(repo: string, token: string | null): Promise<HubModelDetail> {
  return local("hub_model", { repo, token });
}

export function hubDownloadStart(repo: string, token: string | null): Promise<void> {
  return local("hub_download_start", { repo, token });
}

export function hubDownloadStatus(repo: string): Promise<HubDownloadStatus> {
  return local("hub_download_status", { repo });
}

export function hubCancel(repo: string): Promise<void> {
  return local("hub_cancel", { repo });
}

export function hubDelete(repo: string): Promise<void> {
  return local("hub_delete", { repo });
}

// Secrets (desktop app only, Keychain-backed; dev bridge holds them in memory)

export function secretSet(name: string, value: string): Promise<void> {
  return local("secret_set", { name, value });
}

export function secretGet(name: string): Promise<string | null> {
  return local("secret_get", { name });
}

export function secretDelete(name: string): Promise<void> {
  return local("secret_delete", { name });
}

// Browser fallbacks

const browserAborts = new Map<string, AbortController>();

const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1"]);

/**
 * The API key must never leave the machine in the clear: it is only sent
 * when the base URL is https:, or http: to a loopback host.
 *
 * Rust native path (crates/higgs-desktop or equivalent invoke handlers) must
 * apply the same rule before attaching the Authorization header.
 */
function apiKeyTransportAllowed(baseUrl: string): boolean {
  try {
    const url = new URL(baseUrl);
    if (url.protocol === "https:") return true;
    if (url.protocol === "http:") return LOOPBACK_HOSTS.has(url.hostname);
    return false;
  } catch {
    return false;
  }
}

function browserHeaders(connection: Connection): HeadersInit {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (connection.api_key && apiKeyTransportAllowed(connection.base_url)) {
    headers.Authorization = `Bearer ${connection.api_key}`;
  }
  return headers;
}

function browserUrl(connection: Connection, path: string): string {
  return `${connection.base_url.replace(/\/+$/, "")}${path}`;
}

const API_KEY_TRANSPORT_ERROR = "API key is only sent over HTTPS or loopback";

async function browserHealth(connection: Connection): Promise<HealthStatus> {
  if (connection.api_key && !apiKeyTransportAllowed(connection.base_url)) {
    return { ok: false, detail: API_KEY_TRANSPORT_ERROR };
  }
  try {
    const response = await fetch(browserUrl(connection, "/health"), { signal: AbortSignal.timeout(3000) });
    return response.ok ? { ok: true, detail: "ok" } : { ok: false, detail: `HTTP ${response.status}` };
  } catch (error) {
    return { ok: false, detail: (error as Error).message };
  }
}

async function browserError(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const parsed = JSON.parse(text) as { error?: { message?: string } | string };
    const message = typeof parsed.error === "string" ? parsed.error : parsed.error?.message;
    return `HTTP ${response.status}: ${message ?? text}`;
  } catch {
    return `HTTP ${response.status}: ${text}`;
  }
}

async function browserJson<T>(connection: Connection, path: string): Promise<T> {
  if (connection.api_key && !apiKeyTransportAllowed(connection.base_url)) {
    throw new Error(API_KEY_TRANSPORT_ERROR);
  }
  const response = await fetch(browserUrl(connection, path), { headers: browserHeaders(connection) });
  if (!response.ok) throw new Error(await browserError(response));
  return (await response.json()) as T;
}

async function browserStream(
  requestId: string,
  connection: Connection,
  body: unknown,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  if (connection.api_key && !apiKeyTransportAllowed(connection.base_url)) {
    onEvent({ type: "error", message: API_KEY_TRANSPORT_ERROR });
    return;
  }
  const controller = new AbortController();
  browserAborts.set(requestId, controller);
  try {
    const response = await fetch(browserUrl(connection, "/v1/chat/completions"), {
      method: "POST",
      headers: browserHeaders(connection),
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) {
      onEvent({ type: "error", message: await browserError(response) });
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf("\n\n");
        const payload = frame
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (!payload) continue;
        if (payload.trim() === "[DONE]") {
          onEvent({ type: "done" });
          return;
        }
        onEvent({ type: "chunk", data: JSON.parse(payload) as ChatChunk });
      }
    }
    onEvent({ type: "done" });
  } catch (error) {
    if (controller.signal.aborted) onEvent({ type: "cancelled" });
    else onEvent({ type: "error", message: (error as Error).message });
  } finally {
    browserAborts.delete(requestId);
  }
}
