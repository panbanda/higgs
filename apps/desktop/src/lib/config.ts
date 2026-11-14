/**
 * Typed mirror of ~/.config/higgs/config.toml (docs/configuration.md).
 *
 * Files written for earlier versions are not read here: the app asks the
 * `higgs` binary to migrate them (`higgs config migrate`), so the mapping from
 * old keys to new ones lives in one place, the server.
 */

export interface ServerSection {
  host?: string;
  port?: number;
  api_key?: string;
  request_timeout?: string;
  max_body_size?: string | number;
  max_tokens?: number;
  rate_limit?: number;
  cors_origins?: string[];
}

export interface MemorySection {
  guard?: string;
  limit?: string | number;
  max_loaded_models?: number;
  raise_wired_limit?: boolean;
}

export interface KvCacheSection {
  mode?: string;
  bits?: number;
  key_bits?: number;
  value_bits?: number;
  norm_correction?: boolean;
  adaptive_dense_layers?: number;
  seed?: number;
  mla_latent?: boolean;
}

export interface PrefixCacheSection {
  disk_dir?: string;
  disk_budget?: string | number;
}

/** Settings valid both in `[defaults]` and in a `[[models]]` entry. */
export interface ModelSettings {
  load?: string;
  idle_ttl?: string;
  batch?: boolean | "auto";
  profile?: string;
  thinking_budget?: number;
  pinned?: boolean;
  kv_cache?: KvCacheSection;
  prefix_cache?: PrefixCacheSection;
  prefill?: { yield_tokens?: number };
}

export interface ModelEntry extends ModelSettings {
  path: string;
  name?: string;
  aliases?: string[];
  [key: string]: unknown;
}

export interface ProviderEntry {
  url?: string;
  format?: string;
  api_key?: string;
  strip_auth?: boolean;
  stub_count_tokens?: boolean;
  [key: string]: unknown;
}

export interface RouteEntry {
  match?: string;
  provider: string;
  model?: string;
  name?: string;
  description?: string;
  [key: string]: unknown;
}

export interface AutoRouterSection {
  enabled?: boolean;
  force?: boolean;
  model?: string;
  timeout?: string;
}

export interface RoutingSection {
  default?: string;
  auto?: AutoRouterSection;
}

export interface MetricsLogSection {
  enabled?: boolean;
  path?: string;
  max_size?: string | number;
  max_files?: number;
}

export interface ObservabilitySection {
  retention?: string | boolean;
  metrics_log?: MetricsLogSection;
}

export interface ExperimentalSection {
  mtp?: string;
  mtp_draft_tokens?: number;
  mtp_adaptive_draft?: boolean;
  mtp_prompt_lookup?: boolean;
  prompt_lookup?: boolean;
  clear_cache_after_prefill?: boolean;
  chunked_prefill?: { threshold?: number; chunk?: number };
}

export interface HiggsConfig {
  include?: string[];
  server?: ServerSection;
  memory?: MemorySection;
  defaults?: ModelSettings;
  models?: ModelEntry[];
  providers?: Record<string, ProviderEntry>;
  routes?: RouteEntry[];
  routing?: RoutingSection;
  observability?: ObservabilitySection;
  experimental?: ExperimentalSection;
  [key: string]: unknown;
}

export const CONFIG_DEFAULTS = {
  server: {
    host: "127.0.0.1",
    port: 8000,
    request_timeout: "5m",
    max_tokens: 32768,
    max_body_size: "10MB",
    rate_limit: 0,
  },
  memory: { guard: "off", raise_wired_limit: false },
  defaults: { load: "startup", profile: "auto", batch: "auto" },
  observability: { retention: "60m", metrics: { enabled: true, max_size: "50MB", max_files: 5 } },
  routing: { default: "higgs", auto: { enabled: false, force: false, timeout: "2s" } },
} as const;

export const MLX_PROFILES = ["auto", "latency", "balanced", "throughput"] as const;
export const MEMORY_GUARDS = ["off", "safe", "balanced", "aggressive"] as const;
export const LOAD_MODES = ["startup", "on_demand"] as const;
export const MTP_MODES = ["auto", "on", "off"] as const;
export const PROVIDER_FORMATS = ["openai", "anthropic"] as const;

/** Top-level keys and per-model keys that only files written for earlier versions use. */
const LEGACY_TOP_LEVEL = ["local", "provider", "default", "auto_router", "logging", "retention"] as const;
const LEGACY_MODEL_KEYS = [
  "mlx_profile",
  "idle_ttl_seconds",
  "kv_bits",
  "kv_key_bits",
  "kv_value_bits",
  "kv_norm_correction",
  "kv_adaptive_dense_layers",
  "kv_seed",
  "mla_latent_cache",
  "kv_disk_dir",
  "kv_disk_space_mb",
  "prefill_yield_tokens",
] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** True when a parsed file uses keys the server has deprecated. */
export function usesDeprecatedKeys(parsed: unknown): boolean {
  if (!isRecord(parsed)) return false;
  if (LEGACY_TOP_LEVEL.some((key) => key in parsed)) return true;
  const models = parsed.models;
  if (Array.isArray(models)) {
    for (const model of models) {
      if (!isRecord(model)) continue;
      if (LEGACY_MODEL_KEYS.some((key) => key in model)) return true;
      if (typeof model.kv_cache === "string") return true;
    }
  }
  const server = parsed.server;
  if (isRecord(server) && "timeout" in server) return true;
  const routes = parsed.routes;
  if (Array.isArray(routes) && routes.some((route) => isRecord(route) && "pattern" in route)) return true;
  const mcp = parsed.mcp;
  if (isRecord(mcp)) {
    if ("max_result_bytes" in mcp) return true;
    const servers = mcp.servers;
    if (isRecord(servers) && Object.values(servers).some((entry) => isRecord(entry) && "timeout_seconds" in entry)) return true;
  }
  const web = parsed.web;
  return isRecord(web) && "timeout_seconds" in web;
}

/** Lines `higgs config migrate` prints to stderr, one per rewritten key. */
export function parseMigrationNotes(stderr: string): string[] {
  return stderr
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("migrated `"));
}

export function parseConfig(parsed: unknown): HiggsConfig {
  return isRecord(parsed) ? (parsed as HiggsConfig) : {};
}

export function displayModelName(model: ModelEntry): string {
  return model.name ?? model.path;
}

/** Path of the metrics JSONL log the daemon writes for this config and profile. */
export function metricsLogPath(config: HiggsConfig, profile: string | null, configDir: string): string {
  const configured = config.observability?.metrics_log?.path;
  if (configured) return configured;
  const file = profile ? `metrics.${profile}.jsonl` : "metrics.jsonl";
  return `${configDir}/logs/${file}`;
}

export function metricsLoggingEnabled(config: HiggsConfig): boolean {
  return config.observability?.metrics_log?.enabled ?? CONFIG_DEFAULTS.observability.metrics.enabled;
}

/** Server URL implied by the config's [server] section. */
export function serverUrl(config: HiggsConfig): string {
  const host = config.server?.host ?? CONFIG_DEFAULTS.server.host;
  const port = config.server?.port ?? CONFIG_DEFAULTS.server.port;
  const reachable = host === "0.0.0.0" ? "127.0.0.1" : host;
  return `http://${reachable}:${port}`;
}

/** Provider names that routes may reference: configured providers plus the local engine. */
export function providerNames(config: HiggsConfig): string[] {
  return ["higgs", ...Object.keys(config.providers ?? {})];
}

/** Top-level sections the config form edits; everything else passes through to the file untouched. */
const FORM_MANAGED_SECTIONS = [
  "server",
  "memory",
  "defaults",
  "models",
  "providers",
  "routes",
  "routing",
  "observability",
  "experimental",
] as const;

function isEmptyValue(value: unknown): boolean {
  if (value === undefined || value === null) return true;
  if (typeof value === "string") return value === "";
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === "object") return Object.keys(value as Record<string, unknown>).length === 0;
  return false;
}

/** Drops empty values, recursively, so the saved file only lists what was set. */
function prune(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(prune);
  }
  if (typeof value === "object" && value !== null) {
    const result: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
      const cleaned = prune(entry);
      if (!isEmptyValue(cleaned)) result[key] = cleaned;
    }
    return result;
  }
  return value;
}

/**
 * Strips empty strings, unset numbers, and empty tables from the sections the form edits.
 * Sections it does not edit (mcp, web, include, ...) are carried through unchanged, because
 * empty values there can be meaningful: an empty `env` entry overrides the inherited value.
 */
export function sanitizeConfig(config: HiggsConfig): HiggsConfig {
  const sanitized: HiggsConfig = {};
  for (const [key, value] of Object.entries(config)) {
    if (!(FORM_MANAGED_SECTIONS as readonly string[]).includes(key)) {
      sanitized[key] = value;
      continue;
    }
    if (key === "providers" && value && typeof value === "object") {
      const providers: Record<string, unknown> = {};
      for (const [name, entry] of Object.entries(value as Record<string, unknown>)) {
        if (name) providers[name] = prune(entry);
      }
      if (Object.keys(providers).length > 0) sanitized.providers = providers as HiggsConfig["providers"];
      continue;
    }
    const cleaned = prune(value);
    if (!isEmptyValue(cleaned)) sanitized[key] = cleaned;
  }
  return sanitized;
}
