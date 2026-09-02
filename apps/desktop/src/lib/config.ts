/** Typed mirror of ~/.config/higgs/config.toml (docs/configuration.md). */

export interface ServerSection {
  host?: string;
  port?: number;
  max_tokens?: number;
  timeout?: number;
  max_body_size?: number;
  api_key?: string;
  rate_limit?: number;
  cors_origins?: string[];
}

export interface LocalSection {
  mlx_profile?: string;
  raise_wired_limit?: boolean;
}

export interface ModelEntry {
  path: string;
  name?: string;
  mlx_profile?: string;
  batch?: boolean;
  prefill_yield_tokens?: number;
  kv_cache?: string;
  kv_bits?: number;
  kv_key_bits?: number;
  kv_value_bits?: number;
  kv_norm_correction?: boolean;
  kv_adaptive_dense_layers?: number;
  kv_seed?: number;
  kv_disk_dir?: string;
  kv_disk_space_mb?: number;
  mla_latent_cache?: boolean;
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
  pattern?: string;
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
  timeout_ms?: number;
}

export interface HiggsConfig {
  server?: ServerSection;
  local?: LocalSection;
  models?: ModelEntry[];
  provider?: Record<string, ProviderEntry>;
  routes?: RouteEntry[];
  default?: { provider?: string };
  auto_router?: AutoRouterSection;
  retention?: { enabled?: boolean; minutes?: number };
  logging?: { metrics?: { enabled?: boolean; path?: string; max_size_mb?: number; max_files?: number } };
  [key: string]: unknown;
}

export const CONFIG_DEFAULTS = {
  server: { host: "127.0.0.1", port: 8000, timeout: 300, max_tokens: 32768, max_body_size: 10_485_760, rate_limit: 0 },
  local: { mlx_profile: "auto", raise_wired_limit: false },
  retention: { enabled: true, minutes: 60 },
  metrics: { enabled: true, max_size_mb: 50, max_files: 5 },
  auto_router: { enabled: false, force: false, timeout_ms: 2000 },
} as const;

export const MLX_PROFILES = ["auto", "throughput", "latency", "memory"] as const;
export const PROVIDER_FORMATS = ["openai", "anthropic"] as const;

export function parseConfig(parsed: unknown): HiggsConfig {
  return parsed && typeof parsed === "object" ? (parsed as HiggsConfig) : {};
}

export function displayModelName(model: ModelEntry): string {
  return model.name ?? model.path;
}

/** Path of the metrics JSONL log the daemon writes for this config and profile. */
export function metricsLogPath(config: HiggsConfig, profile: string | null, configDir: string): string {
  const configured = config.logging?.metrics?.path;
  if (configured) return configured;
  const file = profile ? `metrics.${profile}.jsonl` : "metrics.jsonl";
  return `${configDir}/logs/${file}`;
}

export function metricsLoggingEnabled(config: HiggsConfig): boolean {
  return config.logging?.metrics?.enabled ?? CONFIG_DEFAULTS.metrics.enabled;
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
  return ["higgs", ...Object.keys(config.provider ?? {})];
}
