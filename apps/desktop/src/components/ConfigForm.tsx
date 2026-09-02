import { useState } from "react";
import {
  CONFIG_DEFAULTS,
  MLX_PROFILES,
  PROVIDER_FORMATS,
  displayModelName,
  providerNames,
  type AutoRouterSection,
  type HiggsConfig,
  type LocalSection,
  type ModelEntry,
  type ProviderEntry,
  type RouteEntry,
  type ServerSection,
} from "../lib/config";

export type ConfigSection = "server" | "local" | "models" | "providers" | "routes" | "metrics";

export const CONFIG_SECTIONS: Array<{ id: ConfigSection; label: string }> = [
  { id: "server", label: "Server" },
  { id: "local", label: "Local defaults" },
  { id: "models", label: "Models" },
  { id: "providers", label: "Providers" },
  { id: "routes", label: "Routes" },
  { id: "metrics", label: "Metrics" },
];

interface ConfigFormProps {
  draft: HiggsConfig;
  onChange: (next: HiggsConfig) => void;
  section: ConfigSection;
}

function toNumber(value: string): number | undefined {
  if (value.trim() === "") return undefined;
  const parsed = Number(value);
  return Number.isNaN(parsed) ? undefined : parsed;
}

function NumberField({ value, onChange, placeholder }: { value: number | undefined; onChange: (value: number | undefined) => void; placeholder?: string }) {
  return <input type="number" value={value ?? ""} placeholder={placeholder} onChange={(event) => onChange(toNumber(event.target.value))} />;
}

function TextField({ value, onChange, placeholder }: { value: string | undefined; onChange: (value: string | undefined) => void; placeholder?: string }) {
  return (
    <input
      type="text"
      value={value ?? ""}
      placeholder={placeholder}
      onChange={(event) => onChange(event.target.value === "" ? undefined : event.target.value)}
    />
  );
}

function PasswordField({ value, onChange }: { value: string | undefined; onChange: (value: string | undefined) => void }) {
  const [show, setShow] = useState(false);
  return (
    <div className="config-password-row">
      <input
        type={show ? "text" : "password"}
        value={value ?? ""}
        onChange={(event) => onChange(event.target.value === "" ? undefined : event.target.value)}
      />
      <button type="button" className="btn small ghost" onClick={() => setShow((current) => !current)}>
        {show ? "Hide" : "Show"}
      </button>
    </div>
  );
}

function Checkbox({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <label className="field inline">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      {label}
    </label>
  );
}

function ReorderButtons({ index, count, onMove, onRemove }: { index: number; count: number; onMove: (delta: -1 | 1) => void; onRemove: () => void }) {
  return (
    <div className="config-list-item-header" style={{ marginLeft: "auto" }}>
      <button type="button" className="btn small ghost" disabled={index === 0} onClick={() => onMove(-1)} title="Move up">
        ↑
      </button>
      <button type="button" className="btn small ghost" disabled={index === count - 1} onClick={() => onMove(1)} title="Move down">
        ↓
      </button>
      <button type="button" className="btn small danger" onClick={onRemove}>
        Remove
      </button>
    </div>
  );
}

export function ConfigForm({ draft, onChange, section }: ConfigFormProps) {
  const server = draft.server ?? {};
  const setServer = (patch: Partial<ServerSection>) => onChange({ ...draft, server: { ...server, ...patch } });

  const local = draft.local ?? {};
  const setLocal = (patch: Partial<LocalSection>) => onChange({ ...draft, local: { ...local, ...patch } });

  const models = draft.models ?? [];
  const setModels = (next: ModelEntry[]) => onChange({ ...draft, models: next });
  const updateModel = (index: number, patch: Partial<ModelEntry>) => setModels(models.map((model, i) => (i === index ? { ...model, ...patch } : model)));
  const addModel = () => setModels([...models, { path: "" }]);
  const removeModel = (index: number) => setModels(models.filter((_, i) => i !== index));
  const moveModel = (index: number, delta: -1 | 1) => {
    const target = index + delta;
    if (target < 0 || target >= models.length) return;
    const next = [...models];
    [next[index], next[target]] = [next[target], next[index]];
    setModels(next);
  };

  const providerEntries = Object.entries(draft.provider ?? {});
  const setProviders = (entries: Array<[string, ProviderEntry]>) => {
    const map: Record<string, ProviderEntry> = {};
    for (const [name, entry] of entries) map[name] = entry;
    onChange({ ...draft, provider: entries.length > 0 ? map : undefined });
  };
  const updateProviderName = (index: number, name: string) => {
    const next = [...providerEntries];
    next[index] = [name, next[index][1]];
    setProviders(next);
  };
  const updateProviderEntry = (index: number, patch: Partial<ProviderEntry>) => {
    const next = [...providerEntries];
    next[index] = [next[index][0], { ...next[index][1], ...patch }];
    setProviders(next);
  };
  const addProvider = () => setProviders([...providerEntries, [`provider${providerEntries.length + 1}`, {}]]);
  const removeProvider = (index: number) => setProviders(providerEntries.filter((_, i) => i !== index));

  const routes = draft.routes ?? [];
  const setRoutes = (next: RouteEntry[]) => onChange({ ...draft, routes: next });
  const updateRoute = (index: number, patch: Partial<RouteEntry>) => setRoutes(routes.map((route, i) => (i === index ? { ...route, ...patch } : route)));
  const addRoute = () => setRoutes([...routes, { provider: providerNames(draft)[0] ?? "higgs" }]);
  const removeRoute = (index: number) => setRoutes(routes.filter((_, i) => i !== index));
  const moveRoute = (index: number, delta: -1 | 1) => {
    const target = index + delta;
    if (target < 0 || target >= routes.length) return;
    const next = [...routes];
    [next[index], next[target]] = [next[target], next[index]];
    setRoutes(next);
  };

  const autoRouter = draft.auto_router ?? {};
  const setAutoRouter = (patch: Partial<AutoRouterSection>) => onChange({ ...draft, auto_router: { ...autoRouter, ...patch } });

  const retention = draft.retention ?? {};
  const setRetention = (patch: Partial<{ enabled?: boolean; minutes?: number }>) => onChange({ ...draft, retention: { ...retention, ...patch } });

  const metrics = draft.logging?.metrics ?? {};
  const setMetrics = (patch: Partial<{ enabled?: boolean; path?: string; max_size_mb?: number; max_files?: number }>) =>
    onChange({ ...draft, logging: { ...draft.logging, metrics: { ...metrics, ...patch } } });

  const providers = providerNames(draft);
  const modelNames = models.map((model) => displayModelName(model));

  if (section === "server") {
    return (
      <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
        <span className="meta">Server</span>
        <div className="form-row">
          <div className="field">
            <label>Host</label>
            <TextField value={server.host} onChange={(v) => setServer({ host: v })} placeholder={CONFIG_DEFAULTS.server.host} />
          </div>
          <div className="field">
            <label>Port</label>
            <NumberField value={server.port} onChange={(v) => setServer({ port: v })} placeholder={String(CONFIG_DEFAULTS.server.port)} />
          </div>
        </div>
        <div className="field">
          <label>API key</label>
          <PasswordField value={server.api_key} onChange={(v) => setServer({ api_key: v })} />
          <span className="field-hint">Required if host is 0.0.0.0.</span>
        </div>
        <div className="form-row">
          <div className="field">
            <label>Max tokens</label>
            <NumberField value={server.max_tokens} onChange={(v) => setServer({ max_tokens: v })} placeholder={String(CONFIG_DEFAULTS.server.max_tokens)} />
          </div>
          <div className="field">
            <label>Timeout (s)</label>
            <NumberField value={server.timeout} onChange={(v) => setServer({ timeout: v })} placeholder={String(CONFIG_DEFAULTS.server.timeout)} />
          </div>
          <div className="field">
            <label>Max body size (bytes)</label>
            <NumberField
              value={server.max_body_size}
              onChange={(v) => setServer({ max_body_size: v })}
              placeholder={String(CONFIG_DEFAULTS.server.max_body_size)}
            />
          </div>
          <div className="field">
            <label>Rate limit (req/min)</label>
            <NumberField value={server.rate_limit} onChange={(v) => setServer({ rate_limit: v })} placeholder={String(CONFIG_DEFAULTS.server.rate_limit)} />
          </div>
        </div>
        <div className="field">
          <label>CORS origins</label>
          <input
            type="text"
            value={(server.cors_origins ?? []).join(", ")}
            placeholder="https://app.example.com, *"
            onChange={(event) => {
              const list = event.target.value
                .split(",")
                .map((item) => item.trim())
                .filter(Boolean);
              setServer({ cors_origins: list.length > 0 ? list : undefined });
            }}
          />
          <span className="field-hint">Comma-separated. Unset means no CORS headers; "*" allows any origin.</span>
        </div>
      </div>
    );
  }

  if (section === "local") {
    return (
      <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
        <span className="meta">Local defaults</span>
        <div className="form-row">
          <div className="field">
            <label>MLX profile</label>
            <select value={local.mlx_profile ?? CONFIG_DEFAULTS.local.mlx_profile} onChange={(event) => setLocal({ mlx_profile: event.target.value })}>
              {MLX_PROFILES.map((profile) => (
                <option key={profile} value={profile}>
                  {profile}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>&nbsp;</label>
            <Checkbox
              label="Raise wired memory limit"
              checked={local.raise_wired_limit ?? CONFIG_DEFAULTS.local.raise_wired_limit}
              onChange={(v) => setLocal({ raise_wired_limit: v })}
            />
          </div>
        </div>
      </div>
    );
  }

  if (section === "models") {
    return (
      <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
        <div className="config-section-title">
          <span className="meta">Models</span>
          <button type="button" className="btn small" onClick={addModel}>
            Add model
          </button>
        </div>
        {models.length === 0 && <div className="config-list-empty">No local models configured.</div>}
        {models.map((model, index) => (
          <div className="config-list-item" key={index}>
            <div className="config-list-item-header">
              <div className="field" style={{ flex: 2 }}>
                <label>Path</label>
                <TextField value={model.path} onChange={(v) => updateModel(index, { path: v ?? "" })} placeholder="mlx-community/…" />
              </div>
              <div className="field" style={{ flex: 1 }}>
                <label>Name</label>
                <TextField value={model.name} onChange={(v) => updateModel(index, { name: v })} placeholder="alias" />
              </div>
              <div className="field" style={{ flex: 1 }}>
                <label>MLX profile</label>
                <select value={model.mlx_profile ?? ""} onChange={(event) => updateModel(index, { mlx_profile: event.target.value || undefined })}>
                  <option value="">inherit</option>
                  {MLX_PROFILES.map((profile) => (
                    <option key={profile} value={profile}>
                      {profile}
                    </option>
                  ))}
                </select>
              </div>
              <Checkbox label="Batch" checked={model.batch ?? false} onChange={(v) => updateModel(index, { batch: v || undefined })} />
              <ReorderButtons index={index} count={models.length} onMove={(delta) => moveModel(index, delta)} onRemove={() => removeModel(index)} />
            </div>
            <details className="config-disclosure">
              <summary>Advanced (KV cache, prefill)</summary>
              <div className="form-row">
                <div className="field">
                  <label>Prefill yield tokens</label>
                  <NumberField value={model.prefill_yield_tokens} onChange={(v) => updateModel(index, { prefill_yield_tokens: v })} placeholder="0" />
                </div>
                <div className="field">
                  <label>KV cache</label>
                  <select value={model.kv_cache ?? ""} onChange={(event) => updateModel(index, { kv_cache: event.target.value || undefined })}>
                    <option value="">off</option>
                    <option value="turboquant">turboquant</option>
                  </select>
                </div>
              </div>
              <div className="form-row">
                <div className="field">
                  <label>KV bits</label>
                  <NumberField value={model.kv_bits} onChange={(v) => updateModel(index, { kv_bits: v })} placeholder="3" />
                </div>
                <div className="field">
                  <label>KV key bits</label>
                  <NumberField value={model.kv_key_bits} onChange={(v) => updateModel(index, { kv_key_bits: v })} />
                </div>
                <div className="field">
                  <label>KV value bits</label>
                  <NumberField value={model.kv_value_bits} onChange={(v) => updateModel(index, { kv_value_bits: v })} />
                </div>
                <div className="field">
                  <label>Adaptive dense layers</label>
                  <NumberField value={model.kv_adaptive_dense_layers} onChange={(v) => updateModel(index, { kv_adaptive_dense_layers: v })} placeholder="0" />
                </div>
              </div>
              <div className="form-row">
                <div className="field">
                  <label>KV seed</label>
                  <NumberField value={model.kv_seed} onChange={(v) => updateModel(index, { kv_seed: v })} placeholder="0" />
                </div>
                <div className="field">
                  <label>KV disk dir</label>
                  <TextField value={model.kv_disk_dir} onChange={(v) => updateModel(index, { kv_disk_dir: v })} />
                </div>
                <div className="field">
                  <label>KV disk space (MB)</label>
                  <NumberField value={model.kv_disk_space_mb} onChange={(v) => updateModel(index, { kv_disk_space_mb: v })} />
                </div>
              </div>
              <div className="form-row">
                <Checkbox
                  label="KV norm correction"
                  checked={model.kv_norm_correction ?? true}
                  onChange={(v) => updateModel(index, { kv_norm_correction: v })}
                />
                <Checkbox label="MLA latent cache" checked={model.mla_latent_cache ?? false} onChange={(v) => updateModel(index, { mla_latent_cache: v })} />
              </div>
            </details>
          </div>
        ))}
      </div>
    );
  }

  if (section === "providers") {
    return (
      <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
        <div className="config-section-title">
          <span className="meta">Providers</span>
          <button type="button" className="btn small" onClick={addProvider}>
            Add provider
          </button>
        </div>
        {providerEntries.length === 0 && <div className="config-list-empty">No remote providers configured.</div>}
        {providerEntries.map(([name, entry], index) => (
          <div className="config-list-item" key={index}>
            <div className="config-list-item-header">
              <div className="field config-key-input">
                <label>Name</label>
                <input type="text" value={name} onChange={(event) => updateProviderName(index, event.target.value)} />
              </div>
              <div className="field" style={{ flex: 2 }}>
                <label>URL</label>
                <TextField value={entry.url} onChange={(v) => updateProviderEntry(index, { url: v })} placeholder="https://api.example.com" />
              </div>
              <div className="field">
                <label>Format</label>
                <select value={entry.format ?? "openai"} onChange={(event) => updateProviderEntry(index, { format: event.target.value })}>
                  {PROVIDER_FORMATS.map((format) => (
                    <option key={format} value={format}>
                      {format}
                    </option>
                  ))}
                </select>
              </div>
              <ReorderButtons index={index} count={providerEntries.length} onMove={() => {}} onRemove={() => removeProvider(index)} />
            </div>
            <div className="form-row">
              <div className="field" style={{ flex: 2 }}>
                <label>API key</label>
                <PasswordField value={entry.api_key} onChange={(v) => updateProviderEntry(index, { api_key: v })} />
              </div>
              <Checkbox label="Strip auth header" checked={entry.strip_auth ?? false} onChange={(v) => updateProviderEntry(index, { strip_auth: v })} />
              <Checkbox
                label="Stub count_tokens"
                checked={entry.stub_count_tokens ?? false}
                onChange={(v) => updateProviderEntry(index, { stub_count_tokens: v })}
              />
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (section === "routes") {
    return (
      <>
        <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
          <div className="config-section-title">
            <span className="meta">Routes</span>
            <button type="button" className="btn small" onClick={addRoute}>
              Add route
            </button>
          </div>
          {routes.length === 0 && <div className="config-list-empty">No routes configured (falls through to the default provider).</div>}
          {routes.map((route, index) => (
            <div className="config-list-item" key={index}>
              <div className="config-list-item-header">
                <div className="field" style={{ flex: 2 }}>
                  <label>Pattern</label>
                  <TextField value={route.pattern} onChange={(v) => updateRoute(index, { pattern: v })} placeholder="claude-.*" />
                </div>
                <div className="field">
                  <label>Provider</label>
                  <select value={route.provider} onChange={(event) => updateRoute(index, { provider: event.target.value })}>
                    {providers.map((provider) => (
                      <option key={provider} value={provider}>
                        {provider}
                      </option>
                    ))}
                  </select>
                </div>
                <ReorderButtons index={index} count={routes.length} onMove={(delta) => moveRoute(index, delta)} onRemove={() => removeRoute(index)} />
              </div>
              <div className="form-row">
                <div className="field">
                  <label>Model rewrite</label>
                  <TextField value={route.model} onChange={(v) => updateRoute(index, { model: v })} />
                </div>
                <div className="field">
                  <label>Name</label>
                  <TextField value={route.name} onChange={(v) => updateRoute(index, { name: v })} />
                </div>
                <div className="field" style={{ flex: 2 }}>
                  <label>Description</label>
                  <TextField value={route.description} onChange={(v) => updateRoute(index, { description: v })} />
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="grid-2">
          <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
            <span className="meta">Default provider</span>
            <select value={draft.default?.provider ?? "higgs"} onChange={(event) => onChange({ ...draft, default: { provider: event.target.value } })}>
              {providers.map((provider) => (
                <option key={provider} value={provider}>
                  {provider}
                </option>
              ))}
            </select>
          </div>

          <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 8 }}>
            <span className="meta">Auto router</span>
            <Checkbox
              label="Enabled"
              checked={autoRouter.enabled ?? CONFIG_DEFAULTS.auto_router.enabled}
              onChange={(v) => setAutoRouter({ enabled: v })}
            />
            <Checkbox label="Force" checked={autoRouter.force ?? CONFIG_DEFAULTS.auto_router.force} onChange={(v) => setAutoRouter({ force: v })} />
            <div className="field">
              <label>Classifier model</label>
              <select value={autoRouter.model ?? ""} onChange={(event) => setAutoRouter({ model: event.target.value || undefined })}>
                <option value="">none</option>
                {modelNames.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label>Timeout (ms)</label>
              <NumberField
                value={autoRouter.timeout_ms}
                onChange={(v) => setAutoRouter({ timeout_ms: v })}
                placeholder={String(CONFIG_DEFAULTS.auto_router.timeout_ms)}
              />
            </div>
          </div>
        </div>
      </>
    );
  }

  return (
    <div className="grid-2">
      <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 8 }}>
        <span className="meta">Retention</span>
        <Checkbox label="Enabled" checked={retention.enabled ?? CONFIG_DEFAULTS.retention.enabled} onChange={(v) => setRetention({ enabled: v })} />
        <div className="field">
          <label>Minutes</label>
          <NumberField value={retention.minutes} onChange={(v) => setRetention({ minutes: v })} placeholder={String(CONFIG_DEFAULTS.retention.minutes)} />
        </div>
      </div>

      <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 8 }}>
        <span className="meta">Metrics logging</span>
        <Checkbox label="Enabled" checked={metrics.enabled ?? CONFIG_DEFAULTS.metrics.enabled} onChange={(v) => setMetrics({ enabled: v })} />
        <div className="field">
          <label>Path</label>
          <TextField value={metrics.path} onChange={(v) => setMetrics({ path: v })} placeholder="~/.config/higgs/logs/metrics.jsonl" />
        </div>
        <div className="form-row">
          <div className="field">
            <label>Max size (MB)</label>
            <NumberField value={metrics.max_size_mb} onChange={(v) => setMetrics({ max_size_mb: v })} placeholder={String(CONFIG_DEFAULTS.metrics.max_size_mb)} />
          </div>
          <div className="field">
            <label>Max files</label>
            <NumberField value={metrics.max_files} onChange={(v) => setMetrics({ max_files: v })} placeholder={String(CONFIG_DEFAULTS.metrics.max_files)} />
          </div>
        </div>
      </div>
    </div>
  );
}

function omitEmpty(source: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(source)) {
    if (value === undefined || value === null) continue;
    if (typeof value === "string" && value === "") continue;
    if (Array.isArray(value) && value.length === 0) continue;
    result[key] = value;
  }
  return result;
}

/** Strips empty strings, null numbers, and empty collections before writing structured config. */
export function sanitizeConfig(config: HiggsConfig): HiggsConfig {
  const sanitized: HiggsConfig = {};

  if (config.server) {
    const server = omitEmpty(config.server as unknown as Record<string, unknown>);
    if (Object.keys(server).length > 0) sanitized.server = server as ServerSection;
  }
  if (config.local) {
    const local = omitEmpty(config.local as unknown as Record<string, unknown>);
    if (Object.keys(local).length > 0) sanitized.local = local as LocalSection;
  }
  if (config.models && config.models.length > 0) {
    sanitized.models = config.models.map((model) => omitEmpty(model as unknown as Record<string, unknown>) as unknown as ModelEntry);
  }
  if (config.provider && Object.keys(config.provider).length > 0) {
    const provider: Record<string, ProviderEntry> = {};
    for (const [name, entry] of Object.entries(config.provider)) {
      if (!name) continue;
      provider[name] = omitEmpty(entry as unknown as Record<string, unknown>) as unknown as ProviderEntry;
    }
    if (Object.keys(provider).length > 0) sanitized.provider = provider;
  }
  if (config.routes && config.routes.length > 0) {
    sanitized.routes = config.routes.map((route) => omitEmpty(route as unknown as Record<string, unknown>) as unknown as RouteEntry);
  }
  if (config.default?.provider) {
    sanitized.default = { provider: config.default.provider };
  }
  if (config.auto_router) {
    const autoRouter = omitEmpty(config.auto_router as unknown as Record<string, unknown>);
    if (Object.keys(autoRouter).length > 0) sanitized.auto_router = autoRouter as AutoRouterSection;
  }
  if (config.retention) {
    const retention = omitEmpty(config.retention as unknown as Record<string, unknown>);
    if (Object.keys(retention).length > 0) sanitized.retention = retention as { enabled?: boolean; minutes?: number };
  }
  if (config.logging?.metrics) {
    const metrics = omitEmpty(config.logging.metrics as unknown as Record<string, unknown>);
    if (Object.keys(metrics).length > 0) sanitized.logging = { metrics };
  }

  return sanitized;
}
