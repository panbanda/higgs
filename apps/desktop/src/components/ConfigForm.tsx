import { useState } from "react";
import {
  CONFIG_DEFAULTS,
  LOAD_MODES,
  MEMORY_GUARDS,
  MLX_PROFILES,
  MTP_MODES,
  PROVIDER_FORMATS,
  displayModelName,
  providerNames,
  type AutoRouterSection,
  type ExperimentalSection,
  type HiggsConfig,
  type MemorySection,
  type MetricsLogSection,
  type ModelEntry,
  type ModelSettings,
  type ProviderEntry,
  type RouteEntry,
  type ServerSection,
} from "../lib/config";

export type ConfigSection = "server" | "memory" | "defaults" | "models" | "providers" | "routes" | "observability" | "experimental";

export const CONFIG_SECTIONS: Array<{ id: ConfigSection; label: string }> = [
  { id: "server", label: "Server" },
  { id: "memory", label: "Memory" },
  { id: "defaults", label: "Model defaults" },
  { id: "models", label: "Models" },
  { id: "providers", label: "Providers" },
  { id: "routes", label: "Routes" },
  { id: "observability", label: "Observability" },
  { id: "experimental", label: "Experimental" },
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

/** A size or duration written as a string such as "5m" or "4GB"; a bare number is kept as typed by the user. */
function UnitField({ value, onChange, placeholder }: { value: string | number | undefined; onChange: (value: string | undefined) => void; placeholder?: string }) {
  return <TextField value={value === undefined ? undefined : String(value)} onChange={onChange} placeholder={placeholder} />;
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

function Select({ value, options, onChange, inherit }: { value: string | undefined; options: readonly string[]; onChange: (value: string | undefined) => void; inherit?: string }) {
  return (
    <select value={value ?? ""} onChange={(event) => onChange(event.target.value || undefined)}>
      <option value="">{inherit ?? "default"}</option>
      {options.map((option) => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
  );
}

/** Keys of `[defaults]` and of every `[[models]]` entry that mean the same thing in both. */
function ModelSettingsEditor({ settings, onChange, inherit }: { settings: ModelSettings; onChange: (patch: Partial<ModelSettings>) => void; inherit: string }) {
  const kv = settings.kv_cache ?? {};
  const prefix = settings.prefix_cache ?? {};
  const setKv = (patch: NonNullable<ModelSettings["kv_cache"]>) => onChange({ kv_cache: { ...kv, ...patch } });
  const setPrefix = (patch: NonNullable<ModelSettings["prefix_cache"]>) => onChange({ prefix_cache: { ...prefix, ...patch } });
  return (
    <>
      <div className="form-row">
        <div className="field">
          <label>Load</label>
          <Select value={settings.load} options={LOAD_MODES} onChange={(v) => onChange({ load: v })} inherit={inherit} />
        </div>
        <div className="field">
          <label>Idle TTL</label>
          <UnitField value={settings.idle_ttl} onChange={(v) => onChange({ idle_ttl: v })} placeholder="15m (0 = never)" />
        </div>
        <div className="field">
          <label>MLX profile</label>
          <Select value={settings.profile} options={MLX_PROFILES} onChange={(v) => onChange({ profile: v })} inherit={inherit} />
        </div>
        <div className="field">
          <label>Concurrent requests</label>
          <select
            value={settings.batch === undefined ? "" : String(settings.batch)}
            onChange={(event) => {
              const choice = event.target.value;
              onChange({ batch: choice === "" ? undefined : choice === "auto" ? "auto" : choice === "true" });
            }}
          >
            <option value="">{inherit}</option>
            <option value="auto">unlimited</option>
            <option value="false">one at a time</option>
          </select>
        </div>
        <div className="field">
          <label>Thinking budget</label>
          <NumberField value={settings.thinking_budget} onChange={(v) => onChange({ thinking_budget: v })} placeholder="256" />
        </div>
      </div>
      <div className="form-row">
        <Checkbox label="Pinned (never evicted or expired)" checked={settings.pinned ?? false} onChange={(v) => onChange({ pinned: v })} />
      </div>
      <details className="config-disclosure">
        <summary>Advanced (KV cache, prefix cache, prefill)</summary>
        <div className="form-row">
          <div className="field">
            <label>KV cache mode</label>
            <Select value={kv.mode} options={["off", "turboquant"]} onChange={(v) => setKv({ mode: v })} inherit={inherit} />
          </div>
          <div className="field">
            <label>Bits</label>
            <NumberField value={kv.bits} onChange={(v) => setKv({ bits: v })} placeholder="3" />
          </div>
          <div className="field">
            <label>Key bits</label>
            <NumberField value={kv.key_bits} onChange={(v) => setKv({ key_bits: v })} />
          </div>
          <div className="field">
            <label>Value bits</label>
            <NumberField value={kv.value_bits} onChange={(v) => setKv({ value_bits: v })} />
          </div>
          <div className="field">
            <label>Adaptive dense layers</label>
            <NumberField value={kv.adaptive_dense_layers} onChange={(v) => setKv({ adaptive_dense_layers: v })} placeholder="0" />
          </div>
          <div className="field">
            <label>Seed</label>
            <NumberField value={kv.seed} onChange={(v) => setKv({ seed: v })} placeholder="0" />
          </div>
        </div>
        <div className="form-row">
          <Checkbox label="Norm correction" checked={kv.norm_correction ?? true} onChange={(v) => setKv({ norm_correction: v })} />
          <Checkbox label="MLA latent cache (DeepSeek-V2)" checked={kv.mla_latent ?? false} onChange={(v) => setKv({ mla_latent: v })} />
        </div>
        <div className="form-row">
          <div className="field">
            <label>Prefix cache directory</label>
            <TextField value={prefix.disk_dir} onChange={(v) => setPrefix({ disk_dir: v })} placeholder="~/.cache/higgs/kv" />
          </div>
          <div className="field">
            <label>Prefix cache budget</label>
            <UnitField value={prefix.disk_budget} onChange={(v) => setPrefix({ disk_budget: v })} placeholder="4GB" />
          </div>
          <div className="field">
            <label>Prefill yield tokens</label>
            <NumberField
              value={settings.prefill?.yield_tokens}
              onChange={(v) => onChange({ prefill: { ...settings.prefill, yield_tokens: v } })}
              placeholder="0"
            />
          </div>
        </div>
      </details>
    </>
  );
}

export function ConfigForm({ draft, onChange, section }: ConfigFormProps) {
  const server = draft.server ?? {};
  const setServer = (patch: Partial<ServerSection>) => onChange({ ...draft, server: { ...server, ...patch } });

  const memory = draft.memory ?? {};
  const setMemory = (patch: Partial<MemorySection>) => onChange({ ...draft, memory: { ...memory, ...patch } });

  const defaults = draft.defaults ?? {};
  const setDefaults = (patch: Partial<ModelSettings>) => onChange({ ...draft, defaults: { ...defaults, ...patch } });

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

  const providerEntries = Object.entries(draft.providers ?? {});
  const setProviders = (entries: Array<[string, ProviderEntry]>) => {
    const map: Record<string, ProviderEntry> = {};
    for (const [name, entry] of entries) map[name] = entry;
    onChange({ ...draft, providers: entries.length > 0 ? map : undefined });
  };
  const [duplicateProviderIndex, setDuplicateProviderIndex] = useState<number | null>(null);
  const updateProviderName = (index: number, name: string) => {
    const isDuplicate = providerEntries.some(([existingName], existingIndex) => existingIndex !== index && existingName === name);
    if (isDuplicate) {
      setDuplicateProviderIndex(index);
      return;
    }
    setDuplicateProviderIndex(null);
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

  const routing = draft.routing ?? {};
  const autoRouter = routing.auto ?? {};
  const setAutoRouter = (patch: Partial<AutoRouterSection>) => onChange({ ...draft, routing: { ...routing, auto: { ...autoRouter, ...patch } } });

  const observability = draft.observability ?? {};
  const metricsLog = observability.metrics_log ?? {};
  const setMetricsLog = (patch: Partial<MetricsLogSection>) => onChange({ ...draft, observability: { ...observability, metrics_log: { ...metricsLog, ...patch } } });
  const retentionOff = observability.retention === false;

  const experimental = draft.experimental ?? {};
  const setExperimental = (patch: Partial<ExperimentalSection>) => onChange({ ...draft, experimental: { ...experimental, ...patch } });

  const providers = providerNames(draft);
  const modelNames = models.map((model) => displayModelName(model));

  const panel = { padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 } as const;

  if (section === "server") {
    return (
      <div className="panel" style={panel}>
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
          <span className="field-hint">Required if host is 0.0.0.0. Prefer an environment reference such as {"${HIGGS_API_KEY}"} over a literal key.</span>
        </div>
        <div className="form-row">
          <div className="field">
            <label>Max tokens</label>
            <NumberField value={server.max_tokens} onChange={(v) => setServer({ max_tokens: v })} placeholder={String(CONFIG_DEFAULTS.server.max_tokens)} />
          </div>
          <div className="field">
            <label>Request timeout</label>
            <UnitField value={server.request_timeout} onChange={(v) => setServer({ request_timeout: v })} placeholder={CONFIG_DEFAULTS.server.request_timeout} />
          </div>
          <div className="field">
            <label>Max body size</label>
            <UnitField value={server.max_body_size} onChange={(v) => setServer({ max_body_size: v })} placeholder={CONFIG_DEFAULTS.server.max_body_size} />
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

  if (section === "memory") {
    return (
      <div className="panel" style={panel}>
        <span className="meta">Memory</span>
        <div className="form-row">
          <div className="field">
            <label>Guard</label>
            <Select value={memory.guard} options={MEMORY_GUARDS} onChange={(v) => setMemory({ guard: v })} inherit={CONFIG_DEFAULTS.memory.guard} />
          </div>
          <div className="field">
            <label>Limit</label>
            <UnitField value={memory.limit} onChange={(v) => setMemory({ limit: v })} placeholder="48GB" />
          </div>
          <div className="field">
            <label>Max loaded models</label>
            <NumberField value={memory.max_loaded_models} onChange={(v) => setMemory({ max_loaded_models: v })} />
          </div>
          <div className="field">
            <label>&nbsp;</label>
            <Checkbox
              label="Raise wired memory limit"
              checked={memory.raise_wired_limit ?? CONFIG_DEFAULTS.memory.raise_wired_limit}
              onChange={(v) => setMemory({ raise_wired_limit: v })}
            />
          </div>
        </div>
        <span className="field-hint">An explicit limit wins over the guard. Both bound MLX active memory across all loaded models.</span>
      </div>
    );
  }

  if (section === "defaults") {
    return (
      <div className="panel" style={panel}>
        <span className="meta">Model defaults</span>
        <span className="field-hint">Applies to every model; a model entry overrides it key by key.</span>
        <ModelSettingsEditor settings={defaults} onChange={setDefaults} inherit="built-in" />
      </div>
    );
  }

  if (section === "models") {
    return (
      <div className="panel" style={panel}>
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
                <TextField value={model.name} onChange={(v) => updateModel(index, { name: v })} placeholder="name" />
              </div>
              <div className="field" style={{ flex: 1 }}>
                <label>Aliases</label>
                <input
                  type="text"
                  value={(model.aliases ?? []).join(", ")}
                  placeholder="default, chat"
                  onChange={(event) => {
                    const list = event.target.value
                      .split(",")
                      .map((item) => item.trim())
                      .filter(Boolean);
                    updateModel(index, { aliases: list.length > 0 ? list : undefined });
                  }}
                />
              </div>
              <ReorderButtons index={index} count={models.length} onMove={(delta) => moveModel(index, delta)} onRemove={() => removeModel(index)} />
            </div>
            <ModelSettingsEditor settings={model} onChange={(patch) => updateModel(index, patch)} inherit="inherit" />
          </div>
        ))}
      </div>
    );
  }

  if (section === "providers") {
    return (
      <div className="panel" style={panel}>
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
                {duplicateProviderIndex === index && (
                  <span className="field-hint" style={{ color: "var(--bad)" }}>
                    A provider with this name already exists.
                  </span>
                )}
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
        <div className="panel" style={panel}>
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
                  <label>Match (regex)</label>
                  <TextField value={route.match} onChange={(v) => updateRoute(index, { match: v })} placeholder="claude-.*" />
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
          <div className="panel" style={panel}>
            <span className="meta">Default route</span>
            <select
              value={routing.default ?? CONFIG_DEFAULTS.routing.default}
              onChange={(event) => onChange({ ...draft, routing: { ...routing, default: event.target.value } })}
            >
              {providers.map((provider) => (
                <option key={provider} value={provider}>
                  {provider}
                </option>
              ))}
            </select>
          </div>

          <div className="panel" style={{ ...panel, gap: 8 }}>
            <span className="meta">Auto router</span>
            <Checkbox
              label="Enabled"
              checked={autoRouter.enabled ?? CONFIG_DEFAULTS.routing.auto.enabled}
              onChange={(v) => setAutoRouter({ enabled: v })}
            />
            <Checkbox label="Force" checked={autoRouter.force ?? CONFIG_DEFAULTS.routing.auto.force} onChange={(v) => setAutoRouter({ force: v })} />
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
              <label>Timeout</label>
              <UnitField value={autoRouter.timeout} onChange={(v) => setAutoRouter({ timeout: v })} placeholder={CONFIG_DEFAULTS.routing.auto.timeout} />
            </div>
          </div>
        </div>
      </>
    );
  }

  if (section === "experimental") {
    const chunked = experimental.chunked_prefill ?? {};
    return (
      <div className="panel" style={panel}>
        <span className="meta">Experimental</span>
        <span className="field-hint">Engine toggles with an environment-variable equivalent; the variable wins when set (see `higgs doctor`).</span>
        <div className="form-row">
          <div className="field">
            <label>MTP head</label>
            <Select value={experimental.mtp} options={MTP_MODES} onChange={(v) => setExperimental({ mtp: v })} inherit="auto" />
          </div>
          <div className="field">
            <label>MTP draft tokens</label>
            <NumberField value={experimental.mtp_draft_tokens} onChange={(v) => setExperimental({ mtp_draft_tokens: v })} placeholder="profile" />
          </div>
          <div className="field">
            <label>Chunked prefill threshold</label>
            <NumberField
              value={chunked.threshold}
              onChange={(v) => setExperimental({ chunked_prefill: { ...chunked, threshold: v } })}
              placeholder="profile"
            />
          </div>
          <div className="field">
            <label>Chunked prefill chunk</label>
            <NumberField value={chunked.chunk} onChange={(v) => setExperimental({ chunked_prefill: { ...chunked, chunk: v } })} placeholder="profile" />
          </div>
        </div>
        <div className="form-row">
          <Checkbox label="Prompt lookup" checked={experimental.prompt_lookup ?? false} onChange={(v) => setExperimental({ prompt_lookup: v })} />
          <Checkbox label="MTP adaptive draft" checked={experimental.mtp_adaptive_draft ?? false} onChange={(v) => setExperimental({ mtp_adaptive_draft: v })} />
          <Checkbox label="MTP prompt lookup" checked={experimental.mtp_prompt_lookup ?? false} onChange={(v) => setExperimental({ mtp_prompt_lookup: v })} />
          <Checkbox
            label="Clear cache after prefill"
            checked={experimental.clear_cache_after_prefill ?? false}
            onChange={(v) => setExperimental({ clear_cache_after_prefill: v })}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="grid-2">
      <div className="panel" style={{ ...panel, gap: 8 }}>
        <span className="meta">Dashboard retention</span>
        <Checkbox
          label="Keep metrics in memory"
          checked={!retentionOff}
          onChange={(on) => onChange({ ...draft, observability: { ...observability, retention: on ? undefined : false } })}
        />
        <div className="field">
          <label>Window</label>
          <UnitField
            value={typeof observability.retention === "string" ? observability.retention : undefined}
            onChange={(v) => onChange({ ...draft, observability: { ...observability, retention: v } })}
            placeholder={CONFIG_DEFAULTS.observability.retention}
          />
        </div>
      </div>

      <div className="panel" style={{ ...panel, gap: 8 }}>
        <span className="meta">Metrics log</span>
        <Checkbox
          label="Enabled"
          checked={metricsLog.enabled ?? CONFIG_DEFAULTS.observability.metrics.enabled}
          onChange={(v) => setMetricsLog({ enabled: v })}
        />
        <div className="field">
          <label>Path</label>
          <TextField value={metricsLog.path} onChange={(v) => setMetricsLog({ path: v })} placeholder="~/.config/higgs/logs/metrics.jsonl" />
        </div>
        <div className="form-row">
          <div className="field">
            <label>Max size</label>
            <UnitField value={metricsLog.max_size} onChange={(v) => setMetricsLog({ max_size: v })} placeholder={CONFIG_DEFAULTS.observability.metrics.max_size} />
          </div>
          <div className="field">
            <label>Max files</label>
            <NumberField
              value={metricsLog.max_files}
              onChange={(v) => setMetricsLog({ max_files: v })}
              placeholder={String(CONFIG_DEFAULTS.observability.metrics.max_files)}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
