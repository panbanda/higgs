import { useEffect, useState } from "react";
import { readTextTail, runHiggs, writeConfigRaw, writeConfigStructured } from "../lib/api";
import { CONFIG_DEFAULTS, type HiggsConfig } from "../lib/config";
import type { ServerData } from "../hooks/useServerData";
import type { CommandOutput, Settings } from "../lib/types";
import { CONFIG_SECTIONS, ConfigForm, sanitizeConfig, type ConfigSection } from "../components/ConfigForm";
import { DoctorOutput } from "../components/DoctorOutput";
import "./views.css";

export interface ConfigViewProps {
  settings: Settings;
  onSettingsChange: (settings: Settings) => void;
  data: ServerData;
}

type Tab = "form" | "raw";

/**
 * Daemon commands must select the profile with `--profile <name>` and never
 * combine it with `--config <path>`; the default profile takes `--config`.
 */
export function daemonSelectorArgs(profile: string | null, configPath: string | null): string[] {
  return profile ? ["--profile", profile] : configPath ? ["--config", configPath] : [];
}

export function ConfigView({ settings, onSettingsChange, data }: ConfigViewProps) {
  const [tab, setTab] = useState<Tab>("form");
  const [section, setSection] = useState<ConfigSection>("server");
  const [draft, setDraft] = useState<HiggsConfig>(() => data.config);
  const [formDirty, setFormDirty] = useState(false);
  const [rawText, setRawText] = useState<string>(() => data.configFile?.raw ?? "");
  const [rawDirty, setRawDirty] = useState(false);
  const [saveNotice, setSaveNotice] = useState<{ ok: boolean; text: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const [doctorBusy, setDoctorBusy] = useState(false);
  const [doctorOutput, setDoctorOutput] = useState<CommandOutput | null>(null);
  const [daemonBusy, setDaemonBusy] = useState<string | null>(null);
  const [logTail, setLogTail] = useState<string>("");
  const [logError, setLogError] = useState<string | null>(null);
  const [logMissing, setLogMissing] = useState(false);
  const [showNewProfile, setShowNewProfile] = useState(false);
  const [newProfileName, setNewProfileName] = useState("");
  const [creatingProfile, setCreatingProfile] = useState(false);
  const [creatingDefault, setCreatingDefault] = useState(false);

  useEffect(() => {
    if (!formDirty) setDraft(data.config);
    if (!rawDirty) setRawText(data.configFile?.raw ?? "");
    // Re-sync from the loaded file only; local edits are tracked by the dirty flags above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.configFile]);

  const configPath = data.configFile?.path ?? null;

  const runDoctor = async () => {
    setDoctorBusy(true);
    try {
      const args = ["doctor", ...daemonSelectorArgs(settings.profile, configPath)];
      const output = await runHiggs(settings.higgsBinary, args);
      setDoctorOutput(output);
    } catch (error) {
      setDoctorOutput({ program: settings.higgsBinary || "higgs", exit_code: null, stdout: "", stderr: String(error) });
    } finally {
      setDoctorBusy(false);
    }
  };

  const saveForm = async () => {
    if (!configPath) return false;
    setSaving(true);
    setSaveNotice(null);
    try {
      await writeConfigStructured(configPath, sanitizeConfig(draft));
      await data.reloadConfig();
      setFormDirty(false);
      setSaveNotice({ ok: true, text: "Saved." });
      return true;
    } catch (error) {
      setSaveNotice({ ok: false, text: String(error) });
      return false;
    } finally {
      setSaving(false);
    }
  };

  const saveRaw = async () => {
    if (!configPath) return false;
    setSaving(true);
    setSaveNotice(null);
    try {
      await writeConfigRaw(configPath, rawText);
      await data.reloadConfig();
      setRawDirty(false);
      setSaveNotice({ ok: true, text: "Saved." });
      return true;
    } catch (error) {
      setSaveNotice({ ok: false, text: String(error) });
      return false;
    } finally {
      setSaving(false);
    }
  };

  const save = async () => {
    const ok = tab === "form" ? await saveForm() : await saveRaw();
    if (ok) await runDoctor();
  };

  const discard = () => {
    setDraft(data.config);
    setFormDirty(false);
    setRawText(data.configFile?.raw ?? "");
    setRawDirty(false);
    setSaveNotice(null);
  };

  const runDaemon = async (label: string, steps: string[][]) => {
    setDaemonBusy(label);
    try {
      for (const args of steps) {
        const output = await runHiggs(settings.higgsBinary, args);
        if (output.exit_code !== 0) {
          setSaveNotice({ ok: false, text: `${args[0]} failed: ${(output.stderr || output.stdout).trim()}` });
          return;
        }
      }
    } catch (error) {
      setSaveNotice({ ok: false, text: String(error) });
    } finally {
      setDaemonBusy(null);
      data.refresh();
    }
  };

  const applyAndRestart = async () => {
    const ok = tab === "form" ? await saveForm() : await saveRaw();
    if (!ok) return;
    const selector = daemonSelectorArgs(settings.profile, configPath);
    await runDaemon("restart", [["stop", ...selector], ["start", ...selector]]);
  };

  const refreshLog = async () => {
    if (!data.daemon?.log_path) return;
    try {
      const text = await readTextTail(data.daemon.log_path, 64 * 1024);
      setLogTail(text);
      setLogError(null);
      setLogMissing(false);
    } catch (error) {
      const message = String(error);
      if (/no such file or directory|cannot find the file|os error 2/i.test(message)) {
        setLogError(null);
        setLogMissing(true);
      } else {
        setLogError(message);
        setLogMissing(false);
      }
    }
  };

  useEffect(() => {
    if (data.daemon?.log_path) void refreshLog();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.daemon?.log_path]);

  const createProfile = async () => {
    const name = newProfileName.trim();
    if (!name || !data.profiles) return;
    setCreatingProfile(true);
    try {
      const path = `${data.profiles.config_dir}/config.${name}.toml`;
      await writeConfigStructured(path, sanitizeConfig(draft));
      setShowNewProfile(false);
      setNewProfileName("");
      onSettingsChange({ ...settings, profile: name });
    } catch (error) {
      setSaveNotice({ ok: false, text: String(error) });
    } finally {
      setCreatingProfile(false);
    }
  };

  const createDefaultConfig = async () => {
    if (!configPath) return;
    setCreatingDefault(true);
    try {
      await writeConfigStructured(configPath, {
        server: { host: CONFIG_DEFAULTS.server.host, port: CONFIG_DEFAULTS.server.port },
        models: [],
      } satisfies HiggsConfig);
      await data.reloadConfig();
    } catch (error) {
      setSaveNotice({ ok: false, text: String(error) });
    } finally {
      setCreatingDefault(false);
    }
  };

  if (!data.localAvailable) {
    return (
      <div className="view">
        <h1>Config</h1>
        <div className="notice">Config editing needs the desktop app and is not available in browser mode.</div>
      </div>
    );
  }

  const running = data.daemon?.running ?? false;
  const healthy = data.health?.ok ?? false;
  const dirty = tab === "form" ? formDirty : rawDirty;

  return (
    <div className="view">
      <div className="view-head">
        <h1>Config</h1>
      </div>

      <div className="config-toolbar">
        <div className="field inline">
          <label>Profile</label>
          <select value={settings.profile ?? ""} onChange={(event) => onSettingsChange({ ...settings, profile: event.target.value || null })}>
            {(data.profiles?.profiles ?? []).map((profile) => (
              <option key={profile.name ?? ""} value={profile.name ?? ""}>
                {profile.name ?? "default"}
              </option>
            ))}
          </select>
        </div>
        {!showNewProfile ? (
          <button type="button" className="btn" onClick={() => setShowNewProfile(true)}>
            New profile
          </button>
        ) : (
          <div className="config-new-profile">
            <input type="text" autoFocus placeholder="profile name" value={newProfileName} onChange={(event) => setNewProfileName(event.target.value)} />
            <button type="button" className="btn hot" disabled={creatingProfile || !newProfileName.trim()} onClick={() => void createProfile()}>
              {creatingProfile ? "Creating…" : "Create"}
            </button>
            <button type="button" className="btn" onClick={() => setShowNewProfile(false)}>
              Cancel
            </button>
          </div>
        )}
        <div className="config-path">
          <span>{configPath ?? "no config path"}</span>
          <span className="state-cell">
            <span className="swatch" style={{ background: data.configFile?.exists ? "var(--ok)" : "var(--warn)" }} />
            <span>{data.configFile?.exists ? "exists" : "missing"}</span>
          </span>
        </div>
      </div>

      {data.configFile && data.configFile.exists === false && (
        <div className="panel" style={{ padding: "16px 20px", marginBottom: 16 }}>
          <span className="meta">No config yet</span>
          <p className="field-hint">Create a default configuration to get started.</p>
          <button type="button" className="btn hot" disabled={creatingDefault} onClick={() => void createDefaultConfig()}>
            {creatingDefault ? "Creating…" : "Create default config"}
          </button>
        </div>
      )}

      {data.configFile?.exists && !healthy && !running && (
        <div className="notice" style={{ marginBottom: 16 }}>
          Config exists but the server isn't reachable. Start the daemon below.
        </div>
      )}

      <div className="tabs" style={{ marginBottom: 16 }}>
        <button type="button" className={`tab ${tab === "form" ? "active" : ""}`} onClick={() => setTab("form")}>
          Form
        </button>
        <button type="button" className={`tab ${tab === "raw" ? "active" : ""}`} onClick={() => setTab("raw")}>
          Raw TOML
        </button>
      </div>

      <div className="config-actions-row">
        <button type="button" className="btn" disabled={!dirty || saving || !configPath} onClick={() => void save()}>
          {saving ? "Saving…" : "Save"}
        </button>
        <button type="button" className="btn" disabled={!dirty || saving} onClick={discard}>
          Discard
        </button>
        {dirty && <span className="pill warn">unsaved changes</span>}
        {saveNotice && <span className={`notice ${saveNotice.ok ? "ok" : "bad"}`}>{saveNotice.text}</span>}
      </div>

      {tab === "form" ? (
        <div className="config-layout">
          <div className="config-nav">
            {CONFIG_SECTIONS.map((entry) => (
              <button
                key={entry.id}
                type="button"
                className={`config-nav-btn ${section === entry.id ? "on" : ""}`}
                onClick={() => setSection(entry.id)}
              >
                {entry.label}
              </button>
            ))}
            <button type="button" className="btn hot config-nav-save" disabled={saving || daemonBusy !== null} onClick={() => void applyAndRestart()}>
              {daemonBusy === "restart" ? "Applying…" : "Save and restart"}
            </button>
          </div>
          <div className="config-panels">
            <div className="notice">Saving from the Form tab rewrites the config file without comments. Use the Raw TOML tab to preserve comments.</div>
            <ConfigForm
              draft={draft}
              section={section}
              onChange={(next) => {
                setDraft(next);
                setFormDirty(true);
              }}
            />
            <DoctorOutput busy={doctorBusy} output={doctorOutput} onRun={() => void runDoctor()} />
            <DaemonPanel
              running={running}
              pid={data.daemon?.pid}
              daemonBusy={daemonBusy}
              logTail={logTail}
              logError={logError}
              logMissing={logMissing}
              logPath={data.daemon?.log_path}
              onStart={() => void runDaemon("start", [["start", ...daemonSelectorArgs(settings.profile, configPath)]])}
              onRestart={() => void runDaemon("restart", [["stop", ...daemonSelectorArgs(settings.profile, configPath)], ["start", ...daemonSelectorArgs(settings.profile, configPath)]])}
              onStop={() => void runDaemon("stop", [["stop", ...daemonSelectorArgs(settings.profile, configPath)]])}
              onRefreshLog={() => void refreshLog()}
            />
          </div>
        </div>
      ) : (
        <div className="config-panels">
          <div className="panel" style={{ padding: "16px 20px" }}>
            {data.configFile?.parse_error && (
              <div className="notice bad" style={{ marginBottom: 10 }}>
                {data.configFile.parse_error}
              </div>
            )}
            <textarea
              className="config-raw-textarea"
              spellCheck={false}
              value={rawText}
              onChange={(event) => {
                setRawText(event.target.value);
                setRawDirty(true);
              }}
            />
            <div className="config-actions-row">
              <button
                type="button"
                className="btn"
                onClick={() => {
                  setRawText(data.configFile?.raw ?? "");
                  setRawDirty(false);
                }}
              >
                Reload
              </button>
            </div>
          </div>
          <DoctorOutput busy={doctorBusy} output={doctorOutput} onRun={() => void runDoctor()} />
          <DaemonPanel
            running={running}
            pid={data.daemon?.pid}
            daemonBusy={daemonBusy}
            logTail={logTail}
            logError={logError}
            logMissing={logMissing}
            logPath={data.daemon?.log_path}
            onStart={() => void runDaemon("start", [["start", ...daemonSelectorArgs(settings.profile, configPath)]])}
            onRestart={() => void runDaemon("restart", [["stop", ...daemonSelectorArgs(settings.profile, configPath)], ["start", ...daemonSelectorArgs(settings.profile, configPath)]])}
            onStop={() => void runDaemon("stop", [["stop", ...daemonSelectorArgs(settings.profile, configPath)]])}
            onRefreshLog={() => void refreshLog()}
          />
        </div>
      )}
    </div>
  );
}

interface DaemonPanelProps {
  running: boolean;
  pid: number | null | undefined;
  daemonBusy: string | null;
  logTail: string;
  logError: string | null;
  logMissing: boolean;
  logPath: string | undefined;
  onStart: () => void;
  onRestart: () => void;
  onStop: () => void;
  onRefreshLog: () => void;
}

function DaemonPanel({ running, pid, daemonBusy, logTail, logError, logMissing, logPath, onStart, onRestart, onStop, onRefreshLog }: DaemonPanelProps) {
  return (
    <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 10 }}>
      <span className="meta">Daemon</span>
      <div className="state-cell">
        <span className="swatch" style={{ background: running ? "var(--ok)" : "var(--bad)" }} />
        <span>{running ? `running (pid ${pid})` : "stopped"}</span>
      </div>
      <div className="config-actions-row">
        {!running ? (
          <button type="button" className="btn hot" disabled={daemonBusy !== null} onClick={onStart}>
            {daemonBusy === "start" ? "Starting…" : "Start"}
          </button>
        ) : (
          <>
            <button type="button" className="btn" disabled={daemonBusy !== null} onClick={onRestart}>
              {daemonBusy === "restart" ? "Restarting…" : "Restart"}
            </button>
            <button type="button" className="btn" disabled={daemonBusy !== null} onClick={onStop}>
              {daemonBusy === "stop" ? "Stopping…" : "Stop"}
            </button>
          </>
        )}
        <button type="button" className="btn" onClick={onRefreshLog}>
          Refresh log
        </button>
      </div>
      {logError && <div className="notice bad">{logError}</div>}
      {logMissing && <div className="notice">No daemon log at {logPath} yet (written by `higgs start`).</div>}
      <pre className="log">{logTail || "No log output yet."}</pre>
    </div>
  );
}
