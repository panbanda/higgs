import { useState } from "react";
import { devBridge, inTauri, runHiggs } from "../lib/api";
import { metricsLogPath, serverUrl } from "../lib/config";
import { DEFAULT_SETTINGS } from "../lib/types";
import type { ServerData } from "../hooks/useServerData";
import type { Settings } from "../lib/types";
import "./views.css";

export interface SettingsViewProps {
  settings: Settings;
  onSettingsChange: (settings: Settings) => void;
  data: ServerData;
}

export function SettingsView({ settings, onSettingsChange, data }: SettingsViewProps) {
  const [showApiKey, setShowApiKey] = useState(false);
  const [showHfToken, setShowHfToken] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [detectResult, setDetectResult] = useState<{ ok: boolean; text: string } | null>(null);

  const detect = async () => {
    setDetecting(true);
    setDetectResult(null);
    try {
      const output = await runHiggs(settings.higgsBinary, ["--version"]);
      if (output.exit_code === 0) {
        setDetectResult({ ok: true, text: `${output.program}: ${output.stdout.trim() || output.stderr.trim()}` });
      } else {
        setDetectResult({ ok: false, text: (output.stderr || output.stdout).trim() || "not found" });
      }
    } catch (error) {
      setDetectResult({ ok: false, text: String(error) });
    } finally {
      setDetecting(false);
    }
  };

  const resetSettings = () => {
    onSettingsChange({ ...DEFAULT_SETTINGS, params: { ...DEFAULT_SETTINGS.params } });
  };

  const configDir = data.profiles?.config_dir ?? null;
  const metricsPath = configDir ? metricsLogPath(data.config, settings.profile, configDir) : null;

  return (
    <div className="view">
      <div className="view-head">
        <h1>Settings</h1>
      </div>

      <div className="config-panels">
        <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
          <span className="meta">Connection</span>
          <div className="field">
            <label>Base URL</label>
            <div className="config-password-row">
              <input type="text" value={settings.baseUrl} onChange={(event) => onSettingsChange({ ...settings, baseUrl: event.target.value })} />
              <button type="button" className="btn" onClick={() => onSettingsChange({ ...settings, baseUrl: serverUrl(data.config) })}>
                Use config's server URL
              </button>
            </div>
          </div>
          <div className="field">
            <label>API key</label>
            <div className="config-password-row">
              <input
                type={showApiKey ? "text" : "password"}
                value={settings.apiKey}
                onChange={(event) => onSettingsChange({ ...settings, apiKey: event.target.value })}
              />
              <button type="button" className="btn" onClick={() => setShowApiKey((current) => !current)}>
                {showApiKey ? "Hide" : "Show"}
              </button>
            </div>
          </div>
          <div className="field">
            <label>Profile</label>
            <select value={settings.profile ?? ""} onChange={(event) => onSettingsChange({ ...settings, profile: event.target.value || null })}>
              {(data.profiles?.profiles ?? []).map((profile) => (
                <option key={profile.name ?? ""} value={profile.name ?? ""}>
                  {profile.name ?? "default"}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Hugging Face token</label>
            <div className="config-password-row">
              <input
                type={showHfToken ? "text" : "password"}
                value={settings.hfToken}
                onChange={(event) => onSettingsChange({ ...settings, hfToken: event.target.value })}
              />
              <button type="button" className="btn" onClick={() => setShowHfToken((current) => !current)}>
                {showHfToken ? "Hide" : "Show"}
              </button>
            </div>
            <span className="field-hint">Needed only for gated repos in the Hub view.</span>
          </div>
        </div>

        <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
          <span className="meta">Higgs binary</span>
          <div className="field">
            <label>Path</label>
            <div className="config-password-row">
              <input
                type="text"
                placeholder="resolved via login shell (higgs)"
                value={settings.higgsBinary}
                onChange={(event) => onSettingsChange({ ...settings, higgsBinary: event.target.value })}
              />
              <button type="button" className="btn" disabled={detecting} onClick={() => void detect()}>
                {detecting ? "Detecting…" : "Detect"}
              </button>
            </div>
            {detectResult && <span className={`notice ${detectResult.ok ? "ok" : "bad"}`}>{detectResult.text}</span>}
          </div>
        </div>

        <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
          <span className="meta">Dashboard</span>
          <div className="form-row">
            <div className="field">
              <label>Refresh interval (seconds)</label>
              <input
                type="number"
                min={1}
                max={60}
                value={settings.refreshSeconds}
                onChange={(event) => onSettingsChange({ ...settings, refreshSeconds: Math.min(60, Math.max(1, Number(event.target.value) || 1)) })}
              />
            </div>
            <div className="field">
              <label>History window (minutes)</label>
              <input
                type="number"
                min={5}
                max={1440}
                value={settings.windowMinutes}
                onChange={(event) => onSettingsChange({ ...settings, windowMinutes: Math.min(1440, Math.max(5, Number(event.target.value) || 5)) })}
              />
            </div>
          </div>
          <button type="button" className="btn" onClick={resetSettings}>
            Reset app settings
          </button>
        </div>

        <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 4 }}>
          <span className="meta" style={{ marginBottom: 6 }}>
            Paths
          </span>
          <div className="settings-readonly-row">
            <span className="settings-readonly-label">Mode</span>
            <span className="settings-readonly-value">{inTauri ? "Desktop app" : devBridge ? "Browser (dev bridge)" : "Browser (HTTP only)"}</span>
          </div>
          <div className="settings-readonly-row">
            <span className="settings-readonly-label">Config dir</span>
            <span className="settings-readonly-value">{configDir ?? "–"}</span>
          </div>
          <div className="settings-readonly-row">
            <span className="settings-readonly-label">Active config path</span>
            <span className="settings-readonly-value">{data.configFile?.path ?? "–"}</span>
          </div>
          <div className="settings-readonly-row">
            <span className="settings-readonly-label">Metrics log path</span>
            <span className="settings-readonly-value">{metricsPath ?? "–"}</span>
          </div>
          <div className="settings-readonly-row">
            <span className="settings-readonly-label">Daemon PID</span>
            <span className="settings-readonly-value">{data.daemon ? (data.daemon.running ? data.daemon.pid : "not running") : "–"}</span>
          </div>
          <div className="settings-readonly-row">
            <span className="settings-readonly-label">Daemon PID file</span>
            <span className="settings-readonly-value">{data.daemon?.pid_path ?? "–"}</span>
          </div>
          <div className="settings-readonly-row">
            <span className="settings-readonly-label">Daemon log path</span>
            <span className="settings-readonly-value">{data.daemon?.log_path ?? "–"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
