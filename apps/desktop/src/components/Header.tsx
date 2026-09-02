import { useState } from "react";
import { runHiggs } from "../lib/api";
import { daemonSelectorArgs } from "../views/ConfigView";
import type { ServerData } from "../hooks/useServerData";
import type { Settings } from "../lib/types";

interface Props {
  settings: Settings;
  onSettingsChange: (settings: Settings) => void;
  data: ServerData;
  onNewChat: () => void;
}

const RANGES: Array<{ minutes: number; label: string }> = [
  { minutes: 5, label: "5m" },
  { minutes: 15, label: "15m" },
  { minutes: 60, label: "60m" },
];

/** Connection, profile, daemon controls, and range picker shared by every section. */
export function Header({ settings, onSettingsChange, data, onNewChat }: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const selectorArgs = daemonSelectorArgs(settings.profile, data.configFile?.path ?? null);
  const run = async (label: string, steps: string[][]) => {
    setBusy(label);
    setNotice(null);
    try {
      for (const args of steps) {
        const output = await runHiggs(settings.higgsBinary, [...args, ...selectorArgs]);
        if (output.exit_code !== 0) {
          setNotice(`${args[0]} failed: ${(output.stderr || output.stdout).trim().split("\n").slice(-3).join(" ")}`);
          return;
        }
      }
      setNotice(`${label} ok`);
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(null);
      data.refresh();
    }
  };

  const running = data.daemon?.running ?? false;
  const healthy = data.health?.ok ?? false;

  return (
    <header className="header" data-tauri-drag-region>
      <div className="header-group">
        <span className={`status-dot ${data.health === null ? "unknown" : healthy ? "ok" : "bad"}`} />
        <span className="header-strong">{healthy ? "Online" : data.health === null ? "Connecting" : "Offline"}</span>
        <code className="header-muted mono">{settings.baseUrl.replace(/^https?:\/\//, "")}</code>
      </div>
      <div className="header-divider" />
      <div className="header-group">
        <span className="header-muted label">Profile</span>
        <span className="header-strong">{settings.profile ?? "default"}</span>
      </div>
      {data.localAvailable && (
        <div className="header-group">
          <span className="header-muted label">Daemon</span>
          <span className="header-strong mono">{data.daemon === null ? "…" : running ? `pid ${data.daemon.pid}` : "stopped"}</span>
        </div>
      )}
      {notice && <span className="header-notice">{notice}</span>}
      <div className="header-spacer" data-tauri-drag-region />
      <div className="seg-group">
        {RANGES.map((range) => (
          <button
            key={range.minutes}
            type="button"
            className={`seg ${settings.windowMinutes === range.minutes ? "on" : ""}`}
            onClick={() => onSettingsChange({ ...settings, windowMinutes: range.minutes })}
          >
            {range.label}
          </button>
        ))}
      </div>
      {data.localAvailable &&
        (running ? (
          <>
            <button type="button" className="btn" disabled={busy !== null} onClick={() => run("restart", [["stop"], ["start"]])}>
              {busy === "restart" ? "Restarting…" : "Restart"}
            </button>
            <button type="button" className="btn" disabled={busy !== null} onClick={() => run("stop", [["stop"]])}>
              {busy === "stop" ? "Stopping…" : "Stop"}
            </button>
          </>
        ) : (
          <button type="button" className="btn hot" disabled={busy !== null} onClick={() => run("start", [["start"]])}>
            {busy === "start" ? "Starting…" : "Start"}
          </button>
        ))}
      <button type="button" className="btn hot" onClick={onNewChat}>
        New chat
      </button>
    </header>
  );
}
