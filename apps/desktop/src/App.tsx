import { useEffect, useRef, useState } from "react";
import { openUrl } from "@tauri-apps/plugin-opener";
import { useServerData } from "./hooks/useServerData";
import { inTauri } from "./lib/api";
import { loadPresets, loadSecrets, loadSettings, saveSecret, savePresets, saveSettings } from "./lib/storage";
import type { Preset, Settings } from "./lib/types";
import { ChatView } from "./views/ChatView";
import { ConfigView } from "./views/ConfigView";
import { HubView } from "./views/HubView";
import { ModelsView } from "./views/ModelsView";
import { Overview } from "./views/Overview";
import { RequestsView } from "./views/RequestsView";
import { RoutingView } from "./views/RoutingView";
import { SettingsView } from "./views/SettingsView";
import { Header } from "./components/Header";
import { SignalStrip } from "./components/SignalStrip";

export type Section = "overview" | "models" | "hub" | "routing" | "requests" | "chat" | "config" | "settings";

const NAV_ICON_PATHS: Record<Section, string> = {
  overview: "M4 20 V10 M10 20 V4 M16 20 V13 M22 20 H2",
  models: "M4 7 L12 3 L20 7 L12 11 Z M4 7 V17 L12 21 L20 17 V7 M12 11 V21",
  hub: "M12 3 V15 M7 10 L12 15 L17 10 M4 21 H20",
  routing: "M6 3 V15 M18 9 A3 3 0 1 0 18 3 A3 3 0 0 0 18 9 M6 21 A3 3 0 1 0 6 15 A3 3 0 0 0 6 21 M18 9 A9 9 0 0 1 9 18",
  requests: "M4 6 H20 M4 12 H14 M4 18 H17",
  chat: "M4 5 H20 V16 H9 L5 19 V16 H4 Z",
  config: "M12 8 A4 4 0 1 0 12 16 A4 4 0 1 0 12 8 M12 2 V5 M12 19 V22 M2 12 H5 M19 12 H22",
  settings: "M4 21 V14 M4 10 V3 M12 21 V12 M12 8 V3 M20 21 V16 M20 12 V3 M1 14 H7 M9 8 H15 M17 16 H23",
};

const SECTIONS: Array<{ id: Section; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "models", label: "Models" },
  { id: "hub", label: "Hub" },
  { id: "routing", label: "Providers & Routing" },
  { id: "requests", label: "Requests" },
  { id: "chat", label: "Chat" },
  { id: "config", label: "Config" },
  { id: "settings", label: "Settings" },
];

const GITHUB_URL = "https://github.com/panbanda/higgs";


function NavIcon({ section }: { section: Section }) {
  return (
    <svg viewBox="0 0 24 24">
      <path d={NAV_ICON_PATHS[section]} />
    </svg>
  );
}

export default function App() {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [presets, setPresets] = useState<Preset[]>(loadPresets);
  const [section, setSection] = useState<Section>("overview");
  const [chatModel, setChatModel] = useState<string | null>(null);
  const data = useServerData(settings);

  useEffect(() => saveSettings(settings), [settings]);
  useEffect(() => savePresets(presets), [presets]);

  // Secrets live in the keychain (or in memory for browser mode), never in localStorage.
  const loadedSecretsRef = useRef<{ apiKey: string; hfToken: string } | null>(null);
  useEffect(() => {
    let cancelled = false;
    loadSecrets().then((secrets) => {
      if (cancelled) return;
      // Settings start with empty secrets (they are stripped from localStorage), so a
      // non-empty value here was typed before the keychain answered. Keep the edit and
      // persist it instead of clobbering it with the stored value.
      setSettings((current) => {
        const apiKey = current.apiKey !== "" ? current.apiKey : secrets.apiKey;
        const hfToken = current.hfToken !== "" ? current.hfToken : secrets.hfToken;
        loadedSecretsRef.current = { apiKey, hfToken };
        if (apiKey !== secrets.apiKey) void saveSecret("apiKey", apiKey);
        if (hfToken !== secrets.hfToken) void saveSecret("hfToken", hfToken);
        return apiKey === current.apiKey && hfToken === current.hfToken ? current : { ...current, apiKey, hfToken };
      });
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const loaded = loadedSecretsRef.current;
    if (loaded === null || settings.apiKey === loaded.apiKey) return;
    loadedSecretsRef.current = { ...loaded, apiKey: settings.apiKey };
    void saveSecret("apiKey", settings.apiKey);
  }, [settings.apiKey]);

  useEffect(() => {
    const loaded = loadedSecretsRef.current;
    if (loaded === null || settings.hfToken === loaded.hfToken) return;
    loadedSecretsRef.current = { ...loaded, hfToken: settings.hfToken };
    void saveSecret("hfToken", settings.hfToken);
  }, [settings.hfToken]);

  // Pick a served model once the list arrives, keeping a valid explicit choice.
  useEffect(() => {
    if (data.models.length === 0) return;
    if (!settings.model || !data.models.some((model) => model.id === settings.model)) {
      setSettings((current) => ({ ...current, model: data.models[0].id }));
    }
  }, [data.models]);

  const openChatWith = (modelId: string) => {
    setChatModel(modelId);
    setSettings((current) => ({ ...current, model: modelId }));
    setSection("chat");
  };

  const openGitHub = () => {
    if (inTauri) void openUrl(GITHUB_URL);
    else window.open(GITHUB_URL, "_blank", "noopener,noreferrer");
  };


  return (
    <div className="app">
      <nav className="rail" data-tauri-drag-region>
        <div className="rail-brand" data-tauri-drag-region>
          <div className="rail-brand-text">
            <span className="rail-brand-name">Higgs</span>
            <span className="rail-brand-tagline">Inference console</span>
          </div>
        </div>
        <div className="rail-nav">
          {SECTIONS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`rail-item ${section === item.id ? "active" : ""}`}
              onClick={() => setSection(item.id)}
            >
              <NavIcon section={item.id} />
              <span>{item.label}</span>
            </button>
          ))}
        </div>
        <div className="rail-spacer" />
        <div className="rail-foot">
          <div className="rail-foot-meta">
            <span className="mono">{data.system ? `v${data.system.version}` : "–"}</span>
            <a
              href={GITHUB_URL}
              onClick={(event) => {
                event.preventDefault();
                openGitHub();
              }}
            >
              GitHub
            </a>
          </div>
        </div>
      </nav>
      <div className="main">
        <Header settings={settings} onSettingsChange={setSettings} data={data} onNewChat={() => setSection("chat")} />
        <SignalStrip settings={settings} data={data} />
        <div className="workspace">
          <div key={section} className="view-swap">
            {section === "overview" && <Overview settings={settings} data={data} onOpenRequests={() => setSection("requests")} />}
            {section === "models" && <ModelsView settings={settings} data={data} onChatWith={openChatWith} onEditConfig={() => setSection("config")} />}
            {section === "hub" && <HubView settings={settings} data={data} />}
            {section === "routing" && <RoutingView settings={settings} data={data} onEditConfig={() => setSection("config")} />}
            {section === "requests" && <RequestsView settings={settings} data={data} />}
            {section === "chat" && (
              <ChatView settings={settings} onSettingsChange={setSettings} presets={presets} onPresetsChange={setPresets} data={data} requestedModel={chatModel} />
            )}
            {section === "config" && <ConfigView settings={settings} onSettingsChange={setSettings} data={data} />}
            {section === "settings" && <SettingsView settings={settings} onSettingsChange={setSettings} data={data} />}
          </div>
        </div>
      </div>
    </div>
  );
}
