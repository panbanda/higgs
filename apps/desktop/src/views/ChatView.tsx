import { useEffect, useMemo, useRef, useState } from "react";
import type { ServerData } from "../hooks/useServerData";
import { newAssistantMessage, replayRequest, runTurn, type TurnHandle } from "../lib/chat";
import { loadConversations, newId, saveConversations } from "../lib/storage";
import { TOOL_DEFINITIONS } from "../lib/tools";
import { DEFAULT_PARAMS, type AssistantMessage, type Conversation, type GenerationParams, type Preset, type ReasoningEffort, type Settings, type SystemInfo, type UserMessage } from "../lib/types";
import { Composer } from "../components/Composer";
import { MessageView } from "../components/MessageView";
import { RequestInspector } from "../components/RequestInspector";
import "./chat.css";

export interface ChatViewProps {
  settings: Settings; onSettingsChange: (settings: Settings) => void; presets: Preset[]; onPresetsChange: (presets: Preset[]) => void; data: ServerData; requestedModel: string | null;
}

const REASONING_OPTIONS: ReasoningEffort[] = ["default", "none", "low", "medium", "high"];

function MenuIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M4 6 H20 M4 12 H20 M4 18 H20" />
    </svg>
  );
}

function ChevronRightIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 5 L16 12 L9 19" />
    </svg>
  );
}

function ChevronLeftIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 5 L8 12 L15 19" />
    </svg>
  );
}

function conversationTitle(conversation: Conversation): string {
  const firstUser = conversation.messages.find((message) => message.role === "user");
  if (!firstUser) return "New conversation";
  const text = (firstUser as UserMessage).content.trim();
  return text.length > 48 ? `${text.slice(0, 48)}…` : text || "New conversation";
}

function newConversation(model: string): Conversation {
  const now = Date.now();
  return { id: newId(), title: "New conversation", model, messages: [], createdAt: now, updatedAt: now };
}

function numberOrNull(value: string): number | null {
  if (value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/** Best-effort "27B · 4-bit · engine" badge detail from what /v1/system reports; falls back to just the engine, or null if the model isn't loaded yet. */
function modelDetail(model: string, system: SystemInfo | null): string | null {
  const entry = system?.models.find((candidate) => candidate.name === model);
  if (!entry) return null;
  const source = `${entry.path ?? ""} ${entry.name}`;
  const sizeMatch = source.match(/(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])/);
  const quantMatch = source.match(/(\d)[- ]?bit/i) ?? source.match(/[qQ](\d)/);
  const parts: string[] = [];
  if (sizeMatch) parts.push(`${sizeMatch[1]}B`);
  if (quantMatch) parts.push(`${quantMatch[1]}-bit`);
  parts.push(entry.engine);
  return parts.join(" · ");
}

function ParamsPanel({ params, onChange }: { params: GenerationParams; onChange: (params: GenerationParams) => void }) {
  const set = <K extends keyof GenerationParams>(key: K, value: GenerationParams[K]) => onChange({ ...params, [key]: value });
  return (
    <div className="params-panel card">
      <div className="field">
        <label htmlFor="chat-system-prompt">System prompt</label>
        <textarea id="chat-system-prompt" rows={3} value={params.systemPrompt} onChange={(event) => set("systemPrompt", event.target.value)} />
      </div>
      <div className="form-row">
        <div className="field">
          <label htmlFor="chat-reasoning">Reasoning effort</label>
          <select id="chat-reasoning" value={params.reasoningEffort} onChange={(event) => set("reasoningEffort", event.target.value as ReasoningEffort)}>
            {REASONING_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>
        <div className="field inline">
          <label htmlFor="chat-tools">Tools</label>
          <input id="chat-tools" type="checkbox" checked={params.toolsEnabled} onChange={(event) => set("toolsEnabled", event.target.checked)} />
        </div>
      </div>
      <div className="form-row">
        <div className="field">
          <label htmlFor="chat-temperature">Temperature</label>
          <input
            id="chat-temperature"
            type="number"
            step="0.1"
            value={params.temperature ?? ""}
            placeholder="default"
            onChange={(event) => set("temperature", numberOrNull(event.target.value))}
          />
        </div>
        <div className="field">
          <label htmlFor="chat-top-p">Top P</label>
          <input id="chat-top-p" type="number" step="0.05" value={params.topP ?? ""} placeholder="default" onChange={(event) => set("topP", numberOrNull(event.target.value))} />
        </div>
        <div className="field">
          <label htmlFor="chat-top-k">Top K</label>
          <input id="chat-top-k" type="number" step="1" value={params.topK ?? ""} placeholder="default" onChange={(event) => set("topK", numberOrNull(event.target.value))} />
        </div>
      </div>
      <div className="form-row">
        <div className="field">
          <label htmlFor="chat-min-p">Min P</label>
          <input id="chat-min-p" type="number" step="0.01" value={params.minP ?? ""} placeholder="default" onChange={(event) => set("minP", numberOrNull(event.target.value))} />
        </div>
        <div className="field">
          <label htmlFor="chat-rep-penalty">Repetition penalty</label>
          <input
            id="chat-rep-penalty"
            type="number"
            step="0.05"
            value={params.repetitionPenalty ?? ""}
            placeholder="default"
            onChange={(event) => set("repetitionPenalty", numberOrNull(event.target.value))}
          />
        </div>
        <div className="field">
          <label htmlFor="chat-max-tokens">Max tokens</label>
          <input
            id="chat-max-tokens"
            type="number"
            step="1"
            value={params.maxTokens ?? ""}
            placeholder="default"
            onChange={(event) => set("maxTokens", numberOrNull(event.target.value))}
          />
        </div>
      </div>
    </div>
  );
}

/** The full override list, shown above the composer whenever any non-default parameter is set. */
function ParamChips({ params }: { params: GenerationParams }) {
  const chips: string[] = [];
  if (params.reasoningEffort !== "default") chips.push(`reasoning: ${params.reasoningEffort}`);
  if (params.temperature !== null) chips.push(`temp ${params.temperature}`);
  if (params.topP !== null) chips.push(`top_p ${params.topP}`);
  if (params.topK !== null) chips.push(`top_k ${params.topK}`);
  if (params.minP !== null) chips.push(`min_p ${params.minP}`);
  if (params.repetitionPenalty !== null) chips.push(`rep ${params.repetitionPenalty}`);
  if (params.maxTokens !== null) chips.push(`max_tokens ${params.maxTokens}`);
  if (!params.toolsEnabled) chips.push("tools off");
  if (params.systemPrompt.trim()) chips.push("system prompt set");
  if (chips.length === 0) return null;
  return (
    <div className="param-chips">
      {chips.map((chip) => (
        <span key={chip} className="pill">
          {chip}
        </span>
      ))}
    </div>
  );
}

/** Three always-visible glance chips in the header; clicking any opens the parameters panel. */
function TopChips({ params, onOpen }: { params: GenerationParams; onOpen: () => void }) {
  const toolCount = params.toolsEnabled ? TOOL_DEFINITIONS.length : 0;
  const chips = [`reasoning ${params.reasoningEffort}`, params.temperature !== null ? `temp ${params.temperature}` : "temp default", `tools ${toolCount}`];
  return (
    <>
      {chips.map((chip) => (
        <button key={chip} type="button" className="chip" onClick={onOpen}>
          {chip}
        </button>
      ))}
    </>
  );
}

export function ChatView({ settings, onSettingsChange, presets, onPresetsChange, data, requestedModel }: ChatViewProps) {
  const [conversations, setConversations] = useState<Conversation[]>(() => loadConversations());
  const [activeId, setActiveId] = useState<string | null>(() => conversations[0]?.id ?? null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [paramsOpen, setParamsOpen] = useState(false);
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);
  const [presetNameDraft, setPresetNameDraft] = useState<string | null>(null);
  const handleRef = useRef<TurnHandle | null>(null);
  const conversationsRef = useRef<Conversation[]>(conversations);

  useEffect(() => saveConversations(conversations), [conversations]);
  useEffect(() => {
    conversationsRef.current = conversations;
  }, [conversations]);

  // On unmount, cancel any in-flight turn and persist a terminal status for
  // whatever message it was streaming into, so a reload never finds a
  // conversation stuck showing "queued"/"thinking"/"streaming"/"tools".
  useEffect(() => {
    return () => {
      handleRef.current?.cancel();
      const terminal = conversationsRef.current.map((conversation) => ({
        ...conversation,
        messages: conversation.messages.map((message) =>
          message.role === "assistant" && ["queued", "thinking", "streaming", "tools"].includes(message.status)
            ? { ...message, status: "cancelled" as const, stats: { ...message.stats, finishedAt: Date.now() } }
            : message,
        ),
      }));
      saveConversations(terminal);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (requestedModel && requestedModel !== settings.model) {
      onSettingsChange({ ...settings, model: requestedModel });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestedModel]);

  const active = conversations.find((conversation) => conversation.id === activeId) ?? null;
  const busy = active?.messages.some((message) => message.role === "assistant" && ["queued", "thinking", "streaming", "tools"].includes(message.status)) ?? false;

  const selectedMessage = useMemo<AssistantMessage | null>(() => {
    if (!active) return null;
    const assistants = active.messages.filter((message): message is AssistantMessage => message.role === "assistant");
    if (assistants.length === 0) return null;
    if (selectedMessageId) {
      const found = assistants.find((message) => message.id === selectedMessageId);
      if (found) return found;
    }
    return assistants[assistants.length - 1];
  }, [active, selectedMessageId]);

  const updateConversation = (id: string, mutate: (conversation: Conversation) => Conversation) => {
    setConversations((current) => current.map((conversation) => (conversation.id === id ? mutate(conversation) : conversation)));
  };

  const updateMessage = (conversationId: string, messageId: string, mutate: (message: AssistantMessage) => void) => {
    updateConversation(conversationId, (conversation) => ({
      ...conversation,
      updatedAt: Date.now(),
      messages: conversation.messages.map((message) => {
        if (message.id !== messageId || message.role !== "assistant") return message;
        const copy: AssistantMessage = { ...message, stats: { ...message.stats }, toolCalls: message.toolCalls.map((call) => ({ ...call })) };
        mutate(copy);
        return copy;
      }),
    }));
  };

  const ensureConversation = (): Conversation => {
    if (active) return active;
    const conversation = newConversation(settings.model);
    setConversations((current) => [conversation, ...current]);
    setActiveId(conversation.id);
    return conversation;
  };

  const send = (text: string) => {
    const conversation = ensureConversation();
    const userMessage: UserMessage = { id: newId(), role: "user", content: text, createdAt: Date.now() };
    const assistantMessage = newAssistantMessage(settings.model, settings.params);
    const history = [...conversation.messages, userMessage];

    updateConversation(conversation.id, (current) => ({
      ...current,
      model: settings.model,
      title: current.messages.length === 0 ? conversationTitle({ ...current, messages: [userMessage] }) : current.title,
      updatedAt: Date.now(),
      messages: [...current.messages, userMessage, assistantMessage],
    }));
    setSelectedMessageId(assistantMessage.id);

    handleRef.current = runTurn(settings, history, (mutate) => updateMessage(conversation.id, assistantMessage.id, mutate));
  };

  const stop = () => handleRef.current?.cancel();

  const replay = (requestBody: unknown) => {
    const conversation = ensureConversation();
    const assistantMessage = newAssistantMessage(settings.model, settings.params);

    updateConversation(conversation.id, (current) => ({
      ...current,
      updatedAt: Date.now(),
      messages: [...current.messages, assistantMessage],
    }));
    setSelectedMessageId(assistantMessage.id);

    handleRef.current = replayRequest(settings, requestBody, (mutate) => updateMessage(conversation.id, assistantMessage.id, mutate));
  };

  const confirmSavePreset = () => {
    const name = presetNameDraft?.trim();
    if (!name) return;
    const preset: Preset = { ...settings.params, id: newId(), name };
    onPresetsChange([...presets, preset]);
    onSettingsChange({ ...settings, activePresetId: preset.id });
    setPresetNameDraft(null);
  };

  const deletePreset = () => {
    if (!settings.activePresetId) return;
    onPresetsChange(presets.filter((preset) => preset.id !== settings.activePresetId));
    onSettingsChange({ ...settings, activePresetId: null });
  };

  const selectPreset = (presetId: string) => {
    if (!presetId) {
      onSettingsChange({ ...settings, activePresetId: null });
      return;
    }
    const preset = presets.find((candidate) => candidate.id === presetId);
    if (!preset) return;
    const { id: _id, name: _name, ...params } = preset;
    onSettingsChange({ ...settings, activePresetId: preset.id, params: { ...DEFAULT_PARAMS, ...params } });
  };

  const canSend = (data.health?.ok ?? false) && Boolean(settings.model);
  const detail = settings.model ? modelDetail(settings.model, data.system) : null;

  return (
    <div className="chat-page">
      <div className={`chat-sidebar ${sidebarOpen ? "" : "collapsed"}`}>
        <div className="chat-sidebar-header">
          <button
            type="button"
            className="btn hot wide"
            style={{ height: 40 }}
            onClick={() => {
              const conversation = newConversation(settings.model);
              setConversations((current) => [conversation, ...current]);
              setActiveId(conversation.id);
              setSelectedMessageId(null);
            }}
          >
            New chat
          </button>
        </div>
        <div className="chat-conversation-list">
          {conversations.map((conversation) => (
            <div key={conversation.id} className={`chat-conversation-item ${conversation.id === activeId ? "active" : ""}`} onClick={() => setActiveId(conversation.id)}>
              <span className="chat-conversation-title">{conversationTitle(conversation)}</span>
              <button
                type="button"
                className="icon-btn"
                onClick={(event) => {
                  event.stopPropagation();
                  setConversations((current) => current.filter((candidate) => candidate.id !== conversation.id));
                  if (activeId === conversation.id) setActiveId(null);
                }}
                title="Delete conversation"
              >
                ×
              </button>
            </div>
          ))}
          {conversations.length === 0 && <div className="muted small pad">No conversations yet.</div>}
        </div>
        <div className="chat-sidebar-presets">
          <span className="meta">Preset</span>
          <select value={settings.activePresetId ?? ""} onChange={(event) => selectPreset(event.target.value)}>
            <option value="">No preset</option>
            {presets.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.name}
              </option>
            ))}
          </select>
          {presetNameDraft === null ? (
            <button type="button" className="btn small" onClick={() => setPresetNameDraft("")}>
              Save as preset
            </button>
          ) : (
            <div className="preset-save-form">
              <input
                autoFocus
                type="text"
                placeholder="Preset name"
                value={presetNameDraft}
                onChange={(event) => setPresetNameDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") confirmSavePreset();
                  if (event.key === "Escape") setPresetNameDraft(null);
                }}
              />
              <button type="button" className="btn small primary" onClick={confirmSavePreset} disabled={!presetNameDraft.trim()}>
                Save
              </button>
              <button type="button" className="btn small ghost" onClick={() => setPresetNameDraft(null)}>
                Cancel
              </button>
            </div>
          )}
          <button type="button" className="btn small ghost" onClick={deletePreset} disabled={!settings.activePresetId}>
            Delete preset
          </button>
        </div>
      </div>

      <div className="chat-main">
        <div className="chat-topbar">
          <button type="button" className="icon-btn" onClick={() => setSidebarOpen((v) => !v)} title="Toggle conversation list">
            <MenuIcon />
          </button>
          <div className="chat-model-badge">
            <span className={`status-square ${data.health?.ok ? "ok" : "neutral"}`} />
            <select value={settings.model} onChange={(event) => onSettingsChange({ ...settings, model: event.target.value })} disabled={data.models.length === 0}>
              {data.models.length === 0 && <option value="">No models</option>}
              {data.models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.id}
                </option>
              ))}
            </select>
            {detail && <span className="label chat-model-detail">{detail}</span>}
          </div>
          <div className="chat-top-chips">
            <TopChips params={settings.params} onOpen={() => setParamsOpen((v) => !v)} />
          </div>
          <button type="button" className={`btn small ${paramsOpen ? "primary" : ""}`} onClick={() => setParamsOpen((v) => !v)}>
            Parameters
          </button>
        </div>

        {paramsOpen && <ParamsPanel params={settings.params} onChange={(params) => onSettingsChange({ ...settings, params })} />}

        {!data.health?.ok && <div className="notice bad">Server is not reachable: {data.health?.detail ?? "unknown"}</div>}

        <div className="messages">
          {(!active || active.messages.length === 0) && (
            <div className="empty">
              <h1>Chat with Higgs</h1>
              <p>Send a message to start a conversation. Every reply keeps a full trace of timing and raw wire data in the inspector.</p>
            </div>
          )}
          {active?.messages.map((message) => (
            <div key={message.id} onClick={() => message.role === "assistant" && setSelectedMessageId(message.id)}>
              <MessageView message={message} />
            </div>
          ))}
        </div>

        <div className="chat-composer-area">
          <ParamChips params={settings.params} />
          <Composer disabled={!canSend} busy={busy} onSend={send} onStop={stop} />
        </div>
      </div>

      {inspectorOpen && selectedMessage && (
        <RequestInspector system={data.system} message={selectedMessage} settings={settings} onReplay={replay} onCollapse={() => setInspectorOpen(false)} />
      )}
      {inspectorOpen && !selectedMessage && (
        <aside className="inspector">
          <div className="inspector-header">
            <span>Inspector</span>
            <div className="inspector-header-spacer" />
            <button type="button" className="icon-btn inspector-collapse-btn" onClick={() => setInspectorOpen(false)} title="Collapse inspector">
              <ChevronRightIcon />
            </button>
          </div>
          <div className="inspector-body">
            <div className="notice">Send a message to see request details here.</div>
          </div>
        </aside>
      )}
      {!inspectorOpen && (
        <aside className="inspector-strip">
          <button type="button" className="icon-btn inspector-collapse-btn" onClick={() => setInspectorOpen(true)} title="Show inspector">
            <ChevronLeftIcon />
          </button>
        </aside>
      )}
    </div>
  );
}
