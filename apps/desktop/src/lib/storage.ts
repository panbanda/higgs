import { DEFAULT_PARAMS, DEFAULT_SETTINGS, type Conversation, type Preset, type Settings } from "./types";

const SETTINGS_KEY = "higgs.settings.v2";
const CONVERSATIONS_KEY = "higgs.conversations.v1";
const PRESETS_KEY = "higgs.presets.v1";

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export function loadSettings(): Settings {
  const stored = read<Partial<Settings>>(SETTINGS_KEY, {});
  return { ...DEFAULT_SETTINGS, ...stored, params: { ...DEFAULT_PARAMS, ...(stored.params ?? {}) } };
}

export function saveSettings(settings: Settings): void {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

export function loadPresets(): Preset[] {
  return read<Preset[]>(PRESETS_KEY, []);
}

export function savePresets(presets: Preset[]): void {
  localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
}

export function loadConversations(): Conversation[] {
  const list = read<Conversation[]>(CONVERSATIONS_KEY, []);
  // A conversation persisted mid-stream must not come back looking live.
  for (const conversation of list) {
    for (const message of conversation.messages) {
      if (message.role === "assistant") {
        message.stats.thinkingMs ??= 0;
        // An open thinking segment would otherwise keep counting from load time.
        message.stats.thinkingStartedAt = undefined;
        if (!["done", "error", "cancelled"].includes(message.status)) message.status = "cancelled";
      }
    }
  }
  return list;
}

/** Conversations kept in localStorage; traces make each one sizeable. */
const MAX_PERSISTED_CONVERSATIONS = 40;

export function saveConversations(conversations: Conversation[]): void {
  const recent = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_PERSISTED_CONVERSATIONS);
  try {
    localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(recent));
  } catch {
    // Quota exceeded: drop raw chunk payloads, which dominate size, and retry.
    const slim = recent.map((conversation) => ({
      ...conversation,
      messages: conversation.messages.map((message) =>
        message.role === "assistant" && message.trace
          ? { ...message, trace: { ...message.trace, rounds: message.trace.rounds.map((round) => ({ ...round, chunks: [] })) } }
          : message,
      ),
    }));
    try {
      localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(slim));
    } catch {
      // Nothing further to shed; keep the in-memory state.
    }
  }
}

export function newId(): string {
  return crypto.randomUUID();
}
