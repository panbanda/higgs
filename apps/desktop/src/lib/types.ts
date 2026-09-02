export type ReasoningEffort = "default" | "none" | "low" | "medium" | "high";

/** Sampling and prompt settings for a chat turn; stored as presets. */
export interface GenerationParams {
  systemPrompt: string;
  reasoningEffort: ReasoningEffort;
  temperature: number | null;
  topP: number | null;
  topK: number | null;
  minP: number | null;
  repetitionPenalty: number | null;
  maxTokens: number | null;
  toolsEnabled: boolean;
}

export interface Preset extends GenerationParams {
  id: string;
  name: string;
}

export const DEFAULT_PARAMS: GenerationParams = {
  systemPrompt: "",
  reasoningEffort: "default",
  temperature: null,
  topP: null,
  topK: null,
  minP: null,
  repetitionPenalty: null,
  maxTokens: null,
  toolsEnabled: true,
};

export interface Settings {
  baseUrl: string;
  apiKey: string;
  model: string;
  /** Config profile name; null means the default config.toml. */
  profile: string | null;
  /** Explicit path to the `higgs` binary; empty resolves via the login shell. */
  higgsBinary: string;
  refreshSeconds: number;
  /** Minutes of request history shown in dashboard views. */
  windowMinutes: number;
  activePresetId: string | null;
  params: GenerationParams;
  /** Hugging Face token, needed only for gated repos in the Hub view. */
  hfToken: string;
}

export const DEFAULT_SETTINGS: Settings = {
  baseUrl: "http://127.0.0.1:8000",
  apiKey: "",
  model: "",
  profile: null,
  higgsBinary: "",
  refreshSeconds: 3,
  windowMinutes: 60,
  activePresetId: null,
  params: DEFAULT_PARAMS,
  hfToken: "",
};

export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
  result?: string;
  error?: string;
  status: "streaming" | "running" | "done" | "error";
  startedAt?: number;
  finishedAt?: number;
}

export interface Usage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

export interface PromptProgress {
  total: number;
  cache: number;
  processed: number;
  time_ms: number;
}

export interface GenerationStats {
  startedAt: number;
  firstTokenAt?: number;
  finishedAt?: number;
  /** Start of the current thinking segment; cleared each tool round. */
  thinkingStartedAt?: number;
  /** Accumulated thinking time across completed segments. */
  thinkingMs: number;
  usage?: Usage;
  promptProgress?: PromptProgress;
  finishReason?: string;
  model?: string;
}

export type TraceChunkKind = "role" | "reasoning" | "content" | "tool_call" | "progress" | "usage" | "finish" | "other";

/** One SSE chunk as received, with arrival time relative to the round's request. */
export interface TraceChunk {
  at: number;
  kind: TraceChunkKind;
  /** Characters of text carried by the delta, for throughput estimates. */
  chars: number;
  raw: unknown;
}

/** One model call inside a turn; a turn has several rounds when tools run. */
export interface TraceRound {
  index: number;
  requestBody: unknown;
  sentAt: number;
  firstChunkAt?: number;
  firstTokenAt?: number;
  finishedAt?: number;
  status: "pending" | "done" | "error" | "cancelled";
  error?: string;
  chunks: TraceChunk[];
  usage?: Usage;
  promptProgress?: PromptProgress;
  finishReason?: string;
  model?: string;
  toolExecution?: { startedAt: number; finishedAt?: number };
}

export interface Trace {
  baseUrl: string;
  rounds: TraceRound[];
}

export interface UserMessage {
  id: string;
  role: "user";
  content: string;
  createdAt: number;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  content: string;
  reasoning: string;
  toolCalls: ToolCall[];
  status: "queued" | "thinking" | "streaming" | "tools" | "done" | "error" | "cancelled";
  error?: string;
  stats: GenerationStats;
  trace?: Trace;
  /** Parameters used for the turn, shown as chips and used for replay. */
  params?: GenerationParams;
  createdAt: number;
}

export type Message = UserMessage | AssistantMessage;

export interface Conversation {
  id: string;
  title: string;
  model: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

export interface ModelInfo {
  id: string;
  owned_by: string;
}

export interface HealthStatus {
  ok: boolean;
  detail: string;
}

/** Shape of GET /metrics (crates/higgs/src/routes/metrics.rs). */
export interface MetricsGroup {
  name: string;
  requests: number;
  input_tokens: number;
  output_tokens: number;
  avg_ms: number;
  p50_ms: number;
  p95_ms: number;
  errors: number;
  ttft_p50_ms: number | null;
  ttft_p95_ms: number | null;
  tokens_per_second: number | null;
  cached_tokens: number;
}

export interface LatencySummary {
  samples: number;
  avg_ms: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
}

export interface Metrics {
  window_minutes: number;
  totals: { requests: number; input_tokens: number; output_tokens: number; errors: number };
  latency: LatencySummary;
  ttft: LatencySummary;
  tokens_per_second: number | null;
  status_counts: Record<string, number>;
  requests_per_minute: number[];
  tokens_per_minute: number[];
  models: MetricsGroup[];
  providers: MetricsGroup[];
}

/** Shape of GET /v1/system (crates/higgs/src/routes/system.rs). */
export interface SystemInfo {
  version: string;
  pid: number;
  uptime_secs: number;
  memory: {
    physical_total_bytes: number | null;
    process_rss_bytes: number | null;
    mlx_active_bytes: number | null;
    mlx_peak_bytes: number | null;
    mlx_cache_bytes: number | null;
  };
  models: Array<{
    name: string;
    path: string | null;
    engine: "simple" | "batch" | string;
    mlx_profile: string | null;
    kv_cache: string | null;
  }>;
  metrics_enabled: boolean;
}

/** One line of the metrics JSONL log (crates/higgs/src/attach.rs LogEntry). */
export interface RequestRecord {
  timestamp: string;
  model: string | null;
  provider: string | null;
  routing_method: string | null;
  status: number;
  duration_ms: number;
  input_tokens: number;
  output_tokens: number;
  error: string | null;
  /** Time to first token; only local streaming requests record it. */
  ttft_ms?: number | null;
  /** Prompt tokens served from the prefix cache. */
  cached_tokens?: number | null;
}

export interface MetricsLog {
  path: string;
  exists: boolean;
  records: RequestRecord[];
  offset: number;
  reset: boolean;
}

export interface Profile {
  name: string | null;
  config_path: string;
}

export interface ProfileList {
  config_dir: string;
  profiles: Profile[];
}

export interface ConfigFile {
  path: string;
  exists: boolean;
  raw: string;
  parsed: unknown;
  parse_error: string | null;
}

export interface DaemonStatus {
  running: boolean;
  pid: number | null;
  pid_path: string;
  log_path: string;
}

export interface CommandOutput {
  program: string;
  exit_code: number | null;
  stdout: string;
  stderr: string;
}

export interface ModelCacheInfo {
  path: string;
  cached: boolean;
  size_bytes: number;
  location: string | null;
}

/** Where a resolved `higgs` binary came from, in resolution-order
 * preference: an explicit path from Settings, `PATH`, or the copy bundled
 * with the desktop app (the dev bridge, which has no bundle, never reports
 * "bundled"). */
export type HiggsBinarySource = "settings" | "path" | "bundled" | "missing";

export interface HiggsBinaryInfo {
  path: string | null;
  source: HiggsBinarySource;
  version: string | null;
}

/** Shape of GET /api/models on huggingface.co, trimmed to what the Hub view needs. */
export interface HubModelSummary {
  id: string;
  downloads: number;
  likes: number;
  last_modified: string | null;
  tags: string[];
  gated: boolean;
}

export interface HubSibling {
  rfilename: string;
  size: number | null;
}

export interface HubModelDetail {
  id: string;
  sha: string;
  siblings: HubSibling[];
  total_bytes: number;
  config_model_type: string | null;
  quantization: string | null;
  tags: string[];
}

export type HubJobState = "idle" | "running" | "done" | "error" | "cancelled";

export interface HubDownloadStatus {
  state: HubJobState;
  file: string | null;
  file_index: number;
  file_count: number;
  bytes_done: number;
  bytes_total: number;
  total_done: number;
  total_bytes: number;
  message: string | null;
  path: string | null;
}
