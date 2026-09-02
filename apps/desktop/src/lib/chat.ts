import { cancelChat, connectionFrom, streamChat, type ChatChunk } from "./api";
import { formatMs, formatRate } from "./format";
import { newId } from "./storage";
import { TOOL_DEFINITIONS, executeTool } from "./tools";
import type { AssistantMessage, GenerationParams, Message, Settings, SystemInfo, ToolCall, Trace, TraceChunkKind, TraceRound } from "./types";

const MAX_TOOL_ROUNDS = 6;

interface WireMessage {
  role: string;
  content?: string | null;
  reasoning_content?: string;
  tool_calls?: Array<{ id: string; type: "function"; function: { name: string; arguments: string } }>;
  tool_call_id?: string;
}

/** Flattens the UI transcript into OpenAI-style wire messages, including tool results. */
export function toWireMessages(messages: Message[], systemPrompt: string): WireMessage[] {
  const wire: WireMessage[] = [];
  if (systemPrompt.trim()) wire.push({ role: "system", content: systemPrompt });
  for (const message of messages) {
    if (message.role === "user") {
      wire.push({ role: "user", content: message.content });
      continue;
    }
    if (message.status === "error" || message.status === "cancelled") {
      if (!message.content && message.toolCalls.length === 0) continue;
    }
    const assistant: WireMessage = { role: "assistant", content: message.content || null };
    if (message.toolCalls.length > 0) {
      assistant.tool_calls = message.toolCalls.map((call) => ({
        id: call.id,
        type: "function",
        function: { name: call.name, arguments: call.arguments },
      }));
    }
    wire.push(assistant);
    for (const call of message.toolCalls) {
      if (call.status === "done" || call.status === "error") {
        wire.push({
          role: "tool",
          tool_call_id: call.id,
          content: call.error ? JSON.stringify({ error: call.error }) : (call.result ?? ""),
        });
      }
    }
  }
  return wire;
}

export function newAssistantMessage(model: string, params: GenerationParams): AssistantMessage {
  return {
    id: newId(),
    role: "assistant",
    content: "",
    reasoning: "",
    toolCalls: [],
    status: "queued",
    stats: { startedAt: Date.now(), model, thinkingMs: 0 },
    params,
    trace: { baseUrl: "", rounds: [] },
    createdAt: Date.now(),
  };
}

export interface TurnHandle {
  cancel: () => void;
}

export type Updater = (mutate: (message: AssistantMessage) => void) => void;

/** Builds the request body for one round exactly as sent to the server. */
export function buildRequestBody(settings: Settings, wireMessages: unknown): Record<string, unknown> {
  const params = settings.params;
  const body: Record<string, unknown> = {
    model: settings.model,
    messages: wireMessages,
    stream: true,
    stream_options: { include_usage: true },
    return_progress: true,
  };
  if (params.toolsEnabled) body.tools = TOOL_DEFINITIONS;
  if (params.reasoningEffort !== "default") body.reasoning = { effort: params.reasoningEffort };
  if (params.temperature !== null) body.temperature = params.temperature;
  if (params.topP !== null) body.top_p = params.topP;
  if (params.topK !== null) body.top_k = params.topK;
  if (params.minP !== null) body.min_p = params.minP;
  if (params.repetitionPenalty !== null) body.repetition_penalty = params.repetitionPenalty;
  if (params.maxTokens !== null) body.max_tokens = params.maxTokens;
  return body;
}

export function toCurl(baseUrl: string, apiKey: string, body: unknown): string {
  const url = `${baseUrl.replace(/\/+$/, "")}/v1/chat/completions`;
  const headerParts = [`-H "Content-Type: application/json"`];
  if (apiKey) headerParts.push(`-H "Authorization: Bearer ${apiKey}"`);
  const payload = JSON.stringify(body, null, 2).replace(/'/g, "'\\''");
  return `curl -N '${url}' \\\n  ${headerParts.join(" \\\n  ")} \\\n  -d '${payload}'`;
}

export interface TraceSummary {
  ttftMs: number | null;
  prefillMs: number | null;
  decodeTokPerSec: number | null;
  promptTokPerSec: number | null;
  totalMs: number | null;
  chunkCount: number;
  avgChunkGapMs: number | null;
  longestStallMs: number | null;
  reasoningChars: number;
  contentChars: number;
  rounds: number;
}

/**
 * Completion tokens for one round: `usage.completion_tokens` when the server
 * reported it, else the number of streamed reasoning/content/tool_call
 * chunks (Higgs emits one chunk per token). Used for decode tok/s so it
 * still works before/without a usage chunk, and matches the server's
 * definition (reasoning tokens count toward decode, same as Overview).
 */
export function roundCompletionTokens(round: TraceRound): number | undefined {
  if (round.usage?.completion_tokens !== undefined) return round.usage.completion_tokens;
  const count = round.chunks.filter((chunk) => chunk.kind === "reasoning" || chunk.kind === "content" || chunk.kind === "tool_call").length;
  return count > 0 ? count : undefined;
}

export function traceSummary(trace: Trace): TraceSummary {
  const rounds = trace.rounds;
  const chunks = rounds.flatMap((round) => round.chunks);
  const last = rounds[rounds.length - 1];
  const first = rounds[0];

  let reasoningChars = 0;
  let contentChars = 0;
  for (const chunk of chunks) {
    if (chunk.kind === "reasoning") reasoningChars += chunk.chars;
    if (chunk.kind === "content") contentChars += chunk.chars;
  }

  // All of firstChunkAt / firstTokenAt / finishedAt are stored relative to
  // their own round's sentAt, matching TraceChunk.at, so they can be
  // subtracted from one another directly without re-touching sentAt.
  const ttftMs = first?.firstTokenAt ?? null;
  const prefillMs = first?.promptProgress ? first.promptProgress.time_ms : ttftMs;

  // Decode throughput is summed across every round so a multi-round tool
  // loop reports one aggregate rate instead of just the last round's.
  let decodeTokens = 0;
  let decodeMs = 0;
  for (const round of rounds) {
    const tokens = roundCompletionTokens(round);
    if (tokens === undefined) continue;
    if (round.firstTokenAt === undefined || round.finishedAt === undefined) continue;
    const span = round.finishedAt - round.firstTokenAt;
    if (span <= 0) continue;
    decodeTokens += tokens;
    decodeMs += span;
  }
  const decodeTokPerSec = decodeMs > 0 ? decodeTokens / (decodeMs / 1000) : null;

  // Uncached prompt throughput: cached tokens are essentially free, so only
  // the newly processed portion is counted against the reported time.
  const uncachedPromptTokens = first?.promptProgress ? first.promptProgress.processed - first.promptProgress.cache : 0;
  const promptTokPerSec =
    first?.promptProgress && first.promptProgress.time_ms > 0 && uncachedPromptTokens > 0
      ? uncachedPromptTokens / (first.promptProgress.time_ms / 1000)
      : null;

  const totalMs = first !== undefined && last?.finishedAt !== undefined ? last.sentAt + last.finishedAt - first.sentAt : null;

  // Gaps start from the first chunk's `at`, not 0, so the prefill time
  // before the first chunk arrives isn't counted as a mid-stream stall.
  const gaps: number[] = [];
  for (const round of rounds) {
    if (round.chunks.length === 0) continue;
    let previous = round.chunks[0].at;
    for (const chunk of round.chunks.slice(1)) {
      gaps.push(chunk.at - previous);
      previous = chunk.at;
    }
  }
  const avgChunkGapMs = gaps.length > 0 ? gaps.reduce((sum, gap) => sum + gap, 0) / gaps.length : null;
  const longestStallMs = gaps.length > 0 ? Math.max(...gaps) : null;

  return {
    ttftMs,
    prefillMs,
    decodeTokPerSec,
    promptTokPerSec,
    totalMs,
    chunkCount: chunks.length,
    avgChunkGapMs,
    longestStallMs,
    reasoningChars,
    contentChars,
    rounds: rounds.length,
  };
}

/** Approximate per-second tokens/s series for a round, scaled to match usage when known. */
export function throughputSeries(round: TraceRound): { labels: string[]; values: number[] } {
  const buckets = new Map<number, number>();
  for (const chunk of round.chunks) {
    if (chunk.kind !== "content" && chunk.kind !== "reasoning") continue;
    const second = Math.floor(chunk.at / 1000);
    buckets.set(second, (buckets.get(second) ?? 0) + chunk.chars / 4);
  }
  const maxSecond = Math.max(0, ...buckets.keys());
  const values: number[] = [];
  for (let second = 0; second <= maxSecond; second += 1) values.push(buckets.get(second) ?? 0);

  const completionTokens = round.usage?.completion_tokens;
  if (completionTokens !== undefined) {
    const approxTotal = values.reduce((sum, v) => sum + v, 0);
    if (approxTotal > 0) {
      const scale = completionTokens / approxTotal;
      for (let i = 0; i < values.length; i += 1) values[i] *= scale;
    }
  }

  return { labels: values.map((_, i) => `${i}s`), values };
}

/**
 * Appends by rebuilding the array rather than pushing in place. React 18
 * StrictMode invokes a `setState` updater function twice per commit to
 * surface impure updaters; mutating an array reachable from the previous
 * state would apply the same push twice and duplicate every chunk.
 */
/** Chunks kept per round; a long generation beyond this drops the oldest raw chunks. */
const MAX_TRACE_CHUNKS = 4000;

function pushChunk(round: TraceRound, kind: TraceChunkKind, chars: number, raw: unknown, at: number): void {
  const next = [...round.chunks, { at: at - round.sentAt, kind, chars, raw }];
  round.chunks = next.length > MAX_TRACE_CHUNKS ? next.slice(next.length - MAX_TRACE_CHUNKS) : next;
}

/**
 * Runs one assistant turn: streams a completion, executes any tool calls
 * locally, and re-invokes the model with the results until it stops calling
 * tools. `history` must already contain the user message; `update` mutates
 * the single assistant message the UI shows for the whole loop.
 */
export function runTurn(settings: Settings, history: Message[], update: Updater): TurnHandle {
  let currentRequestId: string | null = null;
  let cancelled = false;

  const handle: TurnHandle = {
    cancel: () => {
      cancelled = true;
      if (currentRequestId) void cancelChat(currentRequestId);
    },
  };

  update((message) => {
    if (message.trace) message.trace = { ...message.trace, baseUrl: settings.baseUrl };
  });

  void (async () => {
    // Each round appends a fresh assistant turn to the wire history; the UI
    // collapses them into one message with sequential tool calls.
    const rounds: AssistantMessage[] = [];
    try {
      for (let round = 0; round < MAX_TOOL_ROUNDS; round += 1) {
        if (cancelled) break;
        const roundMessage = newAssistantMessage(settings.model, settings.params);
        rounds.push(roundMessage);
        const wire = toWireMessages([...history, ...rounds.slice(0, -1)], settings.params.systemPrompt);
        currentRequestId = newId();
        const result = await streamRound(currentRequestId, round, settings, wire, roundMessage, update);
        currentRequestId = null;
        if (result === "cancelled") {
          cancelled = true;
          break;
        }
        const pending = roundMessage.toolCalls.filter((call) => call.status === "streaming");
        if (pending.length === 0) break;

        update((message) => {
          message.status = "tools";
        });
        const toolExecutionStart = Date.now();
        update((message) => {
          const traceRound = message.trace?.rounds[round];
          if (traceRound)
            message.trace = {
              ...message.trace!,
              rounds: replaceRound(message.trace!.rounds, round, {
                ...traceRound,
                toolExecution: { startedAt: toolExecutionStart - traceRound.sentAt },
              }),
            };
        });
        for (const call of pending) {
          if (cancelled) break;
          await executeCall(call, update);
        }
        update((message) => {
          const traceRound = message.trace?.rounds[round];
          if (traceRound?.toolExecution) {
            message.trace = {
              ...message.trace!,
              rounds: replaceRound(message.trace!.rounds, round, {
                ...traceRound,
                toolExecution: { ...traceRound.toolExecution, finishedAt: Date.now() - traceRound.sentAt },
              }),
            };
          }
        });
      }
      update((message) => {
        if (message.status !== "error") message.status = cancelled ? "cancelled" : "done";
        message.stats.finishedAt = Date.now();
        closeThinkingSegment(message, message.stats.finishedAt);
      });
    } catch (error) {
      update((message) => {
        message.status = "error";
        message.error = (error as Error).message ?? String(error);
        message.stats.finishedAt = Date.now();
      });
    }
  })();

  return handle;
}

/**
 * Re-sends exactly the recorded request body from a past round (model,
 * parameters, message history included) with no tool-execution loop, and
 * appends the result as a new assistant message. Used by the inspector's
 * Replay action so a trace's model/parameters are honored verbatim, unlike
 * `runTurn` which re-derives the request from the live conversation and
 * current settings.
 */
export function replayRequest(settings: Settings, requestBody: unknown, update: Updater): TurnHandle {
  let currentRequestId: string | null = null;
  let cancelled = false;

  const handle: TurnHandle = {
    cancel: () => {
      cancelled = true;
      if (currentRequestId) void cancelChat(currentRequestId);
    },
  };

  const body = requestBody as Record<string, unknown>;
  const model = typeof body.model === "string" ? body.model : settings.model;
  const roundMessage = newAssistantMessage(model, settings.params);

  void (async () => {
    try {
      if (cancelled) return;
      currentRequestId = newId();
      const result = await streamRound(currentRequestId, 0, settings, { body }, roundMessage, update);
      currentRequestId = null;
      update((message) => {
        if (message.status !== "error") message.status = result === "cancelled" ? "cancelled" : "done";
        message.stats.finishedAt = Date.now();
        closeThinkingSegment(message, message.stats.finishedAt);
      });
    } catch (error) {
      update((message) => {
        message.status = "error";
        message.error = (error as Error).message ?? String(error);
        message.stats.finishedAt = Date.now();
      });
    }
  })();

  return handle;
}

async function executeCall(call: ToolCall, update: Updater): Promise<void> {
  const mark = (mutate: (target: ToolCall) => void) => {
    mutate(call);
    update((message) => {
      const target = message.toolCalls.find((candidate) => candidate.id === call.id);
      if (target) mutate(target);
    });
  };
  mark((target) => {
    target.status = "running";
    target.startedAt = Date.now();
  });
  try {
    const result = await executeTool(call.name, call.arguments);
    mark((target) => {
      target.status = "done";
      target.result = result;
      target.finishedAt = Date.now();
    });
  } catch (error) {
    mark((target) => {
      target.status = "error";
      target.error = (error as Error).message ?? String(error);
      target.finishedAt = Date.now();
    });
  }
}

function replaceRound(rounds: TraceRound[], index: number, next: TraceRound): TraceRound[] {
  const copy = rounds.slice();
  copy[index] = next;
  return copy;
}

function streamRound(
  requestId: string,
  roundIndex: number,
  settings: Settings,
  wire: WireMessage[] | { body: Record<string, unknown> },
  roundMessage: AssistantMessage,
  update: Updater,
): Promise<"done" | "cancelled"> {
  const body = Array.isArray(wire) ? buildRequestBody(settings, wire) : wire.body;
  const sentAt = Date.now();
  const traceRound: TraceRound = {
    index: roundIndex,
    requestBody: body,
    sentAt,
    status: "pending",
    chunks: [],
  };

  update((message) => {
    message.status = "queued";
    const rounds = message.trace ? [...message.trace.rounds, traceRound] : [traceRound];
    message.trace = { baseUrl: settings.baseUrl, rounds };
  });

  return new Promise((resolve, reject) => {
    const connection = connectionFrom(settings);
    let settled = false;
    streamChat(requestId, connection, body, (event) => {
      const elapsed = Date.now() - sentAt;
      switch (event.type) {
        case "chunk":
          applyChunk(event.data, roundIndex, sentAt, roundMessage, update);
          break;
        case "done":
          settled = true;
          update((message) => {
            updateRound(message, roundIndex, (round) => {
              round.status = "done";
              round.finishedAt ??= elapsed;
            });
          });
          resolve("done");
          break;
        case "cancelled":
          settled = true;
          update((message) => {
            updateRound(message, roundIndex, (round) => {
              round.status = "cancelled";
              round.finishedAt ??= elapsed;
            });
          });
          resolve("cancelled");
          break;
        case "error":
          settled = true;
          update((message) => {
            updateRound(message, roundIndex, (round) => {
              round.status = "error";
              round.error = event.message;
              round.finishedAt ??= elapsed;
            });
          });
          reject(new Error(event.message));
          break;
      }
    }).catch((error) => {
      if (!settled) reject(error instanceof Error ? error : new Error(String(error)));
    });
  });
}

function updateRound(message: AssistantMessage, roundIndex: number, mutate: (round: TraceRound) => void): void {
  if (!message.trace) return;
  const round = message.trace.rounds[roundIndex];
  if (!round) return;
  // Copy `chunks` too, not just the round object, so `mutate` never touches
  // an array shared with the previous state (see pushChunk).
  const next = { ...round, chunks: round.chunks.slice() };
  mutate(next);
  message.trace = { ...message.trace, rounds: replaceRound(message.trace.rounds, roundIndex, next) };
}

function closeThinkingSegment(message: AssistantMessage, now: number): void {
  if (message.stats.thinkingStartedAt !== undefined) {
    message.stats.thinkingMs += now - message.stats.thinkingStartedAt;
    message.stats.thinkingStartedAt = undefined;
  }
}

function applyChunk(chunk: ChatChunk, roundIndex: number, sentAt: number, roundMessage: AssistantMessage, update: Updater): void {
  const now = Date.now();

  const recordChunk = (kind: TraceChunkKind, chars: number) => {
    update((message) => {
      updateRound(message, roundIndex, (round) => {
        round.firstChunkAt ??= now - round.sentAt;
        pushChunk(round, kind, chars, chunk, now);
      });
    });
  };

  if (chunk.prompt_progress) {
    const progress = chunk.prompt_progress;
    update((message) => {
      message.stats.promptProgress = progress;
      updateRound(message, roundIndex, (round) => {
        round.promptProgress = progress;
      });
    });
    recordChunk("progress", 0);
  }
  if (chunk.usage) {
    const usage = chunk.usage;
    update((message) => {
      message.stats.usage = usage;
      updateRound(message, roundIndex, (round) => {
        round.usage = usage;
      });
    });
    recordChunk("usage", 0);
  }
  const choices = chunk.choices ?? [];
  if (choices.length === 0 && !chunk.prompt_progress && !chunk.usage) {
    recordChunk("other", 0);
  }
  for (const choice of choices) {
    const delta = choice.delta ?? {};
    if (delta.role) recordChunk("role", 0);
    if (delta.reasoning_content) {
      const text = delta.reasoning_content;
      const separator = roundMessage.reasoning === "" ? "\n\n" : "";
      roundMessage.reasoning += text;
      update((message) => {
        message.reasoning += (message.reasoning ? separator : "") + text;
        message.status = "thinking";
        message.stats.firstTokenAt ??= now;
        message.stats.thinkingStartedAt ??= now;
        updateRound(message, roundIndex, (round) => {
          round.firstTokenAt ??= now - sentAt;
        });
      });
      recordChunk("reasoning", text.length);
    }
    if (delta.content) {
      const text = delta.content;
      roundMessage.content += text;
      update((message) => {
        message.content += text;
        message.status = "streaming";
        message.stats.firstTokenAt ??= now;
        closeThinkingSegment(message, now);
        updateRound(message, roundIndex, (round) => {
          round.firstTokenAt ??= now - sentAt;
        });
      });
      recordChunk("content", text.length);
    }
    if (delta.tool_calls && delta.tool_calls.length > 0) {
      for (const toolDelta of delta.tool_calls) {
        mergeToolDelta(roundMessage, toolDelta);
      }
      update((message) => {
        message.status = "streaming";
        message.stats.firstTokenAt ??= now;
        closeThinkingSegment(message, now);
        updateRound(message, roundIndex, (round) => {
          round.firstTokenAt ??= now - sentAt;
        });
        // Tool calls carry ids, so the merged round state can be copied over
        // instead of re-deriving the delta merge on the UI message.
        for (const call of roundMessage.toolCalls) {
          const existing = message.toolCalls.find((candidate) => candidate.id === call.id);
          if (existing) {
            existing.name = call.name;
            existing.arguments = call.arguments;
          } else {
            message.toolCalls.push({ ...call });
          }
        }
      });
      recordChunk(
        "tool_call",
        delta.tool_calls.reduce((sum, d) => sum + (d.function?.arguments?.length ?? 0), 0),
      );
    }
    if (choice.finish_reason) {
      const reason = choice.finish_reason;
      update((message) => {
        message.stats.finishReason = reason;
        closeThinkingSegment(message, now);
        updateRound(message, roundIndex, (round) => {
          round.finishReason = reason;
        });
      });
      recordChunk("finish", 0);
    }
  }
  if (chunk.model) {
    const model = chunk.model;
    update((message) => {
      message.stats.model = model;
      updateRound(message, roundIndex, (round) => {
        round.model = model;
      });
    });
  }
}

/**
 * Maps a round's streaming tool-call deltas (keyed by server-side protocol
 * index, which can arrive out of order or with gaps) to calls in
 * `toolCalls`, kept dense and ordered by that index — not by arrival order —
 * so a delta for index 1 arriving before index 0 doesn't reverse the order
 * calls are later executed in.
 */
const toolDeltaCalls = new WeakMap<AssistantMessage, Map<number, ToolCall>>();

function mergeToolDelta(roundMessage: AssistantMessage, delta: { index: number; id?: string; function?: { name?: string; arguments?: string } }): void {
  let calls = toolDeltaCalls.get(roundMessage);
  if (!calls) {
    calls = new Map();
    toolDeltaCalls.set(roundMessage, calls);
  }
  let call = calls.get(delta.index);
  if (!call) {
    // ids only arrive on the first fragment, so a placeholder id is minted
    // if the server omits it.
    call = { id: delta.id ?? `call_${newId().slice(0, 8)}`, name: "", arguments: "", status: "streaming" };
    calls.set(delta.index, call);
    const byIndex = calls;
    roundMessage.toolCalls = [...byIndex.keys()].sort((a, b) => a - b).map((index) => byIndex.get(index)!);
  }
  if (delta.function?.name) call.name += delta.function.name;
  if (delta.function?.arguments) call.arguments += delta.function.arguments;
}


function markdownNumber(value: number | null | undefined, unit = "", digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "–";
  return `${value.toFixed(digits)}${unit}`;
}

function markdownMs(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "–";
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
}

function paramSummary(params: GenerationParams | undefined): string {
  if (!params) return "defaults";
  const parts: string[] = [];
  if (params.reasoningEffort !== "default") parts.push(`reasoning=${params.reasoningEffort}`);
  if (params.temperature !== null) parts.push(`temperature=${params.temperature}`);
  if (params.topP !== null) parts.push(`top_p=${params.topP}`);
  if (params.topK !== null) parts.push(`top_k=${params.topK}`);
  if (params.minP !== null) parts.push(`min_p=${params.minP}`);
  if (params.repetitionPenalty !== null) parts.push(`repetition_penalty=${params.repetitionPenalty}`);
  if (params.maxTokens !== null) parts.push(`max_tokens=${params.maxTokens}`);
  parts.push(params.toolsEnabled ? "tools=on" : "tools=off");
  return parts.join(", ");
}

/**
 * A fixed-format Markdown report of one reply's trace, meant to be pasted
 * into issues and posts so speed numbers from different machines line up.
 */
export function traceToMarkdown(message: AssistantMessage, settings: Settings, system: SystemInfo | null): string {
  const trace = message.trace;
  if (!trace) return "No trace recorded for this message.";
  const summary = traceSummary(trace);
  const rounds = trace.rounds;
  const usage = rounds.reduce(
    (acc, round) => ({
      prompt: acc.prompt + (round.usage?.prompt_tokens ?? 0),
      completion: acc.completion + (round.usage?.completion_tokens ?? 0),
    }),
    { prompt: 0, completion: 0 },
  );
  const cached = rounds.reduce((sum, round) => sum + (round.promptProgress?.cache ?? 0), 0);
  const model = message.stats.model ?? settings.model;
  const memory = system?.memory;
  const loaded = system?.models.find((entry) => entry.name === model);

  const lines: string[] = [];
  lines.push(`## Higgs request trace`);
  lines.push("");
  lines.push(`| | |`);
  lines.push(`|---|---|`);
  lines.push(`| Model | \`${model}\`${loaded?.path && loaded.path !== model ? ` (\`${loaded.path}\`)` : ""} |`);
  lines.push(`| Engine | ${loaded ? `${loaded.engine}${loaded.mlx_profile ? `, profile ${loaded.mlx_profile}` : ""}${loaded.kv_cache ? `, kv_cache ${loaded.kv_cache}` : ""}` : "–"} |`);
  lines.push(`| Higgs | ${system ? `v${system.version}` : "–"} |`);
  lines.push(`| Machine | ${memory?.physical_total_bytes ? `${(memory.physical_total_bytes / 1024 ** 3).toFixed(0)} GB unified memory` : "–"}${memory?.mlx_active_bytes ? `, MLX active ${(memory.mlx_active_bytes / 1024 ** 3).toFixed(1)} GB` : ""} |`);
  lines.push(`| Parameters | ${paramSummary(message.params)} |`);
  lines.push(`| Date | ${new Date(message.createdAt).toISOString()} |`);
  lines.push("");
  lines.push(`### Speed`);
  lines.push("");
  lines.push(`| Metric | Value |`);
  lines.push(`|---|---|`);
  lines.push(`| Time to first token | ${markdownMs(summary.ttftMs)} |`);
  lines.push(`| Prefill | ${markdownMs(summary.prefillMs)}${summary.promptTokPerSec !== null ? ` (${markdownNumber(summary.promptTokPerSec, " tok/s")} uncached)` : ""} |`);
  lines.push(`| Decode | ${markdownNumber(summary.decodeTokPerSec, " tok/s", 1)} |`);
  lines.push(`| Total time | ${markdownMs(summary.totalMs)} |`);
  lines.push(`| Longest stall | ${markdownMs(summary.longestStallMs)} |`);
  lines.push("");
  lines.push(`### Tokens`);
  lines.push("");
  lines.push(`| Metric | Value |`);
  lines.push(`|---|---|`);
  lines.push(`| Prompt tokens | ${usage.prompt}${cached > 0 ? ` (${cached} from prefix cache)` : ""} |`);
  lines.push(`| Completion tokens | ${usage.completion} |`);
  lines.push(`| Reasoning / visible characters | ${summary.reasoningChars} / ${summary.contentChars} |`);
  lines.push(`| Rounds (tool calls) | ${summary.rounds} |`);
  lines.push(`| Finish reason | ${message.stats.finishReason ?? "–"} |`);
  if (rounds.length > 1) {
    lines.push("");
    lines.push(`### Rounds`);
    lines.push("");
    lines.push(`| Round | TTFT | Duration | Prompt | Completion | Cached | Decode tok/s |`);
    lines.push(`|---|---|---|---|---|---|---|`);
    for (const round of rounds) {
      const decode = round.finishedAt !== undefined && round.firstTokenAt !== undefined ? round.finishedAt - round.firstTokenAt : null;
      const tokens = round.usage?.completion_tokens ?? null;
      const decodeTokens = roundCompletionTokens(round);
      const rate = decode !== null && decode > 0 && decodeTokens !== undefined ? decodeTokens / (decode / 1000) : null;
      lines.push(
        `| ${round.index + 1} | ${markdownMs(round.firstTokenAt)} | ${markdownMs(round.finishedAt)} | ${round.usage?.prompt_tokens ?? "–"} | ${tokens ?? "–"} | ${round.promptProgress?.cache ?? "–"} | ${markdownNumber(rate, "", 1)} |`,
      );
    }
  }
  lines.push("");
  lines.push(`<sub>Generated by Higgs Desktop. TTFT includes prefill; decode tok/s = completion tokens / time after first token.</sub>`);
  return lines.join("\n");
}

/* -------------------------------------------------------------------------
 * Trace waterfall: the spans shown in the inspector's Trace tab.
 * ---------------------------------------------------------------------- */

export type SpanColor = "neutral" | "ok" | "accent" | "reasoning" | "dim";

/** One row of the trace waterfall. `start`/`end`/`stalls` are ms offsets relative to the trace's overall start (the first round's `sentAt`). */
export interface TraceSpan {
  id: string;
  label: string;
  indent: number;
  start: number;
  end: number;
  color: SpanColor;
  detail: string;
  /** Offsets (relative to overall start) of any gap > 1s inside this span. */
  stalls: number[];
}

/** Scope for the inspector: the whole request, or a single round by index. */
export type InspectorScope = "all" | number;

/** Estimated prefill time avoided by the prefix cache, derived from the uncached-token rate of this same round. */
export function estimatePrefillSavedMs(round: TraceRound): number | null {
  const p = round.promptProgress;
  if (!p) return null;
  const uncached = p.processed - p.cache;
  if (p.cache <= 0 || uncached <= 0 || p.time_ms <= 0) return null;
  const uncachedRate = uncached / (p.time_ms / 1000);
  return uncachedRate > 0 ? (p.cache / uncachedRate) * 1000 : null;
}

/** "Prefill · 447 tok" / "Prefill · 614 tok · 442 cached (72%)" — the cache portion is only added when something was actually cached, and only the label's own detail panel spells out a 0% cache hit. Falls back to usage.prompt_tokens when progress is missing or reported as 0. */
function prefillLabel(round: TraceRound): string {
  const p = round.promptProgress;
  const tokens = p && p.processed > 0 ? p.processed : (p?.total ?? round.usage?.prompt_tokens);
  if (tokens === undefined) return "Prefill";
  const cached = p?.cache ?? 0;
  if (cached <= 0) return `Prefill · ${tokens} tok`;
  const pct = p && p.total > 0 ? Math.round((cached / p.total) * 100) : null;
  return `Prefill · ${tokens} tok · ${cached} cached${pct !== null ? ` (${pct}%)` : ""}`;
}

function prefillDetail(round: TraceRound): string {
  const p = round.promptProgress;
  if (!p) return "No prompt-progress data was recorded for this round.";
  const pct = p.total > 0 ? Math.round((p.cache / p.total) * 100) : 0;
  const estSavedMs = estimatePrefillSavedMs(round);
  if (p.cache > 0) {
    return `${p.processed} prompt tokens, ${p.cache} served from the prefix cache (${pct}%)${
      estSavedMs !== null ? `, saving roughly ${formatMs(estSavedMs)} of prefill.` : "."
    }`;
  }
  return `${p.processed} prompt tokens, none served from the prefix cache.`;
}

/**
 * "≈ N tok/s over this span." — a per-span rate for the waterfall's Thinking
 * and Generation/Tool call output rows. Higgs emits one chunk per token, so
 * the chunk count inside the span (not the round's total usage, which would
 * attribute reasoning-phase tokens to the generation span too) is the token
 * count.
 */
function spanRateSuffix(tokenCount: number, durationMs: number): string {
  return tokenCount > 0 && durationMs > 0 ? ` ≈ ${formatRate(tokenCount / (durationMs / 1000))} over this span.` : "";
}

function generationDetail(tokenCount: number, durationMs: number, stallCount: number): string {
  const parts: string[] = [stallCount > 0 ? `${stallCount} stall${stallCount > 1 ? "s" : ""} detected` : "no stalls detected"];
  return `${parts.join(", ")}.${spanRateSuffix(tokenCount, durationMs)}`;
}

/** Distinct tool_calls[].index values seen across this round's tool_call chunks — the number of calls being assembled, not a token count. */
function toolCallCountInRound(round: TraceRound): number {
  const indices = new Set<number>();
  for (const chunk of round.chunks) {
    if (chunk.kind !== "tool_call") continue;
    const raw = chunk.raw as { choices?: Array<{ delta?: { tool_calls?: Array<{ index?: number }> } }> };
    for (const delta of raw.choices?.[0]?.delta?.tool_calls ?? []) {
      if (typeof delta.index === "number") indices.add(delta.index);
    }
  }
  return indices.size;
}

function roundDetail(round: TraceRound, index: number): string {
  const prompt = round.usage?.prompt_tokens;
  const completion = round.usage?.completion_tokens;
  return `Model call ${index + 1}${prompt !== undefined ? `, ${prompt} prompt tokens` : ""}${
    completion !== undefined ? `, ${completion} completion tokens` : ""
  }${round.finishReason ? `, finish_reason ${round.finishReason}` : ""}.`;
}

/**
 * Derives the full waterfall (spans in ms relative to the first round's
 * `sentAt`) from the raw trace. The inspector clips this list to whatever
 * scope window is selected rather than rebuilding it per scope.
 */
export function buildTraceSpans(trace: Trace): TraceSpan[] {
  const rounds = trace.rounds;
  if (rounds.length === 0) return [];
  const overallStart = rounds[0].sentAt;
  const spans: TraceSpan[] = [];

  const first = rounds[0];
  const firstChunkOffset = first.firstChunkAt ?? 0;
  spans.push({
    id: "accepted",
    label: "Request accepted",
    indent: 0,
    start: 0,
    end: firstChunkOffset,
    color: "neutral",
    detail: "Time between the request being sent and the first byte of the response.",
    stalls: [],
  });

  rounds.forEach((round, index) => {
    const roundStart = round.sentAt - overallStart;
    const roundDur = round.finishedAt ?? Date.now() - round.sentAt;
    const roundEnd = roundStart + roundDur;
    const chunks = round.chunks;

    spans.push({
      id: `round-${index}`,
      label: `Round ${index + 1}`,
      indent: 0,
      start: roundStart,
      end: roundEnd,
      color: "dim",
      detail: roundDetail(round, index),
      stalls: [],
    });

    const prefillEndOffset = round.firstTokenAt ?? round.firstChunkAt ?? roundDur;
    const cache = round.promptProgress?.cache ?? 0;
    spans.push({
      id: `round-${index}-prefill`,
      label: prefillLabel(round),
      indent: 1,
      start: roundStart,
      end: roundStart + prefillEndOffset,
      color: cache > 0 ? "ok" : "neutral",
      detail: prefillDetail(round),
      stalls: [],
    });

    const reasoningChunks = chunks.filter((chunk) => chunk.kind === "reasoning");
    const firstGenChunk = chunks.find((chunk) => chunk.kind === "content" || chunk.kind === "tool_call");
    if (reasoningChunks.length > 0) {
      const cutoff = firstGenChunk?.at ?? Infinity;
      const relevant = reasoningChunks.filter((chunk) => chunk.at <= cutoff);
      if (relevant.length > 0) {
        const thinkStart = relevant[0].at;
        const thinkEnd = relevant[relevant.length - 1].at;
        spans.push({
          id: `round-${index}-thinking`,
          label: `Thinking · ${formatMs(thinkEnd - thinkStart)}`,
          indent: 1,
          start: roundStart + thinkStart,
          end: roundStart + thinkEnd,
          color: "reasoning",
          detail: `${relevant.length} reasoning chunk${relevant.length > 1 ? "s" : ""} before the visible answer began.${spanRateSuffix(relevant.length, thinkEnd - thinkStart)}`,
          stalls: [],
        });
      }
    }

    if (firstGenChunk) {
      const genStart = firstGenChunk.at;
      const genEnd = roundDur;
      const stalls: number[] = [];
      let previous = genStart;
      for (const chunk of chunks) {
        if (chunk.at < genStart) continue;
        if (chunk.at - previous > 1000) stalls.push(roundStart + chunk.at);
        previous = chunk.at;
      }
      const isToolRound = round.finishReason === "tool_calls";
      let label: string;
      let detail: string;
      if (isToolRound) {
        // The chunk count here is tool_call deltas, not tokens — Higgs may
        // emit several chunks per call as arguments stream in, so it must
        // not be reported as a token count or fed into a tok/s rate.
        const callCount = toolCallCountInRound(round);
        label = callCount > 0 ? `Tool call output · ${callCount} call${callCount > 1 ? "s" : ""}` : "Tool call output";
        const stallText = stalls.length > 0 ? `${stalls.length} stall${stalls.length > 1 ? "s" : ""} detected.` : "no stalls detected.";
        detail = `Tool call JSON, ${callCount} call${callCount === 1 ? "" : "s"}. ${stallText}`;
      } else {
        // Token count for this span only: the number of content chunks
        // streamed inside it, not the round's whole usage figure (which
        // would also fold in reasoning-phase tokens and inflate the rate).
        const genTokenCount = chunks.filter((chunk) => chunk.at >= genStart && chunk.kind === "content").length;
        label = genTokenCount > 0 ? `Generation · ${genTokenCount} tokens` : "Generation";
        detail = generationDetail(genTokenCount, genEnd - genStart, stalls.length);
      }
      spans.push({
        id: `round-${index}-generation`,
        label,
        indent: 1,
        start: roundStart + genStart,
        end: roundStart + genEnd,
        color: "accent",
        detail,
        stalls,
      });
    }

    if (round.finishReason === "tool_calls") {
      spans.push({
        id: `round-${index}-toolcalls`,
        label: "Tool calls emitted",
        indent: 1,
        start: Math.max(roundStart, roundEnd - Math.min(120, roundDur)),
        end: roundEnd,
        color: "accent",
        detail: "The round ended with finish_reason tool_calls; the requested functions were parsed from the stream.",
        stalls: [],
      });
    }

    const next = rounds[index + 1];
    if (next) {
      const toolStartOffset = round.toolExecution?.startedAt ?? roundDur;
      const toolEndAbs = round.toolExecution?.finishedAt !== undefined ? roundStart + round.toolExecution.finishedAt : next.sentAt - overallStart;
      spans.push({
        id: `tools-${index}`,
        label: "Tools executed locally",
        indent: 0,
        start: roundStart + toolStartOffset,
        end: toolEndAbs,
        color: "ok",
        detail: "Local tool execution between this round and the next model call.",
        stalls: [],
      });
    }
  });

  const last = rounds[rounds.length - 1];
  const lastRoundStart = last.sentAt - overallStart;
  const lastChunkAt = last.chunks.length > 0 ? last.chunks[last.chunks.length - 1].at : 0;
  const lastEnd = lastRoundStart + (last.finishedAt ?? Date.now() - last.sentAt);
  spans.push({
    id: "closed",
    label: "Stream closed",
    indent: 0,
    start: lastRoundStart + lastChunkAt,
    end: lastEnd,
    color: "neutral",
    detail: "Final usage chunk received and the stream terminated.",
    stalls: [],
  });

  return spans;
}

/** Reasoning/content character counts for a single round (traceSummary sums these across every round). */
export function roundChars(round: TraceRound): { reasoningChars: number; contentChars: number } {
  let reasoningChars = 0;
  let contentChars = 0;
  for (const chunk of round.chunks) {
    if (chunk.kind === "reasoning") reasoningChars += chunk.chars;
    if (chunk.kind === "content") contentChars += chunk.chars;
  }
  return { reasoningChars, contentChars };
}

/** The ms window (relative to overall start) that a scope covers. */
export function traceScopeWindow(trace: Trace, scope: InspectorScope): { from: number; to: number } {
  const rounds = trace.rounds;
  if (rounds.length === 0) return { from: 0, to: 0 };
  const overallStart = rounds[0].sentAt;
  if (scope === "all") {
    const last = rounds[rounds.length - 1];
    const lastStart = last.sentAt - overallStart;
    return { from: 0, to: lastStart + (last.finishedAt ?? Date.now() - last.sentAt) };
  }
  const round = rounds[scope];
  if (!round) return { from: 0, to: 0 };
  const start = round.sentAt - overallStart;
  return { from: start, to: start + (round.finishedAt ?? Date.now() - round.sentAt) };
}
