import { traceSummary } from "../lib/chat";
import type { AssistantMessage, Message } from "../lib/types";
import { Markdown } from "./Markdown";
import { ThinkingBlock } from "./ThinkingBlock";
import { ToolCallCard } from "./ToolCallCard";

/**
 * Decode tok/s here must agree with the trace waterfall and the server's own
 * per-request figure (Overview/Requests): completion tokens / (finishedAt -
 * firstTokenAt) per round, summed across rounds. `message.stats` only ever
 * holds the latest round's usage and the overall first/last timestamps, so
 * for a multi-round tool loop it disagreed with both — traceSummary is the
 * single source of truth for this now.
 */
function StatsLine({ message }: { message: AssistantMessage }) {
  const { stats, trace } = message;
  const parts: string[] = [];
  if (trace && trace.rounds.length > 0) {
    const summary = traceSummary(trace);
    const usage = trace.rounds.reduce(
      (acc, round) => ({ prompt: acc.prompt + (round.usage?.prompt_tokens ?? 0), completion: acc.completion + (round.usage?.completion_tokens ?? 0) }),
      { prompt: 0, completion: 0 },
    );
    if (summary.decodeTokPerSec !== null) parts.push(`${summary.decodeTokPerSec.toFixed(1)} tok/s`);
    if (usage.prompt > 0) parts.push(`${usage.prompt} in`);
    if (usage.completion > 0) parts.push(`${usage.completion} out`);
    if (summary.ttftMs !== null) parts.push(`TTFT ${(summary.ttftMs / 1000).toFixed(2)}s`);
  } else {
    if (stats.usage?.completion_tokens && stats.firstTokenAt && stats.finishedAt) {
      const seconds = (stats.finishedAt - stats.firstTokenAt) / 1000;
      if (seconds > 0) parts.push(`${(stats.usage.completion_tokens / seconds).toFixed(1)} tok/s`);
    }
    if (stats.usage?.prompt_tokens !== undefined) parts.push(`${stats.usage.prompt_tokens} in`);
    if (stats.usage?.completion_tokens !== undefined) parts.push(`${stats.usage.completion_tokens} out`);
    if (stats.firstTokenAt) parts.push(`TTFT ${((stats.firstTokenAt - stats.startedAt) / 1000).toFixed(2)}s`);
  }
  if (stats.finishReason && stats.finishReason !== "stop") parts.push(`finish: ${stats.finishReason}`);
  if (stats.model) parts.push(stats.model);
  if (parts.length === 0) return null;
  return <div className="stats">{parts.join("  ·  ")}</div>;
}

function LiveStatus({ message }: { message: AssistantMessage }) {
  const progress = message.stats.promptProgress;
  if (message.status === "queued") {
    if (progress && progress.processed < progress.total) {
      const percent = Math.round((progress.processed / progress.total) * 100);
      return (
        <div className="live-status">
          <span className="spinner" /> Processing prompt {percent}% ({progress.processed}/{progress.total} tokens
          {progress.cache ? `, ${progress.cache} cached` : ""})
        </div>
      );
    }
    return (
      <div className="live-status">
        <span className="spinner" /> Waiting for model
      </div>
    );
  }
  if (message.status === "tools") {
    return (
      <div className="live-status">
        <span className="spinner" /> Running tools
      </div>
    );
  }
  return null;
}

export function MessageView({ message }: { message: Message }) {
  if (message.role === "user") {
    return (
      <div className="message user">
        <div className="bubble">{message.content}</div>
      </div>
    );
  }
  const streaming = message.status === "streaming";
  return (
    <div className="message assistant">
      <div className="avatar">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent-text)" strokeWidth="2.4" strokeLinecap="round">
          <path d="M4 14 L9 6 L13 16 L17 9 L20 14" />
        </svg>
      </div>
      <div className="assistant-body">
        <ThinkingBlock message={message} />
        {message.toolCalls.map((call) => (
          <ToolCallCard key={call.id} call={call} />
        ))}
        <LiveStatus message={message} />
        {message.content && (
          <div className={`content ${streaming ? "streaming" : ""}`}>
            <Markdown>{message.content}</Markdown>
          </div>
        )}
        {message.status === "error" && <div className="error-box">{message.error ?? "Request failed"}</div>}
        {message.status === "cancelled" && <div className="muted">Stopped.</div>}
        {message.status === "done" && <StatsLine message={message} />}
      </div>
    </div>
  );
}
