import { useEffect, useState } from "react";
import type { AssistantMessage } from "../lib/types";

function formatSeconds(ms: number): string {
  return ms < 10_000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms / 1000)}s`;
}

function countSegments(message: AssistantMessage): number {
  const rounds = message.trace?.rounds;
  if (rounds && rounds.length > 0) {
    const count = rounds.filter((round) => round.chunks.some((chunk) => chunk.kind === "reasoning")).length;
    if (count > 0) return count;
  }
  return message.reasoning ? 1 : 0;
}

export function ThinkingBlock({ message }: { message: AssistantMessage }) {
  const active = message.status === "thinking";
  const [open, setOpen] = useState(active);
  const [, tick] = useState(0);

  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => tick((n) => n + 1), 200);
    return () => clearInterval(timer);
  }, [active]);

  useEffect(() => {
    if (!active && message.content) setOpen(false);
  }, [active, message.content]);

  if (!message.reasoning && !active) return null;

  const liveSegment = message.stats.thinkingStartedAt === undefined ? 0 : Date.now() - message.stats.thinkingStartedAt;
  const total = (message.stats.thinkingMs ?? 0) + liveSegment;
  const verb = active ? "Thinking" : "Thought";
  const tokens = Math.round(message.reasoning.length / 4);
  const segments = countSegments(message);

  return (
    <div className={`thinking ${active ? "active" : ""}`}>
      <button type="button" className="thinking-toggle" onClick={() => setOpen((v) => !v)}>
        <span className="thinking-dot" />
        <span className={active ? "shimmer" : ""}>{`${verb} for ${formatSeconds(total)}`}</span>
        {!active && tokens > 0 && (
          <span className="mono label">
            {tokens} tokens · {segments} segment{segments === 1 ? "" : "s"}
          </span>
        )}
        <span className="chevron">{open ? "▾" : "▸"}</span>
      </button>
      {open && <div className="thinking-body">{message.reasoning || "…"}</div>}
    </div>
  );
}
