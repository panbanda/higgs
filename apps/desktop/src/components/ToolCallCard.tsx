import { useState } from "react";
import type { ToolCall } from "../lib/types";

function pretty(raw: string | undefined): string {
  if (!raw) return "";
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
}

/** Renders `arguments` (a JSON object string) as the inline `(key: value, ...)` shown next to the tool name. */
function argsPreview(raw: string): string {
  try {
    const parsed = JSON.parse(raw || "{}") as Record<string, unknown>;
    const parts = Object.entries(parsed).map(([key, value]) => `${key}: ${typeof value === "string" ? value : JSON.stringify(value)}`);
    return `(${parts.join(", ")})`;
  } catch {
    return raw ? `(${raw})` : "()";
  }
}

function durationLabel(call: ToolCall): string | null {
  if (call.startedAt === undefined || call.finishedAt === undefined) return null;
  return `${call.finishedAt - call.startedAt} ms`;
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ok)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12 L10 17 L19 7" />
    </svg>
  );
}

function ErrorIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--bad)" strokeWidth="3" strokeLinecap="round">
      <path d="M12 8 V13 M12 16.5 V17" />
      <circle cx="12" cy="12" r="9" />
    </svg>
  );
}

export function ToolCallCard({ call }: { call: ToolCall }) {
  const [open, setOpen] = useState(false);
  const busy = call.status === "streaming" || call.status === "running";
  const duration = durationLabel(call);
  return (
    <div className={`toolcall toolcall-${call.status}`}>
      <button type="button" className="toolcall-header" onClick={() => setOpen((v) => !v)}>
        <span className="toolcall-icon">{busy ? <span className="spinner" /> : call.status === "error" ? <ErrorIcon /> : <CheckIcon />}</span>
        <code className="toolcall-name">
          {call.name || "…"}
          <span className="toolcall-args">{argsPreview(call.arguments)}</span>
        </code>
        {duration && <span className="mono label">{duration}</span>}
        <span className={`toolcall-status ${call.status === "done" ? "ok" : call.status === "error" ? "bad" : ""}`}>
          {call.status === "done" ? "done" : call.status === "error" ? "failed" : call.status === "running" ? "running" : "preparing"}
        </span>
        <svg className="chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: open ? "rotate(90deg)" : "none" }}><path d="M9 6 L15 12 L9 18" /></svg>
      </button>
      {open && (
        <div className="toolcall-body">
          <div className="toolcall-section">
            <div className="toolcall-label">Arguments</div>
            <pre>{pretty(call.arguments) || "{}"}</pre>
          </div>
          {(call.result || call.error) && (
            <div className="toolcall-section">
              <div className="toolcall-label">{call.error ? "Error" : "Result"}</div>
              <pre>{call.error ?? pretty(call.result)}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
