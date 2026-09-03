import type React from "react";
import { Fragment, useEffect, useMemo, useState } from "react";
import {
  buildTraceSpans,
  estimatePrefillSavedMs,
  roundChars,
  roundCompletionTokens,
  toCurl,
  traceScopeWindow,
  traceSummary,
  traceToMarkdown,
  throughputSeries,
  type InspectorScope,
  type SpanColor,
  type TraceSpan,
} from "../lib/chat";
import { formatMs, formatRate } from "../lib/format";
import type { AssistantMessage, Settings, SystemInfo, Trace, TraceChunkKind, TraceRound } from "../lib/types";
import { Sparkline } from "./charts/Sparkline";

/** Button semantics for clickable rows: Enter and Space activate, everything else passes through. */
function activateOnKey(event: React.KeyboardEvent<HTMLElement>, action: () => void) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    action();
  }
}


type Tab = "trace" | "throughput" | "tokens" | "request" | "response";

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "trace", label: "Trace" },
  { id: "throughput", label: "Throughput" },
  { id: "tokens", label: "Tokens" },
  { id: "request", label: "Request" },
  { id: "response", label: "Response" },
];

const KIND_LABEL: Record<TraceChunkKind, string> = {
  role: "role",
  reasoning: "reasoning",
  content: "content",
  tool_call: "tool call",
  progress: "progress",
  usage: "usage",
  finish: "finish",
  other: "other",
};

const KIND_COLOR: Record<TraceChunkKind, SpanColor> = {
  role: "neutral",
  reasoning: "reasoning",
  content: "accent",
  tool_call: "accent",
  progress: "ok",
  usage: "neutral",
  finish: "neutral",
  other: "neutral",
};

const SPAN_BG: Record<SpanColor, string> = {
  neutral: "var(--neutral)",
  ok: "var(--ok)",
  accent: "var(--accent)",
  reasoning: "var(--thinking)",
  dim: "var(--border-strong)",
};

interface Props {
  message: AssistantMessage;
  settings: Settings;
  system: SystemInfo | null;
  onReplay: (requestBody: unknown) => void;
  replayDisabled: boolean;
  onCollapse: () => void;
}

function ChevronRightIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 5 L16 12 L9 19" />
    </svg>
  );
}

function CopyButton({ text, label = "Copy", title }: { text: string; label?: string; title?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="btn small ghost"
      title={title}
      onClick={() => {
        void navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1200);
        });
      }}
    >
      {copied ? "Copied" : label}
    </button>
  );
}

function chunkPreview(kind: TraceChunkKind, raw: unknown): string {
  try {
    const chunk = raw as { choices?: Array<{ delta?: Record<string, unknown>; finish_reason?: string }> };
    const delta = chunk.choices?.[0]?.delta;
    if (kind === "content" && typeof delta?.content === "string") return delta.content;
    if (kind === "reasoning" && typeof delta?.reasoning_content === "string") return delta.reasoning_content;
    if (kind === "finish") return chunk.choices?.[0]?.finish_reason ?? "";
    if (kind === "tool_call") return JSON.stringify(delta?.tool_calls ?? []);
  } catch {
    // fall through to raw JSON below
  }
  return JSON.stringify(raw);
}

interface ClippedSpan extends TraceSpan {
  left: number;
  width: number;
  dur: number;
  clippedStalls: number[];
}

function clipSpans(spans: TraceSpan[], window: { from: number; to: number }): ClippedSpan[] {
  const total = Math.max(1, window.to - window.from);
  return spans
    .filter((span) => span.end > window.from && span.start < window.to)
    .map((span) => {
      const start = Math.max(span.start, window.from);
      const end = Math.min(span.end, window.to);
      return {
        ...span,
        left: ((start - window.from) / total) * 100,
        width: Math.max(0.3, ((end - start) / total) * 100),
        dur: end - start,
        clippedStalls: span.stalls.filter((at) => at >= window.from && at <= window.to).map((at) => ((at - window.from) / total) * 100),
      };
    });
}

interface StatEntry {
  label: string;
  value: string;
  tone?: "ok" | "bad";
}

function traceStats(trace: Trace, scope: InspectorScope, spans: TraceSpan[]): StatEntry[] {
  const window = traceScopeWindow(trace, scope);
  const totalMs = Math.max(1, window.to - window.from);
  const visible = spans.filter((span) => span.end > window.from && span.start < window.to);
  const stallCount = visible.reduce((sum, span) => sum + span.stalls.filter((at) => at >= window.from && at <= window.to).length, 0);

  if (scope === "all") {
    const overlap = (span: TraceSpan) => Math.max(0, Math.min(span.end, window.to) - Math.max(span.start, window.from));
    const modelMs = visible.filter((span) => /^round-\d+$/.test(span.id)).reduce((sum, span) => sum + overlap(span), 0);
    const toolsMs = visible.filter((span) => /^tools-\d+$/.test(span.id)).reduce((sum, span) => sum + overlap(span), 0);
    return [
      { label: "Total", value: formatMs(totalMs) },
      { label: "Model time", value: `${Math.round((modelMs / totalMs) * 100)}%` },
      { label: "Tools", value: formatMs(toolsMs) },
      { label: "Stalls", value: String(stallCount), tone: stallCount > 0 ? "bad" : undefined },
    ];
  }

  const round = trace.rounds[scope];
  if (!round) return [];
  const cacheHit = round.promptProgress && round.promptProgress.total > 0 ? Math.round((round.promptProgress.cache / round.promptProgress.total) * 100) : null;
  const decodeMs = round.finishedAt !== undefined && round.firstTokenAt !== undefined ? round.finishedAt - round.firstTokenAt : null;
  const roundTokens = roundCompletionTokens(round);
  const decodeRate = decodeMs !== null && decodeMs > 0 && roundTokens !== undefined ? roundTokens / (decodeMs / 1000) : null;
  const thinkingSpan = visible.find((span) => span.id === `round-${scope}-thinking`);
  const thirdStat: StatEntry =
    decodeRate !== null
      ? { label: "Decode", value: formatRate(decodeRate) }
      : thinkingSpan
        ? { label: "Thinking", value: formatMs(thinkingSpan.end - thinkingSpan.start) }
        : { label: "Decode", value: "–" };
  return [
    { label: "TTFT", value: round.firstTokenAt !== undefined ? formatMs(round.firstTokenAt) : "–" },
    { label: "Cache hit", value: cacheHit !== null ? `${cacheHit}%` : "–" },
    thirdStat,
    { label: "Stalls", value: String(stallCount), tone: stallCount > 0 ? "bad" : undefined },
  ];
}

/** Short tick label: "0ms", "0.9s", "12s" — kept to 5-6 chars so 5 ticks never overlap in a 480px inspector. */
function formatTick(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 10_000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.round(ms / 1000)}s`;
}

function TraceTab({ trace, scope, spans }: { trace: Trace; scope: InspectorScope; spans: TraceSpan[] }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  useEffect(() => setSelectedId(null), [scope, trace]);

  const window = traceScopeWindow(trace, scope);
  const clipped = useMemo(() => clipSpans(spans, window), [spans, window.from, window.to]);
  const stats = useMemo(() => traceStats(trace, scope, spans), [trace, scope, spans]);
  const ticks = useMemo(() => {
    const total = Math.max(1, window.to - window.from);
    return Array.from({ length: 4 }, (_, i) => ({ pct: (i / 3) * 100, label: formatTick(window.from + (total * i) / 3), edge: i === 0 ? "start" : i === 3 ? "end" : "mid" }));
  }, [window.from, window.to]);

  const defaultSpan = [...clipped].reverse().find((span) => span.color === "accent") ?? clipped[clipped.length - 1] ?? null;
  const selected = clipped.find((span) => span.id === selectedId) ?? defaultSpan;

  return (
    <div className="inspector-tab">
      <div className="trace-stat-row">
        {stats.map((stat) => (
          <div key={stat.label} className="panel trace-stat">
            <span className="meta">{stat.label}</span>
            <span className="mono" style={{ fontSize: 13, fontWeight: 600, color: stat.tone === "bad" ? "var(--bad)" : "var(--text)" }}>
              {stat.value}
            </span>
          </div>
        ))}
      </div>

      <div className="panel waterfall">
        <div className="waterfall-axis">
          <span className="meta">Span</span>
          <div className="waterfall-ticks">
            {ticks.map((tick, i) => (
              <span
                key={i}
                className="mono"
                style={{ left: `${tick.pct}%`, transform: tick.edge === "start" ? "none" : tick.edge === "end" ? "translateX(-100%)" : "translateX(-50%)" }}
              >
                {tick.label}
              </span>
            ))}
          </div>
          <span className="meta span-dur-head">dur · %</span>
        </div>
        {clipped.map((span) => {
          const open = selected?.id === span.id;
          return (
            <Fragment key={span.id}>
              <div
                className={`span-row ${open ? "on" : ""}`}
                role="button"
                tabIndex={0}
                aria-expanded={open}
                onClick={() => setSelectedId(span.id)}
                onKeyDown={(event) => activateOnKey(event, () => setSelectedId(span.id))}
              >
                <span className="span-label" style={{ paddingLeft: span.indent * 14 }}>
                  <span className="span-swatch" style={{ background: SPAN_BG[span.color] }} />
                  <span className="span-label-text">{span.label}</span>
                </span>
                <div className="span-track">
                  <div className="span-bar" style={{ left: `${span.left}%`, width: `${span.width}%`, background: SPAN_BG[span.color], opacity: span.indent ? 1 : 0.55 }} />
                  {span.clippedStalls.map((pct, i) => (
                    <div key={i} className="span-stall" style={{ left: `${pct}%` }} />
                  ))}
                </div>
                <span className="mono span-dur">
                  {formatMs(span.dur)} · {Math.round((span.dur / Math.max(1, window.to - window.from)) * 100)}%
                </span>
              </div>
              {open && (
                <div className="span-detail">
                  <div className="span-detail-head">
                    <span className="span-detail-title">{span.label}</span>
                    <span className="mono label">
                      {formatMs(span.start)} → {formatMs(span.end)}
                    </span>
                  </div>
                  <span className="span-detail-text">{span.detail}</span>
                </div>
              )}
            </Fragment>
          );
        })}
        {clipped.length === 0 && <div className="notice">No spans in this window.</div>}
      </div>
    </div>
  );
}

function roundSummary(round: TraceRound) {
  const roundTokens = roundCompletionTokens(round);
  const decodeTokPerSec =
    roundTokens !== undefined && round.firstTokenAt !== undefined && round.finishedAt !== undefined && round.finishedAt > round.firstTokenAt
      ? roundTokens / ((round.finishedAt - round.firstTokenAt) / 1000)
      : null;
  const uncachedPromptTokens = round.promptProgress ? round.promptProgress.processed - round.promptProgress.cache : 0;
  const promptTokPerSec = round.promptProgress && round.promptProgress.time_ms > 0 && uncachedPromptTokens > 0 ? uncachedPromptTokens / (round.promptProgress.time_ms / 1000) : null;
  const firstReasoningAt = round.chunks.find((chunk) => chunk.kind === "reasoning")?.at ?? null;
  const firstVisibleAt = round.chunks.find((chunk) => chunk.kind === "content")?.at ?? null;
  // Gaps start from the first chunk's `at`, not 0, so the prefill time
  // before the first chunk arrives isn't counted as a mid-stream stall.
  let previous = round.chunks[0]?.at ?? 0;
  let longestGap = -1;
  let longestGapIndex = -1;
  const gaps: number[] = [];
  round.chunks.slice(1).forEach((chunk, i) => {
    const gap = chunk.at - previous;
    gaps.push(gap);
    if (gap > longestGap) {
      longestGap = gap;
      longestGapIndex = i + 1;
    }
    previous = chunk.at;
  });
  const avgGap = gaps.length > 0 ? gaps.reduce((sum, gap) => sum + gap, 0) / gaps.length : null;
  return {
    decodeTokPerSec,
    promptTokPerSec,
    firstReasoningAt,
    firstVisibleAt,
    chunkCount: round.chunks.length,
    avgGap,
    longestGap: gaps.length > 0 ? longestGap : null,
    longestGapIndex,
  };
}

function ThroughputTab({ trace, scope }: { trace: Trace; scope: InspectorScope }) {
  const rounds = trace.rounds;
  const round = rounds[scope === "all" ? rounds.length - 1 : scope];
  const series = useMemo(() => throughputSeries(round), [round]);
  const window = traceScopeWindow(trace, scope);
  // The chart itself is already round-relative (throughputSeries buckets by
  // second within the round), so the axis footer must be too when a single
  // round is selected — otherwise it shows absolute offsets like "6.14s"
  // next to tiles that all say "this round".
  const axisWindow = scope === "all" ? window : { from: 0, to: window.to - window.from };

  if (scope === "all") {
    const summary = traceSummary(trace);
    return (
      <ThroughputBody
        stats={[
          { label: "Decode", value: summary.decodeTokPerSec !== null ? formatRate(summary.decodeTokPerSec) : "–", hint: "completion tokens / time after first token" },
          { label: "Uncached prompt", value: summary.promptTokPerSec !== null ? formatRate(summary.promptTokPerSec) : "–", hint: "round 1 prefill" },
          { label: "First reasoning token", value: "–", hint: "per round only" },
          { label: "First visible token", value: summary.ttftMs !== null ? formatMs(summary.ttftMs) : "–", hint: "round 1" },
          { label: "Chunks", value: String(summary.chunkCount), hint: summary.avgChunkGapMs !== null ? `avg gap ${formatMs(summary.avgChunkGapMs)}` : "" },
          { label: "Longest stall", value: summary.longestStallMs !== null ? formatMs(summary.longestStallMs) : "–", hint: "across all rounds" },
        ]}
        series={series}
        scopeLabel="entire request"
        window={axisWindow}
      />
    );
  }
  const summary = roundSummary(round);
  return (
    <ThroughputBody
      stats={[
        { label: "Decode", value: summary.decodeTokPerSec !== null ? formatRate(summary.decodeTokPerSec) : "–", hint: "this round" },
        { label: "Uncached prompt", value: summary.promptTokPerSec !== null ? formatRate(summary.promptTokPerSec) : "–", hint: "this round" },
        { label: "First reasoning token", value: summary.firstReasoningAt !== null ? formatMs(summary.firstReasoningAt) : "–", hint: "this round" },
        { label: "First visible token", value: summary.firstVisibleAt !== null ? formatMs(summary.firstVisibleAt) : "–", hint: "this round" },
        { label: "Chunks", value: String(summary.chunkCount), hint: summary.avgGap !== null ? `avg gap ${formatMs(summary.avgGap)}` : "" },
        {
          label: "Longest stall",
          value: summary.longestGap !== null && summary.longestGap > 0 ? formatMs(summary.longestGap) : "–",
          hint: summary.longestGapIndex > 0 ? `between chunks ${summary.longestGapIndex - 1} and ${summary.longestGapIndex}` : "",
        },
      ]}
      series={series}
      scopeLabel={`round ${(scope as number) + 1}`}
      window={axisWindow}
    />
  );
}

function ThroughputBody({
  stats,
  series,
  scopeLabel,
  window,
}: {
  stats: Array<{ label: string; value: string; hint: string }>;
  series: { labels: string[]; values: number[] };
  scopeLabel: string;
  window: { from: number; to: number };
}) {
  const stallIndex = series.values.length > 1 ? series.values.findIndex((v, i) => i > 0 && v === 0 && series.values[i - 1] > 0) : -1;
  return (
    <div className="inspector-tab">
      <div className="stat-row">
        {stats.map((stat) => (
          <div key={stat.label} className="panel throughput-stat">
            <span className="meta">{stat.label}</span>
            <span className="mono" style={{ fontSize: 20, fontWeight: 600 }}>
              {stat.value}
            </span>
            {stat.hint && (
              <span className="label" style={{ fontSize: 11 }}>
                {stat.hint}
              </span>
            )}
          </div>
        ))}
      </div>
      {series.values.length > 0 ? (
        <div className="panel throughput-chart">
          <span className="meta">Tokens per second · {scopeLabel}</span>
          <Sparkline values={series.values} labels={series.labels} formatValue={(v) => v.toFixed(1)} title="Tokens per second" tone="neutral" />
          <div className="mono throughput-chart-footer">
            <span>{formatMs(window.from)}</span>
            {stallIndex >= 0 && <span className="stall-label">stall near second {stallIndex}</span>}
            <span>{formatMs(window.to)}</span>
          </div>
        </div>
      ) : (
        <div className="notice">No streamed tokens recorded for this window.</div>
      )}
    </div>
  );
}

function TokensTab({ trace, scope, message }: { trace: Trace; scope: InspectorScope; message: AssistantMessage }) {
  const rounds = trace.rounds;
  let rows: Array<{ label: string; value: string }>;
  if (scope === "all") {
    const summary = traceSummary(trace);
    const usage = rounds.reduce(
      (acc, round) => ({ prompt: acc.prompt + (round.usage?.prompt_tokens ?? 0), completion: acc.completion + (round.usage?.completion_tokens ?? 0) }),
      { prompt: 0, completion: 0 },
    );
    const cached = rounds.reduce((sum, round) => sum + (round.promptProgress?.cache ?? 0), 0);
    const savedMs = rounds.reduce((sum, round) => sum + (estimatePrefillSavedMs(round) ?? 0), 0);
    const last = rounds[rounds.length - 1];
    rows = [
      { label: "Prompt tokens", value: String(usage.prompt) },
      { label: "From prefix cache", value: `${cached} (${usage.prompt > 0 ? Math.round((cached / usage.prompt) * 100) : 0}%)` },
      { label: "Prefill time saved", value: savedMs > 0 ? `≈ ${formatMs(savedMs)}` : "–" },
      { label: "Completion tokens", value: String(usage.completion) },
      { label: "Reasoning / visible", value: `${summary.reasoningChars} / ${summary.contentChars} chars` },
      { label: "Finish reason", value: last?.finishReason ?? "–" },
      { label: "Model reported", value: message.stats.model ?? "–" },
    ];
  } else {
    const round = rounds[scope];
    const chars = roundChars(round);
    const savedMs = estimatePrefillSavedMs(round);
    rows = [
      { label: "Prompt tokens", value: String(round.usage?.prompt_tokens ?? "–") },
      {
        label: "From prefix cache",
        value: round.promptProgress
          ? `${round.promptProgress.cache} (${round.promptProgress.total > 0 ? Math.round((round.promptProgress.cache / round.promptProgress.total) * 100) : 0}%)`
          : "–",
      },
      { label: "Prefill time saved", value: savedMs !== null ? `≈ ${formatMs(savedMs)}` : "–" },
      { label: "Completion tokens", value: String(round.usage?.completion_tokens ?? "–") },
      { label: "Reasoning / visible", value: `${chars.reasoningChars} / ${chars.contentChars} chars` },
      { label: "Finish reason", value: round.finishReason ?? "–" },
      { label: "Model reported", value: round.model ?? "–" },
    ];
  }
  return (
    <div className="panel token-rows">
      {rows.map((row) => (
        <div key={row.label} className="token-row">
          <span className="label">{row.label}</span>
          <span className="mono">{row.value}</span>
        </div>
      ))}
    </div>
  );
}

function RequestTab({
  round,
  settings,
  onReplay,
  replayDisabled,
}: {
  round: TraceRound;
  settings: Settings;
  onReplay: (requestBody: unknown) => void;
  replayDisabled: boolean;
}) {
  const pretty = JSON.stringify(round.requestBody, null, 2);
  return (
    <div className="inspector-tab">
      <div className="inspector-actions">
        <CopyButton text={pretty} />
        <CopyButton text={toCurl(settings.baseUrl, settings.apiKey, round.requestBody)} label="Copy as curl" />
        <button type="button" className="btn small primary" disabled={replayDisabled} onClick={() => onReplay(round.requestBody)}>
          Replay
        </button>
      </div>
      <pre className="inspector-json">{pretty}</pre>
    </div>
  );
}

function ResponseTab({ round, message }: { round: TraceRound; message: AssistantMessage }) {
  const [expanded, setExpanded] = useState<number | null>(null);
  return (
    <div className="panel token-rows">
      {round.chunks.map((chunk, index) => (
        <Fragment key={index}>
          <div
            className="chunk-row"
            role="button"
            tabIndex={0}
            aria-expanded={expanded === index}
            onClick={() => setExpanded(expanded === index ? null : index)}
            onKeyDown={(event) => activateOnKey(event, () => setExpanded(expanded === index ? null : index))}
          >
            <span className="mono label">{formatMs(chunk.at)}</span>
            <span className="chunk-kind">
              <span className="span-swatch" style={{ background: SPAN_BG[KIND_COLOR[chunk.kind]] }} />
              {KIND_LABEL[chunk.kind]}
            </span>
            <span className="mono chunk-preview">{chunkPreview(chunk.kind, chunk.raw)}</span>
          </div>
          {expanded === index && <pre className="inspector-json">{JSON.stringify(chunk.raw, null, 2)}</pre>}
        </Fragment>
      ))}
      {round.chunks.length === 0 && <div className="notice">No chunks recorded.</div>}
      <div className="assembled-message">
        <div className="meta">Assembled message</div>
        <pre className="inspector-json">{JSON.stringify({ content: message.content, reasoning: message.reasoning, tool_calls: message.toolCalls }, null, 2)}</pre>
      </div>
    </div>
  );
}

export function RequestInspector({ message, settings, system, onReplay, replayDisabled, onCollapse }: Props) {
  const [tab, setTab] = useState<Tab>("trace");
  const [scope, setScope] = useState<InspectorScope>("all");

  useEffect(() => {
    setScope("all");
  }, [message.id]);

  const trace = message.trace;
  const rounds = trace?.rounds ?? [];
  const spans = useMemo(() => (trace ? buildTraceSpans(trace) : []), [trace]);

  if (!trace || rounds.length === 0) {
    return (
      <aside className="inspector">
        <div className="inspector-header">
          <span>Inspector</span>
          <div className="inspector-header-spacer" />
          <button type="button" className="icon-btn inspector-collapse-btn" onClick={onCollapse} title="Collapse inspector">
            <ChevronRightIcon />
          </button>
        </div>
        <div className="inspector-body">
          <div className="notice">No trace recorded for this message.</div>
        </div>
      </aside>
    );
  }

  const safeScope: InspectorScope = scope === "all" || scope < 0 || scope >= rounds.length ? "all" : scope;
  const activeRoundIndex = safeScope === "all" ? rounds.length - 1 : safeScope;
  const activeRound = rounds[activeRoundIndex];

  return (
    <aside className="inspector">
      <div className="inspector-header">
        <span>Inspector</span>
        <div className="inspector-header-spacer" />
        <button type="button" className="icon-btn inspector-collapse-btn" onClick={onCollapse} title="Collapse inspector">
          <ChevronRightIcon />
        </button>
        <div className="inspector-controls">
          <div className="seg-group">
            <button type="button" className={`seg ${scope === "all" ? "on" : ""}`} onClick={() => setScope("all")}>
              Entire request
            </button>
            {rounds.map((_round, index) => (
              <button key={index} type="button" className={`seg ${scope === index ? "on" : ""}`} onClick={() => setScope(index)}>
                Round {index + 1}
              </button>
            ))}
          </div>
          <div className="inspector-header-spacer" />
          <CopyButton text={traceToMarkdown(message, settings, system)} label="Markdown" title="Copy the full trace as Markdown" />
        </div>
      </div>
      <div className="tabs">
        {TABS.map((item) => (
          <button key={item.id} type="button" className={`tab ${tab === item.id ? "active" : ""}`} onClick={() => setTab(item.id)}>
            {item.label}
          </button>
        ))}
      </div>
      <div className="inspector-body">
        {tab === "trace" && <TraceTab trace={trace} scope={safeScope} spans={spans} />}
        {tab === "throughput" && <ThroughputTab trace={trace} scope={safeScope} />}
        {tab === "tokens" && <TokensTab trace={trace} scope={safeScope} message={message} />}
        {tab === "request" && <RequestTab round={activeRound} settings={settings} onReplay={onReplay} replayDisabled={replayDisabled} />}
        {tab === "response" && <ResponseTab round={activeRound} message={message} />}
      </div>
    </aside>
  );
}
