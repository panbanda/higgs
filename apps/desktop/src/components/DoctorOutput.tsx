import type { CommandOutput } from "../lib/types";

interface DoctorLine {
  pill: "PASS" | "WARN" | "FAIL" | null;
  text: string;
}

const PILL_PATTERN = /^\s*\[(PASS|WARN|FAIL)\]\s*(.*)$/;

const PILL_COLOR: Record<"PASS" | "WARN" | "FAIL", string> = { PASS: "var(--ok)", WARN: "var(--warn)", FAIL: "var(--bad)" };

function parseLines(output: CommandOutput): DoctorLine[] {
  const combined = [output.stdout, output.stderr].filter(Boolean).join("\n");
  return combined
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => {
      const match = PILL_PATTERN.exec(line);
      if (!match) return { pill: null, text: line };
      return { pill: match[1] as "PASS" | "WARN" | "FAIL", text: match[2] };
    });
}

interface DoctorOutputProps {
  busy: boolean;
  output: CommandOutput | null;
  onRun: () => void;
}

export function DoctorOutput({ busy, output, onRun }: DoctorOutputProps) {
  const lines = output ? parseLines(output) : [];
  return (
    <div className="panel" style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 10 }}>
      <div className="config-doctor-row">
        <span className="meta" style={{ flex: 1 }}>
          Doctor
        </span>
        <button type="button" className="btn" disabled={busy} onClick={onRun}>
          {busy ? "Running…" : "Run doctor"}
        </button>
        {output && (
          <span className="mono" style={{ fontSize: 12, color: output.exit_code === 0 ? "var(--ok)" : "var(--bad)" }}>
            exit {output.exit_code ?? "?"}
          </span>
        )}
      </div>
      {output ? (
        lines.length > 0 ? (
          <div className="config-doctor-rows">
            {lines.map((line, index) => (
              <div className="config-doctor-line" key={index}>
                <span style={{ color: line.pill ? PILL_COLOR[line.pill] : "var(--text-muted)" }}>{line.pill ?? ""}</span>
                <span>{line.text}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="config-list-empty">No output.</div>
        )
      ) : (
        <div className="config-list-empty">Doctor has not run yet.</div>
      )}
    </div>
  );
}
