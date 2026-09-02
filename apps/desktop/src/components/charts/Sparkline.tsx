import { useState } from "react";

interface Props {
  values: number[];
  labels?: string[];
  height?: number;
  formatValue?: (value: number) => string;
  title: string;
  /** Line/area/marker hue: "accent" (default) or the muted neutral telemetry color. */
  tone?: "accent" | "neutral";
  /** Set false to omit the built-in footer row when the caller renders its own. */
  showFooter?: boolean;
}

/** Single-series line with area fill, 2px stroke, hover crosshair. */
export function Sparkline({ values, labels, height = 60, formatValue = (v) => v.toFixed(1), title, tone = "accent", showFooter = true }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const width = 300;
  const max = Math.max(1e-9, ...values);
  const step = values.length > 1 ? width / (values.length - 1) : width;
  const points = values.map((v, i) => [i * step, height - 4 - (v / max) * (height - 8)] as const);
  const path = points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const area = points.length > 0 ? `${path} L${points[points.length - 1][0].toFixed(1)},${height} L0,${height} Z` : "";

  return (
    <figure className="chart sparkline" aria-label={title}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="chart-svg"
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const ratio = (event.clientX - rect.left) / rect.width;
          setHover(Math.max(0, Math.min(values.length - 1, Math.round(ratio * (values.length - 1)))));
        }}
        onMouseLeave={() => setHover(null)}
      >
        {area && <path className={`chart-area ${tone === "neutral" ? "tone-neutral" : ""}`} d={area} />}
        {path && <path className={`chart-line ${tone === "neutral" ? "tone-neutral" : ""}`} d={path} />}
        {points.length > 0 && (
          <circle className="chart-marker" cx={points[points.length - 1][0]} cy={points[points.length - 1][1]} r={3} />
        )}
        {hover !== null && points[hover] && (
          <>
            <line className="chart-crosshair" x1={points[hover][0]} x2={points[hover][0]} y1={0} y2={height} />
            <circle className="chart-marker" cx={points[hover][0]} cy={points[hover][1]} r={4} />
          </>
        )}
      </svg>
      {showFooter && (
        <div className="chart-footer">
          <span className="chart-tooltip">
            {hover !== null && values[hover] !== undefined ? `${labels?.[hover] ?? hover}: ${formatValue(values[hover])}` : `peak ${formatValue(max)}`}
          </span>
        </div>
      )}
      <table className="sr-only">
        <caption>{title}</caption>
        <tbody>
          {values.map((v, i) => (
            <tr key={i}>
              <td>{labels?.[i] ?? i}</td>
              <td>{formatValue(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
