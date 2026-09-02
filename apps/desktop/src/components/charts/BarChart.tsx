import { useState } from "react";

export interface BarDatum {
  label: string;
  value: number;
  /** Optional second value drawn as a darker cap (for example errors within requests). */
  accent?: number;
}

interface Props {
  data: BarDatum[];
  height?: number;
  formatValue?: (value: number) => string;
  title: string;
  /** Set false to omit the built-in footer row when the caller renders its own. */
  showFooter?: boolean;
}

/**
 * Single-series bar chart: one hue, 2px gaps, rounded data ends, hover
 * tooltip, and a hidden table for screen readers and copy/paste.
 */
export function BarChart({ data, height = 120, formatValue = (v) => v.toLocaleString(), title, showFooter = true }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const max = Math.max(1, ...data.map((d) => d.value));
  const width = 600;
  const padTop = 6;
  const gap = 2;
  const barWidth = Math.max(1, (width - gap * (data.length - 1)) / Math.max(1, data.length));
  const chartHeight = height - padTop;

  return (
    <figure className="chart" aria-label={title}>
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="chart-svg" onMouseLeave={() => setHover(null)}>
        {[0.5, 1].map((fraction) => (
          <line key={fraction} className="chart-grid" x1={0} x2={width} y1={padTop + chartHeight * (1 - fraction)} y2={padTop + chartHeight * (1 - fraction)} />
        ))}
        {data.map((d, i) => {
          const h = Math.max(d.value > 0 ? 2 : 0, (d.value / max) * chartHeight);
          const x = i * (barWidth + gap);
          const y = padTop + chartHeight - h;
          const accentHeight = d.accent ? Math.min(h, Math.max(2, (d.accent / max) * chartHeight)) : 0;
          return (
            <g key={d.label} onMouseEnter={() => setHover(i)}>
              <rect className="chart-hit" x={x} y={0} width={barWidth + gap} height={height} />
              <rect className={`chart-bar ${hover === i ? "hover" : ""}`} x={x} y={y} width={barWidth} height={h} rx={Math.min(2, barWidth / 2)} />
              {accentHeight > 0 && <rect className="chart-bar-accent" x={x} y={y} width={barWidth} height={accentHeight} rx={Math.min(2, barWidth / 2)} />}
            </g>
          );
        })}
      </svg>
      {showFooter && (
        <div className="chart-footer">
          <span className="chart-axis">{data[0]?.label}</span>
          <span className="chart-tooltip">
            {hover !== null && data[hover]
              ? `${data[hover].label}: ${formatValue(data[hover].value)}${data[hover].accent ? ` (${data[hover].accent} errors)` : ""}`
              : `max ${formatValue(max)}`}
          </span>
          <span className="chart-axis">{data[data.length - 1]?.label}</span>
        </div>
      )}
      <table className="sr-only">
        <caption>{title}</caption>
        <tbody>
          {data.map((d) => (
            <tr key={d.label}>
              <td>{d.label}</td>
              <td>{formatValue(d.value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
