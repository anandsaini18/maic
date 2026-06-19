import type { TpmPoint } from "../../api/types";

interface Props {
  points: TpmPoint[];
}

const W = 260;
const H = 80;
const PAD_TOP = 14;
const PAD_BOT = 4;

export default function TpmChart({ points }: Props) {
  const last = points.slice(-20);
  const values = last.map((p) => p.tokPerSec);
  const max = Math.max(...values, 1);
  const latest = values.length > 0 ? values[values.length - 1] : 0;

  const yScale = (v: number) =>
    PAD_TOP + (1 - v / max) * (H - PAD_TOP - PAD_BOT);

  const linePoints = last
    .map((_, i) => {
      const x = last.length === 1 ? W / 2 : (i / (last.length - 1)) * W;
      return `${x},${yScale(values[i])}`;
    })
    .join(" ");

  const areaPoints =
    last.length > 0
      ? `0,${H} ${linePoints} ${W},${H}`
      : "";

  return (
    <div className="px-5 py-4 border-b border-warm-border">
      <div className="flex items-baseline justify-between mb-2">
        <span className="font-display text-[11px] font-medium uppercase tracking-[1.5px] text-warm-muted">
          Throughput
        </span>
        <span className="font-display text-[12px] text-amber tabular-nums">
          {latest > 0 ? `${latest.toFixed(1)} tok/s` : "--"}
        </span>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ height: H }}
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id="tpm-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--amber)" stopOpacity="0.25" />
            <stop offset="100%" stopColor="var(--amber)" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* grid lines */}
        {[0.25, 0.5, 0.75].map((f) => (
          <line
            key={f}
            x1={0}
            y1={yScale(max * f)}
            x2={W}
            y2={yScale(max * f)}
            stroke="var(--warm-border)"
            strokeOpacity={0.4}
            strokeDasharray="2 4"
          />
        ))}

        {last.length > 1 && (
          <>
            <polygon points={areaPoints} fill="url(#tpm-fill)" />
            <polyline
              points={linePoints}
              fill="none"
              stroke="var(--amber)"
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          </>
        )}

        {/* single-point dot */}
        {last.length === 1 && (
          <circle
            cx={W / 2}
            cy={yScale(values[0])}
            r={3}
            fill="var(--amber)"
          />
        )}
      </svg>
    </div>
  );
}
