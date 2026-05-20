import type { RLHFReport } from "../types";

interface Props {
  report: RLHFReport | null;
}

export function StatsBar({ report }: Props) {
  if (!report) return null;

  return (
    <div className="flex items-center gap-6 text-xs">
      <Stat label="Pairs" value={report.total_pairs} />
      <Stat label="Avg Score" value={report.avg_reward_score.toFixed(3)} />
      <Stat
        label="Over-Refusal"
        value={`${(report.over_refusal_ratio * 100).toFixed(1)}%`}
        warn={report.over_refusal_ratio < 0.2}
      />
      <Stat
        label="Violations"
        value={`${(report.violation_ratio * 100).toFixed(1)}%`}
      />
      <div className="flex gap-1.5 flex-wrap">
        {Object.entries(report.pairs_by_failure_mode).map(([mode, count]) => (
          <span
            key={mode}
            className="px-1.5 py-0.5 rounded bg-slate-700 text-slate-300"
          >
            {mode.replace(/_/g, " ")}: {count}
          </span>
        ))}
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  warn,
}: {
  label: string;
  value: string | number;
  warn?: boolean;
}) {
  return (
    <div className="flex flex-col items-center">
      <span className={`font-mono font-bold ${warn ? "text-amber-400" : "text-blue-400"}`}>
        {value}
      </span>
      <span className="text-slate-500">{label}</span>
    </div>
  );
}
