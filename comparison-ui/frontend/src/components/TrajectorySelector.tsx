import type { TrajectoryMeta } from "../types";

interface Props {
  trajectories: TrajectoryMeta[];
  selected: string | null;
  onSelect: (id: string) => void;
  loading: boolean;
}

export function TrajectorySelector({ trajectories, selected, onSelect, loading }: Props) {
  if (loading) {
    return <div className="text-slate-400 text-sm">Loading trajectories...</div>;
  }

  return (
    <div className="flex items-center gap-3">
      <label className="text-sm font-medium text-slate-300">Trajectory:</label>
      <select
        className="bg-slate-800 border border-slate-600 rounded-md px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500 min-w-[320px]"
        value={selected || ""}
        onChange={(e) => onSelect(e.target.value)}
      >
        <option value="">Select a trajectory...</option>
        {trajectories.map((t) => (
          <option key={t.submission_id} value={t.submission_id}>
            {t.task_id} — {t.submission_id.slice(0, 8)}...
            {t.has_rlhf ? " [RLHF]" : ""}
            {t.status ? ` (${t.status})` : ""}
          </option>
        ))}
      </select>
    </div>
  );
}
