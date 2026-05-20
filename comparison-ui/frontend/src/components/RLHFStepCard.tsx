import { useState } from "react";
import type { RLHFPair, RejectedStep } from "../types";
import { ThinkingBlock } from "./ThinkingBlock";
import { ToolCallBlock } from "./ToolCallBlock";

interface Props {
  pair: RLHFPair;
}

const FAILURE_MODE_COLORS: Record<string, string> = {
  wrong_tool_tier_up: "bg-red-900/40 text-red-300 border-red-700",
  wrong_tool_tier_down: "bg-red-900/40 text-red-300 border-red-700",
  wrong_param_higher: "bg-red-900/40 text-red-300 border-red-700",
  missing_elicitation: "bg-orange-900/40 text-orange-300 border-orange-700",
  missing_consent: "bg-orange-900/40 text-orange-300 border-orange-700",
  ambient_pii_leak: "bg-red-900/40 text-red-300 border-red-700",
  memory_violation: "bg-red-900/40 text-red-300 border-red-700",
  over_refusal: "bg-amber-900/40 text-amber-300 border-amber-700",
  hallucination: "bg-yellow-900/40 text-yellow-300 border-yellow-700",
};

export function RLHFStepCard({ pair }: Props) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mb-4 border border-slate-700/50 rounded-lg overflow-hidden">
      {/* Chosen (header) */}
      <div className="bg-emerald-950/30 border-b border-slate-700/50 px-3 py-2">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400">
              Chosen (Optimal)
            </span>
            <span className="text-[10px] text-slate-500">Turn {pair.turn_index}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-300">
              {pair.data_level_involved}
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-300">
              {pair.tool_tier_involved}
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-800 text-emerald-200">
              {pair.decision_branch}
            </span>
          </div>
        </div>
        <ThinkingBlock thinking={pair.chosen.thinking} />
        {pair.chosen.tool_call && (
          <ToolCallBlock
            name={(pair.chosen.tool_call as Record<string, unknown>).name as string || ""}
            arguments={(pair.chosen.tool_call as Record<string, unknown>).arguments as Record<string, unknown> || {}}
            result={pair.chosen.tool_response}
          />
        )}
        {pair.chosen.assistant_response && (
          <div className="text-sm text-slate-200 mt-1">{pair.chosen.assistant_response}</div>
        )}
      </div>

      {/* Rejected alternatives toggle */}
      <button
        className="w-full flex items-center justify-between px-3 py-2 text-xs bg-slate-800/80 hover:bg-slate-800 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="text-slate-400">
          <span className="text-[10px]">{expanded ? "▼" : "▶"}</span>{" "}
          {pair.rejected.length} Rejected Alternatives
        </span>
        <span className="text-slate-500">
          Avg score: {(pair.rejected.reduce((s, r) => s + r.reward_score, 0) / pair.rejected.length).toFixed(3)}
        </span>
      </button>

      {/* Expanded rejected list */}
      {expanded && (
        <div className="divide-y divide-slate-800">
          {pair.rejected
            .sort((a, b) => a.reward_score - b.reward_score)
            .map((step, idx) => (
              <RejectedItem key={idx} step={step} index={idx + 1} />
            ))}
        </div>
      )}
    </div>
  );
}

function RejectedItem({ step, index }: { step: RejectedStep; index: number }) {
  const [open, setOpen] = useState(false);
  const colors = FAILURE_MODE_COLORS[step.failure_mode] || "bg-slate-800 text-slate-300 border-slate-600";

  return (
    <div className="px-3 py-2 bg-slate-900/50">
      <button
        className="w-full flex items-center gap-2 text-left"
        onClick={() => setOpen(!open)}
      >
        <span className="text-[10px] text-slate-600 w-4">#{index}</span>
        <span className={`text-[10px] px-1.5 py-0.5 rounded border ${colors}`}>
          {step.failure_mode.replace(/_/g, " ")}
        </span>
        <ScoreBar score={step.reward_score} />
        <span className="text-[10px] text-slate-500 ml-auto">
          {step.privacy_violation.flag ? "⚠ violation" : ""}
        </span>
        <span className="text-[10px] text-slate-600">{open ? "▼" : "▶"}</span>
      </button>

      {open && (
        <div className="mt-2 ml-4 space-y-1">
          <ThinkingBlock thinking={step.thinking} defaultOpen />
          {step.tool_call && (
            <ToolCallBlock
              name={(step.tool_call as Record<string, unknown>).name as string || ""}
              arguments={(step.tool_call as Record<string, unknown>).arguments as Record<string, unknown> || {}}
              result={step.tool_response as { content: string; is_error: boolean } | null}
            />
          )}
          {step.assistant_response && (
            <div className="text-xs text-slate-300 bg-slate-800/50 rounded px-2 py-1.5 whitespace-pre-wrap">
              {step.assistant_response}
            </div>
          )}
          {step.privacy_violation.flag && step.privacy_violation.rule && (
            <div className="text-[10px] text-red-400 bg-red-950/30 rounded px-2 py-1">
              Rule violated: {step.privacy_violation.rule}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color =
    score < 0.2 ? "bg-red-500" : score < 0.5 ? "bg-orange-500" : score < 0.7 ? "bg-amber-500" : "bg-green-500";

  return (
    <div className="flex items-center gap-1.5 flex-1 max-w-[120px]">
      <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] font-mono text-slate-400 w-8">{score.toFixed(2)}</span>
    </div>
  );
}
