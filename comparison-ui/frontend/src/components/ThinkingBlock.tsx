import { useState } from "react";

interface Props {
  thinking: string;
  defaultOpen?: boolean;
}

export function ThinkingBlock({ thinking, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);

  if (!thinking) return null;

  return (
    <div className="border border-emerald-800/50 rounded-md overflow-hidden my-1">
      <button
        className="w-full flex items-center gap-2 px-3 py-1.5 text-xs bg-emerald-900/30 text-emerald-400 hover:bg-emerald-900/50 transition-colors"
        onClick={() => setOpen(!open)}
      >
        <span className="text-[10px]">{open ? "▼" : "▶"}</span>
        <span className="font-medium">Thinking</span>
        <span className="text-emerald-600 ml-auto">{thinking.length} chars</span>
      </button>
      {open && (
        <div className="px-3 py-2 text-xs text-emerald-200/80 bg-emerald-950/30 whitespace-pre-wrap max-h-48 overflow-y-auto leading-relaxed">
          {thinking}
        </div>
      )}
    </div>
  );
}
