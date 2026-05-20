interface Props {
  name: string;
  arguments: Record<string, unknown>;
  result?: { content: string; is_error: boolean } | null;
}

export function ToolCallBlock({ name, arguments: args, result }: Props) {
  return (
    <div className="border border-blue-800/50 rounded-md overflow-hidden my-1">
      <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-900/30">
        <span className="text-blue-400 text-xs font-mono font-bold">{name}</span>
      </div>
      <div className="px-3 py-2 text-xs font-mono text-blue-200/80 bg-blue-950/20 whitespace-pre-wrap max-h-32 overflow-y-auto">
        {JSON.stringify(args, null, 2)}
      </div>
      {result && (
        <div
          className={`px-3 py-1.5 text-xs border-t ${
            result.is_error
              ? "border-red-800/50 bg-red-950/20 text-red-300"
              : "border-slate-700 bg-slate-900/50 text-slate-400"
          } max-h-24 overflow-y-auto whitespace-pre-wrap`}
        >
          {result.content.slice(0, 300)}
          {result.content.length > 300 && "..."}
        </div>
      )}
    </div>
  );
}
