import type { TrajectoryEvent, ContentBlock } from "../types";
import { ThinkingBlock } from "./ThinkingBlock";
import { ToolCallBlock } from "./ToolCallBlock";

interface Props {
  event: TrajectoryEvent;
  turnIndex?: number;
}

export function MessageBubble({ event, turnIndex }: Props) {
  const msg = event.message;
  if (!msg) return null;

  const role = msg.role;
  const content = msg.content;

  if (role === "user") {
    const text = extractText(content);
    return (
      <div className="mb-3">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] font-bold uppercase tracking-wider text-violet-400">User</span>
          {turnIndex !== undefined && (
            <span className="text-[10px] text-slate-600">Turn {turnIndex}</span>
          )}
        </div>
        <div className="bg-violet-900/20 border border-violet-800/30 rounded-lg px-3 py-2 text-sm text-slate-200 leading-relaxed">
          {text}
        </div>
      </div>
    );
  }

  if (role === "assistant") {
    const blocks = Array.isArray(content) ? content : [];
    return (
      <div className="mb-3">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] font-bold uppercase tracking-wider text-sky-400">Assistant</span>
          {turnIndex !== undefined && (
            <span className="text-[10px] text-slate-600">Turn {turnIndex}</span>
          )}
        </div>
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg px-3 py-2 space-y-1">
          {blocks.map((block, i) => renderBlock(block, i))}
        </div>
      </div>
    );
  }

  if (role === "toolResult") {
    const text = extractText(content);
    return (
      <div className="mb-3">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] font-bold uppercase tracking-wider text-amber-400">
            Tool Result
          </span>
          {msg.toolName && (
            <span className="text-[10px] text-slate-500 font-mono">{msg.toolName}</span>
          )}
        </div>
        <div
          className={`border rounded-lg px-3 py-2 text-xs font-mono max-h-32 overflow-y-auto whitespace-pre-wrap ${
            msg.isError
              ? "border-red-800/40 bg-red-950/20 text-red-300"
              : "border-slate-700/40 bg-slate-900/40 text-slate-400"
          }`}
        >
          {text.slice(0, 500)}
          {text.length > 500 && "..."}
        </div>
      </div>
    );
  }

  return null;
}

function renderBlock(block: ContentBlock, idx: number) {
  if (block.type === "thinking" && block.thinking) {
    return <ThinkingBlock key={idx} thinking={block.thinking} />;
  }
  if (block.type === "text" && block.text) {
    return (
      <div key={idx} className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
        {block.text}
      </div>
    );
  }
  if (block.type === "toolCall" && block.name) {
    return (
      <ToolCallBlock
        key={idx}
        name={block.name}
        arguments={block.arguments || {}}
      />
    );
  }
  return null;
}

function extractText(content: ContentBlock[] | string | unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((b) => b.type === "text")
      .map((b) => b.text || "")
      .join("\n");
  }
  return "";
}
