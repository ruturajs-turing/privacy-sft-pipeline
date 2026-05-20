import { useRef, useEffect, forwardRef, useImperativeHandle } from "react";
import type { TrajectoryEvent, RLHFPair } from "../types";
import { MessageBubble } from "./MessageBubble";
import { RLHFStepCard } from "./RLHFStepCard";

interface TrajectoryColumnProps {
  title: string;
  titleColor: string;
  events: TrajectoryEvent[];
  onScroll?: (scrollTop: number) => void;
  scrollTop?: number;
}

export interface ColumnRef {
  setScrollTop: (top: number) => void;
}

export const TrajectoryColumn = forwardRef<ColumnRef, TrajectoryColumnProps>(
  ({ title, titleColor, events, onScroll, scrollTop }, ref) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const isUserScroll = useRef(true);

    useImperativeHandle(ref, () => ({
      setScrollTop: (top: number) => {
        if (containerRef.current) {
          isUserScroll.current = false;
          containerRef.current.scrollTop = top;
          setTimeout(() => { isUserScroll.current = true; }, 50);
        }
      },
    }));

    useEffect(() => {
      if (scrollTop !== undefined && containerRef.current) {
        isUserScroll.current = false;
        containerRef.current.scrollTop = scrollTop;
        setTimeout(() => { isUserScroll.current = true; }, 50);
      }
    }, [scrollTop]);

    const handleScroll = () => {
      if (isUserScroll.current && onScroll && containerRef.current) {
        onScroll(containerRef.current.scrollTop);
      }
    };

    const messageEvents = events.filter((e) => e.type === "message");
    let turnCounter = 0;

    return (
      <div className="flex flex-col h-full min-h-0 min-w-0">
        <div className={`shrink-0 px-3 py-2 border-b border-slate-700 bg-slate-900`}>
          <h2 className={`text-sm font-bold uppercase tracking-wider ${titleColor}`}>
            {title}
          </h2>
          <span className="text-[10px] text-slate-500">{messageEvents.length} events</span>
        </div>
        <div
          ref={containerRef}
          className="flex-1 overflow-y-auto min-h-0 p-3 space-y-1"
          onScroll={handleScroll}
        >
          {messageEvents.map((event, idx) => {
            const turn = event.message?.role === "user" ? turnCounter++ : undefined;
            return <MessageBubble key={idx} event={event} turnIndex={turn !== undefined ? turn : undefined} />;
          })}
          {messageEvents.length === 0 && (
            <div className="text-center text-slate-600 text-sm mt-8">No data available</div>
          )}
        </div>
      </div>
    );
  }
);

interface RLHFColumnProps {
  pairs: RLHFPair[];
  onScroll?: (scrollTop: number) => void;
  scrollTop?: number;
}

export const RLHFColumn = forwardRef<ColumnRef, RLHFColumnProps>(
  ({ pairs, onScroll, scrollTop }, ref) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const isUserScroll = useRef(true);

    useImperativeHandle(ref, () => ({
      setScrollTop: (top: number) => {
        if (containerRef.current) {
          isUserScroll.current = false;
          containerRef.current.scrollTop = top;
          setTimeout(() => { isUserScroll.current = true; }, 50);
        }
      },
    }));

    useEffect(() => {
      if (scrollTop !== undefined && containerRef.current) {
        isUserScroll.current = false;
        containerRef.current.scrollTop = scrollTop;
        setTimeout(() => { isUserScroll.current = true; }, 50);
      }
    }, [scrollTop]);

    const handleScroll = () => {
      if (isUserScroll.current && onScroll && containerRef.current) {
        onScroll(containerRef.current.scrollTop);
      }
    };

    return (
      <div className="flex flex-col h-full min-h-0 min-w-0">
        <div className="shrink-0 px-3 py-2 border-b border-slate-700 bg-slate-900">
          <h2 className="text-sm font-bold uppercase tracking-wider text-rose-400">
            RLHF Pairs
          </h2>
          <span className="text-[10px] text-slate-500">
            {pairs.length} decision points, {pairs.reduce((s, p) => s + p.rejected.length, 0)} rejected
          </span>
        </div>
        <div
          ref={containerRef}
          className="flex-1 overflow-y-auto min-h-0 p-3"
          onScroll={handleScroll}
        >
          {pairs.map((pair, idx) => (
            <RLHFStepCard key={idx} pair={pair} />
          ))}
          {pairs.length === 0 && (
            <div className="text-center text-slate-600 text-sm mt-8">No RLHF pairs generated</div>
          )}
        </div>
      </div>
    );
  }
);
