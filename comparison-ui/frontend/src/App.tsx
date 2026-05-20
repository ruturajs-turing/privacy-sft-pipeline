import { useState, useRef, useCallback } from "react";
import { useTrajectoryList, useTrajectoryData } from "./hooks/useTrajectory";
import { TrajectorySelector } from "./components/TrajectorySelector";
import { StatsBar } from "./components/StatsBar";
import { TrajectoryColumn, RLHFColumn } from "./components/ColumnPanel";
import type { ColumnRef } from "./components/ColumnPanel";

function App() {
  const { trajectories, loading: listLoading } = useTrajectoryList();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const { original, privacy, rlhf, report, loading: dataLoading } = useTrajectoryData(selectedId);

  const col1Ref = useRef<ColumnRef>(null);
  const col2Ref = useRef<ColumnRef>(null);
  const col3Ref = useRef<ColumnRef>(null);

  const handleScroll = useCallback((_scrollTop: number, _source: number) => {
    // Sync scroll disabled — each column scrolls independently
    // Enable by uncommenting:
    // if (source !== 0) col1Ref.current?.setScrollTop(scrollTop);
    // if (source !== 1) col2Ref.current?.setScrollTop(scrollTop);
    // if (source !== 2) col3Ref.current?.setScrollTop(scrollTop);
  }, []);

  return (
    <div className="h-screen flex flex-col bg-slate-950">
      {/* Top Bar */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm">
        <div className="flex items-center gap-4">
          <h1 className="text-base font-bold text-slate-100">
            OpenClaw Privacy — Comparison Viewer
          </h1>
          <TrajectorySelector
            trajectories={trajectories}
            selected={selectedId}
            onSelect={setSelectedId}
            loading={listLoading}
          />
        </div>
        <StatsBar report={report} />
      </header>

      {/* Loading state */}
      {dataLoading && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-slate-400 text-sm animate-pulse">Loading trajectory data...</div>
        </div>
      )}

      {/* Empty state */}
      {!selectedId && !dataLoading && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="text-slate-500 text-lg mb-2">Select a trajectory to compare</div>
            <div className="text-slate-600 text-sm">
              View the original SFT, privacy rewrite, and RLHF pairs side by side
            </div>
          </div>
        </div>
      )}

      {/* 3-column comparison */}
      {selectedId && !dataLoading && (
        <div className="flex-1 grid grid-cols-3 divide-x divide-slate-800 min-h-0">
          <TrajectoryColumn
            ref={col1Ref}
            title="Original SFT"
            titleColor="text-slate-300"
            events={original}
            onScroll={(st) => handleScroll(st, 0)}
          />
          <TrajectoryColumn
            ref={col2Ref}
            title="Privacy Rewrite"
            titleColor="text-emerald-400"
            events={privacy}
            onScroll={(st) => handleScroll(st, 1)}
          />
          <RLHFColumn
            ref={col3Ref}
            pairs={rlhf}
            onScroll={(st) => handleScroll(st, 2)}
          />
        </div>
      )}
    </div>
  );
}

export default App;
