import { useState, useEffect } from "react";
import type { TrajectoryEvent, RLHFPair, RLHFReport, TrajectoryMeta } from "../types";

const API_BASE = "/api";

export function useTrajectoryList() {
  const [trajectories, setTrajectories] = useState<TrajectoryMeta[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/trajectories`)
      .then((r) => r.json())
      .then((d) => setTrajectories(d.trajectories))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return { trajectories, loading };
}

export function useTrajectoryData(submissionId: string | null) {
  const [original, setOriginal] = useState<TrajectoryEvent[]>([]);
  const [privacy, setPrivacy] = useState<TrajectoryEvent[]>([]);
  const [rlhf, setRlhf] = useState<RLHFPair[]>([]);
  const [report, setReport] = useState<RLHFReport | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!submissionId) return;
    setLoading(true);

    Promise.all([
      fetch(`${API_BASE}/trajectory/${submissionId}/original`).then((r) => r.json()),
      fetch(`${API_BASE}/trajectory/${submissionId}/privacy`).then((r) => r.json()),
      fetch(`${API_BASE}/trajectory/${submissionId}/rlhf`).then((r) => r.json()),
      fetch(`${API_BASE}/trajectory/${submissionId}/metadata`).then((r) => r.json()),
    ])
      .then(([orig, priv, rlhfData, meta]) => {
        setOriginal(orig.events || []);
        setPrivacy(priv.events || []);
        setRlhf(rlhfData.pairs || []);
        setReport(meta.rlhf_report || null);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [submissionId]);

  return { original, privacy, rlhf, report, loading };
}
