export interface TrajectoryEvent {
  type: string;
  id?: string;
  message?: {
    role: "user" | "assistant" | "toolResult";
    content: ContentBlock[] | string;
    timestamp?: number;
    toolCallId?: string;
    toolName?: string;
    isError?: boolean;
  };
  createdAt?: number;
}

export interface ContentBlock {
  type: "text" | "thinking" | "toolCall";
  text?: string;
  thinking?: string;
  id?: string;
  name?: string;
  arguments?: Record<string, unknown>;
}

export interface RLHFPair {
  task_id: string;
  submission_id: string;
  turn_index: number;
  step_criticality: number;
  pair_level: string;
  decision_branch: string;
  data_level_involved: string;
  tool_tier_involved: string;
  context: Array<{ role: string; content: string }>;
  chosen: {
    thinking: string;
    tool_call: Record<string, unknown> | null;
    tool_response: { content: string; is_error: boolean } | null;
    assistant_response: string;
    reward_score: number;
  };
  rejected: RejectedStep[];
}

export interface RejectedStep {
  thinking: string;
  tool_call: Record<string, unknown> | null;
  tool_response: { content: string; is_error: boolean } | null;
  assistant_response: string;
  failure_mode: string;
  privacy_violation: {
    flag: boolean;
    rule: string | null;
    severity: string | null;
    data_level: string;
    tool_tier: string;
  };
  reward_score: number;
  perturbation_type: string;
}

export interface RLHFReport {
  task_id: string;
  submission_id: string;
  total_pairs: number;
  pairs_by_failure_mode: Record<string, number>;
  pairs_by_level: Record<string, number>;
  avg_reward_score: number;
  over_refusal_ratio: number;
  violation_ratio: number;
}

export interface TrajectoryMeta {
  submission_id: string;
  task_id: string;
  has_rlhf: boolean;
  status: string;
}
