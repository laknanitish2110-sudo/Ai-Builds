export interface LearnerProfile {
  id: string;
  createdAt: number;
  lastSessionAt: number;
  totalSessions: number;
  totalTimeMinutes: number;
  strengths: string[];
  struggles: string[];
  preferences: {
    pace: "slow" | "normal" | "fast";
    depth: "simple" | "normal" | "deep";
    style: "visual" | "text" | "interactive";
  };
  emotionPatterns: {
    dominantStates: string[];
    avgFocusRate: number;
    frustrationTriggers: string[];
  };
}

export interface SessionRecord {
  id: string;
  topicId: string;
  lessonIndex: number;
  startedAt: number;
  endedAt: number | null;
  durationMinutes: number;
  messagesCount: number;
  emotionSummary: {
    dominantState: string;
    focusRate: number;
    stateBreakdown: Record<string, number>;
  };
  conceptsCovered: string[];
  struggles: string[];
  plan: SessionPlan | null;
  lastMessage: string;
}

export interface SessionPlan {
  createdAt: number;
  targetDate: string | null;
  goals: string[];
  topicId: string;
  lessonIndex: number;
  notes: string;
}

export interface MemorySnapshot {
  profile: LearnerProfile;
  sessions: SessionRecord[];
  activePlan: SessionPlan | null;
  lastSession: SessionRecord | null;
}
