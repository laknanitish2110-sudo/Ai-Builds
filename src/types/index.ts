export type EmotionLabel =
  | "happy"
  | "sad"
  | "angry"
  | "surprised"
  | "fearful"
  | "disgusted"
  | "neutral";

export type LearningState =
  | "confused"
  | "frustrated"
  | "bored"
  | "focused"
  | "excited"
  | "struggling"
  | "neutral";

export interface EmotionReading {
  timestamp: number;
  emotions: Record<EmotionLabel, number>;
  dominant: EmotionLabel;
  confidence: number;
}

export interface LearningStateReading {
  state: LearningState;
  confidence: number;
  timestamp: number;
  suggestion: TutorAdaptation;
}

export interface TutorAdaptation {
  pacing: "slower" | "normal" | "faster";
  complexity: "simpler" | "normal" | "deeper";
  encouragement: "high" | "normal" | "low";
  format: "visual" | "text" | "interactive" | "mixed";
  shouldBreak: boolean;
  message?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "tutor" | "system";
  content: string;
  timestamp: number;
  emotionContext?: LearningState;
  adaptation?: TutorAdaptation;
}

export interface LearningSession {
  id: string;
  topic: string;
  startedAt: number;
  emotionHistory: EmotionReading[];
  stateHistory: LearningStateReading[];
  messages: ChatMessage[];
  stats: SessionStats;
}

export interface SessionStats {
  totalTime: number;
  focusPercentage: number;
  confusionPoints: number;
  adaptations: number;
  topicsCompleted: number;
}

export interface Topic {
  id: string;
  title: string;
  description: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  icon: string;
  lessons: Lesson[];
}

export interface Lesson {
  id: string;
  title: string;
  content: string;
  examples: string[];
  exercises: string[];
}
