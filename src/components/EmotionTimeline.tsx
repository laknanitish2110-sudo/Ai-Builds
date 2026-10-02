"use client";

import type { EmotionReading, LearningStateReading } from "@/types";

interface EmotionTimelineProps {
  emotionHistory: EmotionReading[];
  learningStateHistory: LearningStateReading[];
  insight: string;
}

const STATE_COLORS: Record<string, string> = {
  confused: "#f97316",
  frustrated: "#ef4444",
  bored: "#a855f7",
  focused: "#06b6d4",
  excited: "#22c55e",
  struggling: "#3b82f6",
  neutral: "#6b7280",
};

export default function EmotionTimeline({
  emotionHistory,
  learningStateHistory,
  insight,
}: EmotionTimelineProps) {
  const recentStates = learningStateHistory.slice(-30);
  const maxPoints = 30;

  return (
    <div className="glass rounded-2xl p-4 space-y-4">
      <h3 className="text-sm font-semibold text-gray-300">
        Emotion Timeline
      </h3>

      <div className="h-24 flex items-end gap-0.5">
        {recentStates.length === 0 ? (
          <div className="w-full h-full flex items-center justify-center">
            <p className="text-xs text-gray-600">
              Waiting for emotion data...
            </p>
          </div>
        ) : (
          recentStates.slice(-maxPoints).map((state, i) => (
            <div
              key={i}
              className="flex-1 rounded-t transition-all duration-300"
              style={{
                height: `${state.confidence * 100}%`,
                backgroundColor:
                  STATE_COLORS[state.state] || STATE_COLORS.neutral,
                opacity: 0.4 + (i / maxPoints) * 0.6,
                minWidth: "3px",
              }}
              title={`${state.state} (${(state.confidence * 100).toFixed(0)}%)`}
            />
          ))
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        {Object.entries(STATE_COLORS).map(([state, color]) => (
          <div key={state} className="flex items-center gap-1">
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: color }}
            />
            <span className="text-xs text-gray-500 capitalize">
              {state}
            </span>
          </div>
        ))}
      </div>

      <div className="bg-gray-800/50 rounded-xl p-3">
        <p className="text-xs text-gray-400 leading-relaxed">{insight}</p>
      </div>

      {emotionHistory.length > 10 && (
        <div className="space-y-2">
          <h4 className="text-xs font-medium text-gray-400">
            Session Summary
          </h4>
          <div className="grid grid-cols-2 gap-2">
            <StatCard
              label="Readings"
              value={emotionHistory.length.toString()}
            />
            <StatCard
              label="Focus Rate"
              value={`${calculateFocusRate(learningStateHistory)}%`}
            />
            <StatCard
              label="Adaptations"
              value={countAdaptations(learningStateHistory).toString()}
            />
            <StatCard
              label="Duration"
              value={formatDuration(emotionHistory)}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-800/30 rounded-lg p-2.5">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="text-lg font-semibold text-white">{value}</p>
    </div>
  );
}

function calculateFocusRate(states: LearningStateReading[]): number {
  if (states.length === 0) return 0;
  const focused = states.filter(
    (s) => s.state === "focused" || s.state === "excited"
  ).length;
  return Math.round((focused / states.length) * 100);
}

function countAdaptations(states: LearningStateReading[]): number {
  return states.filter(
    (s) => s.state !== "neutral" && s.state !== "focused"
  ).length;
}

function formatDuration(history: EmotionReading[]): string {
  if (history.length < 2) return "0s";
  const ms = history[history.length - 1].timestamp - history[0].timestamp;
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m`;
}
