"use client";

import type { EmotionLabel, EmotionReading } from "@/types";

interface EmotionDetectorProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  currentEmotion: EmotionReading | null;
  isActive: boolean;
  isLoading: boolean;
  error: string | null;
  onStart: () => void;
  onDemo: () => void;
}

const EMOTION_COLORS: Record<EmotionLabel, string> = {
  happy: "#22c55e",
  sad: "#3b82f6",
  angry: "#ef4444",
  surprised: "#f59e0b",
  fearful: "#8b5cf6",
  disgusted: "#14b8a6",
  neutral: "#6b7280",
};

const EMOTION_EMOJI: Record<EmotionLabel, string> = {
  happy: "😊",
  sad: "😢",
  angry: "😠",
  surprised: "😲",
  fearful: "😰",
  disgusted: "🤢",
  neutral: "😐",
};

export default function EmotionDetector({
  videoRef,
  canvasRef,
  currentEmotion,
  isActive,
  isLoading,
  error,
  onStart,
  onDemo,
}: EmotionDetectorProps) {
  return (
    <div className="glass rounded-2xl p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">
          Emotion Detection
        </h3>
        <div
          className={`w-2 h-2 rounded-full ${
            isActive ? "bg-green-400 animate-pulse" : "bg-gray-600"
          }`}
        />
      </div>

      <div className="relative aspect-video rounded-xl overflow-hidden bg-gray-900">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
        />

        {!isActive && !isLoading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <div className="text-4xl">🎥</div>
            <p className="text-sm text-gray-400 text-center px-4">
              Enable camera for real-time emotion detection
            </p>
            <div className="flex gap-2">
              <button
                onClick={onStart}
                className="px-4 py-2 rounded-lg bg-sensai-600 hover:bg-sensai-500 text-white text-sm font-medium transition"
              >
                Enable Camera
              </button>
              <button
                onClick={onDemo}
                className="px-4 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-white text-sm font-medium transition"
              >
                Demo Mode
              </button>
            </div>
          </div>
        )}

        {isLoading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-gray-900/80">
            <div className="w-8 h-8 border-2 border-sensai-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-sm text-gray-400">
              Loading emotion models...
            </p>
          </div>
        )}

        {error && (
          <div className="absolute bottom-2 left-2 right-2">
            <div className="bg-yellow-500/20 border border-yellow-500/30 rounded-lg px-3 py-2">
              <p className="text-xs text-yellow-300">
                Camera unavailable — using demo mode
              </p>
            </div>
          </div>
        )}
      </div>

      {isActive && currentEmotion && (
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <span className="text-2xl">
              {EMOTION_EMOJI[currentEmotion.dominant]}
            </span>
            <div>
              <p className="text-sm font-medium capitalize">
                {currentEmotion.dominant}
              </p>
              <p className="text-xs text-gray-500">
                {(currentEmotion.confidence * 100).toFixed(0)}% confidence
              </p>
            </div>
          </div>

          <div className="space-y-1.5">
            {(
              Object.entries(currentEmotion.emotions) as [
                EmotionLabel,
                number,
              ][]
            )
              .sort((a, b) => b[1] - a[1])
              .slice(0, 4)
              .map(([emotion, value]) => (
                <div key={emotion} className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 w-20 capitalize">
                    {emotion}
                  </span>
                  <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${value * 100}%`,
                        backgroundColor: EMOTION_COLORS[emotion],
                      }}
                    />
                  </div>
                  <span className="text-xs text-gray-600 w-8 text-right">
                    {(value * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
