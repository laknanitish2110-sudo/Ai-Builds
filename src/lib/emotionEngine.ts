import type {
  EmotionLabel,
  EmotionReading,
  LearningState,
  LearningStateReading,
  TutorAdaptation,
} from "@/types";

const EMOTION_WINDOW_SIZE = 10;
const CONFUSION_THRESHOLD = 0.4;
const FRUSTRATION_THRESHOLD = 0.5;
const BOREDOM_THRESHOLD = 0.6;

export function mapEmotionToLearningState(
  history: EmotionReading[]
): LearningStateReading {
  const recent = history.slice(-EMOTION_WINDOW_SIZE);
  if (recent.length === 0) {
    return {
      state: "neutral",
      confidence: 1,
      timestamp: Date.now(),
      suggestion: getAdaptation("neutral"),
    };
  }

  const avgEmotions = averageEmotions(recent);
  const dominantPattern = analyzeDominantPattern(recent);
  const volatility = calculateVolatility(recent);

  let state: LearningState = "neutral";
  let confidence = 0.5;

  if (
    avgEmotions.surprised > CONFUSION_THRESHOLD &&
    avgEmotions.fearful > 0.2
  ) {
    state = "confused";
    confidence = (avgEmotions.surprised + avgEmotions.fearful) / 2;
  } else if (
    avgEmotions.angry > FRUSTRATION_THRESHOLD ||
    (avgEmotions.disgusted > 0.3 && avgEmotions.sad > 0.2)
  ) {
    state = "frustrated";
    confidence = Math.max(avgEmotions.angry, avgEmotions.disgusted);
  } else if (
    avgEmotions.neutral > BOREDOM_THRESHOLD &&
    volatility < 0.1
  ) {
    state = "bored";
    confidence = avgEmotions.neutral * (1 - volatility);
  } else if (avgEmotions.happy > 0.4 && avgEmotions.surprised > 0.2) {
    state = "excited";
    confidence = (avgEmotions.happy + avgEmotions.surprised) / 2;
  } else if (
    dominantPattern === "neutral" &&
    volatility < 0.2 &&
    avgEmotions.neutral > 0.3
  ) {
    state = "focused";
    confidence = 0.7;
  } else if (
    avgEmotions.sad > 0.3 &&
    avgEmotions.fearful > 0.2
  ) {
    state = "struggling";
    confidence = (avgEmotions.sad + avgEmotions.fearful) / 2;
  }

  return {
    state,
    confidence,
    timestamp: Date.now(),
    suggestion: getAdaptation(state),
  };
}

function averageEmotions(
  readings: EmotionReading[]
): Record<EmotionLabel, number> {
  const sum: Record<EmotionLabel, number> = {
    happy: 0,
    sad: 0,
    angry: 0,
    surprised: 0,
    fearful: 0,
    disgusted: 0,
    neutral: 0,
  };

  for (const r of readings) {
    for (const key of Object.keys(sum) as EmotionLabel[]) {
      sum[key] += r.emotions[key] || 0;
    }
  }

  const count = readings.length;
  for (const key of Object.keys(sum) as EmotionLabel[]) {
    sum[key] /= count;
  }

  return sum;
}

function analyzeDominantPattern(readings: EmotionReading[]): EmotionLabel {
  const counts: Record<EmotionLabel, number> = {
    happy: 0,
    sad: 0,
    angry: 0,
    surprised: 0,
    fearful: 0,
    disgusted: 0,
    neutral: 0,
  };

  for (const r of readings) {
    counts[r.dominant]++;
  }

  let max: EmotionLabel = "neutral";
  let maxCount = 0;
  for (const [emotion, count] of Object.entries(counts)) {
    if (count > maxCount) {
      max = emotion as EmotionLabel;
      maxCount = count;
    }
  }

  return max;
}

function calculateVolatility(readings: EmotionReading[]): number {
  if (readings.length < 2) return 0;

  let switches = 0;
  for (let i = 1; i < readings.length; i++) {
    if (readings[i].dominant !== readings[i - 1].dominant) {
      switches++;
    }
  }

  return switches / (readings.length - 1);
}

export function getAdaptation(state: LearningState): TutorAdaptation {
  const adaptations: Record<LearningState, TutorAdaptation> = {
    confused: {
      pacing: "slower",
      complexity: "simpler",
      encouragement: "high",
      format: "visual",
      shouldBreak: false,
      message:
        "I notice this might be tricky. Let me break it down differently...",
    },
    frustrated: {
      pacing: "slower",
      complexity: "simpler",
      encouragement: "high",
      format: "interactive",
      shouldBreak: true,
      message:
        "Let's take a step back. You're doing great — this is a tough concept.",
    },
    bored: {
      pacing: "faster",
      complexity: "deeper",
      encouragement: "low",
      format: "interactive",
      shouldBreak: false,
      message:
        "You've got this down! Let me challenge you with something harder...",
    },
    focused: {
      pacing: "normal",
      complexity: "normal",
      encouragement: "normal",
      format: "text",
      shouldBreak: false,
    },
    excited: {
      pacing: "faster",
      complexity: "deeper",
      encouragement: "normal",
      format: "mixed",
      shouldBreak: false,
      message: "Love the energy! Let's dive deeper...",
    },
    struggling: {
      pacing: "slower",
      complexity: "simpler",
      encouragement: "high",
      format: "visual",
      shouldBreak: true,
      message:
        "No worries — everyone finds this part challenging. Let me try a different approach.",
    },
    neutral: {
      pacing: "normal",
      complexity: "normal",
      encouragement: "normal",
      format: "text",
      shouldBreak: false,
    },
  };

  return adaptations[state];
}

export function generateEmotionInsight(
  history: LearningStateReading[]
): string {
  if (history.length < 5) return "Gathering emotion data...";

  const recent = history.slice(-20);
  const stateCounts: Record<string, number> = {};

  for (const r of recent) {
    stateCounts[r.state] = (stateCounts[r.state] || 0) + 1;
  }

  const dominant = Object.entries(stateCounts).sort((a, b) => b[1] - a[1])[0];

  const insights: Record<string, string> = {
    confused:
      "You've been showing signs of confusion. The tutor is simplifying explanations.",
    frustrated:
      "Detecting some frustration. Consider taking a short break.",
    bored:
      "You seem ready for more challenge. Increasing difficulty.",
    focused:
      "Great focus! You're in the zone — keep going.",
    excited:
      "High engagement detected! Perfect learning state.",
    struggling:
      "This topic seems challenging. Extra support has been activated.",
    neutral:
      "Steady learning pace. Everything looks good.",
  };

  return insights[dominant[0]] || "Analyzing your learning pattern...";
}
