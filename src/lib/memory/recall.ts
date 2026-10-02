import type { SessionRecord, MemorySnapshot } from "./types";
import { getSnapshot, getLastSessionForTopic } from "./store";

export interface RecallContext {
  isReturningUser: boolean;
  timeSinceLastSession: string;
  welcomeMessage: string;
  lastTopic: string | null;
  lastLesson: number;
  activePlanSummary: string | null;
  sessionCount: number;
  totalMinutes: number;
  suggestedAction: "continue" | "review" | "new" | "planned";
}

function formatTimeSince(ms: number): string {
  const minutes = Math.floor(ms / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days === 1) return "yesterday";
  if (days < 7) return `${days} days ago`;
  const weeks = Math.floor(days / 7);
  return `${weeks} week${weeks > 1 ? "s" : ""} ago`;
}

export function buildRecallContext(topicId?: string): RecallContext {
  const snapshot = getSnapshot();
  const { profile, lastSession, activePlan } = snapshot;

  const isReturningUser = profile.totalSessions > 0;
  const timeSinceMs = Date.now() - profile.lastSessionAt;
  const timeSinceLastSession = formatTimeSince(timeSinceMs);

  const topicSession = topicId
    ? getLastSessionForTopic(topicId)
    : lastSession;

  let suggestedAction: RecallContext["suggestedAction"] = "new";
  if (activePlan) {
    suggestedAction = "planned";
  } else if (topicSession && timeSinceMs > 86400000) {
    suggestedAction = "review";
  } else if (topicSession) {
    suggestedAction = "continue";
  }

  const welcomeMessage = buildWelcomeMessage(
    snapshot,
    topicSession,
    timeSinceLastSession,
    suggestedAction
  );

  const activePlanSummary = activePlan
    ? `Plan: ${activePlan.goals.join(", ")}${activePlan.notes ? ` — ${activePlan.notes}` : ""}`
    : null;

  return {
    isReturningUser,
    timeSinceLastSession,
    welcomeMessage,
    lastTopic: topicSession?.topicId ?? null,
    lastLesson: topicSession?.lessonIndex ?? 0,
    activePlanSummary,
    sessionCount: profile.totalSessions,
    totalMinutes: profile.totalTimeMinutes,
    suggestedAction,
  };
}

function buildWelcomeMessage(
  snapshot: MemorySnapshot,
  lastTopicSession: SessionRecord | null,
  timeSince: string,
  action: RecallContext["suggestedAction"]
): string {
  const { profile, activePlan } = snapshot;

  if (!snapshot.lastSession) {
    return "Welcome to SensAI! This is your first session. Pick a topic to get started.";
  }

  const parts: string[] = [];

  parts.push(`Welcome back! Last session was **${timeSince}**.`);

  if (profile.totalSessions > 1) {
    parts.push(
      `You've done **${profile.totalSessions} sessions** totaling **${profile.totalTimeMinutes} minutes**.`
    );
  }

  if (lastTopicSession) {
    const concepts = lastTopicSession.conceptsCovered;
    if (concepts.length > 0) {
      parts.push(`Last time you covered: ${concepts.join(", ")}.`);
    }
    if (lastTopicSession.struggles.length > 0) {
      parts.push(
        `You were working through: ${lastTopicSession.struggles.join(", ")}.`
      );
    }
  }

  if (activePlan) {
    parts.push(`\n**Your plan:** ${activePlan.goals.join(", ")}`);
    if (activePlan.notes) parts.push(`Notes: ${activePlan.notes}`);
  }

  switch (action) {
    case "planned":
      parts.push("\nReady to start what you planned?");
      break;
    case "review":
      parts.push(
        "\nIt's been a while — want a quick recap before we continue?"
      );
      break;
    case "continue":
      parts.push("\nPick up where you left off?");
      break;
    default:
      parts.push("\nChoose a topic to begin.");
  }

  return parts.join(" ");
}

export function buildContextForAI(topicId: string): string {
  const snapshot = getSnapshot();
  const { profile } = snapshot;
  const topicSession = getLastSessionForTopic(topicId);

  const lines: string[] = [];

  lines.push(`Returning user: ${profile.totalSessions} past sessions.`);
  lines.push(
    `Time since last session: ${formatTimeSince(Date.now() - profile.lastSessionAt)}.`
  );

  if (profile.strengths.length > 0) {
    lines.push(`Strengths: ${profile.strengths.join(", ")}.`);
  }
  if (profile.struggles.length > 0) {
    lines.push(`Struggles with: ${profile.struggles.join(", ")}.`);
  }

  lines.push(
    `Preferred pace: ${profile.preferences.pace}, depth: ${profile.preferences.depth}.`
  );
  lines.push(
    `Average focus rate: ${(profile.emotionPatterns.avgFocusRate * 100).toFixed(0)}%.`
  );

  if (topicSession) {
    lines.push(
      `Last session on this topic: covered ${topicSession.conceptsCovered.join(", ") || "intro"}, lesson ${topicSession.lessonIndex + 1}.`
    );
    if (topicSession.struggles.length > 0) {
      lines.push(
        `Struggled with: ${topicSession.struggles.join(", ")}.`
      );
    }
  }

  const plan = snapshot.activePlan;
  if (plan && plan.topicId === topicId) {
    lines.push(`User's plan: ${plan.goals.join(", ")}.`);
  }

  return lines.join("\n");
}
