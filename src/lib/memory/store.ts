import type {
  LearnerProfile,
  SessionRecord,
  SessionPlan,
  MemorySnapshot,
} from "./types";

const KEYS = {
  profile: "sensai_profile",
  sessions: "sensai_sessions",
  plan: "sensai_plan",
} as const;

const MAX_SESSIONS = 50;

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function write(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // storage full or unavailable
  }
}

function createDefaultProfile(): LearnerProfile {
  return {
    id: crypto.randomUUID(),
    createdAt: Date.now(),
    lastSessionAt: Date.now(),
    totalSessions: 0,
    totalTimeMinutes: 0,
    strengths: [],
    struggles: [],
    preferences: { pace: "normal", depth: "normal", style: "text" },
    emotionPatterns: {
      dominantStates: [],
      avgFocusRate: 0,
      frustrationTriggers: [],
    },
  };
}

export function getProfile(): LearnerProfile {
  return read<LearnerProfile>(KEYS.profile, createDefaultProfile());
}

export function saveProfile(profile: LearnerProfile): void {
  write(KEYS.profile, profile);
}

export function getSessions(): SessionRecord[] {
  return read<SessionRecord[]>(KEYS.sessions, []);
}

export function getLastSession(): SessionRecord | null {
  const sessions = getSessions();
  return sessions.length > 0 ? sessions[sessions.length - 1] : null;
}

export function getLastSessionForTopic(topicId: string): SessionRecord | null {
  const sessions = getSessions();
  for (let i = sessions.length - 1; i >= 0; i--) {
    if (sessions[i].topicId === topicId) return sessions[i];
  }
  return null;
}

export function saveSession(session: SessionRecord): void {
  const sessions = getSessions();
  const existing = sessions.findIndex((s) => s.id === session.id);
  if (existing >= 0) {
    sessions[existing] = session;
  } else {
    sessions.push(session);
  }
  write(KEYS.sessions, sessions.slice(-MAX_SESSIONS));

  const profile = getProfile();
  profile.lastSessionAt = Date.now();
  profile.totalSessions = sessions.length;
  if (session.durationMinutes > 0) {
    profile.totalTimeMinutes += session.durationMinutes;
  }
  saveProfile(profile);
}

export function getActivePlan(): SessionPlan | null {
  return read<SessionPlan | null>(KEYS.plan, null);
}

export function savePlan(plan: SessionPlan | null): void {
  write(KEYS.plan, plan);
}

export function getSnapshot(): MemorySnapshot {
  const profile = getProfile();
  const sessions = getSessions();
  const activePlan = getActivePlan();
  const lastSession = sessions.length > 0 ? sessions[sessions.length - 1] : null;
  return { profile, sessions, activePlan, lastSession };
}

export function updateStrengths(topic: string): void {
  const profile = getProfile();
  if (!profile.strengths.includes(topic)) {
    profile.strengths.push(topic);
    profile.strengths = profile.strengths.slice(-10);
  }
  saveProfile(profile);
}

export function updateStruggles(topic: string): void {
  const profile = getProfile();
  if (!profile.struggles.includes(topic)) {
    profile.struggles.push(topic);
    profile.struggles = profile.struggles.slice(-10);
  }
  saveProfile(profile);
}

export function updateEmotionPatterns(
  dominantState: string,
  focusRate: number
): void {
  const profile = getProfile();
  profile.emotionPatterns.dominantStates.push(dominantState);
  profile.emotionPatterns.dominantStates =
    profile.emotionPatterns.dominantStates.slice(-20);

  const rates = [profile.emotionPatterns.avgFocusRate, focusRate];
  profile.emotionPatterns.avgFocusRate =
    rates.reduce((a, b) => a + b, 0) / rates.length;

  if (dominantState === "frustrated" || dominantState === "struggling") {
    profile.emotionPatterns.frustrationTriggers.push(
      new Date().toISOString()
    );
    profile.emotionPatterns.frustrationTriggers =
      profile.emotionPatterns.frustrationTriggers.slice(-10);
  }

  saveProfile(profile);
}
