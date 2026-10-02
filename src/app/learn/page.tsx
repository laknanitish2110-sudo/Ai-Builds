"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import EmotionDetector from "@/components/EmotionDetector";
import AITutor from "@/components/AITutor";
import EmotionTimeline from "@/components/EmotionTimeline";
import { useEmotionDetection } from "@/hooks/useEmotionDetection";
import {
  mapEmotionToLearningState,
  generateEmotionInsight,
} from "@/lib/emotionEngine";
import {
  generateAIResponse,
  generateTutorResponse,
  getLessonTitle,
  getTopicList,
  getTopicTitle,
  getWelcomeMessage,
} from "@/lib/tutorAdapter";
import type {
  ChatMessage,
  LearningState,
  LearningStateReading,
} from "@/types";
import Link from "next/link";

const TOPICS = getTopicList();
const TOPIC_ICONS: Record<string, string> = {
  python: "🐍",
  ai: "🤖",
  math: "📐",
};

export default function LearnPage() {
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [lessonIndex, setLessonIndex] = useState(0);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [learningState, setLearningState] = useState<LearningState>("neutral");
  const [stateHistory, setStateHistory] = useState<LearningStateReading[]>([]);
  const [isThinking, setIsThinking] = useState(false);

  const {
    videoRef,
    canvasRef,
    isLoading,
    isActive,
    error,
    currentEmotion,
    emotionHistory,
    startDetection,
    useDemoMode,
  } = useEmotionDetection({ interval: 1500 });

  useEffect(() => {
    if (emotionHistory.length === 0) return;

    const stateReading = mapEmotionToLearningState(emotionHistory);
    setLearningState(stateReading.state);
    setStateHistory((prev) => [...prev.slice(-100), stateReading]);
  }, [emotionHistory]);

  const selectTopic = useCallback((topicId: string) => {
    setSelectedTopic(topicId);
    setLessonIndex(0);
    const welcome = getWelcomeMessage(topicId);
    setMessages([welcome]);
    startDetection();
  }, [startDetection]);

  const handleSendMessage = useCallback(
    async (text: string) => {
      if (!selectedTopic) return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsThinking(true);

      const lower = text.toLowerCase();
      let nextLessonIndex = lessonIndex;

      if (lower === "next" || lower === "continue" || lower === "next lesson") {
        nextLessonIndex = lessonIndex + 1;
        setLessonIndex(nextLessonIndex);
      }

      const stateReading = mapEmotionToLearningState(emotionHistory);
      const history = messages
        .filter((m) => m.role === "user" || m.role === "tutor")
        .map((m) => ({ role: m.role, content: m.content }));

      try {
        const { content } = await generateAIResponse(
          selectedTopic,
          nextLessonIndex,
          stateReading.state,
          text,
          history
        );

        const response: ChatMessage = {
          id: crypto.randomUUID(),
          role: "tutor",
          content,
          timestamp: Date.now(),
          emotionContext: stateReading.state,
        };
        setMessages((prev) => [...prev, response]);
      } catch {
        const response = generateTutorResponse(
          selectedTopic,
          nextLessonIndex,
          stateReading.state,
          stateReading.suggestion,
          text
        );
        setMessages((prev) => [...prev, response]);
      }
      setIsThinking(false);
    },
    [selectedTopic, lessonIndex, emotionHistory, messages]
  );

  const insight = useMemo(
    () => generateEmotionInsight(stateHistory),
    [stateHistory]
  );

  if (!selectedTopic) {
    return (
      <div className="min-h-screen grid-bg flex flex-col">
        <div className="p-4 border-b border-white/5">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sensai-500 to-purple-500 flex items-center justify-center">
              <span className="text-white font-bold text-sm">S</span>
            </div>
            <span className="text-xl font-bold">
              Sens<span className="text-sensai-400">AI</span>
            </span>
          </Link>
        </div>

        <div className="flex-1 flex items-center justify-center px-4">
          <div className="max-w-2xl w-full space-y-8">
            <div className="text-center space-y-4">
              <h1 className="text-3xl sm:text-4xl font-bold">
                What do you want to{" "}
                <span className="gradient-text">learn</span>?
              </h1>
              <p className="text-gray-400">
                Choose a topic and your emotion-aware AI tutor will guide you
              </p>
            </div>

            <div className="grid sm:grid-cols-3 gap-4">
              {TOPICS.map((topic) => (
                <button
                  key={topic.id}
                  onClick={() => selectTopic(topic.id)}
                  className="glass glass-hover rounded-2xl p-6 text-left space-y-3 transition-all hover:scale-105"
                >
                  <span className="text-4xl">
                    {TOPIC_ICONS[topic.id] || "📚"}
                  </span>
                  <h3 className="text-lg font-semibold">{topic.title}</h3>
                  <p className="text-xs text-gray-500">
                    {topic.lessonCount} lessons
                  </p>
                </button>
              ))}
            </div>

            <div className="text-center">
              <p className="text-xs text-gray-600">
                More topics coming soon — contribute at GitHub
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Top bar */}
      <div className="flex items-center justify-between p-3 border-b border-white/5 flex-shrink-0">
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-sensai-500 to-purple-500 flex items-center justify-center">
              <span className="text-white font-bold text-xs">S</span>
            </div>
          </Link>
          <div className="h-5 w-px bg-white/10" />
          <span className="text-sm font-medium">
            {TOPIC_ICONS[selectedTopic]}{" "}
            {TOPICS.find((t) => t.id === selectedTopic)?.title}
          </span>
          <span className="text-xs text-gray-500">
            Lesson {lessonIndex + 1}
          </span>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => {
              setSelectedTopic(null);
              setMessages([]);
              setStateHistory([]);
            }}
            className="text-xs text-gray-500 hover:text-white transition"
          >
            Change Topic
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex gap-3 p-3 min-h-0">
        {/* Left sidebar — Emotion Detection */}
        <div className="hidden lg:flex flex-col gap-3 w-72 flex-shrink-0">
          <EmotionDetector
            videoRef={videoRef}
            canvasRef={canvasRef}
            currentEmotion={currentEmotion}
            isActive={isActive}
            isLoading={isLoading}
            error={error}
            onStart={startDetection}
            onDemo={useDemoMode}
          />
          <EmotionTimeline
            emotionHistory={emotionHistory}
            learningStateHistory={stateHistory}
            insight={insight}
          />
        </div>

        {/* Center — AI Tutor */}
        <div className="flex-1 min-w-0">
          <AITutor
            messages={messages}
            learningState={learningState}
            onSendMessage={handleSendMessage}
            isThinking={isThinking}
          />
        </div>

        {/* Mobile emotion toggle */}
        <div className="lg:hidden fixed bottom-20 right-4 z-50">
          {!isActive && (
            <button
              onClick={useDemoMode}
              className="w-12 h-12 rounded-full bg-sensai-600 text-white flex items-center justify-center shadow-lg"
              title="Enable emotion detection"
            >
              🧠
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
