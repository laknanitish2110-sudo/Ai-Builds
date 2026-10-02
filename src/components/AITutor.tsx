"use client";

import { useEffect, useRef, useState } from "react";
import type { ChatMessage, LearningState } from "@/types";

interface AITutorProps {
  messages: ChatMessage[];
  learningState: LearningState;
  onSendMessage: (message: string) => void;
  isThinking: boolean;
}

const STATE_INDICATORS: Record<
  LearningState,
  { label: string; color: string; icon: string }
> = {
  confused: { label: "Simplifying", color: "text-orange-400", icon: "🔄" },
  frustrated: { label: "Encouraging", color: "text-red-400", icon: "💪" },
  bored: { label: "Challenging", color: "text-purple-400", icon: "🚀" },
  focused: { label: "In the Zone", color: "text-cyan-400", icon: "🎯" },
  excited: { label: "Energized", color: "text-green-400", icon: "⚡" },
  struggling: { label: "Supporting", color: "text-blue-400", icon: "🤝" },
  neutral: { label: "Ready", color: "text-gray-400", icon: "📚" },
};

export default function AITutor({
  messages,
  learningState,
  onSendMessage,
  isThinking,
}: AITutorProps) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    onSendMessage(input.trim());
    setInput("");
  };

  const indicator = STATE_INDICATORS[learningState];

  return (
    <div className="glass rounded-2xl flex flex-col h-full">
      <div className="flex items-center justify-between p-4 border-b border-white/5">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-sensai-500 to-purple-500 flex items-center justify-center">
            <span className="text-sm">🧠</span>
          </div>
          <div>
            <h3 className="text-sm font-semibold">SensAI Tutor</h3>
            <p className={`text-xs ${indicator.color}`}>
              {indicator.icon} {indicator.label}
            </p>
          </div>
        </div>

        {learningState !== "neutral" && learningState !== "focused" && (
          <div
            className={`px-2 py-1 rounded-full text-xs ${indicator.color} bg-white/5`}
          >
            Adapting to your state
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin min-h-0">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${
              msg.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                msg.role === "user"
                  ? "bg-sensai-600 text-white"
                  : msg.role === "system"
                  ? "bg-yellow-500/10 border border-yellow-500/20 text-yellow-200"
                  : "bg-gray-800/50 text-gray-200"
              }`}
            >
              {msg.emotionContext &&
                msg.emotionContext !== "neutral" &&
                msg.emotionContext !== "focused" && (
                  <div className="text-xs text-gray-500 mb-1 flex items-center gap-1">
                    <span>
                      {STATE_INDICATORS[msg.emotionContext].icon}
                    </span>
                    Adapted for:{" "}
                    {msg.emotionContext}
                  </div>
                )}
              <div className="text-sm whitespace-pre-wrap leading-relaxed">
                {formatMessage(msg.content)}
              </div>
            </div>
          </div>
        ))}

        {isThinking && (
          <div className="flex justify-start">
            <div className="bg-gray-800/50 rounded-2xl px-4 py-3">
              <div className="flex gap-1">
                <div className="w-2 h-2 rounded-full bg-gray-500 animate-bounce" />
                <div
                  className="w-2 h-2 rounded-full bg-gray-500 animate-bounce"
                  style={{ animationDelay: "0.1s" }}
                />
                <div
                  className="w-2 h-2 rounded-full bg-gray-500 animate-bounce"
                  style={{ animationDelay: "0.2s" }}
                />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="p-4 border-t border-white/5">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question or type 'next' to continue..."
            className="flex-1 bg-gray-800/50 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-sensai-500/50 transition"
          />
          <button
            type="submit"
            disabled={!input.trim()}
            className="px-4 py-2.5 rounded-xl bg-sensai-600 hover:bg-sensai-500 disabled:opacity-50 disabled:hover:bg-sensai-600 text-white text-sm font-medium transition"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}

function formatMessage(content: string): React.ReactNode {
  const parts = content.split(/(```[\s\S]*?```|\*\*.*?\*\*|`[^`]+`)/g);

  return parts.map((part, i) => {
    if (part.startsWith("```") && part.endsWith("```")) {
      const code = part.slice(3, -3).replace(/^[a-z]*\n/, "");
      return (
        <pre
          key={i}
          className="bg-gray-900 rounded-lg p-3 my-2 overflow-x-auto text-xs font-mono"
        >
          <code>{code}</code>
        </pre>
      );
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} className="font-semibold text-white">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code
          key={i}
          className="bg-gray-800 px-1.5 py-0.5 rounded text-sensai-300 text-xs font-mono"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    return <span key={i}>{part}</span>;
  });
}
