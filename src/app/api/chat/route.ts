import { NextRequest, NextResponse } from "next/server";
import Anthropic from "@anthropic-ai/sdk";

interface ChatRequestBody {
  message: string;
  topic: string;
  lessonTitle: string;
  learningState: string;
  agentName: string;
  conversationHistory: { role: string; content: string }[];
}

const AGENT_PERSONALITIES: Record<string, string> = {
  Py: "You're Py, a friendly Python programming tutor who loves clean code and practical examples. You use snake_case humor and Python analogies.",
  Nova: "You're Nova, an AI & machine learning tutor who makes complex concepts feel intuitive. You connect everything to real-world AI applications.",
  Euler: "You're Euler, a mathematics tutor named after the legendary mathematician. You make math visual and always show your work step by step.",
};

const BASE_RULES = `You teach through conversation — answering questions, explaining concepts, and giving examples.

Rules:
- Answer the student's actual question directly and clearly.
- Keep responses concise (2-4 paragraphs max). Use markdown for formatting.
- If they ask a math question, show the calculation step by step.
- If they ask a coding question, include a short code example.
- Adapt your tone to their emotional state (provided as context).
- If they say "next" or "continue", advance to the next concept in the topic.
- Be encouraging but not patronizing.
- Use analogies and real-world examples to explain complex ideas.`;

export async function POST(req: NextRequest) {
  const body: ChatRequestBody = await req.json();
  const { message, topic, lessonTitle, learningState, agentName, conversationHistory } =
    body;

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "no_api_key", fallback: true },
      { status: 200 }
    );
  }

  const client = new Anthropic({ apiKey });

  const stateContext =
    learningState !== "neutral"
      ? `\n\nThe student's current emotional state is: ${learningState}. Adapt your response accordingly.`
      : "";

  const messages: Anthropic.MessageParam[] = conversationHistory
    .slice(-10)
    .map((m) => ({
      role: (m.role === "user" ? "user" : "assistant") as "user" | "assistant",
      content: m.content,
    }));

  messages.push({ role: "user", content: message });

  try {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      system: `${AGENT_PERSONALITIES[agentName] || "You're SensAI, an emotion-aware AI tutor."}\n\n${BASE_RULES}\n\nCurrent topic: ${topic}\nCurrent lesson: ${lessonTitle}${stateContext}`,
      messages,
    });

    const text =
      response.content[0].type === "text" ? response.content[0].text : "";

    return NextResponse.json({ content: text });
  } catch (err) {
    const errorMessage =
      err instanceof Error ? err.message : "AI service error";
    return NextResponse.json(
      { error: errorMessage, fallback: true },
      { status: 200 }
    );
  }
}
