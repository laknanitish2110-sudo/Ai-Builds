import type {
  ChatMessage,
  LearningState,
  TutorAdaptation,
} from "@/types";

interface TopicContent {
  title: string;
  lessons: LessonContent[];
}

interface LessonContent {
  title: string;
  simple: string;
  normal: string;
  deep: string;
  visual: string;
  exercise: string;
}

const TOPICS: Record<string, TopicContent> = {
  python: {
    title: "Python Programming",
    lessons: [
      {
        title: "Variables & Data Types",
        simple:
          "Think of a variable like a labeled box. You put something in the box and give it a name. For example: `name = \"Alice\"` — you just put \"Alice\" into a box called `name`. You can put numbers too: `age = 25`.",
        normal:
          "Variables in Python are dynamically typed — you don't need to declare their type. Python supports several built-in types: `int`, `float`, `str`, `bool`, `list`, `dict`, `tuple`, and `set`. Assignment uses `=`, and you can check a variable's type with `type()`.",
        deep:
          "Under the hood, Python variables are references to objects in memory. When you write `x = 42`, Python creates an `int` object with value 42 and points `x` to it. This is why `id(x)` returns the memory address. Python uses reference counting + garbage collection. Small integers (-5 to 256) are cached (interned) for performance.",
        visual:
          "```\n[Box: name] --> \"Alice\"\n[Box: age]  --> 25\n[Box: scores] --> [90, 85, 92]\n```\nEach variable is a label pointing to a value in memory.",
        exercise:
          "Create three variables: your name (string), your age (integer), and whether you like coding (boolean). Then print each one with its type.",
      },
      {
        title: "Functions",
        simple:
          "A function is like a recipe. You give it a name, tell it what ingredients (inputs) it needs, and what dish (output) it makes. Example:\n```python\ndef greet(name):\n    return f\"Hello, {name}!\"\n```\nNow `greet(\"Sam\")` gives you `\"Hello, Sam!\"`",
        normal:
          "Functions in Python are defined with `def`, support default arguments, *args, **kwargs, and can return multiple values via tuples. They're first-class objects — you can pass them as arguments, store them in variables, and return them from other functions.",
        deep:
          "Functions in Python are objects of type `function`. They have attributes like `__code__`, `__defaults__`, and `__closure__`. Closures capture variables from enclosing scopes via the LEGB rule (Local, Enclosing, Global, Built-in). Decorators are syntactic sugar for higher-order functions that wrap other functions.",
        visual:
          "```\nINPUT          FUNCTION         OUTPUT\n\"Sam\"    -->  [ greet() ]  -->  \"Hello, Sam!\"\n  5, 3   -->  [  add()  ]  -->  8\n```",
        exercise:
          "Write a function called `calculate_bmi` that takes weight (kg) and height (m) and returns the BMI. Then call it with your own values.",
      },
    ],
  },
  ai: {
    title: "Artificial Intelligence",
    lessons: [
      {
        title: "What is AI?",
        simple:
          "AI is teaching computers to think and learn, kind of like how you learned to recognize cats vs dogs as a kid. You saw hundreds of examples, and now you just *know*. AI does the same thing — it looks at tons of examples and learns patterns.",
        normal:
          "Artificial Intelligence encompasses systems that can perform tasks requiring human-level intelligence. The main branches are: Machine Learning (learning from data), Deep Learning (neural networks), NLP (understanding language), Computer Vision (understanding images), and Robotics. Modern AI is primarily statistical — it finds patterns in large datasets.",
        deep:
          "Modern AI is built on differentiable programming. Neural networks are compositions of parameterized functions optimized via gradient descent on a loss function. The key breakthrough was backpropagation — computing gradients efficiently through the chain rule. Transformers (2017) revolutionized NLP by replacing recurrence with self-attention, enabling parallelization and capturing long-range dependencies via scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V.",
        visual:
          "```\n  DATA          MODEL         PREDICTION\n[Images] --> [Neural Net] --> \"It's a cat!\"\n[Text]   --> [Transformer]--> \"Translation\"\n[Audio]  --> [Wav2Vec]    --> \"Transcription\"\n```",
        exercise:
          "List 3 things you use daily that are powered by AI. For each one, guess what type of AI it uses (computer vision, NLP, recommendation system, etc.).",
      },
    ],
  },
  math: {
    title: "Mathematics",
    lessons: [
      {
        title: "Linear Algebra Basics",
        simple:
          "A matrix is just a grid of numbers — like a spreadsheet. A vector is a single row or column. When you multiply matrices, you're combining these grids in a specific way. This is the foundation of how AI \"thinks\" — every piece of data becomes numbers in a grid.",
        normal:
          "Linear algebra studies vectors, matrices, and linear transformations. Key concepts: vector spaces, matrix multiplication (dot product of rows and columns), eigenvalues/eigenvectors, and matrix decompositions (SVD, LU, QR). In ML, data is represented as matrices, and operations like matrix multiplication are the core computational primitive.",
        deep:
          "A vector space V over field F satisfies closure under addition and scalar multiplication. The rank of a matrix equals the dimension of its column space. The spectral theorem states symmetric matrices are orthogonally diagonalizable. SVD decomposes any m×n matrix A = UΣV^T where U, V are orthogonal and Σ is diagonal. This is fundamental to PCA, LSA, and low-rank approximations in ML.",
        visual:
          "```\nVector:  [3, 5, 2]  ← a point in 3D space\n\nMatrix:  [1  2]    ← a 2×2 grid\n         [3  4]\n\nMultiply: [1 2] × [5] = [1×5 + 2×6] = [17]\n          [3 4]   [6]   [3×5 + 4×6]   [39]\n```",
        exercise:
          "Given matrix A = [[2,1],[1,3]], multiply it by vector v = [4,2]. Show your work step by step.",
      },
    ],
  },
};

export function getTopicList() {
  return Object.entries(TOPICS).map(([id, topic]) => ({
    id,
    title: topic.title,
    lessonCount: topic.lessons.length,
  }));
}

export function getTopicTitle(topicId: string): string {
  return TOPICS[topicId]?.title ?? "General";
}

export function getLessonTitle(topicId: string, lessonIndex: number): string {
  const topic = TOPICS[topicId] || TOPICS.python;
  const lesson = topic.lessons[lessonIndex] || topic.lessons[0];
  return lesson.title;
}

export async function generateAIResponse(
  topicId: string,
  lessonIndex: number,
  state: LearningState,
  userMessage: string,
  conversationHistory: { role: string; content: string }[]
): Promise<{ content: string; fromAI: boolean }> {
  const topic = TOPICS[topicId] || TOPICS.python;
  const lesson = topic.lessons[lessonIndex] || topic.lessons[0];

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessage,
        topic: topic.title,
        lessonTitle: lesson.title,
        learningState: state,
        agentName: AGENT_NAMES[topicId] || "SensAI",
        conversationHistory,
      }),
    });

    const data = await res.json();

    if (data.fallback || data.error) {
      return {
        content: generateLocalResponse(topicId, lessonIndex, state, userMessage),
        fromAI: false,
      };
    }

    return { content: data.content, fromAI: true };
  } catch {
    return {
      content: generateLocalResponse(topicId, lessonIndex, state, userMessage),
      fromAI: false,
    };
  }
}

function generateLocalResponse(
  topicId: string,
  lessonIndex: number,
  state: LearningState,
  userMessage: string
): string {
  const topic = TOPICS[topicId] || TOPICS.python;
  const lesson = topic.lessons[lessonIndex] || topic.lessons[0];
  const lower = userMessage.toLowerCase().trim();

  if (lower === "next" || lower === "continue" || lower === "next lesson") {
    const nextLesson = topic.lessons[lessonIndex + 1];
    if (nextLesson) {
      return `Great, let's move on to **${nextLesson.title}**!\n\n${nextLesson.normal}`;
    }
    return `You've completed all available lessons in ${topic.title}! 🎉\n\nWant to review any topic? Just ask a question about anything we covered.`;
  }

  const mathResult = tryMathEval(userMessage);
  if (mathResult !== null) {
    return mathResult;
  }

  if (isQuestion(lower)) {
    return answerFromContext(lower, lesson, topic.title, state);
  }

  return getLessonContent(lesson, state);
}

function isQuestion(text: string): boolean {
  return (
    text.includes("?") ||
    /^(what|how|why|when|where|who|which|can|do|does|is|are|was|were|explain|tell|show|help|define)\b/.test(
      text
    )
  );
}

function tryMathEval(input: string): string | null {
  let expr = input
    .toLowerCase()
    .replace(/what'?s?\s*/gi, "")
    .replace(/whats\s*/gi, "")
    .replace(/calculate\s*/gi, "")
    .replace(/solve\s*/gi, "")
    .replace(/\?/g, "")
    .replace(/divided\s*by/gi, "/")
    .replace(/multiplied\s*by/gi, "*")
    .replace(/times/gi, "*")
    .replace(/plus/gi, "+")
    .replace(/minus/gi, "-")
    .replace(/mod(ulo)?/gi, "%")
    .replace(/\^/g, "**")
    .replace(/x/gi, "*")
    .replace(/={1,2}/g, "")
    .trim();

  if (!/^[\d\s+\-*/().%*]+$/.test(expr)) return null;
  if (!/\d/.test(expr)) return null;

  try {
    const fn = new Function(`"use strict"; return (${expr});`);
    const result = fn();
    if (typeof result !== "number" || !isFinite(result)) return null;

    const display = Number.isInteger(result)
      ? result.toString()
      : result.toFixed(6).replace(/\.?0+$/, "");

    return `**${expr.trim()} = ${display}**\n\nHere's the breakdown:\n- Expression: \`${expr.trim()}\`\n- Result: **${display}**`;
  } catch {
    return null;
  }
}

function answerFromContext(
  question: string,
  lesson: LessonContent,
  topicTitle: string,
  state: LearningState
): string {
  const statePrefix = getStatePrefix(state);

  const keywords = question.split(/\s+/).filter((w) => w.length > 3);
  const allContent = `${lesson.simple} ${lesson.normal} ${lesson.deep}`;
  const relevantSentences = allContent
    .split(/\.\s+/)
    .filter((s) => keywords.some((k) => s.toLowerCase().includes(k)))
    .slice(0, 3);

  if (relevantSentences.length > 0) {
    return `${statePrefix}${relevantSentences.join(". ")}.\n\n${lesson.visual ? `**Visual:**\n${lesson.visual}` : ""}`;
  }

  return `${statePrefix}That's a great question! It's related to our **${topicTitle}** topic.\n\nHere's what I can share about **${lesson.title}**:\n\n${state === "confused" || state === "struggling" ? lesson.simple : lesson.normal}\n\n💡 *For a smarter AI tutor that answers any question, ask your teacher to add the ANTHROPIC_API_KEY in the Vercel settings.*`;
}

function getStatePrefix(state: LearningState): string {
  switch (state) {
    case "confused":
      return "Let me break this down simply. ";
    case "frustrated":
      return "No worries, let's take it step by step. ";
    case "excited":
      return "Love the energy! ";
    case "bored":
      return "Let's make this more interesting. ";
    case "struggling":
      return "You're doing great, let's work through this together. ";
    default:
      return "";
  }
}

function getLessonContent(
  lesson: LessonContent,
  state: LearningState
): string {
  const prefix = getStatePrefix(state);
  switch (state) {
    case "confused":
    case "struggling":
      return `${prefix}${lesson.simple}\n\n**Visual:**\n${lesson.visual}`;
    case "excited":
    case "bored":
      return `${prefix}${lesson.deep}\n\n**Try this:** ${lesson.exercise}`;
    default:
      return `${prefix}${lesson.normal}`;
  }
}

export function generateTutorResponse(
  topicId: string,
  lessonIndex: number,
  state: LearningState,
  _adaptation: TutorAdaptation,
  userMessage: string
): ChatMessage {
  const content = generateLocalResponse(topicId, lessonIndex, state, userMessage);
  return {
    id: crypto.randomUUID(),
    role: "tutor",
    content,
    timestamp: Date.now(),
    emotionContext: state,
  };
}

const AGENT_NAMES: Record<string, string> = {
  python: "Py",
  ai: "Nova",
  math: "Euler",
};

export function getWelcomeMessage(topicId: string): ChatMessage {
  const topic = TOPICS[topicId] || TOPICS.python;
  const agentName = AGENT_NAMES[topicId] || "SensAI";
  return {
    id: crypto.randomUUID(),
    role: "tutor",
    content: `Hey! I'm **${agentName}**, your ${topic.title} tutor.\n\nI'll be watching your facial expressions through the camera to understand how you're feeling. If you look confused, I'll simplify. If you're bored, I'll challenge you more. If you're frustrated, I'll slow down and encourage you.\n\nAsk me anything — I'm here to help. Let's start with **${topic.lessons[0].title}**.`,
    timestamp: Date.now(),
  };
}
