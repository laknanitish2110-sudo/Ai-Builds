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

export function generateTutorResponse(
  topicId: string,
  lessonIndex: number,
  state: LearningState,
  adaptation: TutorAdaptation,
  userMessage: string
): ChatMessage {
  const topic = TOPICS[topicId] || TOPICS.python;
  const lesson =
    topic.lessons[lessonIndex] || topic.lessons[0];

  let content = "";

  if (adaptation.message) {
    content += adaptation.message + "\n\n";
  }

  switch (adaptation.complexity) {
    case "simpler":
      content += lesson.simple;
      break;
    case "deeper":
      content += lesson.deep;
      break;
    default:
      content += lesson.normal;
  }

  if (adaptation.format === "visual") {
    content += "\n\n**Visual breakdown:**\n" + lesson.visual;
  }

  if (
    adaptation.format === "interactive" ||
    state === "bored"
  ) {
    content += "\n\n**Try this:** " + lesson.exercise;
  }

  if (adaptation.shouldBreak) {
    content +=
      "\n\n💡 *Take a 30-second breather if you need it. There's no rush.*";
  }

  return {
    id: crypto.randomUUID(),
    role: "tutor",
    content,
    timestamp: Date.now(),
    emotionContext: state,
    adaptation,
  };
}

export function getWelcomeMessage(topicId: string): ChatMessage {
  const topic = TOPICS[topicId] || TOPICS.python;
  return {
    id: crypto.randomUUID(),
    role: "tutor",
    content: `Welcome to **${topic.title}**! I'm your SensAI tutor.\n\nI'll be watching your facial expressions through the camera to understand how you're feeling. If you look confused, I'll simplify. If you're bored, I'll challenge you more. If you're frustrated, I'll slow down and encourage you.\n\nReady? Let's start with **${topic.lessons[0].title}**.`,
    timestamp: Date.now(),
  };
}
