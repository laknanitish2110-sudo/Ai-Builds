"use client";

import Link from "next/link";
import Navbar from "@/components/Navbar";

const FEATURES = [
  {
    icon: "🧠",
    title: "Real-Time Emotion Detection",
    description:
      "Advanced facial recognition detects 7 core emotions via your webcam. No data leaves your device.",
  },
  {
    icon: "🔄",
    title: "Adaptive Teaching",
    description:
      "When you're confused, explanations simplify. When you're bored, challenges increase. Frustrated? We slow down and encourage.",
  },
  {
    icon: "📊",
    title: "Learning Analytics",
    description:
      "Track your emotional journey through each lesson. See when you focus best and where you struggle.",
  },
  {
    icon: "🎯",
    title: "Personalized Pacing",
    description:
      "No two learners are the same. SensAI adjusts speed, complexity, and format to match your unique pattern.",
  },
  {
    icon: "🔒",
    title: "Privacy First",
    description:
      "All emotion processing happens in your browser. Zero video data is sent to any server. Ever.",
  },
  {
    icon: "⚡",
    title: "Multiple Subjects",
    description:
      "From Python to AI to Mathematics — learn any subject with an emotionally intelligent tutor.",
  },
];

const STEPS = [
  {
    step: "01",
    title: "Enable Camera",
    description:
      "Grant webcam access so SensAI can read your facial expressions in real-time. Everything stays local on your device.",
  },
  {
    step: "02",
    title: "Choose a Topic",
    description:
      "Pick from Python, AI, Mathematics, or any subject. Each has carefully structured lessons at multiple complexity levels.",
  },
  {
    step: "03",
    title: "Learn Naturally",
    description:
      "Just learn. SensAI continuously reads your emotions and adapts — simpler when confused, harder when bored, supportive when frustrated.",
  },
];

export default function Home() {
  return (
    <main className="min-h-screen">
      <Navbar />

      {/* Hero */}
      <section className="relative min-h-screen flex items-center justify-center px-4 pt-16 grid-bg overflow-hidden">
        <div className="absolute inset-0">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-sensai-500/10 rounded-full blur-[128px]" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-[128px]" />
        </div>

        <div className="relative z-10 max-w-4xl mx-auto text-center space-y-8">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass text-sm text-gray-300">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            The future of learning is emotional
          </div>

          <h1 className="text-5xl sm:text-7xl font-bold leading-tight">
            The AI Tutor That{" "}
            <span className="gradient-text">Feels</span> How You Learn
          </h1>

          <p className="text-lg sm:text-xl text-gray-400 max-w-2xl mx-auto leading-relaxed">
            SensAI detects your emotions in real-time through your webcam and
            adapts its teaching style. Confused? It simplifies. Bored? It
            challenges. Frustrated? It encourages.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/learn"
              className="px-8 py-4 rounded-xl bg-gradient-to-r from-sensai-600 to-purple-600 hover:from-sensai-500 hover:to-purple-500 text-white font-semibold text-lg transition-all emotion-glow"
            >
              Start Learning Free
            </Link>
            <a
              href="#demo"
              className="px-8 py-4 rounded-xl glass glass-hover text-gray-300 font-medium text-lg transition-all"
            >
              Watch Demo
            </a>
          </div>

          <div className="flex items-center justify-center gap-8 pt-4 text-sm text-gray-500">
            <span className="flex items-center gap-2">
              <svg
                className="w-4 h-4 text-green-500"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                  clipRule="evenodd"
                />
              </svg>
              100% Private
            </span>
            <span className="flex items-center gap-2">
              <svg
                className="w-4 h-4 text-green-500"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                  clipRule="evenodd"
                />
              </svg>
              No Sign-Up
            </span>
            <span className="flex items-center gap-2">
              <svg
                className="w-4 h-4 text-green-500"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                  clipRule="evenodd"
                />
              </svg>
              Free to Use
            </span>
          </div>
        </div>

        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
          <svg
            className="w-6 h-6 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 14l-7 7m0 0l-7-7m7 7V3"
            />
          </svg>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="py-24 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              Learning That{" "}
              <span className="gradient-text">Adapts to You</span>
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Traditional AI tutors treat every student the same. SensAI reads
              your emotional state and adjusts everything in real-time.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {FEATURES.map((feature) => (
              <div
                key={feature.title}
                className="glass glass-hover rounded-2xl p-6 space-y-3 transition-all"
              >
                <span className="text-3xl">{feature.icon}</span>
                <h3 className="text-lg font-semibold">{feature.title}</h3>
                <p className="text-sm text-gray-400 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-24 px-4 grid-bg">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              How It <span className="gradient-text">Works</span>
            </h2>
          </div>

          <div className="space-y-12">
            {STEPS.map((step) => (
              <div key={step.step} className="flex gap-6 items-start">
                <div className="flex-shrink-0 w-14 h-14 rounded-2xl bg-gradient-to-br from-sensai-600/20 to-purple-600/20 border border-sensai-500/20 flex items-center justify-center">
                  <span className="text-sensai-400 font-bold text-sm">
                    {step.step}
                  </span>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">{step.title}</h3>
                  <p className="text-gray-400 leading-relaxed">
                    {step.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Emotion Adaptation Demo */}
      <section id="demo" className="py-24 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              See the <span className="gradient-text">Adaptation</span>
            </h2>
            <p className="text-gray-400">
              Here&apos;s how SensAI responds differently based on your emotional
              state
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                emotion: "😕 Confused",
                color: "from-orange-500/20 to-orange-600/20",
                border: "border-orange-500/20",
                response:
                  "Simplifies the explanation, adds visual diagrams, slows down pacing",
              },
              {
                emotion: "😤 Frustrated",
                color: "from-red-500/20 to-red-600/20",
                border: "border-red-500/20",
                response:
                  "Offers encouragement, suggests a break, switches to interactive format",
              },
              {
                emotion: "😴 Bored",
                color: "from-purple-500/20 to-purple-600/20",
                border: "border-purple-500/20",
                response:
                  "Increases difficulty, adds coding challenges, speeds up pacing",
              },
              {
                emotion: "🎯 Focused",
                color: "from-cyan-500/20 to-cyan-600/20",
                border: "border-cyan-500/20",
                response:
                  "Maintains current approach — you're in the zone! No interruptions",
              },
              {
                emotion: "🤩 Excited",
                color: "from-green-500/20 to-green-600/20",
                border: "border-green-500/20",
                response:
                  "Goes deeper into advanced topics, introduces real-world applications",
              },
              {
                emotion: "😰 Struggling",
                color: "from-blue-500/20 to-blue-600/20",
                border: "border-blue-500/20",
                response:
                  "Breaks down concepts further, provides analogies, adds step-by-step guides",
              },
            ].map((item) => (
              <div
                key={item.emotion}
                className={`rounded-2xl p-6 bg-gradient-to-br ${item.color} border ${item.border} space-y-3`}
              >
                <h3 className="text-lg font-semibold">{item.emotion}</h3>
                <p className="text-sm text-gray-300 leading-relaxed">
                  {item.response}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24 px-4">
        <div className="max-w-3xl mx-auto text-center space-y-8">
          <h2 className="text-3xl sm:text-5xl font-bold">
            Ready to Learn{" "}
            <span className="gradient-text">Differently</span>?
          </h2>
          <p className="text-gray-400 text-lg">
            Join the future of education. Where AI doesn&apos;t just teach — it
            understands.
          </p>
          <Link
            href="/learn"
            className="inline-block px-10 py-4 rounded-xl bg-gradient-to-r from-sensai-600 to-purple-600 hover:from-sensai-500 hover:to-purple-500 text-white font-semibold text-lg transition-all emotion-glow"
          >
            Start Learning Now
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 py-8 px-4">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-gradient-to-br from-sensai-500 to-purple-500 flex items-center justify-center">
              <span className="text-white font-bold text-xs">S</span>
            </div>
            <span className="text-sm font-semibold">
              Sens<span className="text-sensai-400">AI</span>
            </span>
          </div>
          <p className="text-xs text-gray-600">
            Built with emotion. Powered by intelligence.
          </p>
        </div>
      </footer>
    </main>
  );
}
