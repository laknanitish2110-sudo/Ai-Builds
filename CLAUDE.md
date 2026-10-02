# SensAI Project Context

## What SensAI Is
Emotion-aware AI education platform. Uses a custom-trained CNN (v2, ResNet+CBAM, 2.8M params) running in-browser via ONNX/WASM to detect facial emotions from webcam, then adapts tutoring in real-time. Built with Next.js 15 + React 19 + TypeScript + TailwindCSS.

## Live
- **Deployed at**: ai-builds-psi.vercel.app
- **Repo**: github.com/laknanitish2110-sudo/Ai-Builds
- **Branch**: claude/intelligent-bohr-3jesu9

## Tutor Agents
- **Py** (green) — Python tutor
- **Nova** (violet) — AI/ML tutor
- **Euler** (blue) — Math tutor

Each has its own personality when Claude API is connected (ANTHROPIC_API_KEY env var on Vercel).

## Memory Layer (src/lib/memory/)
Client-side session continuity system using localStorage. Tracks learner profiles, session history, emotion patterns, plans. Designed as a standalone module that could be extracted into its own package.

---

## STARTUP IDEA — "Memory Layer for AI" (PRIORITY: Future Session)

### The Problem
Every AI app today has amnesia. Users re-explain themselves every session. No continuity, no recall, no learning from past interactions. This affects ChatGPT, Claude, Gemini, and every AI tool.

### The Vision
A universal memory layer that any AI application can plug into:
- **API-based**: `memory.remember(context)`, `memory.recall(userId)`
- **Cross-app**: Works across different AI tools, not just one
- **User-owned**: Users control their data, can export, delete, port it
- **Privacy-first**: Data belongs to the user, not the platform

### What Needs to Be Built
1. **Backend service** — User accounts, cloud sync, data storage
2. **SDK/API** — Simple API any AI app can integrate (`npm install @sensai/memory`)
3. **Privacy controls** — Users own their data, granular permissions
4. **Context engine** — Smart recall that surfaces relevant past context, not just dumps everything
5. **Multi-device sync** — Same memory across phone, laptop, tablet

### Starting Point
The `src/lib/memory/` module in this repo is the prototype. It has:
- LearnerProfile, SessionRecord, SessionPlan types
- Store (read/write to localStorage)
- Recall (build context from past sessions, time awareness)
- Clean TypeScript interfaces, no external deps

### Next Steps for the Startup Session
1. Extract memory module into its own package
2. Add Supabase backend for cloud sync + auth
3. Design the public API surface
4. Build a demo showing the same memory working across two different apps
5. Think about pricing, positioning, go-to-market

### The User's Words
"its a real problem that is today... i want us to work on it... a small layer made by us today can be used by millions of people if it really solves the problem"

---

## Technical Notes
- Model: sensai_emotion.onnx (11MB, v2 architecture) in public/models/
- ONNX export: AdaptiveMaxPool2d replaced with x.view(b,c,-1).max(dim=2)[0] for compatibility
- Webcam: Camera request runs BEFORE model loading to avoid false demo-mode fallback
- AI Tutor: /api/chat route uses Claude API when ANTHROPIC_API_KEY is set, falls back to local math eval + context matching
