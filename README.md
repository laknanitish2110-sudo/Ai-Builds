# SensAI

**Emotion-aware AI tutor that adapts to how you feel.**

SensAI detects your facial emotions in real-time through your webcam and adapts its teaching style accordingly. Confused? It simplifies. Bored? It challenges. Frustrated? It encourages. All processing happens in your browser — zero data leaves your device.

## Features

- **Real-Time Emotion Detection** — Custom CNN detects 7 emotions via webcam at ~30 FPS
- **Adaptive Teaching** — Pacing, complexity, and tone adjust based on your emotional state
- **Learning Analytics** — Emotion timeline tracks your journey through each lesson
- **Privacy First** — All inference runs client-side via WebAssembly. No video data is transmitted
- **Multiple Subjects** — Python, AI/ML, Mathematics, and more
- **No Sign-Up** — Open the app and start learning immediately

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15, React 19, TypeScript, TailwindCSS |
| Emotion Model | Custom CNN (PyTorch) exported to ONNX |
| Browser Inference | onnxruntime-web (WebAssembly) |
| Charts | Recharts |
| Animations | Framer Motion |
| Icons | Lucide React |

## Emotion Detection Model

### Architecture: SensAIEmotionCNN

A lightweight CNN with dual attention mechanisms designed for real-time browser inference.

```
Input (1x48x48 grayscale)
  |
  +--> ConvBlock 1:  1 -> 32 channels   (Conv3x3 + BN + ReLU) x2 + MaxPool
  +--> ConvBlock 2: 32 -> 64 channels   (Conv3x3 + BN + ReLU) x2 + MaxPool
  +--> ConvBlock 3: 64 -> 128 channels  (Conv3x3 + BN + ReLU) x2 + MaxPool + Channel Attention + Spatial Attention
  +--> ConvBlock 4: 128 -> 256 channels (Conv3x3 + BN + ReLU) x2 + MaxPool + Channel Attention + Spatial Attention
  |
  +--> Global Average Pooling
  +--> Dropout(0.4) -> FC(256, 128) -> ReLU -> BN -> Dropout(0.2) -> FC(128, 7)
  |
Output (7 emotion probabilities)
```

### Model Specifications

| Spec | Value |
|------|-------|
| Total Parameters | 1,227,371 (~1.2M) |
| Feature Extractor | 1,193,316 params (4 ConvBlocks) |
| Classifier Head | 34,055 params (2-layer MLP) |
| ONNX Model Size | 95 KB |
| Input | 48x48 grayscale face crop |
| Output | 7 emotion class probabilities |
| Inference Speed | < 1ms on modern CPUs |
| Runtime | onnxruntime-web (WASM backend) |

### Attention Mechanisms

- **Channel Attention (SE-style):** Squeeze-and-excitation blocks on ConvBlocks 3-4. Learns which feature channels are most important via `AdaptiveAvgPool -> FC(C, C/8) -> ReLU -> FC(C/8, C) -> Sigmoid`
- **Spatial Attention:** Learns where to focus in the feature map via channel-wise avg/max pooling concatenated and passed through a 7x7 conv

### Training Details

| Detail | Value |
|--------|-------|
| Dataset | FER2013 (28,709 train / 3,589 test) |
| Emotions | angry, disgusted, fearful, happy, sad, surprised, neutral |
| Best Validation Accuracy | 64.4% (epoch 29) |
| Optimizer | Adam (lr=0.001) |
| Scheduler | ReduceLROnPlateau |
| Augmentation | RandomHorizontalFlip, RandomRotation(10), RandomAffine |
| Training Platform | Google Colab (NVIDIA T4 GPU) |
| Training Time | ~22 seconds/epoch |

> **Note:** 64.4% on FER2013 is competitive for a ~1.2M parameter model. State-of-the-art models with 10-100x more parameters achieve ~73-76%. FER2013 itself has ~65% human agreement due to ambiguous labels, making this close to human-level for this architecture size.

### Inference Pipeline (Browser)

```
Webcam Frame
  -> Extract center face region (60% crop at 40% height)
  -> Convert RGBA to grayscale
  -> Resize to 48x48
  -> Normalize (mean=0.5, std=0.5)
  -> ONNX inference via WebAssembly
  -> Softmax -> 7 emotion probabilities
  -> Map to learning state (confused/frustrated/bored/focused/excited/struggling)
  -> Adapt tutor response
```

## Project Structure

```
Ai-Builds/
├── src/
│   ├── app/
│   │   ├── page.tsx              # Landing page
│   │   ├── learn/page.tsx        # Learning interface
│   │   └── layout.tsx            # Root layout
│   ├── components/
│   │   ├── Navbar.tsx            # Navigation bar
│   │   ├── EmotionDetector.tsx   # Webcam + emotion display
│   │   ├── AITutor.tsx           # Adaptive tutor chat
│   │   └── EmotionTimeline.tsx   # Emotion history chart
│   ├── hooks/
│   │   └── useEmotionDetection.ts # Webcam capture + ONNX inference hook
│   ├── lib/
│   │   └── sensaiModel.ts       # ONNX model loader + predict
│   └── types/
│       └── index.ts              # TypeScript type definitions
├── public/
│   └── models/
│       └── sensai_emotion.onnx   # Trained ONNX model (95KB)
├── sensai-model/
│   ├── models/
│   │   ├── emotion_cnn.py        # SensAIEmotionCNN architecture
│   │   └── mobile_emotion.py     # MobileEmotionNet (alternative)
│   ├── training/
│   │   ├── train.py              # Training loop
│   │   ├── dataset.py            # Dataset loaders
│   │   └── config.py             # Hyperparameters
│   ├── export/
│   │   └── export_onnx.py        # PyTorch -> ONNX export
│   ├── inference/
│   │   └── predictor.py          # Python inference
│   └── checkpoints/
│       └── training_history.json # FER2013 training metrics
├── train_real.py                 # Standalone training script
├── package.json
├── next.config.ts                # WASM + ONNX webpack config
├── tailwind.config.ts
└── tsconfig.json
```

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
git clone https://github.com/laknanitish2110-sudo/Ai-Builds.git
cd Ai-Builds
npm install
```

The `postinstall` script automatically copies the WASM runtime to `public/models/`.

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and navigate to `/learn` to start the emotion-aware tutor.

### Build

```bash
npm run build
npm start
```

## Retraining the Model

To retrain on FER2013:

```bash
# Option 1: Using FER2013 CSV
python train_real.py --fer2013-csv path/to/fer2013.csv

# Option 2: Using HuggingFace (needs internet)
python train_real.py --fer2013

# Option 3: Synthetic data (offline, for testing)
python train_real.py
```

For GPU training, use Google Colab with a T4 runtime. See `train_real.py` for the full training pipeline including ONNX export.

## How Emotion Maps to Teaching

| Detected Emotion | Learning State | Tutor Adaptation |
|-----------------|---------------|-----------------|
| Confused/Fearful | Confused | Simplifies explanation, adds visuals, slows pacing |
| Angry/Disgusted | Frustrated | Encourages, suggests break, switches to interactive |
| Neutral (prolonged) | Bored | Increases difficulty, adds challenges |
| Happy/Neutral | Focused | Maintains approach, no interruption |
| Surprised/Happy | Excited | Goes deeper, introduces advanced topics |
| Sad/Fearful | Struggling | Breaks down concepts, adds step-by-step guides |

## License

MIT
