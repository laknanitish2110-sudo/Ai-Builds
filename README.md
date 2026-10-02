# SensAI

**Emotion-aware AI tutor that adapts to how you feel.**

SensAI detects your facial emotions in real-time through your webcam and adapts its teaching style accordingly. Confused? It simplifies. Bored? It challenges. Frustrated? It encourages. All processing happens in your browser — zero data leaves your device.

## Features

- **Real-Time Emotion Detection** — Custom ResNet-style CNN detects 7 emotions via webcam
- **Adaptive Teaching** — Pacing, complexity, and tone adjust based on your emotional state
- **Learning Analytics** — Emotion timeline tracks your journey through each lesson
- **Privacy First** — All inference runs client-side via WebAssembly. No video data is transmitted
- **Multiple Subjects** — Python, AI/ML, Mathematics, and more
- **No Sign-Up** — Open the app and start learning immediately

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15, React 19, TypeScript, TailwindCSS |
| Emotion Model | Custom ResNet + CBAM CNN (PyTorch) exported to ONNX |
| Browser Inference | onnxruntime-web (WebAssembly) |
| Charts | Recharts |
| Animations | Framer Motion |
| Icons | Lucide React |

## Emotion Detection Model

### Architecture: SensAIEmotionCNN v2

A residual CNN with CBAM (Convolutional Block Attention Module) attention designed for real-time browser inference.

```
Input (1x48x48 grayscale)
  |
  +--> Stem: Conv3x3 (1 -> 64) + BN + ReLU
  |
  +--> Stage 1: 2x ResidualBlock 64   (48x48 -> 24x24)  [skip connections]
  +--> Stage 2: 2x ResidualBlock 128  (24x24 -> 12x12)  [skip + CBAM attention]
  +--> Stage 3: 2x ResidualBlock 256  (12x12 -> 6x6)    [skip + CBAM attention]
  |
  +--> Global Average Pooling (6x6 -> 1x1)
  +--> Dropout(0.5) -> FC(256, 128) -> ReLU -> BN -> Dropout(0.25) -> FC(128, 7)
  |
Output (7 emotion probabilities)
```

### v1 vs v2 Comparison

| Spec | v1 | v2 |
|------|----|----|
| Architecture | Plain ConvBlocks | Residual Blocks + skip connections |
| Parameters | 1,227,371 (1.2M) | 2,833,103 (2.8M) |
| Attention | SE-only on blocks 3-4 | Full CBAM (channel + spatial) on stages 2-3 |
| Skip Connections | None | Every block (identity or 1x1 projection) |
| Dropout | 0.4 | 0.5 |
| Training | Basic Adam, weak augmentation | AdamW + Mixup + Label Smoothing + TTA |
| Target Accuracy | ~64% | ~70%+ |

### Model Specifications

| Spec | Value |
|------|-------|
| Total Parameters | 2,833,103 (~2.8M) |
| Stem | 704 params |
| Stage 1 (64ch) | 152,192 params |
| Stage 2 (128ch + CBAM) | 529,860 params |
| Stage 3 (256ch + CBAM) | 2,116,292 params |
| Classifier Head | 34,055 params |
| ONNX Model Size | ~11 MB |
| Input | 48x48 grayscale face crop |
| Output | 7 emotion class probabilities |
| Inference Speed | < 5ms on modern CPUs |
| Runtime | onnxruntime-web (WASM backend) |

### Key Improvements in v2

**Architecture:**
- **Residual connections** — Skip connections in every block prevent gradient vanishing and allow deeper feature learning
- **CBAM attention** — Channel attention (avg+max pool -> shared FC -> sigmoid) learns *which* features matter. Spatial attention (7x7 conv on pooled features) learns *where* to look in the face
- **Stride-based downsampling** — Replaces MaxPool with strided convolutions for learnable downsampling
- **Deeper classifier** — FC(256->128->7) with BatchNorm between layers

**Training Techniques:**
- **Mixup (alpha=0.2)** — Blends training images and labels to regularize and smooth decision boundaries
- **Label Smoothing (0.1)** — Prevents overconfident predictions on noisy FER2013 labels
- **CosineAnnealingWarmRestarts** — Cyclical LR that escapes local minima and explores the loss landscape
- **AdamW (weight_decay=5e-4)** — Decoupled weight decay for better generalization
- **Stronger augmentation** — RandomRotation(15), RandomAffine(shear=5), ColorJitter(0.3), RandomErasing(p=0.25)
- **Test-Time Augmentation (5x)** — Averages predictions over 5 augmented views at evaluation for +1-2% accuracy
- **Gradient clipping (max_norm=2.0)** — Stabilizes training with mixup

### Training Details

| Detail | Value |
|--------|-------|
| Dataset | FER2013 (28,709 train / 3,589 test) |
| Emotions | angry, disgusted, fearful, happy, sad, surprised, neutral |
| Target Accuracy | 68-72% (v1 was 64.4%) |
| Optimizer | AdamW (lr=0.001, weight_decay=5e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=15, T_mult=2) |
| Loss | CrossEntropy + Label Smoothing (0.1) + Class Weights |
| Regularization | Mixup, Dropout 0.5, Weight Decay, RandomErasing |
| Epochs | 60 (early stopping patience=15) |
| Training Platform | Google Colab (NVIDIA T4 GPU) |

> **Note:** FER2013 has ~65% human inter-rater agreement due to ambiguous labels. State-of-the-art models with 50-100M+ params achieve ~73-76%. Our 2.8M param model targets 68-72% — near human-level accuracy with a model small enough to run in your browser.

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
│       └── sensai_emotion.onnx   # Trained ONNX model
├── sensai-model/
│   ├── models/
│   │   ├── emotion_cnn.py        # SensAIEmotionCNN v2 architecture
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
│       └── training_history.json # Training metrics
├── train_real.py                 # Standalone training script (v2)
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

## Training the Model

### Google Colab (Recommended)

Use a T4 GPU runtime on Google Colab. Copy the training code from `train_real.py` and upload your FER2013 CSV.

### Local Training

```bash
pip install torch torchvision onnx onnxruntime

# Option 1: Using FER2013 CSV
python train_real.py --fer2013-csv path/to/fer2013.csv --epochs 60

# Option 2: Using HuggingFace (auto-downloads)
python train_real.py --fer2013 --epochs 60

# Option 3: Synthetic data (offline, for testing)
python train_real.py
```

After training, copy `sensai-model/export/sensai_emotion.onnx` to `public/models/`.

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
