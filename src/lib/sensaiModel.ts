import * as ort from "onnxruntime-web";
import type { EmotionLabel } from "@/types";

const EMOTION_LABELS: EmotionLabel[] = [
  "angry",
  "disgusted",
  "fearful",
  "happy",
  "sad",
  "surprised",
  "neutral",
];

let session: ort.InferenceSession | null = null;

export async function loadModel(): Promise<void> {
  if (session) return;
  ort.env.wasm.wasmPaths = "/models/";
  session = await ort.InferenceSession.create("/models/sensai_emotion.onnx", {
    executionProviders: ["wasm"],
  });
}

export async function predict(
  imageData: ImageData
): Promise<Record<EmotionLabel, number>> {
  if (!session) throw new Error("Model not loaded");

  const gray = rgbaToGrayscale(imageData, 48, 48);

  const tensor = new ort.Tensor("float32", gray, [1, 1, 48, 48]);
  const results = await session.run({ [session.inputNames[0]]: tensor });
  const output = results[session.outputNames[0]];
  const logits = output.data as Float32Array;

  const probs = softmax(logits);

  const emotions: Record<EmotionLabel, number> = {
    angry: 0,
    disgusted: 0,
    fearful: 0,
    happy: 0,
    sad: 0,
    surprised: 0,
    neutral: 0,
  };

  for (let i = 0; i < EMOTION_LABELS.length; i++) {
    emotions[EMOTION_LABELS[i]] = probs[i];
  }

  return emotions;
}

function rgbaToGrayscale(
  imageData: ImageData,
  targetW: number,
  targetH: number
): Float32Array {
  const { width, height, data } = imageData;
  const out = new Float32Array(targetW * targetH);

  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      const srcX = Math.floor((x / targetW) * width);
      const srcY = Math.floor((y / targetH) * height);
      const idx = (srcY * width + srcX) * 4;

      const gray = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
      out[y * targetW + x] = (gray / 255 - 0.5) / 0.5;
    }
  }

  return out;
}

function softmax(logits: Float32Array): number[] {
  const max = Math.max(...logits);
  const exps = Array.from(logits).map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}
