"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { EmotionLabel, EmotionReading } from "@/types";

interface UseEmotionDetectionOptions {
  interval?: number;
  enabled?: boolean;
}

interface UseEmotionDetectionReturn {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  isLoading: boolean;
  isActive: boolean;
  error: string | null;
  currentEmotion: EmotionReading | null;
  emotionHistory: EmotionReading[];
  startDetection: () => Promise<void>;
  stopDetection: () => void;
  useDemoMode: () => void;
}

export function useEmotionDetection(
  options: UseEmotionDetectionOptions = {}
): UseEmotionDetectionReturn {
  const { interval = 1000, enabled = true } = options;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const faceApiRef = useRef<typeof import("face-api.js") | null>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentEmotion, setCurrentEmotion] =
    useState<EmotionReading | null>(null);
  const [emotionHistory, setEmotionHistory] = useState<EmotionReading[]>([]);
  const [demoMode, setDemoMode] = useState(false);

  const generateDemoEmotion = useCallback((): EmotionReading => {
    const emotions: Record<EmotionLabel, number> = {
      happy: 0,
      sad: 0,
      angry: 0,
      surprised: 0,
      fearful: 0,
      disgusted: 0,
      neutral: 0,
    };

    const states: EmotionLabel[] = [
      "neutral",
      "neutral",
      "happy",
      "surprised",
      "neutral",
      "happy",
      "neutral",
      "surprised",
      "fearful",
      "neutral",
    ];

    const t = (Date.now() / 5000) % states.length;
    const idx = Math.floor(t);
    const dominant = states[idx];

    emotions[dominant] = 0.5 + Math.random() * 0.4;
    for (const key of Object.keys(emotions) as EmotionLabel[]) {
      if (key !== dominant) {
        emotions[key] = Math.random() * 0.2;
      }
    }

    return {
      timestamp: Date.now(),
      emotions,
      dominant,
      confidence: 0.7 + Math.random() * 0.3,
    };
  }, []);

  const useDemoMode = useCallback(() => {
    setDemoMode(true);
    setIsActive(true);
    setIsLoading(false);
    setError(null);
  }, []);

  useEffect(() => {
    if (!demoMode || !isActive) return;

    intervalRef.current = setInterval(() => {
      const reading = generateDemoEmotion();
      setCurrentEmotion(reading);
      setEmotionHistory((prev) => [...prev.slice(-100), reading]);
    }, interval);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [demoMode, isActive, interval, generateDemoEmotion]);

  const loadFaceApi = async () => {
    if (faceApiRef.current) return faceApiRef.current;

    try {
      const faceapi = await import("face-api.js");
      const MODEL_URL =
        "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";

      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
      ]);

      faceApiRef.current = faceapi;
      return faceapi;
    } catch {
      throw new Error(
        "Failed to load face detection models. Check your internet connection."
      );
    }
  };

  const startDetection = async () => {
    if (!enabled) return;
    setIsLoading(true);
    setError(null);

    try {
      const faceapi = await loadFaceApi();

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: "user" },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsActive(true);
      setIsLoading(false);

      intervalRef.current = setInterval(async () => {
        if (!videoRef.current) return;

        const detections = await faceapi
          .detectSingleFace(
            videoRef.current,
            new faceapi.TinyFaceDetectorOptions()
          )
          .withFaceExpressions();

        if (detections) {
          const expressions = detections.expressions;
          const emotions: Record<EmotionLabel, number> = {
            happy: expressions.happy,
            sad: expressions.sad,
            angry: expressions.angry,
            surprised: expressions.surprised,
            fearful: expressions.fearful,
            disgusted: expressions.disgusted,
            neutral: expressions.neutral,
          };

          let dominant: EmotionLabel = "neutral";
          let maxVal = 0;
          for (const [key, val] of Object.entries(emotions)) {
            if (val > maxVal) {
              dominant = key as EmotionLabel;
              maxVal = val;
            }
          }

          const reading: EmotionReading = {
            timestamp: Date.now(),
            emotions,
            dominant,
            confidence: maxVal,
          };

          setCurrentEmotion(reading);
          setEmotionHistory((prev) => [...prev.slice(-100), reading]);
        }
      }, interval);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Camera access denied";
      setError(message);
      setIsLoading(false);
      setDemoMode(true);
      setIsActive(true);
    }
  };

  const stopDetection = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    setIsActive(false);
    setDemoMode(false);
  }, []);

  useEffect(() => {
    return () => {
      stopDetection();
    };
  }, [stopDetection]);

  return {
    videoRef,
    canvasRef,
    isLoading,
    isActive,
    error,
    currentEmotion,
    emotionHistory,
    startDetection,
    stopDetection,
    useDemoMode,
  };
}
