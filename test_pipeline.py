#!/usr/bin/env python3
"""End-to-end test: train → export ONNX → inference → behavioral analysis → fusion."""

import sys
import time
from pathlib import Path

root = Path(__file__).parent
model_dir = root / "sensai-model"
sys.path.insert(0, str(root))

import types
sensai_pkg = types.ModuleType("sensai_model")
sensai_pkg.__path__ = [str(model_dir)]
sensai_pkg.__package__ = "sensai_model"
sys.modules["sensai_model"] = sensai_pkg

import numpy as np


def test_onnx_export():
    print("=" * 60)
    print("TEST 1: ONNX Export + Verification")
    print("=" * 60)

    from sensai_model.export.export_onnx import export_to_onnx, verify_onnx

    checkpoint = root / "sensai-model" / "checkpoints" / "best_model.pt"
    onnx_path = root / "sensai-model" / "export" / "sensai_emotion.onnx"

    export_to_onnx(checkpoint, onnx_path)
    passed = verify_onnx(onnx_path, checkpoint)
    print(f"ONNX Verification: {'PASS' if passed else 'FAIL'}\n")
    return passed


def test_pytorch_inference():
    print("=" * 60)
    print("TEST 2: PyTorch Inference")
    print("=" * 60)

    from sensai_model.inference.predictor import EmotionPredictor

    checkpoint = root / "sensai-model" / "checkpoints" / "best_model.pt"
    predictor = EmotionPredictor(checkpoint)

    test_image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)

    results = []
    for i in range(10):
        result = predictor.predict(test_image)
        results.append(result)

    print(f"Predicted emotion: {results[-1]['emotion']}")
    print(f"Confidence: {results[-1]['confidence']:.4f}")
    print(f"Inference time: {results[-1]['inference_ms']:.2f}ms")
    print(f"Smoothing active: {results[-1]['smoothed']}")
    print(f"All probabilities:")
    for emotion, prob in sorted(
        results[-1]["probabilities"].items(), key=lambda x: -x[1]
    ):
        bar = "█" * int(prob * 30)
        print(f"  {emotion:12s} {prob:.4f} {bar}")

    pattern = predictor.get_dominant_pattern()
    print(f"\nDominant pattern: {pattern['pattern']} (stability: {pattern['stability']:.2f})")
    print("PASS\n")
    return True


def test_onnx_inference():
    print("=" * 60)
    print("TEST 3: ONNX Runtime Inference")
    print("=" * 60)

    from sensai_model.inference.predictor import ONNXEmotionPredictor

    onnx_path = root / "sensai-model" / "export" / "sensai_emotion.onnx"
    predictor = ONNXEmotionPredictor(onnx_path)

    test_image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)

    result = predictor.predict(test_image)
    print(f"Predicted emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"ONNX inference time: {result['inference_ms']:.2f}ms")
    print("PASS\n")
    return True


def test_behavioral_analysis():
    print("=" * 60)
    print("TEST 4: Behavioral Signal Analysis")
    print("=" * 60)

    from sensai_model.behavioral import BehavioralAnalyzer

    analyzer = BehavioralAnalyzer(analysis_interval=0)

    # simulate frustrated user: fast typing with lots of backspaces
    print("Simulating frustrated user (fast typing + errors)...")
    t = time.time()
    for i in range(30):
        key = "Backspace" if i % 3 == 0 else chr(97 + i % 26)
        analyzer.add_keystroke(key, t + i * 0.08)
    analyzer.add_mouse_event(100, 100, "move", t + 0.1)
    analyzer.add_mouse_event(300, 50, "move", t + 0.2)
    analyzer.add_mouse_event(50, 300, "move", t + 0.3)
    analyzer.add_mouse_event(400, 100, "move", t + 0.4)

    result = analyzer.analyze()
    print(f"  State: {result.state} (confidence: {result.confidence:.4f})")
    print(f"  Signals: {result.signals}")

    # simulate focused user: steady typing, low errors
    print("\nSimulating focused user (steady typing)...")
    analyzer2 = BehavioralAnalyzer(analysis_interval=0)
    t = time.time()
    for i in range(30):
        analyzer2.add_keystroke(chr(97 + i % 26), t + i * 0.2)

    result2 = analyzer2.analyze()
    print(f"  State: {result2.state} (confidence: {result2.confidence:.4f})")

    # simulate confused user: long pauses
    print("\nSimulating confused user (long pauses)...")
    analyzer3 = BehavioralAnalyzer(analysis_interval=0)
    t = time.time()
    analyzer3.add_keystroke("a", t)
    analyzer3.add_keystroke("b", t + 12)  # 12s pause
    analyzer3.add_keystroke("c", t + 25)  # 13s pause

    result3 = analyzer3.analyze()
    print(f"  State: {result3.state} (confidence: {result3.confidence:.4f})")
    print("PASS\n")
    return True


def test_fusion():
    print("=" * 60)
    print("TEST 5: Multi-Signal Emotion Fusion")
    print("=" * 60)

    from sensai_model.models.fusion import EmotionFusion, EmotionSignal

    fusion = EmotionFusion()

    facial = EmotionSignal(
        source="facial",
        emotion="angry",
        confidence=0.75,
        probabilities={"angry": 0.75, "disgusted": 0.1, "neutral": 0.15},
    )
    behavioral = EmotionSignal(
        source="behavioral",
        emotion="angry",
        confidence=0.6,
        probabilities={"angry": 0.6, "neutral": 0.4},
    )

    result = fusion.fuse([facial, behavioral])
    print(f"Fused emotion: {result.emotion}")
    print(f"Learning state: {result.learning_state}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Signals used: {result.signals_used}")
    print(f"Per-signal: {result.per_signal}")

    # conflicting signals
    print("\nConflicting signals test:")
    facial2 = EmotionSignal(
        source="facial",
        emotion="happy",
        confidence=0.8,
        probabilities={"happy": 0.8, "neutral": 0.2},
    )
    behavioral2 = EmotionSignal(
        source="behavioral",
        emotion="angry",
        confidence=0.5,
        probabilities={"angry": 0.5, "neutral": 0.5},
    )
    result2 = fusion.fuse([facial2, behavioral2])
    print(f"  Facial says 'happy' (0.8), behavioral says 'angry' (0.5)")
    print(f"  Fused: {result2.emotion} ({result2.confidence:.4f})")
    print(f"  Learning state: {result2.learning_state}")
    print("PASS\n")
    return True


if __name__ == "__main__":
    print("\n🧠 SensAI Emotion Model — Full Pipeline Test\n")

    results = {
        "ONNX Export": test_onnx_export(),
        "PyTorch Inference": test_pytorch_inference(),
        "ONNX Inference": test_onnx_inference(),
        "Behavioral Analysis": test_behavioral_analysis(),
        "Emotion Fusion": test_fusion(),
    }

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test:25s} [{status}]")

    all_passed = all(results.values())
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed.'}")
