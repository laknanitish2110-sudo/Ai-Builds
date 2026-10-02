"""
Real-time emotion predictor with temporal smoothing.

Loads a trained SensAI model and provides a clean prediction API.
Includes temporal smoothing to avoid jittery predictions.
"""

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from sensai_model.models import SensAIEmotionCNN, MobileEmotionNet


EMOTION_LABELS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "sad",
    "surprised",
    "neutral",
]


class EmotionPredictor:
    """
    Production-ready emotion predictor with temporal smoothing.

    Features:
        - Loads from checkpoint with auto model type detection
        - Exponential moving average for smooth predictions
        - Configurable confidence threshold
        - Tracks prediction history for pattern analysis
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        smoothing_window: int = 5,
        smoothing_alpha: float = 0.6,
        confidence_threshold: float = 0.3,
    ):
        self.device = torch.device(device)
        self.smoothing_alpha = smoothing_alpha
        self.confidence_threshold = confidence_threshold
        self.history: deque[dict] = deque(maxlen=100)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model_config = checkpoint.get("config", {})

        model_type = model_config.get("model_type", "sensai_cnn")
        num_classes = model_config.get("num_classes", 7)
        in_channels = model_config.get("in_channels", 1)
        dropout = model_config.get("dropout", 0.4)
        use_attention = model_config.get("use_attention", True)
        self.img_size = model_config.get("img_size", 48)

        if model_type == "mobile_emotion":
            self.model = MobileEmotionNet(
                num_classes=num_classes,
                in_channels=in_channels,
                dropout=dropout,
            )
        else:
            self.model = SensAIEmotionCNN(
                num_classes=num_classes,
                in_channels=in_channels,
                dropout=dropout,
                use_attention=use_attention,
            )

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self._smooth_probs: np.ndarray | None = None
        self._window: deque[np.ndarray] = deque(maxlen=smoothing_window)

    @torch.no_grad()
    def predict(self, image: Image.Image | np.ndarray) -> dict:
        """
        Predict emotion from a face image.

        Args:
            image: PIL Image or numpy array (grayscale or RGB face crop)

        Returns:
            {
                "emotion": str,
                "confidence": float,
                "probabilities": {emotion: float, ...},
                "raw_probabilities": {emotion: float, ...},
                "smoothed": bool,
                "timestamp": float,
            }
        """
        start = time.time()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = Image.fromarray(image).convert("L")
            elif image.ndim == 2:
                image = Image.fromarray(image, mode="L")
            else:
                image = Image.fromarray(image.squeeze(), mode="L")

        if image.mode != "L":
            image = image.convert("L")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        raw_probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # temporal smoothing
        self._window.append(raw_probs)
        if self._smooth_probs is None:
            self._smooth_probs = raw_probs.copy()
        else:
            self._smooth_probs = (
                self.smoothing_alpha * raw_probs
                + (1 - self.smoothing_alpha) * self._smooth_probs
            )

        probs = self._smooth_probs
        dominant_idx = int(np.argmax(probs))
        confidence = float(probs[dominant_idx])

        result = {
            "emotion": EMOTION_LABELS[dominant_idx],
            "confidence": confidence,
            "probabilities": {
                label: float(p) for label, p in zip(EMOTION_LABELS, probs)
            },
            "raw_probabilities": {
                label: float(p) for label, p in zip(EMOTION_LABELS, raw_probs)
            },
            "smoothed": len(self._window) > 1,
            "timestamp": time.time(),
            "inference_ms": (time.time() - start) * 1000,
        }

        self.history.append(result)
        return result

    def get_dominant_pattern(self, window: int = 10) -> dict:
        """Analyze recent prediction history for dominant patterns."""
        recent = list(self.history)[-window:]
        if not recent:
            return {"pattern": "neutral", "confidence": 0, "stability": 0}

        emotions = [r["emotion"] for r in recent]
        counts: dict[str, int] = {}
        for e in emotions:
            counts[e] = counts.get(e, 0) + 1

        dominant = max(counts, key=counts.get)  # type: ignore
        stability = counts[dominant] / len(emotions)

        avg_conf = np.mean([
            r["confidence"] for r in recent if r["emotion"] == dominant
        ])

        return {
            "pattern": dominant,
            "confidence": float(avg_conf),
            "stability": stability,
            "distribution": counts,
        }

    def reset_smoothing(self):
        """Reset temporal smoothing state."""
        self._smooth_probs = None
        self._window.clear()


class ONNXEmotionPredictor:
    """Predictor using ONNX runtime for cross-platform deployment."""

    def __init__(
        self,
        model_path: str | Path,
        smoothing_alpha: float = 0.6,
    ):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.smoothing_alpha = smoothing_alpha
        self._smooth_probs: np.ndarray | None = None

        input_shape = self.session.get_inputs()[0].shape
        self.img_size = input_shape[-1] if len(input_shape) == 4 else 48

        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def predict(self, image: Image.Image | np.ndarray) -> dict:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("L")
        if image.mode != "L":
            image = image.convert("L")

        tensor = self.transform(image).unsqueeze(0).numpy()

        start = time.time()
        outputs = self.session.run(None, {self.input_name: tensor})
        raw_probs = self._softmax(outputs[0][0])

        if self._smooth_probs is None:
            self._smooth_probs = raw_probs.copy()
        else:
            self._smooth_probs = (
                self.smoothing_alpha * raw_probs
                + (1 - self.smoothing_alpha) * self._smooth_probs
            )

        probs = self._smooth_probs
        dominant_idx = int(np.argmax(probs))

        return {
            "emotion": EMOTION_LABELS[dominant_idx],
            "confidence": float(probs[dominant_idx]),
            "probabilities": {
                label: float(p) for label, p in zip(EMOTION_LABELS, probs)
            },
            "inference_ms": (time.time() - start) * 1000,
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()
