"""
Multi-signal emotion fusion.

Combines facial, vocal, and behavioral signals into a unified emotion prediction.
Uses weighted averaging with confidence-based dynamic weighting.
"""

from dataclasses import dataclass


LEARNING_STATE_MAP = {
    "angry": "frustrated",
    "disgusted": "frustrated",
    "fearful": "struggling",
    "happy": "excited",
    "sad": "struggling",
    "surprised": "confused",
    "neutral": "neutral",
}


@dataclass
class EmotionSignal:
    source: str  # "facial", "vocal", "behavioral"
    emotion: str
    confidence: float
    probabilities: dict[str, float]


@dataclass
class FusedEmotion:
    emotion: str
    learning_state: str
    confidence: float
    signals_used: list[str]
    per_signal: dict[str, dict]


class EmotionFusion:
    """
    Fuses multiple emotion signals into one prediction.

    Weighting strategy:
        - Base weights per signal source (facial > vocal > behavioral)
        - Dynamic adjustment based on each signal's confidence
        - If one signal is missing, others redistribute weight
    """

    BASE_WEIGHTS = {
        "facial": 0.50,
        "vocal": 0.30,
        "behavioral": 0.20,
    }

    EMOTIONS = [
        "angry",
        "disgusted",
        "fearful",
        "happy",
        "sad",
        "surprised",
        "neutral",
    ]

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or self.BASE_WEIGHTS.copy()

    def fuse(self, signals: list[EmotionSignal]) -> FusedEmotion:
        if not signals:
            return FusedEmotion(
                emotion="neutral",
                learning_state="neutral",
                confidence=0.0,
                signals_used=[],
                per_signal={},
            )

        if len(signals) == 1:
            s = signals[0]
            return FusedEmotion(
                emotion=s.emotion,
                learning_state=LEARNING_STATE_MAP.get(s.emotion, "neutral"),
                confidence=s.confidence,
                signals_used=[s.source],
                per_signal={s.source: {"emotion": s.emotion, "confidence": s.confidence}},
            )

        active_weights = {}
        total_base = 0.0
        for s in signals:
            w = self.weights.get(s.source, 0.1)
            active_weights[s.source] = w * s.confidence
            total_base += active_weights[s.source]

        if total_base == 0:
            total_base = 1.0

        fused_probs: dict[str, float] = {e: 0.0 for e in self.EMOTIONS}

        for s in signals:
            norm_weight = active_weights[s.source] / total_base
            for emotion in self.EMOTIONS:
                prob = s.probabilities.get(emotion, 0.0)
                fused_probs[emotion] += prob * norm_weight

        dominant = max(fused_probs, key=fused_probs.get)  # type: ignore
        confidence = fused_probs[dominant]

        return FusedEmotion(
            emotion=dominant,
            learning_state=LEARNING_STATE_MAP.get(dominant, "neutral"),
            confidence=min(confidence, 0.99),
            signals_used=[s.source for s in signals],
            per_signal={
                s.source: {"emotion": s.emotion, "confidence": s.confidence}
                for s in signals
            },
        )

    def fuse_with_behavioral(
        self,
        facial_emotion: str | None,
        facial_confidence: float,
        facial_probs: dict[str, float],
        behavioral_state: str,
        behavioral_confidence: float,
    ) -> FusedEmotion:
        """Convenience method for facial + behavioral fusion."""
        signals = []

        if facial_emotion and facial_confidence > 0:
            signals.append(
                EmotionSignal(
                    source="facial",
                    emotion=facial_emotion,
                    confidence=facial_confidence,
                    probabilities=facial_probs,
                )
            )

        if behavioral_state:
            behavioral_emotion_map = {
                "confused": "surprised",
                "frustrated": "angry",
                "bored": "neutral",
                "focused": "neutral",
                "excited": "happy",
                "struggling": "sad",
                "neutral": "neutral",
            }
            mapped = behavioral_emotion_map.get(behavioral_state, "neutral")
            signals.append(
                EmotionSignal(
                    source="behavioral",
                    emotion=mapped,
                    confidence=behavioral_confidence,
                    probabilities={mapped: behavioral_confidence},
                )
            )

        return self.fuse(signals)
