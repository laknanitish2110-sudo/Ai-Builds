"""
Behavioral signal analyzer — emotion detection WITHOUT a camera.

Detects emotional state from user interaction patterns:
  - Typing speed and variance
  - Pause duration between actions
  - Error rate (backspace frequency)
  - Mouse movement patterns (jitter, speed)
  - Click patterns (frequency, force via duration)
  - Scroll behavior

This is SensAI's secret weapon — no camera needed, works everywhere,
and captures signals facial recognition misses.
"""

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class KeystrokeEvent:
    key: str
    timestamp: float
    is_backspace: bool = False


@dataclass
class MouseEvent:
    x: float
    y: float
    timestamp: float
    event_type: str = "move"  # "move", "click", "scroll"


@dataclass
class BehavioralState:
    state: str  # confused, frustrated, bored, focused, excited, struggling, neutral
    confidence: float
    signals: dict = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class BehavioralAnalyzer:
    """
    Analyzes interaction patterns to infer emotional state.

    Feed it keystroke and mouse events, it returns emotional state.
    No ML required — rule-based with tunable thresholds.
    Can be combined with facial emotion for higher accuracy.
    """

    def __init__(
        self,
        typing_window: int = 20,
        mouse_window: int = 50,
        analysis_interval: float = 2.0,
    ):
        self.keystrokes: deque[KeystrokeEvent] = deque(maxlen=typing_window)
        self.mouse_events: deque[MouseEvent] = deque(maxlen=mouse_window)
        self.analysis_interval = analysis_interval
        self._last_analysis = 0.0
        self.history: deque[BehavioralState] = deque(maxlen=100)

    def add_keystroke(self, key: str, timestamp: float | None = None):
        ts = timestamp or time.time()
        self.keystrokes.append(
            KeystrokeEvent(
                key=key,
                timestamp=ts,
                is_backspace=key in ("Backspace", "Delete"),
            )
        )

    def add_mouse_event(
        self,
        x: float,
        y: float,
        event_type: str = "move",
        timestamp: float | None = None,
    ):
        ts = timestamp or time.time()
        self.mouse_events.append(
            MouseEvent(x=x, y=y, timestamp=ts, event_type=event_type)
        )

    def analyze(self) -> BehavioralState:
        """Run analysis on current interaction buffer."""
        now = time.time()
        if now - self._last_analysis < self.analysis_interval:
            if self.history:
                return self.history[-1]

        self._last_analysis = now

        typing_signals = self._analyze_typing()
        mouse_signals = self._analyze_mouse()
        pause_signals = self._analyze_pauses()

        signals = {**typing_signals, **mouse_signals, **pause_signals}
        state = self._infer_state(signals)

        result = BehavioralState(
            state=state["state"],
            confidence=state["confidence"],
            signals=signals,
        )
        self.history.append(result)
        return result

    def _analyze_typing(self) -> dict:
        if len(self.keystrokes) < 3:
            return {
                "typing_speed": 0,
                "typing_variance": 0,
                "error_rate": 0,
                "typing_burst": False,
            }

        keys = list(self.keystrokes)
        intervals = []
        for i in range(1, len(keys)):
            dt = keys[i].timestamp - keys[i - 1].timestamp
            if 0 < dt < 5:  # ignore gaps > 5s
                intervals.append(dt)

        if not intervals:
            return {
                "typing_speed": 0,
                "typing_variance": 0,
                "error_rate": 0,
                "typing_burst": False,
            }

        avg_interval = sum(intervals) / len(intervals)
        typing_speed = 1.0 / avg_interval if avg_interval > 0 else 0

        variance = (
            sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
        ) ** 0.5

        backspaces = sum(1 for k in keys if k.is_backspace)
        error_rate = backspaces / len(keys) if keys else 0

        burst = any(i < 0.08 for i in intervals[-5:]) if len(intervals) >= 5 else False

        return {
            "typing_speed": typing_speed,
            "typing_variance": variance,
            "error_rate": error_rate,
            "typing_burst": burst,
        }

    def _analyze_mouse(self) -> dict:
        moves = [e for e in self.mouse_events if e.event_type == "move"]

        if len(moves) < 3:
            return {
                "mouse_speed": 0,
                "mouse_jitter": 0,
                "mouse_idle": True,
            }

        speeds = []
        direction_changes = 0

        for i in range(1, len(moves)):
            dx = moves[i].x - moves[i - 1].x
            dy = moves[i].y - moves[i - 1].y
            dt = moves[i].timestamp - moves[i - 1].timestamp

            if dt > 0:
                dist = (dx**2 + dy**2) ** 0.5
                speeds.append(dist / dt)

            if i >= 2:
                prev_dx = moves[i - 1].x - moves[i - 2].x
                prev_dy = moves[i - 1].y - moves[i - 2].y
                dot = dx * prev_dx + dy * prev_dy
                if dot < 0:
                    direction_changes += 1

        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        jitter = direction_changes / max(len(moves) - 2, 1)

        return {
            "mouse_speed": avg_speed,
            "mouse_jitter": jitter,
            "mouse_idle": avg_speed < 10,
        }

    def _analyze_pauses(self) -> dict:
        all_events = []
        for k in self.keystrokes:
            all_events.append(k.timestamp)
        for m in self.mouse_events:
            all_events.append(m.timestamp)

        if len(all_events) < 2:
            return {"longest_pause": 0, "pause_frequency": 0}

        all_events.sort()
        pauses = []
        for i in range(1, len(all_events)):
            gap = all_events[i] - all_events[i - 1]
            if gap > 2.0:  # pauses > 2 seconds
                pauses.append(gap)

        return {
            "longest_pause": max(pauses) if pauses else 0,
            "pause_frequency": len(pauses) / max(len(all_events) - 1, 1),
        }

    def _infer_state(self, signals: dict) -> dict:
        scores = {
            "confused": 0.0,
            "frustrated": 0.0,
            "bored": 0.0,
            "focused": 0.0,
            "excited": 0.0,
            "struggling": 0.0,
            "neutral": 0.2,  # small prior
        }

        error_rate = signals.get("error_rate", 0)
        typing_speed = signals.get("typing_speed", 0)
        typing_variance = signals.get("typing_variance", 0)
        mouse_jitter = signals.get("mouse_jitter", 0)
        mouse_idle = signals.get("mouse_idle", True)
        longest_pause = signals.get("longest_pause", 0)
        pause_frequency = signals.get("pause_frequency", 0)
        typing_burst = signals.get("typing_burst", False)

        # frustrated: high error rate + fast typing + jittery mouse
        if error_rate > 0.3:
            scores["frustrated"] += 0.4
        if error_rate > 0.15:
            scores["frustrated"] += 0.2
        if mouse_jitter > 0.4:
            scores["frustrated"] += 0.2
        if typing_speed > 8 and error_rate > 0.2:
            scores["frustrated"] += 0.2

        # confused: long pauses + frequent pauses + slow typing
        if longest_pause > 10:
            scores["confused"] += 0.3
        if pause_frequency > 0.3:
            scores["confused"] += 0.3
        if typing_speed > 0 and typing_speed < 2:
            scores["confused"] += 0.2

        # bored: idle mouse + no typing + no activity
        if mouse_idle and typing_speed == 0:
            scores["bored"] += 0.4
        if mouse_idle and typing_speed < 1:
            scores["bored"] += 0.2

        # focused: steady typing + low error rate + low variance
        if typing_speed > 3 and error_rate < 0.1:
            scores["focused"] += 0.4
        if typing_variance < 0.15 and typing_speed > 2:
            scores["focused"] += 0.3

        # excited: fast typing + bursts + low errors
        if typing_burst:
            scores["excited"] += 0.3
        if typing_speed > 8 and error_rate < 0.1:
            scores["excited"] += 0.3

        # struggling: alternating between typing and long pauses
        if pause_frequency > 0.2 and error_rate > 0.15:
            scores["struggling"] += 0.4
        if typing_variance > 0.5:
            scores["struggling"] += 0.2

        best_state = max(scores, key=scores.get)  # type: ignore
        total = sum(scores.values())
        confidence = scores[best_state] / total if total > 0 else 0

        return {"state": best_state, "confidence": min(confidence, 0.95)}
