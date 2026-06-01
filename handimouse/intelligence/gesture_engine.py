"""
Intelligence Layer: Heuristics-based gesture recognition engine.
Translates normalized 3D hand landmarks into debounced, cooldown-validated
mouse control event states.

Gesture Mapping:
  - MOVE_CURSOR  : Index finger extended only
  - LEFT_CLICK   : Thumb tip  pinch to Ring finger tip  (landmark 4 -> 16)
  - DRAG         : Sustained left-click pinch held beyond drag_hold_frames
  - RIGHT_CLICK  : Thumb tip  pinch to Middle finger tip (landmark 4 -> 12)
  - SCROLL       : All 4 fingers extended, move hand up/down
"""

import collections
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("handimouse.intelligence.gesture_engine")


class GestureEngine:
    """
    A robust rule-based gesture classifier using direct 3D landmark distance
    measurements for click/drag gestures (bypassing debouncing for zero lag),
    and temporal debouncing for scroll/cursor mode transitions.

    Click & Drag use a "tap-on-release" design:
      - Pinch press  -> no event yet
      - Pinch release (short) -> CLICK fires
      - Pinch hold (>drag_hold_frames) -> DRAG_START; release -> DRAG_RELEASE
    """

    # MediaPipe landmark indices
    _THUMB_TIP   = 4
    _MIDDLE_TIP  = 12   # Right-click pinch target
    _RING_TIP    = 16   # Left-click / drag pinch target

    # Default configuration
    DEFAULT_CONFIG = {
        "pinch_threshold":       0.15,   # Thumb-to-ring distance relative to hand_scale  -> left click
        "right_click_threshold": 0.15,   # Thumb-to-middle distance relative to hand_scale -> right click
        "scroll_sensitivity":    0.05,   # Vertical displacement relative to hand_scale to trigger 1 tick
        "debounce_frames":       3,      # History frames to confirm scroll/cursor transition
        "click_cooldown":        0.35,   # Minimum time between any two click events
        "double_click_window":   0.40,   # Max gap between two left-pinches to register as double-click
        "scroll_cooldown":       0.12,   # Rate-limit for scroll events in seconds
        "drag_hold_frames":      8,      # Frames of sustained left-pinch before drag starts
        "min_tracking_confidence": 0.65  # Minimum detection score to process input
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # Debounced state (only used for SCROLL / MOVE_CURSOR)
        self.active_gesture = "IDLE"
        self.raw_history = collections.deque(maxlen=max(1, self.config["debounce_frames"]))

        # Left-click / drag state
        self._left_pinch_frames = 0
        self.in_drag_mode = False

        # Right-click state
        self._right_pinch_frames = 0
        self._right_click_fired = False   # edge-trigger: fire once per pinch hold

        # Double-click detection
        self._last_left_click_time: float = 0.0

        # Scroll anchor
        self.scroll_start_y: Optional[float] = None

        # Event cooldown timestamps
        self._last_event_times: Dict[str, float] = {
            "LEFT_CLICK":  0.0,
            "RIGHT_CLICK": 0.0,
            "SCROLL_UP":   0.0,
            "SCROLL_DOWN": 0.0,
        }

        logger.info(
            f"GestureEngine initialized | "
            f"Left-click: thumb->ring | Right-click: thumb->middle | "
            f"Pinch threshold: {self.config['pinch_threshold']} | "
            f"Double-click window: {self.config['double_click_window']}s"
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_pinching(self) -> bool:
        """True when any pinch gesture is currently held."""
        return self._left_pinch_frames > 0 or self.in_drag_mode or self._right_pinch_frames > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dist(self, p1: Tuple, p2: Tuple) -> float:
        """Euclidean distance between two 3D landmarks."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def _classify_raw_gesture(
        self,
        landmarks: List[Tuple[float, float, float]],
        fingers_up: List[int],
        hand_scale: float,
    ) -> Tuple[str, float]:
        """
        Classifies a single frame for debounced (scroll/cursor) states only.
        Pinch-based clicks are handled directly in process_state.
        """
        # SCROLL: Index, Middle, Ring, Pinky all extended
        if fingers_up[1:] == [1, 1, 1, 1]:
            return "SCROLL", 1.0

        # MOVE_CURSOR: Index only extended
        if fingers_up[1] == 1 and fingers_up[2:] == [0, 0, 0]:
            return "MOVE_CURSOR", 1.0

        return "IDLE", 1.0

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process_state(self, tracking_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single frame's tracking result and return a gesture event.

        Returns:
            {
                "gesture":       str,   # Active gesture label
                "confidence":    float,
                "event_trigger": str,   # "CLICK_LEFT" | "DOUBLE_CLICK" | "CLICK_RIGHT"
                                        # "DRAG_START" | "DRAG_RELEASE" | ""
                "scroll_tick":   int    # 1 = up, -1 = down, 0 = none
            }
        """
        now = time.time()

        idle_event: Dict[str, Any] = {
            "gesture":       "IDLE",
            "confidence":    1.0,
            "event_trigger": "",
            "scroll_tick":   0,
        }

        # ── Guard: no hand detected ────────────────────────────────────
        if (not tracking_result["hand_detected"] or
                tracking_result["confidence"] < self.config["min_tracking_confidence"]):
            event_trigger = ""
            if self.in_drag_mode:
                event_trigger = "DRAG_RELEASE"
                self.in_drag_mode = False
                self._left_pinch_frames = 0
                logger.info("Gesture Engine: DRAG_RELEASE (tracking lost)")
            self.raw_history.clear()
            self.active_gesture = "IDLE"
            self.scroll_start_y = None
            self._right_pinch_frames = 0
            self._right_click_fired = False
            if event_trigger:
                idle_event["event_trigger"] = event_trigger
            return idle_event

        landmarks   = tracking_result["landmarks"]
        fingers_up  = tracking_result["fingers_up"]

        # Hand scale: wrist (0) to middle knuckle (9)
        hand_scale = self._dist(landmarks[0], landmarks[9]) or 0.1

        event_trigger = ""
        scroll_tick   = 0
        confidence    = 1.0

        # ── Debounce non-pinch gestures ────────────────────────────────
        raw, confidence = self._classify_raw_gesture(landmarks, fingers_up, hand_scale)
        self.raw_history.append(raw)

        if len(self.raw_history) == self.config["debounce_frames"]:
            most_common = collections.Counter(self.raw_history).most_common(1)[0][0]
            if self.raw_history.count(most_common) == self.config["debounce_frames"]:
                self.active_gesture = most_common

        # Clear scroll anchor when not scrolling
        if self.active_gesture != "SCROLL":
            self.scroll_start_y = None

        # ── LEFT CLICK / DRAG interceptor (thumb -> ring tip) ──────────
        thumb = landmarks[self._THUMB_TIP]
        ring  = landmarks[self._RING_TIP]
        norm_left = self._dist(thumb, ring) / hand_scale
        left_pinching = norm_left < self.config["pinch_threshold"]

        if left_pinching:
            self._left_pinch_frames += 1
            self.active_gesture = "MOVE_CURSOR"   # keep cursor active during pinch

            if self._left_pinch_frames >= self.config["drag_hold_frames"]:
                self.active_gesture = "DRAG_MODE"
                if not self.in_drag_mode:
                    event_trigger = "DRAG_START"
                    self.in_drag_mode = True
                    logger.info("Gesture Engine Event Triggered: DRAG_START")
        else:
            if self.in_drag_mode:
                event_trigger = "DRAG_RELEASE"
                self.in_drag_mode = False
                logger.info("Gesture Engine Event Triggered: DRAG_RELEASE")
            elif self._left_pinch_frames > 0:
                # Quick tap released → fire LEFT_CLICK or DOUBLE_CLICK
                elapsed = now - self._last_event_times["LEFT_CLICK"]
                if elapsed >= self.config["click_cooldown"]:
                    time_since_last = now - self._last_left_click_time
                    if time_since_last <= self.config["double_click_window"]:
                        event_trigger = "DOUBLE_CLICK"
                        self._last_left_click_time = 0.0
                        logger.info("Gesture Engine Event Triggered: DOUBLE_CLICK")
                    else:
                        event_trigger = "CLICK_LEFT"
                        self._last_left_click_time = now
                        logger.info("Gesture Engine Event Triggered: CLICK_LEFT")
                    self._last_event_times["LEFT_CLICK"] = now
                self.active_gesture = "MOVE_CURSOR"
            self._left_pinch_frames = 0

        # ── RIGHT CLICK interceptor (thumb -> middle tip) ──────────────
        # Only evaluate if we are NOT currently doing a left pinch/drag
        if not left_pinching and not self.in_drag_mode:
            middle = landmarks[self._MIDDLE_TIP]
            norm_right = self._dist(thumb, middle) / hand_scale
            right_pinching = norm_right < self.config["right_click_threshold"]

            if right_pinching:
                self._right_pinch_frames += 1
                self.active_gesture = "MOVE_CURSOR"   # keep cursor active

                # Edge-trigger on first frame of pinch (fire immediately on press)
                if self._right_pinch_frames == 1 and not self._right_click_fired:
                    elapsed = now - self._last_event_times["RIGHT_CLICK"]
                    if elapsed >= self.config["click_cooldown"]:
                        event_trigger = "CLICK_RIGHT"
                        self._last_event_times["RIGHT_CLICK"] = now
                        self._right_click_fired = True
                        logger.info("Gesture Engine Event Triggered: CLICK_RIGHT")
            else:
                self._right_pinch_frames = 0
                self._right_click_fired = False
        else:
            # During left pinch, reset right-click state
            self._right_pinch_frames = 0
            self._right_click_fired = False

        # ── SCROLL processing ──────────────────────────────────────────
        if (self.active_gesture == "SCROLL"
                and not left_pinching
                and not self.in_drag_mode):
            wrist_y = landmarks[0][1]
            if self.scroll_start_y is None:
                self.scroll_start_y = wrist_y
            else:
                delta_y = self.scroll_start_y - wrist_y
                norm_delta = delta_y / hand_scale
                elapsed_scroll = now - max(
                    self._last_event_times["SCROLL_UP"],
                    self._last_event_times["SCROLL_DOWN"]
                )
                if elapsed_scroll >= self.config["scroll_cooldown"]:
                    if norm_delta > self.config["scroll_sensitivity"]:
                        scroll_tick = 1
                        self._last_event_times["SCROLL_UP"] = now
                        self.scroll_start_y = wrist_y
                        logger.debug("Gesture Engine Event: SCROLL_UP")
                    elif norm_delta < -self.config["scroll_sensitivity"]:
                        scroll_tick = -1
                        self._last_event_times["SCROLL_DOWN"] = now
                        self.scroll_start_y = wrist_y
                        logger.debug("Gesture Engine Event: SCROLL_DOWN")

        return {
            "gesture":       self.active_gesture,
            "confidence":    float(confidence),
            "event_trigger": event_trigger,
            "scroll_tick":   scroll_tick,
        }
