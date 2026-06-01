"""
Intelligence Layer: Heuristics-based gesture recognition engine.
Translates normalized 3D hand landmarks and fingers-up binary arrays 
into debounced, cooldown-validated mouse control event states.
"""

import collections
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("handimouse.intelligence.gesture_engine")


class GestureEngine:
    """
    A robust rule-based gesture classifier incorporating a temporal state machine, 
    debouncing, and refractory cooling periods to ensure smooth human-machine interactions.

    Optimizations & Mechanics:
    1. Temporal Debouncing: Raw heuristics are noisy due to finger angles or camera occlusion. 
       We track raw classifications over a rolling queue of `debounce_frames`. A transition is only 
       confirmed if a gesture remains consistent, eliminating state flickering.
    2. Decoupled Click vs. Drag: A pinch starts by registering a LEFT_CLICK event. 
       If the pinch remains held beyond a set timeframe, the engine seamlessly transitions 
       into DRAG_MODE. Releasing the pinch fires a DRAG_RELEASE trigger.
    3. Dynamic Scroll Delta Accumulation: Scroll mode is engaged when fingers are open. 
       Instead of mapping raw coordinates to instant triggers, it tracks vertical wrist deltas. 
       Moving the hand up/down beyond a sensitivity threshold ticks SCROLL_UP/DOWN and resets the 
       anchor, providing linear control.
    """

    # Highly-tuned rule settings
    DEFAULT_CONFIG = {
        "pinch_threshold": 0.15,           # Distance thumb-to-index relative to hand_scale
        "scroll_sensitivity": 0.05,        # Vertical displacement relative to hand_scale to trigger 1 tick
        "debounce_frames": 3,              # History frames to confirm gesture transition
        "click_cooldown": 0.35,            # Anti-double-trigger cooldown in seconds
        "scroll_cooldown": 0.12,           # Telemetry rate-limit for scrolling in seconds
        "drag_hold_frames": 6,             # Frame duration to transition from single click to sustained drag
        "min_tracking_confidence": 0.65    # Minimum detection score to process input
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Gesture Engine.

        Args:
            config: Optional custom parameter dictionary overrides.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # State tracking variables
        self.active_gesture = "IDLE"
        self.raw_history = collections.deque(maxlen=self.config["debounce_frames"])

        # Click vs Drag states
        self._pinch_frame_counter = 0
        self.in_drag_mode = False
        self._left_click_locked = False
        self._right_click_locked = False
        
        # Scroll anchor coordinates
        self.scroll_start_y: Optional[float] = None
        
        # Event rate-limiting (cooldown markers)
        self._last_event_times: Dict[str, float] = {
            "LEFT_CLICK": 0.0,
            "RIGHT_CLICK": 0.0,
            "SCROLL_UP": 0.0,
            "SCROLL_DOWN": 0.0
        }

        logger.info(
            f"GestureEngine initialized | Debounce frames: {self.config['debounce_frames']} | "
            f"Pinch threshold: {self.config['pinch_threshold']}"
        )

    def _euclidean_distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Compute spatial Euclidean distance between two 3D coordinates."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def _classify_raw_gesture(
        self, 
        landmarks: List[Tuple[float, float, float]], 
        fingers_up: List[int],
        hand_scale: float
    ) -> Tuple[str, float]:
        """
        Classifies the immediate frame's layout into a raw gesture before debouncing.

        Returns:
            Tuple[str, float]: (Raw gesture name string, gesture confidence score 0.0 - 1.0).
        """
        # 1. SCROLL MODE check: Index, Middle, Ring, and Pinky are fully extended
        # E.g., [x, 1, 1, 1, 1] -> allows thumb to be closed or open for accessibility
        if fingers_up[1:] == [1, 1, 1, 1]:
            # Scale-independent classification confidence
            confidence = sum(fingers_up[1:]) / 4.0
            return "SCROLL", confidence

        # 2. PINCH CHECK: Measure thumb tip (4) to index tip (8) distance
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch_dist = self._euclidean_distance(thumb_tip, index_tip)
        normalized_pinch = pinch_dist / hand_scale

        if normalized_pinch < self.config["pinch_threshold"]:
            # Proportional confidence mapping (closer pinch -> higher score)
            confidence = 1.0 - (normalized_pinch / self.config["pinch_threshold"])
            confidence = max(0.0, min(1.0, confidence))
            
            # Increment pinch holding counter to differentiate click vs drag
            self._pinch_frame_counter += 1
            if self._pinch_frame_counter >= self.config["drag_hold_frames"] or self.in_drag_mode:
                return "DRAG_MODE", confidence
            else:
                return "LEFT_CLICK", confidence

        # If pinch is released, reset the holding counter immediately
        if not self.in_drag_mode:
            self._pinch_frame_counter = 0

        # 3. RIGHT_CLICK check: Index and Middle fingers extended up, Ring and Pinky folded down
        # E.g., [x, 1, 1, 0, 0]
        if fingers_up[1:3] == [1, 1] and fingers_up[3:] == [0, 0]:
            return "RIGHT_CLICK", 1.0

        # 4. MOVE_CURSOR check: Index finger only is extended up
        # E.g., [x, 1, 0, 0, 0]
        if fingers_up[1] == 1 and fingers_up[2:] == [0, 0, 0]:
            return "MOVE_CURSOR", 1.0

        # 5. Fallback IDLE state
        return "IDLE", 1.0

    def process_state(self, tracking_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the tracking result from the Vision Layer and map it to a debounced, 
        cooldown-validated gesture event.

        Args:
            tracking_result: Output from HandTracker.process_frame.

        Returns:
            Dict: Mapped gesture event details.
                {
                    "gesture": str,          # Mapped active gesture event string
                    "confidence": float,     # Confidence score
                    "event_trigger": str,    # Atomic events: "CLICK_LEFT", "CLICK_RIGHT", "DRAG_START", "DRAG_RELEASE", ""
                    "scroll_tick": int       # Scroll tick offset: 1 (Up), -1 (Down), 0 (None)
                }
        """
        current_time = time.time()
        
        # Default fallback response
        idle_event = {
            "gesture": "IDLE",
            "confidence": 1.0,
            "event_trigger": "",
            "scroll_tick": 0
        }

        # Safe guard checks
        if not tracking_result["hand_detected"] or tracking_result["confidence"] < self.config["min_tracking_confidence"]:
            # If tracking drops out during a drag, ensure we emit a clean release trigger
            event_trigger = ""
            if self.in_drag_mode:
                event_trigger = "DRAG_RELEASE"
                self.in_drag_mode = False
                self._pinch_frame_counter = 0
            
            self.raw_history.clear()
            self.active_gesture = "IDLE"
            self.scroll_start_y = None
            
            if event_trigger:
                idle_event["event_trigger"] = event_trigger
            return idle_event

        landmarks = tracking_result["landmarks"]
        fingers_up = tracking_result["fingers_up"]
        
        # Calculate dynamic hand scale
        wrist = landmarks[0]
        middle_knuckle = landmarks[9]
        hand_scale = self._euclidean_distance(wrist, middle_knuckle)
        if hand_scale == 0.0:
            hand_scale = 0.1

        # 1. Get raw current frame classification
        raw_gesture, confidence = self._classify_raw_gesture(landmarks, fingers_up, hand_scale)
        self.raw_history.append(raw_gesture)

        # 2. Debouncing State Decision
        # Transition active state only if the raw gesture matches consistently over our history queue
        if len(self.raw_history) == self.config["debounce_frames"]:
            most_common = collections.Counter(self.raw_history).most_common(1)[0][0]
            # Ensure 100% agreement within history for state transitions (strict filter)
            if self.raw_history.count(most_common) == self.config["debounce_frames"]:
                self.active_gesture = most_common

        # 3. State Machine Output Translation
        event_trigger = ""
        scroll_tick = 0

        # RESET scroll anchor if we are no longer actively scrolling
        if self.active_gesture != "SCROLL":
            self.scroll_start_y = None
 
        # Unlock click states when gestures are released/changed
        if self.active_gesture != "LEFT_CLICK" and self.active_gesture != "DRAG_MODE":
            self._left_click_locked = False
            
        if self.active_gesture != "RIGHT_CLICK":
            self._right_click_locked = False

        # -- GESTURE STATES ROUTING --
        
        if self.active_gesture == "LEFT_CLICK":
            # Cooldown + Edge-Trigger filtering to prevent multi-firing
            elapsed = current_time - self._last_event_times["LEFT_CLICK"]
            if not self._left_click_locked and elapsed >= self.config["click_cooldown"]:
                event_trigger = "CLICK_LEFT"
                self._last_event_times["LEFT_CLICK"] = current_time
                self._left_click_locked = True
                logger.info("Gesture Engine Event Triggered: CLICK_LEFT")
            # Override state display to keep tracking moving
            self.active_gesture = "MOVE_CURSOR"
 
        elif self.active_gesture == "DRAG_MODE":
            if not self.in_drag_mode:
                event_trigger = "DRAG_START"
                self.in_drag_mode = True
                self._left_click_locked = True  # Pinch hold locks clicks
                logger.info("Gesture Engine Event Triggered: DRAG_START")
 
        elif self.active_gesture == "RIGHT_CLICK":
            elapsed = current_time - self._last_event_times["RIGHT_CLICK"]
            if not self._right_click_locked and elapsed >= self.config["click_cooldown"]:
                event_trigger = "CLICK_RIGHT"
                self._last_event_times["RIGHT_CLICK"] = current_time
                self._right_click_locked = True
                logger.info("Gesture Engine Event Triggered: CLICK_RIGHT")
            self.active_gesture = "MOVE_CURSOR"

        elif self.active_gesture == "SCROLL":
            # Tracking point is the wrist (0)
            wrist_y = landmarks[0][1]
            
            if self.scroll_start_y is None:
                self.scroll_start_y = wrist_y
            else:
                # Calculate normalized vertical displacement delta
                # Note: MediaPipe coordinates increase downward, so positive delta means moving UP
                delta_y = self.scroll_start_y - wrist_y
                normalized_delta = delta_y / hand_scale
                
                # Check scroll rates
                elapsed_scroll = current_time - max(
                    self._last_event_times["SCROLL_UP"], 
                    self._last_event_times["SCROLL_DOWN"]
                )
                
                if elapsed_scroll >= self.config["scroll_cooldown"]:
                    if normalized_delta > self.config["scroll_sensitivity"]:
                        scroll_tick = 1  # SCROLL UP
                        self._last_event_times["SCROLL_UP"] = current_time
                        self.scroll_start_y = wrist_y  # Move anchor
                        logger.debug("Gesture Engine Event: SCROLL_UP")
                    elif normalized_delta < -self.config["scroll_sensitivity"]:
                        scroll_tick = -1  # SCROLL DOWN
                        self._last_event_times["SCROLL_DOWN"] = current_time
                        self.scroll_start_y = wrist_y  # Move anchor
                        logger.debug("Gesture Engine Event: SCROLL_DOWN")

        # Handle DRAG_RELEASE if they opened their hand from drag mode
        if self.in_drag_mode and self.active_gesture != "DRAG_MODE":
            event_trigger = "DRAG_RELEASE"
            self.in_drag_mode = False
            self._pinch_frame_counter = 0
            logger.info("Gesture Engine Event Triggered: DRAG_RELEASE")

        return {
            "gesture": self.active_gesture,
            "confidence": float(confidence),
            "event_trigger": event_trigger,
            "scroll_tick": scroll_tick
        }
