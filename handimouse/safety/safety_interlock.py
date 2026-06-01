"""
Safety Layer: Safety Interlock System.
Provides global emergency hotkey detection and spatial gesture interlocks (e.g., fist hold)
to instantly suspend/abort mouse controls to prevent system freezes or runaway loops.
"""

import sys
import time
import logging
from typing import Dict, Any

logger = logging.getLogger("handimouse.safety.safety_interlock")

# Windows Virtual Key Codes
VK_ESCAPE = 0x1B


class SafetyInterlock:
    """
    Emergency kill-switch and control suspension manager.
    
    Triggers:
    1. Global hotkey interlock: Intercepts physical 'Esc' key globally on Windows
       using high-speed, direct windll ctypes calls (zero dependencies, zero latency).
    2. Emergency fist gesture: Activates if the user maintains a tight fist
       (all 5 fingers folded: fingers_up == [0, 0, 0, 0, 0]) for a sustained timeframe.
    """

    def __init__(self, fist_frames_threshold: int = 60, enable_fist_safety: bool = True):
        """
        Initialize Safety Interlock.

        Args:
            fist_frames_threshold: Number of consecutive frames the fist must be held
                                   to trigger the safety interlock.
            enable_fist_safety: If True, allows emergency fist gestures to trigger suspension.
        """
        self.fist_frames_threshold = fist_frames_threshold
        self.enable_fist_safety = enable_fist_safety
        
        # Interlock states
        self.interlocked = False
        self.reason = ""

        # Fist tracking variables
        self._consecutive_fist_frames = 0

        # Attempt to load ctypes for Windows native high-speed key polling
        self._is_windows = sys.platform == "win32"
        self._user32 = None
        if self._is_windows:
            try:
                import ctypes
                self._user32 = ctypes.windll.user32
                logger.info("Windows detected. High-speed global Esc key interlock activated.")
            except Exception as e:
                logger.warning(f"Could not load Windows user32.dll for global key hook: {e}")

        logger.info(
            f"SafetyInterlock ready | Fist threshold: {fist_frames_threshold} frames | Enabled: {enable_fist_safety}"
        )

    def is_esc_pressed_globally(self) -> bool:
        """
        Checks if the ESC key is physically pressed on the system right now.
        Uses direct Windows ctypes API for sub-microsecond non-blocking evaluation.
        """
        if self._is_windows and self._user32:
            # GetAsyncKeyState checks the state of the key at the moment of the call
            # Most significant bit is set if the key is down
            return bool(self._user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)
        return False

    def process_safety(self, tracking_result: Dict[str, Any], suppress_fist: bool = False) -> bool:
        """
        Processes frame tracking telemetry to check for safety locks or hotkey triggers.

        Args:
            tracking_result: Output result from HandTracker.process_frame.
            suppress_fist: When True (e.g. a pinch gesture is active), skip fist
                           frame counting to avoid false emergency triggers.

        Returns:
            bool: True if the safety interlock is actively triggered/locked, False otherwise.
        """
        # 1. Global Keyboard Check
        if self.is_esc_pressed_globally():
            if not self.interlocked:
                self.interlocked = True
                self.reason = "Global Keyboard ESC Key pressed"
                logger.critical(f"[SAFETY ALARM] {self.reason}! Mouse control suspended.")
            return True

        # 2. Gesture Interlock Check
        if self.enable_fist_safety and tracking_result.get("hand_detected", False):
            fingers_up = tracking_result.get("fingers_up", [0, 0, 0, 0, 0])
            
            # Check if all fingers are folded down (Tight Fist)
            if fingers_up == [0, 0, 0, 0, 0] and not suppress_fist:
                self._consecutive_fist_frames += 1
                
                # Check if held for long enough to declare emergency
                if self._consecutive_fist_frames >= self.fist_frames_threshold:
                    if not self.interlocked:
                        self.interlocked = True
                        self.reason = f"Emergency Fist Gesture held for {self.fist_frames_threshold} frames"
                        logger.critical(f"[SAFETY ALARM] {self.reason}! Mouse control suspended.")
                    return True
            else:
                # Reset consecutive counter if fist is opened
                self._consecutive_fist_frames = 0
                
                # Auto-recover from fist emergency stop if they open their hand
                if self.interlocked and "Emergency Fist" in self.reason:
                    logger.info("Fist opened. Automatically clearing safety interlock.")
                    self.interlocked = False
                    self.reason = ""
        else:
            # If tracking drops, decay the fist timer to avoid accidental triggers
            self._consecutive_fist_frames = max(0, self._consecutive_fist_frames - 1)
            
            # Auto-recover if the hand leaves the frame completely during a fist override
            if self.interlocked and "Emergency Fist" in self.reason:
                logger.info("Hand tracking lost. Automatically clearing safety interlock.")
                self.interlocked = False
                self.reason = ""

        return self.interlocked


    def reset(self) -> None:
        """Clears the safety interlock lock, returning the system to normal emulating."""
        if self.interlocked:
            logger.info(f"Clearing safety interlock (Previous trigger: {self.reason}). Returning to normal.")
            self.interlocked = False
            self.reason = ""
            self._consecutive_fist_frames = 0
