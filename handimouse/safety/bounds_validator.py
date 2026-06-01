"""
Safety Layer: Bounds and Velocity Validator.
Provides screen space boundary clamping and erratic cursor jump gating to filter out
tracking noise, occlusion glitches, and camera-reentry spikes.
"""

import logging
import math
from typing import Tuple

logger = logging.getLogger("handimouse.safety.bounds_validator")


class BoundsValidator:
    """
    Validates proposed mouse actions to ensure safety and smoothness.
    
    Checks:
    1. Screen boundary limits (clamping).
    2. Velocity/Jump gating: Detects anomalous spatial acceleration (e.g., coordinate
       jumps greater than 30% of screen size in a single frame), filtering out tracking glitches.
    3. Dead-reckoning recovery: Allows genuine high-speed movements after a short gating
       timeout (e.g., 3 frames) to prevent the cursor from getting permanently stuck.
    """

    def __init__(
        self,
        max_jump_fraction: float = 0.30,
        max_rejected_frames: int = 3
    ):
        """
        Initialize the Bounds and Velocity Validator.

        Args:
            max_jump_fraction: Maximum percentage of screen size allowed for cursor movement
                               between consecutive frames before it is classified as noise.
            max_rejected_frames: Number of consecutive frames we are allowed to ignore anomalies
                                 before we force-update (re-anchor) to prevent getting stuck.
        """
        import pyautogui
        # Cache screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        self.max_jump_fraction = max_jump_fraction
        self.max_rejected_frames = max_rejected_frames

        # Spatial gating threshold in absolute pixels (using Euclidean distance)
        self.jump_threshold_pixels = math.sqrt(
            (self.screen_width * self.max_jump_fraction) ** 2 +
            (self.screen_height * self.max_jump_fraction) ** 2
        )

        # Track historical coordinate states for delta checks
        self.last_valid_x = None
        self.last_valid_y = None
        self.rejected_frames_counter = 0

        logger.info(
            f"BoundsValidator initialized | Screen: {self.screen_width}x{self.screen_height} | "
            f"Gating Jump limit: {max_jump_fraction * 100:.1f}% ({self.jump_threshold_pixels:.1f} px) | "
            f"Recovery frames: {max_rejected_frames}"
        )

    def validate_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """
        Validates proposed coordinates against physical screen boundaries and noise filters.

        Args:
            x: Target absolute x screen coordinate.
            y: Target absolute y screen coordinate.

        Returns:
            Tuple[int, int]: Validated, filtered, and clamped screen coordinates (x, y).
        """
        # 1. Physical Screen Boundary Clamping
        clamped_x = max(0, min(self.screen_width - 1, x))
        clamped_y = max(0, min(self.screen_height - 1, y))

        # First frame or re-entry anchor condition
        if self.last_valid_x is None or self.last_valid_y is None:
            self.last_valid_x = clamped_x
            self.last_valid_y = clamped_y
            self.rejected_frames_counter = 0
            return clamped_x, clamped_y

        # 2. Gating / Anomalous Jump Protection
        # Calculate spatial Euclidean distance from last valid location
        delta_x = clamped_x - self.last_valid_x
        delta_y = clamped_y - self.last_valid_y
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)

        # If the displacement is abnormally large (likely a tracking jump or occlusion glitch)
        if distance > self.jump_threshold_pixels:
            self.rejected_frames_counter += 1
            
            # If we haven't hit the recovery limit, ignore the jump and return last valid state
            if self.rejected_frames_counter <= self.max_rejected_frames:
                logger.warning(
                    f"[GATING] Ignored erratic cursor jump! Distance: {distance:.1f} px | "
                    f"Attempted: ({clamped_x}, {clamped_y}) | Returning last valid: ({self.last_valid_x}, {self.last_valid_y}) | "
                    f"Anomaly count: {self.rejected_frames_counter}/{self.max_rejected_frames}"
                )
                return self.last_valid_x, self.last_valid_y
            else:
                # Force update/re-anchor if they sustain this position (e.g. quick hand re-entry)
                logger.info(
                    f"[GATING RECOVERY] Recovery limit reached. Re-anchoring cursor to: ({clamped_x}, {clamped_y})"
                )
                self.rejected_frames_counter = 0

        # Accepted valid movement - update anchors
        self.rejected_frames_counter = 0
        self.last_valid_x = clamped_x
        self.last_valid_y = clamped_y

        return clamped_x, clamped_y

    def reset(self) -> None:
        """Reset historical positions. Use when hand tracking drops out."""
        self.last_valid_x = None
        self.last_valid_y = None
        self.rejected_frames_counter = 0
        logger.debug("BoundsValidator state reset.")

