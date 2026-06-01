"""
Control Layer: Abstract base class defining the OS mouse control interface contract.
All concrete controller implementations must fulfill this protocol.
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple

logger = logging.getLogger("handimouse.control.mouse_controller")


class MouseController(ABC):
    """
    Abstract base class for OS mouse interaction.
    Defines the interface all platform-specific controllers must implement.
    """

    def __init__(self):
        import pyautogui
        # Cache screen dimensions once at startup to avoid repeated OS syscalls per frame
        self.screen_width, self.screen_height = pyautogui.size()
        logger.info(f"MouseController initialized | Screen: {self.screen_width}x{self.screen_height}")

    def landmark_to_screen(
        self,
        norm_x: float,
        norm_y: float,
        frame_margin: float = 0.1
    ) -> Tuple[int, int]:
        """
        Convert normalized landmark coordinates (0.0-1.0) to absolute screen pixel coordinates.
        Applies a configurable inset margin to create a comfortable usable zone within the camera FOV.

        Args:
            norm_x: Normalized X coordinate from the Vision Layer (0.0 to 1.0).
            norm_y: Normalized Y coordinate from the Vision Layer (0.0 to 1.0).
            frame_margin: Percentage to inset from camera FOV edges as a dead zone.

        Returns:
            Tuple[int, int]: (screen_x, screen_y) absolute pixel coordinates clamped to screen bounds.
        """
        # The MediaPipe horizontal axis is mirrored relative to screen, correct by flipping
        norm_x = 1.0 - norm_x

        # Remap from the active zone (margin..1-margin) to full screen space (0..1)
        usable_x = (norm_x - frame_margin) / (1.0 - 2 * frame_margin)
        usable_y = (norm_y - frame_margin) / (1.0 - 2 * frame_margin)

        # Scale to screen dimensions and clamp to valid pixel range
        screen_x = int(usable_x * self.screen_width)
        screen_y = int(usable_y * self.screen_height)
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))

        return screen_x, screen_y

    @abstractmethod
    def move_to(self, x: int, y: int) -> None:
        """Move cursor to absolute screen coordinates (x, y)."""
        ...

    @abstractmethod
    def click_left(self) -> None:
        """Execute a left mouse button click at the current cursor position."""
        ...

    @abstractmethod
    def double_click(self) -> None:
        """Execute a double left-click at the current cursor position."""
        ...

    @abstractmethod
    def click_right(self) -> None:
        """Execute a right mouse button click at the current cursor position."""
        ...

    @abstractmethod
    def drag_start(self) -> None:
        """Press and hold the left mouse button to initiate a drag operation."""
        ...

    @abstractmethod
    def drag_release(self) -> None:
        """Release the left mouse button to complete a drag operation."""
        ...

    @abstractmethod
    def scroll(self, ticks: int) -> None:
        """
        Perform a vertical scroll action.

        Args:
            ticks: Positive ticks scroll up, negative ticks scroll down.
        """
        ...

    @abstractmethod
    def get_position(self) -> Tuple[int, int]:
        """Return current OS cursor position as (x, y) absolute screen coordinates."""
        ...
