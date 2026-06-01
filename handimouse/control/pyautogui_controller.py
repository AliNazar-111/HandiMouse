"""
Control Layer: Cross-platform mouse controller using PyAutoGUI.
Used as a fallback on non-Windows platforms or when Win32 APIs are unavailable.

Performance Note:
  PyAutoGUI adds a 0.1s PAUSE after each action by default. We disable this
  entirely for real-time gesture control. Without this, cursor updates lag by
  100ms per frame — completely unacceptable for a live tracking system.
"""

import logging
from typing import Tuple

import pyautogui

from handimouse.control.mouse_controller import MouseController

logger = logging.getLogger("handimouse.control.pyautogui_controller")

# Disable PyAutoGUI's built-in safety delays for real-time performance
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False  # Prevents corner-of-screen crash (we handle safety in SafetyLayer)


class PyAutoGUIController(MouseController):
    """
    Cross-platform mouse controller using PyAutoGUI.
    Adequate for development and non-Windows platforms.
    For lowest-latency on Windows, prefer Win32MouseController.
    """

    def __init__(self):
        super().__init__()
        logger.info("PyAutoGUIController ready.")

    def move_to(self, x: int, y: int) -> None:
        """
        Move cursor to absolute position.
        Uses duration=0 to skip PyAutoGUI's tweening/easing animation.
        """
        pyautogui.moveTo(x, y, duration=0)

    def click_left(self) -> None:
        """Execute a left click at the current position."""
        pyautogui.click(button="left")
        logger.debug("PyAutoGUI: LEFT_CLICK executed.")

    def click_right(self) -> None:
        """Execute a right click at the current position."""
        pyautogui.click(button="right")
        logger.debug("PyAutoGUI: RIGHT_CLICK executed.")

    def drag_start(self) -> None:
        """Press and hold left mouse button."""
        pyautogui.mouseDown(button="left")
        logger.debug("PyAutoGUI: DRAG_START (mouseDown).")

    def drag_release(self) -> None:
        """Release held left mouse button."""
        pyautogui.mouseUp(button="left")
        logger.debug("PyAutoGUI: DRAG_RELEASE (mouseUp).")

    def scroll(self, ticks: int) -> None:
        """
        Scroll vertically. PyAutoGUI scroll units are platform-dependent.
        Positive = up, negative = down.
        """
        pyautogui.scroll(ticks * 3)  # Multiply by 3 for sensible line-scroll amounts
        logger.debug(f"PyAutoGUI: SCROLL {ticks:+d} ticks.")

    def get_position(self) -> Tuple[int, int]:
        """Return current absolute cursor position."""
        return pyautogui.position()
