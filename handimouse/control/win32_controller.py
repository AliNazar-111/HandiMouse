"""
Control Layer: Windows-native ultra-low latency mouse controller.
Directly calls Win32 user32.dll SendInput API via ctypes, bypassing
PyAutoGUI's Python overhead entirely.

Performance:
  PyAutoGUI moveTo()  -> ~100ms per call (Python → subprocess → OS)
  Win32 SendInput()   -> ~0.1ms per call (Python → ctypes → OS direct)
"""

import ctypes
import ctypes.wintypes
import logging
from typing import Tuple

import pyautogui

from handimouse.control.mouse_controller import MouseController

logger = logging.getLogger("handimouse.control.win32_controller")

# Win32 constants for SendInput
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE        = 0x0001
MOUSEEVENTF_LEFTDOWN    = 0x0002
MOUSEEVENTF_LEFTUP      = 0x0004
MOUSEEVENTF_RIGHTDOWN   = 0x0008
MOUSEEVENTF_RIGHTUP     = 0x0010
MOUSEEVENTF_WHEEL       = 0x0800
MOUSEEVENTF_ABSOLUTE    = 0x8000
WHEEL_DELTA             = 120


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx",          ctypes.wintypes.LONG),
        ("dy",          ctypes.wintypes.LONG),
        ("mouseData",   ctypes.wintypes.DWORD),
        ("dwFlags",     ctypes.wintypes.DWORD),
        ("time",        ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("_input",)
    _fields_ = [
        ("type",   ctypes.wintypes.DWORD),
        ("_input", _INPUT),
    ]


class Win32MouseController(MouseController):
    """
    Windows-native mouse controller using the SendInput Win32 API.
    Provides the absolute minimum latency path from gesture event to OS cursor update.
    """

    def __init__(self):
        super().__init__()
        self._user32 = ctypes.windll.user32
        self._is_dragging = False
        logger.info("Win32MouseController ready (SendInput path active).")

    def _send_input(self, flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> None:
        """Low-level helper: build and dispatch a single MOUSEINPUT event."""
        inp = INPUT(
            type=INPUT_MOUSE,
            mi=MOUSEINPUT(
                dx=dx,
                dy=dy,
                mouseData=data,
                dwFlags=flags,
                time=0,
                dwExtraInfo=None
            )
        )
        self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def move_to(self, x: int, y: int) -> None:
        """
        Move cursor using absolute MOUSEEVENTF_ABSOLUTE coordinates.
        Win32 absolute coordinates are scaled to a 0-65535 virtual grid regardless of screen resolution.
        """
        # Normalize to 0-65535 range (Win32 absolute coordinate system)
        abs_x = int(x * 65535 / (self.screen_width - 1))
        abs_y = int(y * 65535 / (self.screen_height - 1))
        self._send_input(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, dx=abs_x, dy=abs_y)

    def click_left(self) -> None:
        """Fire a left mouse button down+up with a 15ms hold gap to guarantee OS registration."""
        import time
        self._send_input(MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.015)
        self._send_input(MOUSEEVENTF_LEFTUP)
        logger.debug("Win32: LEFT_CLICK fired.")

    def double_click(self) -> None:
        """Fire two left clicks with proper hold time and a 50ms interval between them."""
        import time
        self.click_left()
        time.sleep(0.05)
        self.click_left()
        logger.debug("Win32: DOUBLE_CLICK fired.")

    def click_right(self) -> None:
        """Fire a right mouse button down+up with a 15ms hold gap."""
        import time
        self._send_input(MOUSEEVENTF_RIGHTDOWN)
        time.sleep(0.015)
        self._send_input(MOUSEEVENTF_RIGHTUP)
        logger.debug("Win32: RIGHT_CLICK fired.")

    def drag_start(self) -> None:
        """Press and hold left mouse button for drag."""
        if not self._is_dragging:
            self._send_input(MOUSEEVENTF_LEFTDOWN)
            self._is_dragging = True
            logger.debug("Win32: DRAG_START (LEFTDOWN held).")

    def drag_release(self) -> None:
        """Release held left mouse button."""
        if self._is_dragging:
            self._send_input(MOUSEEVENTF_LEFTUP)
            self._is_dragging = False
            logger.debug("Win32: DRAG_RELEASE (LEFTUP).")

    def scroll(self, ticks: int) -> None:
        """
        Scroll wheel via MOUSEEVENTF_WHEEL.
        WHEEL_DELTA=120 represents one standard scroll detent.
        Positive = scroll up, negative = scroll down.
        """
        scroll_amount = ticks * WHEEL_DELTA
        self._send_input(MOUSEEVENTF_WHEEL, data=scroll_amount)
        logger.debug(f"Win32: SCROLL {ticks:+d} ticks ({scroll_amount} wheel delta).")

    def get_position(self) -> Tuple[int, int]:
        """Return current cursor position via GetCursorPos."""
        pt = ctypes.wintypes.POINT()
        self._user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y
