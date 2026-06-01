"""
Control Layer: OS-level mouse action execution.
Provides platform-adaptive controllers for cursor movement, clicking, dragging, and scrolling.
"""

from .mouse_controller import MouseController
from .factory import MouseControllerFactory

__all__ = ["MouseController", "MouseControllerFactory"]
