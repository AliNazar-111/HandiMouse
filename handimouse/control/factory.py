"""
Control Layer: Factory for instantiating the optimal MouseController implementation.
Provides Windows-native SendInput-based controller by default on Windows for ultra-low
latency, with cross-platform PyAutoGUI as fallback.
"""

import sys
import logging
from handimouse.control.mouse_controller import MouseController

logger = logging.getLogger("handimouse.control.factory")


class MouseControllerFactory:
    """
    Factory for producing MouseController instances based on operating system and environment.
    """

    @staticmethod
    def create() -> MouseController:
        """
        Create and return the most optimized MouseController available.

        On Windows (win32 platform), it attempts to instantiate the high-performance
        Win32MouseController which utilizes direct user32.dll SendInput calls (bypassing
        high Python call overheads). If that fails or if the OS is not Windows, it falls back
        to PyAutoGUIController.

        Returns:
            MouseController: An instantiated concrete subclass of MouseController.
        """
        if sys.platform == "win32":
            try:
                from handimouse.control.win32_controller import Win32MouseController
                logger.info("Initializing Windows-native Win32MouseController (ultra-low latency mode).")
                return Win32MouseController()
            except Exception as e:
                logger.warning(
                    f"Failed to initialize Win32MouseController ({e}). "
                    "Falling back to PyAutoGUIController."
                )

        try:
            from handimouse.control.pyautogui_controller import PyAutoGUIController
            logger.info("Initializing cross-platform PyAutoGUIController.")
            return PyAutoGUIController()
        except Exception as e:
            logger.critical(f"Failed to initialize PyAutoGUIController: {e}")
            raise e
