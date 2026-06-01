"""
HandiMouse: A real-time computer vision-based virtual mouse control system.
"""

from handimouse.input import CameraStream
from handimouse.vision import HandTracker, OneEuroFilter
from handimouse.intelligence import GestureEngine
from handimouse.control import MouseController, MouseControllerFactory
from handimouse.safety import BoundsValidator, SafetyInterlock
from handimouse.config import AppSettings, ConfigurationManager
from handimouse.monitoring import setup_logger

__version__ = "0.1.0"

__all__ = [
    "CameraStream",
    "HandTracker",
    "OneEuroFilter",
    "GestureEngine",
    "MouseController",
    "MouseControllerFactory",
    "BoundsValidator",
    "SafetyInterlock",
    "AppSettings",
    "ConfigurationManager",
    "setup_logger"
]

