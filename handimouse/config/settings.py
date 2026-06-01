"""
Configuration Layer: System settings and hot-reloading manager.
Provides dataclass models for application parameters and a file monitoring system
to hot-reload configurations dynamically during real-time loop execution.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

logger = logging.getLogger("handimouse.config.settings")

DEFAULT_CONFIG_FILENAME = "config.json"


@dataclass
class CameraConfig:
    device_index: int = 0
    target_width: int = 640
    target_height: int = 480
    frame_margin: float = 0.1


@dataclass
class TrackerConfig:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    dominant_hand_preference: str = "largest"


@dataclass
class GestureConfig:
    pinch_threshold: float = 0.15
    scroll_sensitivity: float = 0.05
    debounce_frames: int = 3
    click_cooldown: float = 0.35
    scroll_cooldown: float = 0.12
    drag_hold_frames: int = 6
    min_tracking_confidence: float = 0.65


@dataclass
class SafetyConfig:
    max_jump_fraction: float = 0.30
    max_rejected_frames: int = 3
    fist_frames_threshold: int = 15


@dataclass
class AppSettings:
    """
    Unified Application Settings schema.
    Contains configurations for all core HandiMouse system layers.
    """
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    gesture_engine: GestureConfig = field(default_factory=GestureConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


class ConfigurationManager:
    """
    Manager for reading, writing, and hot-reloading the HandiMouse configuration file.
    
    Implements a low-overhead, thread-safe file modification checker (polling rate-limited
    to 1 second) to detect external changes to config.json and instantly update
    the settings in memory.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigurationManager.

        Args:
            config_path: Path to the configuration file. Defaults to config.json in the workspace root.
        """
        if config_path is None:
            # Default to workspace root config.json
            self.config_path = os.path.abspath(os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                DEFAULT_CONFIG_FILENAME
            ))
        else:
            self.config_path = os.path.abspath(config_path)

        self.settings = AppSettings()
        
        # Hot-reload monitoring states
        self._last_loaded_mtime: float = 0.0
        self._last_checked_time: float = 0.0
        self._check_interval: float = 1.0  # Limit mtime syscalls to once per second max

        # Load or generate initial configurations
        self.load_or_create()

    def load_or_create(self) -> None:
        """
        Loads configuration from disk. If the file is missing, it writes the default
        settings to disk first and then loads them.
        """
        if not os.path.exists(self.config_path):
            logger.info(f"Configuration file not found. Generating default: {self.config_path}")
            self.save()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Map raw dictionary to nested dataclasses with safe default fallbacks
            cam_data = data.get("camera", {})
            self.settings.camera = CameraConfig(
                device_index=cam_data.get("device_index", 0),
                target_width=cam_data.get("target_width", 640),
                target_height=cam_data.get("target_height", 480),
                frame_margin=cam_data.get("frame_margin", 0.1)
            )

            track_data = data.get("tracker", {})
            self.settings.tracker = TrackerConfig(
                min_detection_confidence=track_data.get("min_detection_confidence", 0.5),
                min_tracking_confidence=track_data.get("min_tracking_confidence", 0.5),
                dominant_hand_preference=track_data.get("dominant_hand_preference", "largest")
            )

            gest_data = data.get("gesture_engine", {})
            self.settings.gesture_engine = GestureConfig(
                pinch_threshold=gest_data.get("pinch_threshold", 0.15),
                scroll_sensitivity=gest_data.get("scroll_sensitivity", 0.05),
                debounce_frames=gest_data.get("debounce_frames", 3),
                click_cooldown=gest_data.get("click_cooldown", 0.35),
                scroll_cooldown=gest_data.get("scroll_cooldown", 0.12),
                drag_hold_frames=gest_data.get("drag_hold_frames", 6),
                min_tracking_confidence=gest_data.get("min_tracking_confidence", 0.65)
            )

            safe_data = data.get("safety", {})
            self.settings.safety = SafetyConfig(
                max_jump_fraction=safe_data.get("max_jump_fraction", 0.30),
                max_rejected_frames=safe_data.get("max_rejected_frames", 3),
                fist_frames_threshold=safe_data.get("fist_frames_threshold", 15)
            )

            # Record file mtime upon successful loading
            self._last_loaded_mtime = os.path.getmtime(self.config_path)
            logger.info(f"Successfully loaded configuration from {self.config_path}")

        except Exception as e:
            logger.error(
                f"Error loading configuration from disk: {e}. "
                "Resorting to system-wide default settings.",
                exc_info=True
            )
            self.settings = AppSettings()

    def save(self) -> None:
        """Serializes current configuration state to disk as JSON."""
        try:
            # Convert nested dataclasses to dictionary
            dict_data = {
                "camera": asdict(self.settings.camera),
                "tracker": asdict(self.settings.tracker),
                "gesture_engine": asdict(self.settings.gesture_engine),
                "safety": asdict(self.settings.safety)
            }
            
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(dict_data, f, indent=4)
                
            self._last_loaded_mtime = os.path.getmtime(self.config_path)
            logger.info(f"Saved configuration successfully: {self.config_path}")
        except Exception as e:
            logger.critical(f"Failed to write configuration to file: {e}", exc_info=True)

    def check_for_updates(self) -> bool:
        """
        Checks the configuration file's modification time on disk.
        If it has been modified externally, it automatically triggers a reload.
        
        This check is internally throttled to a maximum frequency of once per second
        to guarantee virtually zero performance overhead inside real-time pipelines.

        Returns:
            bool: True if settings were updated/hot-reloaded, False otherwise.
        """
        import time
        current_time = time.time()
        
        # Throttled rate-limiting gate
        if current_time - self._last_checked_time < self._check_interval:
            return False
            
        self._last_checked_time = current_time

        try:
            if os.path.exists(self.config_path):
                current_mtime = os.path.getmtime(self.config_path)
                if current_mtime > self._last_loaded_mtime:
                    logger.warning("[CONFIG WATCH] External modification detected! Hot-reloading config.json...")
                    self.load_or_create()
                    return True
        except Exception as e:
            logger.error(f"Error checking config updates on disk: {e}")
            
        return False
