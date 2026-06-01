"""
Input Layer: Handles physical camera streams, frames buffer management, and ingestion threads.
"""

from .camera import CameraStream

__all__ = ["CameraStream"]
