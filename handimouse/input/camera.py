"""
Input Layer: High-performance camera stream module.
Utilizes a dedicated reader thread for non-blocking frame retrieval,
incorporates graceful retry mechanisms, automatic device fallback, and live FPS telemetry.
"""

import cv2
import logging
import threading
import time
from typing import Optional, Tuple, Union

from handimouse.monitoring.logger import setup_logger

logger = logging.getLogger("handimouse.input.camera")


class CameraStream:
    """
    A production-grade, thread-safe camera stream reader that captures frames 
    in a dedicated background thread to prevent blocking the main pipeline.

    Performance Constraints & Optimizations:
    1. Decoupled Frame Capture: Grabbing frames with `cv2.VideoCapture.read()` is blocking
       and dependent on hardware exposure times (up to 33ms or more in low light). By running
       ingestion in a separate daemon thread, we achieve true zero-blocking frame ingestion.
    2. Zero-Lag Bounded Buffering: Instead of queuing old frames, we overwrite a single 
       buffer slot. This guarantees that the Intelligence/Vision Layers always process the 
       most immediate "photon" capture, eliminating input lag.
    3. Downsampling Performance: Processing 1080p or 4K frames in MediaPipe is extremely 
       expensive and does not yield better landmark accuracy than lower resolutions. We downsample 
       frames directly after acquisition to a lightweight resolution (e.g., 640x480 or 720p) 
       to dramatically decrease inference times.
    """

    def __init__(
        self,
        device_index: int = 0,
        target_resolution: Optional[Tuple[int, int]] = (640, 480),
        max_fallbacks: int = 3,
        reconnect_interval: float = 2.0
    ):
        """
        Initialize the CameraStream stream.

        Args:
            device_index: Primary camera device index (default 0).
            target_resolution: Optional (width, height) to resize frames to. None keeps original.
            max_fallbacks: Maximum consecutive camera indexes to attempt if the primary fails.
            reconnect_interval: Seconds to wait before attempting to reconnect a dropped stream.
        """
        self.device_index = device_index
        self.target_resolution = target_resolution
        self.max_fallbacks = max_fallbacks
        self.reconnect_interval = reconnect_interval

        # Threading and synchronization
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # OpenCV state
        self.cap: Optional[cv2.VideoCapture] = None
        self.active_device_index = device_index

        # Frame buffer
        self._latest_frame: Optional[cv2.Mat] = None
        self._new_frame_available = False

        # Performance & telemetry metrics
        self.fps = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._dropped_frames_count = 0

    def __enter__(self) -> "CameraStream":
        """Context manager entry point."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point with guaranteed resource release."""
        self.stop()

    def _initialize_capture(self, index: int) -> bool:
        """
        Attempts to initialize a VideoCapture device with the specified index.
        Applies fallback logic if the primary device cannot be loaded.
        """
        logger.info(f"Attempting to initialize camera device index {index}...")
        import platform
        if platform.system() == "Windows":
            # DirectShow is much faster to initialize on Windows than MSMF
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index)

        if not cap.isOpened():
            logger.warning(f"Failed to open camera device index {index}.")
            cap.release()
            return False

        # Attempt to configure hardware resolution properties to match target 
        # (reduces camera sensor workload at the driver level if supported)
        if self.target_resolution:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])
            # Set MJPG/H264 compression codec if supported for higher framerates
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # Verify stream is actually generating frames
        success, frame = cap.read()
        if not success or frame is None:
            logger.warning(f"Camera device index {index} opened but returned invalid frames.")
            cap.release()
            return False

        self.cap = cap
        self.active_device_index = index
        logger.info(
            f"Successfully connected to camera device index {index}. "
            f"Sensor dimensions: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
            f"{cap.get(cv2.CAP_PROP_FPS)} FPS"
        )
        return True

    def _establish_connection(self) -> bool:
        """
        Manages primary connection and hardware fallbacks if the primary device fails.
        """
        # Try primary index first
        if self._initialize_capture(self.device_index):
            return True

        # Fallback loop
        logger.info("Executing device fallback logic...")
        for offset in range(1, self.max_fallbacks + 1):
            fallback_index = self.device_index + offset
            if self._initialize_capture(fallback_index):
                logger.info(f"Fallback successful. Using camera device index {fallback_index}.")
                return True

        logger.error("All camera initialization attempts and device fallbacks failed.")
        return False

    def start(self) -> None:
        """Start the background camera frame acquisition thread."""
        with self._lock:
            if self._running:
                logger.warning("Camera stream is already running.")
                return

            self._running = True
            self._thread = threading.Thread(
                target=self._capture_loop, 
                name="CameraReader", 
                daemon=True
            )
            self._thread.start()
            logger.info("Camera Stream acquisition thread started successfully.")

    def stop(self) -> None:
        """Stop the camera stream thread and safely release hardware resources."""
        logger.info("Stopping camera stream reader thread...")
        with self._lock:
            self._running = False

        if self._thread:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning("Camera reader thread join timed out. Proceeding to force close.")
            self._thread = None

        with self._lock:
            self._release_resources()
        logger.info("Camera Stream halted and resources cleanly released.")

    def _release_resources(self) -> None:
        """Helper to release OpenCV capture handle."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self._latest_frame = None
        self._new_frame_available = False

    def _capture_loop(self) -> None:
        """
        Background execution loop. Continuously ingests frames from the camera
        sensor, calculates FPS, resizes frames, and manages auto-reconnections.
        """
        # Initial connection
        while self._running:
            if self._establish_connection():
                break
            logger.warning(f"Could not connect to any camera. Retrying in {self.reconnect_interval}s...")
            time.sleep(self.reconnect_interval)

        self._fps_start_time = time.time()
        self._frame_count = 0

        while self._running:
            if not self.cap or not self.cap.isOpened():
                logger.warning("Camera handle lost. Triggering reconnection procedure...")
                self._handle_reconnection()
                continue

            success, frame = self.cap.read()

            if not success or frame is None:
                logger.warning("Failed to grab frame from video device.")
                self._dropped_frames_count += 1
                self._handle_reconnection()
                continue

            # Performance Optimization: Early Resizing
            # Performing downsizing immediately on acquisition ensures all down-the-line threads
            # (Vision, Tracking) consume much smaller arrays, heavily accelerating pipeline processing.
            if self.target_resolution:
                frame = cv2.resize(
                    frame, 
                    self.target_resolution, 
                    interpolation=cv2.INTER_LINEAR
                )

            # Thread-safe buffer overwrite (keeps latency constant, dropping stale frames)
            with self._lock:
                self._latest_frame = frame
                self._new_frame_available = True
                self._frame_count += 1

            # Continuous telemetry tracking (FPS calculation over moving 1-second intervals)
            current_time = time.time()
            elapsed = current_time - self._fps_start_time
            if elapsed >= 1.0:
                with self._lock:
                    self.fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start_time = current_time
                logger.debug(f"Camera Stream Ingestion Metrics: {self.fps:.1f} FPS | Dropped: {self._dropped_frames_count}")

        # Final cleanup if the thread terminates
        with self._lock:
            self._release_resources()

    def _handle_reconnection(self) -> None:
        """
        Executes a safe retry reconnection protocol without crashing the main application.
        """
        with self._lock:
            self._release_resources()

        logger.warning(f"Camera connection offline. Attempting reconnect in {self.reconnect_interval} seconds...")
        
        # Non-blocking pause within the capture loop thread, check running status continuously
        wait_steps = int(self.reconnect_interval / 0.1)
        for _ in range(wait_steps):
            if not self._running:
                return
            time.sleep(0.1)

        # Attempt to reopen the camera using the last successfully active device index
        self._initialize_capture(self.active_device_index)

    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Non-blocking read interface. Retrieves the latest ingested frame.

        Returns:
            Tuple[bool, Optional[cv2.Mat]]: (Success flag, latest frame image array or None).
        """
        with self._lock:
            if not self._running:
                return False, None

            frame = self._latest_frame
            # We consumed the frame, mark it read (optional logic, usually we want continuous tracking 
            # and don't care if we process the same frame twice if FPS is high, but we track availability)
            self._new_frame_available = False
            
            # Make a light copy of the frame to prevent thread-shared array memory modifications
            return (frame is not None), (frame.copy() if frame is not None else None)

    @property
    def is_running(self) -> bool:
        """Check if background worker thread is active."""
        with self._lock:
            return self._running

    @property
    def current_fps(self) -> float:
        """Retrieve real-time camera ingestion FPS."""
        with self._lock:
            return self.fps
