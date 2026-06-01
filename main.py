"""
HandiMouse: Unified Real-Time Virtual Mouse System.
Main executable script wiring all architectural layers together:
  Input (CameraStream) -> Vision (HandTracker) -> Intelligence (GestureEngine)
  -> Safety (BoundsValidator, SafetyInterlock) -> Control (MouseController)
  with Configuration hot-reloads and Observability monitoring.
"""

import sys
import os
import time
import logging
import cv2
from dataclasses import asdict
from typing import Tuple, List

# Ensure handimouse package is in the import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from handimouse import (
    setup_logger,
    CameraStream,
    HandTracker,
    GestureEngine,
    MouseControllerFactory,
    BoundsValidator,
    SafetyInterlock,
    ConfigurationManager
)

# MediaPipe Hand Connections wireframe indices
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm
]


def draw_hud(
    frame: cv2.Mat,
    gesture: str,
    confidence: float,
    fps: float,
    event: str,
    screen_pos: Tuple[int, int],
    safety_active: bool,
    safety_reason: str
):
    """Draw a premium glassmorphic HUD panel with live telemetry and alarms."""
    h, w, _ = frame.shape

    # 1. Base transparent overlay bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    
    # Draw a warning banner background if safety interlock is thrown
    if safety_active:
        cv2.rectangle(overlay, (0, 80), (w, 120), (0, 0, 180), -1) # Dark red alert
        
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Borders
    cv2.line(frame, (0, 80), (w, 80), (80, 80, 80), 1, cv2.LINE_AA)
    if safety_active:
        cv2.line(frame, (0, 120), (w, 120), (0, 0, 255), 1, cv2.LINE_AA)

    # 2. Text layout & Telemetry Details
    cv2.putText(frame, "HANDIMOUSE REAL-TIME SYSTEM ENGINE", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Active State Text
    gesture_colors = {
        "MOVE_CURSOR": (0, 255, 127),  # Spring Green
        "LEFT_CLICK": (0, 165, 255),   # Deep Orange
        "RIGHT_CLICK": (255, 105, 180), # Hot Pink
        "DRAG_MODE": (30, 144, 255),   # Dodger Blue
        "SCROLL": (186, 85, 211),      # Violet
        "IDLE": (160, 160, 160)        # Gray
    }
    g_color = gesture_colors.get(gesture, (220, 220, 220))
    if safety_active:
        g_color = (0, 0, 255) # Red alert for state text too

    cv2.putText(frame, f"STATE: {gesture if not safety_active else 'SUSPENDED'}", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, g_color, 2, cv2.LINE_AA)

    # Stats Columns
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 300, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"CONFIDENCE: {confidence:.2f}", (w - 300, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Action / Position
    cv2.putText(frame, f"CURSOR ABS: {screen_pos[0]}, {screen_pos[1]}", (w - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    
    if event and not safety_active:
        cv2.putText(frame, f"EVENT: {event}", (w - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # 3. Draw emergency banner details
    if safety_active:
        cv2.putText(frame, "!!! EMERGENCY SAFETY OVERRIDE ACTIVE !!!", (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        reason_desc = safety_reason.upper()
        cv2.putText(frame, f"CAUSE: {reason_desc} | PRESS 'R' TO RESET SYSTEM NORMAL", (w - 380, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_skeleton(frame: cv2.Mat, landmarks: List[Tuple[float, float, float]]):
    """Draw wireframe hand skeletons with stylized joints."""
    h, w, _ = frame.shape
    coords = []
    for lm in landmarks:
        cx, cy = int(lm[0] * w), int(lm[1] * h)
        coords.append((cx, cy))

    # Connect joints
    for connection in HAND_CONNECTIONS:
        s_idx, e_idx = connection
        if s_idx < len(coords) and e_idx < len(coords):
            cv2.line(frame, coords[s_idx], coords[e_idx], (0, 255, 127), 2, cv2.LINE_AA)

    # Nodes drawing
    for i, pt in enumerate(coords):
        if i == 8:  # Index tip pointer
            cv2.circle(frame, pt, 7, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 11, (255, 255, 0), 1, cv2.LINE_AA)
        elif i == 4:  # Thumb tip pinch node
            cv2.circle(frame, pt, 6, (0, 230, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, pt, 4, (0, 80, 255), -1, cv2.LINE_AA)


def update_running_parameters(
    config_manager: ConfigurationManager,
    gesture_engine: GestureEngine,
    bounds_validator: BoundsValidator,
    safety_interlock: SafetyInterlock
):
    """Updates parameters of actively running layers dynamically on hot-reload."""
    settings = config_manager.settings

    # 1. Update gesture thresholds
    gesture_engine.config.update(asdict(settings.gesture_engine))
    logger = logging.getLogger("handimouse")
    logger.info("[HOT-RELOAD] Updated GestureEngine thresholds.")

    # 2. Update bounds validator configs and re-calculate jump pixel limit
    bounds_validator.max_jump_fraction = settings.safety.max_jump_fraction
    bounds_validator.max_rejected_frames = settings.safety.max_rejected_frames
    import math
    bounds_validator.jump_threshold_pixels = math.sqrt(
        (bounds_validator.screen_width * bounds_validator.max_jump_fraction) ** 2 +
        (bounds_validator.screen_height * bounds_validator.max_jump_fraction) ** 2
    )
    logger.info("[HOT-RELOAD] Updated BoundsValidator spatial limits.")

    # 3. Update safety interlock limits
    safety_interlock.fist_frames_threshold = settings.safety.fist_frames_threshold
    logger.info("[HOT-RELOAD] Updated SafetyInterlock timers.")


def main():
    # Setup structured logger
    logger = setup_logger(name="handimouse", level=logging.INFO)
    logger.info("=" * 65)
    logger.info("INITIALIZING HANDIMOUSE VIRTUAL CONTROL SYSTEM")
    logger.info("=" * 65)

    # 1. Initialize Configuration Layer
    config_manager = ConfigurationManager()
    settings = config_manager.settings

    camera = None
    tracker = None

    try:
        # 2. Initialize Control Layer (OS native Mouse Controller)
        controller = MouseControllerFactory.create()

        # 3. Initialize Safety Layer
        bounds_validator = BoundsValidator(
            max_jump_fraction=settings.safety.max_jump_fraction,
            max_rejected_frames=settings.safety.max_rejected_frames
        )
        safety_interlock = SafetyInterlock(
            fist_frames_threshold=settings.safety.fist_frames_threshold
        )

        # 4. Initialize Intelligence Layer
        gesture_engine = GestureEngine(
            config=asdict(settings.gesture_engine)
        )

        # 5. Initialize Input Layer (Camera stream thread)
        camera = CameraStream(
            device_index=settings.camera.device_index,
            target_resolution=(settings.camera.target_width, settings.camera.target_height)
        )

        # 6. Initialize Vision Layer (Hand Landmarker Model)
        tracker = HandTracker(
            min_detection_confidence=settings.tracker.min_detection_confidence,
            min_tracking_confidence=settings.tracker.min_tracking_confidence,
            dominant_hand_preference=settings.tracker.dominant_hand_preference
        )

        # Start Camera frame ingestion
        camera.start()
        logger.info("Ingesting camera thread. Stabilizing feed connection...")

        # Connection warmup
        warmup_start = time.time()
        feed_ready = False
        while time.time() - warmup_start < 10.0:
            success, frame = camera.read()
            if success and frame is not None:
                feed_ready = True
                break
            time.sleep(0.1)

        if not feed_ready:
            logger.critical("Failed to stabilize camera feed connection! Exiting.")
            return

        logger.info("Camera online! System fully initialized.")
        logger.info("=" * 65)
        logger.info("   HANDIMOUSE RUNNING IN BACKGROUND (REAL-TIME MODE)")
        logger.info("=" * 65)
        logger.info("GESTURE CONTROLS SHEET:")
        logger.info("  - Cursor Movement   -> Extend Index finger only")
        logger.info("  - Left Click        -> Briefly pinch Thumb and Index Tip")
        logger.info("  - Click & Drag      -> Sustained pinch of Thumb and Index Tip")
        logger.info("  - Right Click       -> Extend Index and Middle fingers together")
        logger.info("  - Scroll Up/Down    -> Extend 5 fingers & move hand vertically")
        logger.info("  - Emergency Freeze  -> Clench hand in a tight Fist for 0.5s")
        logger.info("  - System Exit       -> Press Keyboard 'ESC' key or close feed window")
        logger.info("=" * 65)

        # Benchmarking / FPS Telemetry variables
        fps_clock = time.time()
        frame_counter = 0
        current_fps = 0.0

        cv2.namedWindow("HandiMouse - Active Stream", cv2.WINDOW_AUTOSIZE)

        while True:
            # Check for config.json modifications on disk once per second
            if config_manager.check_for_updates():
                # Propagate settings updates dynamically
                update_running_parameters(
                    config_manager,
                    gesture_engine,
                    bounds_validator,
                    safety_interlock
                )

            # Read latest camera frame
            success, frame = camera.read()
            if not success or frame is None:
                time.sleep(0.005)
                continue

            frame_counter += 1
            if frame_counter % 30 == 0:
                now = time.time()
                current_fps = 30.0 / (now - fps_clock)
                fps_clock = now

            # 1. Vision Layer processing
            tracking_result = tracker.process_frame(frame)

            # 2. Safety Interlock checks (monitors fist gesture & Esc key)
            is_suspended = safety_interlock.process_safety(tracking_result)

            # 3. Intelligence Layer processing
            gesture_result = gesture_engine.process_state(tracking_result)

            active_gesture = gesture_result["gesture"]
            confidence = gesture_result["confidence"]
            event_trigger = gesture_result["event_trigger"]
            scroll_tick = gesture_result["scroll_tick"]

            # Visual skeleton wiring
            if tracking_result["hand_detected"]:
                draw_skeleton(frame, tracking_result["landmarks"])

            # Current actual pointer position
            screen_pos = controller.get_position()

            if is_suspended:
                # Emergency state: safely release mouse clicks/drags immediately to prevent drag-locks
                controller.drag_release()
                bounds_validator.reset()
            else:
                # 4. Control & Gating Processing (Normal Executing)
                if tracking_result["hand_detected"]:
                    # Determine pointer target
                    landmarks = tracking_result["landmarks"]
                    index_tip = landmarks[8] # Index tip landmark
                    
                    # Convert normalized coordinate spacing to screen coordinates
                    screen_x, screen_y = controller.landmark_to_screen(
                        norm_x=index_tip[0],
                        norm_y=index_tip[1],
                        frame_margin=config_manager.settings.camera.frame_margin
                    )

                    # Pass target through Safety Bounds Validator (gating + clamping)
                    valid_x, valid_y = bounds_validator.validate_coordinates(screen_x, screen_y)

                    # Move cursor if state requests movement
                    if active_gesture in ("MOVE_CURSOR", "DRAG_MODE"):
                        controller.move_to(valid_x, valid_y)
                        screen_pos = (valid_x, valid_y)

                    # Fire click/drag events
                    if event_trigger == "CLICK_LEFT":
                        controller.click_left()
                    elif event_trigger == "CLICK_RIGHT":
                        controller.click_right()
                    elif event_trigger == "DRAG_START":
                        controller.drag_start()
                    elif event_trigger == "DRAG_RELEASE":
                        controller.drag_release()

                    # Fire vertical scrolling
                    if scroll_tick != 0:
                        controller.scroll(scroll_tick)
                else:
                    # Reset filters & release drag if hand drops out completely
                    bounds_validator.reset()
                    controller.drag_release()

            # Render Hud panels and live Telemetry
            draw_hud(
                frame=frame,
                gesture=active_gesture,
                confidence=confidence,
                fps=current_fps,
                event=event_trigger,
                screen_pos=screen_pos,
                safety_active=is_suspended,
                safety_reason=safety_interlock.reason
            )

            # Display feed
            cv2.imshow("HandiMouse - Active Stream", frame)

            # Wait key hooks
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc key exits
                logger.info("Closing HandiMouse via camera exit hook.")
                break
            elif key == ord('r') or key == ord('R'):
                # Reset safety lock
                if safety_interlock.interlocked:
                    safety_interlock.reset()
                    bounds_validator.reset()

        logger.info("Safely terminating unified processing loop.")

    except Exception as e:
        logger.critical(f"System experienced fatal unhandled crash: {e}", exc_info=True)

    finally:
        # Tear down all resources cleanly
        cv2.destroyAllWindows()
        if camera:
            try:
                camera.stop()
            except Exception:
                pass
        if tracker:
            try:
                tracker.close()
            except Exception:
                pass
        
        # Release physical click overrides
        try:
            # In case exception happened during drag
            import pyautogui
            pyautogui.mouseUp(button="left")
        except Exception:
            pass
            
        logger.info("All layers stopped safely. Emulation hardware and camera streams released.")
        logger.info("=" * 65)
        logger.info("   HANDIMOUSE SYSTEM ENGINE SHUT DOWN SUCCESSFULLY")
        logger.info("=" * 65)


if __name__ == "__main__":
    main()
