import sys
import os
import time
import cv2
from typing import List, Tuple

# Dynamically append workspace path to Python's system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handimouse.monitoring.logger import setup_logger
from handimouse.input.camera import CameraStream
from handimouse.vision.tracker import HandTracker
from handimouse.intelligence.gesture_engine import GestureEngine
from handimouse.control.factory import MouseControllerFactory

# MediaPipe Hand Connections for drawing the skeleton overlay
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm
]


def draw_hud(frame: cv2.Mat, gesture: str, confidence: float, fps: float, event: str, screen_pos: Tuple[int, int]):
    """Draw a premium glassmorphic telemetry overlay on the frame."""
    h, w, _ = frame.shape

    # 1. Draw top transparent HUD bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Clean borders
    cv2.line(frame, (0, 75), (w, 75), (80, 80, 80), 1, cv2.LINE_AA)

    # 2. Text layout & status indications
    cv2.putText(frame, "HANDIMOUSE E2E PIPELINE CONTROL PANEL", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Styled active gesture state text
    gesture_colors = {
        "MOVE_CURSOR": (0, 255, 127),  # Spring Green
        "LEFT_CLICK": (0, 165, 255),   # Bright Orange/Blue
        "RIGHT_CLICK": (255, 105, 180), # Hot Pink
        "DRAG_MODE": (30, 144, 255),   # Dodger Blue
        "SCROLL": (186, 85, 211),      # Medium Orchid
        "IDLE": (160, 160, 160)        # Standard Gray
    }
    g_color = gesture_colors.get(gesture, (220, 220, 220))

    cv2.putText(frame, f"STATE: {gesture}", (15, 53),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, g_color, 2, cv2.LINE_AA)

    # Core system telemetry stats
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 280, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"CONFIDENCE: {confidence:.2f}", (w - 280, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Action events & hardware details
    cv2.putText(frame, f"OS CURSOR: {screen_pos[0]}, {screen_pos[1]}", (w - 140, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    if event:
        cv2.putText(frame, f"EVENT: {event}", (w - 140, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)


def draw_skeleton(frame: cv2.Mat, landmarks: List[Tuple[float, float, float]]):
    """Draw neon styled hand wireframe and node joints with anti-aliasing."""
    h, w, _ = frame.shape

    # Calculate absolute coordinates from normalized values
    coords = []
    for lm in landmarks:
        cx, cy = int(lm[0] * w), int(lm[1] * h)
        coords.append((cx, cy))

    # Connect nodes
    for connection in HAND_CONNECTIONS:
        s_idx, e_idx = connection
        if s_idx < len(coords) and e_idx < len(coords):
            cv2.line(frame, coords[s_idx], coords[e_idx], (0, 255, 127), 2, cv2.LINE_AA)

    # Draw individual joints
    for i, pt in enumerate(coords):
        if i == 8:  # Index Tip (Cursor Pointer Node)
            cv2.circle(frame, pt, 7, (255, 255, 0), -1, cv2.LINE_AA)  # Cyan Core
            cv2.circle(frame, pt, 11, (255, 255, 0), 1, cv2.LINE_AA)  # Ring overlay
        elif i == 4:  # Thumb Tip
            cv2.circle(frame, pt, 6, (0, 230, 255), -1, cv2.LINE_AA)  # Gold Tip
        else:
            cv2.circle(frame, pt, 4, (0, 80, 255), -1, cv2.LINE_AA)   # Electric Orange Node


def run_pipeline():
    logger = setup_logger(name="handimouse", level=10)  # DEBUG level logging
    logger.info("=" * 70)
    logger.info("STARTING HANDIMOUSE END-TO-END PIPELINE VALIDATION TEST")
    logger.info("=" * 70)

    try:
        # Initialize optimal MouseController using factory selection
        controller = MouseControllerFactory.create()

        # Initialize core modular layers
        camera = CameraStream(device_index=0, target_resolution=(640, 480))
        tracker = HandTracker(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            dominant_hand_preference="largest"
        )
        gesture_engine = GestureEngine()

        camera.start()
        logger.info("Camera started. Waiting for connection...")

        # Stabilize camera connection
        warmup = time.time()
        connected = False
        while time.time() - warmup < 10.0:
            ok, frame = camera.read()
            if ok and frame is not None:
                connected = True
                break
            time.sleep(0.1)

        if not connected:
            logger.critical("Camera connection could not be established!")
            camera.stop()
            tracker.close()
            return

        logger.info("Connection established! Launching live controller loop.")
        logger.info("INSTRUCTIONS:")
        logger.info("  1. MOVE_CURSOR   -> Extend only your Index Finger")
        logger.info("  2. LEFT_CLICK    -> Briefly pinch Thumb and Index Tip")
        logger.info("  3. DRAG_MODE     -> Sustain the pinch to hold left-down")
        logger.info("  4. RIGHT_CLICK   -> Extend Index and Middle fingers close together")
        logger.info("  5. SCROLL_UP/DN  -> Extend all fingers and lift/lower hand vertically")
        logger.info("  Press 'ESC' in the window to stop safely.")

        # Metrics trackers
        fps_start = time.time()
        frame_counter = 0
        current_fps = 0.0

        # We keep track of the last window position to prevent flashing
        cv2.namedWindow("HandiMouse E2E Controller Pipeline", cv2.WINDOW_AUTOSIZE)

        while True:
            success, frame = camera.read()
            if not success or frame is None:
                time.sleep(0.01)
                continue

            frame_counter += 1
            if frame_counter % 30 == 0:
                now = time.time()
                current_fps = 30.0 / (now - fps_start)
                fps_start = now

            # Process through the pipeline layers:
            # 1. Vision Layer: Track Hand Landmarks
            tracking_result = tracker.process_frame(frame)

            # 2. Intelligence Layer: Recognize Gestures
            gesture_result = gesture_engine.process_state(tracking_result)

            active_gesture = gesture_result["gesture"]
            confidence = gesture_result["confidence"]
            event_trigger = gesture_result["event_trigger"]
            scroll_tick = gesture_result["scroll_tick"]

            # Draw tracking skeleton on visual frame
            if tracking_result["hand_detected"]:
                draw_skeleton(frame, tracking_result["landmarks"])

            # 3. Control Layer: Map & execute real-time actions
            screen_pos = controller.get_position()

            if tracking_result["hand_detected"]:
                # Map active gestures to concrete OS inputs
                if active_gesture in ("MOVE_CURSOR", "DRAG_MODE"):
                    landmarks = tracking_result["landmarks"]
                    index_tip = landmarks[8]
                    # Convert normalized pointer coordinates to absolute screen pixels
                    screen_x, screen_y = controller.landmark_to_screen(index_tip[0], index_tip[1])
                    
                    # Execute move command
                    controller.move_to(screen_x, screen_y)
                    screen_pos = (screen_x, screen_y)

                # Dispatch atomic events
                if event_trigger == "CLICK_LEFT":
                    controller.click_left()
                elif event_trigger == "CLICK_RIGHT":
                    controller.click_right()
                elif event_trigger == "DRAG_START":
                    controller.drag_start()
                elif event_trigger == "DRAG_RELEASE":
                    controller.drag_release()

                # Dispatch scrolling ticks
                if scroll_tick != 0:
                    controller.scroll(scroll_tick)

            # Draw elegant telemetry HUD overlay
            draw_hud(frame, active_gesture, confidence, current_fps, event_trigger, screen_pos)

            # Display feed
            cv2.imshow("HandiMouse E2E Controller Pipeline", frame)

            # Key break condition (Esc key)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Escape
                logger.info("Escape key pressed. Shutting down system gracefully.")
                break

    except Exception as e:
        logger.critical(f"E2E system failure: {e}", exc_info=True)

    finally:
        # Tear down all hardware connections gracefully
        cv2.destroyAllWindows()
        try:
            camera.stop()
        except Exception:
            pass
        try:
            tracker.close()
        except Exception:
            pass
        logger.info("Pipeline closed. Hardware resources successfully released.")


if __name__ == "__main__":
    run_pipeline()
