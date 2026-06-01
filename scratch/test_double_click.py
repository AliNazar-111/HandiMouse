import sys
import os
import time

# Dynamically append workspace path to Python's system path
sys.path.append(r"d:\ali\HandiMouse")

from handimouse.monitoring.logger import setup_logger
from handimouse.intelligence.gesture_engine import GestureEngine
from handimouse.safety.safety_interlock import SafetyInterlock

def test_double_click_and_fist_suppression():
    logger = setup_logger(name="handimouse", level=10)  # DEBUG level
    logger.info("=" * 65)
    logger.info("RUNNING HANDIMOUSE OFFLINE DOUBLE-CLICK & FIST SUPPRESSION TESTS")
    logger.info("=" * 65)

    try:
        # 1. Test double-click detection in GestureEngine
        logger.info("--- Testing GestureEngine Double-Click Logic ---")
        engine = GestureEngine({
            "debounce_frames": 1,        # 1 frame for instant transition in offline test
            "click_cooldown": 0.1,
            "double_click_window": 0.50
        })

        # Mock tracking data representing a pinch (thumb tip to index tip close)
        # Thumb: (0, 0, 0), Index: (0.05, 0, 0) -> distance 0.05
        # Middle knuckle at (0, 0.5, 0) -> hand scale = 0.5, normalized = 0.1 < pinch_threshold (0.15)
        pinch_landmarks = [
            (0.0, 0.0, 0.0),  # 0: Wrist
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),  # 4: Thumb Tip
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.05, 0.0, 0.0), # 8: Index Tip
            (0.0, 0.5, 0.0),  # 9: Middle Knuckle
        ] + [(0,0,0)] * 11

        mock_pinch_tracking = {
            "hand_detected": True,
            "confidence": 0.9,
            "landmarks": pinch_landmarks,
            "fingers_up": [0, 0, 0, 0, 0]  # Note: fingers_up doesn't affect pinch detection
        }

        # Mock data representing moving cursor (no pinch)
        # Index Tip at (0.5, 0.0, 0.0) -> distance 0.5 -> normalized = 1.0 > pinch_threshold
        idle_landmarks = list(pinch_landmarks)
        idle_landmarks[8] = (0.5, 0.0, 0.0)
        mock_idle_tracking = {
            "hand_detected": True,
            "confidence": 0.9,
            "landmarks": idle_landmarks,
            "fingers_up": [1, 1, 0, 0, 0]
        }

        # Fire 1st pinch (starts pinch)
        res1_press = engine.process_state(mock_pinch_tracking)
        logger.info(f"Pinch 1 Press -> Gesture: {res1_press['gesture']}, Event: {res1_press['event_trigger']}")
        assert res1_press["event_trigger"] == "", "Pinch down should not trigger event immediately!"

        # Release 1st pinch -> CLICK_LEFT!
        res1_release = engine.process_state(mock_idle_tracking)
        logger.info(f"Pinch 1 Release -> Gesture: {res1_release['gesture']}, Event: {res1_release['event_trigger']}")
        assert res1_release["event_trigger"] == "CLICK_LEFT", "Pinch 1 Release should trigger CLICK_LEFT!"

        # Fire 2nd pinch quickly (within 0.5s window)
        time.sleep(0.1)
        res2_press = engine.process_state(mock_pinch_tracking)
        logger.info(f"Pinch 2 Press -> Gesture: {res2_press['gesture']}, Event: {res2_press['event_trigger']}")
        
        # Release 2nd pinch -> DOUBLE_CLICK!
        res2_release = engine.process_state(mock_idle_tracking)
        logger.info(f"Pinch 2 Release -> Gesture: {res2_release['gesture']}, Event: {res2_release['event_trigger']}")
        assert res2_release["event_trigger"] == "DOUBLE_CLICK", "Pinch 2 Release should trigger DOUBLE_CLICK!"

        # Fire 3rd pinch slowly (after 0.6s, outside 0.5s window)
        time.sleep(0.6)
        res3_press = engine.process_state(mock_pinch_tracking)
        logger.info(f"Pinch 3 Press -> Gesture: {res3_press['gesture']}, Event: {res3_press['event_trigger']}")
        
        # Release 3rd pinch -> CLICK_LEFT!
        res3_release = engine.process_state(mock_idle_tracking)
        logger.info(f"Pinch 3 Release -> Gesture: {res3_release['gesture']}, Event: {res3_release['event_trigger']}")
        assert res3_release["event_trigger"] == "CLICK_LEFT", "Slow pinch release should trigger CLICK_LEFT, not DOUBLE_CLICK!"

        logger.info("GestureEngine Double-Click verification: PASSED.")

        # 2. Test safety interlock suppress fist during active pinching
        logger.info("\n--- Testing SafetyInterlock Fist Suppression ---")
        interlock = SafetyInterlock(fist_frames_threshold=3)

        # A tight fist state
        mock_fist_tracking = {
            "hand_detected": True,
            "fingers_up": [0, 0, 0, 0, 0]
        }

        # Simulating active pinch - is_pinching is True
        logger.info("Simulating fist frames with suppress_fist=True:")
        for frame in range(1, 5):
            is_locked = interlock.process_safety(mock_fist_tracking, suppress_fist=True)
            logger.info(f"Frame {frame} -> Locked state: {is_locked}")
            assert not is_locked, "Safety incorrectly triggered when suppress_fist is True!"

        # Now without suppression
        logger.info("\nSimulating fist frames with suppress_fist=False:")
        for frame in range(1, 5):
            is_locked = interlock.process_safety(mock_fist_tracking, suppress_fist=False)
            logger.info(f"Frame {frame} -> Locked state: {is_locked}")
            if frame < 3:
                assert not is_locked, "Safety triggered too early!"
            else:
                assert is_locked, "Safety failed to trigger on tight fist!"

        logger.info("SafetyInterlock Fist Suppression verification: PASSED.")

        logger.info("=" * 65)
        logger.info("ALL OFFLINE FUNCTIONAL TESTS PASSED SUCCESSFULLY!")
        logger.info("=" * 65)

    except AssertionError as e:
        logger.error(f"[ASSERTION FAILED] {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected crash during offline tests: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    test_double_click_and_fist_suppression()
