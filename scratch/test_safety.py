import sys
import os
import time

# Dynamically append workspace path to Python's system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handimouse.monitoring.logger import setup_logger
from handimouse.safety.bounds_validator import BoundsValidator
from handimouse.safety.safety_interlock import SafetyInterlock


def test_safety_layer():
    logger = setup_logger(name="handimouse", level=10)  # DEBUG level
    logger.info("=" * 60)
    logger.info("RUNNING HANDIMOUSE SAFETY LAYER UNIT TESTS")
    logger.info("=" * 60)

    try:
        # 1. BoundsValidator test
        logger.info("--- Testing BoundsValidator ---")
        validator = BoundsValidator(max_jump_fraction=0.3, max_rejected_frames=3)
        
        # Test basic clamping
        logger.info("Testing bounds clamping:")
        x, y = validator.validate_coordinates(-100, -100)
        logger.info(f"Target (-100, -100) -> Clamped to ({x}, {y})")
        assert x == 0 and y == 0, "Failed clamping to top-left!"

        validator.reset()
        x, y = validator.validate_coordinates(99999, 99999)
        logger.info(f"Target (99999, 99999) -> Clamped to ({x}, {y})")
        assert x == validator.screen_width - 1 and y == validator.screen_height - 1, "Failed clamping to bottom-right!"


        # Test normal small movement
        validator.reset()
        validator.validate_coordinates(500, 500) # Anchor initial
        x, y = validator.validate_coordinates(510, 495)
        logger.info(f"Target (510, 495) -> Validated to ({x}, {y})")
        assert x == 510 and y == 495, "Failed normal movement validation!"

        # Test large jump gating
        logger.info("\nTesting coordinate jump gating:")
        huge_x = int(500 + validator.screen_width * 0.5)
        x, y = validator.validate_coordinates(huge_x, 500)
        logger.info(f"Huge Jump target ({huge_x}, 500) -> Gated to ({x}, {y}) [Should equal anchor (510, 495)]")
        assert x == 510 and y == 495, "Failed jump gating filter!"

        # Test dead-reckoning recovery (4 frames of huge jumps should recover/re-anchor)
        logger.info("\nTesting jump gating recovery (dead-reckoning timeout):")
        for i in range(1, 5):
            x, y = validator.validate_coordinates(huge_x, 500)
            logger.info(f"Frame {i}/4 Target ({huge_x}, 500) -> Result: ({x}, {y})")
        
        logger.info(f"Final coordinates after timeout: ({x}, {y})")
        assert x == min(validator.screen_width - 1, huge_x) and y == 500, "Failed dead-reckoning recovery re-anchor!"

        # 2. SafetyInterlock test
        logger.info("\n--- Testing SafetyInterlock ---")
        interlock = SafetyInterlock(fist_frames_threshold=5) # Set lower threshold for testing

        # Test basic hand movement (normal state)
        logger.info("Simulating normal hand tracking states:")
        mock_tracking = {
            "hand_detected": True,
            "fingers_up": [1, 1, 0, 0, 0] # Move gesture
        }
        is_locked = interlock.process_safety(mock_tracking)
        logger.info(f"Normal fingers [1, 1, 0, 0, 0] -> Locked state: {is_locked}")
        assert not is_locked, "Safety incorrectly triggered on normal hand state!"

        # Test emergency fist hold gesture
        logger.info("\nSimulating emergency fist gesture hold:")
        mock_fist = {
            "hand_detected": True,
            "fingers_up": [0, 0, 0, 0, 0] # Tight fist emergency
        }
        
        for frame in range(1, 7):
            is_locked = interlock.process_safety(mock_fist)
            logger.info(f"Fist Frame {frame} -> Locked state: {is_locked} (Reason: {interlock.reason})")
            if frame < 5:
                assert not is_locked, "Emergency triggered too early before threshold!"
            else:
                assert is_locked, "Emergency failed to trigger after threshold hold!"

        # Reset interlock
        logger.info("\nTesting interlock manual reset:")
        interlock.reset()
        assert not interlock.interlocked, "Manual reset failed to clear interlocked state!"

        logger.info("=" * 60)
        logger.info("ALL SAFETY LAYER TESTS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

    except AssertionError as e:
        logger.error(f"[ASSERTION FAILED] {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected crash during tests: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    test_safety_layer()
