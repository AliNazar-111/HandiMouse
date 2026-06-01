import sys
import os
import time
import json

# Dynamically append workspace path to Python's system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handimouse.monitoring.logger import setup_logger
from handimouse.config.settings import ConfigurationManager


def test_config_layer():
    logger = setup_logger(name="handimouse", level=10)  # DEBUG level
    logger.info("=" * 60)
    logger.info("RUNNING HANDIMOUSE CONFIGURATION LAYER UNIT TESTS")
    logger.info("=" * 60)

    # Use a separate test configuration file so we don't overwrite workspace settings
    test_config_path = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_config.json"
    ))

    # Clean up any stale test configuration files
    if os.path.exists(test_config_path):
        os.remove(test_config_path)

    try:
        # 1. Verification of default generation
        logger.info("Initializing ConfigurationManager (should generate new test config)...")
        manager = ConfigurationManager(config_path=test_config_path)
        
        assert os.path.exists(test_config_path), "Failed to generate default config.json!"
        logger.info(f"Successfully generated default config at: {test_config_path}")

        # Check default nested values
        logger.info("Verifying default values:")
        logger.info(f"  Camera resolution: {manager.settings.camera.target_width}x{manager.settings.camera.target_height}")
        logger.info(f"  Gesture click cooldown: {manager.settings.gesture_engine.click_cooldown}s")
        assert manager.settings.camera.target_width == 640, "Default target width wrong!"
        assert manager.settings.gesture_engine.click_cooldown == 0.35, "Default click cooldown wrong!"

        # 2. Verification of manual save & reload
        logger.info("\nModifying configuration in-memory and saving...")
        manager.settings.camera.target_width = 1280
        manager.settings.gesture_engine.click_cooldown = 0.50
        manager.save()

        # Instantiate a separate reader manager to verify storage
        logger.info("Reading stored values with new manager instance...")
        reader_manager = ConfigurationManager(config_path=test_config_path)
        logger.info(f"  Saved Camera resolution: {reader_manager.settings.camera.target_width}x{reader_manager.settings.camera.target_height}")
        logger.info(f"  Saved Gesture click cooldown: {reader_manager.settings.gesture_engine.click_cooldown}s")
        assert reader_manager.settings.camera.target_width == 1280, "Failed to save modified width!"
        assert reader_manager.settings.gesture_engine.click_cooldown == 0.50, "Failed to save modified click cooldown!"

        # 3. Verification of Hot-Reload monitoring
        logger.info("\nTesting live hot-reloading check...")
        
        # Check initial update check (should be False because no disk edits happened)
        updated = manager.check_for_updates()
        logger.info(f"Check for updates without file changes -> Updated state: {updated}")
        assert not updated, "Incorrectly reported hot-reload updates before disk changes occurred!"

        # Simulate external disk modifications
        logger.info("Simulating manual external modification in test_config.json...")
        with open(test_config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            
        config_data["camera"]["target_width"] = 1920
        config_data["gesture_engine"]["click_cooldown"] = 0.15

        # We sleep briefly to guarantee file mtime resolution shifts on disk
        time.sleep(0.1)
        with open(test_config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)

        # We bypass the throttled interval gate for unit testing by backing up the last checked clock
        manager._last_checked_time = 0.0

        # Run update check (should trigger dynamic hot-reload)
        updated = manager.check_for_updates()
        logger.info(f"Check for updates after file changes -> Updated state: {updated}")
        logger.info(f"  New Hot-Reloaded Camera width: {manager.settings.camera.target_width}")
        logger.info(f"  New Hot-Reloaded click cooldown: {manager.settings.gesture_engine.click_cooldown}s")
        
        assert updated, "Hot-reload monitoring failed to register file changes!"
        assert manager.settings.camera.target_width == 1920, "Failed to reload updated target width!"
        assert manager.settings.gesture_engine.click_cooldown == 0.15, "Failed to reload updated click cooldown!"

        logger.info("=" * 60)
        logger.info("ALL CONFIGURATION LAYER TESTS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

    except AssertionError as e:
        logger.error(f"[ASSERTION FAILED] {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected crash during config tests: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up testing outputs
        if os.path.exists(test_config_path):
            try:
                os.remove(test_config_path)
            except Exception:
                pass


if __name__ == "__main__":
    test_config_layer()
