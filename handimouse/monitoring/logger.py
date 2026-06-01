"""
Logging & Monitoring Layer: Custom logger setup for HandiMouse.
Provides visual styling and structured formatting for real-time observability.
"""

import logging
import sys
from typing import Optional

# ANSI Color Codes for beautiful CLI output
class LogColors:
    GREY = "\033[90m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD_RED = "\033[1;31m"
    RESET = "\033[0m"


class HandiMouseFormatter(logging.Formatter):
    """
    Custom logging formatter that adds color and structured spacing
    tailored for console observability in real-time pipelines.
    """
    FORMAT = "%(asctime)s | %(levelname)-8s | [%(threadName)-12s] | %(filename)s:%(lineno)d | %(message)s"

    FORMATS = {
        logging.DEBUG: LogColors.GREY + FORMAT + LogColors.RESET,
        logging.INFO: LogColors.BLUE + FORMAT + LogColors.RESET,
        logging.WARNING: LogColors.YELLOW + FORMAT + LogColors.RESET,
        logging.ERROR: LogColors.RED + FORMAT + LogColors.RESET,
        logging.CRITICAL: LogColors.BOLD_RED + FORMAT + LogColors.RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


def setup_logger(
    name: str = "handimouse", 
    level: int = logging.INFO, 
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure a structured thread-safe logger for HandiMouse.

    Args:
        name: Name of the logger instance.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a file where logs should be written.

    Returns:
        logging.Logger: Fully configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if logger is re-initialized
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create console handler with custom colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(HandiMouseFormatter())
    logger.addHandler(console_handler)

    # Optional file handler (uncolored for files)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | [%(threadName)-12s] | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    # Ensure logs from library dependencies don't flood the handimouse console output
    logging.getLogger("mediapipe").setLevel(logging.WARNING)

    return logger
