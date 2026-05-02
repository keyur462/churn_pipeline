import logging
import sys
from datetime import datetime
from pathlib import Path

from config import LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to both stdout and a daily log file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y_%m_%d')}.log"

    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured - return as-is to avoid duplicate handlers
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger
