"""
utils/logger.py — Centralized logger for SEAD-AI.
Every module imports this instead of using print() statements.
"""

import logging
import sys
from config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger with consistent formatting.
    Usage:  from utils.logger import get_logger
            logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
