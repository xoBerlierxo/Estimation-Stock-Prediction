"""Shared logging setup used by every pipeline entry point."""

from __future__ import annotations

import logging
import sys

_CONFIGURED_LOGGERS: set[str] = set()


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a logger with a consistent formatter, configured only once per name."""
    logger = logging.getLogger(name)
    if name not in _CONFIGURED_LOGGERS:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _CONFIGURED_LOGGERS.add(name)
    return logger
