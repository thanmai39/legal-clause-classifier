"""
Centralized logging configuration.

Usage in any module:
    from src.utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("something happened")

We log to console (stdout) so it works the same locally, in Docker, and in
most cloud platforms (which typically capture stdout/stderr as logs
automatically -- no file management needed).

IMPORTANT: never log request/contract text content or API keys here.
Log events and metadata (e.g. "prediction completed in 0.4s"), not the
actual clause text a user submitted -- contract text may be sensitive.
"""

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
