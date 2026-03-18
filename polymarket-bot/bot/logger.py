"""Structured logging setup for the trading bot."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from bot import config

_SENSITIVE_PATTERNS = [
    re.compile(r"(private_key\s*[:=]\s*)[^\s,}\"']+", re.IGNORECASE),
    re.compile(r"(api_key\s*[:=]\s*)[^\s,}\"']+", re.IGNORECASE),
    re.compile(r"(api_secret\s*[:=]\s*)[^\s,}\"']+", re.IGNORECASE),
    re.compile(r"(passphrase\s*[:=]\s*)[^\s,}\"']+", re.IGNORECASE),
    re.compile(r"(0x[0-9a-fA-F]{40,})", re.IGNORECASE),  # Ethereum addresses / keys
]


def _scrub(message: str) -> str:
    for pattern in _SENSITIVE_PATTERNS:
        message = pattern.sub(r"\g<1>***REDACTED***", message)
    return message


class _ScrubFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _scrub(str(record.msg))
        return True


def get_logger(name: str = "polymarket_bot") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    scrub_filter = _ScrubFilter()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.addFilter(scrub_filter)
    logger.addHandler(ch)

    # File handler (optional)
    if config.LOG_FILE:
        log_path = Path(config.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        fh.addFilter(scrub_filter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
