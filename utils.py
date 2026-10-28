"""Shared helpers: custom exceptions, logging, hashing and formatting."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path


class GamePerfError(Exception):
    """Base class for all errors that are safe to show to end users."""


class DataValidationError(GamePerfError):
    """The dataset is missing, unreadable or structurally invalid."""


class InvalidInputError(GamePerfError):
    """A prediction request contains an unsupported or invalid value."""


class ModelUnavailableError(GamePerfError):
    """The model could not be loaded or trained."""


def get_logger(name: str = "gameperf") -> logging.Logger:
    """Return a module logger writing technical details to the console only."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def file_fingerprint(path: Path, length: int = 16) -> str:
    """SHA-256 fingerprint of a file; used to invalidate stale model artifacts."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def fmt_fps(value: float) -> str:
    return f"{value:.0f}"


def fmt_pct(value: float, digits: int = 0) -> str:
    return f"{value * 100:.{digits}f}%"


def impact_label(abs_fps: float, thresholds: list[tuple[float, str]]) -> str:
    """Map an absolute FPS contribution to a human-readable impact level."""
    for limit, label in thresholds:
        if abs_fps >= limit:
            return label
    return "Negligible"
