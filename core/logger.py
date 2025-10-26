"""Structured logging helpers for the Watermark Remover Suite."""

from __future__ import annotations

import logging
import os
from logging import Handler
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Mapping, Optional

DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _remove_handlers(handlers: Iterable[Handler]) -> None:
    for handler in handlers:
        handler.close()


def _resolve_log_path(filename: str) -> Path:
    expanded = Path(os.path.expandvars(filename)).expanduser()
    try:
        expanded.parent.mkdir(parents=True, exist_ok=True)
        return expanded
    except OSError:
        fallback_dir = Path("./logs")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir / Path(filename).name


def setup_logging(settings: Mapping[str, object], *, force: bool = False) -> None:
    """Configure logging handlers based on YAML configuration."""
    level = str(settings.get("level", "INFO")).upper()
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if force:
        _remove_handlers(root_logger.handlers)
        root_logger.handlers.clear()

    console_settings = settings.get("console", {}) or {}
    if console_settings.get("enabled", True):
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            console_handler = logging.StreamHandler()
            console_format = console_settings.get(
                "format", "%(levelname)s | %(name)s | %(message)s"
            )
            console_handler.setFormatter(logging.Formatter(console_format))
            root_logger.addHandler(console_handler)

    file_settings = settings.get("file", {}) or {}
    if file_settings.get("enabled", False):
        filename = file_settings.get("filename")
        if not filename:
            raise ValueError("File logging enabled but no filename provided.")
        log_path = _resolve_log_path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rotate_bytes = int(file_settings.get("rotate_bytes", 1_048_576))
        backups = int(file_settings.get("backups", 5))
        existing = [
            h
            for h in root_logger.handlers
            if isinstance(h, RotatingFileHandler)
            and Path(getattr(h, "baseFilename", "")).resolve() == log_path.resolve()
        ]
        if not existing:
            file_handler = RotatingFileHandler(
                log_path, maxBytes=rotate_bytes, backupCount=backups, encoding="utf-8"
            )
            file_format = file_settings.get("format", DEFAULT_FORMAT)
            file_handler.setFormatter(logging.Formatter(file_format))
            root_logger.addHandler(file_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Helper to retrieve a module-specific logger."""
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger", "DEFAULT_FORMAT"]
