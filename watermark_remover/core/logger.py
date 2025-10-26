"""Logging helpers for the Watermark Remover Suite."""

import logging
import os
import re
from collections.abc import Iterable, Mapping
from logging import Handler
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional, Union

DEFAULT_CONSOLE_FORMAT = "%(levelname)s %(name)s: %(message)s"
DEFAULT_FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _close_handlers(handlers: Iterable[Handler]) -> None:
    for handler in handlers:
        handler.close()


def _coerce_level(level: Union[int, str]) -> int:
    if isinstance(level, int):
        return level
    lvl = logging.getLevelName(level.upper())
    if isinstance(lvl, int):
        return lvl
    raise ValueError(f"Invalid logging level: {level!r}")


def _resolve_log_path(filename: str) -> Path:
    """Expand environment variables (Windows and POSIX styles) for log files."""

    def expand_windows(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    expanded = re.sub(r"%([^%]+)%", expand_windows, filename)
    expanded = os.path.expandvars(expanded)
    path = Path(expanded).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        fallback_dir = Path("./logs")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        path = fallback_dir / Path(filename).name
    return path


def setup_logging(
    level: Union[int, str, Mapping[str, Any]] = "INFO",
    use_rich: bool = True,
    name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    force: bool = False,
) -> logging.Logger:
    """Configure logging using simple defaults or a YAML-style mapping."""

    config_map: dict[str, Any]
    if isinstance(level, Mapping):
        config_map = dict(level)
        if config:
            config_map.update(config)
        level_value = config_map.get("level", "INFO")
    else:
        config_map = dict(config or {})
        level_value = config_map.get("level", level)

    logger_name = name or "watermark_remover"
    logger = logging.getLogger(logger_name)
    root_logger = logging.getLogger()

    numeric_level = _coerce_level(level_value)
    root_logger.setLevel(numeric_level)
    logger.setLevel(numeric_level)

    if force:
        _close_handlers(root_logger.handlers)
        root_logger.handlers.clear()
        if logger is not root_logger:
            _close_handlers(logger.handlers)
            logger.handlers.clear()

    console_cfg = dict(config_map.get("console", {}) or {})
    console_enabled = console_cfg.get("enabled", True)
    console_use_rich = console_cfg.get("use_rich", use_rich)
    console_format = console_cfg.get("format", DEFAULT_CONSOLE_FORMAT)

    if console_enabled:
        existing_console = [
            h
            for h in root_logger.handlers
            if isinstance(h, logging.StreamHandler)
            or h.__class__.__name__ == "RichHandler"
        ]
        handler: Handler | None = None
        if not existing_console:
            if console_use_rich:
                try:
                    from rich.logging import RichHandler  # type: ignore

                    handler = RichHandler(
                        rich_tracebacks=False, show_time=True, show_path=False
                    )
                except Exception:
                    handler = None
            if handler is None:
                handler = logging.StreamHandler()
            try:
                handler.setFormatter(logging.Formatter(console_format))
            except Exception:
                handler.setFormatter(logging.Formatter(DEFAULT_CONSOLE_FORMAT))
            root_logger.addHandler(handler)

    file_cfg = dict(config_map.get("file", {}) or {})
    if file_cfg.get("enabled"):
        filename = file_cfg.get("filename")
        if not filename:
            raise ValueError("File logging enabled but no filename provided.")
        log_path = _resolve_log_path(str(filename))
        rotate_bytes = int(file_cfg.get("rotate_bytes", 1_048_576))
        backups = int(file_cfg.get("backups", 5))

        existing_file = None
        for handler in root_logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                existing_path = Path(getattr(handler, "baseFilename", "")).resolve()
                if existing_path == log_path.resolve():
                    existing_file = handler
                    break
        if existing_file is None:
            file_handler = RotatingFileHandler(
                log_path, maxBytes=rotate_bytes, backupCount=backups, encoding="utf-8"
            )
            file_format = file_cfg.get("format", DEFAULT_FILE_FORMAT)
            file_handler.setFormatter(logging.Formatter(file_format))
            root_logger.addHandler(file_handler)

    return logger
