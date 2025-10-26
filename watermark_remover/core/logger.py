"""Lightweight logger utility for Watermark Remover Suite."""

from __future__ import annotations
import logging
from typing import Optional, Union

def setup_logging(
    level: Union[int, str] = "INFO",
    use_rich: bool = True,
    name: Optional[str] = None,
    config: Optional[dict] = None,
    force: bool = False
) -> logging.Logger:
    """
    Create/configure a logger without global side effects.
    Safe to import in tests.

    Args:
        level: Logging level (e.g., "INFO", logging.INFO)
        use_rich: Whether to use RichHandler if available
        name: Logger name (defaults to "watermark_remover")
        config: Optional config dict (for compatibility with existing code)
        force: Force reconfiguration even if handlers exist

    Returns:
        Configured logger instance
    """
    # Handle config dict parameter (for compatibility with existing code)
    if config is not None:
        level = config.get("level", level)
        use_rich = config.get("use_rich", use_rich)

    logger_name = name or "watermark_remover"
    logger = logging.getLogger(logger_name)

    # Idempotent: don't add duplicate handlers unless force=True
    if not logger.handlers or force:
        # Clear existing handlers if force=True
        if force:
            logger.handlers.clear()

        lvl = logging.getLevelName(level) if isinstance(level, str) else level
        logger.setLevel(lvl)

        handler: logging.Handler
        if use_rich:
            try:
                from rich.logging import RichHandler  # type: ignore
                handler = RichHandler(rich_tracebacks=False, show_time=True, show_path=False)
            except Exception:
                handler = logging.StreamHandler()
        else:
            handler = logging.StreamHandler()

        fmt = logging.Formatter("%(levelname)s %(name)s: %(message)s")
        # RichHandler ignores formatter format; still attach for fallback
        try:
            handler.setFormatter(fmt)
        except Exception:
            pass

        logger.addHandler(handler)
        logger.propagate = False

    return logger
