"""Package-compatible access to configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from config.loader import DEFAULT_CONFIG_PATH as _DEFAULT_CONFIG_PATH
from config.loader import get_section as _get_section
from config.loader import load_config as _load_config

DEFAULT_CONFIG_PATH = _DEFAULT_CONFIG_PATH


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load configuration using the shared loader.

    This mirrors :func:`config.loader.load_config` while keeping imports under the
    ``watermark_remover`` namespace for compatibility with existing callers.
    """

    return _load_config(path, overrides=overrides)


def get_section(config: Mapping[str, Any], section: str, default: Optional[Any] = None) -> Any:
    """Retrieve a defensive copy of a configuration section."""

    return _get_section(config, section, default)


__all__ = ["DEFAULT_CONFIG_PATH", "load_config", "get_section"]
