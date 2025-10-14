"""Configuration loader for the Watermark Remover Suite."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

import yaml

PathLike = Union[str, Path]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _merge_dicts(base: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> None:
    """Recursively merge ``overrides`` into ``base`` in-place."""
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _merge_dicts(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def load_config(path: Optional[PathLike] = None, *, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Load YAML configuration from disk and optionally apply overrides."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if overrides:
        _merge_dicts(data, overrides)
    return data


def get_section(config: Mapping[str, Any], section: str, default: Optional[Any] = None) -> Any:
    """Safely retrieve a configuration subsection."""
    return copy.deepcopy(config.get(section, default))


__all__ = ["load_config", "get_section", "DEFAULT_CONFIG_PATH"]
