"""YAML configuration with strict validation and environment-based secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {path}")
    return value


def require_keys(config: dict[str, Any], keys: list[str], where: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ConfigError(f"Missing keys in {where}: {', '.join(missing)}")


def secret_from_env(env_name: str) -> str:
    value = os.environ.get(env_name)
    if not value:
        raise ConfigError(
            f"Environment variable {env_name} is not set. Set it locally; never put API keys in YAML."
        )
    return value


def dump_yaml(config: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
