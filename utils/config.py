from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping, got {type(cfg)!r}")
    return cfg


def _parse_value(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def set_by_dotted_key(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = cfg
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[parts[-1]] = value


def merge_cli_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    merged = copy.deepcopy(cfg)
    if not overrides:
        return merged
    if len(overrides) % 2 != 0:
        raise ValueError("CLI overrides must be KEY VALUE pairs")
    for key, value in zip(overrides[0::2], overrides[1::2]):
        set_by_dotted_key(merged, key, _parse_value(value))
    return merged


def get_by_path(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
