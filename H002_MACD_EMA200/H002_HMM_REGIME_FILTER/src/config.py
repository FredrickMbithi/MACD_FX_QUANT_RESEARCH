from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(config_path: str | Path, candidate: str | Path) -> Path:
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        return candidate_path
    return (Path(config_path).resolve().parent / candidate_path).resolve()
