"""Synthetic experiment configuration and quota settings; no external datasets."""
from __future__ import annotations
import json
import math
import re
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.json")
DEFAULT_TRAIN_TARGET = 16384

def load_config(path: Path) -> dict:
    path = path.expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "toy-server-1":
        raise ValueError("Expected schema_version=toy-server-1")
    if config["experiment"].get("generator") != "broad-coverage-1":
        raise ValueError("This version requires generator=broad-coverage-1")
    if config["boundary"]["period"] != "peak":
        raise ValueError("Synthetic wave matrices contain Tp; boundary.period must be peak")
    config["_base"] = str(path.parent)
    seen = set()
    for tile in config["tiles"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", tile["id"]) or tile["id"] in seen:
            raise ValueError("Tile IDs must be unique safe names")
        seen.add(tile["id"])
        west, south = float(tile["west"]), float(tile["south"])
        if not all(math.isfinite(v) for v in (west, south)):
            raise ValueError("Tile coordinates must be finite")
        if not (-180 <= west <= 179.5 and -89.5 <= south <= 89):
            raise ValueError("Dateline/polar tiles are unsupported")
    if not seen:
        raise ValueError("At least one tile is required")
    for name in config["resolutions"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Unsafe resolution name")
    return config

def path_for(config: dict, key: str) -> Path:
    path = Path(config["paths"][key]).expanduser()
    return (path if path.is_absolute() else Path(config["_base"]) / path).resolve()

def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def target_settings(config):
    settings = config.get("case_sampling", {})
    target = positive_int(settings.get("target_effective_train_cases", DEFAULT_TRAIN_TARGET),
                          "target_effective_train_cases")
    reserve = float(settings.get("training_reserve_fraction", .25))
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("training_reserve_fraction must be finite and non-negative")
    candidates = positive_int(settings.get("training_candidate_cases", math.ceil(target * (1 + reserve))),
                              "training_candidate_cases")
    if candidates < target:
        raise ValueError("training_candidate_cases must be at least the effective target")
    evaluation = positive_int(settings.get("evaluation_cases_per_group", 512),
                              "evaluation_cases_per_group")
    if not isinstance(settings.get("require_converged", False), bool):
        raise ValueError("require_converged must be a boolean")
    return target, candidates, evaluation
