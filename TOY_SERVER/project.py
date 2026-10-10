"""Small shared contract: configuration, paths, and UTC sampling only."""
from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.json")
PERIODS = {
    "mean": ("mp1", "mean_wave_period_based_on_first_moment", "MEAN", "Tm01 = m0/m1"),
    "peak": ("pp1d", "peak_wave_period", "PEAK", "Tp"),
}


def load_config(path: Path) -> dict:
    path = path.expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "toy-server-1":
        raise ValueError("Expected schema_version=toy-server-1")
    config["_base"] = str(path.parent)
    if config["boundary"]["period"] not in PERIODS:
        raise ValueError("boundary.period must be mean (ERA5 mp1) or peak (ERA5 pp1d)")
    seen = set()
    for tile in config["tiles"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", tile["id"]) or tile["id"] in seen:
            raise ValueError("Tile IDs must be unique and contain only letters, numbers, _ and -")
        seen.add(tile["id"])
        west, south = float(tile["west"]), float(tile["south"])
        if not all(math.isfinite(v) and math.isclose(v * 2, round(v * 2)) for v in (west, south)):
            raise ValueError("Tile southwest corners must lie on the 0.5 degree ERA5 grid")
        if not (-180 <= west <= 179.5 and -89.5 <= south <= 89):
            raise ValueError("This rectangular implementation excludes dateline/polar tiles")
    if not seen:
        raise ValueError("At least one tile is required")
    for name in config["resolutions"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Unsafe resolution name")
    return config


def path_for(config: dict, key: str) -> Path:
    path = Path(config["paths"][key]).expanduser()
    return (path if path.is_absolute() else Path(config["_base"]) / path).resolve()


def utc_datetime(text: str) -> datetime:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    value = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if value.minute or value.second or value.microsecond:
        raise ValueError("ERA5 sampling must use exact UTC hours")
    return value


def sample_times(config: dict) -> list[datetime]:
    sample = config["sampling"]
    start, end = utc_datetime(sample["start_utc"]), utc_datetime(sample["end_utc"])
    step = sample["step_hours"]
    if not isinstance(step, int) or step <= 0 or end < start:
        raise ValueError("Require integer step_hours > 0 and end_utc >= start_utc")
    result = []
    while start <= end:
        result.append(start)
        start += timedelta(hours=step)
    return result


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Windows can briefly deny replacement while another process holds the destination.
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(.02 * 2**attempt)
