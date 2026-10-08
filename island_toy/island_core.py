"""Pure geometry/forcing helpers for the circular-island toy experiment."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
EARTH_RADIUS_M = 6_371_008.8
SPLITS = ("train", "validation", "test")
SIDES = ("NORTH", "EAST", "SOUTH", "WEST")


@dataclass(frozen=True)
class Island:
    terrain_id: str
    split: str
    center_lon: float
    center_lat: float
    radius_m: float
    skirt_width_m: float
    shore_depth_m: float


@dataclass(frozen=True)
class CoupledForcing:
    forcing_id: str
    split: str
    hs_m: float
    tp_s: float
    wave_from_deg: float
    boundary_side: str
    wind_speed_mps: float
    wind_from_deg: float
    wind_u10_mps: float
    wind_v10_mps: float
    current_speed_mps: float
    current_to_deg: float
    current_u_mps: float
    current_v_mps: float
    coupling_mode: str


def load_config(path: str | Path = HERE / "config.json") -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "island-toy-config-1":
        raise ValueError("Unsupported island toy config")
    config["_config_path"] = str(path)
    parent = Path(config["parent_config"])
    if not parent.is_absolute():
        parent = path.parent / parent
    config["_parent_path"] = str(parent.resolve())
    return config


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def local_xy_m(lon: np.ndarray | float, lat: np.ndarray | float, reference_lon: float, reference_lat: float):
    x = EARTH_RADIUS_M * math.cos(math.radians(reference_lat)) * np.deg2rad(np.asarray(lon) - reference_lon)
    y = EARTH_RADIUS_M * np.deg2rad(np.asarray(lat) - reference_lat)
    return x, y


def domain_size_m(tile) -> tuple[float, float]:
    cx, cy = tile.center
    x0, _ = local_xy_m(tile.west, cy, cx, cy)
    x1, _ = local_xy_m(tile.east, cy, cx, cy)
    _, y0 = local_xy_m(cx, tile.south, cx, cy)
    _, y1 = local_xy_m(cx, tile.north, cx, cy)
    return float(x1 - x0), float(y1 - y0)


def generate_islands(config: dict[str, Any], tile) -> list[Island]:
    settings = config["terrain"]
    rng = np.random.default_rng(int(config["seed"]))
    width, height = domain_size_m(tile)
    cx0, cy0 = tile.center
    islands: list[Island] = []
    ordinal = 0
    for split in SPLITS:
        for _ in range(int(settings["count"][split])):
            ordinal += 1
            for attempt in range(10_000):
                radius = rng.uniform(*settings["island_radius_km"]) * 1000.0
                skirt = rng.uniform(*settings["skirt_width_km"]) * 1000.0
                shore = rng.uniform(*settings["shore_depth_m"])
                margin = radius + skirt + float(settings["outer_clearance_km"]) * 1000.0
                if 2 * margin >= min(width, height):
                    continue
                x = rng.uniform(-width / 2 + margin, width / 2 - margin)
                y = rng.uniform(-height / 2 + margin, height / 2 - margin)
                lon = cx0 + math.degrees(x / (EARTH_RADIUS_M * math.cos(math.radians(cy0))))
                lat = cy0 + math.degrees(y / EARTH_RADIUS_M)
                islands.append(Island(f"T{ordinal:03d}", split, lon, lat, radius, skirt, shore))
                break
            else:
                raise RuntimeError("Could not place an island with the requested clearance")
    return islands


def make_island_depth(lon: np.ndarray, lat: np.ndarray, island: Island, base_depth_m: float, exception: float):
    xx, yy = np.meshgrid(lon, lat)
    dx, dy = local_xy_m(xx, yy, island.center_lon, island.center_lat)
    distance = np.hypot(dx, dy)
    wet = distance > island.radius_m
    depth = np.full(distance.shape, exception, dtype=np.float32)
    outer = island.radius_m + island.skirt_width_m
    transition = np.clip((distance - island.radius_m) / island.skirt_width_m, 0.0, 1.0)
    smooth = transition * transition * (3.0 - 2.0 * transition)
    water_depth = island.shore_depth_m + (base_depth_m - island.shore_depth_m) * smooth
    depth[wet] = np.minimum(water_depth[wet], base_depth_m).astype(np.float32)
    depth[distance >= outer] = np.float32(base_depth_m)
    return depth, wet, distance


def vector_from_direction(speed: float, direction_from_deg: float) -> tuple[float, float]:
    angle = math.radians(direction_from_deg)
    return -speed * math.sin(angle), -speed * math.cos(angle)


def vector_to_direction(speed: float, direction_to_deg: float) -> tuple[float, float]:
    angle = math.radians(direction_to_deg)
    return speed * math.sin(angle), speed * math.cos(angle)


def incoming_side(direction_from_deg: float) -> str:
    index = int(math.floor(((direction_from_deg % 360.0) + 45.0) / 90.0)) % 4
    return SIDES[index]


def generate_forcings(config: dict[str, Any]) -> list[CoupledForcing]:
    settings = config["forcing"]
    rng = np.random.default_rng(int(config["seed"]) + 1)
    total = sum(int(settings["count"][split]) for split in SPLITS)
    speeds = np.linspace(float(settings["wind_speed_mps"][0]), float(settings["wind_speed_mps"][1]), total)
    rng.shuffle(speeds)
    wind_directions = (np.arange(total) * 137.50776405 + 15.0) % 360.0
    current_offsets = list(map(float, settings["current_direction_offsets_deg"]))
    wave_offsets = list(map(float, settings["wave_wind_offsets_deg"]))
    mode_names = {0.0: "downwind", -45.0: "right_crosswind", 45.0: "left_crosswind", 180.0: "opposing"}
    forcings: list[CoupledForcing] = []
    ordinal = 0
    for split in SPLITS:
        for _ in range(int(settings["count"][split])):
            wind_speed = float(speeds[ordinal])
            wind_from = float(wind_directions[ordinal])
            wind_to = (wind_from + 180.0) % 360.0
            current_offset = current_offsets[ordinal % len(current_offsets)]
            current_to = (wind_to + current_offset) % 360.0
            current_speed = float(settings["current_base_speed_mps"]) + float(settings["current_per_wind"]) * wind_speed
            wave_from = (wind_from + wave_offsets[ordinal % len(wave_offsets)]) % 360.0
            hs = float(settings["wave_hs_base_m"]) + float(settings["wave_hs_per_wind"]) * wind_speed
            tp = float(settings["wave_tp_base_s"]) + float(settings["wave_tp_per_wind"]) * wind_speed
            wind_u, wind_v = vector_from_direction(wind_speed, wind_from)
            current_u, current_v = vector_to_direction(current_speed, current_to)
            forcings.append(CoupledForcing(
                f"F{ordinal + 1:03d}", split, hs, tp, wave_from, incoming_side(wave_from),
                wind_speed, wind_from, wind_u, wind_v, current_speed, current_to,
                current_u, current_v, mode_names.get(current_offset, f"offset_{current_offset:g}"),
            ))
            ordinal += 1
    return forcings


def case_plan(islands: list[Island], forcings: list[CoupledForcing]) -> list[dict[str, Any]]:
    terrain = {split: [item for item in islands if item.split == split] for split in SPLITS}
    forcing = {split: [item for item in forcings if item.split == split] for split in SPLITS}
    pairs: list[tuple[str, str, list[Island], list[CoupledForcing]]] = [
        ("train", "seen_island_seen_forcing", terrain["train"], forcing["train"]),
        ("validation", "new_island", terrain["validation"], forcing["train"]),
        ("validation", "new_forcing", terrain["train"], forcing["validation"]),
        ("validation", "new_both", terrain["validation"], forcing["validation"]),
        ("test", "new_island", terrain["test"], forcing["train"]),
        ("test", "new_forcing", terrain["train"], forcing["test"]),
        ("test", "new_both", terrain["test"], forcing["test"]),
    ]
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for split, generalization, terrain_items, forcing_items in pairs:
        for island in terrain_items:
            for condition in forcing_items:
                ordinal += 1
                row = {"case_id": f"IT{ordinal:06d}", "terrain_id": island.terrain_id,
                       "forcing_id": condition.forcing_id, "split": split,
                       "generalization": generalization}
                row.update(forcing_to_row(condition))
                rows.append(row)
    return rows


def island_to_row(item: Island) -> dict[str, Any]:
    return {"terrain_id": item.terrain_id, "terrain_split": item.split, "center_lon": item.center_lon,
            "center_lat": item.center_lat, "radius_m": item.radius_m, "skirt_width_m": item.skirt_width_m,
            "shore_depth_m": item.shore_depth_m}


def forcing_to_row(item: CoupledForcing) -> dict[str, Any]:
    return {
        "forcing_id": item.forcing_id, "forcing_split": item.split, "hs_m": item.hs_m,
        "tp_s": item.tp_s, "wave_from_deg": item.wave_from_deg, "boundary_side": item.boundary_side,
        "wind_speed_mps": item.wind_speed_mps, "wind_from_deg": item.wind_from_deg,
        "wind_u10_mps": item.wind_u10_mps, "wind_v10_mps": item.wind_v10_mps,
        "current_speed_mps": item.current_speed_mps, "current_to_deg": item.current_to_deg,
        "current_u_mps": item.current_u_mps, "current_v_mps": item.current_v_mps,
        "coupling_mode": item.coupling_mode,
    }

