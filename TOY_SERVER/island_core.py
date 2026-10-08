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
    wave: np.ndarray  # (Hs, Tp, direction FROM), 2x2
    wind: np.ndarray  # (u10, v10), 3x3; nine independent input nodes
    family: str
    wave_age: float
    steepness: float


@dataclass(frozen=True)
class Rectangle:
    west: float
    south: float

    @property
    def east(self):
        return self.west + .5

    @property
    def north(self):
        return self.south + .5

    @property
    def center(self):
        return (self.west + .25, self.south + .25)


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


def generate_forcings(config: dict[str, Any]) -> list[CoupledForcing]:
    """Prescribed joint sea states, not a fitted ERA5 climatology or ocean model."""
    settings = config["forcing"]
    families = ("swell_no_wind", "aligned_windsea", "swell_weak_wind", "wind_only")
    g = 9.81
    result = []
    for split_number, split in enumerate(SPLITS):
        count = int(settings["count"][split])
        if count < len(families):
            raise ValueError("Each forcing split needs at least four samples to cover all sea-state families")
        rng = np.random.default_rng(int(config["seed"]) + 1 + 1009 * split_number)
        # Every split covers the full compass, including oblique directions.
        directions = (np.arange(count) * 360 / count + rng.uniform(0, 360 / count, count)) % 360
        rng.shuffle(directions)
        for local in range(count):
            family = families[local % len(families)]
            direction = float(directions[local])
            gradient = float(settings["maximum_relative_gradient"])
            sx, sy = rng.uniform(-gradient, gradient, 2)
            dx, dy = rng.uniform(-float(settings["maximum_direction_gradient_deg"]),
                                  float(settings["maximum_direction_gradient_deg"]), 2)
            y2, x2 = np.meshgrid([-1., 1.], [-1., 1.], indexing="ij")
            y3, x3 = np.meshgrid([-1., 0., 1.], [-1., 0., 1.], indexing="ij")
            wind_from = direction
            wave_age, steepness = 0., 0.
            if family == "aligned_windsea":
                speed = rng.uniform(*settings["windsea_speed_mps"])
                wave_age = rng.uniform(*settings["windsea_wave_age"])
                steepness = rng.uniform(*settings["windsea_steepness"])
                # cp/U = g*Tp/(2*pi*U): period and height follow wind and wave age.
                local_speed = speed * (1 + sx * x2 + sy * y2)
                period = 2 * np.pi * local_speed * wave_age / g
                hs = steepness * g * period**2 / (2 * np.pi)
                wave_from = (direction + dx * x2 + dy * y2) % 360
            elif family == "wind_only":
                speed = rng.uniform(*settings["wind_only_speed_mps"])
                period = np.full((2, 2), 8.)  # dummy positive period for zero incoming energy
                hs = np.zeros((2, 2))
                wave_from = (direction + dx * x2 + dy * y2) % 360
            else:
                speed = 0. if family == "swell_no_wind" else rng.uniform(*settings["swell_wind_speed_mps"])
                wind_from = (direction + float(rng.choice(settings["swell_wind_offsets_deg"]))) % 360
                period = rng.uniform(*settings["swell_period_s"]) * (1 + .03 * x2 * rng.uniform(-1, 1))
                hs = rng.uniform(*settings["swell_height_m"]) * (1 + sx * x2 + sy * y2)
                wave_from = (direction + dx * x2 + dy * y2) % 360
            wind_speed = speed * (1 + sx * x3 + sy * y3)
            wind_angle = np.deg2rad(wind_from + dx * x3 + dy * y3)
            wind = np.stack((-wind_speed * np.sin(wind_angle), -wind_speed * np.cos(wind_angle)))
            wave = np.stack((hs, period, wave_from))
            actual_steepness = hs / (g * period**2 / (2 * np.pi))
            if np.any(actual_steepness > float(settings["maximum_steepness"])):
                raise ValueError("Requested forcing is too steep; reduce Hs or increase Tp")
            if period.min() < 1 / .5477526003 or period.max() > 1 / .03453:
                raise ValueError("Wave peak falls outside the configured spectral grid")
            result.append(CoupledForcing(f"F{len(result)+1:03d}", split, wave, wind, family,
                                         float(wave_age), float(actual_steepness.max())))
    return result


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
    return {"forcing_id": item.forcing_id, "forcing_split": item.split,
            "forcing_file": f"forcings/{item.forcing_id}.npz", "family": item.family,
            "wave_age": item.wave_age, "maximum_steepness": item.steepness,
            "mean_hs_m": float(item.wave[0].mean()), "mean_tp_s": float(item.wave[1].mean()),
            "maximum_wind_speed_mps": float(np.hypot(*item.wind).max())}


def runtime_config(config, parent):
    return {"grid": {"exception": parent["grid"]["bottom_exception"]},
            "boundary": {"jonswap_gamma": config["forcing"]["jonswap_gamma"],
                         "directional_spread_degrees": config["forcing"]["directional_spread_deg"]},
            "spectral_grid": {"directions": 36, "lowest_frequency_hz": .03453,
                              "highest_frequency_hz": .5477526003, "frequency_intervals": 29},
            "physics_commands": config["physics_commands"], "output_quantities": parent["output"]["quantities"]}
