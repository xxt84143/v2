"""Shared geometry and forcing utilities for the 0.5-degree v2 experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
EARTH_RADIUS_M = 6_371_008.8
CORNER_NAMES = ("sw", "se", "ne", "nw")
WIND_NAMES = ("south", "east", "north", "west", "center")


@dataclass(frozen=True)
class Tile:
    tile_id: str
    west: float
    east: float
    south: float
    north: float
    wave_indices: tuple[tuple[int, int], ...]  # SW, SE, NE, NW as (lat, lon)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.west + self.east) / 2.0, (self.south + self.north) / 2.0)


@dataclass(frozen=True)
class GridSpec:
    profile: str
    tile: Tile
    lon: np.ndarray
    lat: np.ndarray
    nominal_spacing_m: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.lat.size, self.lon.size)


def load_config(path: str | Path = HERE / "config.json") -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "toy-v2-config-1":
        raise ValueError(f"Unsupported v2 config schema: {payload.get('schema_version')!r}")
    payload["_config_path"] = str(path)
    return payload


def resolve_config_path(config: dict[str, Any], value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_path"]).parent / path
    return path.resolve()


def coordinate_step(values: np.ndarray, expected: float, name: str) -> None:
    values = np.asarray(values, dtype=float)
    diff = np.diff(values)
    if values.ndim != 1 or values.size < 3:
        raise ValueError(f"{name} must be a one-dimensional coordinate")
    if not (np.allclose(np.abs(diff), expected, atol=1e-10, rtol=0.0)):
        raise ValueError(f"{name} must have {expected:g}-degree spacing")


def validate_era5_geometry(ds: xr.Dataset) -> None:
    required = {
        "valid_time", "wave_longitude", "wave_latitude", "wind_longitude",
        "wind_latitude", "swh", "mwp", "mwd", "u10", "v10",
    }
    missing = sorted(required.difference(ds.variables))
    if missing:
        raise ValueError(f"ERA5 file is missing: {', '.join(missing)}")
    coordinate_step(ds.wave_longitude.values, 0.5, "wave_longitude")
    coordinate_step(ds.wave_latitude.values, 0.5, "wave_latitude")
    coordinate_step(ds.wind_longitude.values, 0.25, "wind_longitude")
    coordinate_step(ds.wind_latitude.values, 0.25, "wind_latitude")
    for wave in (ds.wave_longitude.values, ds.wave_latitude.values):
        doubled = wave * 2.0
        if not np.allclose(doubled, np.round(doubled), atol=1e-10, rtol=0.0):
            raise ValueError("ERA5 wave coordinates are not on the 0.5-degree lattice")
    for wind in (ds.wind_longitude.values, ds.wind_latitude.values):
        quadrupled = wind * 4.0
        if not np.allclose(quadrupled, np.round(quadrupled), atol=1e-10, rtol=0.0):
            raise ValueError("ERA5 wind coordinates are not on the 0.25-degree lattice")


def _coord_token(prefix: str, value: float) -> str:
    return prefix + f"{value:.1f}".replace("-", "m").replace(".", "p")


def make_tile_id(west: float, south: float) -> str:
    return f"{_coord_token('E', west)}_{_coord_token('N', south)}"


def enumerate_tiles(ds: xr.Dataset) -> list[Tile]:
    """Return cells whose four wave corners are finite for every source hour."""
    validate_era5_geometry(ds)
    lon = np.asarray(ds.wave_longitude.values, dtype=float)
    lat = np.asarray(ds.wave_latitude.values, dtype=float)
    valid = np.ones((lat.size, lon.size), dtype=bool)
    for name in ("swh", "mwp", "mwd"):
        valid &= np.asarray(ds[name].notnull().all("valid_time").values, dtype=bool)
    tiles: list[Tile] = []
    # ERA5 latitude is descending. iy is the north row; iy+1 is the south row.
    for iy in range(lat.size - 1):
        for ix in range(lon.size - 1):
            corners = ((iy + 1, ix), (iy + 1, ix + 1), (iy, ix + 1), (iy, ix))
            if not all(valid[row, col] for row, col in corners):
                continue
            west, east = float(lon[ix]), float(lon[ix + 1])
            south, north = float(lat[iy + 1]), float(lat[iy])
            tiles.append(Tile(make_tile_id(west, south), west, east, south, north, corners))
    return tiles


def tile_map(ds: xr.Dataset) -> dict[str, Tile]:
    return {tile.tile_id: tile for tile in enumerate_tiles(ds)}


def _distance_m(lon0: float, lat0: float, lon1: float, lat1: float) -> float:
    mean_lat = math.radians((lat0 + lat1) / 2.0)
    dx = EARTH_RADIUS_M * math.cos(mean_lat) * math.radians(lon1 - lon0)
    dy = EARTH_RADIUS_M * math.radians(lat1 - lat0)
    return math.hypot(dx, dy)


def make_grid_spec(tile: Tile, profile: str, settings: dict[str, Any]) -> GridSpec:
    kind = settings["kind"]
    if kind == "angular":
        step = float(settings["step_degrees"])
        nx_intervals = int(round((tile.east - tile.west) / step))
        ny_intervals = int(round((tile.north - tile.south) / step))
        if not math.isclose(nx_intervals * step, tile.east - tile.west, abs_tol=1e-10):
            raise ValueError(f"{profile} does not divide the tile longitude exactly")
        if not math.isclose(ny_intervals * step, tile.north - tile.south, abs_tol=1e-10):
            raise ValueError(f"{profile} does not divide the tile latitude exactly")
    elif kind == "nominal_metre":
        spacing = float(settings["spacing_m"])
        center_lon, center_lat = tile.center
        nx_intervals = max(1, int(round(_distance_m(tile.west, center_lat, tile.east, center_lat) / spacing)))
        ny_intervals = max(1, int(round(_distance_m(center_lon, tile.south, center_lon, tile.north) / spacing)))
    else:
        raise ValueError(f"Unknown resolution kind: {kind}")
    lon = np.linspace(tile.west, tile.east, nx_intervals + 1, dtype=np.float64)
    lat = np.linspace(tile.south, tile.north, ny_intervals + 1, dtype=np.float64)
    if not (lon[0] == tile.west and lon[-1] == tile.east and lat[0] == tile.south and lat[-1] == tile.north):
        raise AssertionError("Grid endpoints do not equal ERA5 wave corners")
    return GridSpec(profile, tile, lon, lat, int(settings["nominal_spacing_m"]))


def wind_points(tile: Tile) -> dict[str, tuple[float, float]]:
    cx, cy = tile.center
    return {
        "south": (cx, tile.south), "east": (tile.east, cy),
        "north": (cx, tile.north), "west": (tile.west, cy), "center": (cx, cy),
    }


def reconstruct_wind_grid(tile: Tile, values: dict[str, float], power: float = 2.0) -> np.ndarray:
    """IDW the five declared wind controls onto the required 3x3 SWAN grid."""
    points = wind_points(tile)
    missing = sorted(set(WIND_NAMES).difference(values))
    if missing:
        raise ValueError(f"Missing wind controls: {', '.join(missing)}")
    lon = np.asarray([tile.west, tile.center[0], tile.east])
    lat = np.asarray([tile.south, tile.center[1], tile.north])
    out = np.empty((3, 3), dtype=np.float32)
    controls = [(points[name][0], points[name][1], float(values[name])) for name in WIND_NAMES]
    for iy, y in enumerate(lat):
        for ix, x in enumerate(lon):
            distances = np.asarray([_distance_m(x, y, px, py) for px, py, _ in controls])
            exact = np.flatnonzero(distances < 1e-7)
            if exact.size:
                out[iy, ix] = controls[int(exact[0])][2]
            else:
                weights = distances ** (-power)
                out[iy, ix] = float(np.dot(weights, [item[2] for item in controls]) / weights.sum())
    return out


def split_for_case(case_id: str, settings: dict[str, Any]) -> str:
    digest = hashlib.sha256(f"{int(settings['seed'])}:{case_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    train = float(settings["train_fraction"])
    validation = float(settings["validation_fraction"])
    if train <= 0 or validation < 0 or train + validation >= 1:
        raise ValueError("split fractions must satisfy train>0, validation>=0, train+validation<1")
    return "train" if value < train else ("validation" if value < train + validation else "test")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def padded_shape(shape: tuple[int, int], multiple: int) -> tuple[int, int]:
    return tuple(int(math.ceil(size / multiple) * multiple) for size in shape)  # type: ignore[return-value]


def read_swan_block(path: Path, shape: tuple[int, int], quantities: list[str], name: str) -> np.ndarray:
    quantities = [item.upper() for item in quantities]
    # SWAN writes vector WIND as two consecutive fixed-width blocks although
    # it appears as one keyword in BLOCK.  Counting keywords therefore gives
    # the wrong block size (and was the fragile assumption in the v1 reader).
    expanded: list[str] = []
    for quantity in quantities:
        expanded.extend(("WIND_X", "WIND_Y") if quantity == "WIND" else (quantity,))
    target = name.upper()
    index = expanded.index(target)
    size = path.stat().st_size
    if size % len(expanded):
        raise ValueError(f"{path}: size is not divisible by {len(expanded)} scalar output blocks")
    block_bytes = size // len(expanded)
    with path.open("rb") as handle:
        handle.seek(index * block_bytes)
        raw = handle.read(block_bytes)
    values = np.fromstring(raw.decode("ascii"), sep=" ", dtype=np.float32)
    expected = shape[0] * shape[1]
    if values.size != expected:
        raise ValueError(f"{path}: {name} has {values.size} values, expected {expected}")
    return values.reshape(shape)


def forcing_columns() -> list[str]:
    wave = [f"wave_{corner}_{field}" for corner in CORNER_NAMES for field in ("hs_m", "mwp_s", "mwd_deg")]
    wind = [f"wind_{point}_{component}" for point in WIND_NAMES for component in ("u10", "v10")]
    return wave + wind
