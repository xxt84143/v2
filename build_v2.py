"""Build the v2 tile catalog, two grid tiers, and ERA5 forcing index."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from v2_core import (
    CORNER_NAMES, HERE, enumerate_tiles, load_config, make_grid_spec,
    resolve_config_path, split_for_case, write_csv,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_wet_mask(initial_wet: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Reuse the established grid-builder rule: connected water with 2-D cells."""
    ny, nx = initial_wet.shape
    seeds = np.zeros_like(initial_wet, dtype=bool)
    seeds[0, :] = initial_wet[0, :]
    seeds[-1, :] = initial_wet[-1, :]
    seeds[:, 0] = initial_wet[:, 0]
    seeds[:, -1] = initial_wet[:, -1]
    if not np.any(seeds):
        raise ValueError("Tile has no wet nodes on its outer boundary")
    wet = np.zeros_like(initial_wet, dtype=bool)
    queue = deque((int(iy), int(ix)) for iy, ix in np.argwhere(seeds))
    while queue:
        iy, ix = queue.popleft()
        if wet[iy, ix] or not initial_wet[iy, ix]:
            continue
        wet[iy, ix] = True
        for jy, jx in ((iy - 1, ix), (iy + 1, ix), (iy, ix - 1), (iy, ix + 1)):
            if 0 <= jy < ny and 0 <= jx < nx and initial_wet[jy, jx] and not wet[jy, jx]:
                queue.append((jy, jx))
    disconnected = int(np.count_nonzero(initial_wet & ~wet))
    removed = 0
    iterations = 0
    while True:
        cells = wet[:-1, :-1] & wet[1:, :-1] & wet[:-1, 1:] & wet[1:, 1:]
        belongs = np.zeros_like(wet)
        belongs[:-1, :-1] |= cells
        belongs[1:, :-1] |= cells
        belongs[:-1, 1:] |= cells
        belongs[1:, 1:] |= cells
        keep = wet & belongs
        count = int(np.count_nonzero(wet & ~keep))
        if not count:
            break
        wet = keep
        removed += count
        iterations += 1
    return wet, {
        "initial_wet_node_count": int(initial_wet.sum()),
        "open_boundary_seed_node_count": int(seeds.sum()),
        "disconnected_wet_nodes_removed": disconnected,
        "non_2d_wet_nodes_removed": removed,
        "stencil_cleanup_iterations": iterations,
    }


def interpolate_bathymetry(gebco: xr.Dataset, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Bilinear interpolation on the regular GEBCO grid using only NumPy."""
    source_lon = np.asarray(gebco.lon.values, dtype=np.float64)
    source_lat = np.asarray(gebco.lat.values, dtype=np.float64)
    source = np.asarray(gebco.elevation.values, dtype=np.float32)
    if source.shape != (source_lat.size, source_lon.size):
        raise ValueError("GEBCO elevation dimensions do not match lat/lon")
    if lon[0] < source_lon[0] or lon[-1] > source_lon[-1] or lat[0] < source_lat[0] or lat[-1] > source_lat[-1]:
        raise ValueError("GEBCO does not cover the requested tile")
    horizontal = np.empty((source_lat.size, lon.size), dtype=np.float32)
    for row in range(source_lat.size):
        horizontal[row] = np.interp(lon, source_lon, source[row])
    result = np.empty((lat.size, lon.size), dtype=np.float32)
    for column in range(lon.size):
        result[:, column] = np.interp(lat, source_lat, horizontal[:, column])
    return result


def write_catalog(config: dict, era5: xr.Dataset, output: Path) -> dict[str, object]:
    rows = []
    for tile in enumerate_tiles(era5):
        row: dict[str, object] = {
            "tile_id": tile.tile_id, "west": tile.west, "east": tile.east,
            "south": tile.south, "north": tile.north, "center_lon": tile.center[0],
            "center_lat": tile.center[1], "wave_corner_valid_all_hours": 1,
        }
        for name, (iy, ix) in zip(CORNER_NAMES, tile.wave_indices):
            row[f"{name}_wave_lat_index"] = iy
            row[f"{name}_wave_lon_index"] = ix
        rows.append(row)
    write_csv(output / "index" / "tile_catalog.csv", rows)
    selected = set(config["tiles"])
    available = {str(row["tile_id"]) for row in rows}
    missing = sorted(selected.difference(available))
    if missing:
        raise ValueError(f"Configured tiles are not four-corner ocean cells: {', '.join(missing)}")
    return {tile.tile_id: tile for tile in enumerate_tiles(era5)}


def write_grid(config: dict, gebco: xr.Dataset, spec, output: Path, overwrite: bool) -> None:
    grid_dir = output / "grids" / spec.profile / spec.tile.tile_id
    metadata_path = grid_dir / "grid_metadata.json"
    if metadata_path.exists() and not overwrite:
        print(f"keep existing grid: {grid_dir}")
        return
    grid_dir.mkdir(parents=True, exist_ok=True)
    elevation = interpolate_bathymetry(gebco, spec.lon, spec.lat)
    if elevation.shape != spec.shape or not np.isfinite(elevation).all():
        raise ValueError(f"GEBCO interpolation failed for {spec.tile.tile_id}/{spec.profile}")
    exception = float(config["grid"]["bottom_exception"])
    minimum = float(config["grid"]["minimum_wet_depth_m"])
    wet, cleanup = clean_wet_mask(elevation < 0.0)
    depth = np.full(spec.shape, exception, dtype=np.float32)
    depth[wet] = np.maximum(-elevation[wet], minimum)
    domain = np.ones(spec.shape, dtype=np.uint8)
    ds = xr.Dataset(
        data_vars={
            "depth": (("lat", "lon"), depth, {"units": "m", "positive": "down"}),
            "domain_mask": (("lat", "lon"), domain),
            "wet_mask": (("lat", "lon"), wet.astype(np.uint8)),
        },
        coords={"lon": spec.lon, "lat": spec.lat},
        attrs={
            "schema_version": "toy-v2-rectangular-grid-1", "tile_id": spec.tile.tile_id,
            "profile": spec.profile, "nominal_spacing_m": spec.nominal_spacing_m,
            "storage_order": "rows south-to-north; columns west-to-east; SWAN IDLA=3",
            "bathymetry_source": str(resolve_config_path(config, config["paths"]["gebco"])),
        },
    )
    ds.to_netcdf(grid_dir / "grid.nc")
    np.savetxt(grid_dir / "bottom.dat", depth, fmt="%.2f")
    xlen = spec.tile.east - spec.tile.west
    ylen = spec.tile.north - spec.tile.south
    dx = xlen / (spec.lon.size - 1)
    dy = ylen / (spec.lat.size - 1)
    commands = [
        "COORD SPHERICAL CCM",
        f"CGRID REG {spec.tile.west:.10f} {spec.tile.south:.10f} 0 {xlen:.10f} {ylen:.10f} "
        f"{spec.lon.size - 1} {spec.lat.size - 1} CIRCLE 36 0.0345300000 0.5477526003 29",
        f"INPGRID BOTTOM {spec.tile.west:.10f} {spec.tile.south:.10f} 0 "
        f"{spec.lon.size - 1} {spec.lat.size - 1} {dx:.12f} {dy:.12f} EXCEPTION {exception:.1f}",
        "READINP BOTTOM 1 'bottom.dat' 3 0 FREE",
    ]
    (grid_dir / "grid_commands.swn").write_text("\n".join(commands) + "\n", encoding="ascii")
    # SW, SE, NE, NW; every domain vertex is exactly an ERA5 wave point.
    vertex_rows = []
    locations = ((0, 0), (spec.lon.size - 1, 0), (spec.lon.size - 1, spec.lat.size - 1), (0, spec.lat.size - 1))
    for order, ((iy_era, ix_era), (ix, iy), corner) in enumerate(zip(spec.tile.wave_indices, locations, CORNER_NAMES), start=1):
        vertex_rows.append({
            "boundary_order": order, "selection_order": order, "corner": corner,
            "era5_lat_index": iy_era, "era5_lon_index": ix_era, "ix": ix, "iy": iy,
            "lon": f"{spec.lon[ix]:.10f}", "lat": f"{spec.lat[iy]:.10f}",
            "source_depth_m": f"{max(float(depth[iy, ix]), 0.0):.3f}",
            "domain_mask": 1, "active_wet_mask": int(wet[iy, ix]),
        })
    write_csv(grid_dir / "boundary_vertices.csv", vertex_rows)
    center_lat = spec.tile.center[1]
    dx_m = 111_195.08 * math.cos(math.radians(center_lat)) * dx
    dy_m = 111_195.08 * dy
    metadata = {
        "schema_version": "toy-v2-rectangular-grid-1", "created_at_utc": utc_now(),
        "tile_id": spec.tile.tile_id, "profile": spec.profile,
        "bounds": [spec.tile.west, spec.tile.south, spec.tile.east, spec.tile.north],
        "shape": list(spec.shape), "intervals": [spec.lon.size - 1, spec.lat.size - 1],
        "dx_degrees": dx, "dy_degrees": dy, "dx_m_at_center": dx_m,
        "dy_m": dy_m, "nominal_spacing_m": spec.nominal_spacing_m,
        "corner_wave_points_exact": True, "wet_node_count": int(wet.sum()),
        "land_node_count": int((~wet).sum()), "bottom_exception": exception,
        "bathymetry": {"mask_cleanup": cleanup},
        "grid_commands": commands,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {spec.profile}/{spec.tile.tile_id}: {spec.shape[1]}x{spec.shape[0]} nodes")


def write_cases(config: dict, era5: xr.Dataset, tiles: dict, output: Path) -> None:
    source_path = resolve_config_path(config, config["paths"]["source_cases"])
    import pandas as pd
    source = pd.read_csv(source_path, parse_dates=["time"])
    time_values = np.asarray(era5.valid_time.values)
    rows = []
    for source_row in source.itertuples(index=False):
        timestamp = np.datetime64(source_row.time.to_datetime64(), "ns")
        index = int(np.searchsorted(time_values, timestamp))
        if index >= time_values.size or time_values[index] != timestamp:
            raise ValueError(f"Source case time is absent from ERA5: {source_row.time}")
        for tile_id in config["tiles"]:
            tile = tiles[tile_id]
            row: dict[str, object] = {
                "sample_id": f"{tile_id}__{source_row.case_id}", "tile_id": tile_id,
                "source_case_id": source_row.case_id, "case_id": str(source_row.case_id),
                "time": source_row.time.isoformat(), "era5_time_index": index,
                "split": split_for_case(str(source_row.case_id), config["split"]),
            }
            wave = np.empty((3, 2, 2), dtype=np.float32)
            wind = np.empty((2, 3, 3), dtype=np.float32)
            for y, latitude in enumerate((tile.south, tile.north)):
                for x, longitude in enumerate((tile.west, tile.east)):
                    for channel, name in enumerate(("swh", "mwp", "mwd")):
                        wave[channel, y, x] = float(era5[name].sel(wave_latitude=latitude, wave_longitude=longitude).isel(valid_time=index))
            wave[2] %= 360
            for y, latitude in enumerate((tile.south, tile.center[1], tile.north)):
                for x, longitude in enumerate((tile.west, tile.center[0], tile.east)):
                    for channel, name in enumerate(("u10", "v10")):
                        wind[channel, y, x] = float(era5[name].sel(wind_latitude=latitude, wind_longitude=longitude).isel(valid_time=index))
            from forcing_arrays import validate
            validate(wave, wind)
            forcing_file = Path("forcings") / tile_id / f"{source_row.case_id}.npz"
            destination = output / "index" / forcing_file
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, wave=wave, wind=wind)
            row["forcing_file"] = forcing_file.as_posix()
            rows.append(row)
    write_csv(output / "index" / "cases.csv", rows)
    summary = {
        "schema_version": "toy-v2-index-1", "created_at_utc": utc_now(),
        "sample_count": len(rows), "tile_count": len(config["tiles"]),
        "source_case_count": len(source), "wind_shape": [2, 3, 3], "wave_shape": [3, 2, 2],
        "matrix_order": "channel, south-to-north, west-to-east",
        "split_counts": {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")},
        "leakage_rule": "same source_case_id has the same split for every tile and resolution",
    }
    (output / "index" / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} forcing rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("catalog", "grids", "cases", "all"), nargs="?", default="all")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    era5_path = resolve_config_path(config, config["paths"]["era5"])
    gebco_path = resolve_config_path(config, config["paths"]["gebco"])
    with xr.open_dataset(era5_path) as era5:
        tiles = write_catalog(config, era5, args.output)
        if args.command in ("grids", "all"):
            with xr.open_dataset(gebco_path) as gebco:
                for profile, settings in config["resolutions"].items():
                    for tile_id in config["tiles"]:
                        write_grid(config, gebco, make_grid_spec(tiles[tile_id], profile, settings), args.output, args.overwrite)
        if args.command in ("cases", "all"):
            write_cases(config, era5, tiles, args.output)
    if args.command == "catalog":
        print(f"wrote catalog with {len(tiles)} valid tiles")


if __name__ == "__main__":
    main()
