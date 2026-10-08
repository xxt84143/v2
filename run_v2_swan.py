"""Prepare, run, and inspect rectangular v2 SWAN cases.

The mature SWAN_YEARLY runner is reused for boundary tracing, JONSWAP forcing,
INPUT rendering, case execution, and status files.  This wrapper only supplies
the v2 rectangle/grid contract and uses the full nine-node ERA5 wind matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from v2_core import HERE, load_config, read_csv, resolve_config_path


SWAN_YEARLY = HERE.parents[1] / "SWAN_YEARLY"
sys.path.insert(0, str(SWAN_YEARLY))
import swan_batch as core  # noqa: E402


def core_config(config: dict, nominal_spacing_m: int, dx: float, dy: float) -> dict:
    # A land corner can legitimately have an ERA5 wave value because ERA5's
    # coarse land mask differs from GEBCO.  Permit snapping along either
    # adjacent rectangle side, up to 0.25 degree, and record the mapping.
    snap_cells = max(2.0, math.ceil(0.25 / min(dx, dy)))
    return {
        "grid_resolution_m": int(nominal_spacing_m),
        "boundary": {
            "mode": "jonswap",
            "jonswap_gamma": float(config["boundary"]["jonswap_gamma"]),
            "directional_spread_degrees": float(config["boundary"]["directional_spread_degrees"]),
            "inactive_vertex_snap_cells": snap_cells,
        },
        "physics_commands": list(config["physics_commands"]),
        "output": dict(config["output"]),
        "variables": {
            "time": "valid_time", "wind_longitude": "wind_longitude",
            "wind_latitude": "wind_latitude", "wave_longitude": "wave_longitude",
            "wave_latitude": "wave_latitude", "u10": "u10", "v10": "v10",
            "swh": "swh", "mwp": "mwp", "mwd": "mwd",
        },
    }


def load_grid(grid_dir: Path) -> tuple[core.GridContext, dict]:
    metadata = json.loads((grid_dir / "grid_metadata.json").read_text(encoding="utf-8"))
    with xr.open_dataset(grid_dir / "grid.nc") as ds:
        lon = np.asarray(ds.lon.values, dtype=np.float64)
        lat = np.asarray(ds.lat.values, dtype=np.float64)
        domain = np.asarray(ds.domain_mask.values, dtype=bool)
        wet = np.asarray(ds.wet_mask.values, dtype=bool)
    vertices = []
    with (grid_dir / "boundary_vertices.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            vertices.append(core.Vertex(
                boundary_order=int(row["boundary_order"]), selection_order=int(row["selection_order"]),
                era5_lat_index=int(row["era5_lat_index"]), era5_lon_index=int(row["era5_lon_index"]),
                ix=int(row["ix"]), iy=int(row["iy"]), lon=float(row["lon"]), lat=float(row["lat"]),
                active_wet=bool(int(row["active_wet_mask"])),
            ))
    commands = [line.strip() for line in (grid_dir / "grid_commands.swn").read_text(encoding="ascii").splitlines() if line.strip()]
    grid = core.GridContext(grid_dir, metadata, lon, lat, domain, wet, vertices, commands)
    return grid, metadata


def selected_rows(index: Path, tile_id: str, case_ids: list[str] | None, limit: int | None) -> list[dict[str, str]]:
    rows = [row for row in read_csv(index) if row["tile_id"] == tile_id]
    if case_ids:
        requested = set(case_ids)
        rows = [row for row in rows if row["case_id"] in requested]
        missing = requested.difference(row["case_id"] for row in rows)
        if missing:
            raise ValueError(f"Unknown case IDs for {tile_id}: {', '.join(sorted(missing))}")
    return rows[:limit] if limit else rows


def to_case(row: dict[str, str]) -> core.CaseRecord:
    stamp = datetime.fromisoformat(row["time"].replace("Z", "+00:00"))
    return core.CaseRecord(row["case_id"], stamp, row)


def use_nine_node_wind(data: core.ExtractedData, rows: list[dict[str, str]], tile) -> core.ExtractedData:
    from forcing_arrays import load
    matrices = np.stack([load(HERE / "index" / row["forcing_file"])[1] for row in rows])
    return core.ExtractedData(data.time_indices, matrices[:, 0], matrices[:, 1], data.swh, data.mwp, data.mwd,
                              np.asarray([tile.west, tile.center[0], tile.east]),
                              np.asarray([tile.south, tile.center[1], tile.north]))


def context(config: dict, profile: str, tile_id: str, case_ids: list[str] | None, limit: int | None):
    from v2_core import tile_map
    era5_path = resolve_config_path(config, config["paths"]["era5"])
    grid_dir = HERE / "grids" / profile / tile_id
    if not grid_dir.is_dir():
        raise FileNotFoundError(f"Grid is absent: {grid_dir}; run build_v2.py all")
    grid, metadata = load_grid(grid_dir)
    cfg = core_config(config, metadata["nominal_spacing_m"], metadata["dx_degrees"], metadata["dy_degrees"])
    rows = selected_rows(HERE / "index" / "cases.csv", tile_id, case_ids, limit)
    cases = [to_case(row) for row in rows]
    if not cases:
        raise ValueError("No v2 cases selected")
    with xr.open_dataset(era5_path) as era5:
        tile = tile_map(era5)[tile_id]
    geometry = core.orient_and_map_boundary(grid, cfg)
    data = core.extract_era5(era5_path, cases, grid, cfg)
    data = use_nine_node_wind(data, rows, tile)
    return cfg, cases, rows, grid, geometry, data, tile


def run_root(profile: str, tile_id: str) -> Path:
    return HERE / "runs" / profile / tile_id


def prepare(config: dict, profile: str, tile_id: str, case_ids: list[str] | None, limit: int | None, overwrite: bool) -> None:
    cfg, cases, rows, grid, geometry, data, tile = context(config, profile, tile_id, case_ids, limit)
    root = run_root(profile, tile_id)
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, (case, row) in enumerate(zip(cases, rows)):
        case_dir = root / case.case_id
        if case_dir.exists() and not overwrite:
            record = {
                "case_id": case.case_id, "time_utc": case.time_utc.isoformat().replace("+00:00", "Z"),
                "era5_time_index": int(data.time_indices[index]), "case_directory": case.case_id,
                "status": core.read_status(case_dir).get("status", "prepared"),
            }
        else:
            record = core.prepare_case(root, case, index, grid, geometry, data, cfg, core.utc_now(), overwrite)
        from forcing_arrays import load
        wave, wind = load(HERE / "index" / row["forcing_file"])
        np.savez_compressed(case_dir / "forcing.npz", wave=wave, wind=wind)
        metadata_path = case_dir / "case_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["v2_contract"] = {
            "schema_version": "toy-v2-case-extension-1", "tile_id": tile_id,
            "resolution_profile": profile,
            "wave_inputs": "four ERA5 corners: swh/mwp/mwd",
            "wind_inputs": "all nine ERA5 wind nodes: wind(2,3,3)",
            "swan_wind_corner_policy": "direct nine-node inputs; no five-point reconstruction",
            "forcing_file": "forcing.npz",
        }
        core.atomic_write_json(metadata_path, metadata)
        record.update({"sample_id": row["sample_id"], "tile_id": tile_id, "profile": profile, "split": row["split"]})
        manifest.append(record)
        print(f"prepared {index + 1}/{len(cases)} {case.case_id}", flush=True)
    core.atomic_write_csv(root / "manifest.csv", list(manifest[0]), manifest)


def run(config: dict, profile: str, tile_id: str, case_ids: list[str] | None, limit: int | None, workers: int) -> int:
    root = run_root(profile, tile_id)
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} is absent; run prepare first")
    rows = read_csv(manifest_path)
    if case_ids:
        wanted = set(case_ids)
        rows = [row for row in rows if row["case_id"] in wanted]
    if limit:
        rows = rows[:limit]
    executable = resolve_config_path(config, config["paths"]["swan_executable"])
    expected = Path(config["output"]["filename"])
    def one(row):
        return core.run_one_case(row, root, executable, expected, None)
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, row): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            failed += result["status"] != "completed"
            print(result["case_id"], result["status"], result.get("runtime_seconds"), flush=True)
    return 1 if failed else 0


def status(profile: str, tile_id: str) -> None:
    root = run_root(profile, tile_id)
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        print({"not_prepared": 0})
        return
    counts: dict[str, int] = {}
    for row in read_csv(manifest):
        state = core.read_status(root / row["case_id"]).get("status", "missing")
        counts[state] = counts.get(state, 0) + 1
    print(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--profile", choices=("gebco15s", "1km"), required=True)
    parser.add_argument("--tile", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    prepare_parser = sub.add_parser("prepare")
    run_parser = sub.add_parser("run")
    sub.add_parser("status")
    for item in (check, prepare_parser, run_parser):
        item.add_argument("--case-id", action="append", dest="case_ids")
        item.add_argument("--limit", type=int)
    prepare_parser.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "check":
        cfg, cases, _, grid, geometry, data, _ = context(config, args.profile, args.tile, args.case_ids, args.limit)
        print({
            "cases": len(cases), "grid_shape": [grid.lat.size, grid.lon.size],
            "open_segments": len(geometry.segments), "open_nodes": geometry.open_node_count,
            "wind_grid": list(data.u10.shape[1:]), "wave_vertices": data.swh.shape[1],
            "snap_cells": cfg["boundary"]["inactive_vertex_snap_cells"],
        })
    elif args.command == "prepare":
        prepare(config, args.profile, args.tile, args.case_ids, args.limit, args.overwrite)
    elif args.command == "run":
        raise SystemExit(run(config, args.profile, args.tile, args.case_ids, args.limit, args.workers))
    else:
        status(args.profile, args.tile)


if __name__ == "__main__":
    main()
