"""Generate circular-island terrains, coupled forcings, and the case plan."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import numpy as np
import xarray as xr

import sys
from island_core import (
    HERE, SPLITS, case_plan, generate_forcings, generate_islands, island_to_row,
    forcing_to_row, load_config, make_island_depth, write_csv,
)

sys.path.insert(0, str(HERE.parent))
from v2_core import load_config as load_parent_config, make_grid_spec, resolve_config_path, tile_map  # noqa: E402


def plot_terrain(path: Path, terrain_id: str, lon: np.ndarray, lat: np.ndarray,
                 depth: np.ndarray, wet: np.ndarray, island) -> None:
    """Write the same per-terrain diagnostic view retained by the v1.3 pilot."""
    xx, yy = np.meshgrid(lon, lat)
    masked_depth = np.ma.masked_where(~wet, depth)
    try:
        import cmocean
        cmap = cmocean.cm.deep
    except ImportError:
        cmap = "Blues"
    fig = plt.figure(figsize=(8.8, 8.0), dpi=150)
    ax = fig.add_axes([0.12, 0.10, 0.72, 0.80])
    cax = fig.add_axes([0.87, 0.20, 0.025, 0.60])
    mesh = ax.pcolormesh(
        xx, yy, masked_depth, shading="nearest", cmap=cmap,
        norm=Normalize(0.0, 100.0), rasterized=True,
    )
    ax.contour(xx, yy, wet.astype(float), levels=[0.5], colors="#222222", linewidths=1.0)
    ax.scatter([island.center_lon], [island.center_lat], marker="+", s=85,
               linewidths=1.3, color="#d62728", label="island center", zorder=5)
    ax.set_xlim(float(lon[0]), float(lon[-1]))
    ax.set_ylim(float(lat[0]), float(lat[-1]))
    ax.set_aspect(1 / np.cos(np.deg2rad(float(np.mean(lat)))), adjustable="box")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}\N{DEGREE SIGN}E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}\N{DEGREE SIGN}N"))
    ax.set_title(
        f"Circular-island water depth (m)\n{terrain_id} | radius={island.radius_m / 1000:.2f} km, "
        f"skirt={island.skirt_width_m / 1000:.2f} km",
        fontsize=14, fontweight="bold", pad=10,
    )
    ax.legend(loc="lower left", framealpha=0.9)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("water depth (m)")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def write_terrain(root: Path, profile: str, spec, island, settings: dict, parent: dict, overwrite: bool) -> None:
    directory = root / "terrains" / profile / island.terrain_id
    metadata_path = directory / "metadata.json"
    if metadata_path.exists() and not overwrite:
        return
    directory.mkdir(parents=True, exist_ok=True)
    exception = float(parent["grid"]["bottom_exception"])
    depth, wet, distance = make_island_depth(
        spec.lon, spec.lat, island, float(settings["base_depth_m"]), exception,
    )
    if np.any(~wet[[0, -1], :]) or np.any(~wet[:, [0, -1]]):
        raise ValueError(f"{island.terrain_id}: island/skirt reached the open boundary")
    dataset = xr.Dataset(
        data_vars={
            "depth": (("lat", "lon"), depth, {"units": "m", "positive": "down"}),
            "wet_mask": (("lat", "lon"), wet.astype(np.uint8)),
            "domain_mask": (("lat", "lon"), np.ones_like(wet, dtype=np.uint8)),
            "distance_to_island_center": (("lat", "lon"), distance.astype(np.float32), {"units": "m"}),
        }, coords={"lon": spec.lon, "lat": spec.lat},
        attrs={"schema_version": "island-toy-terrain-1", "terrain_id": island.terrain_id,
               "profile": profile, "base_depth_m": float(settings["base_depth_m"])},
    )
    dataset.to_netcdf(directory / "grid.nc")
    np.savez_compressed(
        directory / "terrain.npz",
        lon=spec.lon,
        lat=spec.lat,
        depth=depth,
        domain_mask=np.ones_like(wet, dtype=np.uint8),
        wet_mask=wet.astype(np.uint8),
        distance_to_island_center=distance.astype(np.float32),
    )
    plot_terrain(directory / "terrain.png", island.terrain_id, spec.lon, spec.lat, depth, wet, island)
    np.savetxt(directory / "bottom.dat", depth, fmt="%.3f")
    dx = (spec.tile.east - spec.tile.west) / (spec.lon.size - 1)
    dy = (spec.tile.north - spec.tile.south) / (spec.lat.size - 1)
    commands = [
        "COORD SPHERICAL CCM",
        f"CGRID REG {spec.tile.west:.10f} {spec.tile.south:.10f} 0 0.5000000000 0.5000000000 "
        f"{spec.lon.size - 1} {spec.lat.size - 1} CIRCLE 36 0.0345300000 0.5477526003 29",
        f"INPGRID BOTTOM {spec.tile.west:.10f} {spec.tile.south:.10f} 0 {spec.lon.size - 1} {spec.lat.size - 1} "
        f"{dx:.12f} {dy:.12f} EXCEPTION {exception:.1f}",
        "READINP BOTTOM 1 'bottom.dat' 3 0 FREE",
    ]
    (directory / "grid_commands.swn").write_text("\n".join(commands) + "\n", encoding="ascii")
    metadata = island_to_row(island)
    metadata.update({
        "schema_version": "island-toy-terrain-1", "profile": profile,
        "shape": list(depth.shape), "base_depth_m": float(settings["base_depth_m"]),
        "land_node_count": int((~wet).sum()), "wet_node_count": int(wet.sum()),
        "minimum_wet_depth_m": float(depth[wet].min()), "maximum_wet_depth_m": float(depth[wet].max()),
        "outer_boundary_depth_m": float(settings["base_depth_m"]), "grid_commands": commands,
        "visualization": "terrain.png", "array_archive": "terrain.npz",
    })
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    parent = load_parent_config(config["_parent_path"])
    era5_path = resolve_config_path(parent, parent["paths"]["era5"])
    with xr.open_dataset(era5_path) as ds:
        tiles = tile_map(ds)
    tile_id = config["tile_id"]
    if tile_id not in tiles:
        raise ValueError(f"Toy tile is not a valid four-wave-corner cell: {tile_id}")
    tile = tiles[tile_id]
    islands = generate_islands(config, tile)
    forcings = generate_forcings(config)
    for profile in config["profiles"]:
        spec = make_grid_spec(tile, profile, parent["resolutions"][profile])
        for island in islands:
            write_terrain(HERE, profile, spec, island, config["terrain"], parent, args.overwrite)
    terrain_rows = [island_to_row(item) for item in islands]
    forcing_rows = [forcing_to_row(item) for item in forcings]
    cases = case_plan(islands, forcings)
    write_csv(HERE / "index" / "terrains.csv", terrain_rows)
    write_csv(HERE / "index" / "forcings.csv", forcing_rows)
    write_csv(HERE / "index" / "cases.csv", cases)
    manifest = {
        "schema_version": "island-toy-index-1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tile_id": tile_id, "profiles": config["profiles"], "terrain_count": len(islands),
        "forcing_count": len(forcings), "case_count": len(cases),
        "terrain_split_counts": {split: sum(item.split == split for item in islands) for split in SPLITS},
        "forcing_split_counts": {split: sum(item.split == split for item in forcings) for split in SPLITS},
        "case_split_counts": {split: sum(row["split"] == split for row in cases) for split in SPLITS},
        "generalization_counts": {name: sum(row["generalization"] == name for row in cases)
                                  for name in ("seen_island_seen_forcing", "new_island", "new_forcing", "new_both")},
        "coupling_scope": "diagnostic kinematic wind-current relation; SWAN consumes prescribed fields and does not solve ocean circulation",
    }
    (HERE / "index" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
