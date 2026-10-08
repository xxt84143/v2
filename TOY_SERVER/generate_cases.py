"""Generate the circular-island experiment without ERA5, GEBCO or SWAN."""
from __future__ import annotations

import argparse
from pathlib import Path

from island_core import Rectangle, case_plan, generate_forcings, generate_islands, make_island_depth
from project import DEFAULT_CONFIG, load_config, path_for
from swan_inputs import grid_coordinates, prepare_case, write_json, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", action="append")
    parser.add_argument("--limit", type=int, help="Maximum cases per tile and profile")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]
    profiles = args.profile or experiment["profiles"]
    if set(profiles) - set(config["resolutions"]):
        raise ValueError("Unknown profile")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if config["boundary"]["period"] != "peak":
        raise ValueError("Synthetic island wave matrices contain Tp; boundary.period must be peak")
    if config["boundary"]["jonswap_gamma"] != experiment["forcing"]["jonswap_gamma"] or config["boundary"]["directional_spread_degrees"] != experiment["forcing"]["directional_spread_deg"]:
        raise ValueError("Boundary spectrum settings must agree with experiment.forcing")
    forcings = {item.forcing_id: item for item in generate_forcings(experiment)}
    root = path_for(config, "cases")
    records = []
    for tile in config["tiles"]:
        rectangle = Rectangle(west=tile["west"], south=tile["south"])
        islands = {item.terrain_id: item for item in generate_islands(experiment, rectangle)}
        cases = case_plan(list(islands.values()), list(forcings.values()))
        if args.split:
            cases = [case for case in cases if case["split"] == args.split]
        selected = cases[:args.limit]
        for profile in profiles:
            lon, lat = grid_coordinates(tile, config["resolutions"][profile])
            terrain_cache = {}
            for case in selected:
                terrain_id = case["terrain_id"]
                if terrain_id not in terrain_cache:
                    depth, wet, _ = make_island_depth(lon, lat, islands[terrain_id],
                                                     experiment["terrain"]["base_depth_m"], config["grid"]["exception"])
                    if not (wet[0].all() and wet[-1].all() and wet[:, 0].all() and wet[:, -1].all()):
                        raise ValueError("Island touches the open boundary")
                    if wet.all():
                        raise ValueError("This resolution does not resolve the island; use a finer grid")
                    terrain_cache[terrain_id] = depth, wet
                depth, wet = terrain_cache[terrain_id]
                forcing = forcings[case["forcing_id"]]
                sample_id = f"{profile}__{tile['id']}__{case['case_id']}"
                directory = root / profile / tile["id"] / case["case_id"]
                prepare_case(directory, sample_id, case, profile, tile, lon, lat, depth, wet,
                             forcing.wave, forcing.wind, config, args.overwrite)
                records.append({"sample_id": sample_id, "case_id": case["case_id"], "profile": profile,
                                "tile_id": tile["id"], "terrain_id": terrain_id, "forcing_id": case["forcing_id"],
                                "split": case["split"], "generalization": case["generalization"],
                                "family": case["family"], "case_directory": directory.relative_to(root).as_posix()})
                print(f"Prepared {sample_id}: {depth.shape}", flush=True)
    if not records:
        raise ValueError("No cases selected")
    write_manifest(root / "manifest.csv", records)
    write_json(root / "experiment.json", {"schema_version": "island-matrix-experiment-2",
                                          "experiment": experiment, "selected_in_this_call": len(records),
                                          "wave_shape": [3, 2, 2], "wind_shape": [2, 3, 3],
                                          "matrix_order": "channel, south-to-north, west-to-east"})
    print(f"Prepared {len(records)} cases; manifest: {root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
