"""Generate circle/ellipse experiments without ERA5, GEBCO or SWAN."""
from __future__ import annotations

import argparse
from pathlib import Path

from island_core import Rectangle, case_plan, generate_forcings, generate_islands, make_island_depth
from project import DEFAULT_CONFIG, load_config, path_for
from swan_inputs import grid_coordinates, prepare_case, write_json, write_manifest
from coverage_generator import coverage_summary, positive_int, target_settings, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", action="append")
    parser.add_argument("--limit", type=int, help="Maximum cases per tile and profile")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target-train-cases", type=int, help="Effective training target per profile")
    parser.add_argument("--training-candidates", type=int, help="Total candidate prefix, including replacements")
    parser.add_argument("--plan-only", action="store_true", help="Write a compact plan/coverage report; no case files")
    args = parser.parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]
    if args.target_train_cases is not None or args.training_candidates is not None:
        if experiment.get("generator") != "broad-coverage-1":
            raise ValueError("Quota overrides require broad-coverage-1")
        sampling = experiment["case_sampling"]
        if args.target_train_cases is not None:
            sampling["target_effective_train_cases"] = positive_int(args.target_train_cases, "--target-train-cases")
        if args.training_candidates is not None:
            sampling["training_candidate_cases"] = positive_int(args.training_candidates, "--training-candidates")
    broad = experiment.get("generator") == "broad-coverage-1"
    if broad:
        # Use the actual solver frequency limits in coverage validation.
        experiment["spectral_grid"] = config["spectral_grid"]
        validate_config(experiment)
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
    snapshot = root / "experiment.json"
    if snapshot.is_file():
        import json
        previous = json.loads(snapshot.read_text(encoding="utf-8"))["experiment"]
        old_recipe, new_recipe = dict(previous), dict(experiment)
        # Candidate count can increase with a stable recipe/pair prefix.
        for recipe in (old_recipe, new_recipe):
            recipe["case_sampling"] = dict(recipe.get("case_sampling", {}))
            for name in ("target_effective_train_cases", "training_candidate_cases", "training_reserve_fraction"):
                recipe["case_sampling"].pop(name, None)
        if old_recipe != new_recipe:
            raise ValueError("Generator recipe changed; choose a fresh paths.cases directory")
    summaries = []
    records = []
    for tile in config["tiles"]:
        rectangle = Rectangle(west=tile["west"], south=tile["south"])
        islands = {item.terrain_id: item for item in generate_islands(experiment, rectangle)}
        cases = case_plan(list(islands.values()), list(forcings.values()), experiment)
        summaries.append({"tile_id": tile["id"], **coverage_summary(list(islands.values()), list(forcings.values()), cases)})
        if args.plan_only:
            continue
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
    if args.plan_only:
        destination = Path(config["_base"]) / "coverage_plan.json"
        write_json(destination, {"version": "2.1.1", "experiment": experiment, "profiles": profiles,
                                "per_tile": summaries, "target_per_profile": target_settings(experiment)[0] if broad else None,
                                "note": "planned candidates only; no SWAN or training labels"})
        print(f"Wrote plan only: {destination}")
        return
    if not records:
        raise ValueError("No cases selected")
    write_manifest(root / "manifest.csv", records)
    write_json(root / "experiment.json", {"schema_version": "island-matrix-experiment-2",
                                          "version": "2.1.1", "experiment": experiment,
                                          "coverage": summaries, "selected_in_this_call": len(records),
                                          "wave_shape": [3, 2, 2], "wind_shape": [2, 3, 3],
                                          "matrix_order": "channel, south-to-north, west-to-east"})
    print(f"Prepared {len(records)} cases; manifest: {root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
