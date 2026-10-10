"""Prepare self-contained island cases; run them only when explicitly requested."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import xarray as xr

from island_core import HERE, load_config, read_csv, runtime_config

sys.path.insert(0, str(HERE.parent))
from forcing_arrays import load
from swan_inputs import prepare_case, render_input as render_arrays, write_manifest
from swan_runtime import inspect_case, run_case
from v2_core import load_config as load_parent_config, resolve_config_path


def context(config_path, profile, case_ids=None, limit=None):
    config = load_config(config_path)
    parent = load_parent_config(config["_parent_path"])
    if profile not in config["profiles"]:
        raise ValueError("Profile not enabled in island config")
    cases = read_csv(HERE / "index" / "cases.csv")
    if case_ids:
        missing = set(case_ids) - {case["case_id"] for case in cases}
        if missing:
            raise ValueError(f"Unknown cases: {missing}")
        cases = [case for case in cases if case["case_id"] in case_ids]
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    cases = cases[:limit]
    if not cases:
        raise ValueError("No cases selected")
    return config, parent, cases


def case_arrays(case, profile):
    wave, wind = load(HERE / "index" / case["forcing_file"])
    with xr.open_dataset(HERE / "terrains" / profile / case["terrain_id"] / "grid.nc") as ds:
        return wave, wind, ds.lon.values.copy(), ds.lat.values.copy(), ds.depth.values.copy(), ds.wet_mask.values.astype(bool)


def render_input(case, profile, config, parent, case_dir=None):
    wave, wind, lon, lat, _, _ = case_arrays(case, profile)
    return render_arrays(case["case_id"], lon, lat, wave, wind, runtime_config(config, parent))


def prepare(config_path, profile, case_ids=None, limit=None, overwrite=False):
    config, parent, cases = context(config_path, profile, case_ids, limit)
    settings = runtime_config(config, parent)
    root = HERE / "runs" / profile
    rows = []
    for case in cases:
        wave, wind, lon, lat, depth, wet = case_arrays(case, profile)
        sample_id = f"{profile}__{config['tile_id']}__{case['case_id']}"
        tile = {"id": config["tile_id"], **config["tile"]}
        prepare_case(root / case["case_id"], sample_id, case, profile, tile, lon, lat, depth, wet,
                     wave, wind, settings, overwrite)
        rows.append({"sample_id": sample_id, "case_id": case["case_id"], "profile": profile,
                     "tile_id": tile["id"], "terrain_id": case["terrain_id"], "forcing_id": case["forcing_id"],
                     "split": case["split"], "generalization": case["generalization"],
                     "family": case["family"], "case_directory": case["case_id"]})
    write_manifest(root / "manifest.csv", rows, overwrite_legacy=overwrite)
    print(f"Prepared {len(rows)} cases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--profile", help="Use the sole configured profile by default")
    parser.add_argument("command", choices=("check", "prepare", "run", "status"))
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--swan")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.profile is None:
        profiles = load_config(args.config)["profiles"]
        if len(profiles) != 1:
            parser.error("Specify --profile when multiple profiles are configured")
        args.profile = profiles[0]
    if args.command == "prepare":
        prepare(args.config, args.profile, args.case_id, args.limit, args.overwrite)
        return
    root = HERE / "runs" / args.profile
    if args.command == "check":
        config, parent, cases = context(args.config, args.profile, args.case_id, args.limit)
        for case in cases:
            render_input(case, args.profile, config, parent)
        print(f"Matrix and terrain contracts checked: {len(cases)} cases; SWAN was not run")
        return
    rows = read_csv(root / "manifest.csv")
    if args.case_id:
        missing = set(args.case_id) - {row["case_id"] for row in rows}
        if missing:
            raise ValueError(f"Unknown cases: {missing}")
        rows = [row for row in rows if row["case_id"] in args.case_id]
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    rows = rows[:args.limit]
    if args.command == "status":
        counts = {}
        for row in rows:
            value = json.loads((root / row["case_directory"] / "run_status.json").read_text(encoding="utf-8"))["status"]
            counts[value] = counts.get(value, 0) + 1
        print(counts)
        return
    config = load_config(args.config)
    parent = load_parent_config(config["_parent_path"])
    candidate = resolve_config_path(parent, args.swan or parent["paths"]["swan_executable"])
    found = str(candidate) if candidate.is_file() else shutil.which(args.swan or parent["paths"]["swan_executable"])
    if not found:
        raise FileNotFoundError("Set --swan or parent paths.swan_executable to the SWAN calculation program")
    if not rows or args.workers <= 0 or args.timeout_seconds <= 0:
        raise ValueError("Select cases and use positive workers/timeout")
    directories = [(root / row["case_directory"]).resolve() for row in rows]
    for directory in directories:
        if not directory.is_relative_to(root.resolve()):
            raise ValueError("Manifest path escapes case root")
        inspect_case(directory)
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, directory, Path(found), args.timeout_seconds, 1, args.rerun) for directory in directories]
        for future in as_completed(futures):
            result = future.result()
            failed += result["status"] not in ("completed", "skipped")
            print(json.dumps(result, ensure_ascii=False), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
