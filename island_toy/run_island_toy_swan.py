"""Check, prepare, run, and report circular-island SWAN cases."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from island_core import HERE, load_config, read_csv

sys.path.insert(0, str(HERE.parent))
from v2_core import load_config as load_parent_config, resolve_config_path  # noqa: E402

parent_hint = load_config()["_parent_path"]
parent_config_hint = load_parent_config(parent_hint)
SWAN_YEARLY = Path(parent_config_hint["_config_path"]).parents[2] / "SWAN_YEARLY"
sys.path.insert(0, str(SWAN_YEARLY))
import swan_batch as core  # noqa: E402


SIDE_TOKEN = {"NORTH": "N", "EAST": "E", "SOUTH": "S", "WEST": "W"}


def select_cases(path: Path, case_ids: list[str] | None, limit: int | None) -> list[dict[str, str]]:
    rows = read_csv(path)
    if case_ids:
        wanted = set(case_ids)
        rows = [row for row in rows if row["case_id"] in wanted]
        missing = wanted.difference(row["case_id"] for row in rows)
        if missing:
            raise ValueError(f"Unknown island toy cases: {', '.join(sorted(missing))}")
    return rows[:limit] if limit else rows


def context(config_path: Path, profile: str, case_ids: list[str] | None, limit: int | None):
    config = load_config(config_path)
    parent = load_parent_config(config["_parent_path"])
    if profile not in config["profiles"]:
        raise ValueError(f"Profile {profile} is not enabled in island config")
    terrains = {row["terrain_id"]: row for row in read_csv(HERE / "index" / "terrains.csv")}
    cases = select_cases(HERE / "index" / "cases.csv", case_ids, limit)
    if not cases:
        raise ValueError("No island toy cases selected")
    for row in cases:
        terrain_id = row["terrain_id"]
        terrain_dir = HERE / "terrains" / profile / terrain_id
        for filename in ("grid.nc", "bottom.dat", "grid_commands.swn", "metadata.json", "terrain.png", "terrain.npz"):
            if not (terrain_dir / filename).is_file():
                raise FileNotFoundError(terrain_dir / filename)
        if terrain_id not in terrains:
            raise ValueError(f"Case references unknown terrain: {terrain_id}")
        if row["boundary_side"] not in SIDE_TOKEN:
            raise ValueError(f"Invalid boundary side: {row['boundary_side']}")
        numeric = [float(row[name]) for name in (
            "hs_m", "tp_s", "wave_from_deg", "wind_u10_mps", "wind_v10_mps",
            "current_u_mps", "current_v_mps",
        )]
        if not np.isfinite(numeric).all() or numeric[0] <= 0 or numeric[1] <= 0:
            raise ValueError(f"Invalid forcing in {row['case_id']}")
    return config, parent, cases


def write_vector(path: Path, u: float, v: float) -> None:
    core.write_wind(path, np.full((2, 2), u, dtype=np.float32), np.full((2, 2), v, dtype=np.float32))


def render_input(case: dict[str, str], profile: str, config: dict, parent: dict, case_dir: Path) -> str:
    terrain_dir = HERE / "terrains" / profile / case["terrain_id"]
    commands = [line.strip() for line in (terrain_dir / "grid_commands.swn").read_text(encoding="ascii").splitlines() if line.strip()]
    bottom_relative = os.path.relpath(terrain_dir / "bottom.dat", case_dir).replace("\\", "/")
    commands = [
        re.sub(r"'[^']+'", f"'{bottom_relative}'", line, count=1) if line.upper().startswith("READINP BOTTOM") else line
        for line in commands
    ]
    metadata = json.loads((terrain_dir / "metadata.json").read_text(encoding="utf-8"))
    parent_tile = metadata  # bounds are fixed by grid command; extract origin from first command below.
    cgrid = next(line for line in commands if line.startswith("CGRID REG"))
    parts = cgrid.split()
    west, south = float(parts[2]), float(parts[3])
    gamma = float(config["forcing"]["jonswap_gamma"])
    spread = float(config["forcing"]["directional_spread_deg"])
    output = parent["output"]
    lines = [f"PROJ '{case['case_id'][:16]}' 'IT'", "MODE STATIONARY TWODIMENSIONAL", "SET NAUTICAL"]
    lines.extend(commands)
    lines.extend([
        f"INPGRID WIND {west:.10f} {south:.10f} 0 1 1 0.5000000000 0.5000000000",
        "READINP WIND 1 'wind.dat' 3 0 FREE",
        f"INPGRID CURRENT {west:.10f} {south:.10f} 0 1 1 0.5000000000 0.5000000000",
        "READINP CURRENT 1 'current.dat' 3 0 FREE",
        f"BOUND SHAPESPEC JONSWAP {gamma:.3f} PEAK DSPR DEGREES",
        f"BOUNDSPEC SIDE {SIDE_TOKEN[case['boundary_side']]} CCW CONSTANT PAR "
        f"{float(case['hs_m']):.6f} {float(case['tp_s']):.6f} "
        f"{float(case['wave_from_deg']):.6f} {spread:.3f}",
    ])
    lines.extend(config["physics_commands"])
    lines.extend([
        f"BLOCK 'COMPGRID' NOHEADER '{output['filename']}' LAYOUT {int(output['layout'])} "
        + " ".join(output["quantities"]),
        "COMPUTE", "STOP", "",
    ])
    return "\n".join(lines)


def prepare(config_path: Path, profile: str, case_ids: list[str] | None, limit: int | None, overwrite: bool) -> None:
    config, parent, cases = context(config_path, profile, case_ids, limit)
    root = HERE / "runs" / profile
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, case in enumerate(cases, start=1):
        case_dir = root / case["case_id"]
        if case_dir.exists() and not overwrite:
            status = core.read_status(case_dir).get("status", "prepared")
        else:
            if case_dir.exists():
                core.remove_direct_child(case_dir, root)
            case_dir.mkdir(parents=False)
            output = case_dir / parent["output"]["filename"]
            output.parent.mkdir(parents=True, exist_ok=True)
            core.atomic_write_text(case_dir / "INPUT", render_input(case, profile, config, parent, case_dir), encoding="ascii")
            core.atomic_write_text(case_dir / "swaninit", core.SWANINIT_TEXT, encoding="ascii")
            write_vector(case_dir / "wind.dat", float(case["wind_u10_mps"]), float(case["wind_v10_mps"]))
            write_vector(case_dir / "current.dat", float(case["current_u_mps"]), float(case["current_v_mps"]))
            core.atomic_write_json(case_dir / "forcing.json", {
                "schema_version": "island-toy-forcing-1", **case,
                "wind_storage": "uniform 2x2 u then v, south-to-north",
                "current_storage": "uniform 2x2 u then v, south-to-north",
                "coupling_scope": "prescribed diagnostic relation; not a circulation model",
            })
            core.atomic_write_json(case_dir / "run_status.json", {"schema_version": "island-toy-run-status-1", "status": "prepared"})
            status = "prepared"
        manifest.append({
            "case_id": case["case_id"], "terrain_id": case["terrain_id"], "forcing_id": case["forcing_id"],
            "split": case["split"], "generalization": case["generalization"],
            "case_directory": case["case_id"], "status": status,
        })
        if index % 50 == 0 or index == len(cases):
            print(f"prepared {index}/{len(cases)}", flush=True)
    core.atomic_write_csv(root / "manifest.csv", list(manifest[0]), manifest)


def run(config_path: Path, profile: str, case_ids: list[str] | None, limit: int | None, workers: int) -> int:
    config, parent, _ = context(config_path, profile, case_ids, limit)
    root = HERE / "runs" / profile
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path}; run prepare first")
    rows = read_csv(manifest_path)
    if case_ids:
        wanted = set(case_ids); rows = [row for row in rows if row["case_id"] in wanted]
    if limit:
        rows = rows[:limit]
    executable = resolve_config_path(parent, parent["paths"]["swan_executable"])
    expected = Path(parent["output"]["filename"])
    def one(row):
        return core.run_one_case(row, root, executable, expected, None)
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, row): row for row in rows}
        for future in as_completed(futures):
            result = future.result(); failed += result["status"] != "completed"
            print(result["case_id"], result["status"], result.get("runtime_seconds"), flush=True)
    return 1 if failed else 0


def status(profile: str) -> None:
    root = HERE / "runs" / profile
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        print({"not_prepared": 0}); return
    counts: dict[str, int] = {}
    for row in read_csv(manifest):
        value = core.read_status(root / row["case_id"]).get("status", "missing")
        counts[value] = counts.get(value, 0) + 1
    print(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--profile", default="1km")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare", "run"):
        item = sub.add_parser(name); item.add_argument("--case-id", action="append", dest="case_ids"); item.add_argument("--limit", type=int)
        if name == "prepare": item.add_argument("--overwrite", action="store_true")
        if name == "run": item.add_argument("--workers", type=int, default=1)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "check":
        config, _, cases = context(args.config, args.profile, args.case_ids, args.limit)
        print({"cases": len(cases), "profile": args.profile, "current_enabled": True,
               "single_side_wave_boundary": True, "tile_id": config["tile_id"]})
    elif args.command == "prepare":
        prepare(args.config, args.profile, args.case_ids, args.limit, args.overwrite)
    elif args.command == "run":
        raise SystemExit(run(args.config, args.profile, args.case_ids, args.limit, args.workers))
    else:
        status(args.profile)


if __name__ == "__main__":
    main()
