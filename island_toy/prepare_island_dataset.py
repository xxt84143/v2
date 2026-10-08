"""Convert completed island-toy SWAN Hs outputs to v2-compatible shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from island_core import HERE, SIDES, load_config, read_csv, write_csv

sys.path.insert(0, str(HERE.parent))
from v2_core import load_config as load_parent_config, padded_shape, read_swan_block  # noqa: E402


def pad(array: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    widths = [(0, 0)] * array.ndim
    widths[-2] = (0, target[0] - array.shape[-2]); widths[-1] = (0, target[1] - array.shape[-1])
    return np.pad(array, widths)


def inputs(case: dict[str, str], depth: np.ndarray, wet: np.ndarray, config: dict):
    norm = config["normalization"]
    yy, xx = np.meshgrid(np.linspace(-1, 1, depth.shape[0]), np.linspace(-1, 1, depth.shape[1]), indexing="ij")
    wave = np.deg2rad(float(case["wave_from_deg"]))
    values = [
        np.where(wet, depth / float(norm["depth_scale_m"]), 0.0), wet.astype(float), xx, yy,
        np.full(depth.shape, float(case["hs_m"]) / float(norm["hs_scale_m"])),
        np.full(depth.shape, float(case["tp_s"]) / float(norm["period_scale_s"])),
        np.full(depth.shape, np.sin(wave)), np.full(depth.shape, np.cos(wave)),
    ]
    names = ["depth", "wet_mask", "x", "y", "wave_hs", "wave_tp", "wave_dir_sin", "wave_dir_cos"]
    for side in SIDES:
        values.append(np.full(depth.shape, float(case["boundary_side"] == side)))
        names.append(f"boundary_{side.lower()}")
    for prefix, scale in (("wind", norm["wind_scale_mps"]), ("current", norm["current_scale_mps"])):
        for component in ("u", "v"):
            source = f"{prefix}_{component + ('10' if prefix == 'wind' else '')}_mps"
            values.append(np.full(depth.shape, float(case[source]) / float(scale)))
            names.append(source)
    return np.stack(values).astype(np.float32), names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--profile", default="1km")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config); parent = load_parent_config(config["_parent_path"])
    output = args.output or (HERE / "dataset" / args.profile)
    cases = {row["case_id"]: row for row in read_csv(HERE / "index" / "cases.csv")}
    run_manifest = HERE / "runs" / args.profile / "manifest.csv"
    if not run_manifest.is_file(): raise FileNotFoundError(run_manifest)
    rows = []; channel_names = None
    for run_row in read_csv(run_manifest):
        case = cases[run_row["case_id"]]
        result = HERE / "runs" / args.profile / case["case_id"] / parent["output"]["filename"]
        if not result.is_file(): continue
        with xr.open_dataset(HERE / "terrains" / args.profile / case["terrain_id"] / "grid.nc") as ds:
            depth = np.asarray(ds.depth.values, dtype=np.float32); wet = np.asarray(ds.wet_mask.values, dtype=bool)
        hs = read_swan_block(result, depth.shape, parent["output"]["quantities"], "HSIGN")
        valid = wet & np.isfinite(hs) & (hs >= 0)
        x, names = inputs(case, depth, wet, config); channel_names = names if channel_names is None else channel_names
        if names != channel_names: raise AssertionError("channel order changed")
        shape = padded_shape(depth.shape, int(parent["grid"]["model_pad_multiple"]))
        shard = output / f"{case['case_id']}.npz"; shard.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shard, x=pad(x, shape), y=pad((hs / float(config["normalization"]["hs_scale_m"]))[None], shape),
                            mask=pad(valid[None].astype(np.uint8), shape), raw_shape=np.asarray(depth.shape))
        rows.append({**{key: case[key] for key in ("case_id", "terrain_id", "forcing_id", "split", "generalization")},
                     "shard": shard.relative_to(output).as_posix(), "padded_ny": shape[0], "padded_nx": shape[1]})
    if not rows: raise RuntimeError("No completed island SWAN outputs")
    write_csv(output / "manifest.csv", rows)
    metadata = {"schema_version": "toy-v2-dataset-1", "profile": args.profile, "sample_count": len(rows),
                "input_channel_count": len(channel_names), "input_channels": channel_names,
                "padded_shape": [int(rows[0]["padded_ny"]), int(rows[0]["padded_nx"])],
                "normalization": config["normalization"], "generalization_column": "generalization"}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} island toy shards to {output}")


if __name__ == "__main__": main()
