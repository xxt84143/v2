"""Convert completed rectangular SWAN runs to padded, model-ready NPZ shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from v2_core import (
    CORNER_NAMES, HERE, WIND_NAMES, load_config, padded_shape, read_csv,
    read_swan_block, write_csv,
)


def dense_input(row: dict[str, str], depth: np.ndarray, wet: np.ndarray, config: dict) -> tuple[np.ndarray, list[str]]:
    norm = config["normalization"]
    depth_channel = np.log1p(np.minimum(np.maximum(depth, 0.0), float(norm["depth_cap_m"]))) / np.log1p(float(norm["depth_cap_m"]))
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, depth.shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, depth.shape[1], dtype=np.float32), indexing="ij",
    )
    channels = [depth_channel.astype(np.float32), wet.astype(np.float32), xx, yy]
    names = ["log_depth", "wet_mask", "x_normalized", "y_normalized"]
    for corner in CORNER_NAMES:
        angle = np.deg2rad(float(row[f"wave_{corner}_mwd_deg"]))
        values = (
            float(row[f"wave_{corner}_hs_m"]) / float(norm["hs_scale_m"]),
            float(row[f"wave_{corner}_mwp_s"]) / float(norm["period_scale_s"]),
            float(np.sin(angle)), float(np.cos(angle)),
        )
        for suffix, value in zip(("hs", "period", "dir_sin", "dir_cos"), values):
            channels.append(np.full(depth.shape, value, dtype=np.float32))
            names.append(f"wave_{corner}_{suffix}")
    for point in WIND_NAMES:
        for component in ("u10", "v10"):
            value = float(row[f"wind_{point}_{component}"]) / float(norm["wind_scale_mps"])
            channels.append(np.full(depth.shape, value, dtype=np.float32))
            names.append(f"wind_{point}_{component}")
    return np.stack(channels), names


def pad(array: np.ndarray, target: tuple[int, int], value: float = 0.0) -> np.ndarray:
    dh, dw = target[0] - array.shape[-2], target[1] - array.shape[-1]
    if dh < 0 or dw < 0:
        raise ValueError("target padding shape is smaller than the array")
    widths = [(0, 0)] * array.ndim
    widths[-2] = (0, dh)
    widths[-1] = (0, dw)
    return np.pad(array, widths, constant_values=value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--profile", choices=("gebco15s", "1km"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or (HERE / "dataset" / args.profile)
    index_rows = read_csv(HERE / "index" / "cases.csv")
    by_sample = {row["sample_id"]: row for row in index_rows}
    candidates = []
    for tile_id in config["tiles"]:
        manifest_path = HERE / "runs" / args.profile / tile_id / "manifest.csv"
        if not manifest_path.is_file():
            continue
        for run_row in read_csv(manifest_path):
            index_row = by_sample.get(run_row["sample_id"])
            if index_row is None:
                raise ValueError(f"Run sample is absent from the v2 index: {run_row['sample_id']}")
            result = HERE / "runs" / args.profile / tile_id / run_row["case_id"] / config["output"]["filename"]
            if result.is_file():
                candidates.append((tile_id, run_row, index_row, result))
    if args.limit:
        candidates = candidates[:args.limit]
    if not candidates:
        raise RuntimeError("No completed v2 SWAN outputs were found")
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    channel_names = None
    multiple = int(config["grid"]["model_pad_multiple"])
    for number, (tile_id, run_row, row, result) in enumerate(candidates, start=1):
        grid_path = HERE / "grids" / args.profile / tile_id / "grid.nc"
        with xr.open_dataset(grid_path) as ds:
            depth = np.asarray(ds.depth.values, dtype=np.float32)
            wet = np.asarray(ds.wet_mask.values, dtype=bool)
        hs = read_swan_block(result, depth.shape, list(config["output"]["quantities"]), "HSIGN")
        valid = wet & np.isfinite(hs) & (hs >= 0.0)
        x, names = dense_input(row, depth, wet, config)
        if channel_names is None:
            channel_names = names
        elif names != channel_names:
            raise AssertionError("v2 channel order changed within one conversion")
        shape = padded_shape(depth.shape, multiple)
        target = hs[None] / float(config["normalization"]["hs_scale_m"])
        shard = output / tile_id / f"{row['case_id']}.npz"
        shard.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            shard, x=pad(x, shape), y=pad(target.astype(np.float32), shape),
            mask=pad(valid[None].astype(np.uint8), shape), raw_shape=np.asarray(depth.shape),
        )
        manifest.append({
            "sample_id": row["sample_id"], "case_id": row["case_id"], "tile_id": tile_id,
            "time": row["time"], "split": row["split"], "profile": args.profile,
            "shard": shard.relative_to(output).as_posix(), "raw_ny": depth.shape[0],
            "raw_nx": depth.shape[1], "padded_ny": shape[0], "padded_nx": shape[1],
            "valid_target_nodes": int(valid.sum()),
        })
        print(f"converted {number}/{len(candidates)} {row['sample_id']}", flush=True)
    padded_shapes = {(row["padded_ny"], row["padded_nx"]) for row in manifest}
    if len(padded_shapes) != 1:
        raise ValueError(f"Profile {args.profile} produced incompatible padded shapes: {sorted(padded_shapes)}")
    write_csv(output / "manifest.csv", manifest)
    metadata = {
        "schema_version": "toy-v2-dataset-1", "profile": args.profile,
        "sample_count": len(manifest), "input_channel_count": len(channel_names),
        "input_channels": channel_names, "target": "SWAN HSIGN / hs_scale_m",
        "padded_shape": list(next(iter(padded_shapes))), "pad_policy": "north/east zero pad to model_pad_multiple",
        "mask_policy": "GEBCO wet and finite non-negative SWAN HSIGN; padded cells excluded",
        "normalization": config["normalization"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(manifest)} shards with {len(channel_names)} channels to {output}")


if __name__ == "__main__":
    main()
