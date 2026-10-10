"""Change only the terrain channel of a delivered v2 dataset; never run SWAN."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from forcing_arrays import terrain_feature
from training_core import validate_dataset, write_json, shard_path
from wave_geometry import RELATIVE_DEPTH_SCHEMA, recover_log_depth, feature_definition


def convert_dataset(source, destination, require_splits=True):
    source, destination = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
    metadata, rows, source_report = validate_dataset(source, require_splits=require_splits)
    if metadata["input_channels"][0] != "log_depth":
        raise ValueError("Migration input must be a baseline log_depth dataset")
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and output must be separate dataset directories")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Choose a fresh output directory; baseline data will be preserved")
    norm = copy.deepcopy(metadata["normalization"])
    norm.update(terrain_channel="relative_depth", relative_depth_gravity_mps2=9.81,
                zero_boundary_reference_period_s=8.0)
    destination.mkdir(parents=True, exist_ok=True)
    reconstructed, zero_boundary = 0, 0
    for row in rows:
        with np.load(shard_path(source, row["shard"]), allow_pickle=False) as shard:
            arrays = {name: shard[name].copy() for name in shard.files}
        ny, nx = map(int, arrays["raw_shape"])
        wet = arrays["mask"][0, :ny, :nx].astype(bool)
        if "depth_m" in arrays:
            depth = arrays["depth_m"]
        else:
            depth = recover_log_depth(arrays["x"][0, :ny, :nx], wet, metadata["normalization"])
            reconstructed += 1
        first, name = terrain_feature(arrays["wave"], depth, wet, norm)
        assert name == "relative_depth"
        arrays["x"][0] = 0
        arrays["x"][0, :ny, :nx] = first
        arrays["depth_m"] = np.where(wet, depth, 0).astype(np.float32)
        zero_boundary += int(not np.any(arrays["wave"][0] > 0))
        target = shard_path(destination, row["shard"])
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
    updated = copy.deepcopy(metadata)
    updated.update(schema_version=RELATIVE_DEPTH_SCHEMA, version="2.1", normalization=norm,
                   terrain_feature=feature_definition(norm))
    updated["input_channels"][0] = "relative_depth"
    updated["migration"] = {"source_sha256": source_report["sha256"],
                            "changed_arrays": ["x[0]"], "added_arrays": ["depth_m"],
                            "raw_depth_recovered_cases": reconstructed,
                            "depth_recovery_limit": "inverse log feature recovers min(h,cap); use raw cases for depths exceeding the old cap",
                            "zero_boundary_reference_cases": zero_boundary}
    (destination / "manifest.csv").write_bytes((source / "manifest.csv").read_bytes())
    write_json(destination / "metadata.json", updated)
    _, _, report = validate_dataset(destination, require_splits=require_splits)
    write_json(destination / "migration_check.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Baseline directory containing metadata.json and manifest.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete-splits", action="store_true")
    args = parser.parse_args()
    report = convert_dataset(args.data, args.output, require_splits=not args.allow_incomplete_splits)
    print(report)


if __name__ == "__main__":
    main()
