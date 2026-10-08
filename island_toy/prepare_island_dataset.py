"""Convert completed island-toy SWAN Hs outputs to v2-compatible shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from island_core import HERE, load_config, read_csv, write_csv

sys.path.insert(0, str(HERE.parent))
from v2_core import load_config as load_parent_config, padded_shape, read_swan_block  # noqa: E402


def pad(array: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    widths = [(0, 0)] * array.ndim
    widths[-2] = (0, target[0] - array.shape[-2]); widths[-1] = (0, target[1] - array.shape[-1])
    return np.pad(array, widths)


def inputs(wave, wind, depth, wet, config):
    from forcing_arrays import model_inputs
    return model_inputs(wave, wind, depth, wet, config["normalization"], period_name="tp")


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
        status_path = result.parent.parent / "run_status.json"
        if not result.is_file() or not status_path.is_file(): continue
        if json.loads(status_path.read_text(encoding="utf-8"))["status"] != "completed": continue
        from swan_runtime import inspect_case, check_output, fingerprint
        directory = result.parent.parent
        metadata = inspect_case(directory)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        metrics = check_output(directory, metadata)
        if status.get("input_signature") != fingerprint(metadata["input_hashes"]) or status.get("output_sha256") != metrics["output_sha256"]:
            raise ValueError(f"Completed case changed: {directory}; rerun before conversion")
        with np.load(directory / "inputs.npz", allow_pickle=False) as data:
            depth, wet = data["depth_m"], data["wet_mask"].astype(bool)
            wave, wind = data["wave"], data["wind"]
        hs = read_swan_block(result, depth.shape, metadata["output_quantities"], "HSIGN")
        valid = wet & np.isfinite(hs) & (hs >= 0)
        x, names = inputs(wave, wind, depth, wet, config); channel_names = names if channel_names is None else channel_names
        if names != channel_names: raise AssertionError("channel order changed")
        shape = padded_shape(depth.shape, int(parent["grid"]["model_pad_multiple"]))
        shard = output / f"{case['case_id']}.npz"; shard.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shard, wave=wave, wind=wind, x=pad(x, shape), y=pad((np.where(valid, hs, 0) / float(config["normalization"]["hs_scale_m"]))[None], shape),
                            mask=pad(valid[None].astype(np.uint8), shape), raw_shape=np.asarray(depth.shape))
        rows.append({**{key: case[key] for key in ("case_id", "terrain_id", "forcing_id", "split", "generalization")},
                     "shard": shard.relative_to(output).as_posix(), "padded_ny": shape[0], "padded_nx": shape[1]})
    if not rows: raise RuntimeError("No completed island SWAN outputs")
    write_csv(output / "manifest.csv", rows)
    metadata = {"schema_version": "toy-v2-dataset-matrix-2", "profile": args.profile, "sample_count": len(rows),
                "input_channel_count": len(channel_names), "input_channels": channel_names,
                "padded_shape": [int(rows[0]["padded_ny"]), int(rows[0]["padded_nx"])],
                "normalization": config["normalization"], "generalization_column": "generalization"}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} island toy shards to {output}")


if __name__ == "__main__": main()
