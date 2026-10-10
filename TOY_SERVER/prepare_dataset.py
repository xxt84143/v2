"""Convert completed self-contained SWAN cases to Hs training shards."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from forcing_arrays import model_inputs
from project import DEFAULT_CONFIG, load_config, path_for
from swan_inputs import write_json
from swan_runtime import CONVERGENCE_FIELDS, SAVED_STATUSES, check_output, fingerprint, inspect_case


def pad(array, shape):
    widths = [(0, 0)] * array.ndim
    widths[-2:] = [(0, shape[0] - array.shape[-2]), (0, shape[1] - array.shape[-1])]
    return np.pad(array, widths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", help="Use the sole experiment profile by default")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.profile is None:
        profiles = config["experiment"]["profiles"]
        if len(profiles) != 1:
            parser.error("Specify --profile when the experiment has multiple profiles")
        args.profile = profiles[0]
    root = path_for(config, "cases")
    output = (args.output or Path(config["_base"]) / "dataset" / args.profile).resolve()
    if output.exists() and (output / "manifest.csv").exists():
        raise FileExistsError("Choose a fresh output directory to avoid mixing dataset versions")
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["profile"] == args.profile]
    norm = config["experiment"]["normalization"]
    converted, shapes, channel_names = [], set(), None
    for row in rows:
        directory = (root / row["case_directory"]).resolve()
        if not directory.is_relative_to(root) or directory == root:
            raise ValueError("Manifest path escapes case root")
        status = json.loads((directory / "run_status.json").read_text(encoding="utf-8"))
        if status.get("status") not in SAVED_STATUSES:
            continue
        if status.get("return_code") != 0 or status.get("print_error"):
            raise ValueError(f"Saved case has a SWAN execution error: {directory}")
        metadata = inspect_case(directory)
        metrics = check_output(directory, metadata)
        if status.get("input_signature") != fingerprint(metadata["input_hashes"]) or status.get("output_sha256") != metrics["output_sha256"]:
            raise ValueError(f"Completed case changed: {directory}; rerun before conversion")
        with np.load(directory / "inputs.npz", allow_pickle=False) as data:
            wave, wind = data["wave"], data["wind"]
            depth, wet = data["depth_m"], data["wet_mask"].astype(bool)
        quantities = [component for name in metadata["output_quantities"]
                      for component in (["WIND_X", "WIND_Y"] if name == "WIND" else [name])]
        fields = np.fromstring((directory / "output/compgrid.tab").read_text(encoding="ascii"), sep=" ").reshape(len(quantities), *depth.shape)
        hs = fields[quantities.index("HSIGN")]
        x, names = model_inputs(wave, wind, depth, wet, norm, period_name="tp")
        if channel_names is not None and channel_names != names:
            raise ValueError("Model channel order changed")
        channel_names = names
        shape = tuple((size + 15) // 16 * 16 for size in depth.shape)
        shapes.add(shape)
        shard = Path(row["tile_id"]) / f"{row['case_id']}.npz"
        destination = output / shard
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, wave=wave, wind=wind, x=pad(x, shape),
                            y=pad((np.where(wet, hs, 0)[None] / norm["hs_scale_m"]).astype(np.float32), shape),
                            mask=pad(wet[None].astype(np.uint8), shape), raw_shape=np.asarray(depth.shape))
        converted.append({"sample_id": row["sample_id"], "case_id": row["case_id"],
                          "terrain_id": row["terrain_id"], "forcing_id": row["forcing_id"],
                          "split": row["split"], "generalization": row["generalization"],
                          "shard": shard.as_posix(), "padded_ny": shape[0], "padded_nx": shape[1],
                          **{name: status.get(name) for name in CONVERGENCE_FIELDS}})
    if not converted:
        raise ValueError("No completed cases available")
    if len(shapes) != 1:
        raise ValueError("A dataset must use one padded grid shape")
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(converted[0]))
        writer.writeheader()
        writer.writerows(converted)
    write_json(output / "metadata.json", {"schema_version": "toy-v2-dataset-matrix-2", "profile": args.profile,
                                           "sample_count": len(converted), "input_channel_count": len(channel_names),
                                           "input_channels": channel_names, "padded_shape": list(next(iter(shapes))),
                                           "normalization": norm, "generalization_column": "generalization",
                                           "target": "SWAN HSIGN / hs_scale_m", "convergence_policy": "record_only"})
    print(f"Converted {len(converted)} completed cases to {output}")


if __name__ == "__main__":
    main()
