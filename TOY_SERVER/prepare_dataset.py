"""Convert completed self-contained SWAN cases to Hs training shards."""

import _bootstrap  # locate shared contracts
import argparse
import csv
from pathlib import Path

import numpy as np

from shared.wave_features import model_inputs
from shared.wave_features import dataset_schema, feature_definition
from project import DEFAULT_CONFIG, load_config, path_for
from shared.io_utils import write_json
from swan_runtime import CONVERGENCE_FIELDS
from effective_cases import experiment_settings, quota, select_effective


def pad(array, shape):
    widths = [(0, 0)] * array.ndim
    widths[-2:] = [(0, shape[0] - array.shape[-2]), (0, shape[1] - array.shape[-1])]
    return np.pad(array, widths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", help="Use the sole experiment profile by default")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly export fewer valid training labels than target")
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
    experiment = experiment_settings(config, root)
    norm = experiment["normalization"]
    target, require_converged = quota(config, root)
    # Preflight every selected label before creating shards or a dataset manifest.
    selected, train_counts, rejected = select_effective(root, rows, target, require_converged,
                                                       strict=not args.allow_partial)
    converted, shapes, channel_names = [], set(), None
    for row, (metadata, status, metrics) in selected:
        directory = (root / row["case_directory"]).resolve()
        if not directory.is_relative_to(root) or directory == root:
            raise ValueError("Manifest path escapes case root")
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
        np.savez_compressed(destination, wave=wave, wind=wind, depth_m=np.where(wet, depth, 0).astype(np.float32), x=pad(x, shape),
                            y=pad((np.where(wet, hs, 0)[None] / norm["hs_scale_m"]).astype(np.float32), shape),
                            mask=pad(wet[None].astype(np.uint8), shape), raw_shape=np.asarray(depth.shape))
        converted.append({"sample_id": row["sample_id"], "case_id": row["case_id"],
                          "terrain_id": row["terrain_id"], "forcing_id": row["forcing_id"],
                          "split": row["split"], "generalization": row["generalization"],
                          "family": row["family"], "terrain_shape": metadata.get("terrain_shape", "circle"),
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
    write_json(output / "metadata.json", {"schema_version": dataset_schema(norm), "version": "2.1.1", "terrain_feature": feature_definition(norm), "profile": args.profile,
                                           "sample_count": len(converted), "input_channel_count": len(channel_names),
                                           "input_channels": channel_names, "padded_shape": list(next(iter(shapes))),
                                           "normalization": norm, "generalization_column": "generalization",
                                           "target": "SWAN HSIGN / hs_scale_m", "convergence_policy": "required" if require_converged else "record_only",
                                           "generator": experiment.get("generator", "legacy"),
                                           "target_effective_train_cases": target, "effective_train_count": train_counts.get(args.profile, 0),
                                           "training_target_reached": target is None or train_counts.get(args.profile, 0) >= target,
                                           "partial_export_requested": args.allow_partial, "ineligible_case_count": rejected})
    print(f"Converted {len(converted)} completed cases to {output}")


if __name__ == "__main__":
    main()
