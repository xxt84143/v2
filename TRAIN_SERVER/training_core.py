"""Configuration and dataset contracts shared by independent training entry points."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from forcing_arrays import terrain_feature
from wave_geometry import BASELINE_SCHEMA, RELATIVE_DEPTH_SCHEMA, feature_definition

HERE = Path(__file__).resolve().parent
CHANNELS = ["log_depth", "wet_mask", "x", "y", "wave_hs", "wave_tp",
            "wave_dir_sin", "wave_dir_cos", "wind_u10", "wind_v10"]
RELATIVE_DEPTH_CHANNELS = ["relative_depth", *CHANNELS[1:]]
DATASET_CHANNELS = {BASELINE_SCHEMA: CHANNELS, RELATIVE_DEPTH_SCHEMA: RELATIVE_DEPTH_CHANNELS}
SPLITS = ("train", "validation", "test")


def load_config(path=HERE / "config.json"):
    path = Path(path).expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "toy-training-1":
        raise ValueError("Expected schema_version=toy-training-1")
    value["_config_path"] = str(path)
    return value


def resolve_config_path(config, value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(config["_config_path"]).parent / path).resolve()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def shard_path(root, name):
    # Archives and manifests always use portable POSIX relative paths.
    relative = Path(name)
    root = Path(root).resolve()
    destination = (root / relative).resolve()
    if "\\" in name or relative.is_absolute() or not destination.is_relative_to(root) or destination == root:
        raise ValueError(f"Shard path escapes the dataset or is not portable: {name}")
    return destination


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_dataset(root, require_splits=True):
    """Inspect every shard; convergence is counted and never used as an exclusion."""
    root = Path(root).expanduser().resolve()
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    rows = read_csv(root / "manifest.csv")
    expected_channels = DATASET_CHANNELS.get(metadata.get("schema_version"))
    if expected_channels is None:
        raise ValueError("Regenerate the dataset with the current TOY_SERVER converter")
    relative_terrain = metadata["schema_version"] == RELATIVE_DEPTH_SCHEMA
    if relative_terrain and metadata.get("terrain_feature") != feature_definition(metadata["normalization"]):
        raise ValueError("Relative-depth feature definition does not match normalization")
    if relative_terrain and metadata.get("normalization", {}).get("terrain_channel") != "relative_depth":
        raise ValueError("Relative-depth schema requires terrain_channel=relative_depth")
    if metadata.get("input_channels") != expected_channels or metadata.get("input_channel_count") != 10:
        raise ValueError("Expected the current ten-channel, nine-wind-node / four-wave-node contract")
    if metadata.get("target") != "SWAN HSIGN / hs_scale_m":
        raise ValueError("This package currently trains normalized Hs only")
    shape = tuple(metadata["padded_shape"])
    if len(shape) != 2 or any(not isinstance(size, int) or size < 16 or size % 16 for size in shape):
        raise ValueError("U-Net spatial dimensions must be positive multiples of 16")
    if metadata.get("sample_count") != len(rows) or not rows:
        raise ValueError("metadata.sample_count does not match a nonempty manifest")
    for name in ("hs_scale_m", "period_scale_s", "wind_scale_mps"):
        scale = float(metadata["normalization"][name])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid normalization scale: {name}")
    required = {"sample_id", "case_id", "terrain_id", "forcing_id", "split", "generalization", "shard"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing manifest columns: {sorted(required.difference(rows[0]))}")
    counts = Counter(row["split"] for row in rows)
    if set(counts).difference(SPLITS):
        raise ValueError(f"Unknown splits: {counts}")
    if require_splits and any(not counts[name] for name in SPLITS):
        raise ValueError(f"Training requires train, validation and test samples; found {dict(counts)}")
    for field in ("sample_id", "shard"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate {field}: a sample may not occur in several splits")
    pairs = [(str(Path(row["shard"]).parent), row["terrain_id"], row["forcing_id"]) for row in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Repeated island/forcing pairs across dataset rows")
    for row in rows:
        allowed = {"seen_island_seen_forcing"} if row["split"] == "train" else {"new_island", "new_forcing", "new_both"}
        if row["generalization"] not in allowed:
            raise ValueError(f"Invalid generalization group for {row['split']}: {row['generalization']}")
    train = [row for row in rows if row["split"] == "train"]
    def identifiers(items, field):
        return {(str(Path(row["shard"]).parent), row[field]) for row in items}
    novel = {}
    for split in ("validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        for field, groups in (("terrain_id", {"new_island", "new_both"}),
                              ("forcing_id", {"new_forcing", "new_both"})):
            values = identifiers([row for row in selected if row["generalization"] in groups], field)
            if values & identifiers(train, field):
                raise ValueError(f"{split} {field} declared new is present in training")
            novel[split, field] = values
    for field in ("terrain_id", "forcing_id"):
        if novel["validation", field] & novel["test", field]:
            raise ValueError(f"Validation and test share held-out {field}")
    signature = hashlib.sha256()
    for name in ("metadata.json", "manifest.csv"):
        signature.update(name.encode())
        signature.update(digest_file(root / name).encode())
    wet_pixels = Counter()
    convergence = Counter()
    for row in rows:
        path = shard_path(root, row["shard"])
        with np.load(path, allow_pickle=False) as shard:
            arrays = {name: shard[name] for name in ("x", "y", "mask", "raw_shape", "wave", "wind")}
            if relative_terrain:
                arrays["depth_m"] = shard["depth_m"]
        x, y, mask = (arrays[name] for name in ("x", "y", "mask"))
        if x.shape != (10, *shape) or y.shape != (1, *shape) or mask.shape != (1, *shape):
            raise ValueError(f"{path}: incorrect x/y/mask shape")
        if x.dtype != np.float32 or y.dtype != np.float32:
            raise ValueError(f"{path}: x and y must be float32")
        if not all(np.isfinite(arrays[name]).all() for name in ("x", "y", "wave", "wind")):
            raise ValueError(f"{path}: nonfinite values")
        if not np.isin(mask, [0, 1]).all() or not mask.any() or not np.array_equal(x[1], mask[0]):
            raise ValueError(f"{path}: wet mask is empty, nonbinary or inconsistent with input")
        if np.any(y < 0) or np.any(y[mask == 0] != 0):
            raise ValueError(f"{path}: negative Hs or nonzero target outside wet mask")
        raw = arrays["raw_shape"]
        if raw.shape != (2,) or raw.dtype.kind not in "iu":
            raise ValueError(f"{path}: invalid raw_shape")
        ny, nx = map(int, raw)
        if not (0 < ny <= shape[0] and 0 < nx <= shape[1]):
            raise ValueError(f"{path}: raw_shape exceeds padded_shape")
        for array in (x, y, mask):
            if np.any(array[..., ny:, :]) or np.any(array[..., :, nx:]):
                raise ValueError(f"{path}: padding must be zero and excluded from loss")
        wave, wind = arrays["wave"], arrays["wind"]
        if wave.shape != (3, 2, 2) or wind.shape != (2, 3, 3):
            raise ValueError(f"{path}: wrong raw wave/wind matrix shapes")
        if np.any(wave[0] < 0) or np.any(wave[1] <= 0) or np.any((wave[2] < 0) | (wave[2] >= 360)):
            raise ValueError(f"{path}: invalid boundary Hs/period/direction")
        if metadata.get("profile") == "gebco15s" and (ny, nx) != (121, 121):
            raise ValueError(f"{path}: 15 arc-sec over 0.5 degrees requires 121x121 nodes")
        if relative_terrain:
            depth = arrays["depth_m"]
            wet = mask[0, :ny, :nx].astype(bool)
            if depth.shape != (ny, nx) or not np.isfinite(depth).all() or np.any(depth[wet] <= 0) or np.any(depth[~wet] != 0):
                raise ValueError(f"{path}: invalid raw depth field")
            expected, _ = terrain_feature(wave, depth, wet, metadata["normalization"])
            if not np.allclose(x[0, :ny, :nx], expected, rtol=2e-6, atol=2e-7):
                raise ValueError(f"{path}: terrain channel does not match finite-depth dispersion")
        wet_pixels[row["split"]] += int(mask.sum())
        convergence[row.get("converged", "").lower() or "unknown"] += 1
        signature.update(row["shard"].encode())
        signature.update(digest_file(path).encode())
    report = {"dataset": str(root), "profile": metadata["profile"], "samples": len(rows),
              "splits": {name: counts[name] for name in SPLITS}, "padded_shape": list(shape),
              "input_channels": expected_channels, "terrain_channel": expected_channels[0], "target": "Hs", "wet_pixels": dict(wet_pixels),
              "convergence": dict(convergence), "convergence_policy": "record_only",
              "sha256": signature.hexdigest()}
    return metadata, rows, report
