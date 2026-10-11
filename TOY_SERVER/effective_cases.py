"""One shared definition of valid saved labels for runner and converter."""
from __future__ import annotations

import json
from pathlib import Path

from coverage_generator import target_settings
from swan_runtime import (SAVED_STATUSES, check_output, convergence,
                          fingerprint, inspect_case)


def experiment_settings(config, root):
    """Read the generator snapshot so CLI targets survive subsequent commands."""
    path = root / "experiment.json"
    experiment = config["experiment"]
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("experiment", {}).get("generator") == "broad-coverage-1":
            experiment = saved["experiment"]
    return experiment


def quota(config, root):
    experiment = experiment_settings(config, root)
    if experiment.get("generator") != "broad-coverage-1":
        return None, False
    target, _, _ = target_settings(experiment)
    return target, experiment["case_sampling"].get("require_converged", False)


def saved_case(root, row, require_converged=False):
    """Return validated metadata/status/metrics, or None for an ineligible case.

    Corrupt/mismatched completed cases raise rather than silently counting.
    Non-converged labels are eligible under the existing record_only policy
    unless require_converged was explicitly requested.
    """
    directory = (root / row["case_directory"]).resolve()
    if directory == root or not directory.is_relative_to(root):
        raise ValueError("Manifest case path escapes case root")
    status_path = directory / "run_status.json"
    if not status_path.is_file():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") not in SAVED_STATUSES:
        return None
    if status.get("return_code") != 0 or status.get("print_error"):
        raise ValueError(f"Saved case has a SWAN execution error: {directory}")
    metadata = inspect_case(directory)
    for name in ("sample_id", "case_id", "profile", "terrain_id", "forcing_id", "split", "generalization", "family"):
        if metadata.get(name) != row[name]:
            raise ValueError(f"Manifest {name} differs from saved case: {directory}")
    metrics = check_output(directory, metadata)
    if (status.get("input_signature") != fingerprint(metadata["input_hashes"])
            or status.get("output_sha256") != metrics["output_sha256"]):
        raise ValueError(f"Saved case changed: {directory}; rerun before counting/conversion")
    current = convergence(directory)
    if current["print_error"]:
        raise ValueError(f"Saved case has a SWAN log error: {directory}")
    status = {**status, **current}
    if require_converged and status.get("converged") is not True:
        return None
    return metadata, status, metrics


def select_effective(root, rows, target, require_converged=False, strict=True):
    """Cap training per profile at the target; evaluation remains uncapped."""
    selected, train_counts, rejected = [], {}, 0
    rows = sorted(rows, key=lambda r: (r["profile"], r["tile_id"], r["case_id"]))
    identities = set()
    for row in rows:
        pair = (row["profile"], row["tile_id"], row["terrain_id"], row["forcing_id"])
        if pair in identities:
            raise ValueError("Duplicate geometry/forcing pair in manifest")
        identities.add(pair)
        profile = row["profile"]
        if row["split"] == "train" and target is not None and train_counts.get(profile, 0) >= target:
            continue
        saved = saved_case(root, row, require_converged)
        if saved is None:
            rejected += 1
            continue
        selected.append((row, saved))
        if row["split"] == "train":
            train_counts[profile] = train_counts.get(profile, 0) + 1
    profiles = {row["profile"] for row in rows}
    if strict and target is not None:
        short = {p: train_counts.get(p, 0) for p in profiles if train_counts.get(p, 0) < target}
        if short:
            raise ValueError(f"Effective train quota not reached ({short}; target {target} per profile). "
                             "Prepare additional training candidates or rerun failed cases; "
                             "use --allow-partial only for an explicitly partial dataset.")
    return selected, train_counts, rejected
