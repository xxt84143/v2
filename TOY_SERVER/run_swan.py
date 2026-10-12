"""Run prepared cases, validate output and record convergence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


from project import DEFAULT_CONFIG, load_config, path_for


from swan_runtime import inspect_case, run_case
from effective_cases import quota, saved_case
from shared.io_utils import write_json


def run_with_quota(root, rows, executable, timeout, threads, workers, rerun,
                   executable_sha256, target, require_converged):
    """Bound parallel work to the remaining quota; continue past failed candidates."""
    counts, attempted, failures, evaluation_failures = {}, 0, 0, 0
    profiles = sorted({row["profile"] for row in rows})
    rows = sorted(rows, key=lambda r: (r["profile"], r["tile_id"], r["case_id"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for profile in profiles:
            training = [r for r in rows if r["profile"] == profile and r["split"] == "train"]
            evaluation = [r for r in rows if r["profile"] == profile and r["split"] != "train"]
            counts[profile] = 0
            cursor = 0
            while cursor < len(training) and counts[profile] < target:
                size = min(workers, target-counts[profile])
                batch = training[cursor:cursor+size]
                cursor += len(batch)
                futures = {pool.submit(run_case, root / r["case_directory"], executable, timeout,
                                       threads, rerun, executable_sha256): r for r in batch}
                for future in as_completed(futures):
                    row = futures[future]
                    attempted += 1
                    try:
                        result = future.result()
                        accepted = saved_case(root, row, require_converged) is not None
                    except Exception as exc:
                        result = {"sample_id": row["sample_id"], "status": "failed", "error": str(exc)}
                        accepted = False
                    result["effective_train_label"] = accepted
                    counts[profile] += int(accepted)
                    failures += int(not accepted)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
            # Evaluation uses its fixed candidate list and does not consume training quota.
            for start in range(0, len(evaluation), workers):
                futures = {pool.submit(run_case, root / r["case_directory"], executable, timeout,
                                       threads, rerun, executable_sha256): r
                           for r in evaluation[start:start+workers]}
                for future in as_completed(futures):
                    row = futures[future]
                    attempted += 1
                    try:
                        result = future.result()
                        accepted = saved_case(root, row, require_converged) is not None
                    except Exception as exc:
                        result = {"sample_id": row["sample_id"], "status": "failed", "error": str(exc)}
                        accepted = False
                    failures += int(not accepted)
                    evaluation_failures += int(not accepted)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
    requested = {r["profile"] for r in rows if r["split"] == "train"}
    reached = all(counts[p] >= target for p in requested) if requested else None
    report = {"version": "2.1.1", "target_effective_train_cases_per_profile": target,
              "effective_train_counts_in_selection": counts, "training_target_reached": reached,
              "attempted_or_reused": attempted, "ineligible_attempts": failures,
              "require_converged": require_converged, "scope": "selected manifest rows"}
    write_json(root / "quota_status.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if reached is False:
        print("Insufficient valid training labels. Increase --training-candidates with the same "
              "generator recipe, or rerun failed cases; no effective target was claimed.")
    # Replacement failures do not fail a run that reached target; failed evaluation does.
    return 0 if reached is not False and not evaluation_failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, help="Prepared case root; no ERA5 or GEBCO needed")
    parser.add_argument("--swan", help="Executable path or program name on PATH")
    parser.add_argument("--profile", action="append")
    parser.add_argument("--tile", action="append")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--limit", type=int, help="Maximum selected cases (across profiles/tiles)")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--check", action="store_true", help="Check inputs and executable without running SWAN")
    parser.add_argument("--rerun", action="store_true", help="Recompute completed cases")
    args = parser.parse_args()
    config = load_config(args.config)
    root = (args.cases or path_for(config, "cases")).expanduser().resolve()
    executable_text = args.swan or config["paths"]["swan_executable"]
    candidate = Path(executable_text).expanduser()
    candidate = candidate if candidate.is_absolute() else Path(config["_base"]) / candidate
    found = str(candidate) if candidate.is_file() else shutil.which(executable_text)
    if not found:
        raise FileNotFoundError(f"SWAN executable not found: {executable_text}; set paths.swan_executable or --swan")
    executable = Path(found).resolve()
    if os.name != "nt" and not os.access(executable, os.X_OK):
        raise ValueError(f"SWAN file is not executable: {executable}")
    workers = args.workers if args.workers is not None else config["run"]["workers"]
    timeout = args.timeout_seconds if args.timeout_seconds is not None else config["run"]["timeout_seconds"]
    threads = config["run"].get("threads_per_case", 1)
    if workers <= 0 or timeout <= 0 or threads <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("workers, timeout, threads_per_case and limit must be positive")
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for column, wanted in (("profile", args.profile), ("tile_id", args.tile), ("case_id", args.case_id)):
        if wanted:
            missing = set(wanted) - {row[column] for row in rows}
            if missing:
                raise ValueError(f"Unknown {column}: {sorted(missing)}")
            rows = [row for row in rows if row[column] in wanted]
    if args.split:
        rows = [row for row in rows if row["split"] == args.split]
    rows = rows[:args.limit]
    if not rows:
        raise ValueError("No cases selected")
    directories = []
    for row in rows:
        directory = (root / row["case_directory"]).resolve()
        if not directory.is_relative_to(root) or directory == root:
            raise ValueError("Manifest case path escapes case root")
        metadata = inspect_case(directory)
        if metadata["sample_id"] != row["sample_id"]:
            raise ValueError("Manifest sample ID does not match case metadata")
        directories.append(directory)
    print(f"SWAN: {executable}; cases: {len(directories)}; workers: {workers}; threads/case: {threads}", flush=True)
    if args.check:
        print("Input checks passed; SWAN was not run.")
        return 0
    executable_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    target, require_converged = quota(config, root)
    if args.case_id or args.limit is not None:
        print("Diagnostic subset selected; full training quota is not enforced in this run.", flush=True)
        target = None
    if target is not None:
        return run_with_quota(root, rows, executable, timeout, threads, workers, args.rerun,
                              executable_sha256, target, require_converged)
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_case, directory, executable, timeout, threads, args.rerun, executable_sha256): directory for directory in directories}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = {"case_directory": str(futures[future]), "status": "failed", "error": str(exc)}
            failed += result["status"] not in {"completed", "skipped"}
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(f"Finished: {len(directories)-failed} completed/skipped, {failed} failed; convergence recorded only", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
