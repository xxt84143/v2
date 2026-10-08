"""Run prepared cases independently of ERA5; verify convergence and output fields."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, help="Prepared case root; no ERA5 or GEBCO needed")
    parser.add_argument("--swan", help="Executable path or program name on PATH")
    parser.add_argument("--profile", action="append")
    parser.add_argument("--tile", action="append")
    parser.add_argument("--case-id", action="append")
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
    print(f"Finished: {len(directories)-failed} passed/skipped, {failed} failed/unconverged/unverified", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
