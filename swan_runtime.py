"""Validate saved SWAN output and record convergence without rejecting it."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from swan_inputs import write_json
from forcing_arrays import load as load_forcing

SAVED_STATUSES = frozenset({"completed", "unconverged", "unverified"})
CONVERGENCE_FIELDS = ("convergence_percent", "required_percent", "converged",
                      "iterations", "convergence_log_recognized")


def fingerprint(hashes: dict) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def inspect_case(directory: Path) -> dict:
    metadata = json.loads((directory / "case.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "toy-server-case-2":
        raise ValueError(f"Unsupported case schema: {directory}")
    for name in ("INPUT", "bottom.dat", "wind.dat", "swaninit", "inputs.npz"):
        expected = metadata["input_hashes"][name]
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared input changed: {directory / name}; regenerate this case")
    bottom = np.loadtxt(directory / "bottom.dat")
    if list(bottom.shape) != metadata["shape"] or not np.isfinite(bottom).all():
        raise ValueError(f"Invalid bottom shape/values in {directory}")
    if np.count_nonzero(bottom > 0) != metadata["wet_nodes"]:
        raise ValueError("Case wet-mask metadata does not agree with bottom.dat")
    if np.loadtxt(directory / "wind.dat").shape != (6, 3):
        raise ValueError("Expected u10 then v10, each 3x3, in wind.dat")
    wave, wind = load_forcing(directory / "inputs.npz")
    if not np.allclose(np.loadtxt(directory / "wind.dat"), wind.reshape(6, 3), atol=1e-7):
        raise ValueError("wind.dat does not match the nine-node matrix")
    return metadata


def convergence(directory: Path) -> dict:
    print_path = directory / "PRINT"
    text = print_path.read_text(encoding="utf-8", errors="replace") if print_path.is_file() else ""
    err_path = directory / "Errfile"
    errors = err_path.read_text(encoding="utf-8", errors="replace") if err_path.is_file() else ""
    combined = text + "\n" + errors
    failed = bool(re.search(r"\*\*\s+Error\b|Terminating\s+error", combined, re.I))
    matches = re.findall(r"accuracy\s+OK\s+in\s+([0-9.]+)\s*%\s+of\s+wet\s+grid\s+points\s*"
                         r"\(\s*([0-9.]+)\s*%\s+required", text, re.I)
    actual, required = (map(float, matches[-1]) if matches else (None, None))
    iterations = re.findall(r"^\s*iteration\s+(\d+)\s*;", text, re.I | re.M)
    return {"print_error": failed, "convergence_percent": actual, "required_percent": required,
            "converged": actual >= required if matches else None,
            "iterations": int(iterations[-1]) if iterations else None,
            "convergence_log_recognized": bool(matches)}


def check_output(directory: Path, metadata: dict) -> dict:
    path = directory / "output" / "compgrid.tab"
    if not path.is_file() or not path.stat().st_size:
        raise ValueError("SWAN output is absent or empty")
    expanded = []
    for quantity in metadata["output_quantities"]:
        expanded.extend(["WIND_X", "WIND_Y"] if quantity == "WIND" else [quantity])
    shape = tuple(metadata["shape"])
    values = np.fromstring(path.read_text(encoding="ascii"), sep=" ")
    expected = len(expanded) * math_product(shape)
    if values.size != expected:
        raise ValueError(f"SWAN output has {values.size} values; expected {expected}")
    fields = dict(zip(expanded, values.reshape(len(expanded), *shape)))
    wet = np.loadtxt(directory / "bottom.dat") > 0
    hs = fields["HSIGN"][wet]
    if not np.isfinite(hs).all() or np.any(hs < 0):
        raise ValueError("Invalid HSIGN at wet nodes")
    tile = metadata["tile"]
    ny, nx = shape
    yy, xx = np.meshgrid(np.linspace(tile["south"], tile["south"]+.5, ny),
                         np.linspace(tile["west"], tile["west"]+.5, nx), indexing="ij")
    if not np.allclose(fields["XP"][wet], xx[wet], atol=2e-4, rtol=0) or not np.allclose(fields["YP"][wet], yy[wet], atol=2e-4, rtol=0):
        raise ValueError("Output coordinates/order do not match the prepared grid")
    energetic = wet & (fields["HSIGN"] > .01)
    for name in ("TM01", "TM02", "RTP", "TMM10"):
        if name in fields and (not np.isfinite(fields[name][energetic]).all() or np.any(fields[name][energetic] <= 0)):
            raise ValueError(f"Invalid {name} at energetic wet nodes")
    for name in ("DIR", "PDIR", "DSPR"):
        if name in fields:
            selected = fields[name][energetic]
            maximum = 180 if name == "DSPR" else 360
            if not np.isfinite(selected).all() or np.any((selected < 0) | (selected > maximum)):
                raise ValueError(f"Invalid {name} at energetic wet nodes")
    return {"hs_min_m": float(hs.min()), "hs_max_m": float(hs.max()), "wet_nodes": int(wet.sum()),
            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def math_product(shape: tuple) -> int:
    return int(np.prod(shape))


def run_case(directory: Path, executable: Path, timeout: float, threads: int, rerun: bool,
             executable_sha256: str | None = None) -> dict:
    """Reserve this case so separate runner processes cannot overwrite the same output."""
    lock = directory / ".swan.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Case is already running: {lock}; remove a stale lock only after checking the process stopped") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
        return _run_case(directory, executable, timeout, threads, rerun, executable_sha256)
    finally:
        lock.unlink(missing_ok=True)


def _run_case(directory: Path, executable: Path, timeout: float, threads: int, rerun: bool,
              executable_sha256: str | None = None) -> dict:
    metadata = inspect_case(directory)
    executable_sha256 = executable_sha256 or hashlib.sha256(executable.read_bytes()).hexdigest()
    signature = fingerprint(metadata["input_hashes"])
    status_path = directory / "run_status.json"
    previous = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    if previous.get("status") in SAVED_STATUSES and previous.get("input_signature") == signature and not rerun:
        if previous.get("executable_sha256") != executable_sha256:
            raise ValueError("SWAN executable changed; use --rerun explicitly")
        output_metrics = check_output(directory, metadata)
        if output_metrics["output_sha256"] != previous.get("output_sha256"):
            raise ValueError("Completed case output changed; use --rerun explicitly")
        recorded = convergence(directory)
        if previous.get("return_code") != 0 or recorded["print_error"]:
            raise ValueError("Saved case has a SWAN execution error; inspect logs or use --rerun")
        previous.update(recorded)
        previous.update(status="completed", error=None, convergence_policy="record_only")
        write_json(status_path, previous)
        return {"sample_id": metadata["sample_id"], "status": "skipped",
                "convergence_policy": "record_only", **recorded, **output_metrics}
    write_json(status_path, {"status": "running", "input_signature": signature,
                             "started_at_utc": datetime.now(timezone.utc).isoformat()})
    environment = os.environ.copy()
    environment["PATH"] = str(executable.parent) + os.pathsep + environment.get("PATH", "")
    environment["OMP_NUM_THREADS"] = str(threads)
    started = time.perf_counter()
    result = {"sample_id": metadata["sample_id"], "status": "failed", "executable": str(executable),
              "executable_sha256": executable_sha256, "input_signature": signature, "return_code": None, "error": None,
              "convergence_policy": "record_only"}
    try:
        for name in ("PRINT", "Errfile", "ERRPTS", "norm_end", "output/compgrid.tab"):
            path = directory / name
            if path.is_file():
                path.unlink()
        with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
            process = subprocess.run([str(executable)], cwd=directory, env=environment,
                                     stdout=out, stderr=err, timeout=timeout, check=False,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        result["return_code"] = process.returncode
        result.update(convergence(directory))
        if process.returncode != 0 or result["print_error"]:
            raise ValueError("SWAN returned an error; inspect PRINT, stdout.log and stderr.log")
        result.update(check_output(directory, metadata))
        result["status"] = "completed"
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)
    result["runtime_seconds"] = round(time.perf_counter() - started, 3)
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(status_path, result)
    return result
