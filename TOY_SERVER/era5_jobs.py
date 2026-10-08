"""ERA5 request planning and bounded download scheduling; no SWAN dependencies."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import threading
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

from project import PERIODS, sample_times, write_json

DEFAULT_DOWNLOAD = {
    "workers": 5,
    "variables_per_request": 1,
    "max_days_per_request": 31,
    "max_fields_per_request": 10000,
    "max_estimated_mib": 256,
    "http_timeout_seconds": 120,
    "http_retries": 10,
    "poll_seconds": 30,
    "download_attempts": 3,
    "retry_seconds": 30,
    "merge_time_chunk": 24,
}
_NETCDF_VALIDATION_LOCK = threading.Lock()


def download_settings(config):
    settings = {**DEFAULT_DOWNLOAD, **config["era5"].get("download", {})}
    unknown = set(settings) - set(DEFAULT_DOWNLOAD)
    if unknown:
        raise ValueError(f"Unknown era5.download options: {sorted(unknown)}")
    for key, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"era5.download.{key} must be a positive number")
        if isinstance(DEFAULT_DOWNLOAD[key], int) and not isinstance(value, int):
            raise ValueError(f"era5.download.{key} must be an integer")
    if settings["workers"] > 5:
        raise ValueError("workers must be 1..5; this downloader reserves at most five slots")
    if settings["max_days_per_request"] > 31:
        raise ValueError("max_days_per_request must be <=31")
    return settings


def estimates(request):
    fields = len(request["variable"]) * len(request["day"]) * len(request["time"])
    north, west, south, east = request["area"]
    dy, dx = request["grid"]
    # A planning estimate of float32 values, not the CDS conversion cost or actual ZIP size.
    cells = (math.ceil((north - south) / dy) + 1) * (math.ceil((east - west) / dx) + 1)
    return fields, fields * cells * 4 / 1024**2


def make_plan(group, dataset, request, raw_root):
    digest = hashlib.sha256(json.dumps({"dataset": dataset, "request": request}, sort_keys=True).encode()).hexdigest()[:20]
    fields, mib = estimates(request)
    return {"group": group, "dataset": dataset, "request": request,
            "target": Path(raw_root) / f"{group}_{request['year'][0]}{request['month'][0]}_{digest}.download",
            "fields": fields, "estimated_mib": round(mib, 3)}


def split_plan(plan):
    """Reduce variables first, then dates, then hours. Keep one common spatial area."""
    for key in ("variable", "day", "time"):
        values = plan["request"][key]
        if len(values) > 1:
            middle = len(values) // 2
            children = []
            for half in (values[:middle], values[middle:]):
                request = copy.deepcopy(plan["request"])
                request[key] = half
                children.append(make_plan(plan["group"], plan["dataset"], request, plan["target"].parent))
            return children
    raise ValueError("One variable/hour still exceeds limits; reduce the configured area or raise the local estimate budget")


def requests_for(config):
    settings = download_settings(config)
    dataset = config["era5"]["dataset"]
    if dataset != "reanalysis-era5-single-levels":
        raise ValueError("This planner supports reanalysis-era5-single-levels bulk data only")
    north, west, south, east = config["era5"]["area_north_west_south_east"]
    if not all(math.isfinite(v) for v in (north, west, south, east)) or not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise ValueError("Invalid ERA5 area; expected [north, west, south, east]")
    for tile in config["tiles"]:
        if not (west <= tile["west"] and tile["west"] + .5 <= east and south <= tile["south"] and tile["south"] + .5 <= north):
            raise ValueError(f"ERA5 area does not cover tile {tile['id']}")
    days = defaultdict(list)
    for stamp in sample_times(config):
        days[(stamp.year, stamp.month, stamp.day)].append(stamp.strftime("%H:%M"))
    # Days with different hour selections must not be multiplied together.
    patterns = defaultdict(list)
    for (year, month, day), hours in sorted(days.items()):
        patterns[(year, month, tuple(sorted(set(hours))))].append(f"{day:02d}")
    groups = {
        "wind": (["10m_u_component_of_wind", "10m_v_component_of_wind"], .25),
        "waves": (list(dict.fromkeys(["significant_height_of_combined_wind_waves_and_swell",
                                      PERIODS[config["boundary"]["period"]][1], "mean_wave_direction", "mean_wave_period"])), .5),
    }
    raw_root = Path(config["_base"]) / "data" / "raw"
    result = []
    for (year, month, hours), selected_days in sorted(patterns.items()):
        for group, (variables, spacing) in groups.items():
            for vstart in range(0, len(variables), settings["variables_per_request"]):
                for dstart in range(0, len(selected_days), settings["max_days_per_request"]):
                    request = {"product_type": ["reanalysis"],
                               "variable": variables[vstart:vstart + settings["variables_per_request"]],
                               "year": [str(year)], "month": [f"{month:02d}"],
                               "day": selected_days[dstart:dstart + settings["max_days_per_request"]],
                               "time": list(hours), "area": [north, west, south, east], "grid": [spacing, spacing],
                               "data_format": "netcdf", "download_format": "zip"}
                    pending = deque([make_plan(group, dataset, request, raw_root)])
                    while pending:
                        plan = pending.popleft()
                        fields, mib = estimates(plan["request"])
                        if fields > settings["max_fields_per_request"] or mib > settings["max_estimated_mib"]:
                            pending.extend(split_plan(plan))
                        else:
                            result.append(plan)
    return result


def state_path(plan):
    return plan["target"].with_suffix(".status.json")


def read_state(plan):
    path = state_path(plan)
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if state and (state.get("dataset") != plan["dataset"] or state.get("request") != plan["request"]):
        raise ValueError(f"State does not match request: {path}")
    return state


def save_state(plan, state):
    write_json(state_path(plan), {**state, "dataset": plan["dataset"], "request": plan["request"]})


def effective_plans(plans):
    """Follow persisted adaptive splits, also when running --merge-only."""
    leaves = []
    pending = deque(plans)
    while pending:
        plan = pending.popleft()
        state = read_state(plan)
        if state.get("status") == "split":
            children = split_plan(plan)
            if state.get("children") != [child["target"].name for child in children]:
                raise ValueError(f"Invalid saved split: {state_path(plan)}")
            pending.extend(children)
        else:
            leaves.append(plan)
    return leaves


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cached(plan, verify_hash=False):
    state = read_state(plan)
    target = plan["target"]
    if state.get("status") != "completed" or not target.is_file() or target.stat().st_size != state.get("bytes"):
        return False
    if verify_hash and sha256_file(target) != state.get("sha256"):
        raise ValueError(f"Cached file checksum changed: {target}; quarantine it before resuming")
    return True


@contextmanager
def download_lock(raw_root):
    raw_root.mkdir(parents=True, exist_ok=True)
    path = raw_root / ".download.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Another downloader may be active: {path}; remove only after confirming it stopped") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
        yield
    finally:
        path.unlink(missing_ok=True)


class RequestTooLarge(RuntimeError):
    pass


def safe_error(exc):
    # Signed result URLs belong to the API client, not to our manifest or console.
    return re.sub(r"https?://[^\s]+", lambda match: match.group().split("?")[0], str(exc))[:2000]


def cost_error(exc):
    message = str(exc).lower()
    return any(text in message for text in ("cost limits exceeded", "request is too large", "too many fields", "maximum number of fields"))


def cds_client(settings, submission=False):
    import cdsapi
    client = cdsapi.Client(timeout=settings["http_timeout_seconds"],
                           retry_max=1 if submission else settings["http_retries"],
                           sleep_max=settings["poll_seconds"], wait_until_complete=False, progress=False, quiet=True)
    service = getattr(client, "client", None)
    if service is None or not hasattr(service, "get_remote"):
        raise RuntimeError("Requires cdsapi>=0.7.7 with the current CDS API; upgrade requirements.txt")
    return client, service


def download_one(plan, settings, validate_file, stop):
    """Keep the same CDS job through polling/download retries and process restarts."""
    import requests
    from ecmwf.datastores.processing import DownloadError

    state = read_state(plan)
    if state.get("status") == "submitting" and not state.get("request_id"):
        raise RuntimeError(f"Submission outcome unknown: {state_path(plan)}; reconcile with CDS Your Requests before retrying")
    if state.get("status") == "terminal_failed":
        raise RuntimeError(f"CDS task failed: {state_path(plan)}; inspect the saved error before resetting this state")
    if stop.is_set():
        raise RuntimeError("Stopped before starting this job")
    if not state.get("request_id"):
        # Do not repeat POST automatically after a lost response: its outcome may be unknown.
        client, _ = cds_client(settings, submission=True)
        state = {"status": "submitting"}
        save_state(plan, state)
        try:
            remote = client.retrieve(plan["dataset"], plan["request"])
        except Exception as exc:
            response = getattr(exc, "response", None)
            # An explicit client rejection is safe to split/retry. A lost POST response is ambiguous.
            rejected = response is not None and 400 <= response.status_code < 500 and response.status_code not in (408, 429)
            if rejected:
                save_state(plan, {"status": "rejected", "error": safe_error(exc)})
                if cost_error(exc):
                    raise RequestTooLarge(safe_error(exc)) from exc
            raise
        state = {"status": "submitted", "request_id": remote.request_id}
        save_state(plan, state)
        print(f"SUBMITTED {plan['target'].name}: {remote.request_id}", flush=True)
    _, service = cds_client(settings)
    partial = plan["target"].with_suffix(".partial")
    for attempt in range(settings["download_attempts"]):
        if stop.is_set():
            raise RuntimeError("Stopped; saved CDS job will be resumed next time")
        remote_terminal = False
        try:
            remote = service.get_remote(state["request_id"])
            while True:
                status = remote.status
                state.update(status=status)
                save_state(plan, state)
                if status == "successful":
                    break
                if status not in ("accepted", "running"):
                    remote_terminal = True
                    remote.get_results()  # Obtain the server's detailed failure message.
                    raise RuntimeError(f"Unexpected CDS state: {status}")
                if stop.wait(settings["poll_seconds"]):
                    raise RuntimeError("Stopped; saved CDS job will be resumed next time")
            results = remote.get_results()
            expected = results.content_length
            # A crash after writing a full file may leave only the completion marker missing.
            candidate = plan["target"] if plan["target"].is_file() else partial
            if not candidate.is_file() or candidate.stat().st_size != expected:
                results.download(str(partial))
                candidate = partial
            if not candidate.is_file() or candidate.stat().st_size != expected or expected <= 0:
                raise ValueError("CDS download size differs from its result metadata")
            # netCDF4/HDF5 readers must not be used concurrently in these worker threads.
            with _NETCDF_VALIDATION_LOCK:
                validate_file(plan, candidate)
            digest = sha256_file(candidate)
            if candidate != plan["target"]:
                candidate.replace(plan["target"])
            state.update(status="completed", bytes=expected, sha256=digest)
            state.pop("error", None)
            save_state(plan, state)
            return
        except Exception as exc:
            state["error"] = safe_error(exc)
            if remote_terminal:
                state["status"] = "terminal_failed"
            save_state(plan, state)
            if remote_terminal and cost_error(exc):
                raise RequestTooLarge(safe_error(exc)) from exc
            # Retry network/transfer errors against this request ID; validation errors need inspection.
            transient = isinstance(exc, (requests.ConnectionError, requests.Timeout, DownloadError)) or (
                isinstance(exc, requests.HTTPError) and exc.response is not None and
                (exc.response.status_code >= 500 or exc.response.status_code in (408, 429)))
            if not transient or attempt + 1 == settings["download_attempts"]:
                raise
            if stop.wait(min(300, settings["retry_seconds"] * 2**attempt)):
                raise RuntimeError("Stopped; saved CDS job will be resumed next time") from exc


def download_all(plans, settings, validate_file, verify_hash=False):
    leaves = effective_plans(plans)
    pending = [plan for plan in leaves if not cached(plan, verify_hash)]
    # Recover old server-side jobs before submitting fresh ones. They already occupy remote slots.
    pending.sort(key=lambda plan: not bool(read_state(plan).get("request_id")))
    recoveries = [plan for plan in pending if read_state(plan).get("request_id")]
    if len(recoveries) > settings["workers"]:
        raise RuntimeError("More saved CDS jobs than worker slots; increase workers up to five or finish them in CDS first")
    for plan in pending:
        if read_state(plan).get("status") == "submitting" and not read_state(plan).get("request_id"):
            raise RuntimeError(f"Uncertain submission requires reconciliation: {state_path(plan)}")
    if pending:
        _, service = cds_client(settings)
        owned_ids = {read_state(plan).get("request_id") for plan in pending}
        other_ids = set(service.get_jobs(status=["accepted", "running"], limit=100).request_ids) - owned_ids
        if other_ids:
            raise RuntimeError("Other CDS jobs are already active on this account; finish them before starting this five-slot queue")
    pending = deque(pending)
    stop = threading.Event()
    failure = None
    with ThreadPoolExecutor(max_workers=settings["workers"]) as executor:
        active = {}
        try:
            while pending or active:
                while pending and len(active) < settings["workers"] and failure is None:
                    plan = pending.popleft()
                    active[executor.submit(download_one, plan, settings, validate_file, stop)] = plan
                if not active:
                    break
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    plan = active.pop(future)
                    try:
                        future.result()
                        print(f"COMPLETE {plan['target'].name}; {len(pending)} pending", flush=True)
                    except RequestTooLarge as exc:
                        try:
                            children = split_plan(plan)
                            state = read_state(plan)
                            state.update(status="split", children=[child["target"].name for child in children])
                            save_state(plan, state)
                            pending.extendleft(reversed(children))
                            print(f"SPLIT {plan['target'].name}: {safe_error(exc)}", flush=True)
                        except Exception as split_exc:
                            failure = failure or split_exc
                            stop.set()
                    except Exception as exc:
                        failure = failure or exc
                        stop.set()
                        print(f"FAILED {plan['target'].name}: {safe_error(exc)}", flush=True)
            if failure is not None:
                raise RuntimeError("Download stopped after failure; saved jobs/files can be resumed") from failure
        except BaseException:
            stop.set()
            raise
    return effective_plans(plans)
