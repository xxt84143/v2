"""Validate bulk ERA5 archives and write canonical NetCDF incrementally."""
from __future__ import annotations

import zipfile
from pathlib import Path

from project import PERIODS, path_for, sample_times

ALIASES = {
    "u10": ("u10", "10u", "10m_u_component_of_wind"),
    "v10": ("v10", "10v", "10m_v_component_of_wind"),
    "swh": ("swh", "significant_height_of_combined_wind_waves_and_swell"),
    "mp1": ("mp1", "p140220", "mean_wave_period_based_on_first_moment"),
    "pp1d": ("pp1d", "peak_wave_period"),
    "mwd": ("mwd", "mean_wave_direction"),
    "mwp": ("mwp", "p140232", "mean_wave_period"),
}


def netcdf_paths(target: Path):
    if not zipfile.is_zipfile(target):
        return [target]
    directory = target.with_suffix(".files")
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with zipfile.ZipFile(target) as archive:
        for index, info in enumerate(archive.infolist()):
            if info.is_dir() or not info.filename.lower().endswith(".nc"):
                continue
            path = directory / f"part_{index:03d}.nc"
            temporary = path.with_suffix(".partial")
            with archive.open(info) as source, temporary.open("wb") as destination:
                import shutil
                shutil.copyfileobj(source, destination)
            temporary.replace(path)  # A full read also verifies the member's ZIP CRC.
            paths.append(path)
    if not paths:
        raise ValueError(f"Archive contains no NetCDF files: {target}")
    return paths


def canonical_dataset(source):
    """Normalize coordinates without loading the complete field arrays."""
    rename = {old: new for old, new in (("valid_time", "time"), ("lat", "latitude"), ("lon", "longitude"))
              if old in source.dims and new not in source.dims}
    data = source.rename(rename)
    if "expver" in data.dims:
        merged = data.isel(expver=0, drop=True)
        for index in range(1, data.sizes["expver"]):
            merged = merged.combine_first(data.isel(expver=index, drop=True))
        data = merged
    for dimension in list(data.dims):
        if dimension not in {"time", "latitude", "longitude"}:
            if data.sizes[dimension] != 1:
                raise ValueError(f"Unexpected dimension {dimension}")
            data = data.isel({dimension: 0}, drop=True)
    if not {"time", "latitude", "longitude"}.issubset(data.dims):
        raise ValueError("Expected time/latitude/longitude dimensions")
    data = data.assign_coords(longitude=((data.longitude + 180) % 360) - 180)
    return data.sortby("time").sortby("latitude", ascending=False).sortby("longitude")


def variable_names(request):
    return [next(name for name, aliases in ALIASES.items() if variable in aliases)
            for variable in request["variable"]]


def request_times(request):
    import numpy as np
    year, month = request["year"][0], request["month"][0]
    return np.asarray([np.datetime64(f"{year}-{month}-{day}T{hour}", "ns")
                       for day in request["day"] for hour in request["time"]])


def validate_download(plan, target):
    import numpy as np
    import xarray as xr
    expected = request_times(plan["request"])
    wanted = variable_names(plan["request"])
    found = set()
    reference = None
    for path in netcdf_paths(target):
        with xr.open_dataset(path) as source:
            data = canonical_dataset(source)
            if not np.array_equal(data.time.values.astype("datetime64[ns]"), expected):
                raise ValueError(f"ERA5 timestamps differ from request: {path}")
            latitude, longitude = data.latitude.values, data.longitude.values
            spacing = plan["request"]["grid"][0]
            for coordinate in (latitude, longitude):
                if coordinate.size < 2 or not np.allclose(np.abs(np.diff(coordinate)), spacing, atol=1e-8):
                    raise ValueError(f"Unexpected {spacing}-degree grid: {path}")
            north, west, south, east = plan["request"]["area"]
            if not (latitude[0] >= north - spacing - 1e-8 and latitude[-1] <= south + spacing + 1e-8 and
                    longitude[0] <= west + spacing + 1e-8 and longitude[-1] >= east - spacing - 1e-8):
                raise ValueError(f"Returned ERA5 grid does not cover the requested area: {path}")
            if reference is not None and not all(np.array_equal(a, b) for a, b in zip(reference, (latitude, longitude))):
                raise ValueError("Coordinates differ between files in one result")
            reference = latitude, longitude
            for name in wanted:
                aliases = [alias for alias in ALIASES[name] if alias in data.data_vars]
                if aliases:
                    if name in found or len(aliases) != 1:
                        raise ValueError(f"Duplicated field {name}: {path}")
                    if set(data[aliases[0]].dims) != {"time", "latitude", "longitude"}:
                        raise ValueError(f"Unexpected dimensions for {name}: {path}")
                    found.add(name)
    if found != set(wanted):
        raise ValueError(f"Downloaded variables differ from request: missing {set(wanted) - found}")


def merge_group(config, plans, group, overwrite=False, time_chunk=24):
    """Write one request/time block at a time; memory does not grow with the number of years."""
    import numpy as np
    import xarray as xr
    from netCDF4 import Dataset

    target = path_for(config, group)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; use --overwrite")
    selected = [plan for plan in plans if plan["group"] == group]
    if not selected:
        raise ValueError(f"No downloaded plans for {group}")
    requested = np.asarray([np.datetime64(stamp.replace(tzinfo=None), "ns") for stamp in sample_times(config)])
    names = ["u10", "v10"] if group == "wind" else ["swh", PERIODS[config["boundary"]["period"]][0], "mwd", "mwp"]
    names = list(dict.fromkeys(names))
    coverage = {name: np.zeros(len(requested), dtype=bool) for name in names}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    reference = None
    with Dataset(temporary, "w", format="NETCDF4") as output:
        for plan in selected:
            for path in netcdf_paths(plan["target"]):
                with xr.open_dataset(path) as source:
                    data = canonical_dataset(source)
                    latitude, longitude = data.latitude.values, data.longitude.values
                    if reference is None:
                        reference = latitude, longitude
                        output.createDimension("time", len(requested))
                        output.createDimension("latitude", len(latitude))
                        output.createDimension("longitude", len(longitude))
                        axis = output.createVariable("time", "i8", ("time",))
                        axis.units = "seconds since 1970-01-01 00:00:00"
                        axis.calendar = "proleptic_gregorian"
                        axis[:] = requested.astype("datetime64[s]").astype(np.int64)
                        for name, values, units in (("latitude", latitude, "degrees_north"), ("longitude", longitude, "degrees_east")):
                            coordinate = output.createVariable(name, "f8", (name,))
                            coordinate.units = units
                            coordinate[:] = values
                        for name in names:
                            output.createVariable(name, "f4", ("time", "latitude", "longitude"), fill_value=np.nan,
                                                  zlib=True, complevel=1,
                                                  chunksizes=(min(len(requested), time_chunk), min(len(latitude), 128), min(len(longitude), 128)))
                    elif not all(np.array_equal(a, b) for a, b in zip(reference, (latitude, longitude))):
                        raise ValueError("Coordinates differ between downloaded requests")
                    stamps = data.time.values.astype("datetime64[ns]")
                    indices = np.searchsorted(requested, stamps)
                    if np.any(indices >= len(requested)) or not np.array_equal(requested[indices], stamps):
                        raise ValueError("Downloaded time outside configured sampling")
                    if np.unique(indices).size != len(indices):
                        raise ValueError("Duplicate timestamps within one downloaded file")
                    for name in names:
                        found = [alias for alias in ALIASES[name] if alias in data.data_vars]
                        if not found:
                            continue
                        if len(found) != 1 or coverage[name][indices].any():
                            raise ValueError(f"Duplicate field/time for {name}")
                        field = data[found[0]].transpose("time", "latitude", "longitude")
                        attrs = {key: value for key, value in field.attrs.items() if key not in {"_FillValue", "missing_value"}}
                        output[name].setncatts(attrs)
                        for start in range(0, len(indices), time_chunk):
                            stop = start + time_chunk
                            output[name][indices[start:stop], :, :] = field.isel(time=slice(start, stop)).values
                        coverage[name][indices] = True
        for name, present in coverage.items():
            if not present.all():
                raise ValueError(f"Missing {int((~present).sum())} timestamps for {name}")
        output.setncatts({"schema_version": "toy-server-era5-1",
                         "period_definition": PERIODS[config["boundary"]["period"]][3],
                         "direction_convention": "mwd: clockwise from north, waves coming FROM"})
        if group == "waves":
            output["mwp"].setncatts({"units": "s", "period_definition": "Tm-1,0 = m-1/m0"})
    temporary.replace(target)
    print(f"Wrote {target}: {len(requested)} times", flush=True)
