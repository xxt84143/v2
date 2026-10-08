"""Download wind and bulk waves separately; write two canonical NetCDF files."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from project import DEFAULT_CONFIG, PERIODS, load_config, path_for, sample_times, write_json


def requests_for(config: dict) -> list[dict]:
    """Small monthly requests; temporal cross-products are filtered after download."""
    north, west, south, east = config["era5"]["area_north_west_south_east"]
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise ValueError("Invalid ERA5 area; expected [north, west, south, east]")
    for tile in config["tiles"]:
        if not (west <= tile["west"] and tile["west"] + .5 <= east and
                south <= tile["south"] and tile["south"] + .5 <= north):
            raise ValueError(f"ERA5 area does not cover tile {tile['id']}")
    months = defaultdict(list)
    for stamp in sample_times(config):
        months[(stamp.year, stamp.month)].append(stamp)
    period_variable = PERIODS[config["boundary"]["period"]][1]
    groups = {
        "wind": (["10m_u_component_of_wind", "10m_v_component_of_wind"], .25),
        "waves": (["significant_height_of_combined_wind_waves_and_swell",
                   period_variable, "mean_wave_direction", "mean_wave_period"], .5),
    }
    result = []
    raw_root = Path(config["_base"]) / "data" / "raw"
    for (year, month), times in sorted(months.items()):
        for group, (variables, spacing) in groups.items():
            request = {
                "product_type": ["reanalysis"], "variable": variables,
                "year": [str(year)], "month": [f"{month:02d}"],
                "day": sorted({f"{stamp.day:02d}" for stamp in times}),
                "time": sorted({stamp.strftime("%H:%M") for stamp in times}),
                "area": [north, west, south, east], "grid": [spacing, spacing],
                "data_format": "netcdf", "download_format": "zip",
            }
            digest = hashlib.sha256(json.dumps({"dataset": config["era5"]["dataset"],
                                               "request": request}, sort_keys=True).encode()).hexdigest()[:12]
            result.append({"group": group, "request": request,
                           "target": raw_root / f"{group}_{year}{month:02d}_{digest}.download"})
    return result


def netcdf_paths(target: Path) -> list[Path]:
    if not zipfile.is_zipfile(target):
        return [target]
    directory = target.with_suffix(".files")
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    with zipfile.ZipFile(target) as archive:
        for index, info in enumerate(archive.infolist()):
            if info.is_dir() or not info.filename.lower().endswith(".nc"):
                continue
            # Use our own filename, so ZIP paths can never escape this directory.
            path = directory / f"part_{index:03d}.nc"
            if not path.exists() or path.stat().st_size != info.file_size:
                temporary = path.with_suffix(".partial")
                with archive.open(info) as source, temporary.open("wb") as destination:
                    import shutil
                    shutil.copyfileobj(source, destination)
                temporary.replace(path)
            result.append(path)
    if not result:
        raise ValueError(f"Archive contains no NetCDF files: {target}")
    return result


def canonical_dataset(path: Path):
    import xarray as xr
    with xr.open_dataset(path) as source:
        rename = {old: new for old, new in (("valid_time", "time"), ("lat", "latitude"),
                                           ("lon", "longitude")) if old in source.dims and new not in source.dims}
        data = source.rename(rename).load()
    if "expver" in data.dims:
        merged = data.isel(expver=0, drop=True)
        for index in range(1, data.sizes["expver"]):
            merged = merged.combine_first(data.isel(expver=index, drop=True))
        data = merged
    for dimension in list(data.dims):
        if dimension not in {"time", "latitude", "longitude"}:
            if data.sizes[dimension] != 1:
                raise ValueError(f"Unexpected dimension {dimension} in {path}")
            data = data.isel({dimension: 0}, drop=True)
    if not {"time", "latitude", "longitude"}.issubset(data.dims):
        raise ValueError(f"Expected time/latitude/longitude dimensions in {path}")
    data = data.assign_coords(longitude=((data.longitude + 180) % 360) - 180)
    return data.sortby("time").sortby("latitude", ascending=False).sortby("longitude")


def merge_group(config: dict, plans: list[dict], group: str, overwrite: bool) -> None:
    import numpy as np
    import xarray as xr
    target = path_for(config, group)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; use --overwrite to replace canonical data")
    period = PERIODS[config["boundary"]["period"]][0]
    aliases = {
        "u10": ("u10", "10u", "10m_u_component_of_wind"),
        "v10": ("v10", "10v", "10m_v_component_of_wind"),
        "swh": ("swh", "significant_height_of_combined_wind_waves_and_swell"),
        "mp1": ("mp1", "p140220", "mean_wave_period_based_on_first_moment"),
        "pp1d": ("pp1d", "peak_wave_period"),
        "mwd": ("mwd", "mean_wave_direction"),
        "mwp": ("mwp", "p140232", "mean_wave_period"),
    }
    names = ["u10", "v10"] if group == "wind" else ["swh", period, "mwd", "mwp"]
    datasets = []
    for plan in plans:
        if plan["group"] != group:
            continue
        parts = [canonical_dataset(path) for path in netcdf_paths(plan["target"])]
        monthly = xr.combine_by_coords(parts, combine_attrs="drop_conflicts")
        rename = {}
        for name in names:
            found = next((alias for alias in aliases[name] if alias in monthly.data_vars), None)
            if found is None:
                raise ValueError(f"Missing {name} in downloaded fields: {list(monthly.data_vars)}")
            rename[found] = name
        datasets.append(monthly.rename(rename)[names])
    data = xr.concat(datasets, dim="time", join="exact").sortby("time")
    if np.unique(data.time.values).size != data.sizes["time"]:
        raise ValueError("Duplicate ERA5 timestamps")
    requested = np.asarray([np.datetime64(stamp.replace(tzinfo=None), "ns") for stamp in sample_times(config)])
    data = data.sel(time=requested).transpose("time", "latitude", "longitude")
    spacing = .25 if group == "wind" else .5
    for coordinate in (data.latitude.values, data.longitude.values):
        if coordinate.size < 2 or not np.allclose(np.abs(np.diff(coordinate)), spacing, atol=1e-8):
            raise ValueError(f"Unexpected {group} grid; expected {spacing} degree steps")
    data.attrs.update(schema_version="toy-server-era5-1", period_definition=PERIODS[config["boundary"]["period"]][3],
                      direction_convention="mwd: clockwise from north, waves coming FROM")
    if group == "waves":
        data["mwp"].attrs.update(units="s", period_definition="Tm-1,0 = m-1/m0")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    data.to_netcdf(temporary)
    temporary.replace(target)
    print(f"Wrote {target}: {data.sizes}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Print requests only; no network or file writes")
    parser.add_argument("--merge-only", action="store_true", help="Use existing downloaded monthly files")
    parser.add_argument("--overwrite", action="store_true", help="Replace the canonical wind/wave NetCDF outputs")
    args = parser.parse_args()
    config = load_config(args.config)
    plans = requests_for(config)
    if args.dry_run:
        for plan in plans:
            print(json.dumps({"dataset": config["era5"]["dataset"], **plan}, default=str, indent=2))
        return
    for key in ("wind", "waves"):
        if path_for(config, key).exists() and not args.overwrite:
            raise FileExistsError(f"{path_for(config, key)} exists; use --overwrite")
    client = None
    for plan in plans:
        target = plan["target"]
        if target.is_file() and target.stat().st_size:
            print(f"Reuse cached download: {target.name}", flush=True)
            continue
        if args.merge_only:
            raise FileNotFoundError(target)
        if client is None:
            import cdsapi
            client = cdsapi.Client()
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target.with_suffix(".request.json"), {"dataset": config["era5"]["dataset"], "request": plan["request"]})
        temporary = target.with_name(target.name + ".partial")
        client.retrieve(config["era5"]["dataset"], plan["request"], str(temporary))
        if not temporary.is_file() or not temporary.stat().st_size:
            raise ValueError("CDS returned an empty download")
        temporary.replace(target)
    for group in ("wind", "waves"):
        merge_group(config, plans, group, args.overwrite)


if __name__ == "__main__":
    main()
