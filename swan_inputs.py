"""Pure SWAN input rendering and self-contained case serialization."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np

from forcing_arrays import validate


def array_text(array):
    stream = io.StringIO()
    np.savetxt(stream, array, fmt="%.8f")
    return stream.getvalue()


def boundary_blocks(wave, nx, ny, spread):
    """One shared rectangular sea field; SWAN uses each side's incoming components."""
    edges = [((0, 0), (1, 0), (0, 0), (nx, 0)),
             ((1, 0), (1, 1), (nx, 0), (nx, ny)),
             ((1, 1), (0, 1), (nx, ny), (0, ny)),
             ((0, 1), (0, 0), (0, ny), (0, 0))]
    result = []
    for first, last, start, end in edges:
        # SEGMENT IJ with VARIABLE PAR distances in degrees on a spherical grid.
        length = .5
        values = []
        reference = float(wave[2, first[1], first[0]])
        for distance, (x, y) in ((0, first), (length, last)):
            hs, period, direction = wave[:, y, x]
            direction = reference + (float(direction) - reference + 180) % 360 - 180
            values.append(f"  {distance:.10f} {hs:.8f} {period:.8f} {direction:.8f} {spread:.6f}")
        result.append(f"BOUNDSPEC SEGMENT IJ {start[0]} {start[1]} {end[0]} {end[1]} &\n"
                      "  VARIABLE PAR &\n" + values[0] + " &\n" + values[1])
    return result


def render_input(case_id, lon, lat, wave, wind, config):
    validate(wave, wind)
    nx, ny = len(lon) - 1, len(lat) - 1
    spectrum = config["spectral_grid"]
    lines = [f"PROJECT '{case_id[:16]}' 'ISLD'", "MODE STATIONARY TWODIMENSIONAL", "SET NAUTICAL",
             "COORD SPHERICAL CCM",
             f"CGRID REG {lon[0]:.10f} {lat[0]:.10f} 0 0.5 0.5 {nx} {ny} "
             f"CIRCLE {spectrum['directions']} {spectrum['lowest_frequency_hz']} "
             f"{spectrum['highest_frequency_hz']} {spectrum['frequency_intervals']}",
             f"INPGRID BOTTOM {lon[0]:.10f} {lat[0]:.10f} 0 {nx} {ny} "
             f"{.5/nx:.12f} {.5/ny:.12f} EXCEPTION {config['grid']['exception']}",
             "READINP BOTTOM 1 'bottom.dat' 3 0 FREE",
             f"INPGRID WIND {lon[0]:.10f} {lat[0]:.10f} 0 2 2 0.25 0.25",
             "READINP WIND 1 'wind.dat' 3 0 FREE",
             f"BOUND SHAPESPEC JONSWAP {config['boundary']['jonswap_gamma']} PEAK DSPR DEGREES"]
    lines.extend(boundary_blocks(wave, nx, ny, config["boundary"]["directional_spread_degrees"]))
    lines.extend(config["physics_commands"])
    # Default four-digit BLOCK output rounds geographic coordinates too much for grid validation.
    lines.append("OUTPUT OPTIONS BLOCK 8 6")
    lines.append("BLOCK 'COMPGRID' NOHEADER 'output/compgrid.tab' LAYOUT 3 " + " ".join(config["output_quantities"]))
    return "\n".join(lines + ["COMPUTE", "STOP", ""])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_case(directory, sample_id, case, profile, tile, lon, lat, depth, wet, wave, wind, config, overwrite=False):
    validate(wave, wind)
    stream = io.BytesIO()
    np.savez_compressed(stream, depth_m=depth, wet_mask=wet.astype(np.uint8), longitude=lon, latitude=lat,
                        wave=wave.astype(np.float32), wind=wind.astype(np.float32))
    files = {"INPUT": render_input(case["case_id"], lon, lat, wave, wind, config).encode("ascii"),
             "bottom.dat": array_text(depth).encode("ascii"),
             "wind.dat": array_text(wind.reshape(6, 3)).encode("ascii"),
             "swaninit": swaninit_text().encode("ascii"), "inputs.npz": stream.getvalue()}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    metadata = {"schema_version": "toy-server-case-2", "sample_id": sample_id,
                "case_id": case["case_id"], "profile": profile, "tile": tile,
                "shape": list(depth.shape), "wet_nodes": int(wet.sum()),
                "wave_shape": [3, 2, 2], "wave_channels": ["hs_m", "tp_s", "direction_from_deg"],
                "wind_shape": [2, 3, 3], "wind_channels": ["u10_mps", "v10_mps"],
                "matrix_order": "channel, south-to-north, west-to-east", "period_definition": "Tp",
                "terrain_id": case["terrain_id"], "forcing_id": case["forcing_id"],
                "split": case["split"], "generalization": case["generalization"],
                "family": case["family"], "boundary_policy": "one shared field on all edges; incoming components only",
                "physics_commands": config["physics_commands"], "spectral_grid": config["spectral_grid"],
                "output_quantities": config["output_quantities"], "input_hashes": hashes}
    previous_path = directory / "case.json"
    if directory.exists() and not overwrite:
        previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
        if previous != metadata or not all((directory/name).is_file() and
                                          hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
                                          for name, digest in hashes.items()):
            raise FileExistsError(f"Case changed or incomplete: {directory}; use --overwrite")
        return
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "output").mkdir(exist_ok=True)
    for name, data in files.items():
        (directory / name).write_bytes(data)
    write_json(previous_path, metadata)
    write_json(directory / "run_status.json", {"status": "prepared"})


def write_manifest(path, records, overwrite_legacy=False):
    """Keep previous subsets while adding cases, without coupling to a runner."""
    rows = {}
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as stream:
            previous = list(csv.DictReader(stream))
        if any("sample_id" not in row for row in previous):
            if not overwrite_legacy:
                raise ValueError("Legacy manifest format; regenerate with --overwrite")
        else:
            rows = {row["sample_id"]: row for row in previous}
    rows.update({row["sample_id"]: row for row in records})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    temporary.replace(path)


def swaninit_text() -> str:
    # SWAN replaces the first separator with the second; second must match this OS.
    return ("    4                                   version of initialisation file\n"
            "Delft University of Technology          name of institute\n"
            "    3                                   command file ref. number\n"
            "INPUT                                   command file name\n"
            "    4                                   print file ref. number\n"
            "PRINT                                   print file name\n"
            "    4                                   test file ref. number\n"
            "                                        test file name\n"
            "    6                                   screen ref. number\n"
            "99999                                   highest file ref. number\n"
            "$                                       comment identifier\n"
            "\t                                       TAB character\n"
            "/                                       dir sep char in input file\n"
            f"{os.sep}                                       dir sep char replacing previous one\n"
            "    1                                   default time coding option\n")



def grid_coordinates(tile: dict, resolution: dict) -> tuple[np.ndarray, np.ndarray]:
    west, south = float(tile["west"]), float(tile["south"])
    spacing = float(resolution["spacing"])
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Grid spacing must be positive")
    if resolution["kind"] == "degrees":
        nx = ny = round(.5 / spacing)
        if not math.isclose(nx * spacing, .5, abs_tol=1e-9):
            raise ValueError("Angular spacing must exactly divide 0.5 degrees")
    elif resolution["kind"] == "metres":
        nx = round(111195.08 * math.cos(math.radians(south + .25)) * .5 / spacing)
        ny = round(111195.08 * .5 / spacing)
    else:
        raise ValueError("Resolution kind must be metres or degrees")
    if min(nx, ny) < 2:
        raise ValueError("Spatial grid is too small")
    return np.linspace(west, west + .5, nx + 1), np.linspace(south, south + .5, ny + 1)
