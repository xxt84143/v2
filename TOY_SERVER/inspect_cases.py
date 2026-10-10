"""Build an offline visual report from experiment settings and existing case files.

This entry point reads files only. It never generates a case or starts SWAN.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from island_core import Rectangle, case_plan, generate_forcings, generate_islands, make_island_depth
from project import DEFAULT_CONFIG, load_config, path_for
from swan_inputs import grid_coordinates
from swan_runtime import check_output, inspect_case

FAMILIES = {"swell_no_wind": "无风涌浪", "aligned_windsea": "同向风浪",
            "swell_weak_wind": "弱风涌浪", "wind_only": "纯风生浪"}


def propagation(direction_from):
    """Nautical FROM angle -> east/north components of the travel direction."""
    radians = np.deg2rad(direction_from)
    return -np.sin(radians), -np.cos(radians)


def save_figure(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def axes_for_map(axis, tile):
    width = 111.19508 * np.cos(np.deg2rad(tile["south"] + .25)) * .5
    height = 111.19508 * .5
    axis.set(xlabel="East from western edge (km)", ylabel="North from southern edge (km)")
    axis.set_aspect("equal")
    return [0, width, 0, height]


def depth_plot(path, depth, wet, tile, title):
    figure, axis = plt.subplots(figsize=(6.3, 5.5), layout="constrained")
    extent = axes_for_map(axis, tile)
    axis.set_facecolor("#bbb7a7")
    image = axis.imshow(np.where(wet, depth, np.nan), origin="lower", extent=extent,
                        cmap="Blues", vmin=0, vmax=max(1, float(depth[wet].max())))
    figure.colorbar(image, ax=axis, label="Water depth (m)")
    if wet.any() and not wet.all():
        axis.contour(np.linspace(0, extent[1], depth.shape[1]),
                     np.linspace(0, extent[3], depth.shape[0]), wet, levels=[.5], colors="#333", linewidths=.8)
    axis.set_title(title + f"\n{depth.shape[0]} x {depth.shape[1]} nodes; land in grey")
    save_figure(figure, path)


def forcing_plot(path, wave, wind, tile, title):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained")
    extent = axes_for_map(axes[0], tile)
    axes_for_map(axes[1], tile)
    width, height = extent[1], extent[3]
    xs, ys = np.meshgrid(np.linspace(0, width, 3), np.linspace(0, height, 3))
    speed = np.hypot(wind[0], wind[1])
    axes[0].quiver(xs, ys, wind[0], wind[1], color="#15836e", angles="xy", scale_units="xy",
                   scale=max(1, speed.max()) / (min(width, height) * .12), width=.007)
    for x, y, s in zip(xs.flat, ys.flat, speed.flat):
        axes[0].scatter(x, y, color="#15836e", s=12)
        axes[0].annotate(f"{s:.2f}", (x, y), xytext=(4, 5), textcoords="offset points", fontsize=9)
    axes[0].set_title("Nine original wind nodes\nArrows: travel direction; labels: speed (m/s)")
    xs, ys = np.meshgrid([0, width], [0, height])
    u, v = propagation(wave[2])
    active = wave[0] > 0
    axes[1].quiver(xs[active], ys[active], u[active], v[active], color="#d47220", angles="xy",
                   scale_units="xy", scale=1 / (min(width, height) * .13), width=.007)
    for index in np.ndindex((2, 2)):
        x, y = xs[index], ys[index]
        text = (f"Hs {wave[0][index]:.2f} m\nTp {wave[1][index]:.2f} s\nFROM {wave[2][index]:.1f} deg"
                if active[index] else "Hs 0 m\nNo incoming wave energy")
        axes[1].scatter(x, y, color="#d47220", s=22)
        axes[1].annotate(text, (x, y), xytext=(6 if index[1] == 0 else -6, 6 if index[0] == 0 else -6),
                         textcoords="offset points", ha="left" if index[1] == 0 else "right",
                         va="bottom" if index[0] == 0 else "top", fontsize=9)
    axes[1].set_title("Four shared wave boundary nodes\nArrows: propagation (opposite to FROM)")
    for axis in axes:
        axis.set(xlim=(-.18 * width, 1.18 * width), ylim=(-.18 * height, 1.18 * height))
        axis.plot([0, width, width, 0, 0], [0, 0, height, height, 0], color="#586b78", lw=1)
        axis.grid(alpha=.2)
    figure.suptitle(title, fontsize=13)
    save_figure(figure, path)


def forcing_values(wave, wind):
    speed = np.hypot(wind[0], wind[1])
    wind_from = np.rad2deg(np.arctan2(-wind[0], -wind[1])) % 360
    corner_speed = speed[::2, ::2]
    offset = (wave[2] - wind_from[::2, ::2] + 180) % 360 - 180
    offset = np.where((corner_speed > .01) & (wave[0] > 0), offset, np.nan)
    return {"wave": wave.tolist(), "wind": wind.tolist(), "wind_speed": speed.tolist(),
            "wind_from": np.where(speed > .01, wind_from, np.nan).tolist(),
            "wave_wind_offset": offset.tolist()}


def read_output(directory, metadata):
    names = [part for name in metadata["output_quantities"]
             for part in (["WIND_X", "WIND_Y"] if name == "WIND" else [name])]
    values = np.fromstring((directory / "output/compgrid.tab").read_text(encoding="ascii"), sep=" ")
    return dict(zip(names, values.reshape(len(names), *metadata["shape"])))


def output_plot(path, fields, wet, tile, title):
    labels = [("HSIGN", "Hs (m)"), ("TM01", "Tm01 (s)"), ("TM02", "Tm02 (s)"),
              ("RTP", "Tp (s)"), ("TMM10", "Tm-1,0 (s)"), ("DSPR", "Directional spread (deg)"),
              ("DIR", "Mean direction FROM (deg)"), ("PDIR", "Peak direction FROM (deg)"),
              ("WIND_SPEED", "SWAN wind speed (m/s)")]
    if "WIND_X" in fields and "WIND_Y" in fields:
        fields["WIND_SPEED"] = np.hypot(fields["WIND_X"], fields["WIND_Y"])
    figure, axes = plt.subplots(3, 3, figsize=(15, 14), layout="constrained")
    for axis, (name, label) in zip(axes.flat, labels):
        extent = axes_for_map(axis, tile)
        axis.set_facecolor("#bbb7a7")
        axis.set_title(label)
        if name not in fields:
            axis.text(.5, .5, "Not requested", transform=axis.transAxes, ha="center")
            continue
        valid = wet if name in ("HSIGN", "WIND_SPEED") else wet & (fields["HSIGN"] > .01)
        options = {"vmin": 0, "vmax": 360, "cmap": "twilight"} if name in ("DIR", "PDIR") else {"cmap": "viridis"}
        image = axis.imshow(np.where(valid, fields[name], np.nan), origin="lower", extent=extent, **options)
        figure.colorbar(image, ax=axis, shrink=.75)
        if name == "HSIGN" and "DIR" in fields:
            skip = max(1, wet.shape[0] // 12)
            small = wet[::skip, ::skip] & (fields["HSIGN"][::skip, ::skip] > .01)
            u, v = propagation(fields["DIR"][::skip, ::skip])
            xx, yy = np.meshgrid(np.linspace(0, extent[1], wet.shape[1])[::skip],
                                 np.linspace(0, extent[3], wet.shape[0])[::skip])
            axis.quiver(xx[small], yy[small], u[small], v[small], angles="xy", scale_units="xy",
                        scale=1 / (min(extent[1], extent[3]) * .035), color="white", width=.004)
    figure.suptitle(title + "\nHs arrows: mean wave propagation; low-energy periods/directions masked", fontsize=14)
    save_figure(figure, path)


def convergence_history(text):
    rows = []
    for match in re.finditer(r"^\s*iteration\s+(\d+)\s*;([\s\S]*?)(?=^\s*iteration\s+\d+\s*;|\Z)", text, re.I | re.M):
        accuracy = re.findall(r"accuracy\s+OK\s+in\s+([0-9.]+)\s*%.*?\(\s*([0-9.]+)\s*%\s+required", match[2], re.I)
        if accuracy:
            actual, reference = map(float, accuracy[-1])
            rows.append([int(match[1]), actual, reference])
    return rows


def history_plot(path, history):
    figure, axis = plt.subplots(figsize=(8, 3.7), layout="constrained")
    if history:
        values = np.asarray(history)
        axis.plot(values[:, 0], values[:, 1], "o-", color="#15836e", label="Recorded accuracy OK (%)")
        axis.plot(values[:, 0], values[:, 2], "--", color="#d47220", label="SWAN log reference (record only)")
        axis.legend(fontsize=9)
        axis.set_ylim(0, 102)
    else:
        axis.text(.5, .5, "No recognized iteration percentages", transform=axis.transAxes, ha="center")
    axis.set(xlabel="Iteration", ylabel="Wet grid points (%)", title="Convergence is recorded; no acceptance threshold")
    axis.grid(alpha=.2)
    save_figure(figure, path)


def existing_cases(root, profile, output):
    results = {}
    if not root.exists():
        return results
    for metadata_path in sorted(root.rglob("case.json")):
        directory = metadata_path.parent.resolve()
        if not directory.is_relative_to(root):
            raise ValueError(f"Case path escapes case root: {directory}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("profile") != profile:
            continue
        status_path = directory / "run_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"status": "unknown"}
        record = {**{name: metadata.get(name) for name in ("sample_id", "case_id", "terrain_id", "forcing_id", "family", "split", "generalization")},
                  "tile_id": metadata["tile"]["id"], "source": "prepared", "status": status.get("status", "unknown"),
                  "run": status, "checks": [], "directory": str(directory), "output_image": None, "history_image": None}
        key = hashlib.sha256(metadata["sample_id"].encode()).hexdigest()[:16]
        try:
            inspect_case(directory)
            record["checks"].append({"ok": True, "text": "输入文件哈希、地形和九点风矩阵校验通过"})
            with np.load(directory / "inputs.npz", allow_pickle=False) as inputs:
                wave, wind = inputs["wave"], inputs["wind"]
                depth, wet = inputs["depth_m"], inputs["wet_mask"].astype(bool)
            record.update(forcing_values(wave, wind))
            record.update(shape=list(depth.shape), depth_image=f"assets/case_{key}_depth.png", forcing_image=f"assets/case_{key}_forcing.png")
            depth_plot(output / record["depth_image"], depth, wet, metadata["tile"], metadata["sample_id"])
            forcing_plot(output / record["forcing_image"], wave, wind, metadata["tile"], metadata["sample_id"])
            if (directory / "output/compgrid.tab").exists():
                metrics = check_output(directory, metadata)
                if status.get("output_sha256") and metrics["output_sha256"] != status["output_sha256"]:
                    raise ValueError("Output differs from the saved result hash")
                record["checks"].append({"ok": True, "text": "SWAN 输出尺寸、坐标、数值与已记录哈希校验通过"})
                record.update(metrics, output_image=f"assets/case_{key}_swan.png")
                output_plot(output / record["output_image"], read_output(directory, metadata), wet, metadata["tile"], metadata["sample_id"])
            else:
                record["checks"].append({"ok": None, "text": "尚无 SWAN 输出"})
        except (OSError, ValueError, KeyError) as exc:
            record["checks"].append({"ok": False, "text": str(exc)})
        print_path = directory / "PRINT"
        text = print_path.read_text(encoding="utf-8", errors="replace") if print_path.exists() else ""
        record["history"] = convergence_history(text)
        if text:
            record["history_image"] = f"assets/case_{key}_history.png"
            history_plot(output / record["history_image"], record["history"])
        record["input_text"] = (directory / "INPUT").read_text(encoding="utf-8", errors="replace") if (directory / "INPUT").exists() else ""
        record["log_tail"] = "\n".join(text.splitlines()[-25:])
        record["log_errors"] = "\n".join(line for line in text.splitlines() if re.search(r"\*\*\s+(?:Error|Warning)|Terminating", line, re.I))
        results[metadata["sample_id"]] = record
    return results


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, help="Existing case root; defaults to config paths.cases")
    parser.add_argument("--profile", help="Defaults to the sole experiment profile")
    parser.add_argument("--output", type=Path, default=Path("inspection"))
    args = parser.parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]
    profiles = experiment["profiles"]
    profile = args.profile or (profiles[0] if len(profiles) == 1 else None)
    if not profile or profile not in config["resolutions"]:
        parser.error("Specify a valid --profile")
    root = (args.cases or path_for(config, "cases")).resolve()
    output = args.output.resolve()
    if output == root or output.is_relative_to(root):
        parser.error("Put the report outside the case root")
    output.mkdir(parents=True, exist_ok=True)
    records, terrains, forcings = [], {}, {}
    for tile in config["tiles"]:
        islands = generate_islands(experiment, Rectangle(tile["west"], tile["south"]))
        sea_states = generate_forcings(experiment)
        lon, lat = grid_coordinates(tile, config["resolutions"][profile])
        for island in islands:
            key = f"{tile['id']}__{island.terrain_id}"
            image = f"assets/terrain_{key}.png"
            depth, wet, _ = make_island_depth(lon, lat, island, experiment["terrain"]["base_depth_m"], config["grid"]["exception"])
            depth_plot(output / image, depth, wet, tile, f"{key}; {island.split}")
            terrains[key] = {**vars(island), "image": image, "shape": list(depth.shape)}
        for forcing in sea_states:
            key = f"{tile['id']}__{forcing.forcing_id}"
            image = f"assets/forcing_{key}.png"
            forcing_plot(output / image, forcing.wave, forcing.wind, tile, f"{key}; {forcing.family}")
            forcings[key] = {"image": image, "family": forcing.family, "split": forcing.split,
                             "wave_age": forcing.wave_age, "steepness": forcing.steepness,
                             **forcing_values(forcing.wave, forcing.wind)}
        for case in case_plan(islands, sea_states):
            records.append({**case, "sample_id": f"{profile}__{tile['id']}__{case['case_id']}",
                            "tile_id": tile["id"], "source": "plan", "status": "planned", "run": {},
                            "terrain_key": f"{tile['id']}__{case['terrain_id']}",
                            "forcing_key": f"{tile['id']}__{case['forcing_id']}", "checks": []})
    actual = existing_cases(root, profile, output)
    planned_ids = {record["sample_id"] for record in records}
    for record in records:
        if record["sample_id"] in actual:
            record.update(actual[record["sample_id"]])
    records.extend(record for key, record in actual.items() if key not in planned_ids)
    payload = clean_json({"generated_at": datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat(),
                          "profile": profile, "cases_root": str(root), "families": FAMILIES,
                          "physics": config["physics_commands"], "spectral_grid": config["spectral_grid"],
                          "convergence_policy": "record_only", "planned_count": len(planned_ids),
                          "prepared_count": len(actual), "valid_output_count": sum(bool(item.get("output_image")) for item in actual.values()),
                          "status_counts": dict(Counter(record["status"] for record in records)),
                          "split_counts": dict(Counter(record["split"] for record in records)),
                          "cases": records, "terrains": terrains, "forcings": forcings})
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    (output / "inspection.json").write_text(serialized, encoding="utf-8")
    template = Path(__file__).with_name("inspection_template.html").read_text(encoding="utf-8")
    html = template.replace("__INSPECTION_DATA__", serialized.replace("<", "\\u003c"))
    (output / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("planned_count", "prepared_count", "valid_output_count", "status_counts")}, ensure_ascii=False))
    print(f"Open in a browser: {output / 'index.html'}")


if __name__ == "__main__":
    main()
