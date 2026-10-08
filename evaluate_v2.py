"""Evaluate a trained v2 Hs field model without changing its checkpoint.

The evaluator reports physical-unit, mask-aware metrics at both pooled-pixel
(micro) and equal-case (macro) levels.  Validation and test are deliberately
separate invocations so that test data need not be inspected during iteration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from model_factory import build_model
from v2_core import HERE, load_config, read_csv, resolve_config_path


METRIC_COLUMNS = (
    "mae_m", "rmse_m", "bias_m", "centered_rmse_m", "scatter_index",
    "pearson_r", "r2", "p90_abs_error_m", "p95_abs_error_m",
)


class EvaluationDataset:
    def __init__(self, root: Path, rows: list[dict[str, str]]):
        self.root = root
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        path = self.root / self.rows[index]["shard"]
        with np.load(path) as shard:
            x = torch.from_numpy(np.asarray(shard["x"], dtype=np.float32))
            y = torch.from_numpy(np.asarray(shard["y"], dtype=np.float32))
            mask = torch.from_numpy(np.asarray(shard["mask"], dtype=np.uint8))
            raw_shape = torch.from_numpy(np.asarray(shard["raw_shape"], dtype=np.int32))
        return x, y, mask, raw_shape, index


def field_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    """Return conventional continuous-field metrics for already-masked 1-D arrays."""
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    valid = np.isfinite(prediction) & np.isfinite(target)
    prediction, target = prediction[valid], target[valid]
    if target.size == 0:
        raise ValueError("No finite values supplied to field_metrics")
    residual = prediction - target
    bias = float(np.mean(residual))
    mse = float(np.mean(residual ** 2))
    centered_mse = float(np.mean((residual - bias) ** 2))
    target_mean = float(np.mean(target))
    target_centered = target - target_mean
    prediction_centered = prediction - float(np.mean(prediction))
    target_ss = float(np.sum(target_centered ** 2))
    pred_ss = float(np.sum(prediction_centered ** 2))
    cross = float(np.sum(target_centered * prediction_centered))
    abs_error = np.abs(residual)
    eps = np.finfo(np.float64).eps
    return {
        "n_valid": int(target.size),
        "target_mean_m": target_mean,
        "prediction_mean_m": float(np.mean(prediction)),
        "mae_m": float(np.mean(abs_error)),
        "rmse_m": math.sqrt(mse),
        "bias_m": bias,
        "relative_bias_pct": 100.0 * bias / target_mean if abs(target_mean) > eps else math.nan,
        "centered_rmse_m": math.sqrt(centered_mse),
        "scatter_index": math.sqrt(centered_mse) / abs(target_mean) if abs(target_mean) > eps else math.nan,
        "pearson_r": cross / math.sqrt(target_ss * pred_ss) if target_ss > eps and pred_ss > eps else math.nan,
        "r2": 1.0 - float(np.sum(residual ** 2)) / target_ss if target_ss > eps else math.nan,
        "p90_abs_error_m": float(np.quantile(abs_error, 0.90)),
        "p95_abs_error_m": float(np.quantile(abs_error, 0.95)),
    }


def bootstrap_mean_ci(values: np.ndarray, samples: int, rng: np.random.Generator,
                      confidence: float = 0.95) -> tuple[float, float]:
    """Case-level percentile bootstrap CI for a macro mean."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1 or samples <= 0:
        value = float(values.mean())
        return value, value
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def aggregate_group(records: list[dict[str, Any]], bootstrap_samples: int,
                    rng: np.random.Generator) -> dict[str, Any]:
    target = np.concatenate([record["_target"] for record in records])
    prediction = np.concatenate([record["_prediction"] for record in records])
    baseline = np.concatenate([record["_baseline"] for record in records])
    result: dict[str, Any] = {"n_cases": len(records)}
    result.update(field_metrics(prediction, target))
    baseline_metrics = field_metrics(baseline, target)
    for key in METRIC_COLUMNS:
        result[f"baseline_{key}"] = baseline_metrics[key]
    baseline_mse = float(baseline_metrics["rmse_m"]) ** 2
    result["mse_skill_vs_boundary_hs"] = (
        1.0 - float(result["rmse_m"]) ** 2 / baseline_mse if baseline_mse > 0 else math.nan
    )
    result["negative_prediction_fraction"] = (
        sum(int(record["_negative_count"]) for record in records) / int(result["n_valid"])
    )
    for metric in ("mae_m", "rmse_m", "bias_m"):
        values = np.asarray([record[metric] for record in records], dtype=np.float64)
        low, high = bootstrap_mean_ci(values, bootstrap_samples, rng)
        result[f"macro_{metric}_mean"] = float(np.mean(values))
        result[f"macro_{metric}_median"] = float(np.median(values))
        result[f"macro_{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        result[f"macro_{metric}_ci95_low"] = low
        result[f"macro_{metric}_ci95_high"] = high
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if not key.startswith("_") and key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fields:
                value = row.get(key, "")
                clean[key] = "" if isinstance(value, float) and not math.isfinite(value) else value
            writer.writerow(clean)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_results(output: Path, records: list[dict[str, Any]], groups: list[dict[str, Any]],
                 checkpoint: Path, max_scatter_points: int, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    target = np.concatenate([record["_target"] for record in records])
    prediction = np.concatenate([record["_prediction"] for record in records])
    residual = prediction - target
    if target.size > max_scatter_points:
        choice = rng.choice(target.size, max_scatter_points, replace=False)
        target_plot, prediction_plot = target[choice], prediction[choice]
    else:
        target_plot, prediction_plot = target, prediction

    limit = max(float(np.max(target_plot)), float(np.max(prediction_plot)), 0.1)
    fig, ax = plt.subplots(figsize=(6.4, 5.5), dpi=150)
    image = ax.hist2d(target_plot, prediction_plot, bins=90, range=[[0, limit], [min(0, float(prediction_plot.min())), limit]], cmap="viridis")
    ax.plot([0, limit], [0, limit], "r--", linewidth=1.2, label="1:1")
    ax.set(xlabel="SWAN Hs (m)", ylabel="Predicted Hs (m)", title="Observed-predicted density")
    ax.legend()
    fig.colorbar(image[3], ax=ax, label="pixel count")
    fig.tight_layout(); fig.savefig(output / "parity.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    ax.hist(residual, bins=80, color="#4472c4", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(float(np.mean(residual)), color="#d62728", linestyle="--", label=f"bias={np.mean(residual):.3f} m")
    ax.set(xlabel="Prediction - SWAN Hs (m)", ylabel="valid pixel count", title="Residual distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "residual_distribution.png", dpi=220); plt.close(fig)

    subgroup = [row for row in groups if row["generalization"] != "all"]
    if subgroup:
        labels = [row["generalization"].replace("_", "\n") for row in subgroup]
        positions = np.arange(len(subgroup)); width = 0.36
        fig, ax = plt.subplots(figsize=(max(6.4, 1.7 * len(subgroup)), 4.8), dpi=150)
        ax.bar(positions - width / 2, [row["rmse_m"] for row in subgroup], width, label="model")
        ax.bar(positions + width / 2, [row["baseline_rmse_m"] for row in subgroup], width, label="boundary-Hs baseline")
        ax.set_xticks(positions, labels)
        ax.set(ylabel="RMSE (m)", title="RMSE by generalization group")
        ax.legend(); fig.tight_layout(); fig.savefig(output / "rmse_by_generalization.png", dpi=220); plt.close(fig)

    ordered = sorted(records, key=lambda record: float(record["rmse_m"]))
    colors = {"new_island": "#4c78a8", "new_forcing": "#f58518", "new_both": "#e45756", "seen_island_seen_forcing": "#72b7b2"}
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
    ax.bar(np.arange(len(ordered)), [row["rmse_m"] for row in ordered],
           color=[colors.get(row.get("generalization", ""), "#777777") for row in ordered])
    ax.set(xlabel="cases ordered by RMSE", ylabel="RMSE (m)", title="Per-case error distribution")
    fig.tight_layout(); fig.savefig(output / "case_rmse.png", dpi=220); plt.close(fig)

    selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.5), dpi=150)
    labels = ("best", "median", "worst")
    for row_index, (label, record) in enumerate(zip(labels, selected)):
        ny, nx = record["_raw_shape"]
        mask = record["_mask"][:ny, :nx]
        truth = np.ma.masked_where(~mask, record["_target_field"][:ny, :nx])
        pred = np.ma.masked_where(~mask, record["_prediction_field"][:ny, :nx])
        error = np.ma.masked_where(~mask, pred - truth)
        vmax = max(float(truth.max()), float(pred.max()), 0.1)
        emax = max(float(np.max(np.abs(error.compressed()))), 0.01)
        for column, (field, title, cmap, vmin, upper) in enumerate((
            (truth, "SWAN", "viridis", 0.0, vmax),
            (pred, "prediction", "viridis", 0.0, vmax),
            (error, "error", "RdBu_r", -emax, emax),
        )):
            image = axes[row_index, column].imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=upper)
            axes[row_index, column].set_title(f"{label}: {record['case_id']} | {title}")
            axes[row_index, column].set_xticks([]); axes[row_index, column].set_yticks([])
            fig.colorbar(image, ax=axes[row_index, column], fraction=0.046, pad=0.04, label="m")
        axes[row_index, 0].set_ylabel(f"{record['generalization']}\nRMSE={record['rmse_m']:.3f} m")
    fig.tight_layout(); fig.savefig(output / "spatial_examples.png", dpi=220); plt.close(fig)

    history = checkpoint.parent / "history.csv"
    if history.is_file():
        rows = read_csv(history)
        if rows:
            fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=150)
            ax.plot([int(row["epoch"]) for row in rows], [float(row["train_loss"]) for row in rows], label="train")
            ax.plot([int(row["epoch"]) for row in rows], [float(row["validation_loss"]) for row in rows], label="validation")
            ax.set(xlabel="epoch", ylabel="masked MSE (normalized)", title="Training history")
            ax.legend(); fig.tight_layout(); fig.savefig(output / "training_history.png", dpi=220); plt.close(fig)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.6f}"
    lines = [
        f"# {summary['split'].title()} evaluation", "",
        f"- Checkpoint: `{summary['checkpoint']['path']}`", 
        f"- Cases / valid pixels: {overall['n_cases']} / {overall['n_valid']}",
        f"- Inference: {summary['runtime']['seconds']:.3f} s ({summary['runtime']['cases_per_second']:.3f} cases/s)",
        "", "## Primary results", "",
        "| Metric | Model | Boundary-Hs baseline |", "| --- | ---: | ---: |",
        f"| MAE (m) | {fmt(overall['mae_m'])} | {fmt(overall['baseline_mae_m'])} |",
        f"| RMSE (m) | {fmt(overall['rmse_m'])} | {fmt(overall['baseline_rmse_m'])} |",
        f"| Bias (m) | {fmt(overall['bias_m'])} | {fmt(overall['baseline_bias_m'])} |",
        f"| Pearson r | {fmt(overall['pearson_r'])} | {fmt(overall['baseline_pearson_r'])} |",
        f"| R2 | {fmt(overall['r2'])} | {fmt(overall['baseline_r2'])} |",
        f"| Scatter index | {fmt(overall['scatter_index'])} | {fmt(overall['baseline_scatter_index'])} |",
        "",
        f"MSE skill relative to the boundary-Hs constant-field baseline: **{fmt(overall['mse_skill_vs_boundary_hs'])}**.",
        "", "## Statistical convention", "",
        "Micro metrics pool all valid wet pixels. Macro metrics first score each case and then give cases equal weight. "
        "The reported 95% confidence intervals are percentile bootstraps over cases, not over spatial pixels. "
        "Scatter index is centered RMSE divided by mean SWAN Hs. Bias is prediction minus SWAN. "
        "R2 is evaluated against the pooled SWAN mean. Percentage errors are intentionally omitted near calm-water values.",
        "", "Use validation for iteration and checkpoint/model choices. Run the frozen choice on test once for final reporting.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    item = argparse.ArgumentParser(description=__doc__)
    item.add_argument("--config", type=Path, default=HERE / "config.json")
    item.add_argument("--data", type=Path, required=True, help="Dataset directory containing metadata.json and manifest.csv")
    item.add_argument("--checkpoint", type=Path, required=True, help="Selected checkpoint, normally best.pt")
    item.add_argument("--output", type=Path, help="Default: CHECKPOINT_DIR/evaluation/SPLIT")
    item.add_argument("--split", choices=("validation", "test", "train"), default="validation")
    item.add_argument("--device", default="auto")
    item.add_argument("--batch-size", type=int, default=8)
    item.add_argument("--num-workers", type=int, default=0)
    item.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision for inference")
    item.add_argument("--clip-min-m", type=float, help="Optional declared post-processing; raw predictions are the default")
    item.add_argument("--bootstrap-samples", type=int, default=2000)
    item.add_argument("--seed", type=int, default=20260925)
    item.add_argument("--max-scatter-points", type=int, default=100000)
    item.add_argument("--save-predictions", action="store_true")
    item.add_argument("--no-plots", action="store_true")
    item.add_argument("--case-id", action="append", dest="case_ids")
    item.add_argument("--limit", type=int, help="Diagnostic subset only; do not use for reported results")
    item.add_argument("--max-rmse-m", type=float, help="Optional CI/production gate; exit 2 if violated")
    item.add_argument("--min-r2", type=float, help="Optional CI/production gate; exit 2 if violated")
    item.add_argument("--min-skill", type=float, help="Optional CI/production gate; exit 2 if violated")
    item.add_argument("--check", action="store_true", help="Load contracts/model/checkpoint and perform one dummy forward only")
    return item


def main() -> None:
    args = parser().parse_args()
    import torch
    from torch.utils.data import DataLoader

    config_path = args.config.expanduser().resolve()
    data = args.data.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config = load_config(config_path)
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    rows = [row for row in read_csv(data / "manifest.csv") if row["split"] == args.split]
    if args.case_ids:
        wanted = set(args.case_ids)
        rows = [row for row in rows if row["case_id"] in wanted]
        missing = wanted.difference(row["case_id"] for row in rows)
        if missing:
            raise ValueError(f"Requested cases are absent from split {args.split}: {sorted(missing)}")
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError(f"No cases selected for split {args.split}")
    missing_shards = [str(data / row["shard"]) for row in rows if not (data / row["shard"]).is_file()]
    if missing_shards:
        raise FileNotFoundError(f"Missing {len(missing_shards)} shard(s), first: {missing_shards[0]}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    channels = list(metadata["input_channels"])
    if list(checkpoint.get("input_channels", [])) != channels:
        raise ValueError("Checkpoint input channels do not exactly match dataset metadata")
    if checkpoint.get("dataset_schema") != metadata.get("schema_version"):
        raise ValueError("Checkpoint and dataset schema differ")
    if checkpoint.get("profile") != metadata.get("profile"):
        raise ValueError("Checkpoint and dataset profile differ")
    if checkpoint.get("backend") != config["model"]["backend"]:
        raise ValueError("Checkpoint backend differs from config model backend")
    repo = resolve_config_path(config, config["paths"].get("model_repo"))
    model = build_model(config["model"], len(channels), 1, repo)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = model.to(device).eval()
    padded_shape = tuple(int(value) for value in metadata["padded_shape"])
    dummy = torch.zeros((1, len(channels), *padded_shape), dtype=torch.float32, device=device)
    with torch.no_grad():
        dummy_output = model(dummy)
    if list(dummy_output.shape) != [1, 1, *padded_shape]:
        raise ValueError(f"Model output shape mismatch: {list(dummy_output.shape)}")
    contract = {
        "status": "ok", "split": args.split, "selected_cases": len(rows),
        "profile": metadata["profile"], "dataset_schema": metadata["schema_version"],
        "backend": checkpoint["backend"], "checkpoint_epoch": checkpoint["epoch"],
        "device": str(device), "input_channels": len(channels), "output_shape": list(dummy_output.shape),
    }
    if args.check:
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return

    hs_scale = float(metadata["normalization"]["hs_scale_m"])
    wave_hs_index = channels.index("wave_hs")
    loader = DataLoader(
        EvaluationDataset(data, rows), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    output = (args.output or (checkpoint_path.parent / "evaluation" / args.split)).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        (output / "predictions").mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for x, y, mask, raw_shapes, indices in loader:
            x_device = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                predicted = model(x_device)
            raw_prediction_m = predicted.detach().cpu().numpy()[:, 0] * hs_scale
            target_m = y.numpy()[:, 0] * hs_scale
            valid = mask.numpy()[:, 0].astype(bool)
            baseline_m = x.numpy()[:, wave_hs_index] * hs_scale
            for batch_index, row_index in enumerate(indices.tolist()):
                row = rows[row_index]
                raw = raw_prediction_m[batch_index]
                prediction = np.maximum(raw, args.clip_min_m) if args.clip_min_m is not None else raw
                target = target_m[batch_index]
                active = valid[batch_index] & np.isfinite(target) & np.isfinite(prediction)
                if not np.any(active):
                    raise ValueError(f"{row['case_id']} has no valid evaluation pixels")
                record: dict[str, Any] = {
                    "case_id": row["case_id"], "terrain_id": row.get("terrain_id", ""),
                    "forcing_id": row.get("forcing_id", ""), "split": row["split"],
                    "generalization": row.get(metadata.get("generalization_column", "generalization"), "unspecified"),
                    "negative_prediction_fraction": float(np.mean(raw[active] < 0)),
                }
                record.update(field_metrics(prediction[active], target[active]))
                base_metrics = field_metrics(baseline_m[batch_index][active], target[active])
                record["baseline_rmse_m"] = base_metrics["rmse_m"]
                record["baseline_mae_m"] = base_metrics["mae_m"]
                baseline_mse = float(base_metrics["rmse_m"]) ** 2
                record["mse_skill_vs_boundary_hs"] = 1.0 - float(record["rmse_m"]) ** 2 / baseline_mse if baseline_mse > 0 else math.nan
                record.update({
                    "_target": target[active].astype(np.float32),
                    "_prediction": prediction[active].astype(np.float32),
                    "_baseline": baseline_m[batch_index][active].astype(np.float32),
                    "_negative_count": int(np.count_nonzero(raw[active] < 0)),
                    "_target_field": target.astype(np.float32),
                    "_prediction_field": prediction.astype(np.float32),
                    "_mask": active,
                    "_raw_shape": tuple(int(value) for value in raw_shapes[batch_index].tolist()),
                })
                records.append(record)
                if args.save_predictions:
                    ny, nx = record["_raw_shape"]
                    np.savez_compressed(
                        output / "predictions" / f"{row['case_id']}.npz",
                        prediction_hs_m=prediction[:ny, :nx].astype(np.float32),
                        target_hs_m=target[:ny, :nx].astype(np.float32),
                        valid_mask=active[:ny, :nx].astype(np.uint8),
                    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    rng = np.random.default_rng(args.seed)
    group_rows: list[dict[str, Any]] = []
    overall = aggregate_group(records, args.bootstrap_samples, rng)
    group_rows.append({"split": args.split, "generalization": "all", **overall})
    group_names = sorted({record["generalization"] for record in records})
    for name in group_names:
        selected = [record for record in records if record["generalization"] == name]
        group_rows.append({"split": args.split, "generalization": name,
                           **aggregate_group(selected, args.bootstrap_samples, rng)})

    gates = {
        "max_rmse_m": None if args.max_rmse_m is None else {"threshold": args.max_rmse_m, "value": overall["rmse_m"], "passed": overall["rmse_m"] <= args.max_rmse_m},
        "min_r2": None if args.min_r2 is None else {"threshold": args.min_r2, "value": overall["r2"], "passed": overall["r2"] >= args.min_r2},
        "min_skill": None if args.min_skill is None else {"threshold": args.min_skill, "value": overall["mse_skill_vs_boundary_hs"], "passed": overall["mse_skill_vs_boundary_hs"] >= args.min_skill},
    }
    gate_values = [item for item in gates.values() if item is not None]
    summary = {
        "schema_version": "toy-v2-evaluation-1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "subset_limited": args.limit is not None or args.case_ids is not None,
        "dataset": {"path": str(data), "schema": metadata["schema_version"], "profile": metadata["profile"], "input_channels": channels},
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path), "epoch": checkpoint["epoch"],
                       "training_validation_loss_normalized": checkpoint["validation_loss"], "backend": checkpoint["backend"]},
        "postprocessing": {"clip_min_m": args.clip_min_m},
        "statistics": {"bootstrap_unit": "case", "bootstrap_samples": args.bootstrap_samples,
                       "confidence": 0.95, "seed": args.seed},
        "runtime": {"device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda,
                    "amp": bool(args.amp and device.type == "cuda"), "seconds": elapsed,
                    "cases_per_second": len(records) / elapsed},
        "overall": overall, "groups": group_rows, "quality_gates": gates,
        "quality_gates_passed": all(item["passed"] for item in gate_values) if gate_values else None,
    }
    write_csv(output / "metrics_by_case.csv", records)
    write_csv(output / "metrics_by_group.csv", group_rows)
    safe_summary = json_safe(summary)
    (output / "summary.json").write_text(json.dumps(safe_summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(output / "report.md", safe_summary)
    if not args.no_plots:
        plot_results(output, records, group_rows, checkpoint_path, args.max_scatter_points, args.seed)
    print(json.dumps({
        "output": str(output), "split": args.split, "cases": len(records),
        "rmse_m": overall["rmse_m"], "mae_m": overall["mae_m"], "bias_m": overall["bias_m"],
        "r2": overall["r2"], "mse_skill_vs_boundary_hs": overall["mse_skill_vs_boundary_hs"],
        "quality_gates_passed": summary["quality_gates_passed"],
    }, ensure_ascii=False, indent=2))
    if gate_values and not all(item["passed"] for item in gate_values):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
