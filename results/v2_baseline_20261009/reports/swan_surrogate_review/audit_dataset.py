"""Read-only audit of the archived dataset and existing validation metrics.

No archive extraction, model loading, prediction or training. Test labels are not
read. Run from any directory with a Python environment containing numpy/pandas.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "TOY_SERVER"))
from island_core import Rectangle, case_plan, forcing_to_row, generate_forcings, generate_islands, make_island_depth
from swan_inputs import grid_coordinates
from forcing_arrays import model_inputs


def records(frame):
    return json.loads(frame.to_json(orient="records"))


def table(frame, digits=4):
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value)
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |"]
    rows.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def aggregate_metrics(frame, columns):
    result = []
    all_sse = (frame.n_valid * frame.rmse_m ** 2).sum()
    for key, sub in frame.groupby(columns, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(columns, key))
        count = sub.n_valid.sum()
        mse = (sub.n_valid * sub.rmse_m ** 2).sum() / count
        base_mse = (sub.n_valid * sub.baseline_rmse_m ** 2).sum() / count
        bias = (sub.n_valid * sub.bias_m).sum() / count
        row.update(n_cases=len(sub), n_valid=int(count), rmse_m=np.sqrt(mse),
                   bias_m=bias, centered_rmse_m=np.sqrt(max(0.0, mse-bias*bias)),
                   baseline_rmse_m=np.sqrt(base_mse), skill=1-mse/base_mse,
                   fraction_total_sse=(sub.n_valid * sub.rmse_m ** 2).sum()/all_sse,
                   target_mean_m=(sub.n_valid * sub.target_mean_m).sum()/count,
                   prediction_mean_m=(sub.n_valid * sub.prediction_mean_m).sum()/count,
                   macro_rmse_m=sub.rmse_m.mean(),
                   negative_fraction=(sub.n_valid * sub.negative_prediction_fraction).sum()/count)
        result.append(row)
    return pd.DataFrame(result)


def main():
    config = json.loads((ROOT / "TOY_SERVER/config.json").read_text(encoding="utf-8"))
    exp = config["experiment"]
    forcings = generate_forcings(exp)
    force_lookup = {f.forcing_id: f for f in forcings}
    tile = config["tiles"][0]
    islands = generate_islands(exp, Rectangle(tile["west"], tile["south"]))
    island_lookup = {i.terrain_id: i for i in islands}
    plan = pd.DataFrame(case_plan(islands, forcings))
    lon, lat = grid_coordinates(tile, config["resolutions"]["gebco15s"])
    archive = ROOT / "dataset_gebco15s.tar.gz"
    with tarfile.open(archive, "r:gz") as tar:
        metadata_bytes = tar.extractfile("gebco15s/metadata.json").read()
        manifest_bytes = tar.extractfile("gebco15s/manifest.csv").read()
        metadata = json.loads(metadata_bytes)
        manifest = pd.read_csv(io.BytesIO(manifest_bytes))
        dataset_signature = hashlib.sha256()
        for name, payload in [("metadata.json", metadata_bytes), ("manifest.csv", manifest_bytes)]:
            dataset_signature.update(name.encode())
            dataset_signature.update(hashlib.sha256(payload).hexdigest().encode())
        shard_digests = {}
        names = {m.name for m in tar.getmembers() if m.isfile()}
        manifest["shard_exists"] = manifest.shard.map(lambda s: f"gebco15s/{s}" in names)
        bycase = manifest.set_index("case_id")
        observations, force_seen, terrain_seen = [], {}, {}
        x_max_error, wave_max_error, wind_max_error = 0.0, 0.0, 0.0
        raw_shapes, mask_errors, varying_forcings, varying_terrains = set(), [], [], []
        for member in tar.getmembers():
            if not member.name.endswith(".npz"):
                continue
            case_id = Path(member.name).stem
            row = bycase.loc[case_id]
            shard_bytes = tar.extractfile(member).read()
            shard_digests[member.name.removeprefix("gebco15s/")] = hashlib.sha256(shard_bytes).hexdigest()
            with np.load(io.BytesIO(shard_bytes), allow_pickle=False) as shard:
                wave, wind, x, mask = shard["wave"], shard["wind"], shard["x"], shard["mask"].astype(bool)
                raw_shape = tuple(shard["raw_shape"].tolist())
                raw_shapes.add(raw_shape)
                fid, tid = row.forcing_id, row.terrain_id
                force_digest = hashlib.sha256(wave.tobytes()+wind.tobytes()).hexdigest()
                terrain_digest = hashlib.sha256(x[:2].tobytes()).hexdigest()
                if fid in force_seen and force_seen[fid] != force_digest:
                    varying_forcings.append(case_id)
                if tid in terrain_seen and terrain_seen[tid] != terrain_digest:
                    varying_terrains.append(case_id)
                force_seen[fid], terrain_seen[tid] = force_digest, terrain_digest
                expected_force = force_lookup[fid]
                wave_max_error = max(wave_max_error, float(np.abs(wave-expected_force.wave.astype(np.float32)).max()))
                wind_max_error = max(wind_max_error, float(np.abs(wind-expected_force.wind.astype(np.float32)).max()))
                depth, wet, _ = make_island_depth(lon, lat, island_lookup[tid], exp["terrain"]["base_depth_m"], config["grid"]["exception"])
                expected_x, _ = model_inputs(wave, wind, depth, wet, exp["normalization"], period_name="tp")
                x_max_error = max(x_max_error, float(np.abs(x[:, :raw_shape[0], :raw_shape[1]]-expected_x).max()))
                if not np.array_equal(mask[0, :raw_shape[0], :raw_shape[1]], wet):
                    mask_errors.append(case_id)
                obs = dict(case_id=case_id, n_wet=int(mask.sum()))
                # Preserve untouched test evaluation: no access to shard['y'] for test.
                if row.split != "test":
                    target = shard["y"][mask] * metadata["normalization"]["hs_scale_m"]
                    obs.update(target_mean_m=float(target.mean()), target_std_m=float(target.std()),
                               target_min_m=float(target.min()), target_max_m=float(target.max()),
                               target_finite=bool(np.isfinite(target).all()))
                observations.append(obs)

    for shard_path in manifest.shard:
        dataset_signature.update(shard_path.encode())
        dataset_signature.update(shard_digests[shard_path].encode())
    expected_signature = json.loads((ROOT / "toy_runs/hs_unet/dataset_check.json").read_text(encoding="utf-8"))["sha256"]

    merged = plan.merge(manifest.drop(columns=["terrain_id", "forcing_id", "split", "generalization"]), on="case_id", how="left", indicator=True)
    merged["available"] = merged._merge == "both"
    merged["converged_bool"] = merged.converged == True
    merged["unconverged"] = merged.available & ~merged.converged_bool
    merged.to_csv(OUT / "planned_vs_actual_cases.csv", index=False, encoding="utf-8-sig")
    identity_columns = ["case_id", "terrain_id", "forcing_id", "split", "generalization"]
    identity = manifest[identity_columns].merge(plan[identity_columns], on="case_id", suffixes=("_actual", "_plan"))
    identity_mismatch = sum(any(row[f"{c}_actual"] != row[f"{c}_plan"] for c in identity_columns[1:]) for row in identity.to_dict("records"))
    split_counts = merged.groupby(["split", "generalization"], sort=False).agg(planned=("case_id", "size"), actual=("available", "sum"), unconverged=("unconverged", "sum")).reset_index()
    split_counts["missing"] = split_counts.planned - split_counts.actual
    forcing_counts = merged.groupby(["forcing_id", "forcing_split", "family"], sort=True).agg(planned=("case_id", "size"), actual=("available", "sum"), unconverged=("unconverged", "sum")).reset_index()
    forcing_counts["missing"] = forcing_counts.planned - forcing_counts.actual
    family_counts = merged.groupby(["split", "family"], sort=False).agg(planned=("case_id", "size"), actual=("available", "sum"), unconverged=("unconverged", "sum")).reset_index()
    family_counts["missing"] = family_counts.planned-family_counts.actual
    force_values = []
    for f in forcings:
        row = forcing_to_row(f)
        angle = np.deg2rad(f.wave[2])
        row.update(mean_direction_deg=float(np.rad2deg(np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())) % 360),
                   mean_wind_speed_mps=float(np.hypot(*f.wind).mean()),
                   wave_hs_min_m=float(f.wave[0].min()), wave_hs_max_m=float(f.wave[0].max()),
                   wave_tp_min_s=float(f.wave[1].min()), wave_tp_max_s=float(f.wave[1].max()))
        force_values.append(row)
    forcing_values = pd.DataFrame(force_values)
    forcing_counts = forcing_counts.merge(forcing_values.drop(columns=["forcing_split", "family"]), on="forcing_id")
    forcing_counts.to_csv(OUT / "forcing_coverage.csv", index=False, encoding="utf-8-sig")
    unconverged = merged[merged.unconverged][["case_id", "terrain_id", "forcing_id", "split", "family", "convergence_percent", "iterations"]]
    val = pd.read_csv(ROOT / "toy_runs/hs_unet/evaluation/validation/metrics_by_case.csv")
    val = val.merge(plan[["case_id", "family"]], on="case_id").merge(manifest[["case_id", "converged"]], on="case_id")
    forcing_metrics = aggregate_metrics(val, ["forcing_id", "family"])
    forcing_group_metrics = aggregate_metrics(val, ["generalization", "forcing_id", "family"])
    convergence_metrics = aggregate_metrics(val, ["converged"])
    for name, frame in [("validation_metrics_by_forcing", forcing_metrics), ("validation_metrics_by_generalization_forcing", forcing_group_metrics), ("validation_metrics_by_convergence", convergence_metrics)]:
        frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    obs = pd.DataFrame(observations).merge(manifest[["case_id", "terrain_id", "forcing_id", "split", "generalization"]], on="case_id")
    obs.to_csv(OUT / "dataset_case_descriptives.csv", index=False, encoding="utf-8-sig")
    cross_counts = merged.pivot_table(index="forcing_id", columns="split", values="available", aggfunc="sum", fill_value=0).reset_index()
    audit = dict(archive_sha256=hashlib.file_digest(archive.open("rb"), "sha256").hexdigest(),
                 dataset_signature_same_algorithm_as_training=dataset_signature.hexdigest(),
                 saved_training_dataset_signature=expected_signature,
                 dataset_matches_training_signature=(dataset_signature.hexdigest() == expected_signature),
                 source="dataset_gebco15s.tar.gz, current TOY_SERVER/config.json and deterministic generators; existing validation CSV",
                 metadata=metadata, planned_count=len(plan), actual_count=len(manifest), missing_count=int((~merged.available).sum()),
                 identity_mismatch_count=identity_mismatch, missing_archive_shards=int((~manifest.shard_exists).sum()),
                 unique_forcings=len(force_seen), unique_terrains=len(terrain_seen), raw_shapes=sorted(raw_shapes),
                 wave_max_abs_difference_vs_current_generator=wave_max_error,
                 wind_max_abs_difference_vs_current_generator=wind_max_error,
                 x_max_abs_difference_vs_current_generator=x_max_error, mask_mismatch_cases=mask_errors,
                 nonconstant_forcing_ids_within_same_id=varying_forcings, nonconstant_terrains_within_same_id=varying_terrains,
                 split_counts=records(split_counts), forcing_counts=records(forcing_counts), family_counts=records(family_counts),
                 forcing_split_counts=records(cross_counts), unconverged=records(unconverged),
                 validation_forcing_metrics=records(forcing_metrics), validation_generalization_forcing_metrics=records(forcing_group_metrics),
                 validation_convergence_metrics=records(convergence_metrics),
                 test_labels_read=False, predictions_run=False)
    (OUT / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    parts = ["# 数据与验证误差定量审计", "",
             "本审计直接只读打开压缩包，不解压、不训练、不运行预测，也不读取 test 的 y 数组。计划来自当前确定性生成器，验证误差来自已有 metrics_by_case.csv。缺失表示没有进入已交付数据集，不能由此单独判定运行失败或超时。", "",
             f"计划 {len(plan)} 例，实得 {len(manifest)} 例，缺失 {int((~merged.available).sum())} 例。原始网格 {sorted(raw_shapes)}；已补零至 128 × 128。",
             f"实际 forcing 数 {len(force_seen)}，地形数 {len(terrain_seen)}。已存在样本与当前生成器：wave 最大差 {wave_max_error}，wind 最大差 {wind_max_error}，x 最大差 {x_max_error}，mask 不一致 {len(mask_errors)} 例，ID/划分不一致 {identity_mismatch} 例。", "",
             f"按 TRAIN_SERVER/training_core.py 的字节哈希算法复算 dataset signature：`{dataset_signature.hexdigest()}`。与 toy_runs/hs_unet/dataset_check.json 一致：{dataset_signature.hexdigest() == expected_signature}。", "",
             "## 划分覆盖", "", table(split_counts), "", "## 每组 forcing 的覆盖和物理条件", "",
             table(forcing_counts[["forcing_id", "forcing_split", "family", "planned", "actual", "missing", "unconverged", "mean_hs_m", "mean_tp_s", "mean_direction_deg", "mean_wind_speed_mps"]]), "",
             "## 按 split 和族的缺失", "", table(family_counts), "",
             "## 未收敛但进入数据集的样本", "", table(unconverged), "",
             "## 验证按 forcing 汇总", "", "RMSE 从每例 SSE 与有效像元数重构，不是简单平均每例 RMSE；fraction_total_sse 是该 forcing 对全部验证 SSE 的贡献比例。", "",
             table(forcing_metrics[["forcing_id", "family", "n_cases", "rmse_m", "bias_m", "centered_rmse_m", "baseline_rmse_m", "skill", "fraction_total_sse"]]), "",
             "## 验证按泛化类别和 forcing 汇总", "", table(forcing_group_metrics[["generalization", "forcing_id", "n_cases", "rmse_m", "bias_m", "baseline_rmse_m", "skill"]]), "",
             "## 验证按收敛状态汇总", "", table(convergence_metrics), "",
             "## 解释边界", "", "当前配置与已存在数组匹配，只能验证输入的来源一致，不能追回缺失样本的运行原因。压缩包没有全部 608 个 run_status.json/PRINT，因此不要将 missing 自动称为 unconverged。有效独立 forcing 数应按训练中出现的 forcing_id 计数，不能把同一 forcing 在 24 个地形上的重复组合算成 24 组新海况。", "",
             "复现：`C:/Users/15507/.conda/envs/DTP_env/python.exe reports/swan_surrogate_review/audit_dataset.py`。所有 CSV/JSON/Markdown 均写入本报告目录。"]
    (OUT / "quantitative_audit_notes.md").write_text("\n".join(parts)+"\n", encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ["planned_count", "actual_count", "missing_count", "unique_forcings", "unique_terrains", "wave_max_abs_difference_vs_current_generator", "wind_max_abs_difference_vs_current_generator", "x_max_abs_difference_vs_current_generator"]}, indent=2))
    print(forcing_counts[["forcing_id", "family", "planned", "actual", "missing", "mean_hs_m", "mean_tp_s", "mean_direction_deg", "mean_wind_speed_mps"]].to_string(index=False))
    print(forcing_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
