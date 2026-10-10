"""Read existing validation predictions; never train or evaluate the test set.

Run from any directory with Python + numpy + matplotlib. Outputs are
diagnostics, not a new model evaluation: target-informed corrections are labelled
as oracle decompositions and cannot be used as performance claims.
"""
from pathlib import Path
import csv
import io
import json
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EVAL = ROOT / "toy_runs/hs_unet/evaluation/validation"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def metrics(p, t, b):
    e = p.astype(np.float64) - t
    be = b.astype(np.float64) - t
    return {"n": int(t.size), "rmse_m": float(np.sqrt(np.mean(e**2))),
            "bias_m": float(e.mean()), "mae_m": float(np.abs(e).mean()),
            "baseline_rmse_m": float(np.sqrt(np.mean(be**2)))}


def main():
    rows = read_csv(EVAL / "metrics_by_case.csv")
    archive_data = {}
    metadata = None
    wanted = {r["case_id"] for r in rows}
    # A streaming read avoids extracting the archive or repeated gzip seeks.
    with tarfile.open(ROOT / "dataset_gebco15s.tar.gz", "r|gz") as archive:
        for member in archive:
            if member.isfile() and Path(member.name).name == "metadata.json":
                metadata = json.load(archive.extractfile(member))
            if member.isfile() and Path(member.name).suffix == ".npz" and Path(member.name).stem in wanted:
                with np.load(io.BytesIO(archive.extractfile(member).read()), allow_pickle=False) as z:
                    archive_data[Path(member.name).stem] = {k: z[k] for k in ["x", "y", "mask", "wave", "wind"]}
    assert set(archive_data) == wanted
    scale = float(metadata["normalization"]["hs_scale_m"])
    cap = float(metadata["normalization"].get("depth_cap_m", metadata["normalization"]["depth_scale_m"]))
    reference = float(metadata["normalization"]["depth_reference_m"])
    case_records, fields, buckets = [], {}, {}
    agreement = {"max_target_difference_m": 0., "mask_mismatches": 0, "max_csv_rmse_difference_m": 0.}
    for row in rows:
        cid = row["case_id"]
        with np.load(EVAL / "predictions" / f"{cid}.npz", allow_pickle=False) as z:
            p, t, m = z["prediction_hs_m"], z["target_hs_m"], z["valid_mask"].astype(bool)
        data = archive_data[cid]
        ny, nx = p.shape
        x = data["x"][:, :ny, :nx]
        b = x[4] * scale
        agreement["max_target_difference_m"] = max(agreement["max_target_difference_m"], float(np.max(np.abs(t - data["y"][0, :ny, :nx] * scale))))
        agreement["mask_mismatches"] += int(np.count_nonzero(m != data["mask"][0, :ny, :nx]))
        depth = reference * np.expm1(x[0].astype(np.float64) * np.log1p(cap/reference))
        pp, tt, bb = p[m].astype(np.float64), t[m].astype(np.float64), b[m].astype(np.float64)
        stat = metrics(pp, tt, bb)
        agreement["max_csv_rmse_difference_m"] = max(agreement["max_csv_rmse_difference_m"], abs(stat["rmse_m"] - float(row["rmse_m"])))
        error = pp - tt
        stat.update({"case_id": cid, "terrain_id": row["terrain_id"], "forcing_id": row["forcing_id"],
                     "generalization": row["generalization"], "bias_squared_m2": float(error.mean()**2),
                     "spatial_centered_mse_m2": float(np.var(error)),
                     "target_spatial_std_m": float(np.std(tt)), "prediction_spatial_std_m": float(np.std(pp)),
                     "target_mean_m": float(tt.mean()), "prediction_mean_m": float(pp.mean()),
                     "spatial_r": float(np.corrcoef(pp,tt)[0,1])})
        case_records.append(stat)
        iy, ix = np.indices(p.shape)
        edge = np.minimum.reduce([iy, ix, ny-1-iy, nx-1-ix]) < 5
        regions = {"all": m, "depth_lt20m": m & (depth < 20),
                   "depth20to80m": m & (depth >= 20) & (depth < 80),
                   "depth_ge80m": m & (depth >= 80), "edge5nodes": m & edge,
                   "interior": m & ~edge}
        for name, mask in regions.items():
            for group in ["all", row["generalization"], row["forcing_id"]]:
                buckets.setdefault((group, name), []).append((p[mask], t[mask], b[mask]))
        fields[cid] = (p, t, m, b)

    region_records = []
    for (group, region), chunks in buckets.items():
        values = [np.concatenate([c[i] for c in chunks]) for i in range(3)]
        if values[0].size:
            region_records.append({"group": group, "region": region, **metrics(*values)})

    decomposition = []
    for group in ["all", "new_island", "new_forcing", "new_both", "F013", "F014", "F015"]:
        selected = [r for r in case_records if group == "all" or r["generalization"] == group or r["forcing_id"] == group]
        if not selected:
            continue
        weights = np.array([r["n"] for r in selected], dtype=float)
        total = np.average([r["rmse_m"]**2 for r in selected], weights=weights)
        bias2 = np.average([r["bias_squared_m2"] for r in selected], weights=weights)
        space = np.average([r["spatial_centered_mse_m2"] for r in selected], weights=weights)
        assert abs(total-bias2-space) < 1e-12
        decomposition.append({"group": group, "n_cases": len(selected), "mse_m2": float(total),
                              "case_mean_bias_mse_m2": float(bias2), "spatial_mse_m2": float(space),
                              "case_mean_bias_share": float(bias2/total),
                              "oracle_case_debiased_rmse_m": float(np.sqrt(space)),
                              "median_spatial_r": float(np.median([r["spatial_r"] for r in selected])),
                              "wins_over_baseline": sum(r["rmse_m"] < r["baseline_rmse_m"] for r in selected)})

    total_sse = sum(r["n"] * r["rmse_m"]**2 for r in case_records)
    forcing_sse = {f: sum(r["n"] * r["rmse_m"]**2 for r in case_records if r["forcing_id"] == f) / total_sse
                   for f in sorted({r["forcing_id"] for r in case_records})}
    p = np.concatenate([v[0][v[2]] for v in fields.values()]).astype(float)
    t = np.concatenate([v[1][v[2]] for v in fields.values()]).astype(float)
    b = np.concatenate([v[3][v[2]] for v in fields.values()]).astype(float)
    result = {"validation_artifact_agreement": agreement, "decomposition": decomposition,
              "forcing_sse_fraction": forcing_sse,
              "raw": metrics(p,t,b), "clip_zero_diagnostic": metrics(np.maximum(p,0),t,b),
              "oracle_global_debiased_rmse_m": float(np.std(p-t)),
              "case_records": case_records, "regions": region_records,
              "note": "Oracle debiasing uses validation targets; not deployable, not a new performance score."}
    (OUT / "prediction_diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, values in [("case_diagnostics.csv", case_records), ("region_diagnostics.csv", region_records)]:
        with (OUT / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.7), layout="constrained")
    hist = read_csv(ROOT / "toy_runs/hs_unet/history.csv")
    epochs = [int(r["epoch"]) for r in hist]
    axes[0,0].plot(epochs, [scale*np.sqrt(float(r["train_loss"])) for r in hist], label="train (online)")
    axes[0,0].plot(epochs, [float(r["validation_rmse_m"]) for r in hist], label="validation")
    axes[0,0].axvline(65, color="gray", ls="--", lw=1)
    axes[0,0].set(xlabel="Epoch", ylabel="RMSE (m)", title="A. Training fits; validation stalls")
    axes[0,0].legend()
    groups = ["new_island", "new_forcing", "new_both"]
    summary = json.loads((EVAL / "summary.json").read_text(encoding="utf-8"))
    stats = {r["generalization"]:r for r in summary["groups"]}
    pos=np.arange(3)
    axes[0,1].bar(pos-.18,[stats[g]["rmse_m"] for g in groups], .36, label="model")
    axes[0,1].bar(pos+.18,[stats[g]["baseline_rmse_m"] for g in groups], .36, label="boundary baseline")
    axes[0,1].set(xticks=pos,xticklabels=groups,ylabel="RMSE (m)",title="B. Failure is forcing generalization")
    axes[0,1].legend()
    colors={"new_island":"#277DA1", "new_forcing":"#F8961E", "new_both":"#F94144"}
    for group in groups:
        chosen=[r for r in case_records if r["generalization"]==group]
        axes[1,0].scatter([r["target_mean_m"] for r in chosen], [r["prediction_mean_m"] for r in chosen], s=25,alpha=.7,label=group,color=colors[group])
    axes[1,0].plot([0,3],[0,3],color="black",ls="--",lw=1)
    axes[1,0].set(xlabel="SWAN case mean Hs (m)",ylabel="Predicted case mean Hs (m)",title="C. Many unseen forcings lose amplitude")
    axes[1,0].legend(fontsize=8)
    dec={r["group"]:r for r in decomposition}
    axes[1,1].bar(pos,[dec[g]["case_mean_bias_mse_m2"] for g in groups],label="case mean bias squared")
    axes[1,1].bar(pos,[dec[g]["spatial_mse_m2"] for g in groups],bottom=[dec[g]["case_mean_bias_mse_m2"] for g in groups],label="within-case residual variance")
    axes[1,1].set(xticks=pos,xticklabels=groups,ylabel="MSE (m²)",title="D. Exact MSE decomposition")
    axes[1,1].legend(fontsize=8)
    fig.savefig(OUT / "diagnostic_overview.png",dpi=180); plt.close(fig)

    selected = ["IT000329", "IT000339", "IT000393"]
    fig,axes=plt.subplots(len(selected),4,figsize=(14,10),layout="constrained")
    lookup={r["case_id"]:r for r in case_records}
    for i,cid in enumerate(selected):
        p,t,m,b=fields[cid]; bias=lookup[cid]["bias_m"]
        vmax=max(t[m].max(),p[m].max(),b[m].max())
        for j,(field,title) in enumerate([(t,"SWAN"),(p,"Model"),(b,"Boundary baseline"),(p-t-bias,"Error after removing case bias")]):
            upper=max(np.abs(field[m]).max(),.05) if j==3 else vmax
            im=axes[i,j].imshow(np.ma.masked_where(~m,field),origin="lower",cmap="RdBu_r" if j==3 else "viridis",vmin=-upper if j==3 else 0,vmax=upper)
            axes[i,j].set_title(title,fontsize=10);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            fig.colorbar(im,ax=axes[i,j],shrink=.8,label="m")
        axes[i,0].set_ylabel(f"{cid} / {lookup[cid]['forcing_id']}\nBias={bias:+.3f} m")
    fig.savefig(OUT / "spatial_diagnostics.png",dpi=180);plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k not in {"case_records","regions"}},indent=2))


if __name__ == "__main__":
    main()
