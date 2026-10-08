"""Train an external U-Net backend on one v2 resolution profile."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from v2_core import HERE, load_config, read_csv, resolve_config_path
from model_factory import build_model


class ShardDataset:
    def __init__(self, root: Path, rows: list[dict[str, str]]):
        self.root, self.rows = root, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        with np.load(self.root / self.rows[index]["shard"]) as shard:
            return tuple(torch.from_numpy(np.asarray(shard[name])) for name in ("x", "y", "mask"))


def masked_mse(prediction, target, mask):
    weights = mask.float()
    return (((prediction - target) ** 2) * weights).sum() / weights.sum().clamp_min(1.0)


def evaluate(model, loader, device):
    import torch
    model.eval()
    total, weight = 0.0, 0
    with torch.no_grad():
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            value = masked_mse(model(x), y, mask)
            total += float(value.item()) * len(x)
            weight += len(x)
    return total / max(weight, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--profile", choices=("gebco15s", "1km"), required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--check-model", action="store_true")
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader
    config = load_config(args.config)
    data = args.data or (HERE / "dataset" / args.profile)
    output = args.output or (HERE / "training" / args.profile)
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    rows = read_csv(data / "manifest.csv")
    groups = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    repo = resolve_config_path(config, config["paths"].get("model_repo"))
    model = build_model(config["model"], int(metadata["input_channel_count"]), 1, repo)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = model.to(device)
    dummy = torch.zeros((1, int(metadata["input_channel_count"]), *metadata["padded_shape"]), device=device)
    with torch.no_grad():
        output_shape = list(model(dummy).shape)
    if output_shape != [1, 1, *metadata["padded_shape"]]:
        raise ValueError(f"External model changed field shape: {output_shape}")
    print({"backend": config["model"]["backend"], "device": str(device), "output_shape": output_shape})
    if args.check_model:
        return
    if not groups["train"] or not groups["validation"] or not groups["test"]:
        raise ValueError({name: len(value) for name, value in groups.items()})
    loaders = {
        name: DataLoader(ShardDataset(data, items), batch_size=args.batch_size, shuffle=name == "train", num_workers=args.num_workers)
        for name, items in groups.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    output.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y, mask in loaders["train"]:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            value = masked_mse(model(x), y, mask)
            value.backward()
            optimizer.step()
            total += float(value.item()) * len(x)
            count += len(x)
        record = {
            "epoch": epoch, "train_loss": total / count,
            "validation_loss": evaluate(model, loaders["validation"], device),
        }
        history.append(record)
        if record["validation_loss"] < best:
            best = record["validation_loss"]
            torch.save({
                "model": model.state_dict(), "epoch": epoch, "validation_loss": best,
                "profile": args.profile, "backend": config["model"]["backend"],
                "input_channels": metadata["input_channels"], "dataset_schema": metadata["schema_version"],
            }, output / "best.pt")
        print(f"epoch {epoch:03d} train={record['train_loss']:.6g} val={record['validation_loss']:.6g}", flush=True)
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    summary = {
        "profile": args.profile, "backend": config["model"]["backend"], "device": str(device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "samples": {name: len(value) for name, value in groups.items()},
        "best_validation_loss": best, "test_loss_at_selected_checkpoint": None,
        "selection_rule": "validation only; test remains untouched during training",
    }
    checkpoint = torch.load(output / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    summary["test_loss_at_selected_checkpoint"] = evaluate(model, loaders["test"], device)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
