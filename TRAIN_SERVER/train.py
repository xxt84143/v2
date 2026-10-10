"""Train the island Hs U-Net from converted shards, with epoch checkpoints."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import random
import time

import numpy as np

from model_factory import build_model
from training_core import HERE, load_config, resolve_config_path, validate_dataset, write_json


class ShardDataset:
    def __init__(self, root, rows):
        self.root, self.rows = root, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        with np.load(self.root / self.rows[index]["shard"], allow_pickle=False) as shard:
            return tuple(torch.from_numpy(shard[name].copy()) for name in ("x", "y", "mask"))


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    import torch
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def masked_mse(prediction, target, mask):
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    weights = mask.float()
    return ((prediction.float() - target.float()).square() * weights).sum() / weights.sum().clamp_min(1)


def run_epoch(model, loader, device, amp, optimizer=None, scaler=None):
    import torch
    training = optimizer is not None
    model.train(training)
    total, pixels = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y, mask in loader:
            x, y, mask = (item.to(device, non_blocking=True) for item in (x, y, mask))
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                loss = masked_mse(model(x), y, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; checkpoint from last completed epoch is retained")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            count = int(mask.sum().item())
            total += float(loss.item()) * count
            pixels += count
    return total / pixels


def save_checkpoint(path, value):
    import torch
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_history(path, history):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, help="Total epoch count, including resumed epochs")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", type=Path, help="Resume last.pt into its original run directory")
    parser.add_argument("--check-model", action="store_true", help="Data and model forward checks only")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = dict(config["training"])
    for cli, key in ((args.epochs, "epochs"), (args.batch_size, "batch_size"),
                     (args.learning_rate, "learning_rate"), (args.num_workers, "num_workers"), (args.amp, "amp")):
        if cli is not None:
            settings[key] = cli
    for key in ("epochs", "batch_size"):
        if not isinstance(settings[key], int) or settings[key] <= 0:
            parser.error(f"{key} must be a positive integer")
    if settings["num_workers"] < 0 or not math.isfinite(settings["learning_rate"]) or settings["learning_rate"] <= 0:
        parser.error("Invalid worker count or learning rate")
    data = (args.data or resolve_config_path(config, config["paths"]["dataset"])).expanduser().resolve()
    output = (args.output or resolve_config_path(config, config["paths"]["output"])).expanduser().resolve()
    # Check before importing torch / allocating GPU memory.
    metadata, rows, report = validate_dataset(data)
    print(report, flush=True)
    import torch
    from torch.utils.data import DataLoader
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run check_environment.py; CPU requires explicit --device cpu")
    amp = bool(settings["amp"] and device.type == "cuda")
    seed_everything(int(settings["seed"]))
    repo = resolve_config_path(config, config["paths"].get("model_repo"))
    model = build_model(config["model"], 10, 1, repo).to(device)
    model.eval()
    with torch.no_grad():
        shape = list(model(torch.zeros((1, 10, *metadata["padded_shape"]), device=device)).shape)
    if shape != [1, 1, *metadata["padded_shape"]]:
        raise ValueError(f"Model changed output shape: {shape}")
    print({"device": str(device), "amp": amp, "output_shape": shape}, flush=True)
    if args.check_model:
        return
    if args.resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty; choose --output or resume its last.pt")
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    contract = {"profile": metadata["profile"], "backend": config["model"]["backend"],
                "model_settings": config["model"], "input_channels": metadata["input_channels"],
                "dataset_schema": metadata["schema_version"], "normalization": metadata["normalization"],
                "dataset_sha256": report["sha256"], "padded_shape": metadata["padded_shape"]}
    history, best, start = [], float("inf"), 1
    if args.resume:
        resume = args.resume.expanduser().resolve()
        if resume.parent != output or not (output / "best.pt").is_file():
            raise ValueError("Resume into the original output directory containing best.pt")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if any(checkpoint.get(key) != value for key, value in contract.items()):
            raise ValueError("Resume dataset or model differs from the saved checkpoint")
        previous = checkpoint["training_settings"]
        for key in ("batch_size", "learning_rate", "weight_decay", "seed", "amp"):
            if previous[key] != settings[key]:
                raise ValueError(f"Resume changed {key}; start a fresh run for a new experiment")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        history, best = checkpoint["history"], checkpoint["best_validation_loss"]
        start = checkpoint["epoch"] + 1
        if settings["epochs"] < start:
            raise ValueError("--epochs must exceed the checkpoint epoch")
    groups = {name: [row for row in rows if row["split"] == name] for name in ("train", "validation")}
    generator = torch.Generator()
    loaders = {name: DataLoader(ShardDataset(data, items), batch_size=settings["batch_size"],
               shuffle=name == "train", num_workers=settings["num_workers"],
               pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=generator)
               for name, items in groups.items()}
    write_json(output / "dataset_check.json", report)
    write_json(output / "run_config.json", {"model": config["model"], "training": settings,
               "data": str(data), "device": str(device), "torch": str(torch.__version__),
               "cuda_runtime": torch.version.cuda, "target": "Hs", "selection": "validation only"})
    for epoch in range(start, settings["epochs"] + 1):
        seed_everything(int(settings["seed"]) + epoch)
        generator.manual_seed(int(settings["seed"]) + epoch)
        started = time.perf_counter()
        train_loss = run_epoch(model, loaders["train"], device, amp, optimizer, scaler)
        validation_loss = run_epoch(model, loaders["validation"], device, amp)
        record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                  "validation_rmse_m": math.sqrt(validation_loss) * metadata["normalization"]["hs_scale_m"],
                  "seconds": time.perf_counter() - started}
        history.append(record)
        improved = validation_loss < best
        best = min(best, validation_loss)
        checkpoint = {**contract, "model": model.state_dict(), "epoch": epoch,
                      "validation_loss": validation_loss, "best_validation_loss": best}
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
        save_checkpoint(output / "last.pt", {**checkpoint, "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "history": history, "training_settings": settings})
        save_history(output / "history.csv", history)
        write_json(output / "summary.json", {"epoch": epoch, "samples": report["splits"],
                   "best_validation_loss": best, "best_validation_rmse_m": math.sqrt(best) * metadata["normalization"]["hs_scale_m"],
                   "test_evaluated": False, "selection_rule": "validation only", "device": str(device)})
        print(f"epoch {epoch:03d} train={train_loss:.6g} val={validation_loss:.6g} "
              f"val_RMSE={record['validation_rmse_m']:.4f}m seconds={record['seconds']:.2f}", flush=True)


if __name__ == "__main__":
    main()
