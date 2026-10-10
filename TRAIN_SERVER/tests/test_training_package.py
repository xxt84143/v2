"""Small synthetic fixtures check contracts and checkpoint recovery, not model skill."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training_core import CHANNELS, read_csv, validate_dataset, write_json


def fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": "toy-v2-dataset-matrix-2", "profile": "gebco15s",
                "sample_count": 4, "input_channel_count": 10, "input_channels": CHANNELS,
                "padded_shape": [128, 128], "normalization": {"hs_scale_m": 5.,
                "period_scale_s": 15., "wind_scale_mps": 25.}, "target": "SWAN HSIGN / hs_scale_m",
                "generalization_column": "generalization", "convergence_policy": "record_only"}
    write_json(root / "metadata.json", metadata)
    specifications = [("train", "T1", "F1", "seen_island_seen_forcing"),
                      ("train", "T1", "F2", "seen_island_seen_forcing"),
                      ("validation", "T2", "F3", "new_both"), ("test", "T3", "F4", "new_both")]
    rows = []
    for index, (split, terrain, forcing, group) in enumerate(specifications):
        wet = np.zeros((1, 128, 128), dtype=np.uint8)
        wet[:, :121, :121] = 1
        wet[:, 55:65, 55:65] = 0
        x = np.zeros((10, 128, 128), dtype=np.float32)
        x[0] = wet[0] * .6
        x[1] = wet[0]
        yy, xx = np.meshgrid(np.linspace(-1, 1, 121), np.linspace(-1, 1, 121), indexing="ij")
        x[2, :121, :121], x[3, :121, :121] = xx, yy
        x[4, :121, :121], x[5, :121, :121], x[7, :121, :121] = .2, .5, 1
        x[8, :121, :121] = .3
        y = (wet * (.2 + .01 * index)).astype(np.float32)
        wave = np.stack([np.full((2, 2), 1.), np.full((2, 2), 7.5), np.zeros((2, 2))]).astype(np.float32)
        wind = np.stack([np.full((3, 3), 7.5), np.zeros((3, 3))]).astype(np.float32)
        case = f"CASE{index}"
        np.savez_compressed(root / f"{case}.npz", x=x, y=y, mask=wet, wave=wave, wind=wind,
                            raw_shape=np.asarray([121, 121]))
        rows.append({"sample_id": case, "case_id": case, "terrain_id": terrain, "forcing_id": forcing,
                     "split": split, "generalization": group, "shard": f"{case}.npz",
                     "converged": "False" if index == 0 else "True"})
    write_manifest(root, rows)
    return rows


def write_manifest(root, rows):
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class DatasetContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "data"
        self.rows = fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def mutate_shard(self, change):
        path = self.root / "CASE0.npz"
        with np.load(path) as data:
            arrays = {name: data[name].copy() for name in data.files}
        change(arrays)
        np.savez_compressed(path, **arrays)

    def test_unconverged_cases_are_retained_and_fingerprinted(self):
        _, _, report = validate_dataset(self.root)
        self.assertEqual(report["samples"], 4)
        self.assertEqual(report["convergence"]["false"], 1)
        self.mutate_shard(lambda arrays: arrays["y"].__setitem__((0, 0, 0), .21))
        self.assertNotEqual(report["sha256"], validate_dataset(self.root)[2]["sha256"])

    def test_shard_cannot_escape_dataset(self):
        self.rows[0]["shard"] = "../CASE0.npz"
        write_manifest(self.root, self.rows)
        with self.assertRaisesRegex(ValueError, "path escapes"):
            validate_dataset(self.root)

    def test_incomplete_transfer_fails(self):
        (self.root / "CASE0.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            validate_dataset(self.root)

    def test_nonfinite_shard_fails(self):
        self.mutate_shard(lambda arrays: arrays["x"].__setitem__((0, 0, 0), np.nan))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            validate_dataset(self.root)

    def test_padding_is_excluded(self):
        self.mutate_shard(lambda arrays: arrays["x"].__setitem__((0, 127, 127), 1))
        with self.assertRaisesRegex(ValueError, "padding"):
            validate_dataset(self.root)

    def test_validation_new_terrain_cannot_leak_from_training(self):
        self.rows[2]["terrain_id"] = "T1"
        write_manifest(self.root, self.rows)
        with self.assertRaisesRegex(ValueError, "present in training"):
            validate_dataset(self.root)

    def test_validation_test_holdouts_are_distinct(self):
        self.rows[3]["terrain_id"] = "T2"
        write_manifest(self.root, self.rows)
        with self.assertRaisesRegex(ValueError, "share held-out"):
            validate_dataset(self.root)

    def test_partial_delivery_is_not_trainable(self):
        rows = self.rows[:2]
        write_manifest(self.root, rows)
        metadata = json.loads((self.root / "metadata.json").read_text())
        metadata["sample_count"] = 2
        write_json(self.root / "metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "requires train"):
            validate_dataset(self.root)
        self.assertEqual(validate_dataset(self.root, require_splits=False)[2]["samples"], 2)


class TrainingIntegrationTests(unittest.TestCase):
    def test_masked_loss_has_no_land_gradient(self):
        import torch
        from train import masked_mse
        prediction = torch.tensor([[[[2., 100.]]]], requires_grad=True)
        target = torch.tensor([[[[1., 0.]]]])
        mask = torch.tensor([[[[1, 0]]]])
        loss = masked_mse(prediction, target, mask)
        loss.backward()
        self.assertEqual(loss.item(), 1.)
        self.assertEqual(prediction.grad[0, 0, 0, 1].item(), 0.)

    def test_train_resume_evaluate_and_dataset_change_rejection(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = directory / "dataset"
            fixture(data)
            resumed, full = directory / "resumed", directory / "full"
            environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            def run(script, *args, success=True):
                result = subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                                        env=environment, capture_output=True, text=True, timeout=120)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result
            common = ("--data", data, "--device", "cpu", "--no-amp", "--num-workers", 0, "--batch-size", 2)
            run("train.py", *common, "--output", resumed, "--epochs", 1)
            first = torch.load(resumed / "last.pt", weights_only=True)
            run("train.py", *common, "--output", resumed, "--resume", resumed / "last.pt", "--epochs", 2)
            run("train.py", *common, "--output", full, "--epochs", 2)
            last = torch.load(resumed / "last.pt", weights_only=True)
            reference = torch.load(full / "last.pt", weights_only=True)
            self.assertEqual(last["epoch"], 2)
            self.assertEqual(len(read_csv(resumed / "history.csv")), 2)
            self.assertTrue(any(not torch.equal(first["model"][key], last["model"][key]) for key in last["model"]))
            for key, value in last["model"].items():
                self.assertTrue(torch.equal(value, reference["model"][key]), key)
            summary = json.loads((resumed / "summary.json").read_text())
            self.assertFalse(summary["test_evaluated"])
            run("evaluate.py", "--data", data, "--checkpoint", resumed / "best.pt", "--device", "cpu",
                "--split", "test", "--save-predictions", "--bootstrap-samples", 10)
            evaluation = resumed / "evaluation/test"
            metrics = json.loads((evaluation / "summary.json").read_text())
            self.assertEqual(metrics["overall"]["n_cases"], 1)
            self.assertGreaterEqual(metrics["overall"]["rmse_m"], 0.)
            self.assertTrue((evaluation / "spatial_examples.png").is_file())
            self.assertTrue((evaluation / "predictions/CASE3.npz").is_file())
            metadata = json.loads((data / "metadata.json").read_text())
            metadata["normalization"]["hs_scale_m"] = 6.
            write_json(data / "metadata.json", metadata)
            result = run("train.py", *common, "--output", resumed, "--resume", resumed / "last.pt", "--epochs", 3, success=False)
            self.assertIn("Resume dataset or model differs", result.stderr)
            result = run("evaluate.py", "--data", data, "--checkpoint", resumed / "best.pt", "--device", "cpu", success=False)
            self.assertIn("Checkpoint differs", result.stderr)


if __name__ == "__main__":
    unittest.main()
