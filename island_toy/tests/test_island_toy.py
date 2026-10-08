from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from island_core import load_config, read_csv  # noqa: E402
from run_island_toy_swan import context, render_input  # noqa: E402


class IslandToyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "config.json")
        cls.terrains = read_csv(ROOT / "index" / "terrains.csv")
        cls.forcings = read_csv(ROOT / "index" / "forcings.csv")
        cls.cases = read_csv(ROOT / "index" / "cases.csv")

    def test_exact_split_and_case_counts(self):
        terrain_counts = {split: sum(row["terrain_split"] == split for row in self.terrains) for split in ("train", "validation", "test")}
        forcing_counts = {split: sum(row["forcing_split"] == split for row in self.forcings) for split in ("train", "validation", "test")}
        case_counts = {split: sum(row["split"] == split for row in self.cases) for split in ("train", "validation", "test")}
        self.assertEqual(terrain_counts, {"train": 24, "validation": 4, "test": 4})
        self.assertEqual(forcing_counts, {"train": 8, "validation": 2, "test": 2})
        self.assertEqual(case_counts, {"train": 192, "validation": 88, "test": 88})
        self.assertEqual(len({row["case_id"] for row in self.cases}), 368)

    def test_no_geometry_or_forcing_leakage(self):
        terrain_split = {row["terrain_id"]: row["terrain_split"] for row in self.terrains}
        forcing_split = {row["forcing_id"]: row["forcing_split"] for row in self.forcings}
        for row in self.cases:
            ts, fs = terrain_split[row["terrain_id"]], forcing_split[row["forcing_id"]]
            if row["split"] == "train":
                self.assertEqual((ts, fs), ("train", "train"))
            elif row["generalization"] == "new_island":
                self.assertEqual(fs, "train")
                self.assertEqual(ts, row["split"])
            elif row["generalization"] == "new_forcing":
                self.assertEqual(ts, "train")
                self.assertEqual(fs, row["split"])
            else:
                self.assertEqual((ts, fs), (row["split"], row["split"]))

    def test_terrain_is_100m_at_boundary_and_has_smooth_skirt(self):
        terrain = self.terrains[0]
        with xr.open_dataset(ROOT / "terrains" / "1km" / terrain["terrain_id"] / "grid.nc") as ds:
            depth = np.asarray(ds.depth.values); wet = np.asarray(ds.wet_mask.values, dtype=bool)
            distance = np.asarray(ds.distance_to_island_center.values)
        boundary = np.concatenate((depth[0], depth[-1], depth[:, 0], depth[:, -1]))
        np.testing.assert_allclose(boundary, 100.0)
        self.assertGreater(int((~wet).sum()), 0)
        radius = float(terrain["radius_m"]); skirt = float(terrain["skirt_width_m"])
        band = wet & (distance > radius) & (distance < radius + skirt)
        self.assertGreater(int(band.sum()), 5)
        order = np.argsort(distance[band])
        sampled = depth[band][order]
        self.assertGreater(float(sampled[-1]), float(sampled[0]))
        self.assertGreaterEqual(float(sampled.min()), float(terrain["shore_depth_m"]))

    def test_wind_current_coupling_is_exact(self):
        settings = self.config["forcing"]
        for row in self.forcings:
            wind = math.hypot(float(row["wind_u10_mps"]), float(row["wind_v10_mps"]))
            current = math.hypot(float(row["current_u_mps"]), float(row["current_v_mps"]))
            self.assertAlmostEqual(wind, float(row["wind_speed_mps"]), places=8)
            expected = float(settings["current_base_speed_mps"]) + float(settings["current_per_wind"]) * wind
            self.assertAlmostEqual(current, expected, places=8)

    def test_rendered_input_has_current_and_one_side_boundary(self):
        config, parent, cases = context(ROOT / "config.json", "1km", None, 1)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            text = render_input(cases[0], "1km", config, parent, Path(directory))
        self.assertIn("INPGRID CURRENT", text)
        self.assertIn("READINP CURRENT", text)
        self.assertEqual(text.count("BOUNDSPEC SIDE"), 1)
        self.assertIn("BOUND SHAPESPEC JONSWAP", text)
        self.assertIn("BREAKING CONSTANT ALPHA 1.0 GAMMA 0.73", text)
        self.assertNotIn("BREAKING BJ", text)

    def test_terrain_visualization_assets_exist(self):
        for terrain in self.terrains:
            directory = ROOT / "terrains" / "1km" / terrain["terrain_id"]
            self.assertGreater((directory / "terrain.png").stat().st_size, 0)
            with np.load(directory / "terrain.npz") as archive:
                self.assertEqual(archive["depth"].shape, archive["wet_mask"].shape)
                self.assertEqual(archive["depth"].shape, archive["distance_to_island_center"].shape)


if __name__ == "__main__":
    unittest.main()
