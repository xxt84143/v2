"""Synthetic geometry and joint-condition checks; no SWAN process."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from island_core import Rectangle, case_plan, generate_forcings, generate_islands, load_config, make_island_depth


class IslandToyTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config.json")
        self.islands = generate_islands(self.config, Rectangle(**self.config["tile"]))
        self.forcings = generate_forcings(self.config)
        self.cases = case_plan(self.islands, self.forcings)

    def test_split_counts_and_families(self):
        self.assertEqual(len(self.cases), 608)
        self.assertEqual(sum(case["split"] == "train" for case in self.cases), 288)
        for split in ("train", "validation", "test"):
            self.assertEqual(len({item.family for item in self.forcings if item.split == split}), 4)

    def test_no_geometry_or_forcing_leakage(self):
        terrain = {item.terrain_id: item.split for item in self.islands}
        forcing = {item.forcing_id: item.split for item in self.forcings}
        for case in self.cases:
            ts, fs = terrain[case["terrain_id"]], forcing[case["forcing_id"]]
            if case["split"] == "train":
                self.assertEqual((ts, fs), ("train", "train"))
            elif case["generalization"] == "new_island":
                self.assertEqual((ts, fs), (case["split"], "train"))
            elif case["generalization"] == "new_forcing":
                self.assertEqual((ts, fs), ("train", case["split"]))
            else:
                self.assertEqual((ts, fs), (case["split"], case["split"]))

    def test_windsea_height_period_wind_relation(self):
        for forcing in self.forcings:
            self.assertEqual(forcing.wind.shape, (2, 3, 3))
            if forcing.family == "aligned_windsea":
                corner_speed = np.hypot(*forcing.wind[:, ::2, ::2])
                np.testing.assert_allclose(9.81 * forcing.wave[1] / (2 * np.pi * corner_speed), forcing.wave_age)
                steepness = forcing.wave[0] / (9.81 * forcing.wave[1]**2 / (2*np.pi))
                np.testing.assert_allclose(steepness, forcing.steepness)
            if forcing.family == "wind_only":
                self.assertTrue((forcing.wave[0] == 0).all())
            if forcing.family == "swell_no_wind":
                self.assertTrue((forcing.wind == 0).all())

    def test_terrain_boundary_and_radial_skirt(self):
        island = self.islands[0]
        depth, wet, distance = make_island_depth(np.linspace(121.5, 122, 121), np.linspace(23.5, 24, 121), island, 100, -999)
        boundary = np.concatenate((depth[0], depth[-1], depth[:, 0], depth[:, -1]))
        np.testing.assert_allclose(boundary, 100)
        self.assertTrue((~wet).any())
        band = wet & (distance < island.radius_m + island.skirt_width_m)
        ordered = depth[band][np.argsort(distance[band])]
        self.assertTrue((np.diff(ordered) >= -1e-5).all())
