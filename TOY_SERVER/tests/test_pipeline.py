"""Offline file contracts. Maintained, but not run during this static-only update."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from island_core import Rectangle, case_plan, generate_forcings, generate_islands, make_island_depth
from project import load_config
from request_era5 import requests_for
from swan_inputs import grid_coordinates, prepare_case
from swan_runtime import inspect_case


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config.json")
        self.experiment = self.config["experiment"]
        self.tile = self.config["tiles"][0]
        self.islands = generate_islands(self.experiment, Rectangle(self.tile["west"], self.tile["south"]))
        self.forcings = generate_forcings(self.experiment)

    def test_matrix_and_spectrum_settings(self):
        self.assertEqual(self.config["boundary"]["period"], "peak")
        self.assertEqual(self.config["boundary"]["jonswap_gamma"], self.experiment["forcing"]["jonswap_gamma"])
        self.assertEqual(self.forcings[0].wave.shape, (3, 2, 2))
        self.assertEqual(self.forcings[0].wind.shape, (2, 3, 3))

    def test_self_contained_nine_point_wind_and_corruption_checks(self):
        lon, lat = grid_coordinates(self.tile, self.config["resolutions"]["1km"])
        depth, wet, _ = make_island_depth(lon, lat, self.islands[0], 100, -999)
        case = case_plan(self.islands, self.forcings)[0]
        forcing = self.forcings[0]
        wind = np.arange(18.).reshape(2, 3, 3)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "case"
            prepare_case(directory, "sample", case, "1km", self.tile, lon, lat, depth, wet,
                         forcing.wave, wind, self.config)
            inspect_case(directory)
            np.testing.assert_allclose(np.loadtxt(directory / "wind.dat"), wind.reshape(6, 3))
            self.assertEqual((directory / "INPUT").read_text().count("BOUNDSPEC SEGMENT"), 4)
            prepare_case(directory, "sample", case, "1km", self.tile, lon, lat, depth, wet,
                         forcing.wave, wind, self.config)
            (directory / "wind.dat").write_text("corrupt")
            with self.assertRaises(ValueError):
                inspect_case(directory)

    def test_era5_request_preserves_mwp(self):
        config = load_config(ROOT / "era5_config.json")
        waves = [plan for plan in requests_for(config) if plan["group"] == "waves"]
        for plan in waves:
            self.assertIn("mean_wave_period", plan["request"]["variable"])
            self.assertIn("mean_wave_period_based_on_first_moment", plan["request"]["variable"])
            self.assertEqual(plan["request"]["grid"], [.5, .5])
