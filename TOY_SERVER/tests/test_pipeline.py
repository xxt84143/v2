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
from project import load_config, sample_times
from era5_jobs import download_settings, requests_for, split_plan
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
        variables = {variable for plan in waves for variable in plan["request"]["variable"]}
        self.assertIn("mean_wave_period", variables)
        self.assertIn("mean_wave_period_based_on_first_moment", variables)
        for plan in waves:
            self.assertEqual(len(plan["request"]["variable"]), 1)
            self.assertEqual(plan["request"]["grid"], [.5, .5])

    def test_download_plan_month_and_hour_boundaries(self):
        config = load_config(ROOT / "era5_config.json")
        config["sampling"] = {"start_utc": "2024-02-28T21:00:00", "end_utc": "2024-03-01T03:00:00", "step_hours": 3}
        plans = requests_for(config)
        actual = {(plan["request"]["year"][0], plan["request"]["month"][0], day, hour)
                  for plan in plans if plan["group"] == "wind"
                  for day in plan["request"]["day"] for hour in plan["request"]["time"]}
        expected = {(stamp.strftime("%Y"), stamp.strftime("%m"), stamp.strftime("%d"), stamp.strftime("%H:%M"))
                    for stamp in sample_times(config)}
        self.assertEqual(actual, expected)
        self.assertIn(("2024", "02", "29", "00:00"), actual)

    def test_split_variables_before_dates_and_reject_six_workers(self):
        config = load_config(ROOT / "era5_config.json")
        config["era5"]["download"]["variables_per_request"] = 2
        plan = next(plan for plan in requests_for(config) if plan["group"] == "wind")
        children = split_plan(plan)
        self.assertEqual([child["request"]["variable"] for child in children],
                         [[plan["request"]["variable"][0]], [plan["request"]["variable"][1]]])
        self.assertTrue(all(child["request"]["day"] == plan["request"]["day"] for child in children))
        config["era5"]["download"]["workers"] = 6
        with self.assertRaises(ValueError):
            download_settings(config)
