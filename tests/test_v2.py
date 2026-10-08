"""Offline contracts using synthetic fixtures; no original ERA5/GEBCO files."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forcing_arrays import model_inputs, resize, validate
from v2_core import Tile, load_config, make_grid_spec, padded_shape, read_swan_block, split_for_case


class V2ContractTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config.json")
        self.tile = Tile("test", 121.5, 122., 23.5, 24., ((1, 0), (1, 1), (0, 1), (0, 0)))

    def test_resolution_shapes_and_exact_corners(self):
        self.assertEqual(make_grid_spec(self.tile, "gebco15s", self.config["resolutions"]["gebco15s"]).shape, (121, 121))
        self.assertEqual(make_grid_spec(self.tile, "1km", self.config["resolutions"]["1km"]).shape, (57, 52))
        self.assertEqual(padded_shape((57, 52), 16), (64, 64))

    def test_all_nine_wind_nodes_survive_feature_resize(self):
        wind = np.arange(18.).reshape(2, 3, 3)
        resized = resize(wind, (5, 5))
        np.testing.assert_allclose(resized[:, ::2, ::2], wind)

    def test_wave_direction_wrap_and_land_depth_are_safe(self):
        wave = np.stack((np.ones((2, 2)), np.full((2, 2), 8.), np.array([[359., 1.], [359., 1.]])))
        depth = np.full((5, 5), 100.)
        depth[2, 2] = -999
        wet = depth > 0
        inputs, names = model_inputs(wave, np.zeros((2, 3, 3)), depth, wet, self.config["normalization"])
        self.assertEqual(inputs.shape, (10, 5, 5))
        self.assertTrue(np.isfinite(inputs).all())
        self.assertEqual(inputs[0, 2, 2], 0.)
        self.assertAlmostEqual(inputs[names.index("wave_dir_cos"), 0, 2], 1.)

    def test_legacy_flat_controls_are_rejected(self):
        with self.assertRaises(ValueError):
            validate(np.zeros((4, 3)), np.zeros((5, 2)))

    def test_split_is_case_stable(self):
        self.assertEqual(split_for_case("C0042", self.config["split"]), split_for_case("C0042", self.config["split"]))

    def test_swan_reader_expands_vector_wind(self):
        quantities = ["XP", "YP", "DEPTH", "HSIGN", "WIND"]
        blocks = ["".join(f"{block + value / 10:12.4E}" for value in range(6)) + "\n" for block in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "block.tab"
            path.write_text("".join(blocks), encoding="ascii", newline="")
            hs = read_swan_block(path, (2, 3), quantities, "HSIGN")
        np.testing.assert_allclose(hs.ravel(), 3 + np.arange(6) / 10)
