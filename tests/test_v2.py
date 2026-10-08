from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v2_core import (  # noqa: E402
    WIND_NAMES, enumerate_tiles, load_config, make_grid_spec, padded_shape,
    read_swan_block, reconstruct_wind_grid, resolve_config_path, split_for_case, wind_points,
)


class V2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "config.json")
        cls.era5_path = resolve_config_path(cls.config, cls.config["paths"]["era5"])
        with xr.open_dataset(cls.era5_path) as ds:
            cls.tiles = {tile.tile_id: tile for tile in enumerate_tiles(ds)}
        cls.tile = cls.tiles["E121p5_N23p5"]

    def test_source_has_expected_valid_cell_catalog(self):
        self.assertEqual(len(self.tiles), 173)
        self.assertEqual(self.tile.east - self.tile.west, 0.5)
        self.assertEqual(self.tile.north - self.tile.south, 0.5)

    def test_resolution_shapes_and_exact_corners(self):
        fine = make_grid_spec(self.tile, "gebco15s", self.config["resolutions"]["gebco15s"])
        coarse = make_grid_spec(self.tile, "1km", self.config["resolutions"]["1km"])
        self.assertEqual(fine.shape, (121, 121))
        self.assertEqual(coarse.shape, (57, 52))
        for spec in (fine, coarse):
            self.assertEqual(spec.lon[0], self.tile.west)
            self.assertEqual(spec.lon[-1], self.tile.east)
            self.assertEqual(spec.lat[0], self.tile.south)
            self.assertEqual(spec.lat[-1], self.tile.north)
        self.assertEqual(padded_shape(fine.shape, 16), (128, 128))
        self.assertEqual(padded_shape(coarse.shape, 16), (64, 64))

    def test_five_wind_controls_are_preserved_exactly(self):
        values = {name: float(index + 1) for index, name in enumerate(WIND_NAMES)}
        result = reconstruct_wind_grid(self.tile, values)
        location_to_index = {
            (self.tile.center[0], self.tile.south): (0, 1),
            (self.tile.east, self.tile.center[1]): (1, 2),
            (self.tile.center[0], self.tile.north): (2, 1),
            (self.tile.west, self.tile.center[1]): (1, 0),
            self.tile.center: (1, 1),
        }
        for name, location in wind_points(self.tile).items():
            iy, ix = location_to_index[location]
            self.assertAlmostEqual(float(result[iy, ix]), values[name])
        self.assertTrue(np.isfinite(result).all())

    def test_split_is_case_stable_across_tiles_and_resolutions(self):
        first = split_for_case("C0042", self.config["split"])
        self.assertEqual(first, split_for_case("C0042", self.config["split"]))
        self.assertIn(first, {"train", "validation", "test"})

    def test_swan_reader_expands_vector_wind_to_two_blocks(self):
        quantities = ["XP", "YP", "DEPTH", "HSIGN", "TM01", "TM02", "RTP", "DIR", "PDIR", "WIND"]
        blocks = []
        for block in range(11):
            blocks.append("".join(f"{block + value / 10:12.4E}" for value in range(6)) + "\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "block.tab"
            path.write_text("".join(blocks), encoding="ascii", newline="")
            hs = read_swan_block(path, (2, 3), quantities, "HSIGN")
        np.testing.assert_allclose(hs.ravel(), 3.0 + np.arange(6) / 10.0)


if __name__ == "__main__":
    unittest.main()
