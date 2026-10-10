"""Physical limits, feature isolation, data migration and checkpoint compatibility."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from forcing_arrays import model_inputs, terrain_feature
from migrate_dataset import convert_dataset
from training_core import validate_dataset, RELATIVE_DEPTH_CHANNELS
from wave_geometry import relative_depth, recover_log_depth, RELATIVE_DEPTH_SCHEMA
from test_training_package import fixture


class RelativeDepthPhysicsTests(unittest.TestCase):
    def test_dispersion_residual_over_shallow_to_deep_water(self):
        h, period = np.meshgrid(np.geomspace(.01, 10000, 40), np.geomspace(2, 30, 20))
        ratio = relative_depth(h, period)
        z = 2 * np.pi * ratio
        q = (2 * np.pi / period)**2 * h / 9.81
        np.testing.assert_allclose(z * np.tanh(z), q, rtol=5e-13, atol=1e-14)

    def test_shallow_water_limit(self):
        h, period = .0001, 12.
        self.assertAlmostEqual(float(relative_depth(h, period)), np.sqrt(h / 9.81) / period, delta=1e-9)

    def test_deep_water_limit(self):
        h, period = 10000., 8.
        wavelength = 9.81 * period**2 / (2 * np.pi)
        self.assertAlmostEqual(float(relative_depth(h, period)), h / wavelength, places=11)

    def test_deeper_water_or_shorter_period_increases_relative_depth(self):
        values = relative_depth(np.asarray([1., 10., 100.]), 8.)
        self.assertTrue(np.all(np.diff(values) > 0))
        values = relative_depth(20., np.asarray([4., 8., 16.]))
        self.assertTrue(np.all(np.diff(values) < 0))

    def test_land_exceptions_are_zero_and_not_used_as_water(self):
        h = np.asarray([[10., -999.], [np.nan, 20.]])
        wet = np.asarray([[True, False], [False, True]])
        values = relative_depth(h, 8., wet)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all(values[~wet] == 0))
        with self.assertRaises(ValueError):
            relative_depth(np.asarray([-1., 0., np.nan]), 8.)

    def test_period_and_gravity_are_validated(self):
        for period in (0., -1., np.nan):
            with self.assertRaises(ValueError):
                relative_depth(10., period)
        with self.assertRaises(ValueError):
            relative_depth(10., 8., gravity_mps2=0)

    def test_only_first_channel_changes(self):
        wave = np.stack((np.ones((2, 2)), np.asarray([[5., 8.], [11., 14.]]), np.full((2, 2), 30.)))
        wind = np.arange(18.).reshape(2, 3, 3)
        depth = np.full((5, 5), 200.)
        depth[2, 2] = -999.
        wet = depth > 0
        norm = {"hs_scale_m": 5., "period_scale_s": 15., "wind_scale_mps": 25., "depth_scale_m": 100.}
        baseline, _ = model_inputs(wave, wind, depth, wet, norm, period_name="tp")
        modified, names = model_inputs(wave, wind, depth, wet, {**norm, "terrain_channel": "relative_depth"}, period_name="tp")
        np.testing.assert_array_equal(baseline[1:], modified[1:])
        self.assertEqual(names, RELATIVE_DEPTH_CHANNELS)
        self.assertEqual(modified[0, 2, 2], 0)
        # The physical path uses 200 m even though the baseline clips at 100 m.
        self.assertAlmostEqual(float(modified[0, 0, 0]), float(relative_depth(200., 5.)), places=6)

    def test_zero_boundary_uses_declared_reference_not_dummy_period(self):
        wave = np.stack((np.zeros((2, 2)), np.full((2, 2), 3.), np.zeros((2, 2))))
        norm = {"terrain_channel": "relative_depth", "zero_boundary_reference_period_s": 8.}
        first, _ = terrain_feature(wave, np.full((5, 5), 10.), np.ones((5, 5), bool), norm)
        wave[1] = 20.
        second, _ = terrain_feature(wave, np.full((5, 5), 10.), np.ones((5, 5), bool), norm)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, relative_depth(10., 8.))

    def test_legacy_inverse_recovers_capped_depth(self):
        norm = {"depth_scale_m": 100., "depth_reference_m": 10.}
        h = np.asarray([[1., 10., 100.]])
        channel = np.log1p(h / 10.) / np.log1p(100. / 10.)
        np.testing.assert_allclose(recover_log_depth(channel, np.ones_like(h, bool), norm), h, rtol=1e-14)


class RelativeDepthMigrationTests(unittest.TestCase):
    def test_migration_preserves_targets_splits_and_other_channels(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "baseline", Path(temporary) / "relative"
            rows = fixture(source)
            source_signature = validate_dataset(source)[2]["sha256"]
            report = convert_dataset(source, target)
            metadata, _, checked = validate_dataset(target)
            self.assertEqual(metadata["schema_version"], RELATIVE_DEPTH_SCHEMA)
            self.assertEqual(checked["input_channels"], RELATIVE_DEPTH_CHANNELS)
            self.assertNotEqual(report["sha256"], source_signature)
            self.assertEqual((source / "manifest.csv").read_bytes(), (target / "manifest.csv").read_bytes())
            for row in rows:
                with np.load(source / row["shard"]) as old, np.load(target / row["shard"]) as new:
                    np.testing.assert_array_equal(old["x"][1:], new["x"][1:])
                    for name in ("y", "mask", "wave", "wind", "raw_shape"):
                        np.testing.assert_array_equal(old[name], new[name])
                    self.assertTrue(np.all(new["x"][0, 121:] == 0))
            self.assertEqual(validate_dataset(source)[2]["sha256"], source_signature)
            with self.assertRaises(FileExistsError):
                convert_dataset(source, target)

    def test_inconsistent_relative_depth_channel_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "baseline", Path(temporary) / "relative"
            fixture(source)
            convert_dataset(source, target)
            path = target / "CASE0.npz"
            with np.load(path) as data:
                arrays = {name: data[name].copy() for name in data.files}
            arrays["x"][0, 0, 0] += .1
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "finite-depth dispersion"):
                validate_dataset(target)


if __name__ == "__main__":
    unittest.main()
