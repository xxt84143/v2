from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_v2 import aggregate_group, bootstrap_mean_ci, field_metrics  # noqa: E402


class EvaluationMetricTests(unittest.TestCase):
    def test_field_metrics_are_in_physical_units_and_signed_bias(self):
        target = np.asarray([1.0, 2.0, 3.0])
        prediction = np.asarray([2.0, 2.0, 2.0])
        result = field_metrics(prediction, target)
        self.assertAlmostEqual(result["mae_m"], 2.0 / 3.0)
        self.assertAlmostEqual(result["rmse_m"], np.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(result["bias_m"], 0.0)
        self.assertAlmostEqual(result["r2"], 0.0)

    def test_perfect_prediction(self):
        target = np.asarray([0.4, 1.1, 2.3, 3.2])
        result = field_metrics(target, target)
        self.assertEqual(result["rmse_m"], 0.0)
        self.assertEqual(result["mae_m"], 0.0)
        self.assertAlmostEqual(result["pearson_r"], 1.0)
        self.assertAlmostEqual(result["r2"], 1.0)

    def test_bootstrap_is_deterministic_and_case_level(self):
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        first = bootstrap_mean_ci(values, 200, np.random.default_rng(7))
        second = bootstrap_mean_ci(values, 200, np.random.default_rng(7))
        self.assertEqual(first, second)
        self.assertLess(first[0], values.mean())
        self.assertGreater(first[1], values.mean())

    def test_skill_score_uses_mse_ratio(self):
        records = []
        for prediction in (np.asarray([1.0, 2.0]), np.asarray([2.0, 3.0])):
            target = np.asarray([1.0, 1.0])
            baseline = np.asarray([3.0, 3.0])
            metrics = field_metrics(prediction, target)
            records.append({
                **metrics, "_prediction": prediction, "_target": target, "_baseline": baseline,
                "_negative_count": 0,
            })
        result = aggregate_group(records, 50, np.random.default_rng(2))
        expected = 1.0 - result["rmse_m"] ** 2 / result["baseline_rmse_m"] ** 2
        self.assertAlmostEqual(result["mse_skill_vs_boundary_hs"], expected)


if __name__ == "__main__":
    unittest.main()
