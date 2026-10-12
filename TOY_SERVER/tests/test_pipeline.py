"""Offline contracts for synthetic cases, locks and solver output integrity."""
import json
import copy
import io
from contextlib import redirect_stdout
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from generation import Rectangle, case_plan, generate_forcings, generate_islands, make_island_depth
from project import load_config, resume_sampling
import generate_cases
from shared.io_utils import write_json, read_csv, digest_file
from swan_inputs import grid_coordinates, prepare_case
from swan_runtime import inspect_case, run_case


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config.json")
        self.experiment = self.config["experiment"]
        self.experiment["terrain"]["count"] = {"train": 16, "validation": 4, "test": 4}
        self.experiment["forcing"]["count"] = {"train": 18, "validation": 9, "test": 9}
        self.experiment["case_sampling"].update(target_effective_train_cases=16, evaluation_cases_per_group=8)
        self.tile = self.config["tiles"][0]
        self.islands = generate_islands(self.experiment, Rectangle(self.tile["west"], self.tile["south"]))
        self.forcings = generate_forcings(self.experiment)

    def test_matrix_and_spectrum_settings(self):
        self.assertEqual(self.config["boundary"]["period"], "peak")
        self.assertEqual(self.config["boundary"]["jonswap_gamma"], self.experiment["forcing"]["jonswap_gamma"])
        self.assertEqual(self.forcings[0].wave.shape, (3, 2, 2))
        self.assertEqual(self.forcings[0].wind.shape, (2, 3, 3))

    def test_incremental_prefix_preserves_inputs_and_saved_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cases"
            config = copy.deepcopy(self.config)
            config["paths"]["cases"] = str(root)

            def generate(target):
                with patch("generate_cases.load_config", side_effect=lambda _: copy.deepcopy(config)), \
                     patch.object(sys, "argv", ["generate_cases.py", "--split", "train",
                                                "--target-train-cases", str(target)]), \
                     redirect_stdout(io.StringIO()):
                    generate_cases.main()
                return json.loads((root / "experiment.json").read_text())

            first = generate(8)
            self.assertEqual((first["created_in_this_call"], first["reused_in_this_call"]), (10, 0))
            original_rows = read_csv(root / "manifest.csv")
            preserved = {}
            for row in original_rows:
                directory = root / row["case_directory"]
                # Sentinel saved results: this checks preservation without launching SWAN.
                (directory / "run_status.json").write_text('{"status":"completed","sentinel":true}')
                (directory / "output/compgrid.tab").write_bytes(b"preserve prior solver output")
                for path in directory.rglob("*"):
                    if path.is_file():
                        preserved[path] = digest_file(path)
            second = generate(16)
            self.assertEqual((second["created_in_this_call"], second["reused_in_this_call"]), (10, 10))
            current_rows = read_csv(root / "manifest.csv")
            self.assertEqual(len(current_rows), 20)
            self.assertEqual(len({row["sample_id"] for row in current_rows}), 20)
            self.assertEqual(current_rows[:10], original_rows)
            for path, digest in preserved.items():
                self.assertEqual(digest_file(path), digest)
            repeated = generate(16)
            self.assertEqual((repeated["created_in_this_call"], repeated["reused_in_this_call"]), (0, 20))

    def test_resume_keeps_cli_target_and_grows_an_old_reserve_prefix(self):
        configured = copy.deepcopy(self.experiment["case_sampling"])
        saved = {"configured_case_sampling": configured,
                 "experiment": {"case_sampling": {**configured,
                                "target_effective_train_cases": 8, "training_candidate_cases": 10}}}
        self.assertEqual(resume_sampling(configured, saved)["target_effective_train_cases"], 8)
        larger = resume_sampling(configured, saved, target_override=16)
        self.assertEqual(larger["training_candidate_cases"], 20)
        edited = {**configured, "target_effective_train_cases": 32}
        self.assertEqual(resume_sampling(edited, saved)["target_effective_train_cases"], 32)
        self.assertEqual(configured, self.experiment["case_sampling"])

    def test_status_write_recovers_from_temporary_windows_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "status.json"
            destination.write_text('{"status":"old"}')
            original_replace = Path.replace
            attempts = []

            def replace(path, target):
                attempts.append(path)
                if len(attempts) == 1:
                    raise PermissionError("destination briefly held open")
                return original_replace(path, target)

            with patch("shared.io_utils.os.name", "nt"), patch("shared.io_utils.time.sleep"), patch.object(Path, "replace", replace):
                write_json(destination, {"status": "completed", "converged": False})
            self.assertEqual(json.loads(destination.read_text()), {"status": "completed", "converged": False})
            self.assertFalse(destination.with_name("status.json.tmp").exists())

    def test_self_contained_nine_point_wind_and_corruption_checks(self):
        lon, lat = grid_coordinates(self.tile, self.config["resolutions"]["gebco15s"])
        self.assertEqual((len(lat), len(lon)), (121, 121))
        np.testing.assert_allclose(np.diff(lon), 15 / 3600, atol=1e-12)
        np.testing.assert_allclose(np.diff(lat), 15 / 3600, atol=1e-12)
        depth, wet, _ = make_island_depth(lon, lat, self.islands[0], 100, -999)
        case = case_plan(self.islands, self.forcings, self.experiment)[0]
        forcing = self.forcings[0]
        wind = np.arange(18.).reshape(2, 3, 3)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "case"
            prepare_case(directory, "sample", case, "gebco15s", self.tile, lon, lat, depth, wet,
                         forcing.wave, wind, self.config)
            inspect_case(directory)
            np.testing.assert_allclose(np.loadtxt(directory / "wind.dat"), wind.reshape(6, 3))
            self.assertEqual((directory / "INPUT").read_text().count("BOUNDSPEC SEGMENT"), 4)
            prepare_case(directory, "sample", case, "gebco15s", self.tile, lon, lat, depth, wet,
                         forcing.wave, wind, self.config)
            (directory / "wind.dat").write_text("corrupt")
            with self.assertRaises(ValueError):
                inspect_case(directory)

    def test_case_lock_rejects_overlap_and_releases_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            lock = directory / ".swan.lock"
            lock.write_text("pid=other\n")
            with patch("swan_runtime._run_case") as execute:
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    run_case(directory, Path("unused"), 1, 1, False)
                execute.assert_not_called()
            self.assertTrue(lock.exists())
            lock.unlink()
            with patch("swan_runtime._run_case", side_effect=ValueError("bad input")):
                with self.assertRaisesRegex(ValueError, "bad input"):
                    run_case(directory, Path("unused"), 1, 1, False)
            self.assertFalse(lock.exists())

    def test_low_or_unknown_convergence_is_saved_and_reused(self):
        logs = [("iteration 50 ;\naccuracy OK in 92.49 % of wet grid points (99.5 % required)\n", False, "unconverged"),
                ("finished with an unfamiliar log format\n", None, "unverified")]
        for log, converged, legacy_status in logs:
            with self.subTest(legacy_status=legacy_status), tempfile.TemporaryDirectory() as temporary:
                directory, executable = self.prepare_runtime_fixture(Path(temporary))

                def compute(*args, **kwargs):
                    (directory / "PRINT").write_text(log)
                    self.write_output_fixture(directory)
                    return SimpleNamespace(returncode=0)

                with patch("swan_runtime.subprocess.run", side_effect=compute) as process:
                    result = run_case(directory, executable, 10, 1, False)
                    process.assert_called_once()
                self.assertEqual(result["status"], "completed")
                self.assertIs(result["converged"], converged)
                self.assertIsNone(result["error"])
                self.assertEqual(result["convergence_policy"], "record_only")
                self.assertTrue((directory / "output/compgrid.tab").exists())
                saved = json.loads((directory / "run_status.json").read_text())
                self.assertIs(saved["converged"], converged)
                # Outputs produced under the previous policy are recovered without a new SWAN run.
                saved.update(status=legacy_status, error="old convergence rejection")
                (directory / "run_status.json").write_text(json.dumps(saved))
                with patch("swan_runtime.subprocess.run") as process:
                    reused = run_case(directory, executable, 10, 1, False)
                    process.assert_not_called()
                self.assertEqual(reused["status"], "skipped")
                self.assertIs(reused["converged"], converged)
                saved = json.loads((directory / "run_status.json").read_text())
                self.assertEqual(saved["status"], "completed")
                self.assertIsNone(saved["error"])

    def test_execution_errors_and_invalid_output_still_fail(self):
        for code, log, write_output in [(1, "", True), (0, "** Error fatal", True), (0, "", False)]:
            with self.subTest(code=code, log=log, output=write_output), tempfile.TemporaryDirectory() as temporary:
                directory, executable = self.prepare_runtime_fixture(Path(temporary))

                def compute(*args, **kwargs):
                    (directory / "PRINT").write_text(log)
                    if write_output:
                        self.write_output_fixture(directory)
                    return SimpleNamespace(returncode=code)

                with patch("swan_runtime.subprocess.run", side_effect=compute):
                    result = run_case(directory, executable, 10, 1, False)
                self.assertEqual(result["status"], "failed")
                self.assertIsNotNone(result["error"])
                self.assertFalse((directory / ".swan.lock").exists())

    def prepare_runtime_fixture(self, root):
        lon, lat = grid_coordinates(self.tile, self.config["resolutions"]["gebco15s"])
        depth, wet, _ = make_island_depth(lon, lat, self.islands[0], 100, -999)
        directory = root / "case"
        prepare_case(directory, "sample", case_plan(self.islands, self.forcings, self.experiment)[0], "gebco15s",
                     self.tile, lon, lat, depth, wet, self.forcings[0].wave, self.forcings[0].wind, self.config)
        executable = root / "swan.exe"
        executable.write_bytes(b"test executable; subprocess is mocked")
        return directory, executable

    def write_output_fixture(self, directory):
        with np.load(directory / "inputs.npz") as data:
            depth = data["depth_m"]
        ny, nx = depth.shape
        yy, xx = np.meshgrid(np.linspace(self.tile["south"], self.tile["south"] + .5, ny),
                             np.linspace(self.tile["west"], self.tile["west"] + .5, nx), indexing="ij")
        values = {"XP": xx, "YP": yy, "DEPTH": depth, "HSIGN": np.ones_like(depth),
                  **{name: np.full_like(depth, 5) for name in ("TM01", "TM02", "RTP", "TMM10")},
                  **{name: np.full_like(depth, 30) for name in ("DIR", "PDIR", "DSPR")},
                  "WIND_X": np.zeros_like(depth), "WIND_Y": np.zeros_like(depth)}
        names = [component for name in self.config["output_quantities"]
                 for component in (["WIND_X", "WIND_Y"] if name == "WIND" else [name])]
        np.savetxt(directory / "output/compgrid.tab", np.stack([values[name] for name in names]).reshape(-1, 1), fmt="%.8e")
