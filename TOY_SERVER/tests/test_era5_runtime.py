"""Offline regression checks for real NetCDF I/O and the CDS queue lifecycle."""
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from era5_jobs import download_all, download_settings, read_state, requests_for
from era5_merge import merge_group, request_times, validate_download, variable_names
from project import load_config


class FakeCDS:
    def __init__(self, reject_multiple=False, interrupt_once=False):
        self.reject_multiple = reject_multiple
        self.interrupt_once = interrupt_once
        self.jobs, self.events = {}, []
        self.lock = threading.Lock()
        self.active = self.peak = 0

    def get_jobs(self, **kwargs):
        return SimpleNamespace(request_ids=[])

    def retrieve(self, dataset, request):
        if self.reject_multiple and len(request["variable"]) > 1:
            response = requests.Response()
            response.status_code = 403
            raise requests.HTTPError("cost limits exceeded", response=response)
        with self.lock:
            number = len(self.jobs) + 1
            remote = SimpleNamespace(request_id=str(number), status="successful")
            remote.get_results = lambda: self.results(number)
            self.jobs[str(number)] = remote
            self.events.append(("submit", number))
            self.active += 1
            self.peak = max(self.peak, self.active)
            return remote

    def get_remote(self, request_id):
        return self.jobs[request_id]

    def results(self, number):
        def download(target):
            if self.interrupt_once:
                self.interrupt_once = False
                raise requests.ConnectionError("simulated interrupted transfer")
            # A slow first job proves later jobs can fill a slot before the first finishes.
            time.sleep(.15 if number == 1 else .01)
            Path(target).write_bytes(b"X")
            with self.lock:
                self.events.append(("downloaded", number))
                self.active -= 1
        return SimpleNamespace(content_length=1, download=download)


class ERA5RuntimeTests(unittest.TestCase):
    def config(self, root):
        config = load_config(ROOT / "era5_config.json")
        config["_base"] = str(root)
        config["sampling"] = {"start_utc": "2023-07-01T00:00:00", "end_utc": "2023-07-01T03:00:00", "step_hours": 3}
        return config

    @staticmethod
    def validate_fake(plan, path):
        if path.read_bytes() != b"X":
            raise ValueError("incomplete fake transfer")

    def test_rolling_queue_and_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            plans = requests_for(config)
            api = FakeCDS()
            with patch("era5_jobs.cds_client", return_value=(api, api)):
                leaves = download_all(plans, download_settings(config), self.validate_fake)
            self.assertEqual(len(leaves), 6)
            self.assertLessEqual(api.peak, 5)
            self.assertLess(api.events.index(("submit", 6)), api.events.index(("downloaded", 1)))
            self.assertTrue(all(read_state(plan)["status"] == "completed" for plan in leaves))
            with patch("era5_jobs.cds_client") as client:
                download_all(plans, download_settings(config), self.validate_fake, verify_hash=True)
                client.assert_not_called()

    def test_cost_splits_persist(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            config["era5"]["download"]["variables_per_request"] = 2
            plans = requests_for(config)
            api = FakeCDS(reject_multiple=True)
            with patch("era5_jobs.cds_client", return_value=(api, api)):
                leaves = download_all(plans, download_settings(config), self.validate_fake)
            self.assertEqual(len(leaves), 6)
            self.assertTrue(all(read_state(plan)["status"] == "split" for plan in plans))
            with patch("era5_jobs.cds_client") as client:
                restored = download_all(plans, download_settings(config), self.validate_fake)
                client.assert_not_called()
            self.assertEqual({p["target"] for p in leaves}, {p["target"] for p in restored})

    def test_transfer_retry_keeps_request_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            config["era5"]["download"]["retry_seconds"] = 1
            plans = requests_for(config)[:1]
            api = FakeCDS(interrupt_once=True)
            with patch("era5_jobs.cds_client", return_value=(api, api)):
                download_all(plans, download_settings(config), self.validate_fake)
            self.assertEqual(len(api.jobs), 1)
            self.assertEqual(read_state(plans[0])["request_id"], "1")

    def test_netcdf_validation_and_incremental_merge(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            plans = requests_for(config)
            for index, plan in enumerate(plans):
                request = plan["request"]
                north, west, south, east = request["area"]
                spacing = request["grid"][0]
                latitude = np.arange(north, south - spacing / 2, -spacing)
                longitude = np.arange(west, east + spacing / 2, spacing)
                stamps = request_times(request)
                name = variable_names(request)[0]
                values = np.full((len(stamps), len(latitude), len(longitude)), index + 1., dtype="float32")
                dataset = xr.Dataset({name: (("valid_time", "latitude", "longitude"), values)},
                                     coords={"valid_time": stamps, "latitude": latitude, "longitude": longitude})
                plan["target"].parent.mkdir(parents=True, exist_ok=True)
                dataset.to_netcdf(plan["target"], engine="netcdf4")
                validate_download(plan, plan["target"])
            for group in ("wind", "waves"):
                merge_group(config, plans, group, time_chunk=1)
                with xr.open_dataset(Path(temporary) / config["paths"][group]) as data:
                    self.assertEqual(data.sizes["time"], 2)
                    for plan in plans:
                        if plan["group"] == group:
                            name = variable_names(plan["request"])[0]
                            np.testing.assert_allclose(data[name], plans.index(plan) + 1.)
                    if group == "waves":
                        self.assertEqual(data["mwp"].attrs["period_definition"], "Tm-1,0 = m-1/m0")


if __name__ == "__main__":
    unittest.main()
