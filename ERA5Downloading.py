#!/usr/bin/env python
"""Download ERA5 Complete two-dimensional wave spectra.

The CDS/MARS service limits a request to 2,500,000 fields.  A four-month
request contains at most 123 * 24 * 24 * 30 = 2,127,360 fields, so it is a
safe compromise between request overhead and file size.  Requests are made in
fixed batches of at most five.  No request from the next batch is submitted
until every request in the current batch has downloaded successfully.
"""

from __future__ import annotations

import concurrent.futures
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import cdsapi


DATASET = "reanalysis-era5-complete"
DOWNLOAD_DIR = Path(r"F:\TXG\ERA5Data_new\WaveSpectra")
YEARS = (2022, 2023, 2024)
MAX_BATCH_SIZE = 5

# A queued MARS task may legitimately take days.  These settings apply to
# transient network failures while polling the same server-side task and do
# not submit a new CDS task.
CDS_HTTP_TIMEOUT = 600       # seconds per HTTP request
CDS_RETRY_MAX = 10_000       # enough for several days of transient failures
CDS_RETRY_SLEEP = 120        # seconds between failed HTTP attempts

# MARS area order is North/West/South/East.  Positive 117 and 125 mean
# 117E and 125E; W is the western boundary, not west longitude.
AREA = [27, 117, 20, 125]
DIRECTION = "/".join(str(i) for i in range(1, 25))
FREQUENCY = "/".join(str(i) for i in range(1, 31))


@dataclass(frozen=True)
class Job:
    year: int
    start_month: int
    end_month: int

    @property
    def start(self) -> date:
        return date(self.year, self.start_month, 1)

    @property
    def end(self) -> date:
        # The first day of the following month minus one day avoids a
        # hard-coded leap-year special case.
        if self.end_month == 12:
            next_month = date(self.year + 1, 1, 1)
        else:
            next_month = date(self.year, self.end_month + 1, 1)
        return date.fromordinal(next_month.toordinal() - 1)

    @property
    def label(self) -> str:
        return f"{self.year}_{self.start_month:02d}-{self.end_month:02d}"


def make_jobs(years: Iterable[int]) -> list[Job]:
    return [
        Job(year, start_month, start_month + 3)
        for year in years
        for start_month in (1, 5, 9)
    ]


def request_for(job: Job) -> dict[str, object]:
    return {
        # class and expver are imposed by reanalysis-era5-complete and are
        # intentionally omitted.
        "param": "251.140",  # 2D wave spectra (single)
        "direction": DIRECTION,
        "frequency": FREQUENCY,
        "stream": "wave",
        "type": "an",
        "date": f"{job.start.isoformat()}/to/{job.end.isoformat()}",
        "time": "00/to/23/by/1",
        "area": AREA,
        "grid": "0.5/0.5",
        # cdsapi currently warns that the old 'format' key is deprecated.
        "data_format": "grib",
    }


def output_path(job: Job) -> Path:
    return DOWNLOAD_DIR / (
        f"ERA5_2Dwave_spectra_{job.label}_N27E117S20E125.grib"
    )


def download_job(job: Job) -> tuple[Job, str]:
    target = output_path(job)
    if target.exists() and target.stat().st_size > 0:
        return job, "SKIP (file already exists)"

    # A .part file prevents an interrupted download from being mistaken for
    # a complete file on the next run.
    partial = target.with_suffix(target.suffix + ".part")
    request = request_for(job)
    client = cdsapi.Client(
        timeout=CDS_HTTP_TIMEOUT,
        retry_max=CDS_RETRY_MAX,
        sleep_max=CDS_RETRY_SLEEP,
        wait_until_complete=True,
        progress=False,
    )
    client.retrieve(DATASET, request, str(partial))
    os.replace(partial, target)
    return job, f"DONE ({target.stat().st_size / 1e9:.2f} GB)"


def main() -> int:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(YEARS)

    print(f"Dataset: {DATASET}")
    print(f"Jobs: {len(jobs)} four-month requests")
    print(f"CDS requests: fixed batches of at most {MAX_BATCH_SIZE}")
    print(f"Output: {DOWNLOAD_DIR}")

    batches = [
        jobs[start : start + MAX_BATCH_SIZE]
        for start in range(0, len(jobs), MAX_BATCH_SIZE)
    ]

    for batch_index, batch in enumerate(batches, start=1):
        labels = ", ".join(job.label for job in batch)
        print(f"\nBatch {batch_index}/{len(batches)}: {labels}")
        batch_failures: list[tuple[Job, Exception]] = []

        # Only this batch is submitted.  A completed or rejected future is
        # never replaced while another request in the batch is still active.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(batch)
        ) as executor:
            future_to_job = {
                executor.submit(download_job, job): job for job in batch
            }
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    completed_job, status = future.result()
                    print(f"[{completed_job.label}] {status}")
                except Exception as exc:
                    batch_failures.append((job, exc))
                    print(f"[{job.label}] FAILED: {exc}")

        if batch_failures:
            print("\nThe current batch had failures.")
            print("No request from the next batch has been submitted.")
            return 1

        print(f"Batch {batch_index} downloaded successfully.")

    print("\nAll requested chunks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
