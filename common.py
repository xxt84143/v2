"""Shared paths and small helpers for the v2 rectangular ERA5/GEBCO pilot."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_WNE = HERE.parents[1]
SWAN_YEARLY = CODE_WNE / "SWAN_YEARLY"
ERA5 = CODE_WNE / "ERA5DATA" / "ERA5_merged_2022_2024.nc"
GEBCO = CODE_WNE / "ERA5DATA" / "gebco_2026.nc"
CASES = HERE / "cases.csv"
CONFIG = HERE / "swan_batch_config.json"
GRIDS = HERE / "grids"
RUNS = HERE / "runs"
DATA = HERE / "dataset"
TRAINING = HERE / "training"

sys.path.insert(0, str(SWAN_YEARLY))
import swan_batch as core  # noqa: E402


def resolution_dir(resolution: str | int) -> Path:
    text = str(resolution).lower().replace("arcsec", "15arcsec").replace("km", "km")
    if text in {"15", "15arcsec", "gebco"}:
        return GRIDS / "15arcsec"
    if text in {"1000", "1", "1km", "1000m"}:
        return GRIDS / "1000m"
    raise ValueError(f"Unsupported v2 resolution: {resolution!r}")
