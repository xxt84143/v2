"""Thin adapters to externally maintained U-Net implementations.

No U-Net implementation is vendored here.  Point ``paths.model_repo`` at a
user-downloaded checkout, or install the selected package in the environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _add_repo(repo_path: str | Path | None) -> None:
    if repo_path is None:
        return
    path = Path(repo_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"External model repository is absent: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def build_model(settings: dict[str, Any], in_channels: int, out_channels: int, repo_path=None):
    backend = str(settings["backend"])
    _add_repo(repo_path)
    if backend == "monai_basic_unet":
        try:
            from monai.networks.nets import BasicUNet
        except ImportError as exc:
            raise RuntimeError(
                "MONAI is unavailable. Download Project-MONAI/MONAI tag 1.3.2 and set "
                "paths.model_repo, or install a compatible released package."
            ) from exc
        return BasicUNet(
            spatial_dims=2, in_channels=in_channels, out_channels=out_channels,
            features=tuple(int(value) for value in settings["features"]),
            norm=("group", {"num_groups": 4}), dropout=0.0,
        )
    if backend == "smp_unet":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError(
                "segmentation_models.pytorch is unavailable; download the official repository "
                "and set paths.model_repo before choosing backend=smp_unet."
            ) from exc
        return smp.Unet(
            encoder_name="resnet18", encoder_weights=None, in_channels=in_channels,
            classes=out_channels, activation=None, encoder_depth=4,
            decoder_channels=(128, 64, 32, 16),
        )
    raise ValueError(f"Unsupported external model backend: {backend}")

