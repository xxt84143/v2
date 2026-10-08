"""Spatial forcing contract: channels, south-to-north rows, west-to-east columns."""
from pathlib import Path

import numpy as np

WAVE_SHAPE = (3, 2, 2)  # Hs, characteristic period, nautical direction FROM
WIND_SHAPE = (2, 3, 3)  # eastward u10, northward v10; all nine nodes observed


def validate(wave, wind):
    wave, wind = np.asarray(wave), np.asarray(wind)
    if wave.shape != WAVE_SHAPE or wind.shape != WIND_SHAPE:
        raise ValueError(f"Expected wave{WAVE_SHAPE}, wind{WIND_SHAPE}; regenerate old forcing files")
    if not np.isfinite(wave).all() or not np.isfinite(wind).all():
        raise ValueError("Forcing matrices must contain finite values")
    if np.any(wave[0] < 0) or np.any(wave[1] <= 0):
        raise ValueError("Hs must be non-negative; period must be positive")
    if np.any((wave[2] < 0) | (wave[2] >= 360)):
        raise ValueError("Wave directions must be in [0, 360)")
    return wave, wind


def load(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return validate(data["wave"].copy(), data["wind"].copy())


def resize(field, shape):
    """Bilinear features on the fine grid; raw nine-node wind is never reconstructed."""
    field = np.asarray(field)
    ny, nx = field.shape[-2:]
    yy = np.linspace(0, ny - 1, shape[0])
    xx = np.linspace(0, nx - 1, shape[1])
    y0, x0 = yy.astype(int), xx.astype(int)
    y1, x1 = np.minimum(y0 + 1, ny - 1), np.minimum(x0 + 1, nx - 1)
    wy, wx = yy - y0, xx - x0
    bottom = field[..., y0[:, None], x0] * (1 - wx) + field[..., y0[:, None], x1] * wx
    top = field[..., y1[:, None], x0] * (1 - wx) + field[..., y1[:, None], x1] * wx
    return bottom * (1 - wy[:, None]) + top * wy[:, None]


def model_inputs(wave, wind, depth, wet, norm, period_name="period"):
    wave, wind = validate(wave, wind)
    depth = np.asarray(depth)
    shape = depth.shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing="ij")
    cap = float(norm.get("depth_cap_m", norm.get("depth_scale_m", 100)))
    reference = float(norm.get("depth_reference_m", 10))
    if cap <= 0 or reference <= 0:
        raise ValueError("Depth normalization scales must be positive")
    water_depth = np.where(wet, np.clip(depth, 0, cap), 0)
    angle = np.deg2rad(wave[2])
    sine, cosine = resize(np.sin(angle), shape), resize(np.cos(angle), shape)
    length = np.hypot(sine, cosine)
    if np.any(length < 1e-8):
        raise ValueError("Opposite wave directions cannot define one interpolated mean direction")
    values = [np.log1p(water_depth / reference) / np.log1p(cap / reference), wet, xx, yy,
              resize(wave[0], shape) / float(norm["hs_scale_m"]),
              resize(wave[1], shape) / float(norm["period_scale_s"]), sine / length, cosine / length,
              *list(resize(wind, shape) / float(norm["wind_scale_mps"]))]
    names = ["log_depth", "wet_mask", "x", "y", "wave_hs", f"wave_{period_name}",
             "wave_dir_sin", "wave_dir_cos", "wind_u10", "wind_v10"]
    return np.stack(values).astype(np.float32), names
