"""Relative water depth from input periods and the finite-depth dispersion law."""
from __future__ import annotations

import numpy as np

BASELINE_SCHEMA = "toy-v2-dataset-matrix-2"
RELATIVE_DEPTH_SCHEMA = "toy-v2.1-dataset-relative-depth-1"


def relative_depth(depth_m, period_s, wet=None, gravity_mps2=9.81):
    """Return h/lambda = kh/(2*pi); solve z*tanh(z) = omega**2*h/g.

    The period is an input characteristic period, not a predicted local Tp.
    Land cells are zero. No deep-water approximation or depth clipping is used.
    """
    depth, period = np.broadcast_arrays(np.asarray(depth_m, dtype=np.float64),
                                        np.asarray(period_s, dtype=np.float64))
    active = np.ones(depth.shape, dtype=bool) if wet is None else np.broadcast_to(np.asarray(wet, dtype=bool), depth.shape)
    gravity = float(gravity_mps2)
    if not np.isfinite(gravity) or gravity <= 0:
        raise ValueError("gravity_mps2 must be positive and finite")
    if not np.isfinite(depth[active]).all() or np.any(depth[active] <= 0):
        raise ValueError("Wet water depths must be positive and finite")
    if not np.isfinite(period[active]).all() or np.any(period[active] <= 0):
        raise ValueError("Wet input periods must be positive and finite")
    result = np.zeros(depth.shape, dtype=np.float64)
    if not active.any():
        return result
    q = np.square(2 * np.pi / period[active]) * depth[active] / gravity
    if not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError("Depth/period combination is outside numerical range")
    lower, upper = np.zeros_like(q), q + 1
    # z*tanh(z) is monotone; 0 and q+1 bracket its unique positive root.
    for _ in range(64):
        middle = (lower + upper) * .5
        below = middle * np.tanh(middle) < q
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    result[active] = (lower + upper) / (4 * np.pi)
    return result


def feature_definition(norm):
    mode = norm.get("terrain_channel", "log_depth")
    if mode == "log_depth":
        return {"name": "log_depth", "definition": "log1p(min(h,cap)/reference)/log1p(cap/reference)"}
    if mode != "relative_depth":
        raise ValueError(f"Unknown terrain_channel: {mode}")
    gravity = float(norm.get("relative_depth_gravity_mps2", 9.81))
    calm_period = float(norm.get("zero_boundary_reference_period_s", 8.0))
    if not np.isfinite([gravity, calm_period]).all() or gravity <= 0 or calm_period <= 0:
        raise ValueError("Gravity and zero-boundary reference period must be positive and finite")
    return {"name": "relative_depth", "definition": "h/lambda = kh/(2*pi)",
            "dispersion": "(2*pi/T)^2 = g*k*tanh(k*h)", "gravity_mps2": gravity,
            "period_source": "bilinear input Tp; fixed reference if all incoming Hs are zero",
            "zero_boundary_reference_period_s": calm_period,
            "zero_boundary_meaning": "depth encoding with a reference wavelength; not the generated wind-wave wavelength"}


def dataset_schema(norm):
    return RELATIVE_DEPTH_SCHEMA if feature_definition(norm)["name"] == "relative_depth" else BASELINE_SCHEMA


def recover_log_depth(channel, wet, norm):
    """Invert the old feature; depths formerly above its cap cannot be recovered."""
    cap = float(norm.get("depth_cap_m", norm.get("depth_scale_m", 100)))
    reference = float(norm.get("depth_reference_m", 10))
    if not np.isfinite([cap, reference]).all() or cap <= 0 or reference <= 0:
        raise ValueError("Depth normalization scales must be positive and finite")
    channel, active = np.asarray(channel, dtype=np.float64), np.asarray(wet, dtype=bool)
    if np.any(channel[active] < -1e-6) or np.any(channel[active] > 1 + 1e-6):
        raise ValueError("Legacy log_depth is outside its expected range")
    recovered = reference * np.expm1(np.clip(channel, 0, 1) * np.log1p(cap / reference))
    return np.where(active, recovered, 0)
