"""Wave matrix contract, finite-depth physics and the ten input features."""
from __future__ import annotations
from pathlib import Path
import numpy as np

BASELINE_SCHEMA = "toy-v2-dataset-matrix-2"
RELATIVE_DEPTH_SCHEMA = "toy-v2.1-dataset-relative-depth-1"
WAVE_SHAPE = (3, 2, 2)
WIND_SHAPE = (2, 3, 3)

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


def terrain_feature(wave, depth, wet, norm):
    """Encode depth using the declared baseline or finite-depth wave geometry."""
    depth, wet = np.asarray(depth), np.asarray(wet, dtype=bool)
    if depth.ndim != 2 or wet.shape != depth.shape:
        raise ValueError("depth and wet must be matching two-dimensional fields")
    definition = feature_definition(norm)
    if definition["name"] == "relative_depth":
        periods = resize(np.asarray(wave)[1], depth.shape)
        if not np.any(np.asarray(wave)[0] > 0):
            periods = np.full(depth.shape, definition["zero_boundary_reference_period_s"])
        return relative_depth(depth, periods, wet, definition["gravity_mps2"]), "relative_depth"
    cap = float(norm.get("depth_cap_m", norm.get("depth_scale_m", 100)))
    reference = float(norm.get("depth_reference_m", 10))
    if not np.isfinite([cap, reference]).all() or cap <= 0 or reference <= 0:
        raise ValueError("Depth normalization scales must be positive and finite")
    water_depth = np.where(wet, np.clip(depth, 0, cap), 0)
    return np.log1p(water_depth / reference) / np.log1p(cap / reference), "log_depth"


def model_inputs(wave, wind, depth, wet, norm, period_name="period"):
    wave, wind = validate(wave, wind)
    depth = np.asarray(depth)
    shape = depth.shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing="ij")
    terrain, terrain_name = terrain_feature(wave, depth, wet, norm)
    angle = np.deg2rad(wave[2])
    sine, cosine = resize(np.sin(angle), shape), resize(np.cos(angle), shape)
    length = np.hypot(sine, cosine)
    if np.any(length < 1e-8):
        raise ValueError("Opposite wave directions cannot define one interpolated mean direction")
    values = [terrain, wet, xx, yy,
              resize(wave[0], shape) / float(norm["hs_scale_m"]),
              resize(wave[1], shape) / float(norm["period_scale_s"]), sine / length, cosine / length,
              *list(resize(wind, shape) / float(norm["wind_scale_mps"]))]
    names = [terrain_name, "wet_mask", "x", "y", "wave_hs", f"wave_{period_name}",
             "wave_dir_sin", "wave_dir_cos", "wind_u10", "wind_v10"]
    return np.stack(values).astype(np.float32), names
