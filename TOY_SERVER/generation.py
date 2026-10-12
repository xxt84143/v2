"""Pure v2.1.1 geometry, single-peak forcing and sparse case planning."""
from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any
import numpy as np
from project import positive_int, target_settings

EARTH_RADIUS_M = 6_371_008.8
SPLITS = ("train", "validation", "test")
FAMILIES = ("calm", "near_calm", "swell_no_wind", "young_windsea",
            "mature_windsea", "swell_weak_wind", "swell_moderate_wind",
            "swell_strong_wind", "wind_only")

@dataclass(frozen=True)
class Island:
    terrain_id: str
    split: str
    center_lon: float
    center_lat: float
    radius_m: float
    skirt_width_m: float
    shore_depth_m: float
    shape: str = "circle"
    semi_major_m: float = 0.0
    semi_minor_m: float = 0.0
    orientation_deg: float = 0.0  # counterclockwise from east


@dataclass(frozen=True)
class CoupledForcing:
    forcing_id: str
    split: str
    wave: np.ndarray  # (Hs, Tp, direction FROM), 2x2
    wind: np.ndarray  # (u10, v10), 3x3; nine independent input nodes
    family: str
    wave_age: float
    steepness: float


@dataclass(frozen=True)
class Rectangle:
    west: float
    south: float

    @property
    def east(self):
        return self.west + .5

    @property
    def north(self):
        return self.south + .5

    @property
    def center(self):
        return (self.west + .25, self.south + .25)


def local_xy_m(lon: np.ndarray | float, lat: np.ndarray | float, reference_lon: float, reference_lat: float):
    x = EARTH_RADIUS_M * math.cos(math.radians(reference_lat)) * np.deg2rad(np.asarray(lon) - reference_lon)
    y = EARTH_RADIUS_M * np.deg2rad(np.asarray(lat) - reference_lat)
    return x, y


def domain_size_m(tile) -> tuple[float, float]:
    cx, cy = tile.center
    x0, _ = local_xy_m(tile.west, cy, cx, cy)
    x1, _ = local_xy_m(tile.east, cy, cx, cy)
    _, y0 = local_xy_m(cx, tile.south, cx, cy)
    _, y1 = local_xy_m(cx, tile.north, cx, cy)
    return float(x1 - x0), float(y1 - y0)


def make_island_depth(lon: np.ndarray, lat: np.ndarray, island: Island, base_depth_m: float, exception: float):
    xx, yy = np.meshgrid(lon, lat)
    dx, dy = local_xy_m(xx, yy, island.center_lon, island.center_lat)
    distance = np.hypot(dx, dy)
    shore_radius = island.radius_m
    if island.shape == "ellipse":
        angle = math.radians(island.orientation_deg)
        along = dx * math.cos(angle) + dy * math.sin(angle)
        across = -dx * math.sin(angle) + dy * math.cos(angle)
        theta = np.arctan2(across, along)
        a, b = island.semi_major_m, island.semi_minor_m
        if not (a >= b > 0):
            raise ValueError("Ellipse requires positive semi_major_m >= semi_minor_m")
        shore_radius = a * b / np.hypot(b * np.cos(theta), a * np.sin(theta))
    elif island.shape != "circle":
        raise ValueError("Only circle and ellipse terrains are supported")
    wet = distance > shore_radius
    depth = np.full(distance.shape, exception, dtype=np.float32)
    outer = shore_radius + island.skirt_width_m
    transition = np.clip((distance - shore_radius) / island.skirt_width_m, 0.0, 1.0)
    smooth = transition * transition * (3.0 - 2.0 * transition)
    water_depth = island.shore_depth_m + (base_depth_m - island.shore_depth_m) * smooth
    depth[wet] = np.minimum(water_depth[wet], base_depth_m).astype(np.float32)
    depth[distance >= outer] = np.float32(base_depth_m)
    return depth, wet, distance


def island_to_row(item: Island) -> dict[str, Any]:
    return {"terrain_id": item.terrain_id, "terrain_split": item.split, "center_lon": item.center_lon,
            "center_lat": item.center_lat, "radius_m": item.radius_m, "skirt_width_m": item.skirt_width_m,
            "shore_depth_m": item.shore_depth_m, "terrain_shape": item.shape,
            "semi_major_m": item.semi_major_m or item.radius_m,
            "semi_minor_m": item.semi_minor_m or item.radius_m,
            "orientation_deg": item.orientation_deg}


def forcing_to_row(item: CoupledForcing) -> dict[str, Any]:
    return {"forcing_id": item.forcing_id, "forcing_split": item.split,
            "forcing_file": f"forcings/{item.forcing_id}.npz", "family": item.family,
            "wave_age": item.wave_age, "maximum_steepness": item.steepness,
            "mean_hs_m": float(item.wave[0].mean()), "mean_tp_s": float(item.wave[1].mean()),
            "maximum_wind_speed_mps": float(np.hypot(*item.wind).max())}


def bounds(settings, name, lower=0.0, strict=True):
    values = settings[name]
    if len(values) != 2:
        raise ValueError(f"{name} requires [minimum, maximum]")
    lo, hi = map(float, values)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi
            and (lo > lower if strict else lo >= lower)):
        raise ValueError(f"Invalid bounds for {name}")
    return lo, hi


def rng_for(config, purpose, split, family=0):
    return np.random.default_rng(np.random.SeedSequence(
        [int(config["seed"]), purpose, SPLITS.index(split), family]))


def strata(count, dimensions, rng):
    """Latin hypercube in the unit cube, without an additional dependency."""
    return np.column_stack([(rng.permutation(count) + rng.random(count)) / count
                            for _ in range(dimensions)])


def between(pair, fraction):
    return pair[0] + (pair[1] - pair[0]) * fraction


def validate_config(config):
    target, candidates, evaluation = target_settings(config)
    terrain, forcing = config["terrain"], config["forcing"]
    nt, nf = {}, {}
    for split in SPLITS:
        nt[split] = positive_int(terrain["count"][split], f"terrain.count.{split}")
        nf[split] = positive_int(forcing["count"][split], f"forcing.count.{split}")
        if nt[split] < 2 or nf[split] < len(FAMILIES):
            raise ValueError("Each split needs both shapes and all nine forcing families")
    for name in ("island_radius_km", "skirt_width_km", "shore_depth_m"):
        bounds(terrain, name)
    if bounds(terrain, "ellipse_axis_ratio", lower=1, strict=False)[1] <= 1:
        raise ValueError("ellipse_axis_ratio must include non-circular ellipses")
    base = float(terrain["base_depth_m"])
    if not math.isfinite(base) or base < float(terrain["shore_depth_m"][1]):
        raise ValueError("base_depth_m must be finite and at least the maximum shore depth")
    clearance = float(terrain["outer_clearance_km"])
    if not math.isfinite(clearance) or clearance <= 0:
        raise ValueError("outer_clearance_km must be positive")
    limits = forcing["coverage"]
    for name in ("period_s", "swell_period_s", "swell_height_m", "near_calm_height_m",
                 "young_wave_age", "mature_wave_age", "windsea_steepness"):
        bounds(limits, name)
    for name in ("windsea_speed_mps", "wind_only_speed_mps", "weak_wind_speed_mps",
                 "moderate_wind_speed_mps", "strong_wind_speed_mps", "near_calm_wind_speed_mps"):
        bounds(limits, name, strict=False)
    for name in ("maximum_relative_gradient", "maximum_direction_gradient_deg", "maximum_steepness"):
        value = float(forcing[name])
        if not math.isfinite(value) or value < 0 or (name == "maximum_steepness" and value == 0):
            raise ValueError(f"Invalid {name}")
    if forcing["maximum_relative_gradient"] >= .5 or forcing["maximum_direction_gradient_deg"] >= 45:
        raise ValueError("Gradients must preserve positive scales and a well-defined interpolated direction")
    for name in ("maximum_hs_m", "maximum_wind_speed_mps"):
        if not math.isfinite(float(limits[name])) or limits[name] <= 0:
            raise ValueError(f"{name} must be positive")
    spectral = config.get("spectral_grid", {"lowest_frequency_hz": .03453,
                                            "highest_frequency_hz": .5477526003})
    lo, hi = map(float, limits["period_s"])
    if not (1 / spectral["highest_frequency_hz"] <= lo <= hi <= 1 / spectral["lowest_frequency_hz"]):
        raise ValueError("coverage.period_s must be inside the spectral grid")
    if not (lo <= limits["swell_period_s"][0] <= limits["swell_period_s"][1] <= hi):
        raise ValueError("swell periods must lie within coverage.period_s")
    if candidates > nt["train"] * nf["train"]:
        raise ValueError("Training target plus reserve exceeds unique pair capacity; increase terrain/forcing counts")
    for split in ("validation", "test"):
        capacities = (nt[split] * nf["train"], nt["train"] * nf[split], nt[split] * nf[split])
        if evaluation > min(capacities):
            raise ValueError(f"{split} evaluation quota exceeds unique pair capacity")
    return target


def generate_islands(config, tile):
    validate_config(config)
    settings = config["terrain"]
    width, height = domain_size_m(tile)
    cx, cy = tile.center
    if width <= 0 or height <= 0:
        raise ValueError("Invalid local domain dimensions")
    result = []
    for split in SPLITS:
        count = int(settings["count"][split])
        rng = rng_for(config, 10, split)
        samples = strata(count, 8, rng)
        for index, u in enumerate(samples):
            shape = "circle" if index % 2 == 0 else "ellipse"
            # Equal-area radius, leaving existing radius/skirt/depth ranges unchanged.
            skirt = between(settings["skirt_width_km"], u[1]) * 1000
            shore = between(settings["shore_depth_m"], u[2])
            ratio = 1.0 if shape == "circle" else between(settings["ellipse_axis_ratio"], u[3])
            orientation = 180 * u[4]
            angle = math.radians(orientation)
            margin = skirt + float(settings["outer_clearance_km"]) * 1000
            scale_x = math.hypot(math.sqrt(ratio)*math.cos(angle), math.sin(angle)/math.sqrt(ratio))
            scale_y = math.hypot(math.sqrt(ratio)*math.sin(angle), math.cos(angle)/math.sqrt(ratio))
            radius_lo, radius_hi = np.asarray(settings["island_radius_km"]) * 1000
            radius_hi = min(radius_hi, (width/2-margin-1)/scale_x, (height/2-margin-1)/scale_y)
            if radius_lo > radius_hi:
                raise ValueError("No feasible radius for this ellipse/skirt/tile configuration")
            radius = between((radius_lo, radius_hi), u[0])
            a, b = radius * math.sqrt(ratio), radius / math.sqrt(ratio)
            # Bound the rotated ellipse and its radial skirt conservatively.
            mx = math.hypot(a * math.cos(angle), b * math.sin(angle)) + margin
            my = math.hypot(a * math.sin(angle), b * math.cos(angle)) + margin
            if 2 * mx >= width or 2 * my >= height:
                raise ValueError("Requested ellipse/skirt cannot fit this tile; reduce size/axis ratio")
            x = (2 * u[5] - 1) * (width / 2 - mx)
            y = (2 * u[6] - 1) * (height / 2 - my)
            lon = cx + math.degrees(x / (EARTH_RADIUS_M * math.cos(math.radians(cy))))
            lat = cy + math.degrees(y / EARTH_RADIUS_M)
            result.append(Island(f"T{split[:2].upper()}{index+1:06d}", split, lon, lat,
                                 radius, skirt, shore, shape, a, b, orientation))
    return result


def generate_forcings(config):
    validate_config(config)
    settings, limits = config["forcing"], config["forcing"]["coverage"]
    result = []
    y2, x2 = np.meshgrid([-1., 1.], [-1., 1.], indexing="ij")
    y3, x3 = np.meshgrid([-1., 0., 1.], [-1., 0., 1.], indexing="ij")
    for split in SPLITS:
        count = int(settings["count"][split])
        per_family = []
        for family_index, family in enumerate(FAMILIES):
            size = len(range(family_index, count, len(FAMILIES)))
            rng = rng_for(config, 20, split, family_index)
            samples = strata(size, 12, rng)
            items = []
            for u in samples:
                direction = 360 * u[0]
                speed, age, hs, period = 0., 0., None, None
                sx, sy = (2 * u[1:3] - 1) * float(settings["maximum_relative_gradient"])
                dx, dy = (2 * u[3:5] - 1) * float(settings["maximum_direction_gradient_deg"])
                factor2, factor3 = 1 + sx * x2 + sy * y2, 1 + sx * x3 + sy * y3
                offsets = 180 * (2 * u[5] - 1)
                if family in {"young_windsea", "mature_windsea"}:
                    # Conditional U/age and steepness: local periods and heights
                    # stay within the declared envelope even with gradients.
                    ages = limits["young_wave_age" if family == "young_windsea" else "mature_wave_age"]
                    age = between(ages, u[6])
                    pmin, pmax = limits["period_s"]
                    speed_lo = max(limits["windsea_speed_mps"][0], pmin * 9.81 / (2*np.pi*age*factor2.min()))
                    speed_hi = min(limits["windsea_speed_mps"][1], pmax * 9.81 / (2*np.pi*age*factor2.max()),
                                   limits["maximum_wind_speed_mps"] / factor3.max())
                    if speed_lo > speed_hi:
                        raise ValueError("No feasible wind speed for requested wave age/period bounds")
                    speed = between((speed_lo, speed_hi), u[7])
                    period = 2*np.pi*speed*factor2*age/9.81
                    lp = 9.81 * period**2 / (2*np.pi)
                    smin, smax = limits["windsea_steepness"]
                    smax = min(smax, settings["maximum_steepness"], limits["maximum_hs_m"]/lp.max())
                    if smin > smax:
                        raise ValueError("No feasible windsea steepness under the configured Hs cap")
                    hs = between((smin, smax), u[8]) * lp
                    offsets = 0.
                elif family in {"calm", "wind_only"}:
                    hs, period = np.zeros((2, 2)), np.full((2, 2), 8.)
                    speed = 0. if family == "calm" else between(limits["wind_only_speed_mps"], u[7])
                    offsets = 0.
                else:
                    wind_bounds = {"near_calm": "near_calm_wind_speed_mps",
                                   "swell_weak_wind": "weak_wind_speed_mps",
                                   "swell_moderate_wind": "moderate_wind_speed_mps",
                                   "swell_strong_wind": "strong_wind_speed_mps"}
                    speed = 0. if family == "swell_no_wind" else between(limits[wind_bounds[family]], u[7])
                    periods = limits["period_s"] if family == "near_calm" else limits["swell_period_s"]
                    # A separate, mild period gradient is represented by all four corners.
                    pg = .03 * (2*u[9]-1)
                    pscale = 1 + pg*x2
                    p_lo = max(periods[0], limits["period_s"][0]/pscale.min())
                    p_hi = min(periods[1], limits["period_s"][1]/pscale.max())
                    if p_lo > p_hi:
                        raise ValueError("No feasible period with the configured spatial gradient")
                    period = between((p_lo, p_hi), u[6]) * pscale
                    hmin, hmax = limits["near_calm_height_m" if family == "near_calm" else "swell_height_m"]
                    lp = 9.81 * period**2 / (2*np.pi)
                    hmax = min(hmax, limits["maximum_hs_m"]/factor2.max(),
                               float(np.min(settings["maximum_steepness"]*lp/factor2)))
                    if hmin > hmax:
                        raise ValueError("No feasible swell height under the steepness/Hs constraints")
                    hs = between((hmin, hmax), u[8]) * factor2
                speed = min(speed, limits["maximum_wind_speed_mps"]/factor3.max())
                wave_from = (direction + dx*x2 + dy*y2) % 360
                wind_angle = np.deg2rad(direction + offsets + dx*x3 + dy*y3)
                wind_speed = speed * factor3
                wind = np.stack((-wind_speed*np.sin(wind_angle), -wind_speed*np.cos(wind_angle)))
                wave = np.stack((hs, period, wave_from))
                steepness = hs / (9.81*period**2/(2*np.pi))
                if not (np.isfinite(wave).all() and np.isfinite(wind).all()
                        and np.all(hs >= 0) and hs.max() <= limits["maximum_hs_m"] + 1e-9
                        and steepness.max() <= settings["maximum_steepness"] + 1e-9):
                    raise ValueError("Invalid generated forcing")
                if hs.max() > 0 and (period.min() < limits["period_s"][0]-1e-9
                                     or period.max() > limits["period_s"][1]+1e-9):
                    raise ValueError("Generated period outside coverage envelope")
                items.append((wave, wind, family, float(age), float(steepness.max())))
            per_family.append(iter(items))
        # Interleave families so early candidate prefixes retain every family.
        for index in range(count):
            values = next(per_family[index % len(FAMILIES)])
            result.append(CoupledForcing(f"F{split[:2].upper()}{index+1:06d}", split, *values))
    return result


def pair_indices(nt, nf, count, seed):
    """Unique balanced cycles without constructing the Cartesian product.

    Each full cycle visits the larger pool once and covers the smaller pool.
    Changing count preserves the existing pair prefix.
    """
    if count > nt*nf:
        raise ValueError("Sparse pair quota exceeds capacity")
    rng = np.random.default_rng(seed)
    terrains, forcings = rng.permutation(nt), rng.permutation(nf)
    for ordinal in range(count):
        cycle, position = divmod(ordinal, max(nt, nf))
        if nf >= nt:
            yield int(terrains[(position+cycle) % nt]), int(forcings[position])
        else:
            yield int(terrains[position]), int(forcings[(position+cycle) % nf])


def case_plan(islands, forcings, config):
    validate_config(config)
    _, training_candidates, evaluation = target_settings(config)
    terrain = {s: [x for x in islands if x.split == s] for s in SPLITS}
    conditions = {s: [x for x in forcings if x.split == s] for s in SPLITS}
    groups = [("train", "seen_island_seen_forcing", "train", "train", training_candidates)]
    for split in ("validation", "test"):
        groups.extend([(split, "new_island", split, "train", evaluation),
                       (split, "new_forcing", "train", split, evaluation),
                       (split, "new_both", split, split, evaluation)])
    rows = []
    for group_number, (split, group, ts, fs, count) in enumerate(groups):
        for ordinal, (ti, fi) in enumerate(pair_indices(len(terrain[ts]), len(conditions[fs]), count,
                                                       int(config["seed"])+3000+group_number)):
            island, forcing = terrain[ts][ti], conditions[fs][fi]
            # Group-local IDs keep evaluation identity stable when target changes.
            row = {"case_id": f"BC{group_number}{ordinal+1:07d}",
                   "terrain_id": island.terrain_id, "forcing_id": forcing.forcing_id,
                   "split": split, "generalization": group,
                   "candidate_rank": ordinal, "generator": "broad-coverage-1"}
            row.update(island_to_row(island))
            row.update(forcing_to_row(forcing))
            rows.append(row)
    return rows


def coverage_summary(islands, forcings, cases):
    return {"terrain_count": len(islands), "forcing_count": len(forcings),
            "terrain_shapes": dict(Counter(x.shape for x in islands)),
            "forcing_families": dict(Counter(x.family for x in forcings)),
            "candidate_split_counts": dict(Counter(x["split"] for x in cases)),
            "groups": dict(Counter(f"{x['split']}:{x['generalization']}" for x in cases)),
            "scope": "fixed-spectrum stationary single-peak conditions; synthetic, not climatological"}
