"""Camera trajectories around the canonical person.

All driving sequences in the bank are placed in ONE world: the capture rig of the canonical
rendering person (NeRSemble id 124). A `Rig` = that person's head centre, median camera radius,
world-up and spherical frame, azimuth/elevation envelope and the intrinsics of the frontal camera.

A trajectory spec is a list of legs, each moving the camera from where the previous leg ended to
a target (az, el, rad) over `n` frames with an easing curve; a leg without a target holds the
camera. Units:
  * default ("norm"): az/el normalised, +-1 = the outermost real rig camera (az_half = 65 deg here),
    or half of the spec's "az_span_deg" / "el_span_deg" when it gives them
  * "units": "deg":   az/el in degrees relative to the rig centre / median elevation
rad multiplies the rig radius (<1 dollies in). Presets: vda/assets/cameras/*.json (docs/cameras.md).
"""
import json
from pathlib import Path

import numpy as np

from .. import paths

PRESET_DIR = paths.ASSETS / "cameras"


def lookat_c2w(pos, target, up):
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, pos
    return c2w


class Rig:
    """Spherical camera placement around the canonical head."""

    def __init__(self, rig_vec, K):
        r = np.asarray(rig_vec, np.float64)
        self.center, self.radius = r[0:3], float(r[3])
        self.up, self.e1, self.e2, self.e3 = r[4:7], r[7:10], r[10:13], r[13:16]
        self.az_c, self.az_half, self.el_c, self.el_half = (float(x) for x in r[16:20])
        self.K = np.asarray(K, np.float64)          # intrinsics of the 1100x1604 (2x-downscaled) camera

    @classmethod
    def from_state(cls, state):
        return cls(state["rig"], state["K"])

    def place(self, az_n, el_n, rad_m):
        a = self.az_c + az_n * self.az_half
        e = self.el_c + el_n * self.el_half
        p = self.center + self.radius * rad_m * (
            np.cos(e) * (np.sin(a) * self.e1 + np.cos(a) * self.e2) + np.sin(e) * self.e3)
        return lookat_c2w(p, self.center, self.up)


def load_spec(name_or_path):
    """A preset name (vda/assets/cameras/<name>.json) or a path to a spec .json."""
    p = Path(name_or_path)
    if not p.suffix and (PRESET_DIR / f"{name_or_path}.json").exists():
        p = PRESET_DIR / f"{name_or_path}.json"
    if not p.exists():
        raise SystemExit(f"camera preset {name_or_path!r} not found; presets: {', '.join(list_presets())}")
    return json.load(open(p))


def list_presets():
    return sorted(x.stem for x in PRESET_DIR.glob("*.json"))


def _ease(name, t):
    """Leg-local time t in (0,1] -> progress in [0,1]; f(1) = 1 for every curve but `hold`."""
    n = name.lower()
    if n == "hold":
        return np.zeros_like(t)
    if n == "linear":
        return t
    if n.startswith("accel"):                       # slow -> fast; accel3 = t**3
        return t ** float(n[5:] or 2.0)
    if n.startswith("decel"):                       # fast -> slow
        return 1.0 - (1.0 - t) ** float(n[5:] or 2.0)
    if n == "smooth":                               # smoothstep
        return t * t * (3.0 - 2.0 * t)
    if n == "smoother":                             # quintic: zero velocity and acceleration at both ends
        return t * t * t * (t * (6.0 * t - 15.0) + 10.0)
    raise ValueError(f"unknown ease {name!r} (linear, smooth, smoother, accelN, decelN)")


def total_frames(spec):
    return sum(int(leg["n"]) for leg in spec["legs"])


def build(spec):
    """Expand the legs into per-frame (az, el, rad), in the spec's own units and length."""
    start = spec.get("start", {})
    az, el, rad = float(start.get("az", 0.0)), float(start.get("el", 0.0)), float(start.get("rad", 1.0))
    AZ, EL, RD = [], [], []
    for li, leg in enumerate(spec["legs"]):
        n = int(leg["n"])
        if n <= 0:
            raise ValueError(f"leg {li}: n must be positive")
        moves = ("az" in leg) or ("el" in leg) or ("rad" in leg)
        ease = leg.get("ease", "smooth") if moves else "hold"
        az_t, el_t, rd_t = float(leg.get("az", az)), float(leg.get("el", el)), float(leg.get("rad", rad))
        # frame 0 of a leg is one step into it, so the boundary frame is not repeated
        t = (np.arange(n, dtype=np.float64) + 1.0) / n
        p = _ease(ease, t)
        seg_az = az + (az_t - az) * p
        seg_el = el + (el_t - el) * p
        seg_rd = rad + (rd_t - rad) * p
        bump = float(leg.get("el_bump", 0.0))       # half-sine rise over the leg: an arc over the top
        if bump:
            seg_el = seg_el + bump * np.sin(np.pi * t)
        AZ.append(seg_az); EL.append(seg_el); RD.append(seg_rd)
        az, el, rad = seg_az[-1], seg_el[-1], seg_rd[-1]
    return np.concatenate(AZ), np.concatenate(EL), np.concatenate(RD)


def schedule(spec, n_frames):
    """-> (az, el, rad) per output frame, in the spec's units.

    The camera is never sped up: a spec longer than the driving is cut to its first n_frames (e.g. the 41 s presets
    on a 77-frame clip), a shorter one is stretched to n_frames (slower)."""
    n_spec = total_frames(spec)
    az, el, rad = build(spec)
    if n_spec > n_frames:
        az, el, rad = az[:n_frames], el[:n_frames], rad[:n_frames]
    elif n_spec < n_frames:
        u, v = np.linspace(0.0, 1.0, n_spec), np.linspace(0.0, 1.0, n_frames)
        az, el, rad = np.interp(v, u, az), np.interp(v, u, el), np.interp(v, u, rad)
    return np.asarray(az, np.float64), np.asarray(el, np.float64), np.asarray(rad, np.float64)


def angles_deg(spec, rig, n_frames):
    """-> (az_deg, el_deg, rad) per frame, relative to the rig centre / median elevation."""
    az, el, rad = schedule(spec, n_frames)
    if spec.get("units", "norm") == "deg":
        return az, el, rad
    az_half = spec["az_span_deg"] / 2.0 if "az_span_deg" in spec else np.degrees(rig.az_half)
    el_half = spec["el_span_deg"] / 2.0 if "el_span_deg" in spec else np.degrees(rig.el_half)
    return az * az_half, el * el_half, rad


def trajectory(spec, rig, n_frames):
    """-> c2w [T, 4, 4] (world = canonical rig world)."""
    az, el, rad = angles_deg(spec, rig, n_frames)
    az_n, el_n = np.radians(az) / rig.az_half, np.radians(el) / rig.el_half
    return np.stack([rig.place(a, e, r) for a, e, r in zip(az_n, el_n, rad)])
