"""View volumes: the 3D isovist of one eye point.

The view volume of an eye at ``o`` is the set of points visible from ``o``.
It is star-shaped around the eye and bounded by the visible depth
``r(omega)`` in each direction ``omega``, so its volume is

    V = 1/3 int_{S^2} r(omega)^3 d omega,

the 3D analogue of the isovist area ``A = 1/2 int r(theta)^2 d theta``. With
``N`` directions that each carry the solid angle ``4 pi / N`` the estimate is

    V = (4 pi / 3N) sum_i min(r_i, R)^3,

where ``R`` is an optional range limit. Fibonacci directions (the default)
make this a quasi-Monte Carlo rule, and independent random directions make
it a Monte Carlo rule (see ``pysovist.directions``).

Occluders are a triangle ``Mesh`` (exact surfaces) or a ``PointCloud``
(every point a ball of radius r). For a prism room (vertical walls between a
horizontal floor and ceiling of height H) every eye between floor and
ceiling sees ``V = H * A``, with ``A`` the exact 2D isovist area of the plan
closed along the same outline as the prism.

Policies
--------
escape
    A ray escapes when it hits nothing within ``R``. ``"clip"`` counts it
    with depth ``R`` (with ``R = inf`` the volume is then ``inf``),
    ``"zero"`` counts it with depth 0 (the Unity reference), ``"nan"`` makes
    the volume and the depth statistics NaN (``escape_fraction`` stays a
    number, and so does ``volume_up`` or ``volume_down`` when no ray escapes
    through that half). ``flags.unbounded`` is set whenever a ray escapes with
    ``R = inf``, whatever the policy, unless the eye is inside an occluder.
inside
    An eye inside a ball of a point cloud, or closer than ``eps`` to an
    occluder, sees nothing. ``"nan"`` makes its metrics NaN and ``"zero"``
    sets them to 0 (the Unity reference: every depth is clamped to 0).
    The flag is ``inside_occluder`` for point clouds and ``on_wall`` for meshes.
near_clip
    Occluders closer than ``near_clip`` to the eye are ignored: for a point
    cloud, the balls whose centre lies within ``near_clip``, for a mesh, the
    surfaces within ``near_clip`` along each ray. Rows whose clearance is
    below ``near_clip`` are flagged ``near_clipped``.

Metrics
-------
- ``volume``: V in cubic metres.
- ``r_mean``, ``r_std``, ``r_min``, ``r_max``: moments and extremes of the
  depth over the sphere, each direction weighted by its solid angle.
- ``equivalent_radius``: radius of the ball of the same volume, (3V / 4 pi)^(1/3).
- ``escape_fraction``: share of the full solid angle in which no occluder is
  hit within ``R``.
- ``volume_up``, ``volume_down``: the parts of V above and below the eye's
  horizontal plane (directions in the plane count half to each).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .directions import fibonacci_sphere, random_sphere, solid_angle_weights
from .results import Flags

VOLUME_METRIC_NAMES = (
    "volume", "r_mean", "r_std", "r_min", "r_max", "equivalent_radius", "escape_fraction",
    "volume_up", "volume_down",
)
"""Names of the view-volume metrics, in order.

They are the keys of [`ViewVolume.metrics`][pysovist.ViewVolume.metrics] and
the metric columns of [`view_volume_field`][pysovist.view_volume_field].
"""
ESCAPE_POLICIES = ("clip", "zero", "nan")
INSIDE_POLICIES = ("nan", "zero")
DIRECTION_SETS = ("fibonacci", "random")
_CHUNK_RAYS = 1 << 22  # rays per batch of observers in view_volume_field
_FIELD_KEYS = frozenset({"n_rays", "directions", "seed", "max_distance", "near_clip", "escape",
                         "inside", "eps", "radius"})


@dataclass(frozen=True)
class ViewVolume:
    """View volume of one eye point.

    ``distances[i]`` is the depth along ``directions[i]``, clipped to
    ``max_distance``, and ``hit[i]`` tells whether an occluder was hit within
    ``max_distance``. Each direction carries the solid angle ``weights[i]``.

    Attributes
    ----------
    origin : ndarray of shape (3,)
        Eye position in metres.
    directions : ndarray of shape (n, 3)
        Unit ray directions.
    weights : ndarray of shape (n,)
        Solid angle of each direction, ``4 pi / n``.
    distances : ndarray of shape (n,)
        Depth along each direction, clipped to ``max_distance``. It is ``inf``
        where a ray escapes with no range limit. The escape policy does not
        change it.
    hit : ndarray of shape (n,)
        True where an occluder is hit within ``max_distance``.
    max_distance : float
        Range limit R in metres, ``inf`` for none.
    clearance : float
        Distance from the eye to the nearest occluder surface, negative inside
        a ball of a point cloud.
    flags : Flags
        Conditions under which the metrics are not ordinary numbers.
    near_clipped : bool
        True when ``near_clip`` is positive and the clearance lies below it.
    metrics : dict
        The metrics, keyed by the names of
        [`VOLUME_METRIC_NAMES`][pysovist.VOLUME_METRIC_NAMES].
    """

    origin: np.ndarray
    directions: np.ndarray
    weights: np.ndarray
    distances: np.ndarray
    hit: np.ndarray
    max_distance: float
    clearance: float
    flags: Flags = field(default_factory=Flags)
    near_clipped: bool = False
    metrics: dict = field(default_factory=dict)

    @property
    def volume(self) -> float:
        """View volume in cubic metres, the same as ``metrics["volume"]``."""
        return self.metrics["volume"]

    @property
    def n_rays(self) -> int:
        """Number of ray directions."""
        return len(self.directions)

    def points(self) -> np.ndarray:
        """End point of every ray, shape (n, 3) (``inf`` where a ray escapes to infinity)."""
        return self.origin + self.directions * self.distances[:, None]


def view_volume(
    occluders3d,
    origin,
    *,
    n_rays: int = 16384,
    directions="fibonacci",
    seed=None,
    max_distance: float = np.inf,
    near_clip: float = 0.0,
    escape: str = "clip",
    inside: str = "nan",
    eps: float = 1e-9,
    radius: float | None = None,
) -> ViewVolume:
    """View volume of one eye point.

    Parameters
    ----------
    occluders3d : Mesh or PointCloud
        The scene.
    origin : (x, y, z)
        Eye position in metres.
    n_rays : int
        Number of directions, ignored when ``directions`` is an array.
    directions : str or array_like of shape (n, 3)
        Direction set: ``"fibonacci"`` (an equal-area spiral), ``"random"``
        (independent uniform directions) or an array. An array is normalised
        and each row weighted 4 pi / n, so it should be an equal-area design.
    seed : int or numpy.random.Generator, optional
        Seed for ``directions="random"``.
    max_distance : float
        Range limit R in metres. Each depth is clipped to R.
    near_clip : float
        Distance in metres below which occluders are ignored. For a point
        cloud, the balls whose centre lies within ``near_clip`` of the eye are
        ignored. For a mesh, each ray starts at distance ``near_clip`` from
        the eye.
    escape : {"clip", "zero", "nan"}
        Depth counted for a ray that hits nothing within R: R, 0 or NaN. With
        ``"clip"`` and no range limit, one escaping ray makes the volume
        ``inf``. With ``"nan"``, it makes the volume and the depth statistics
        NaN. ``escape_fraction`` stays a number, and so does ``volume_up`` or
        ``volume_down`` when no ray escapes through that half.
    inside : {"nan", "zero"}
        Metrics of an eye inside an occluder or closer than ``eps`` to one:
        NaN or 0.
    eps : float
        Clearance in metres below which the eye counts as inside.
    radius : float, optional
        Ball radius for a point cloud, overriding the cloud's own radius.

    Returns
    -------
    ViewVolume
        Depths, flags and metrics of the eye.
    """
    occ = _occluder(occluders3d, radius)
    d = _directions(directions, n_rays, seed)
    R = _check(escape, inside, max_distance, near_clip)
    o = np.asarray(origin, dtype=float).reshape(3)
    t = occ.cast(o[None], d, max_distance=R, near_clip=near_clip)
    clearance = float(occ.clearance(o[None])[0])
    blocked = occ.blocked(o[None], eps, near_clip)
    metrics = _metrics(t, d[:, 2], R, escape, blocked, inside)
    hit = np.isfinite(t[0]) & (t[0] <= R)
    unbounded = bool(not blocked[0] and np.isinf(R) and not hit.all())
    flags = Flags(unbounded=unbounded, **{occ.block_flag: bool(blocked[0])})
    return ViewVolume(
        origin=o, directions=d, weights=solid_angle_weights(len(d)),
        distances=np.minimum(t[0], R), hit=hit, max_distance=R, clearance=clearance,
        flags=flags, near_clipped=bool(near_clip > 0 and clearance < near_clip),
        metrics={k: float(v[0]) for k, v in metrics.items()},
    )


def view_volume_field(occluders3d, origins, *, n_jobs: int = -1, **kw) -> pd.DataFrame:
    """View volumes of many eye points, one row per origin.

    Parameters
    ----------
    occluders3d : Mesh or PointCloud
        The scene.
    origins : array_like of shape (m, 3) or DataFrame
        Eye positions in metres. A DataFrame gives them in its columns
        ``x, y, z``.
    n_jobs : int
        Threads for ray casting, ``-1`` for all cores. Both back ends are
        multi-threaded (numba for point clouds, Embree for meshes), so the
        observers are batched into one kernel call per chunk.
    **kw
        Keyword arguments of [`view_volume`][pysovist.view_volume]. Every
        origin uses the same directions.

    Returns
    -------
    DataFrame
        One row per origin, in order. The columns are ``x, y, z``, the metrics
        in the order of [`VOLUME_METRIC_NAMES`][pysovist.VOLUME_METRIC_NAMES],
        ``clearance`` and the flags ``on_wall``, ``inside_occluder``,
        ``unbounded`` and ``near_clipped``.
    """
    unknown = set(kw) - _FIELD_KEYS
    if unknown:
        raise TypeError(f"unexpected keyword arguments {sorted(unknown)}")
    if n_jobs != -1 and n_jobs < 1:
        raise ValueError("n_jobs must be -1 (all cores) or a positive number of threads")
    occ = _occluder(occluders3d, kw.get("radius"))
    d = _directions(kw.get("directions", "fibonacci"), kw.get("n_rays", 16384), kw.get("seed"))
    escape, inside = kw.get("escape", "clip"), kw.get("inside", "nan")
    near_clip, eps = kw.get("near_clip", 0.0), kw.get("eps", 1e-9)
    R = _check(escape, inside, kw.get("max_distance", np.inf), near_clip)
    o = _origins(origins)
    threads = None if n_jobs == -1 else int(n_jobs)

    clearance = occ.clearance(o)
    blocked = occ.blocked(o, eps, near_clip)
    metrics = {k: np.full(len(o), np.nan) for k in VOLUME_METRIC_NAMES}
    unbounded = np.zeros(len(o), dtype=bool)
    active = np.flatnonzero(~blocked)
    step = max(1, _CHUNK_RAYS // len(d))
    for i0 in range(0, len(active), step):
        rows = active[i0:i0 + step]
        t = occ.cast(o[rows], d, max_distance=R, near_clip=near_clip, n_threads=threads)
        part = _metrics(t, d[:, 2], R, escape, np.zeros(len(rows), dtype=bool), inside)
        for k in VOLUME_METRIC_NAMES:
            metrics[k][rows] = part[k]
        if np.isinf(R):
            unbounded[rows] = ~np.isfinite(t).all(axis=1)
    if inside == "zero":
        for k in VOLUME_METRIC_NAMES:
            metrics[k][blocked] = 0.0

    flags = {"on_wall": np.zeros(len(o), dtype=bool), "inside_occluder": np.zeros(len(o), bool)}
    flags[occ.block_flag] = blocked
    return pd.DataFrame({
        "x": o[:, 0], "y": o[:, 1], "z": o[:, 2], **metrics, "clearance": clearance,
        **flags, "unbounded": unbounded,
        "near_clipped": (near_clip > 0) & (clearance < near_clip),
    })


# --- helpers ----------------------------------------------------------------------------------

def _metrics(t, dz, R, escape, blocked, inside) -> dict:
    """Metric arrays over the rows of first-hit distances ``t`` (m, n)."""
    hit = np.isfinite(t) & (t <= R)
    fill = {"clip": R, "zero": 0.0, "nan": np.nan}[escape]
    r = np.where(hit, np.minimum(t, R), fill)
    k = 4 * np.pi / (3 * t.shape[1])
    up, down, flat = dz > 0, dz < 0, dz == 0
    with np.errstate(invalid="ignore"):
        r3 = r**3
        half = 0.5 * r3[:, flat].sum(axis=1)
        volume = k * r3.sum(axis=1)
        out = {
            "volume": volume,
            "r_mean": r.mean(axis=1),
            "r_std": r.std(axis=1),
            "r_min": r.min(axis=1),
            "r_max": r.max(axis=1),
            "equivalent_radius": np.cbrt(3 * volume / (4 * np.pi)),
            "escape_fraction": 1.0 - hit.mean(axis=1),
            "volume_up": k * (r3[:, up].sum(axis=1) + half),
            "volume_down": k * (r3[:, down].sum(axis=1) + half),
        }
    value = np.nan if inside == "nan" else 0.0
    for v in out.values():
        v[blocked] = value
    return out


def _directions(directions, n_rays, seed) -> np.ndarray:
    if isinstance(directions, str):
        if directions == "fibonacci":
            return fibonacci_sphere(n_rays)
        if directions == "random":
            return random_sphere(n_rays, seed)
        if directions == "equiangular":
            raise ValueError("'equiangular' is a horizontal ring and cannot weight a volume")
        raise ValueError(f"directions must be one of {DIRECTION_SETS} or an (n, 3) array")
    d = np.asarray(directions, dtype=float).reshape(-1, 3)
    norm = np.linalg.norm(d, axis=1)
    if len(d) == 0 or np.any(norm == 0):
        raise ValueError("directions must be non-zero vectors")
    return d / norm[:, None]


def _check(escape, inside, max_distance, near_clip) -> float:
    if escape not in ESCAPE_POLICIES:
        raise ValueError(f"escape must be one of {ESCAPE_POLICIES}")
    if inside not in INSIDE_POLICIES:
        raise ValueError(f"inside must be one of {INSIDE_POLICIES}")
    R = float(max_distance)
    if not R > 0:
        raise ValueError("max_distance must be positive")
    if near_clip < 0:
        raise ValueError("near_clip must be non-negative")
    return R


def _occluder(occ, radius):
    if radius is None:
        return occ
    if not hasattr(occ, "with_radius"):
        raise TypeError("radius applies to point clouds only")
    return occ.with_radius(radius)


def _origins(origins) -> np.ndarray:
    if hasattr(origins, "columns"):
        origins = origins[["x", "y", "z"]].to_numpy()
    o = np.asarray(origins, dtype=float)
    if o.ndim != 2 or o.shape[1] != 3:
        raise ValueError("origins must have shape (m, 3)")
    return np.ascontiguousarray(o)
