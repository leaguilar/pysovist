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

With ``shape=True`` the shape metrics of ``pysovist.shape3d`` follow: the
surface and occluding area, sphericity, jaggedness, drift, elongation, convex
deficiency and the radial statistics weighted by solid angle.

Directed views
--------------
A field of view (``hfov``, ``vfov``, ``direction``, ``elevation``, see
``pysovist.directions.in_field_of_view``) restricts the volume to the wedge of
directions inside it, V = 1/3 int_{field} r^3 d omega. Only the directions
inside the field are cast, each still carrying its solid angle ``w_i``
(``4 pi / N`` for the N directions of an equal-area set, or the cell solid
angles of ``angle_grid`` passed as ``weights``), so that

    V = 1/3 sum_i w_i r_i^3.

The depth statistics become averages weighted by solid angle over the field,
and ``escape_fraction``, ``volume_up`` and ``volume_down`` refer to the field.
With the default full field and no weights, the metrics are those above, to
the last bit.

Objects and view factors
------------------------
With ``object_ids=True`` every ray reports the object it hits: the object id
of the triangle hit in a mesh (``Mesh.combine`` gives one id per input mesh),
or the index of the point whose ball is hit in a point cloud, and -1 when it
hits nothing within R. ``ViewVolume.factors`` splits the field's solid angle
into three shares that sum to 1:

- ``sky_factor``: the rays that hit nothing within R (``escape_fraction``),
- ``ground_factor``: the rays that hit an object designated as ground,
- ``object_factor``: the rays that hit any other object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._angles import direction_angle
from .directions import (
    check_field_of_view,
    fibonacci_sphere,
    in_field_of_view,
    random_sphere,
    solid_angle_weights,
)
from .results import Flags
from .shape3d import SHAPE_METRIC_NAMES, shape_metrics, triangulate

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
                         "inside", "eps", "radius", "hfov", "vfov", "direction", "elevation",
                         "weights", "ground", "shape", "jump_factor"})
FACTOR_NAMES = ("sky_factor", "ground_factor", "object_factor")
"""Names of the view factors, in order.

They are the keys of [`ViewVolume.factors`][pysovist.ViewVolume.factors] and
the factor columns of [`view_volume_field`][pysovist.view_volume_field] when it
is given ``ground``.
"""


@dataclass(frozen=True)
class ViewVolume:
    """View volume of one eye point.

    ``distances[i]`` is the depth along ``directions[i]``, clipped to
    ``max_distance``, and ``hit[i]`` tells whether an occluder was hit within
    ``max_distance``. Each direction carries the solid angle ``weights[i]``.
    A directed view holds only the directions inside its field of view.

    Attributes
    ----------
    origin : ndarray of shape (3,)
        Eye position in metres.
    directions : ndarray of shape (n, 3)
        Unit ray directions.
    weights : ndarray of shape (n,)
        Solid angle of each direction: ``4 pi / n`` for a full view, the
        share ``4 pi / N`` of the N directions of the full set for a directed
        view, or the weights given to ``view_volume``.
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
        [`VOLUME_METRIC_NAMES`][pysovist.VOLUME_METRIC_NAMES], followed with
        ``shape=True`` by those of
        [`SHAPE_METRIC_NAMES`][pysovist.SHAPE_METRIC_NAMES].
    object_ids : ndarray of shape (n,) or None
        The object each ray hits within ``max_distance``, -1 for none. Set by
        ``view_volume(..., object_ids=True)``, else None. For a mesh it is the
        object id of the triangle hit, for a point cloud the index of the
        point whose ball is hit.
    n_objects : int or None
        Number of objects of the scene, set with ``object_ids``: the
        ``n_objects`` of a mesh, the number of points of a point cloud.
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
    object_ids: np.ndarray | None = None
    n_objects: int | None = None

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

    def hits_per_object(self, minlength: int = 0) -> np.ndarray:
        """Number of rays that hit each object, indexed by object id.

        Parameters
        ----------
        minlength : int
            Minimum length of the result, for example the number of objects
            of the scene, so that objects no ray hits count 0.
        """
        ids = self._ids()
        return np.bincount(ids[ids >= 0], minlength=minlength)

    def solid_angle_per_object(self, minlength: int = 0) -> np.ndarray:
        """Solid angle in steradians in which each object is seen, indexed by object id.

        It is the sum of the weights of the rays that hit the object, so it
        counts only the part of the object inside the field of view and not
        hidden by nearer objects.
        """
        ids = self._ids()
        hit = ids >= 0
        return np.bincount(ids[hit], weights=self.weights[hit], minlength=minlength)

    def factors(self, ground=()) -> dict:
        """Sky, ground and object factors: shares of the field's solid angle.

        Parameters
        ----------
        ground : sequence of int or array_like of bool
            The object ids that count as ground, or a boolean mask with one
            entry per object. For a point cloud, the indices of the ground
            points or a mask over the points. Default: no ground.

        Returns
        -------
        dict
            ``sky_factor``, ``ground_factor`` and ``object_factor``, keyed by
            the names of [`FACTOR_NAMES`][pysovist.volume3d.FACTOR_NAMES].
            They sum to 1. They are NaN for an eye flagged ``on_wall`` or
            ``inside_occluder``.

        Raises
        ------
        ValueError
            When the view volume has no object ids, or ``ground`` names an
            object the scene does not have, holds ids that are not whole
            numbers, or is a mask without one entry per object.
        """
        ids = self._ids()
        blocked = np.array([self.flags.on_wall or self.flags.inside_occluder])
        f = _factors(ids[None], self.weights, _ground_ids(ground, self.n_objects), blocked)
        return {k: float(v[0]) for k, v in f.items()}

    def _ids(self) -> np.ndarray:
        if self.object_ids is None:
            raise ValueError("this view volume has no object ids: call view_volume with "
                             "object_ids=True")
        return self.object_ids


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
    hfov: float = 2 * np.pi,
    vfov: float = np.pi,
    direction: float | tuple[float, float] = 0.0,
    elevation: float = 0.0,
    weights=None,
    object_ids: bool = False,
    shape: bool = False,
    jump_factor: float = 1.0,
) -> ViewVolume:
    """View volume of one eye point, over the full sphere or a field of view.

    Parameters
    ----------
    occluders3d : Mesh or PointCloud
        The scene.
    origin : (x, y, z)
        One eye position in metres.
    n_rays : int
        Number of directions over the whole sphere, a whole number of at
        least 1, ignored when ``directions`` is an array. Default 16384. The
        quadrature error falls as the count grows: on a cube seen from an
        off-centre eye it is 3.7e-4 at 1,024 Fibonacci directions and 3.2e-6
        at 65,536.
    directions : str or array_like of shape (n, 3)
        Direction set: ``"fibonacci"`` (an equal-area spiral, the default),
        ``"random"`` (independent uniform directions) or an array of finite,
        non-zero vectors. An array is normalised and each row weighted
        4 pi / n, so it should be an equal-area design.
    seed : int or numpy.random.Generator, optional
        Seed for ``directions="random"``.
    max_distance : float
        Range limit R in metres, positive. Each depth is clipped to R.
        Default: none.
    near_clip : float
        Distance in metres below which occluders are ignored, non-negative
        and below ``max_distance``. Default 0. For a point cloud, the balls
        whose centre lies within ``near_clip`` of the eye are ignored. For a
        mesh, each ray starts at distance ``near_clip`` from the eye.
    escape : {"clip", "zero", "nan"}
        Depth counted for a ray that hits nothing within R: R, 0 or NaN. With
        ``"clip"`` and no range limit, one escaping ray makes the volume
        ``inf``. With ``"nan"``, it makes the volume and the depth statistics
        NaN. ``escape_fraction`` stays a number, and so does ``volume_up`` or
        ``volume_down`` when no ray escapes through that half.
    inside : {"nan", "zero"}
        Metrics of an eye inside an occluder or closer than ``eps`` to one:
        NaN (the default) or 0.
    eps : float
        Clearance in metres below which the eye counts as inside,
        non-negative. Default 1e-9.
    radius : float, optional
        Ball radius in metres for a point cloud, positive and finite,
        overriding the cloud's own radius.
    hfov : float
        Horizontal field of view in radians, in (0, 2 pi], centred on
        ``direction``. Default: full turn. A value up to a relative 1e-6
        above 2 pi, such as 2 pi in float32, is the full turn. A larger
        value raises a ``ValueError``: an angle in degrees converts with
        ``math.radians``.
    vfov : float
        Vertical field of view in radians, in (0, pi], centred on
        ``elevation``. Default: pole to pole. A value up to a relative 1e-6
        above pi is pole to pole too.
    direction : float or (dx, dy)
        Azimuth of the centre of the view: an angle in radians,
        counter-clockwise from +x, between -2 pi and 2 pi, or a horizontal
        vector ``(dx, dy)`` that points along it, which gives the same result
        as the angle ``atan2(dy, dx)``. Default 0. It matters only when
        ``hfov`` is below a full turn, and it is checked in every call. A
        larger angle raises a ``ValueError``: an angle in degrees converts
        with ``math.radians``, and an angle in radians of more than a turn
        wraps with ``math.remainder(a, math.tau)``.
    elevation : float
        Elevation of the centre of the view, in radians above the eye's
        horizontal plane (the vertical offset of the view). Default 0. The
        vertical field must lie between the poles.
    weights : array_like of shape (n,), optional
        Solid angle of each direction of an array ``directions``, positive,
        for example the cell solid angles of
        [`angle_grid`][pysovist.directions.angle_grid]. Default: ``4 pi / n``
        each.
    object_ids : bool
        Also report, for every ray, the object it hits (``ViewVolume.object_ids``),
        which [`ViewVolume.factors`][pysovist.ViewVolume.factors] and
        [`ViewVolume.hits_per_object`][pysovist.ViewVolume.hits_per_object] use.
        Default False.
    shape : bool
        Also compute the shape metrics: surface and occluding area,
        sphericity, jaggedness, drift and its direction, elongation, convex
        deficiency and the weighted radial statistics, keyed by
        [`SHAPE_METRIC_NAMES`][pysovist.SHAPE_METRIC_NAMES] after the volume
        metrics. Default False. They follow the escape and inside policies,
        and a clipped ray ends at the range limit. Directions with weights
        that cover part of the sphere need the field of view they cover. The
        [metric definitions](../metrics.md#3d-shape-metrics) page defines
        them.
    jump_factor : float
        Neighbouring rays ``i`` and ``j`` see a depth jump, the edge of an
        occluding face, when the farther ray ends behind the planes of the
        nearer ray's surface, and the nearer ray in front of the planes of the
        farther ray's surface, each by more than ``jump_factor |u_i - u_j|``
        in relative depth, with ``u`` their unit directions. Positive.
        Default 1. At any value, no wall, floor or ceiling counts as
        occluding, however close to the eye or grazing, and no edge where two
        of them meet, unless one is seen as a strip one ray wide. A larger
        value leaves smaller jumps out. Used with ``shape=True``.

    Returns
    -------
    ViewVolume
        Depths, flags and metrics of the eye.

    Raises
    ------
    ValueError
        When the scene is empty (a mesh with no triangles, a point cloud with
        no points), the origin is not one finite point, an option lies outside
        the domain given above, or no direction lies inside the field of view.
        With ``shape=True``, also when the weights of a full view do not sum
        to 4 pi (give the field of view they cover), the directions leave a
        gap wider than 10 ray spacings, or no three neighbouring directions
        lie inside the field of view.
    TypeError
        When ``radius`` is given for a mesh.
    """
    occ = _occluder(occluders3d, radius)
    d = _directions(directions, n_rays, seed)
    R = _check(escape, inside, max_distance, near_clip, eps)
    jump_factor = _jump_factor(jump_factor)
    direction = direction_angle(direction)
    d, w = _field(d, weights, hfov, vfov, direction, elevation)
    o = _eye(origin)
    if object_ids:
        t, ids = _cast_ids(occ, o[None], d, R, near_clip)
    else:
        t, ids = occ.cast(o[None], d, max_distance=R, near_clip=near_clip), None
    clearance = float(occ.clearance(o[None])[0])
    blocked = occ.blocked(o[None], eps, near_clip)
    if w is None:
        metrics = _metrics(t, d[:, 2], R, escape, blocked, inside)
    else:
        metrics = _weighted_metrics(t, d[:, 2], w, R, escape, blocked, inside)
    if shape:
        surface, wt = _shape_surface(d, w, weights is not None, hfov, vfov, direction, elevation)
        metrics.update(_shape(t, d, wt, R, escape, blocked, inside, metrics["volume"], surface,
                              jump_factor))
    hit = np.isfinite(t[0]) & (t[0] <= R)
    unbounded = bool(not blocked[0] and np.isinf(R) and not hit.all())
    flags = Flags(unbounded=unbounded, **{occ.block_flag: bool(blocked[0])})
    return ViewVolume(
        origin=o, directions=d, weights=solid_angle_weights(len(d)) if w is None else w,
        distances=np.minimum(t[0], R), hit=hit, max_distance=R, clearance=clearance,
        flags=flags, near_clipped=bool(near_clip > 0 and clearance < near_clip),
        metrics={k: float(v[0]) for k, v in metrics.items()},
        object_ids=None if ids is None else ids[0],
        n_objects=_n_objects(occ) if object_ids else None,
    )


def view_volume_field(occluders3d, origins, *, n_jobs: int = -1, **kw) -> pd.DataFrame:
    """View volumes of many eye points, one row per origin.

    Parameters
    ----------
    occluders3d : Mesh or PointCloud
        The scene.
    origins : array_like of shape (m, 3) or DataFrame
        Eye positions in metres, finite. A DataFrame gives them in its
        columns ``x, y, z``. No origins give an empty table.
    n_jobs : int
        Threads for ray casting, ``-1`` (the default) for all cores. Both back
        ends are multi-threaded (numba for point clouds, Embree for meshes), so
        the observers are batched into one kernel call per chunk.
    **kw
        Keyword arguments of [`view_volume`][pysovist.view_volume], including
        a field of view, ``weights``, ``shape`` and ``jump_factor``, except
        ``object_ids``. Every origin uses the same directions and the same
        ``direction``, one angle in radians. A vector ``(dx, dy)`` converts
        with ``math.atan2(dy, dx)``. For a direction per eye, call
        ``view_volume`` for each eye in a loop. ``ground`` (the object ids that
        count as ground, as in [`ViewVolume.factors`][pysovist.ViewVolume.factors])
        adds the view factors.

    Returns
    -------
    DataFrame
        One row per origin, in order. The columns are ``x, y, z``, the metrics
        in the order of [`VOLUME_METRIC_NAMES`][pysovist.VOLUME_METRIC_NAMES],
        ``clearance`` and the flags ``on_wall``, ``inside_occluder``,
        ``unbounded`` and ``near_clipped``. With ``ground``, the factors
        follow in the order of [`FACTOR_NAMES`][pysovist.volume3d.FACTOR_NAMES],
        NaN for flagged rows. With ``shape=True``, the shape metrics come last,
        in the order of [`SHAPE_METRIC_NAMES`][pysovist.SHAPE_METRIC_NAMES].
        Each row equals the shape metrics of its eye alone, to the last bit.

    Raises
    ------
    ValueError
        For the inputs that ``view_volume`` refuses, a ``direction`` that is
        not one angle, an origin that is not finite, ``n_jobs`` that is
        neither -1 nor positive, and a ``ground`` that names an object the
        scene does not have.
    TypeError
        For a keyword argument that ``view_volume_field`` does not take.
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
    R = _check(escape, inside, kw.get("max_distance", np.inf), near_clip, eps)
    jump_factor = _jump_factor(kw.get("jump_factor", 1.0))
    # one angle for the field and for the shape surface, whether given as an angle or a vector
    view = [kw.get("hfov", 2 * np.pi), kw.get("vfov", np.pi),
            direction_angle(kw.get("direction", 0.0), loop="view_volume"), kw.get("elevation", 0.0)]
    d, w = _field(d, kw.get("weights"), *view, loop="view_volume")
    shape = bool(kw.get("shape", False))
    if shape:
        surface, wt = _shape_surface(d, w, kw.get("weights") is not None, *view)
    ground = kw.get("ground")
    ground_ids = None if ground is None else _ground_ids(ground, _n_objects(occ))
    o = _origins(origins)
    threads = None if n_jobs == -1 else int(n_jobs)

    clearance = occ.clearance(o)
    blocked = occ.blocked(o, eps, near_clip)
    metrics = {k: np.full(len(o), np.nan) for k in VOLUME_METRIC_NAMES}
    factors = {k: np.full(len(o), np.nan) for k in FACTOR_NAMES}
    shapes = {k: np.full(len(o), np.nan) for k in SHAPE_METRIC_NAMES}
    unbounded = np.zeros(len(o), dtype=bool)
    active = np.flatnonzero(~blocked)
    step = max(1, _CHUNK_RAYS // len(d))
    for i0 in range(0, len(active), step):
        rows = active[i0:i0 + step]
        if ground_ids is None:
            t = occ.cast(o[rows], d, max_distance=R, near_clip=near_clip, n_threads=threads)
        else:
            t, ids = _cast_ids(occ, o[rows], d, R, near_clip, threads)
            for k, v in _factors(ids, _weights(w, len(d)), ground_ids).items():
                factors[k][rows] = v
        none = np.zeros(len(rows), dtype=bool)
        if w is None:
            part = _metrics(t, d[:, 2], R, escape, none, inside)
        else:
            part = _weighted_metrics(t, d[:, 2], w, R, escape, none, inside)
        for k in VOLUME_METRIC_NAMES:
            metrics[k][rows] = part[k]
        if shape:
            part = _shape(t, d, wt, R, escape, none, inside, part["volume"], surface,
                          jump_factor)
            for k in SHAPE_METRIC_NAMES:
                shapes[k][rows] = part[k]
        if np.isinf(R):
            unbounded[rows] = ~np.isfinite(t).all(axis=1)
    if inside == "zero":
        for k in VOLUME_METRIC_NAMES:
            metrics[k][blocked] = 0.0
        for k in SHAPE_METRIC_NAMES:
            shapes[k][blocked] = 0.0

    flags = {"on_wall": np.zeros(len(o), dtype=bool), "inside_occluder": np.zeros(len(o), bool)}
    flags[occ.block_flag] = blocked
    return pd.DataFrame({
        "x": o[:, 0], "y": o[:, 1], "z": o[:, 2], **metrics, "clearance": clearance,
        **flags, "unbounded": unbounded,
        "near_clipped": (near_clip > 0) & (clearance < near_clip),
        **({} if ground_ids is None else factors),
        **(shapes if shape else {}),
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


def _shape(t, d, w, R, escape, blocked, inside, volume, surface, jump_factor) -> dict:
    """Shape metric arrays over the rows of first-hit distances ``t`` (m, n)."""
    hit = np.isfinite(t) & (t <= R)
    fill = {"clip": R, "zero": 0.0, "nan": np.nan}[escape]
    r = np.where(hit, np.minimum(t, R), fill)
    value = np.nan if inside == "nan" else 0.0
    out = {k: np.full(len(r), value) for k in SHAPE_METRIC_NAMES}
    free = ~np.asarray(blocked, dtype=bool)
    if free.any():
        part = shape_metrics(r[free], d, w, volume[free], surface, jump_factor)
        for k in SHAPE_METRIC_NAMES:
            out[k][free] = part[k]
    return out


def _jump_factor(value) -> float:
    v = float(value)
    if not v > 0:
        raise ValueError("jump_factor must be a positive number: 1 is the default, and larger "
                         "values flag fewer jumps")
    return v


def _shape_surface(d, w, weighted, hfov, vfov, direction, elevation):
    """Triangulation of the directions for the shape metrics, and the solid angle of each ray."""
    wt = _weights(w, len(d))
    if hfov >= 2 * np.pi and vfov >= np.pi:
        total = wt.sum()
        if weighted and abs(total / (4 * np.pi) - 1) > 1e-6:
            raise ValueError(
                f"the weights sum to {total:.4g} sr, not the 4 pi sr of the sphere: give the "
                "field of view the directions cover (hfov, vfov, direction, elevation), so that "
                "the shape metrics can close the view along its edges")
        return triangulate(d, wt), wt
    field = {"hfov": hfov, "vfov": vfov, "direction": direction, "elevation": elevation}
    return triangulate(d, wt, field), wt


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
    if len(d) == 0 or not np.all(np.isfinite(norm) & (norm > 0)):
        raise ValueError("directions must be finite, non-zero vectors")
    return d / norm[:, None]


def _check(escape, inside, max_distance, near_clip, eps) -> float:
    if escape not in ESCAPE_POLICIES:
        raise ValueError(f"escape must be one of {ESCAPE_POLICIES}")
    if inside not in INSIDE_POLICIES:
        raise ValueError(f"inside must be one of {INSIDE_POLICIES}")
    R = float(max_distance)
    if not R > 0:
        raise ValueError("max_distance must be positive")
    if not near_clip >= 0:
        raise ValueError("near_clip must be a non-negative distance in metres")
    if not near_clip < R:
        raise ValueError("near_clip must be smaller than max_distance, or every occluder is "
                         "ignored")
    if not eps >= 0:
        raise ValueError("eps must be a non-negative distance in metres")
    return R


def _occluder(occ, radius):
    if radius is not None:
        if not hasattr(occ, "with_radius"):
            raise TypeError("radius applies to point clouds only")
        occ = occ.with_radius(radius)
    if _n_objects(occ) == 0 or (hasattr(occ, "triangles") and len(occ.triangles) == 0):
        raise ValueError("the scene is empty: a mesh with no triangles or a point cloud with no "
                         "points hides nothing. Check the file, the crop or the classes that "
                         "produced it")
    return occ


def _n_objects(occ) -> int:
    """Objects of a scene: the objects of a mesh, the points of a point cloud."""
    return int(occ.n_objects) if hasattr(occ, "n_objects") else len(occ.points)


def _eye(origin) -> np.ndarray:
    o = np.asarray(origin, dtype=float)
    if o.size != 3 or not np.all(np.isfinite(o)):
        raise ValueError("view_volume takes one finite eye point (x, y, z): for many points "
                         "use view_volume_field")
    return o.reshape(3)


def _origins(origins) -> np.ndarray:
    if hasattr(origins, "columns"):
        origins = origins[["x", "y", "z"]].to_numpy()
    o = np.asarray(origins, dtype=float)
    if o.ndim != 2 or o.shape[1] != 3:
        raise ValueError("origins must have shape (m, 3)")
    bad = int((~np.isfinite(o).all(axis=1)).sum())
    if bad:
        raise ValueError(f"{bad} of {len(o)} origins are not finite: remove those rows or give "
                         "their coordinates")
    return np.ascontiguousarray(o)


# --- directed views, object ids and view factors -----------------------------------------------

def _field(d, weights, hfov, vfov, direction, elevation, loop=None):
    """Directions inside the field of view and their solid angles.

    The weights are None for the full field without given weights, the case
    whose arithmetic stays that of 0.1.2. ``loop`` names the function of one
    eye, for the message of a field given one direction per eye.
    """
    check_field_of_view(hfov, vfov, elevation)
    direction = direction_angle(direction, loop=loop)
    full = hfov >= 2 * np.pi and vfov >= np.pi
    if weights is None and full:
        return d, None
    if weights is None:
        w = solid_angle_weights(len(d))
    else:
        w = np.array(weights, dtype=float, copy=True).reshape(-1)
        if len(w) != len(d):
            raise ValueError("weights needs one solid angle per direction")
        if not np.all(np.isfinite(w) & (w > 0)):
            raise ValueError("weights must be positive solid angles")
    if not full:
        keep = in_field_of_view(d, hfov=hfov, vfov=vfov, direction=direction,
                                elevation=elevation)
        if not keep.any():
            raise ValueError("no direction lies inside the field of view: use more rays")
        d, w = np.ascontiguousarray(d[keep]), w[keep]
    return d, w


def _weights(w, n):
    return solid_angle_weights(n) if w is None else w


def _weighted_metrics(t, dz, w, R, escape, blocked, inside) -> dict:
    """Metric arrays over the rows of ``t`` (m, n), each direction weighted by its solid angle."""
    hit = np.isfinite(t) & (t <= R)
    fill = {"clip": R, "zero": 0.0, "nan": np.nan}[escape]
    r = np.where(hit, np.minimum(t, R), fill)
    total = w.sum()
    up, down, flat = dz > 0, dz < 0, dz == 0
    with np.errstate(invalid="ignore"):
        wr3 = w * r**3
        half = 0.5 * wr3[:, flat].sum(axis=1)
        volume = wr3.sum(axis=1) / 3
        mean = (w * r).sum(axis=1) / total
        var = (w * (r - mean[:, None]) ** 2).sum(axis=1) / total
        out = {
            "volume": volume,
            "r_mean": mean,
            "r_std": np.sqrt(var),
            "r_min": r.min(axis=1),
            "r_max": r.max(axis=1),
            "equivalent_radius": np.cbrt(3 * volume / (4 * np.pi)),
            "escape_fraction": 1.0 - (w * hit).sum(axis=1) / total,
            "volume_up": (wr3[:, up].sum(axis=1) + half) / 3,
            "volume_down": (wr3[:, down].sum(axis=1) + half) / 3,
        }
    value = np.nan if inside == "nan" else 0.0
    for v in out.values():
        v[blocked] = value
    return out


def _cast_ids(occ, origins, d, R, near_clip, n_threads=None):
    """First-hit distances (m, n) and the object hit by each ray, -1 for none."""
    if hasattr(occ, "cast_hits"):
        t, tri = occ.cast_hits(origins, d, max_distance=R, near_clip=near_clip,
                               n_threads=n_threads)
        ids = np.where(tri >= 0, occ.object_ids[np.maximum(tri, 0)], -1)
        return t, ids
    t = occ.cast(origins, d, max_distance=R, near_clip=near_clip, n_threads=n_threads)
    return t, _ball_ids(occ, origins, d, t, near_clip)


def _ball_ids(cloud, origins, d, t, near_clip) -> np.ndarray:
    """Index of the ball each ray hits: the centre nearest to the hit point.

    A first hit lies on the surface of the ball it enters and outside every
    other ball the ray meets, so no centre lies nearer to it than the radius.
    Balls ignored by ``near_clip`` may still contain the hit point, so among
    the nearest centres the first one outside ``near_clip`` is taken.
    """
    ids = np.full(t.shape, -1, dtype=np.int64)
    i, j = np.nonzero(np.isfinite(t))
    if len(i) == 0:
        return ids
    o = np.asarray(origins, dtype=float).reshape(-1, 3)
    p = o[i] + d[j] * t[i, j][:, None]
    tree = cloud._tree()
    if near_clip <= 0:
        ids[i, j] = tree.query(p)[1]
        return ids
    k = min(16, len(cloud.points))
    _, near = tree.query(p, k=k)
    near = near.reshape(len(p), k)
    far = np.linalg.norm(cloud.points[near] - o[i][:, None, :], axis=2) >= near_clip
    ids[i, j] = near[np.arange(len(p)), far.argmax(axis=1)]
    return ids


def _ground_ids(ground, n_objects=None) -> np.ndarray:
    """Object ids of the ground, checked against the ``n_objects`` objects of the scene."""
    g = np.asarray(ground)
    if g.dtype == bool:
        if n_objects is not None and g.shape != (n_objects,):
            raise ValueError(f"a ground mask needs one entry per object: {n_objects} entries, "
                             f"not {g.size}")
        return np.flatnonzero(g)
    g = g.reshape(-1)
    if g.dtype.kind not in "iu":
        f = g.astype(float)
        if not np.all(np.isfinite(f) & (f == np.round(f))):
            raise ValueError("ground ids must be whole numbers, the object ids of the ground")
    g = g.astype(np.int64)
    top = np.inf if n_objects is None else n_objects
    bad = g[(g < 0) | (g >= top)]
    if len(bad):
        have = "" if n_objects is None else f": its objects are 0 to {n_objects - 1}"
        raise ValueError(f"ground names object {bad[0]}, which the scene does not have{have}")
    return g


def _factors(ids, w, ground_ids, blocked=None) -> dict:
    """Sky, ground and object factors of the rows of ``ids`` (m, n)."""
    total = w.sum()
    hit = ids >= 0
    on_ground = hit & np.isin(ids, ground_ids)
    out = {
        "sky_factor": (w * ~hit).sum(axis=1) / total,
        "ground_factor": (w * on_ground).sum(axis=1) / total,
        "object_factor": (w * (hit & ~on_ground)).sum(axis=1) / total,
    }
    if blocked is not None:
        for v in out.values():
            v[blocked] = np.nan
    return out
