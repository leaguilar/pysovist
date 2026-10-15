"""Isovists along a path.

A path is a polyline in the plan. Observers stand on it at a fixed spacing
of arc length and look along it: the centre of each field of view is the
direction of the path where the observer stands. Each observer gets the
exact isovist and the standard metrics of
[`isovist`][pysovist.isovist].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .isovist2d import TWO_PI, isovist
from .metrics2d import METRIC_NAMES
from .plan import Plan

_COLUMNS = ["s", "x", "y", "direction", *METRIC_NAMES, "clearance", "on_wall", "unbounded"]
_TOL = 1e-9  # relative tolerance of an arc length that falls on the end of the path


def sample_path(path, spacing: float, *, endpoint: bool = True
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points every ``spacing`` metres of arc length along a polyline.

    Parameters
    ----------
    path : array_like of shape (n, 2)
        Vertices of the polyline, in walking order. Columns after the
        second are ignored, and repeated vertices are skipped.
    spacing : float
        Arc length between two samples, in metres.
    endpoint : bool
        Also sample the last vertex when the length of the path is not a
        multiple of ``spacing``. The last step is then shorter.

    Returns
    -------
    s : ndarray of shape (k,)
        Arc length of each sample from the first vertex: ``0, spacing,
        2 spacing, ...``.
    points : ndarray of shape (k, 2)
        Position of each sample.
    direction : ndarray of shape (k,)
        Direction of the path at each sample, in radians counter-clockwise
        from +x. A sample on a vertex takes the direction of the segment
        that leaves it, and the last vertex that of the segment that ends
        there.

    Raises
    ------
    ValueError
        When ``spacing`` is not a positive finite number, when a vertex is
        not finite, or when the path has fewer than two distinct vertices.
    """
    if not spacing > 0:
        raise ValueError("spacing must be positive")
    if not np.isfinite(spacing):
        raise ValueError(f"spacing must be finite, got {spacing}")
    v = np.asarray(path, dtype=float)
    v = v.reshape(-1, v.shape[-1])[:, :2]
    if not np.isfinite(v).all():
        raise ValueError("path vertices must be finite")
    piece = np.diff(v, axis=0)
    length = np.hypot(piece[:, 0], piece[:, 1])
    keep = length > 0
    if not keep.any():
        raise ValueError("a path needs two distinct vertices")
    start, piece, length = v[:-1][keep], piece[keep], length[keep]
    cum = np.r_[0.0, np.cumsum(length)]
    total = cum[-1]

    s = spacing * np.arange(int(np.floor(total / spacing + _TOL)) + 1)
    tol = _TOL * max(total, spacing)
    if abs(total - s[-1]) <= tol:
        if len(s) > 1:
            s[-1] = total  # the last step reaches the end, up to rounding
    elif endpoint:
        s = np.r_[s, total]
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(length) - 1)
    frac = (s - cum[k]) / length[k]
    points = start[k] + frac[:, None] * piece[k]
    direction = np.arctan2(piece[k, 1], piece[k, 0])
    return s, points, direction


def path_isovists(
    occluders,
    path,
    spacing: float,
    *,
    fov: float = TWO_PI,
    max_distance: float = np.inf,
    endpoint: bool = True,
    on_wall: str = "nan",
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Exact isovists of observers walking along a path and looking along it.

    The observers stand at the samples of
    [`sample_path`][pysovist.paths.sample_path], and the field of view of
    each is centred on the direction of the path at its position.

    Parameters
    ----------
    occluders : Plan or array_like of shape (n, 2, 2)
        Wall segments.
    path : array_like of shape (n, 2)
        Vertices of the polyline, in walking order.
    spacing : float
        Arc length between two observers, in metres.
    fov : float
        Field of view in radians, in (0, 2 pi], centred on the direction of
        the path, as in [`isovist`][pysovist.isovist]. Default: the full
        turn, for which the direction plays no role.
    max_distance : float
        Range limit in metres, as in [`isovist`][pysovist.isovist].
    endpoint : bool
        Also place an observer on the last vertex, as in ``sample_path``.
    on_wall : {"nan", "raise"}
        What to do with an observer closer than ``eps`` to a wall, as in
        ``isovist``.
    eps : float
        Distance in metres, positive, below which an observer counts as
        standing on a wall. Default 1e-9.

    Returns
    -------
    DataFrame
        One row per observer, in walking order. The columns are ``s`` (the
        arc length), ``x, y``, ``direction`` (the centre of the field of
        view), the metrics in the order of
        [`METRIC_NAMES`][pysovist.METRIC_NAMES], ``clearance`` and the flags
        ``on_wall`` and ``unbounded``. Without ``s`` and ``direction`` it
        has the columns of [`isovist_field`][pysovist.isovist_field].

    Raises
    ------
    ValueError
        For a path or spacing that ``sample_path`` refuses, for an argument
        that [`isovist`][pysovist.isovist] refuses, and with
        ``on_wall="raise"`` when an observer stands on a wall.
    """
    plan = occluders if isinstance(occluders, Plan) else Plan(occluders)
    s, points, direction = sample_path(path, spacing, endpoint=endpoint)
    rows = []
    for si, o, theta in zip(s, points, direction, strict=True):
        iso = isovist(plan, o, max_distance=max_distance, fov=fov, direction=theta,
                      on_wall=on_wall, eps=eps)
        m = iso.metrics
        rows.append([si, o[0], o[1], theta, *(m[k] for k in METRIC_NAMES), iso.clearance,
                     iso.flags.on_wall, iso.flags.unbounded])
    df = pd.DataFrame(rows, columns=_COLUMNS)
    return df.astype({"on_wall": bool, "unbounded": bool})
