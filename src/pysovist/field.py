"""Isovists for many observers at once."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ._angles import direction_angle
from .isovist2d import isovist
from .metrics2d import METRIC_NAMES
from .plan import Plan, _points2d, _view_settings, _workers

_COLUMNS = ["x", "y", *METRIC_NAMES, "clearance", "on_wall", "unbounded"]


def _rows(plan: Plan, origins: np.ndarray, kw: dict) -> list[list]:
    rows = []
    for o in origins:
        iso = isovist(plan, o, **kw)
        m = iso.metrics
        rows.append([o[0], o[1], *(m[k] for k in METRIC_NAMES), iso.clearance,
                     iso.flags.on_wall, iso.flags.unbounded])
    return rows


def isovist_field(occluders, origins, *, n_jobs: int = 1, chunk: int = 256, **kw) -> pd.DataFrame:
    """Exact isovist metrics for many observers, one row per observer.

    Parameters
    ----------
    occluders : Plan or array_like of shape (n, 2, 2)
        Wall segments.
    origins : array_like of shape (m, 2) or (m, 3)
        Observer positions. Columns after the second are ignored, and one
        point of shape (2,) gives one row.
    n_jobs : int
        Worker processes that share the observers, a positive integer. ``-1``
        uses all cores, and 1 runs in the calling process.
    chunk : int
        Observers per task sent to a worker process, a positive integer. With
        no more than ``chunk`` observers the field runs in the calling
        process.
    **kw
        Keyword arguments of [`isovist`][pysovist.isovist], applied to every observer.
        ``direction`` is one angle or one vector for every observer, and a
        sequence of two numbers is always a vector ``(dx, dy)``. For a
        direction per observer, call ``isovist`` for each observer in a loop.

    Returns
    -------
    DataFrame
        One row per observer, in the order of ``origins``. The columns are
        ``x, y``, the metrics in the order of
        [`METRIC_NAMES`][pysovist.METRIC_NAMES], ``clearance`` (the distance to
        the nearest wall) and the flags ``on_wall`` and ``unbounded``.

    Raises
    ------
    ValueError
        When an origin is not finite, when ``n_jobs`` or ``chunk`` is not a
        positive integer (``n_jobs`` may be -1), when ``direction`` holds one
        angle per observer, or for any argument that
        [`isovist`][pysovist.isovist] refuses.
    TypeError
        For a keyword argument that ``isovist`` does not take.
    """
    plan = occluders if isinstance(occluders, Plan) else Plan(occluders)
    origins = _points2d(origins, "origins")
    n_jobs, chunk = _workers(n_jobs, chunk)
    unknown = set(kw) - {"max_distance", "fov", "direction", "on_wall", "eps"}
    if unknown:
        raise TypeError(f"unexpected keyword arguments {sorted(unknown)}")
    if kw.get("on_wall", "nan") not in ("nan", "raise"):
        raise ValueError("on_wall must be one of ('nan', 'raise')")
    if "direction" in kw:
        kw = {**kw, "direction": direction_angle(kw["direction"], loop="isovist")}
    _view_settings(kw.get("max_distance", np.inf), kw.get("fov", 2 * np.pi),
                   kw.get("direction", 0.0), kw.get("eps", 1e-9))
    if n_jobs <= 1 or len(origins) <= chunk:
        rows = _rows(plan, origins, kw)
    else:
        parts = [origins[i:i + chunk] for i in range(0, len(origins), chunk)]
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            rows = [r for part in pool.map(_rows, [plan] * len(parts), parts, [kw] * len(parts))
                    for r in part]
    df = pd.DataFrame(rows, columns=_COLUMNS)
    return df.astype({"on_wall": bool, "unbounded": bool})
