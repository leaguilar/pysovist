"""Isovists for many observers at once."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .isovist2d import isovist
from .metrics2d import METRIC_NAMES
from .plan import Plan

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
    """Exact isovist metrics for every observer in ``origins``, shape (m, 2).

    Keyword arguments go to :func:`pysovist.isovist`. ``n_jobs`` processes share
    the work (``-1``: all cores). Rows keep the order of ``origins``.
    """
    plan = occluders if isinstance(occluders, Plan) else Plan(occluders)
    origins = np.asarray(origins, dtype=float)[:, :2]
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if n_jobs <= 1 or len(origins) <= chunk:
        rows = _rows(plan, origins, kw)
    else:
        parts = [origins[i:i + chunk] for i in range(0, len(origins), chunk)]
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            rows = [r for part in pool.map(_rows, [plan] * len(parts), parts, [kw] * len(parts))
                    for r in part]
    df = pd.DataFrame(rows, columns=_COLUMNS)
    return df.astype({"on_wall": bool, "unbounded": bool})
