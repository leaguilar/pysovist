"""Floor plans as sets of wall segments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_CHUNK = 512  # rows per block in the pairwise intersection search


class Plan:
    """A 2D floor plan: wall segments in the (x, y) plane, in metres.

    Parameters
    ----------
    segments : array_like of shape (n, 2, 2)
        Segment endpoints ``[[x0, y0], [x1, y1]]``. A third coordinate, if
        present, is dropped. Zero-length segments are removed.
    eps : float
        Length below which a segment counts as zero-length. It is also the
        tolerance of the search for crossing points.

    Attributes
    ----------
    segments : ndarray of shape (n, 2, 2)
        The segments that remain.
    vertices : ndarray of shape (k, 2)
        Every segment endpoint and every crossing point, without duplicates.

    Notes
    -----
    Between the viewing angles of the vertices, no two walls change their
    order of depth as seen from any observer, which is what makes the exact
    isovist kernel exact.
    """

    def __init__(self, segments, *, eps: float = 1e-12):
        s = np.asarray(segments, dtype=float)
        if s.size == 0:
            s = np.empty((0, 2, 2))
        s = s.reshape(-1, 2, s.shape[-1])[:, :, :2]
        length = np.linalg.norm(s[:, 1] - s[:, 0], axis=1)
        self.segments: np.ndarray = np.ascontiguousarray(s[length > eps])
        self.vertices: np.ndarray = _vertices(self.segments, eps)

    def __len__(self) -> int:
        return len(self.segments)

    def __repr__(self) -> str:
        return f"Plan({len(self)} segments, {len(self.vertices)} vertices)"

    @classmethod
    def from_json(cls, path, keys=("start", "end")) -> Plan:
        """Read a list of ``{"start": [x, y(, z)], "end": [x, y(, z)]}`` records.

        ``keys`` names the two endpoint fields of a record.
        """
        records = json.loads(Path(path).read_text())
        return cls([[r[keys[0]][:2], r[keys[1]][:2]] for r in records])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` of all segment endpoints."""
        p = self.segments.reshape(-1, 2)
        return (*p.min(axis=0), *p.max(axis=0))

    def grid(self, spacing: float, domain=None) -> np.ndarray:
        """Regular grid of points, shape (m, 2), over ``domain`` or the plan bounds.

        ``domain`` is ``(xmin, ymin, xmax, ymax)``. The points lie ``spacing``
        apart, starting half a ``spacing`` inside the lower corner.
        """
        xmin, ymin, xmax, ymax = self.bounds if domain is None else domain
        xs = np.arange(xmin + spacing / 2, xmax, spacing)
        ys = np.arange(ymin + spacing / 2, ymax, spacing)
        gx, gy = np.meshgrid(xs, ys)
        return np.c_[gx.ravel(), gy.ravel()]


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _vertices(seg: np.ndarray, eps: float) -> np.ndarray:
    """Segment endpoints plus pairwise crossing points, deduplicated."""
    pts = [seg.reshape(-1, 2)]
    n = len(seg)
    p, r = seg[:, 0], seg[:, 1] - seg[:, 0]
    for i0 in range(0, n, _CHUNK):
        pi, ri = p[i0:i0 + _CHUNK, None], r[i0:i0 + _CHUNK, None]
        denom = _cross(ri, r[None])
        qp = p[None] - pi
        with np.errstate(divide="ignore", invalid="ignore"):
            t = _cross(qp, r[None]) / denom
            u = _cross(qp, ri) / denom
        hit = (np.abs(denom) > eps) & (t > 0) & (t < 1) & (u > 0) & (u < 1)
        ii, jj = np.nonzero(hit)
        if len(ii):
            pts.append(pi[ii, 0] + t[ii, jj, None] * ri[ii, 0])
    v = np.concatenate(pts) if pts else np.empty((0, 2))
    if len(v) == 0:
        return v
    key = np.round(v / 1e-9).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return v[np.sort(idx)]
