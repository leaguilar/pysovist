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

    @classmethod
    def from_ifc(cls, path, height: float, *, classes="walls", storey=None) -> Plan:
        """Cut the elements of a BIM model from an IFC file at ``height``.

        The plan is the horizontal section of ``Mesh.from_ifc(path,
        classes=classes, storey=storey)`` at ``height`` metres above the
        storey's elevation, or above the project origin without a storey. It
        holds the outline of every element the plane cuts, so an observer in
        a room sees the inner faces of its walls. Needs the ``ifc`` extra
        (ifcopenshell).

        Parameters
        ----------
        path : str or os.PathLike
            IFC file.
        height : float
            Height of the section in metres. A solid from ``z0`` to ``z1`` is
            cut for ``z0 <= z < z1``, as in ``Mesh.section``.
        classes : str or sequence of str
            ``"walls"``, ``"furnished"``, an IFC class, or a sequence of
            presets and classes, as in ``Mesh.from_ifc``.
        storey : str, optional
            Name or GlobalId of an IfcBuildingStorey. Only its elements are
            read, and ``height`` is measured from its elevation: the z of its
            placement, or its Elevation attribute when it has no placement.
        """
        from .ifc import read_ifc
        from .mesh import Mesh

        vertices, triangles, elevation = read_ifc(path, classes=classes, storey=storey)
        return Mesh(vertices, triangles).section(elevation + height)

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

    def flood_grid(self, spacing: float, seed, domain=None, *,
                   eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
        """Cells of the grid that a flood from ``seed`` reaches without crossing a wall.

        The grid is that of [`grid`][pysovist.Plan.grid]: square cells of
        edge ``spacing`` over ``domain`` or the plan bounds, centred on the
        points of ``grid(spacing, domain)``. The flood starts in the cell
        that contains ``seed`` and passes from a cell to each of its four
        side neighbours when the straight segment between their centres
        passes no closer than ``eps`` to any wall. It stops at the walls and
        at the edge of the grid, so in an open plan it fills the bounding
        rectangle of the walls. This is the Analysis Grid of DeCodingSpaces.

        Parameters
        ----------
        spacing : float
            Edge of a square cell in metres.
        seed : (x, y)
            Start point of the flood.
        domain : (xmin, ymin, xmax, ymax), optional
            Rectangle covered by the grid, as in ``grid``. Default: the plan
            bounds.
        eps : float
            Clearance in metres. A segment between two centres that passes
            closer than ``eps`` to a wall, its endpoints included, is
            blocked, so a larger value closes gaps narrower than ``2 eps``.

        Returns
        -------
        centres : ndarray of shape (k, 2)
            Centres of the cells reached: the points of
            ``grid(spacing, domain)`` that the flood reaches, in the same
            order and with the same values.
        cells : ndarray of shape (k, 4)
            ``(xmin, ymin, xmax, ymax)`` of each cell reached.

        Raises
        ------
        ValueError
            When ``seed`` lies outside the cells of the grid, or when the
            segment from ``seed`` to the centre of its cell passes closer
            than ``eps`` to a wall.

        Notes
        -----
        A cell is reached or not as a whole, by its centre. A cell cut by a
        wall is kept whole when its centre is reached, and so belongs to the
        side of the wall on which its centre lies. A centre closer than
        ``eps`` to a wall is never reached. A gap between walls lets the
        flood through when a segment between two neighbouring centres passes
        through it.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        if not spacing > 0:
            raise ValueError("spacing must be positive")
        if not eps > 0:
            raise ValueError("eps must be positive")
        xmin, ymin, xmax, ymax = self.bounds if domain is None else domain
        # The same lattice as grid(), to the last bit.
        xs = np.arange(xmin + spacing / 2, xmax, spacing)
        ys = np.arange(ymin + spacing / 2, ymax, spacing)
        nx, ny = len(xs), len(ys)
        s = np.asarray(seed, dtype=float)[:2]
        i, j = (int(k) for k in np.floor((s - (xmin, ymin)) / spacing))
        if not (0 <= i < nx and 0 <= j < ny):
            raise ValueError(f"seed {tuple(s.tolist())} lies outside the cells of the grid")
        centre = np.array([[xs[i], ys[j]]])
        seg = self.segments
        if len(seg) and (_segment_distance(s[None], centre, seg[:, 0], seg[:, 1]) < eps).any():
            raise ValueError(f"a wall separates the seed {tuple(s.tolist())} from the centre "
                             f"{tuple(centre[0].tolist())} of its cell")

        right = _blocked_links(seg, xs, ys, eps, spacing)  # (ny, nx - 1)
        up = _blocked_links(seg[:, :, ::-1], ys, xs, eps, spacing).T  # (ny - 1, nx)
        node = np.arange(nx * ny).reshape(ny, nx)
        a = np.r_[node[:, :-1][~right], node[:-1][~up]]
        b = np.r_[node[:, 1:][~right], node[1:][~up]]
        graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(nx * ny, nx * ny))
        _, label = connected_components(graph, directed=False)
        keep = label == label[node[j, i]]
        gx, gy = np.meshgrid(xs, ys)
        centres = np.c_[gx.ravel(), gy.ravel()][keep]
        half = spacing / 2
        return centres, np.c_[centres - half, centres + half]


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
        # Near-parallel pairs give tiny denominators. Their quotients may overflow, and the
        # mask |denom| > eps below drops them.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
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


def _expand(start, count):
    """Pairs ``(k, start[k] + r)`` for ``r < count[k]``: owner indices and values."""
    owner = np.repeat(np.arange(len(start)), count)
    offset = np.arange(len(owner)) - np.repeat(np.cumsum(count) - count, count)
    return owner, np.repeat(start, count) + offset


def _point_segment_distance(p, a, d):
    """Distance from points ``p`` to segments ``a + t d``, t in [0, 1], row by row."""
    dd = np.einsum("ij,ij->i", d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dd > 0, np.einsum("ij,ij->i", p - a, d) / dd, 0.0)
    q = a + np.clip(t, 0.0, 1.0)[:, None] * d
    return np.hypot(*(p - q).T)


def _segment_distance(a, b, c, d):
    """Distance between segments ``a b`` and ``c d``, row by row (rows broadcast)."""
    a, b, c, d = np.broadcast_arrays(a, b, c, d)
    r, q = b - a, d - c
    crossing = ((_cross(r, c - a) * _cross(r, d - a) < 0)
                & (_cross(q, a - c) * _cross(q, b - c) < 0))
    dist = np.minimum.reduce([_point_segment_distance(a, c, q), _point_segment_distance(b, c, q),
                              _point_segment_distance(c, a, r), _point_segment_distance(d, a, r)])
    return np.where(crossing, 0.0, dist)


def _blocked_links(seg, xs, ys, eps, spacing):
    """Links between neighbouring centres of each row that pass closer than ``eps`` to a wall.

    The link ``(j, i)`` joins ``(xs[i], ys[j])`` and ``(xs[i + 1], ys[j])``. Each wall is
    tested only against the links of the rows it comes near, and in each row only against
    the links near the part of the wall that lies within a band around the row. The band
    and the link range are wider than ``eps``, so no blocked link is missed, and the exact
    distance decides.
    """
    nx, ny = len(xs), len(ys)
    out = np.zeros((ny, max(nx - 1, 0)), dtype=bool)
    if nx < 2 or ny == 0 or len(seg) == 0:
        return out
    pad = 2 * eps + 1e-6 * spacing
    p, q = seg[:, 0], seg[:, 1]
    j0 = np.searchsorted(ys, np.minimum(p[:, 1], q[:, 1]) - pad, side="left")
    j1 = np.searchsorted(ys, np.maximum(p[:, 1], q[:, 1]) + pad, side="right")
    w, j = _expand(j0, np.maximum(j1 - j0, 0))
    pw, dw = p[w], q[w] - p[w]
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (ys[j] - pad - pw[:, 1]) / dw[:, 1]
        t1 = (ys[j] + pad - pw[:, 1]) / dw[:, 1]
    flat = dw[:, 1] == 0
    lo = np.where(flat, 0.0, np.clip(np.minimum(t0, t1), 0.0, 1.0))
    hi = np.where(flat, 1.0, np.clip(np.maximum(t0, t1), 0.0, 1.0))
    xa, xb = pw[:, 0] + lo * dw[:, 0], pw[:, 0] + hi * dw[:, 0]
    i0 = np.clip(np.searchsorted(xs, np.minimum(xa, xb) - pad, side="left") - 2, 0, nx - 2)
    i1 = np.clip(np.searchsorted(xs, np.maximum(xa, xb) + pad, side="right") + 1, 0, nx - 1)
    k, i = _expand(i0, np.maximum(i1 - i0, 0))
    wk, jk = w[k], j[k]
    y = ys[jk]
    dist = _segment_distance(np.c_[xs[i], y], np.c_[xs[i + 1], y], p[wk], q[wk])
    hit = dist < eps
    out[jk[hit], i[hit]] = True
    return out
