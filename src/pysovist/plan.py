"""Floor plans as sets of wall segments."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_CHUNK = 512  # rows per block in the pairwise intersection search


class Plan:
    """A 2D floor plan: wall segments in the (x, y) plane, in metres.

    Parameters
    ----------
    segments : array_like of shape (n, 2, 2)
        Segment endpoints ``[[x0, y0], [x1, y1]]``. A third coordinate, if
        present, is dropped. Zero-length segments are removed. One segment
        of shape (2, 2) and nested lists of segments, such as one list per
        room, are accepted as well: every array whose last two axes have
        shape (2, 2) or (2, 3).
    eps : float
        Length in metres, finite and not negative, below which a segment
        counts as zero-length. Default 1e-12. The search for crossing points
        uses it too: two segments whose direction vectors have a cross
        product of at most ``eps`` (in square metres) count as parallel and
        do not cross.

    Attributes
    ----------
    segments : ndarray of shape (n, 2, 2)
        The segments that remain.
    vertices : ndarray of shape (k, 2)
        Every segment endpoint and every crossing point, without duplicates:
        points that round to the same multiple of 1e-9 m count once.

    Raises
    ------
    ValueError
        When ``segments`` is not an array of that shape, such as rows of
        ``x0, y0, x1, y1`` or a list of points, when it holds a coordinate
        that is not finite, or when ``eps`` is negative or not finite.

    Notes
    -----
    Between the viewing angles of the vertices, no two walls change their
    order of depth as seen from any observer, which is what makes the exact
    isovist kernel exact.
    """

    def __init__(self, segments, *, eps: float = 1e-12):
        eps = float(eps)
        if not 0 <= eps < np.inf:
            raise ValueError(f"eps must be a finite non-negative length, got {eps}")
        s = _as_segments(segments)
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

        Parameters
        ----------
        path : str or os.PathLike
            JSON file holding a list of records.
        keys : (str, str)
            Names of the two endpoint fields of a record. A third coordinate
            is dropped.

        Returns
        -------
        Plan
            One segment per record, in file order, less zero-length ones.

        Raises
        ------
        FileNotFoundError
            When ``path`` does not exist.
        KeyError
            When a record lacks one of ``keys``.
        ValueError
            When a coordinate is not finite.
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

        Returns
        -------
        Plan
            The outlines of the elements cut at that height, in metres.

        Raises
        ------
        ImportError
            When ifcopenshell is not installed.
        ValueError
            For a name in ``classes`` that is neither a preset nor an IFC
            class, when no element of those classes has geometry, for a
            ``storey`` that names no storey of the model, and for a name that
            several storeys share (give the GlobalId instead).
        """
        from .ifc import read_ifc
        from .mesh import Mesh

        vertices, triangles, elevation = read_ifc(path, classes=classes, storey=storey)
        return Mesh(vertices, triangles).section(elevation + height)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` of all segment endpoints.

        Raises ``ValueError`` for a plan without segments.
        """
        if len(self.segments) == 0:
            raise ValueError("the plan has no segments, so it has no bounds")
        p = self.segments.reshape(-1, 2)
        return (*p.min(axis=0), *p.max(axis=0))

    def grid(self, spacing: float, domain=None) -> np.ndarray:
        """Regular grid of points, shape (m, 2), over ``domain`` or the plan bounds.

        ``domain`` is ``(xmin, ymin, xmax, ymax)``. The points lie ``spacing``
        apart, starting half a ``spacing`` inside the lower corner, and every
        point lies below ``xmax`` and ``ymax``. A ``spacing`` more than twice
        the width or the height of the domain gives no points.

        Raises ``ValueError`` when ``spacing`` is not a positive finite
        number, when ``domain`` is not four finite numbers with
        ``xmin < xmax`` and ``ymin < ymax``, or when the plan has no segments
        and no ``domain`` is given.
        """
        spacing = _positive_finite(spacing, "spacing")
        xmin, ymin, xmax, ymax = self._domain(domain)
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
            When ``seed`` lies outside the cells of the grid or is not one
            finite point, when the segment from ``seed`` to the centre of its
            cell passes closer than ``eps`` to a wall, when ``spacing`` or
            ``eps`` is not a positive number (``spacing`` also finite), or for
            a ``domain`` that ``grid`` refuses.

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
        spacing = _positive_finite(spacing, "spacing")
        xmin, ymin, xmax, ymax = self._domain(domain)
        # The same lattice as grid(), to the last bit.
        xs = np.arange(xmin + spacing / 2, xmax, spacing)
        ys = np.arange(ymin + spacing / 2, ymax, spacing)
        nx, ny = len(xs), len(ys)
        s = _point2d(seed, "seed")
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

    def _domain(self, domain) -> tuple[float, float, float, float]:
        """``domain`` as four floats, or the plan bounds when it is None."""
        if domain is None:
            if len(self.segments) == 0:
                raise ValueError("the plan has no segments: pass domain=(xmin, ymin, xmax, ymax)")
            return self.bounds
        d = np.asarray(domain, dtype=float).ravel()
        if d.shape != (4,) or not np.isfinite(d).all() or not (d[0] < d[2] and d[1] < d[3]):
            raise ValueError(f"domain must be four finite numbers (xmin, ymin, xmax, ymax) with "
                             f"xmin < xmax and ymin < ymax, got {domain!r}")
        return tuple(float(v) for v in d)


def _as_segments(segments) -> np.ndarray:
    """Segments as a float array of shape (n, 2, 2), checked for shape and finite values."""
    s = np.asarray(segments, dtype=float)
    if s.size == 0:
        return np.empty((0, 2, 2))
    if s.ndim < 2 or s.shape[-2] != 2 or s.shape[-1] not in (2, 3):
        raise ValueError(f"segments must have shape (n, 2, 2) or (n, 2, 3), got {s.shape}. "
                         f"Rows of x0, y0, x1, y1 become segments with "
                         f"np.reshape(rows, (-1, 2, 2))")
    s = s.reshape(-1, 2, s.shape[-1])[:, :, :2]
    bad = ~np.isfinite(s).all(axis=(1, 2))
    if bad.any():
        k = int(np.flatnonzero(bad)[0])
        raise ValueError(f"segment coordinates must be finite: segment {k} is {s[k].tolist()}")
    return s


def _point2d(value, name: str) -> np.ndarray:
    """One point (x, y) from (x, y) or (x, y, z), refused when x or y is not finite."""
    p = np.asarray(value, dtype=float)
    if p.shape not in ((2,), (3,)):
        raise ValueError(f"{name} must be one point (x, y) or (x, y, z), got shape {p.shape}")
    if not np.isfinite(p[:2]).all():
        raise ValueError(f"{name} must be finite, got {tuple(p.tolist())}")
    return p[:2]


def _points2d(value, name: str) -> np.ndarray:
    """Points of shape (m, 2) from (m, 2), (m, 3) or one point, refused when not finite."""
    p = np.asarray(value, dtype=float)
    if p.ndim == 1:
        p = p[None]
    if p.size == 0:
        return np.empty((0, 2))
    if p.ndim != 2 or p.shape[1] not in (2, 3):
        raise ValueError(f"{name} must have shape (m, 2) or (m, 3), got {p.shape}")
    p = p[:, :2]
    bad = ~np.isfinite(p).all(axis=1)
    if bad.any():
        k = int(np.flatnonzero(bad)[0])
        raise ValueError(f"{name} must be finite: row {k} is {p[k].tolist()}")
    return p


def _positive_finite(value, name: str) -> float:
    v = float(value)
    if not (v > 0 and np.isfinite(v)):
        raise ValueError(f"{name} must be a positive finite number, got {v}")
    return v


def _view_settings(max_distance, fov, direction, eps=1.0) -> tuple[float, float, float, float]:
    """Range, field of view, direction and eps of a 2D view, checked.

    The range is positive (``inf`` for none), the field of view positive and at most a full
    turn (larger values mean the full turn), the direction finite and eps positive.
    """
    R, f, d, e = float(max_distance), float(fov), float(direction), float(eps)
    if not R > 0:
        raise ValueError(f"max_distance must be positive (inf for no limit), got {R}")
    if not f > 0:
        raise ValueError(f"fov must be a positive angle in radians (2 pi for the full turn), "
                         f"got {f}")
    if not np.isfinite(d):
        raise ValueError(f"direction must be a finite angle in radians, got {d}")
    if not e > 0:
        raise ValueError(f"eps must be a positive distance, got {e}")
    return R, min(f, 2 * np.pi), d, e


def _workers(n_jobs, chunk) -> tuple[int, int]:
    """Worker count (-1: all cores) and chunk size of a field, checked."""
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    if int(n_jobs) != n_jobs or n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive integer or -1 for all cores, got {n_jobs}")
    if int(chunk) != chunk or chunk < 1:
        raise ValueError(f"chunk must be a positive integer, got {chunk}")
    return int(n_jobs), int(chunk)


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
    # A horizontal or almost horizontal wall gives 0, inf or nan here: it spans the whole band.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
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
