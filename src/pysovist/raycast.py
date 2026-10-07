"""Ray casting against balls on a uniform grid ("splat" point clouds).

Every point of a cloud is a solid ball of radius r. The depth of a ray is
the distance to the first ball it meets, and 0 for a ray that starts inside a
ball. This is the occluder model of the Unity reference implementation and
the one used here for raw point clouds.

Grid. A uniform grid of cubic cells (edge ``h``, default 2 r) covers the
balls. Each ball is stored in every cell that its bounding box overlaps (8
cells when h = 2 r, unless it is aligned with the grid), in compressed rows:
the balls of cell ``c`` are ``items[starts[c]:starts[c + 1]]``. Bounding
boxes are padded by 1e-6 h so that rounding never loses a ball from a cell
it touches. Storing a ball in all cells it reaches,
rather than in the cell of its centre, means a traversed cell is searched on
its own, with no neighbouring cells.

Traversal. A 3D digital differential analyser (Amanatides and Woo 1987)
visits the cells along the ray in order of distance. In each cell the
nearest hit among the cell's balls is accepted once it lies within the
cell's own distance interval (t <= t_exit), else the ray moves on. This is
exact: the true first hit point lies in some cell, its ball is stored in that
cell, and an earlier cell can only return a hit before its own exit, which
would then be the first hit. A ball found in an earlier cell whose hit lies
beyond that cell is never accepted early.

The same scheme in 2D casts horizontal rays against the disks in which a
horizontal plane cuts the balls (``section_isovist_area``).
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from .directions import equiangular
from .results import Flags
from .volume3d import ESCAPE_POLICIES, INSIDE_POLICIES

try:
    import numba
    from numba import prange
except ImportError:  # pragma: no cover - optional dependency
    numba = None
    prange = range

MAX_CELLS = 2**27  # the grid coarsens beyond this many cells
_PAD = 1e-6  # bounding boxes are padded by this fraction of a cell against rounding
_CHUNK_RAYS = 1 << 21  # rays per kernel call
_BRUTE_BLOCK = 1 << 22  # ray-ball pairs per block in the brute-force reference


def _jit(**kw):
    if numba is None:  # pragma: no cover
        return lambda f: f
    return numba.njit(cache=True, nogil=True, **kw)


def _require_numba():
    if numba is None:  # pragma: no cover
        raise ImportError("point clouds need numba: pip install 'pysovist[pointcloud]'")


@contextmanager
def _threads(n: int | None):
    """Run numba kernels on ``n`` threads (``None``: all) inside the block.

    open3d ships an older TBB. Once it is loaded, numba's first parallel launch
    rejects that TBB with a warning and runs on OpenMP instead. The warning is
    silenced here, and only here.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*TBB.*", category=numba.NumbaWarning)
        if n is None:
            yield
            return
        old = numba.get_num_threads()
        numba.set_num_threads(max(1, min(int(n), numba.config.NUMBA_NUM_THREADS)))
        try:
            yield
        finally:
            numba.set_num_threads(old)


# --- grid construction ------------------------------------------------------------------------

@dataclass(frozen=True)
class BallGrid:
    """Balls of one radius sorted into a uniform grid (compressed rows).

    ``points`` holds the ball centres in cell order. ``starts`` has one entry
    per cell plus one, ``items`` indexes ``points``.
    """

    points: np.ndarray
    radius: float
    lo: np.ndarray
    cell_size: float
    dims: np.ndarray
    starts: np.ndarray
    items: np.ndarray

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.dims))


def _grid_frame(lo, hi, cell, max_cells):
    """Cell edge, padded lower corner and cell counts of a grid over the box [lo, hi]."""
    dim = len(lo)
    ext = np.maximum(hi - lo, cell)
    if np.prod(np.ceil(ext / cell)) > max_cells:
        cell = float(np.prod(ext) / max_cells) ** (1 / dim) * 1.01
    pad = _PAD * cell
    lo, hi = lo - 2 * pad, hi + 2 * pad
    dims = np.maximum(np.ceil((hi - lo) / cell), 1).astype(np.int64)
    return float(cell), pad, lo, dims


@_jit()
def _cell_range(x, half, lo, h, n):
    a = int(np.floor((x - half - lo) / h))
    b = int(np.floor((x + half - lo) / h))
    return max(a, 0), min(b, n - 1)


@_jit()
def _count_ball_cells(pts, half, lo, h, dims, counts):
    nx, ny, nz = dims[0], dims[1], dims[2]
    for i in range(pts.shape[0]):
        x0, x1 = _cell_range(pts[i, 0], half, lo[0], h, nx)
        y0, y1 = _cell_range(pts[i, 1], half, lo[1], h, ny)
        z0, z1 = _cell_range(pts[i, 2], half, lo[2], h, nz)
        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                for iz in range(z0, z1 + 1):
                    counts[(ix * ny + iy) * nz + iz] += 1


@_jit()
def _fill_ball_cells(pts, half, lo, h, dims, fill, items):
    nx, ny, nz = dims[0], dims[1], dims[2]
    for i in range(pts.shape[0]):
        x0, x1 = _cell_range(pts[i, 0], half, lo[0], h, nx)
        y0, y1 = _cell_range(pts[i, 1], half, lo[1], h, ny)
        z0, z1 = _cell_range(pts[i, 2], half, lo[2], h, nz)
        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                for iz in range(z0, z1 + 1):
                    c = (ix * ny + iy) * nz + iz
                    items[fill[c]] = i
                    fill[c] += 1


def _csr(counts):
    """Row starts (int32 when they fit) and an empty item array for per-cell counts."""
    ends = np.cumsum(counts, dtype=np.int64)
    total = int(ends[-1]) if len(ends) else 0
    dtype = np.int32 if total < np.iinfo(np.int32).max else np.int64
    starts = np.zeros(len(counts) + 1, dtype=dtype)
    starts[1:] = ends
    return starts, np.empty(total, dtype=np.int32)


def build_ball_grid(points, radius: float, cell_size: float | None = None,
                    max_cells: int = MAX_CELLS) -> BallGrid:
    """Sort balls of radius ``radius`` centred at ``points`` (n, 3) into a grid.

    Parameters
    ----------
    points : array_like of shape (n, 3)
        Ball centres in metres.
    radius : float
        Ball radius in metres.
    cell_size : float, optional
        Cell edge. Default ``2 * radius``. The result does not depend on it,
        only the speed does. The grid coarsens when it would exceed
        ``max_cells`` cells.
    max_cells : int
        Largest number of grid cells.
    """
    _require_numba()
    pts = np.ascontiguousarray(np.asarray(points, dtype=float).reshape(-1, 3))
    r = float(radius)
    if r <= 0:
        raise ValueError("radius must be positive")
    if len(pts) == 0:
        z3 = np.zeros(3)
        return BallGrid(pts, r, z3, 1.0, np.ones(3, np.int64), np.zeros(2, np.int32),
                        np.empty(0, np.int32))
    h = 2 * r if cell_size is None else float(cell_size)
    h, pad, lo, dims = _grid_frame(pts.min(axis=0) - r, pts.max(axis=0) + r, h, max_cells)
    # Store the centres in cell order so that the balls of one cell sit together in memory.
    cell_of = np.floor((pts - lo) / h).astype(np.int64)
    np.clip(cell_of, 0, dims - 1, out=cell_of)
    key = (cell_of[:, 0] * dims[1] + cell_of[:, 1]) * dims[2] + cell_of[:, 2]
    pts = np.ascontiguousarray(pts[np.argsort(key, kind="stable")])
    counts = np.zeros(int(np.prod(dims)), dtype=np.int32)
    _count_ball_cells(pts, r + pad, lo, h, dims, counts)
    starts, items = _csr(counts)
    del counts
    fill = starts[:-1].copy()
    _fill_ball_cells(pts, r + pad, lo, h, dims, fill, items)
    return BallGrid(pts, r, lo, h, dims, starts, items)


# --- traversal --------------------------------------------------------------------------------

@_jit(inline="always")
def _ball_t(ox, oy, oz, dx, dy, dz, cx, cy, cz, r2, nc2):
    """Distance along a unit ray to a ball: 0 inside the ball, inf on a miss."""
    ex = ox - cx
    ey = oy - cy
    ez = oz - cz
    dd = ex * ex + ey * ey + ez * ez
    if dd < nc2:
        return np.inf
    q = dd - r2
    if q <= 0.0:
        return 0.0
    b = ex * dx + ey * dy + ez * dz
    if b >= 0.0:
        return np.inf
    disc = b * b - q
    if disc < 0.0:
        return np.inf
    return q / (np.sqrt(disc) - b)


@_jit(inline="always")
def _axis_setup(o, d, lo, h, i):
    """Step direction and distance to the next cell boundary along one axis."""
    if d > 0.0:
        return 1, (lo + (i + 1) * h - o) / d
    if d < 0.0:
        return -1, (lo + i * h - o) / d
    return 0, np.inf


@_jit()
def _slab(o, d, lo, hi):
    """Entry and exit distance of a ray through the box [lo, hi] along one axis."""
    if d == 0.0:
        if o < lo or o > hi:
            return np.inf, -np.inf
        return -np.inf, np.inf
    t1 = (lo - o) / d
    t2 = (hi - o) / d
    if t1 > t2:
        return t2, t1
    return t1, t2


@_jit()
def _trace_balls(ox, oy, oz, dx, dy, dz, pts, r2, nc2, lo, h, dims, starts, items, rmax):
    nx, ny, nz = dims[0], dims[1], dims[2]
    ax0, ax1 = _slab(ox, dx, lo[0], lo[0] + nx * h)
    ay0, ay1 = _slab(oy, dy, lo[1], lo[1] + ny * h)
    az0, az1 = _slab(oz, dz, lo[2], lo[2] + nz * h)
    t0 = max(max(ax0, ay0), max(az0, 0.0))
    t1 = min(ax1, min(ay1, az1))
    if t0 > t1 or t0 > rmax:
        return np.inf
    ix = min(max(int(np.floor((ox + t0 * dx - lo[0]) / h)), 0), nx - 1)
    iy = min(max(int(np.floor((oy + t0 * dy - lo[1]) / h)), 0), ny - 1)
    iz = min(max(int(np.floor((oz + t0 * dz - lo[2]) / h)), 0), nz - 1)
    # Boundary distances are recomputed from the cell index at each step, so they never drift.
    sx, tx = _axis_setup(ox, dx, lo[0], h, ix)
    sy, ty = _axis_setup(oy, dy, lo[1], h, iy)
    sz, tz = _axis_setup(oz, dz, lo[2], h, iz)
    while True:
        t_exit = min(tx, min(ty, tz))
        c = (ix * ny + iy) * nz + iz
        best = np.inf
        for k in range(starts[c], starts[c + 1]):
            j = items[k]
            t = _ball_t(ox, oy, oz, dx, dy, dz, pts[j, 0], pts[j, 1], pts[j, 2], r2, nc2)
            if t < best:
                best = t
        if best <= t_exit:
            return best if best <= rmax else np.inf
        if t_exit > rmax:
            return np.inf
        if tx <= ty and tx <= tz:
            ix += sx
            if ix < 0 or ix >= nx:
                return np.inf
            tx = (lo[0] + (ix + 1 if sx > 0 else ix) * h - ox) / dx
        elif ty <= tz:
            iy += sy
            if iy < 0 or iy >= ny:
                return np.inf
            ty = (lo[1] + (iy + 1 if sy > 0 else iy) * h - oy) / dy
        else:
            iz += sz
            if iz < 0 or iz >= nz:
                return np.inf
            tz = (lo[2] + (iz + 1 if sz > 0 else iz) * h - oz) / dz


@_jit(parallel=True)
def _cast_balls_kernel(origins, dirs, pts, r2, nc2, lo, h, dims, starts, items, rmax, out):
    n_dir = dirs.shape[0]
    for k in prange(origins.shape[0] * n_dir):
        i = k // n_dir
        j = k - i * n_dir
        out[i, j] = _trace_balls(origins[i, 0], origins[i, 1], origins[i, 2],
                                 dirs[j, 0], dirs[j, 1], dirs[j, 2],
                                 pts, r2, nc2, lo, h, dims, starts, items, rmax)


def _unit(directions, dim):
    d = np.asarray(directions, dtype=float).reshape(-1, dim)
    norm = np.linalg.norm(d, axis=1)
    if np.any(norm == 0):
        raise ValueError("directions must be non-zero")
    return np.ascontiguousarray(d / norm[:, None])


def cast_balls(grid: BallGrid, origins, directions, *, max_distance: float = np.inf,
               near_clip: float = 0.0, n_threads: int | None = None) -> np.ndarray:
    """First-hit distance of every ray ``origins[i] + t directions[j]``.

    Parameters
    ----------
    grid : BallGrid
        Balls from ``build_ball_grid``.
    origins : array_like of shape (m, 3)
        Ray origins in metres.
    directions : array_like of shape (n, 3)
        Ray directions, normalised internally.
    max_distance : float
        Hits farther than this are reported as misses.
    near_clip : float
        Balls whose centre lies closer than this to the ray's origin are ignored.
    n_threads : int, optional
        Threads to use. Default: all.

    Returns
    -------
    ndarray of shape (m, n)
        Distance to the first ball, ``0`` for a ray that starts inside a ball,
        ``inf`` when no ball is hit within ``max_distance``.
    """
    _require_numba()
    o = np.ascontiguousarray(np.asarray(origins, dtype=float).reshape(-1, 3))
    d = _unit(directions, 3)
    out = np.full((len(o), len(d)), np.inf)
    if len(grid.points) == 0 or out.size == 0:
        return out
    r2 = grid.radius**2
    nc2 = float(near_clip) ** 2 if near_clip > 0 else -1.0
    step = max(1, _CHUNK_RAYS // len(d))
    with _threads(n_threads):
        for i0 in range(0, len(o), step):
            _cast_balls_kernel(o[i0:i0 + step], d, grid.points, r2, nc2, grid.lo,
                               grid.cell_size, grid.dims, grid.starts, grid.items,
                               float(max_distance), out[i0:i0 + step])
    return out


# --- brute-force reference ---------------------------------------------------------------------

def first_hit_bruteforce(points, radius: float, origins, directions, *,
                         near_clip: float = 0.0) -> np.ndarray:
    """Reference for ``cast_balls``: the minimum over all balls, with the same arithmetic.

    Quadratic in the number of balls. Meant for tests.
    """
    c = np.asarray(points, dtype=float).reshape(-1, 3)
    o = np.asarray(origins, dtype=float).reshape(-1, 3)
    d = _unit(directions, 3)
    m, n = len(o), len(d)
    oo = np.repeat(o, n, axis=0)
    dd_ = np.tile(d, (m, 1))
    out = np.full(m * n, np.inf)
    r2 = float(radius) ** 2
    nc2 = float(near_clip) ** 2 if near_clip > 0 else -1.0
    block = max(1, _BRUTE_BLOCK // max(len(c), 1))
    for k0 in range(0, m * n, block):
        ro, rd = oo[k0:k0 + block, None, :], dd_[k0:k0 + block, None, :]
        ex = ro[..., 0] - c[None, :, 0]
        ey = ro[..., 1] - c[None, :, 1]
        ez = ro[..., 2] - c[None, :, 2]
        dist2 = ex * ex + ey * ey + ez * ez
        q = dist2 - r2
        b = ex * rd[..., 0] + ey * rd[..., 1] + ez * rd[..., 2]
        disc = b * b - q
        with np.errstate(invalid="ignore", divide="ignore"):
            t = q / (np.sqrt(np.maximum(disc, 0.0)) - b)
        t = np.where((b >= 0) | (disc < 0), np.inf, t)
        t = np.where(q <= 0, 0.0, t)
        t = np.where(dist2 < nc2, np.inf, t)
        if len(c):
            out[k0:k0 + block] = t.min(axis=1)
    return out.reshape(m, n)


# --- horizontal sections ----------------------------------------------------------------------

def section_disks(points, radius: float, z_eye: float) -> tuple[np.ndarray, np.ndarray]:
    """Disks in which the plane ``z = z_eye`` cuts balls of radius ``radius``.

    A point with ``|z - z_eye| < radius`` gives a disk of radius
    ``sqrt(radius^2 - (z - z_eye)^2)`` around its (x, y). Returns the centres
    (k, 2) and radii (k,).
    """
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    dz = p[:, 2] - float(z_eye)
    keep = np.abs(dz) < radius
    return np.ascontiguousarray(p[keep, :2]), np.sqrt(radius**2 - dz[keep] ** 2)


@dataclass(frozen=True)
class DiskGrid:
    """Disks sorted into a uniform 2D grid (compressed rows), as ``BallGrid`` in 3D."""

    centers: np.ndarray
    radii: np.ndarray
    lo: np.ndarray
    cell_size: float
    dims: np.ndarray
    starts: np.ndarray
    items: np.ndarray


@_jit()
def _count_disk_cells(c, rad, pad, lo, h, dims, counts):
    nx, ny = dims[0], dims[1]
    for i in range(c.shape[0]):
        x0, x1 = _cell_range(c[i, 0], rad[i] + pad, lo[0], h, nx)
        y0, y1 = _cell_range(c[i, 1], rad[i] + pad, lo[1], h, ny)
        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                counts[ix * ny + iy] += 1


@_jit()
def _fill_disk_cells(c, rad, pad, lo, h, dims, fill, items):
    nx, ny = dims[0], dims[1]
    for i in range(c.shape[0]):
        x0, x1 = _cell_range(c[i, 0], rad[i] + pad, lo[0], h, nx)
        y0, y1 = _cell_range(c[i, 1], rad[i] + pad, lo[1], h, ny)
        for ix in range(x0, x1 + 1):
            for iy in range(y0, y1 + 1):
                c_ = ix * ny + iy
                items[fill[c_]] = i
                fill[c_] += 1


def build_disk_grid(centers, radii, cell_size: float | None = None,
                    max_cells: int = MAX_CELLS) -> DiskGrid:
    """Sort disks (centres (k, 2), radii (k,)) into a 2D grid of cell edge ``cell_size``.

    The default cell edge is the largest disk diameter.
    """
    _require_numba()
    c = np.ascontiguousarray(np.asarray(centers, dtype=float).reshape(-1, 2))
    r = np.ascontiguousarray(np.asarray(radii, dtype=float).reshape(-1))
    if len(c) == 0:
        return DiskGrid(c, r, np.zeros(2), 1.0, np.ones(2, np.int64), np.zeros(2, np.int32),
                        np.empty(0, np.int32))
    h = 2 * float(r.max()) if cell_size is None else float(cell_size)
    if h <= 0:
        raise ValueError("cell size must be positive")
    h, pad, lo, dims = _grid_frame((c - r[:, None]).min(axis=0), (c + r[:, None]).max(axis=0),
                                   h, max_cells)
    cell_of = np.clip(np.floor((c - lo) / h).astype(np.int64), 0, dims - 1)
    order = np.argsort(cell_of[:, 0] * dims[1] + cell_of[:, 1], kind="stable")
    c, r = np.ascontiguousarray(c[order]), np.ascontiguousarray(r[order])
    counts = np.zeros(int(np.prod(dims)), dtype=np.int32)
    _count_disk_cells(c, r, pad, lo, h, dims, counts)
    starts, items = _csr(counts)
    fill = starts[:-1].copy()
    _fill_disk_cells(c, r, pad, lo, h, dims, fill, items)
    return DiskGrid(c, r, lo, h, dims, starts, items)


@_jit(inline="always")
def _disk_t(ox, oy, dx, dy, cx, cy, r2):
    """Distance along a unit 2D ray to a disk: 0 inside the disk, inf on a miss."""
    ex = ox - cx
    ey = oy - cy
    q = ex * ex + ey * ey - r2
    if q <= 0.0:
        return 0.0
    b = ex * dx + ey * dy
    if b >= 0.0:
        return np.inf
    disc = b * b - q
    if disc < 0.0:
        return np.inf
    return q / (np.sqrt(disc) - b)


@_jit()
def _trace_disks(ox, oy, dx, dy, c, r2, lo, h, dims, starts, items, rmax):
    nx, ny = dims[0], dims[1]
    ax0, ax1 = _slab(ox, dx, lo[0], lo[0] + nx * h)
    ay0, ay1 = _slab(oy, dy, lo[1], lo[1] + ny * h)
    t0 = max(max(ax0, ay0), 0.0)
    t1 = min(ax1, ay1)
    if t0 > t1 or t0 > rmax:
        return np.inf
    ix = min(max(int(np.floor((ox + t0 * dx - lo[0]) / h)), 0), nx - 1)
    iy = min(max(int(np.floor((oy + t0 * dy - lo[1]) / h)), 0), ny - 1)
    sx, tx = _axis_setup(ox, dx, lo[0], h, ix)
    sy, ty = _axis_setup(oy, dy, lo[1], h, iy)
    while True:
        t_exit = min(tx, ty)
        cell = ix * ny + iy
        best = np.inf
        for k in range(starts[cell], starts[cell + 1]):
            j = items[k]
            t = _disk_t(ox, oy, dx, dy, c[j, 0], c[j, 1], r2[j])
            if t < best:
                best = t
        if best <= t_exit:
            return best if best <= rmax else np.inf
        if t_exit > rmax:
            return np.inf
        if tx <= ty:
            ix += sx
            if ix < 0 or ix >= nx:
                return np.inf
            tx = (lo[0] + (ix + 1 if sx > 0 else ix) * h - ox) / dx
        else:
            iy += sy
            if iy < 0 or iy >= ny:
                return np.inf
            ty = (lo[1] + (iy + 1 if sy > 0 else iy) * h - oy) / dy


@_jit(parallel=True)
def _cast_disks_kernel(origins, dirs, c, r2, lo, h, dims, starts, items, rmax, out):
    n_dir = dirs.shape[0]
    for k in prange(origins.shape[0] * n_dir):
        i = k // n_dir
        j = k - i * n_dir
        out[i, j] = _trace_disks(origins[i, 0], origins[i, 1], dirs[j, 0], dirs[j, 1],
                                 c, r2, lo, h, dims, starts, items, rmax)


def cast_disks(grid: DiskGrid, origins, directions, *, max_distance: float = np.inf,
               n_threads: int | None = None) -> np.ndarray:
    """First-hit distance (m, n) of 2D rays from ``origins`` (m, 2) along ``directions`` (n, 2).

    ``0`` for rays that start inside a disk, ``inf`` when no disk is hit
    within ``max_distance``.
    """
    _require_numba()
    o = np.ascontiguousarray(np.asarray(origins, dtype=float).reshape(-1, 2))
    d = _unit(directions, 2)
    out = np.full((len(o), len(d)), np.inf)
    if len(grid.centers) == 0 or out.size == 0:
        return out
    r2 = grid.radii**2
    step = max(1, _CHUNK_RAYS // len(d))
    with _threads(n_threads):
        for i0 in range(0, len(o), step):
            _cast_disks_kernel(o[i0:i0 + step], d, grid.centers, r2, grid.lo, grid.cell_size,
                               grid.dims, grid.starts, grid.items, float(max_distance),
                               out[i0:i0 + step])
    return out


def first_hit_disks_bruteforce(centers, radii, origins, directions) -> np.ndarray:
    """Reference for ``cast_disks``: the minimum over all disks, with the same arithmetic."""
    c = np.asarray(centers, dtype=float).reshape(-1, 2)
    r2 = np.asarray(radii, dtype=float).reshape(-1) ** 2
    o = np.asarray(origins, dtype=float).reshape(-1, 2)
    d = _unit(directions, 2)
    oo = np.repeat(o, len(d), axis=0)[:, None, :]
    dd_ = np.tile(d, (len(o), 1))[:, None, :]
    out = np.full(len(o) * len(d), np.inf)
    block = max(1, _BRUTE_BLOCK // max(len(c), 1))
    for k0 in range(0, len(out), block):
        ro, rd = oo[k0:k0 + block], dd_[k0:k0 + block]
        ex = ro[..., 0] - c[None, :, 0]
        ey = ro[..., 1] - c[None, :, 1]
        q = ex * ex + ey * ey - r2[None, :]
        b = ex * rd[..., 0] + ey * rd[..., 1]
        disc = b * b - q
        with np.errstate(invalid="ignore", divide="ignore"):
            t = q / (np.sqrt(np.maximum(disc, 0.0)) - b)
        t = np.where((b >= 0) | (disc < 0), np.inf, t)
        t = np.where(q <= 0, 0.0, t)
        if len(c):
            out[k0:k0 + block] = t.min(axis=1)
    return out.reshape(len(o), len(d))


@dataclass(frozen=True)
class SectionIsovist:
    """Isovist of the horizontal section of a ball model at eye height ``z_eye``.

    ``depths[i]`` is the depth along ``angles[i]`` (radians from +x),
    clipped to ``max_distance``. ``area`` is ``(dtheta / 2) sum depths^2``,
    with ``dtheta = 2 pi / n`` and the escape and inside policies applied.

    Attributes
    ----------
    origin : ndarray of shape (2,)
        Eye position in plan.
    z_eye : float
        Eye height in metres.
    angles : ndarray of shape (n,)
        Ray angles ``2 pi k / n``, in radians counter-clockwise from +x.
    depths : ndarray of shape (n,)
        Distance along each ray to the first disk, clipped to
        ``max_distance``. It is ``inf`` where a ray escapes with no range
        limit. The escape policy does not change it.
    hit : ndarray of shape (n,)
        True where a disk is hit within ``max_distance``.
    max_distance : float
        Range limit R in metres, ``inf`` for none.
    clearance : float
        Distance in the section plane from the eye to the nearest disk edge.
        Negative inside a disk, ``inf`` when the plane cuts no ball.
    flags : Flags
        ``inside_occluder`` when the clearance is below ``eps``, and
        ``unbounded`` when a ray escapes with no range limit.
    area : float
        Isovist area of the section in square metres.
    """

    origin: np.ndarray
    z_eye: float
    angles: np.ndarray
    depths: np.ndarray
    hit: np.ndarray
    max_distance: float
    clearance: float
    flags: Flags
    area: float


def section_isovist_area(pointcloud, origin_xy, z_eye: float, *, n_rays: int = 3600,
                         radius: float | None = None, max_distance: float = np.inf,
                         escape: str = "clip", inside: str = "nan",
                         eps: float = 1e-9) -> SectionIsovist:
    """Isovist area of the horizontal section of a point cloud at eye height.

    The plane ``z = z_eye`` cuts every ball (radius r) whose centre lies
    within r of it in a disk of radius ``sqrt(r^2 - (z - z_eye)^2)``. Rays
    at ``n_rays`` equal angle steps ``dtheta`` meet the first disk at depth
    ``r_i``, and the area is ``(dtheta / 2) sum r_i^2``.

    Parameters
    ----------
    pointcloud : PointCloud or array_like of shape (n, 3)
        The scan. An array is read as points of a cloud of radius 0.05 m.
    origin_xy : (x, y)
        Eye position in plan.
    z_eye : float
        Eye height in metres.
    n_rays : int
        Number of equiangular rays, starting along +x.
    radius : float, optional
        Ball radius in metres. Default: the cloud's own radius.
    max_distance : float
        Range limit R in metres. Each depth is clipped to R.
    escape : {"clip", "zero", "nan"}
        Depth counted for a ray that hits no disk within R: R, 0 or NaN. With
        ``"clip"`` and no range limit, one escaping ray makes the area
        ``inf``. With ``"nan"``, it makes the area NaN.
    inside : {"nan", "zero"}
        Area of an eye inside a disk or closer than ``eps`` to one: NaN or 0.
    eps : float
        Clearance in metres below which the eye counts as inside.

    Returns
    -------
    SectionIsovist
        Depths, flags and area of the section.
    """
    from .pointcloud import PointCloud

    if escape not in ESCAPE_POLICIES:
        raise ValueError(f"escape must be one of {ESCAPE_POLICIES}")
    if inside not in INSIDE_POLICIES:
        raise ValueError(f"inside must be one of {INSIDE_POLICIES}")
    pc = pointcloud if isinstance(pointcloud, PointCloud) else PointCloud(pointcloud)
    r_ball = pc.radius if radius is None else float(radius)
    centers, radii = section_disks(pc.points, r_ball, z_eye)
    o = np.asarray(origin_xy, dtype=float).reshape(-1)[:2]
    R = float(max_distance)
    n = int(n_rays)
    t = cast_disks(build_disk_grid(centers, radii), o[None], equiangular(n)[:, :2],
                   max_distance=R)[0]
    clearance = float((np.hypot(*(centers - o).T) - radii).min()) if len(radii) else np.inf
    blocked = clearance < eps
    hit = np.isfinite(t) & (t <= R)
    fill = {"clip": R, "zero": 0.0, "nan": np.nan}[escape]
    r = np.where(hit, np.minimum(t, R), fill)
    area = float(np.pi / n * np.sum(r * r))
    if blocked:
        area = np.nan if inside == "nan" else 0.0
    flags = Flags(inside_occluder=bool(blocked),
                  unbounded=bool(not blocked and np.isinf(R) and not hit.all()))
    return SectionIsovist(o, float(z_eye), 2 * np.pi * np.arange(n) / n, np.minimum(t, R), hit,
                          R, clearance, flags, area)
