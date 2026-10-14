"""Ray directions for view volumes and horizontal sections.

A view volume is the integral V = 1/3 int r(omega)^3 d omega over the unit
sphere, where r(omega) is the visible depth in direction omega. It is
estimated from N directions, each carrying the solid angle 4 pi / N:

- ``fibonacci_sphere``: a deterministic equal-area spiral. Every direction
  owns the same patch of the sphere, so the estimate is a quasi-Monte Carlo
  rule whose error falls faster than N^(-1/2).
- ``random_sphere``: independent uniform directions, the Monte Carlo rule.
  Its error is a sampling error with standard deviation O(N^(-1/2)).
- ``equiangular``: N equally spaced directions on the horizontal ring, for
  2D sections. It does not cover the sphere and cannot weight a volume.
- ``angle_grid``: the centres of a grid of cells in azimuth and elevation,
  each with the exact solid angle of its cell. The cells shrink towards the
  poles, so the directions are not equal-area and carry their own weights.

A field of view restricts a view to the directions whose azimuth lies within
``hfov / 2`` of ``direction`` and whose elevation lies within ``vfov / 2`` of
``elevation`` (``in_field_of_view``). Its solid angle is
``hfov (sin e1 - sin e0)``, with ``e0`` and ``e1`` the elevations of its lower
and upper edge.
"""

from __future__ import annotations

import numpy as np

GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def fibonacci_sphere(n: int) -> np.ndarray:
    """Equal-area spiral of ``n`` unit vectors, shape (n, 3).

    Direction ``i`` has height ``z = 1 - (2 i + 1) / n`` and azimuth
    ``i * pi (3 - sqrt 5)``. The heights split the sphere into ``n`` bands of
    equal area, so each direction stands for the same solid angle.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    i = np.arange(n, dtype=float)
    z = 1.0 - (2.0 * i + 1.0) / n
    rho = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    phi = np.mod(i * GOLDEN_ANGLE, 2 * np.pi)
    return np.c_[rho * np.cos(phi), rho * np.sin(phi), z]


def random_sphere(n: int, seed=None) -> np.ndarray:
    """``n`` independent unit vectors, uniform on the sphere, shape (n, 3).

    Parameters
    ----------
    n : int
        Number of directions.
    seed : int or numpy.random.Generator, optional
        Seed of the generator. The same seed gives the same directions.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    rng = np.random.default_rng(seed)
    # Uniform height and azimuth give a uniform direction (Archimedes' hat-box theorem).
    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    rho = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    d = np.c_[rho * np.cos(phi), rho * np.sin(phi), z]
    return d / np.linalg.norm(d, axis=1)[:, None]


def equiangular(n: int) -> np.ndarray:
    """``n`` horizontal unit vectors at angles ``2 pi k / n`` from +x, shape (n, 3)."""
    n = int(n)
    if n < 1:
        raise ValueError("n must be at least 1")
    theta = 2 * np.pi * np.arange(n) / n
    return np.c_[np.cos(theta), np.sin(theta), np.zeros(n)]


def solid_angle_weights(n: int) -> np.ndarray:
    """Solid angle carried by each of ``n`` equal-area directions: ``4 pi / n`` each."""
    return np.full(int(n), 4 * np.pi / int(n))


def check_field_of_view(hfov: float, vfov: float, elevation: float) -> tuple[float, float]:
    """Lower and upper elevation of a field of view, after checking that it lies on the sphere.

    Raises
    ------
    ValueError
        When ``hfov`` is not in (0, 2 pi], ``vfov`` is not in (0, pi], or the
        vertical field reaches beyond a pole.
    """
    hfov, vfov, elevation = float(hfov), float(vfov), float(elevation)
    if not 0 < hfov <= 2 * np.pi:
        raise ValueError("hfov must lie in (0, 2 pi]")
    if not 0 < vfov <= np.pi:
        raise ValueError("vfov must lie in (0, pi]")
    lo, hi = elevation - vfov / 2, elevation + vfov / 2
    if lo < -np.pi / 2 - 1e-12 or hi > np.pi / 2 + 1e-12:
        raise ValueError("the vertical field of view must lie between the poles: "
                         "|elevation| + vfov / 2 <= pi / 2")
    return max(lo, -np.pi / 2), min(hi, np.pi / 2)


def in_field_of_view(directions, *, hfov: float = 2 * np.pi, vfov: float = np.pi,
                     direction: float = 0.0, elevation: float = 0.0) -> np.ndarray:
    """Which directions lie inside a field of view, as a boolean array.

    A direction lies inside when its azimuth lies in
    ``[direction - hfov / 2, direction + hfov / 2)`` and its elevation in
    ``[elevation - vfov / 2, elevation + vfov / 2)``, with the top pole
    included in a field that reaches it. Each lower edge belongs to the field
    and each upper edge to its neighbour, so that adjacent fields partition
    the sphere.

    Parameters
    ----------
    directions : array_like of shape (n, 3)
        Unit vectors.
    hfov : float
        Horizontal field of view in radians, centred on ``direction``. Default:
        full turn.
    vfov : float
        Vertical field of view in radians, centred on ``elevation``. Default:
        pole to pole.
    direction : float
        Azimuth of the centre of the field, in radians counter-clockwise from +x.
    elevation : float
        Elevation of the centre of the field, in radians above the horizontal
        plane. The vertical field must lie between the poles.

    Returns
    -------
    ndarray of shape (n,)
        True for the directions inside the field.
    """
    lo, hi = check_field_of_view(hfov, vfov, elevation)
    d = np.asarray(directions, dtype=float).reshape(-1, 3)
    keep = np.ones(len(d), dtype=bool)
    if hfov < 2 * np.pi:
        az = np.arctan2(d[:, 1], d[:, 0])
        off = np.mod(az - direction + np.pi, 2 * np.pi) - np.pi  # in [-pi, pi)
        keep &= (off >= -hfov / 2) & (off < hfov / 2)
    if lo > -np.pi / 2 or hi < np.pi / 2:
        el = np.arcsin(np.clip(d[:, 2], -1.0, 1.0))
        keep &= (el >= lo) & ((el < hi) | (hi >= np.pi / 2))
    return keep


def angle_grid(n_azimuth: int, n_elevation: int, *, hfov: float = 2 * np.pi,
               vfov: float = np.pi, direction: float = 0.0,
               elevation: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Cell centres of an azimuth-elevation grid over a field of view, with their solid angles.

    The field is cut into ``n_azimuth`` equal steps of azimuth and
    ``n_elevation`` equal steps of elevation. Each cell is represented by the
    direction at its centre and weighted by its exact solid angle

        w = (hfov / n_azimuth) (sin e_hi - sin e_lo),

    with ``e_lo`` and ``e_hi`` the elevations of its lower and upper edge.
    The weights sum to the solid angle of the field, ``4 pi`` for the full
    sphere, so ``V = 1/3 sum w_i r_i^3`` is the midpoint rule of the view
    volume.

    Parameters
    ----------
    n_azimuth, n_elevation : int
        Number of cells along azimuth and along elevation.
    hfov, vfov, direction, elevation : float
        The field of view, as in [`in_field_of_view`][pysovist.directions.in_field_of_view].

    Returns
    -------
    directions : ndarray of shape (n_azimuth * n_elevation, 3)
        Unit vectors, azimuth-major: azimuths rise counter-clockwise, and the
        elevations rise within each azimuth. ``reshape(n_azimuth,
        n_elevation, 3)`` gives the grid.
    weights : ndarray of shape (n_azimuth * n_elevation,)
        Solid angle of each cell in steradians.
    """
    na, ne = int(n_azimuth), int(n_elevation)
    if na < 1 or ne < 1:
        raise ValueError("the grid needs at least one cell along each axis")
    lo, hi = check_field_of_view(hfov, vfov, elevation)
    step = hfov / na
    az = direction - hfov / 2 + step * (np.arange(na) + 0.5)
    edges = lo + (hi - lo) * np.arange(ne + 1) / ne
    el = 0.5 * (edges[:-1] + edges[1:])
    w = step * (np.sin(edges[1:]) - np.sin(edges[:-1]))
    a, e = np.meshgrid(az, el, indexing="ij")
    d = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1)
    return d.reshape(-1, 3), np.tile(w, na)
