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

import math

import numpy as np

from ._angles import beyond, degrees_hint, direction_angle, full_turn_hint

GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def _count(n, name: str = "n") -> int:
    """A whole number of at least 1, or a ValueError naming ``name``."""
    v = float(n)
    if not (math.isfinite(v) and v == int(v)):
        raise ValueError(f"{name} must be a whole number")
    if v < 1:
        raise ValueError(f"{name} must be at least 1")
    return int(v)


def fibonacci_sphere(n: int) -> np.ndarray:
    """Equal-area spiral of ``n`` unit vectors, shape (n, 3).

    Direction ``i`` has height ``z = 1 - (2 i + 1) / n`` and azimuth
    ``i * pi (3 - sqrt 5)``. The heights split the sphere into ``n`` bands of
    equal area, so each direction stands for the same solid angle.

    Raises
    ------
    ValueError
        When ``n`` is not a whole number of at least 1.
    """
    n = _count(n)
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
        Number of directions, a whole number of at least 1.
    seed : int or numpy.random.Generator, optional
        Seed of the generator. The same seed gives the same directions.
        Default: fresh entropy, different directions on every call.

    Raises
    ------
    ValueError
        When ``n`` is not a whole number of at least 1.
    """
    n = _count(n)
    rng = np.random.default_rng(seed)
    # Uniform height and azimuth give a uniform direction (Archimedes' hat-box theorem).
    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    rho = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    d = np.c_[rho * np.cos(phi), rho * np.sin(phi), z]
    return d / np.linalg.norm(d, axis=1)[:, None]


def equiangular(n: int) -> np.ndarray:
    """``n`` horizontal unit vectors at angles ``2 pi k / n`` from +x, shape (n, 3).

    Raises
    ------
    ValueError
        When ``n`` is not a whole number of at least 1.
    """
    n = _count(n)
    theta = 2 * np.pi * np.arange(n) / n
    return np.c_[np.cos(theta), np.sin(theta), np.zeros(n)]


def solid_angle_weights(n: int) -> np.ndarray:
    """Solid angle in steradians carried by each of ``n`` equal-area directions: ``4 pi / n`` each.

    Raises
    ------
    ValueError
        When ``n`` is not a whole number of at least 1.
    """
    n = _count(n)
    return np.full(n, 4 * np.pi / n)


def check_field_of_view(hfov: float, vfov: float, elevation: float) -> tuple[float, float]:
    """Lower and upper elevation of a field of view, after checking that it lies on the sphere.

    Parameters
    ----------
    hfov : float
        Horizontal field of view in radians, in (0, 2 pi]. A value up to a
        relative 1e-6 above 2 pi, such as 2 pi in float32, is the full turn.
    vfov : float
        Vertical field of view in radians, in (0, pi]. A value up to a
        relative 1e-6 above pi is pole to pole.
    elevation : float
        Elevation of the centre of the field, in radians.

    Returns
    -------
    tuple of float
        Elevations of the lower and upper edge, in radians.

    Raises
    ------
    ValueError
        When ``hfov`` is not in (0, 2 pi], ``vfov`` is not in (0, pi],
        ``elevation`` is not finite, or the vertical field reaches beyond a
        pole. A field of view of 7 or more, or a vertical one above 2 pi,
        reads as degrees, and the message suggests ``math.radians``.
    """
    _, _, lo, hi = _field_of_view(hfov, vfov, elevation)
    return lo, hi


def _field_of_view(hfov, vfov, elevation) -> tuple[float, float, float, float]:
    """``hfov`` and ``vfov`` clamped to 2 pi and pi, and the elevations of the two edges."""
    hfov, vfov, elevation = float(hfov), float(vfov), float(elevation)
    if not math.isfinite(elevation):
        raise ValueError("elevation must be a finite angle in radians")
    if not (hfov > 0 and not beyond(hfov, 2 * np.pi)):
        hint = f" {full_turn_hint(hfov)}" if hfov > 2 * np.pi else ""
        raise ValueError(f"hfov must be an angle in radians in (0, 2 pi], got {hfov}.{hint}")
    if not (vfov > 0 and not beyond(vfov, np.pi)):
        hint = ""
        if beyond(vfov, 2 * np.pi):
            hint = f" {degrees_hint(vfov)}"
        elif vfov > np.pi:
            hint = " Pole to pole is pi: for the full height, pass math.pi."
        raise ValueError(f"vfov must be an angle in radians in (0, pi], got {vfov}.{hint}")
    hfov, vfov = min(hfov, 2 * np.pi), min(vfov, np.pi)
    lo, hi = elevation - vfov / 2, elevation + vfov / 2
    if lo < -np.pi / 2 - 1e-12 or hi > np.pi / 2 + 1e-12:
        hint = f". {degrees_hint(elevation)}" if abs(elevation) > np.pi / 2 else ""
        raise ValueError("the vertical field of view must lie between the poles: "
                         f"|elevation| + vfov / 2 <= pi / 2{hint}")
    return hfov, vfov, max(lo, -np.pi / 2), min(hi, np.pi / 2)


def in_field_of_view(directions, *, hfov: float = 2 * np.pi, vfov: float = np.pi,
                     direction: float | tuple[float, float] = 0.0,
                     elevation: float = 0.0) -> np.ndarray:
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
        Unit vectors, to 1e-9.
    hfov : float
        Horizontal field of view in radians, in (0, 2 pi], centred on
        ``direction``. Default: full turn. A larger value raises a
        ``ValueError``: an angle in degrees converts with ``math.radians``.
    vfov : float
        Vertical field of view in radians, in (0, pi], centred on
        ``elevation``. Default: pole to pole.
    direction : float or (dx, dy)
        Azimuth of the centre of the field: an angle in radians,
        counter-clockwise from +x, between -2 pi and 2 pi, or a horizontal
        vector ``(dx, dy)`` that points along it, which gives the same result
        as the angle ``atan2(dy, dx)``. Default 0. A larger angle raises a
        ``ValueError``: an angle in degrees converts with ``math.radians``,
        and an angle in radians of more than a turn wraps with
        ``math.remainder(a, math.tau)``.
    elevation : float
        Elevation of the centre of the field, in radians above the horizontal
        plane. Default 0. The vertical field must lie between the poles.

    Returns
    -------
    ndarray of shape (n,)
        True for the directions inside the field.

    Raises
    ------
    ValueError
        When a direction is not a unit vector, or the field is one that
        [`check_field_of_view`][pysovist.directions.check_field_of_view]
        refuses or has a ``direction`` that is neither a finite angle between
        -2 pi and 2 pi nor a finite vector with an x or y component.
    """
    lo, hi = check_field_of_view(hfov, vfov, elevation)
    direction = direction_angle(direction)
    d = np.asarray(directions, dtype=float).reshape(-1, 3)
    norm = np.linalg.norm(d, axis=1)
    if not np.all(np.abs(norm - 1.0) <= 1e-9):
        raise ValueError("directions must be unit vectors: divide each by its length")
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
               vfov: float = np.pi, direction: float | tuple[float, float] = 0.0,
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
        Number of cells along azimuth and along elevation, whole numbers of at
        least 1.
    hfov, vfov, direction, elevation : float, and (dx, dy) for direction
        The field of view, as in [`in_field_of_view`][pysovist.directions.in_field_of_view].
        Default: the full sphere.

    Returns
    -------
    directions : ndarray of shape (n_azimuth * n_elevation, 3)
        Unit vectors, azimuth-major: azimuths rise counter-clockwise, and the
        elevations rise within each azimuth. ``reshape(n_azimuth,
        n_elevation, 3)`` gives the grid.
    weights : ndarray of shape (n_azimuth * n_elevation,)
        Solid angle of each cell in steradians.

    Raises
    ------
    ValueError
        When a count is not a whole number of at least 1, or the field is one
        that [`in_field_of_view`][pysovist.directions.in_field_of_view] refuses.
    """
    na, ne = _count(n_azimuth, "n_azimuth"), _count(n_elevation, "n_elevation")
    hfov, _, lo, hi = _field_of_view(hfov, vfov, elevation)
    direction = direction_angle(direction)
    step = hfov / na
    az = direction - hfov / 2 + step * (np.arange(na) + 0.5)
    edges = lo + (hi - lo) * np.arange(ne + 1) / ne
    el = 0.5 * (edges[:-1] + edges[1:])
    w = step * (np.sin(edges[1:]) - np.sin(edges[:-1]))
    a, e = np.meshgrid(az, el, indexing="ij")
    d = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1)
    return d.reshape(-1, 3), np.tile(w, na)

