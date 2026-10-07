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
