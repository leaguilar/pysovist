"""Scene builders shared by the tests."""

import numpy as np


def polygon_walls(vertices):
    """Closed polygon as wall segments, shape (n, 2, 2)."""
    v = np.asarray(vertices, dtype=float)
    return np.stack([v, np.roll(v, -1, axis=0)], axis=1)


def rect(x0, y0, x1, y1):
    return polygon_walls([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def regular_ngon(n, rho, center=(0.0, 0.0)):
    k = np.arange(n)
    v = np.c_[np.cos(2 * np.pi * k / n), np.sin(2 * np.pi * k / n)] * rho + np.asarray(center)
    return polygon_walls(v)


def room_walls(side=10.0, s=0.01, z0=1.0, z1=2.0):
    """Vertical walls of the square room [0, side]^2, sampled on a lattice of spacing s."""
    u = np.linspace(0, side, int(round(side / s)) + 1)
    z = np.linspace(z0, z1, int(round((z1 - z0) / s)) + 1)
    uu, zz = (g.ravel() for g in np.meshgrid(u, z))
    zero, full = np.zeros_like(uu), np.full_like(uu, side)
    return np.concatenate([np.c_[uu, zero, zz], np.c_[uu, full, zz],
                           np.c_[zero, uu, zz], np.c_[full, uu, zz]])


def square_rule(side, n):
    """Equiangular estimate (dtheta / 2) sum r_i^2 for a centred square of edge ``side``."""
    th = 2 * np.pi * np.arange(n) / n
    off = np.mod(th + np.pi / 4, np.pi / 2) - np.pi / 4
    r = side / 2 / np.cos(off)
    return np.pi / n * np.sum(r * r)
