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
