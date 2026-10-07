"""Result types."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ARC = 0  # wedge bounded by the range circle
SEGMENT = 1  # wedge bounded by a wall


@dataclass(frozen=True)
class Flags:
    """Conditions under which the metrics are not ordinary numbers."""

    on_wall: bool = False  # observer closer than eps to a wall: metrics are NaN
    unbounded: bool = False  # some ray escapes and no range limit is set: area is inf
    inside_occluder: bool = False  # observer inside a solid occluder (point clouds): metrics NaN


@dataclass(frozen=True)
class Isovist:
    """Exact 2D isovist from one observer.

    The visible region is a union of angular wedges. Wedge ``i`` spans the
    angles ``[theta0[i], theta1[i]]`` (absolute, radians). It is bounded
    either by a wall (``kind == SEGMENT``), whose supporting line lies at
    perpendicular distance ``p[i]`` from the observer in direction ``phi[i]``,
    so that the visible depth is ``r(theta) = p / cos(theta - phi)``, or by
    the range circle (``kind == ARC``), where ``r = max_distance``.
    """

    origin: np.ndarray
    max_distance: float
    fov: float
    direction: float
    theta0: np.ndarray
    theta1: np.ndarray
    kind: np.ndarray
    p: np.ndarray
    phi: np.ndarray
    clearance: float
    flags: Flags = field(default_factory=Flags)
    metrics: dict = field(default_factory=dict)

    @property
    def area(self) -> float:
        return self.metrics["area"]

    @property
    def perimeter(self) -> float:
        return self.metrics["perimeter"]

    def radial(self, n: int, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Exact visible depth along ``n`` equally spaced rays.

        The rays start at ``offset`` (radians, absolute) for a full circle, or
        span the field of view otherwise. Returns ``(angles, depths)``.
        This is how a ray-sampling tool (Grasshopper, depthmapX) sees the
        same scene.
        """
        full = self.fov >= 2 * np.pi - 1e-12
        if full:
            ang = offset + 2 * np.pi * np.arange(n) / n
        else:
            start = self.direction - self.fov / 2
            ang = start + self.fov * np.arange(n) / (n - 1)
        return ang, self.depth(ang)

    def depth(self, angles) -> np.ndarray:
        """Exact visible depth at absolute angles (NaN outside the field of view)."""
        ang = np.asarray(angles, dtype=float)
        base = self.theta0[0] if len(self.theta0) else 0.0
        rel = np.mod(ang - base, 2 * np.pi)
        edges = self.theta1 - base
        i = np.searchsorted(edges, rel, side="left")
        out = np.full(ang.shape, np.nan)
        ok = i < len(edges)
        i = np.minimum(i, len(edges) - 1)
        seg = ok & (self.kind[i] == SEGMENT)
        out[seg] = self.p[i[seg]] / np.cos(ang[seg] - self.phi[i[seg]])
        arc = ok & (self.kind[i] == ARC)
        out[arc] = self.max_distance
        return out

    def polygon(self, arc_step: float = np.radians(0.5)) -> np.ndarray:
        """Boundary as a closed ring, shape (k, 2), in absolute coordinates.

        Wall pieces are exact. Range arcs are sampled every ``arc_step`` radians.
        """
        if self.flags.on_wall or self.flags.unbounded or len(self.theta0) == 0:
            return np.empty((0, 2))
        pts = []
        full = self.fov >= 2 * np.pi - 1e-12
        if not full:
            pts.append(np.zeros((1, 2)))
        wedges = zip(self.theta0, self.theta1, self.kind, self.p, self.phi, strict=True)
        for t0, t1, k, p, phi in wedges:
            if k == SEGMENT:
                th = np.array([t0, t1])
                r = p / np.cos(th - phi)
            else:
                m = max(2, int(np.ceil((t1 - t0) / arc_step)) + 1)
                th = np.linspace(t0, t1, m)
                r = np.full(m, self.max_distance)
            pts.append(np.c_[r * np.cos(th), r * np.sin(th)])
        ring = np.concatenate(pts)
        keep = np.r_[True, np.linalg.norm(np.diff(ring, axis=0), axis=1) > 1e-12]
        return ring[keep] + self.origin
