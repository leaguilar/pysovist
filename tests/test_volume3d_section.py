"""Horizontal sections of the ball model: disks, 2D grid traversal and section areas."""

import math

import numpy as np
import pytest

from pysovist.directions import equiangular
from pysovist.pointcloud import PointCloud
from pysovist.raycast import (
    build_disk_grid,
    cast_disks,
    first_hit_disks_bruteforce,
    section_disks,
    section_isovist_area,
)


def assert_same_hits(t_grid, t_ref):
    np.testing.assert_array_equal(np.isinf(t_grid), np.isinf(t_ref))
    fin = np.isfinite(t_ref)
    np.testing.assert_allclose(t_grid[fin], t_ref[fin], rtol=0, atol=1e-12)


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


# --- section disks ----------------------------------------------------------------------------

def test_points_near_the_eye_plane_become_disks():
    pts = np.array([[0, 0, 1.0], [1, 0, 1.03], [2, 0, 0.95], [3, 0, 1.05], [4, 0, 2.0]])
    centers, radii = section_disks(pts, 0.05, 1.0)
    # |z - z_eye| = 0.05 does not cut the ball of radius 0.05.
    np.testing.assert_array_equal(centers, [[0, 0], [1, 0]])
    np.testing.assert_allclose(radii, [0.05, 0.04], atol=1e-12)


# --- 2D kernel against the brute-force minimum ------------------------------------------------

@pytest.mark.parametrize("n,seed", [(1, 0), (40, 1), (2000, 2), (10000, 3)])
def test_disk_kernel_matches_brute_force(n, seed):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0, 4, size=(n, 2))
    r = rng.uniform(0.005, 0.1, size=n)
    k = min(n, 10)
    inside = c[:k] + rng.uniform(-0.5, 0.5, size=(k, 2)) * r[:k, None]
    origins = np.concatenate([rng.uniform(0, 4, size=(10, 2)), inside,
                              rng.uniform(6, 9, size=(3, 2))])
    dirs = equiangular(360)[:, :2]
    t = cast_disks(build_disk_grid(c, r), origins, dirs)
    ref = first_hit_disks_bruteforce(c, r, origins, dirs)
    assert_same_hits(t, ref)
    assert (ref == 0).any() and np.isinf(ref).any()


@pytest.mark.parametrize("cell", [0.02, 0.5])
def test_disk_kernel_any_cell_size_and_range(cell):
    rng = np.random.default_rng(9)
    c = rng.uniform(-1, 1, size=(3000, 2))
    r = rng.uniform(0.001, 0.03, size=3000)
    origins = rng.uniform(-1, 1, size=(8, 2))
    dirs = np.concatenate([equiangular(100)[:, :2], [[1.0, 0.0], [0.0, -1.0]]])
    t = cast_disks(build_disk_grid(c, r, cell_size=cell), origins, dirs, max_distance=0.3)
    ref = first_hit_disks_bruteforce(c, r, origins, dirs)
    ref[ref > 0.3] = np.inf
    assert_same_hits(t, ref)


# --- section isovist of a walled room ---------------------------------------------------------

@pytest.mark.parametrize("z_eye", [1.5, 1.505])  # on a sample row, and halfway between rows
def test_square_room_section_area(z_eye):
    side, s, radius, n = 10.0, 0.01, 0.02, 3600
    pc = PointCloud(room_walls(side, s), radius=radius)
    sec = section_isovist_area(pc, (side / 2, side / 2), z_eye, n_rays=n)
    # Each ray ends between the square eroded by r (no disk reaches inside it) and the square
    # eroded by sqrt(r^2 - s^2 / 2) (every point closer to a wall lies inside some disk).
    inner = square_rule(side - 2 * radius, n)
    outer = square_rule(side - 2 * math.sqrt(radius**2 - s**2 / 2), n)
    assert inner <= sec.area <= outer
    # The equiangular rule errs by 1.0e-6 on a square (the corners are kinks of r^2), and the
    # erosion bracket is 5.2e-4 wide, so the area is (10 - 2r)^2 to within 5.2e-4.
    target = (side - 2 * radius) ** 2
    assert inner == pytest.approx(target, rel=2e-6)
    assert outer / target - 1 < 5.2e-4
    assert sec.depths.shape == (n,) and sec.hit.all()
    assert sec.area == pytest.approx(np.pi / n * np.sum(sec.depths**2), rel=1e-14)
    assert not sec.flags.unbounded and not sec.flags.inside_occluder


def test_section_policies():
    pc = PointCloud(room_walls(4.0, 0.02, 0.0, 3.0), radius=0.03)
    o = (2.0, 2.0)
    # Above the walls the section is empty: every ray escapes.
    assert np.isinf(section_isovist_area(pc, o, 5.0, n_rays=360).area)
    assert section_isovist_area(pc, o, 5.0, n_rays=360).flags.unbounded
    assert section_isovist_area(pc, o, 5.0, n_rays=360, escape="zero").area == 0.0
    assert np.isnan(section_isovist_area(pc, o, 5.0, n_rays=360, escape="nan").area)
    ranged = section_isovist_area(pc, o, 5.0, n_rays=360, max_distance=1.0)
    assert ranged.area == pytest.approx(np.pi, rel=1e-12)
    # Inside a wall.
    inside = section_isovist_area(pc, (0.0, 2.0), 1.0, n_rays=360)
    assert inside.flags.inside_occluder and np.isnan(inside.area)
    assert section_isovist_area(pc, (0.0, 2.0), 1.0, n_rays=360, inside="zero").area == 0.0
    # The radius can be set per call.
    a = section_isovist_area(pc, o, 1.0, n_rays=360, radius=0.1).area
    b = section_isovist_area(pc.with_radius(0.1), o, 1.0, n_rays=360).area
    assert a == b
