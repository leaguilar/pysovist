"""Grid-traversal ray casting against balls, checked against the brute-force minimum."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pysovist.directions import fibonacci_sphere, random_sphere
from pysovist.raycast import build_ball_grid, cast_balls, first_hit_bruteforce

ATOL = 1e-12


def assert_same_hits(t_grid, t_ref):
    assert t_grid.shape == t_ref.shape
    np.testing.assert_array_equal(np.isinf(t_grid), np.isinf(t_ref))
    fin = np.isfinite(t_ref)
    np.testing.assert_allclose(t_grid[fin], t_ref[fin], rtol=0, atol=ATOL)


def mixed_origins(rng, pts, radius, lo, hi, m):
    """Free points, points inside balls and points outside the cloud's box."""
    free = rng.uniform(lo, hi, size=(m, 3))
    k = min(m, len(pts))
    off = random_sphere(k, seed=rng) * rng.uniform(0, 0.9 * radius, size=(k, 1))
    inside = pts[rng.choice(len(pts), k, replace=False)] + off
    outside = rng.uniform(lo, hi, size=(m, 3)) + (hi - lo) * 1.5
    return np.concatenate([free, inside, outside])


# --- single balls with known answers ----------------------------------------------------------

def test_single_ball_through_centre_tangent_and_inside():
    pts = np.array([[5.0, 0.0, 0.0]])
    g = build_ball_grid(pts, 1.0)
    dirs = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0]])
    t = cast_balls(g, np.zeros((1, 3)), dirs)
    np.testing.assert_allclose(t[0], [4.0, np.inf, np.inf])
    # Ray that starts inside the ball: depth clamped to 0 in every direction.
    t_in = cast_balls(g, np.array([[5.5, 0.0, 0.0]]), fibonacci_sphere(64))
    np.testing.assert_array_equal(t_in, 0.0)
    # Grazing ray at distance 0.6 from the centre: t = 5 - sqrt(1 - 0.36).
    t_g = cast_balls(g, np.array([[0.0, 0.6, 0.0]]), np.array([[1.0, 0, 0]]))
    assert t_g[0, 0] == pytest.approx(5 - 0.8, abs=1e-14)


def test_max_distance_and_near_clip():
    pts = np.array([[1.0, 0, 0], [3.0, 0, 0]])
    g = build_ball_grid(pts, 0.25)
    o, d = np.zeros((1, 3)), np.array([[1.0, 0, 0]])
    assert cast_balls(g, o, d)[0, 0] == pytest.approx(0.75)
    assert cast_balls(g, o, d, max_distance=0.5)[0, 0] == np.inf
    # The near ball's centre is within 1.5 of the eye and is ignored.
    assert cast_balls(g, o, d, near_clip=1.5)[0, 0] == pytest.approx(2.75)
    ref = first_hit_bruteforce(pts, 0.25, o, d, near_clip=1.5)
    assert ref[0, 0] == pytest.approx(2.75)


# --- random clouds against the brute-force minimum --------------------------------------------

@pytest.mark.parametrize("n,radius,seed", [
    (1, 0.3, 0), (50, 0.2, 1), (1000, 0.05, 2), (1000, 0.15, 3), (10000, 0.02, 4),
    (10000, 0.08, 5),
])
def test_random_cloud_matches_brute_force(n, radius, seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 2, size=(n, 3))
    origins = mixed_origins(rng, pts, radius, 0.0, 2.0, 20)
    dirs = random_sphere(300, seed=seed)
    g = build_ball_grid(pts, radius)
    t = cast_balls(g, origins, dirs)
    ref = first_hit_bruteforce(pts, radius, origins, dirs)
    assert_same_hits(t, ref)
    # The cases the test is meant to cover all occur.
    assert (ref == 0).any(axis=1).sum() >= 1
    assert np.isinf(ref).any()
    assert (np.isfinite(ref) & (ref > 0)).any()


@pytest.mark.parametrize("cell_factor", [0.3, 1.0, 2.0, 7.0])
def test_any_cell_size_is_exact(cell_factor):
    rng = np.random.default_rng(11)
    radius = 0.07
    pts = rng.uniform(-1, 1, size=(3000, 3))
    origins = mixed_origins(rng, pts, radius, -1.0, 1.0, 10)
    dirs = random_sphere(200, seed=3)
    g = build_ball_grid(pts, radius, cell_size=cell_factor * radius)
    assert_same_hits(cast_balls(g, origins, dirs), first_hit_bruteforce(pts, radius, origins, dirs))


def test_lattice_cloud_axis_aligned_rays():
    # Ball centres on cell corners and rays with zero direction components exercise ties.
    ax = np.arange(0.0, 1.01, 0.1)
    pts = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)
    radius = 0.05
    dirs = np.concatenate([np.eye(3), -np.eye(3), fibonacci_sphere(64),
                           np.array([[1.0, 1.0, 0.0], [0.0, -1.0, 1.0]]) / np.sqrt(2)])
    origins = np.array([[0.05, 0.05, 0.05], [0.5, 0.45, 0.55], [-0.3, 0.5, 0.5],
                        [0.0, 0.0, 0.0], [1.5, 0.05, 0.05], [0.25, 0.25, 0.25]])
    g = build_ball_grid(pts, radius)
    assert_same_hits(cast_balls(g, origins, dirs), first_hit_bruteforce(pts, radius, origins, dirs))


def test_rays_that_miss_everything():
    rng = np.random.default_rng(5)
    pts = rng.uniform(0, 1, size=(500, 3))
    g = build_ball_grid(pts, 0.05)
    origins = np.array([[3.0, 0.5, 0.5], [0.5, 0.5, -4.0]])
    away = np.array([[1.0, 0, 0], [0.0, 0, -1.0]])
    t = cast_balls(g, origins, away)
    assert np.isinf(t).all()
    assert np.isinf(first_hit_bruteforce(pts, 0.05, origins, away)).all()


@pytest.mark.parametrize("max_distance,near_clip", [(0.4, 0.0), (1.0, 0.2), (np.inf, 0.3)])
def test_range_and_near_clip_match_brute_force(max_distance, near_clip):
    rng = np.random.default_rng(8)
    radius = 0.06
    pts = rng.uniform(0, 2, size=(4000, 3))
    origins = mixed_origins(rng, pts, radius, 0.0, 2.0, 10)
    dirs = random_sphere(200, seed=9)
    g = build_ball_grid(pts, radius)
    t = cast_balls(g, origins, dirs, max_distance=max_distance, near_clip=near_clip)
    ref = first_hit_bruteforce(pts, radius, origins, dirs, near_clip=near_clip)
    ref[ref > max_distance] = np.inf
    assert_same_hits(t, ref)


@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(1, 300),
    radius=st.floats(0.005, 0.5),
    seed=st.integers(0, 2**32 - 1),
)
def test_small_clouds_property(n, radius, seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1, 1, size=(n, 3))
    origins = mixed_origins(rng, pts, radius, -1.0, 1.0, 4)
    dirs = random_sphere(64, seed=rng)
    g = build_ball_grid(pts, radius)
    assert_same_hits(cast_balls(g, origins, dirs), first_hit_bruteforce(pts, radius, origins, dirs))
