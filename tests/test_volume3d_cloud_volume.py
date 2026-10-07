"""View volumes of point clouds: erosion bounds, policies and the per-origin table."""

import math

import numpy as np
import pandas as pd
import pytest

from pysovist.directions import fibonacci_sphere
from pysovist.pointcloud import PointCloud
from pysovist.volume3d import VOLUME_METRIC_NAMES, ViewVolume, view_volume, view_volume_field

# Relative quadrature error of 65536 Fibonacci directions on piecewise smooth depth fields
# (measured on the cube and the prisms in the mesh tests: below 3e-4).
QUAD_REL = 1e-3


def cube_faces(a, s):
    """Points on the six faces of the cube [-a, a]^3, on a square lattice of spacing s."""
    ax = np.linspace(-a, a, int(round(2 * a / s)) + 1)
    u, v = (g.ravel() for g in np.meshgrid(ax, ax))
    w = np.full_like(u, a)
    faces = [np.c_[sgn * w, u, v] for sgn in (-1, 1)]
    faces += [np.c_[u, sgn * w, v] for sgn in (-1, 1)]
    faces += [np.c_[u, v, sgn * w] for sgn in (-1, 1)]
    return np.concatenate(faces)


def floor_cloud(half=3.0, s=0.02, radius=0.03):
    ax = np.arange(-half, half + s / 2, s)
    gx, gy = np.meshgrid(ax, ax)
    return PointCloud(np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)], radius=radius)


# --- erosion bound ----------------------------------------------------------------------------

@pytest.mark.parametrize("radius", [0.05, 0.03])
@pytest.mark.parametrize("origin", [(0.0, 0.0, 0.0), (0.3, -0.2, 0.1)])
def test_cube_of_balls_lies_between_erosion_bounds(radius, origin):
    a, s = 1.0, 0.04
    assert radius > s / math.sqrt(2)
    pc = PointCloud(cube_faces(a, s), radius=radius)
    vv = view_volume(pc, origin, n_rays=65536)
    lower = 8 * (a - radius) ** 3
    upper = 8 * (a - math.sqrt(radius**2 - s**2 / 2)) ** 3
    assert lower * (1 - QUAD_REL) < vv.volume < upper * (1 + QUAD_REL)
    assert not vv.flags.unbounded and not vv.flags.inside_occluder
    assert vv.metrics["escape_fraction"] == 0.0


# --- result object ----------------------------------------------------------------------------

def test_result_fields_and_metric_identities():
    pc = PointCloud(cube_faces(1.0, 0.04), radius=0.05)
    vv = view_volume(pc, (0.1, 0.2, -0.1), n_rays=4096)
    assert isinstance(vv, ViewVolume)
    assert vv.directions.shape == (4096, 3) and vv.distances.shape == (4096,)
    assert vv.weights.sum() == pytest.approx(4 * math.pi, rel=1e-14)
    assert vv.hit.all()
    m = vv.metrics
    assert set(VOLUME_METRIC_NAMES) <= set(m)
    r = vv.distances
    assert vv.volume == pytest.approx((vv.weights * r**3).sum() / 3, rel=1e-13)
    assert m["volume_up"] + m["volume_down"] == pytest.approx(vv.volume, rel=1e-13)
    assert m["r_mean"] == pytest.approx(r.mean(), rel=1e-13)
    assert m["r_std"] == pytest.approx(r.std(), rel=1e-12)
    assert m["r_min"] == r.min() and m["r_max"] == r.max()
    assert m["equivalent_radius"] == pytest.approx((3 * vv.volume / (4 * math.pi)) ** (1 / 3))
    assert vv.clearance == pytest.approx(pc.clearance([(0.1, 0.2, -0.1)])[0])
    np.testing.assert_allclose(vv.points(), vv.origin + vv.directions * r[:, None])


def test_direction_options():
    pc = PointCloud(cube_faces(1.0, 0.1), radius=0.08)
    a = view_volume(pc, (0, 0, 0), n_rays=500, directions="random", seed=3)
    b = view_volume(pc, (0, 0, 0), n_rays=500, directions="random", seed=3)
    assert a.volume == b.volume
    custom = view_volume(pc, (0, 0, 0), directions=fibonacci_sphere(300) * 2.0)
    assert custom.directions.shape == (300, 3)
    np.testing.assert_allclose(np.linalg.norm(custom.directions, axis=1), 1.0)
    with pytest.raises(ValueError, match="ring"):
        view_volume(pc, (0, 0, 0), directions="equiangular")
    with pytest.raises(ValueError):
        view_volume(pc, (0, 0, 0), escape="drop")
    with pytest.raises(ValueError):
        view_volume(pc, (0, 0, 0), inside="raise")


# --- observer inside a ball -------------------------------------------------------------------

def test_inside_a_ball_policies():
    pc = PointCloud(cube_faces(1.0, 0.04), radius=0.05)
    o = (1.0 - 0.02, 0.0, 0.0)  # 2 cm from a face point, inside its ball
    vv = view_volume(pc, o, n_rays=1024)
    assert vv.flags.inside_occluder and vv.clearance < 0
    assert all(np.isnan(vv.metrics[k]) for k in VOLUME_METRIC_NAMES)
    vz = view_volume(pc, o, n_rays=1024, inside="zero")
    assert vz.flags.inside_occluder
    assert all(vz.metrics[k] == 0.0 for k in VOLUME_METRIC_NAMES)
    # Within eps of a ball surface counts as inside.
    near = (1.0 - 0.05 - 1e-4, 0.0, 0.0)
    assert view_volume(pc, near, n_rays=64, eps=1e-3).flags.inside_occluder
    assert not view_volume(pc, near, n_rays=64, eps=1e-6).flags.inside_occluder


def test_near_clip_ignores_close_points_and_flags_the_row():
    pts = cube_faces(1.0, 0.04)
    pc = PointCloud(pts, radius=0.05)
    o = np.array([1.0 - 0.02, 0.0, 0.0])
    vv = view_volume(pc, o, n_rays=2048, near_clip=0.15)
    assert vv.near_clipped and not vv.flags.inside_occluder
    far = pts[np.linalg.norm(pts - o, axis=1) >= 0.15]
    ref = view_volume(PointCloud(far, radius=0.05), o, n_rays=2048)
    assert vv.volume == ref.volume
    assert not view_volume(pc, (0, 0, 0), n_rays=64, near_clip=0.15).near_clipped


def test_radius_override_matches_a_cloud_with_that_radius():
    pts = cube_faces(1.0, 0.04)
    a = view_volume(PointCloud(pts, radius=0.05), (0, 0, 0), n_rays=2048, radius=0.08)
    b = view_volume(PointCloud(pts, radius=0.08), (0, 0, 0), n_rays=2048)
    assert a.volume == b.volume


# --- rays that escape -------------------------------------------------------------------------

def test_escape_policies_over_an_open_floor():
    pc = floor_cloud()
    o = (0.0, 0.0, 1.0)
    clip = view_volume(pc, o, n_rays=4096)
    assert clip.flags.unbounded and math.isinf(clip.volume)
    # Only the solid angle of the 6 m x 6 m floor seen from 1 m above it is hit.
    a, h = 3.0, 1.0 - 0.03
    omega = 4 * math.atan(a * a / (h * math.sqrt(2 * a * a + h * h)))
    assert clip.metrics["escape_fraction"] == pytest.approx(1 - omega / (4 * math.pi), abs=0.01)
    zero = view_volume(pc, o, n_rays=4096, escape="zero")
    assert zero.flags.unbounded
    t = np.where(zero.hit, zero.distances, 0.0)
    assert zero.volume == pytest.approx(4 * math.pi / 3 * np.mean(t**3), rel=1e-13)
    assert zero.metrics["escape_fraction"] == clip.metrics["escape_fraction"]
    nan = view_volume(pc, o, n_rays=4096, escape="nan")
    assert nan.flags.unbounded and math.isnan(nan.volume)


def test_range_limit_over_an_open_floor():
    pc = floor_cloud(s=0.02, radius=0.03)
    R = 2.0
    vv = view_volume(pc, (0.0, 0.0, 1.0), n_rays=65536, max_distance=R)
    assert not vv.flags.unbounded
    assert vv.distances.max() == R
    # The floor's free surface sits between depth sqrt(r^2 - s^2 / 2) and r above z = 0.
    vols = []
    for depth in (0.03, math.sqrt(0.03**2 - 0.02**2 / 2)):
        h = R - (1.0 - depth)
        vols.append(4 * math.pi / 3 * R**3 - math.pi * h**2 * (3 * R - h) / 3)
    assert vols[0] * (1 - QUAD_REL) < vv.volume < vols[1] * (1 + QUAD_REL)
    assert vv.metrics["volume_up"] == pytest.approx(2 * math.pi / 3 * R**3, rel=QUAD_REL)


# --- many origins -----------------------------------------------------------------------------

def test_field_rows_match_single_volumes():
    pc = PointCloud(cube_faces(1.0, 0.04), radius=0.05)
    origins = np.array([[0.0, 0, 0], [0.5, 0.5, -0.5], [0.98, 0, 0], [-0.3, 0.2, 0.7]])
    df = view_volume_field(pc, origins, n_rays=1024, inside="zero")
    assert isinstance(df, pd.DataFrame) and len(df) == 4
    for col in ("x", "y", "z", *VOLUME_METRIC_NAMES, "clearance", "inside_occluder",
                "unbounded", "near_clipped"):
        assert col in df.columns
    for i, o in enumerate(origins):
        vv = view_volume(pc, o, n_rays=1024, inside="zero")
        for k in VOLUME_METRIC_NAMES:
            assert df[k].iloc[i] == pytest.approx(vv.metrics[k], rel=1e-12, abs=0)
        assert df["inside_occluder"].iloc[i] == vv.flags.inside_occluder
    assert df["inside_occluder"].tolist() == [False, False, True, False]
    one = view_volume_field(pc, origins, n_rays=1024, inside="zero", n_jobs=1)
    pd.testing.assert_frame_equal(df, one)
