"""Metric definitions on scenes with known values, and against fine numerical integration."""

import json
import math

import numpy as np
import pytest

from pysovist import Plan, isovist

from .geometry import polygon_walls, rect


def _numeric_radial_moments(iso, n=2_000_001):
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    r = iso.depth(ang)
    mean = r.mean()
    return {
        "r_mean": mean,
        "r_mad": np.abs(r - mean).mean(),
        "r_skew": ((r - mean) ** 3).mean() / r.std() ** 3,
    }


def test_depth_and_radial_sampling_on_the_square():
    iso = isovist(Plan(rect(0, 0, 10, 10)), (5, 5))
    assert iso.depth([0.0, math.pi / 4, math.pi]) == pytest.approx([5, 5 * math.sqrt(2), 5])
    ang, r = iso.radial(4)
    assert ang == pytest.approx([0, math.pi / 2, math.pi, 3 * math.pi / 2])
    assert r == pytest.approx([5, 5, 5, 5])


@pytest.mark.parametrize("origin", [(5, 5), (2, 3), (8.5, 1.5)])
def test_radial_moments_match_fine_integration(origin):
    walls = np.concatenate([rect(0, 0, 10, 10), rect(4, 6, 6, 7)])
    iso = isovist(Plan(walls), origin)
    num = _numeric_radial_moments(iso)
    for key, value in num.items():
        assert iso.metrics[key] == pytest.approx(value, rel=1e-5), key


def test_drift_points_from_the_observer_to_the_centroid():
    iso = isovist(Plan(rect(0, 0, 10, 10)), (2, 3))
    assert iso.metrics["drift"] == pytest.approx(math.hypot(3, 2))
    assert iso.metrics["drift_angle"] == pytest.approx(math.atan2(2, 3))


@pytest.mark.parametrize("origin", [(0, 0), (-12, 0.5)])
def test_elongation_of_a_rectangle_is_its_aspect_ratio(origin):
    iso = isovist(Plan(rect(-20, -1, 20, 1)), origin)
    assert iso.metrics["elongation"] == pytest.approx(20.0)


def test_convex_deficiency():
    assert isovist(Plan(rect(0, 0, 10, 10)), (3, 3)).metrics["convex_deficiency"] == pytest.approx(
        0.0, abs=1e-12)
    walls = polygon_walls([(0, 0), (10, 0), (10, 2), (2, 2), (2, 10), (0, 10)])
    iso = isovist(Plan(walls), (1, 6))
    # Visible region: [0,2]x[0,10] plus triangle (2,2),(2,0),(2.5,0). Hull area 22.5.
    assert iso.metrics["convex_deficiency"] == pytest.approx((22.5 - 20.5) / 22.5)


def test_polygon_ring_has_the_isovist_area():
    walls = np.concatenate([rect(-5, -5, 5, 5), rect(-1, -1, 1, 1)])
    iso = isovist(Plan(walls), (3, 0))
    ring = iso.polygon()
    x, y = ring[:, 0], ring[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert area == pytest.approx(70.0)


def test_plan_from_json(tmp_path):
    path = tmp_path / "walls.json"
    corners = [(0, 0), (10, 0), (10, 10), (0, 10)]
    pairs = zip(corners, corners[1:] + corners[:1], strict=True)
    records = [{"start": [*a, 0], "end": [*b, 0]} for a, b in pairs]
    path.write_text(json.dumps(records))
    plan = Plan.from_json(path)
    assert len(plan) == 4
    assert isovist(plan, (5, 5)).area == pytest.approx(100.0)
