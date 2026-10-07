"""Batch evaluation over many observers, and the ray-sampled mode of other tools."""

import math

import numpy as np
import pytest

from pysovist import METRIC_NAMES, Plan, isovist, isovist_field, sampled_metrics

from .geometry import rect


def test_field_returns_one_row_per_origin_with_all_metrics():
    plan = Plan(rect(0, 0, 10, 10))
    origins = np.array([[5, 5], [2, 3], [5, 0], [9, 9]])
    df = isovist_field(plan, origins)
    assert len(df) == 4
    assert {"x", "y", "clearance", "on_wall", "unbounded", *METRIC_NAMES} <= set(df.columns)
    assert df.loc[[0, 1, 3], "area"].to_numpy() == pytest.approx([100, 100, 100])
    assert df.loc[2, "on_wall"]
    assert math.isnan(df.loc[2, "area"])


def test_field_in_parallel_equals_serial():
    plan = Plan(np.concatenate([rect(-5, -5, 5, 5), rect(-1, -1, 1, 1)]))
    rng = np.random.default_rng(0)
    origins = rng.uniform(-4.9, 4.9, size=(200, 2))
    serial = isovist_field(plan, origins, n_jobs=1, max_distance=6)
    parallel = isovist_field(plan, origins, n_jobs=2, max_distance=6)
    np.testing.assert_array_equal(serial.to_numpy(), parallel.to_numpy())


def test_sampled_square_with_four_rays_is_a_diamond():
    iso = isovist(Plan(rect(0, 0, 10, 10)), (5, 5))
    m = sampled_metrics(iso, n_rays=4)
    assert m["area"] == pytest.approx(50.0)
    assert m["perimeter"] == pytest.approx(20 * math.sqrt(2))
    assert m["r_mean"] == pytest.approx(5.0)


def _error_slope(scene, origin, ns, n_offsets=32):
    # The error at one ray count depends on where each corner falls between two rays, so
    # average over evenly spread offsets of the first ray before fitting the rate.
    iso = isovist(Plan(scene), origin)
    err = []
    for n in ns:
        offsets = 2 * math.pi / n * (np.arange(n_offsets) + 0.5) / n_offsets
        err.append(np.mean([abs(sampled_metrics(iso, n_rays=n, offset=o)["area"] - iso.area)
                            for o in offsets]))
    return np.polyfit(np.log(ns), np.log(err), 1)[0]


def test_sampling_error_falls_as_n_squared_in_a_convex_room():
    slope = _error_slope(rect(0, 0, 10, 10), (3.1, 4.3), [64, 128, 256, 512, 1024])
    assert slope == pytest.approx(-2.0, abs=0.2)


def test_sampling_error_falls_as_n_with_occluding_edges():
    scene = np.concatenate([rect(-5, -5, 5, 5), rect(-1, -1, 1, 1)])
    slope = _error_slope(scene, (3.05, 0.37), [101, 211, 401, 809, 1601])
    assert slope == pytest.approx(-1.0, abs=0.25)
