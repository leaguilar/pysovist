"""The plan: segments, vertices, bounds, grid and the JSON reader, against closed forms."""

import json

import numpy as np
import pytest

from pysovist import Plan

from .geometry import rect


def test_vertices_are_the_endpoints_and_the_crossings():
    """Closed form: the two diagonals of a square of side 2 cross at (1, 1). A wall that ends
    on another (a T-junction) and two collinear walls add no crossing."""
    plan = Plan([[(0, 0), (2, 2)], [(0, 2), (2, 0)]])
    assert sorted(map(tuple, plan.vertices.tolist())) == [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
    tee = Plan([[(0, 0), (0, 10)], [(0, 5), (5, 5)]])
    assert sorted(map(tuple, tee.vertices.tolist())) == [(0, 0), (0, 5), (0, 10), (5, 5)]
    collinear = Plan([[(0, 0), (4, 0)], [(2, 0), (6, 0)]])
    assert len(collinear.vertices) == 4


def test_shared_endpoints_count_once():
    plan = Plan(rect(0, 0, 10, 10))
    assert sorted(map(tuple, plan.vertices.tolist())) == [(0, 0), (0, 10), (10, 0), (10, 10)]
    assert len(plan) == 4
    assert repr(plan) == "Plan(4 segments, 4 vertices)"


def test_zero_length_segments_are_removed_below_eps():
    walls = [[(0, 0), (1, 0)], [(2, 2), (2, 2 + 1e-13)], [(3, 3), (3, 3 + 1e-11)]]
    assert len(Plan(walls)) == 2
    assert len(Plan(walls, eps=1e-10)) == 1
    assert len(Plan(walls, eps=0.0)) == 3


def test_bounds_are_the_extremes_of_the_endpoints():
    plan = Plan([[(-1, 2), (3, 2)], [(0, -4), (0, 7.5)]])
    assert plan.bounds == (-1.0, -4.0, 3.0, 7.5)


def test_grid_starts_half_a_spacing_inside_and_stays_below_the_upper_edge():
    """Closed form: spacing 1 over (0, 0, 3, 2) gives the six cell centres, x running fastest."""
    pts = Plan(rect(0, 0, 10, 10)).grid(1.0, domain=(0, 0, 3, 2))
    np.testing.assert_array_equal(pts, [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5],
                                        [0.5, 1.5], [1.5, 1.5], [2.5, 1.5]])
    full = Plan(rect(0, 0, 10, 10)).grid(2.5)
    assert full.shape == (16, 2)
    assert full.min() == 1.25 and full.max() == 8.75
    assert Plan(rect(0, 0, 10, 10)).grid(25.0).shape == (0, 2)


def test_from_json_reads_records_with_any_keys_and_drops_z(tmp_path):
    path = tmp_path / "walls.json"
    path.write_text(json.dumps([{"a": [0, 0, 1.5], "b": [4, 0, 1.5]},
                                {"a": [4, 0, 0], "b": [4, 3, 0]}]))
    plan = Plan.from_json(path, keys=("a", "b"))
    np.testing.assert_array_equal(plan.segments, [[[0, 0], [4, 0]], [[4, 0], [4, 3]]])
    with pytest.raises(KeyError):
        Plan.from_json(path)
