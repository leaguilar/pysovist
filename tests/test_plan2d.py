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
    with pytest.raises(ValueError, match=r'record 0 .*"start": \[x, y\], "end": \[x, y\]'):
        Plan.from_json(path)


@pytest.mark.parametrize("z", [False, True])
def test_from_json_reads_a_list_of_segments_as_the_records_of_the_same_walls(tmp_path, z):
    """Reference: the same walls written as records, the format from_json has always read.
    The two plans have the same segments and vertices, to the last bit."""
    walls = np.concatenate([rect(0, 0, 10, 10), [[(2.5, 2.0), (2.5, 3.5)]]]) + 0.1
    points = [[[*a, 1.5], [*b, 1.5]] if z else [a, b] for a, b in walls.tolist()]
    records = tmp_path / "records.json"
    records.write_text(json.dumps([{"start": a, "end": b} for a, b in points]))
    segments = tmp_path / "segments.json"
    segments.write_text(json.dumps(points))
    ref, plan = Plan.from_json(records), Plan.from_json(segments)
    assert plan.segments.dtype == ref.segments.dtype
    np.testing.assert_array_equal(plan.segments, ref.segments)
    np.testing.assert_array_equal(plan.vertices, ref.vertices)
    np.testing.assert_array_equal(plan.segments, walls)


@pytest.mark.parametrize("content", [
    '{"start": [0, 0], "end": [4, 0]}',  # one record, not a list
    "[[0, 0], [4, 0], [4, 3]]",  # a polygon: points, not segments
    "[[0, 0, 4, 0], [4, 0, 4, 3]]",  # rows of x0, y0, x1, y1
    '[{"start": [0, 0], "end": [4, 0]}, [[4, 0], [4, 3]]]',  # the two formats mixed
    '[{"start": [0, 0]}]',  # a record without its end
    '[{"start": [0], "end": [4, 0]}]',  # a record with one coordinate
    '[{"start": "a", "end": [4, 0]}]',  # a record with a name for a point
    '[[[0, 0], [4, 0]], [[4, 0, 1], [4, 3]]]',  # points of two and three coordinates
    '"walls"',
])
def test_from_json_refuses_other_content_and_names_both_formats(tmp_path, content):
    """These used to fail with a KeyError or a TypeError that named no format."""
    path = tmp_path / "walls.json"
    path.write_text(content)
    with pytest.raises(ValueError, match=r'"start": \[x, y\], "end": \[x, y\]\}.* or a list of '
                                         r"segments \[\[x0, y0\], \[x1, y1\]\]"):
        Plan.from_json(path)


def test_from_json_reads_an_empty_list_as_a_plan_without_walls(tmp_path):
    path = tmp_path / "walls.json"
    path.write_text("[]")
    assert len(Plan.from_json(path)) == 0


def test_flood_grid_over_a_domain_inside_one_room():
    """Closed form: a 4 m x 4 m domain inside a 10 m square holds no wall, so the flood at
    spacing 1 reaches all 16 cells, centred at 2.5 to 5.5 m."""
    centres, cells = Plan(rect(0, 0, 10, 10)).flood_grid(1.0, (3.0, 3.0), domain=(2, 2, 6, 6))
    assert len(centres) == 16
    np.testing.assert_array_equal(np.unique(centres[:, 0]), [2.5, 3.5, 4.5, 5.5])
    np.testing.assert_array_equal(cells[0], [2.0, 2.0, 3.0, 3.0])
