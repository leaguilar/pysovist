"""Isovists along a path, looking along its tangent."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from pysovist import METRIC_NAMES, Plan, isovist, isovist_field
from pysovist.paths import path_isovists, sample_path

from .geometry import rect

CORRIDOR = Plan(rect(-20, -1, 20, 1))


def test_straight_path_samples_every_spacing_and_looks_ahead():
    s, pts, direction = sample_path([(-15, 0), (15, 0)], 5.0)
    assert s.tolist() == [0, 5, 10, 15, 20, 25, 30]
    assert pts.tolist() == [[x, 0] for x in (-15, -10, -5, 0, 5, 10, 15)]
    assert direction.tolist() == [0.0] * 7


def test_the_end_point_closes_a_path_whose_length_is_not_a_multiple():
    s, pts, _ = sample_path([(0, 0), (10, 0)], 3.0)
    assert s.tolist() == [0, 3, 6, 9, 10]
    assert pts[-1].tolist() == [10, 0]
    s, _, _ = sample_path([(0, 0), (10, 0)], 3.0, endpoint=False)
    assert s.tolist() == [0, 3, 6, 9]


def test_an_exact_multiple_is_not_sampled_twice():
    s, _, _ = sample_path([(0, 0), (1, 0)], 0.1)
    assert len(s) == 11
    assert s[-1] == pytest.approx(1.0)


def test_a_vertex_takes_the_direction_of_the_segment_that_leaves_it():
    s, pts, direction = sample_path([(0, 0), (4, 0), (4, 3)], 1.0)
    assert s.tolist() == list(range(8))
    assert pts.tolist() == [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3]]
    assert direction == pytest.approx([0, 0, 0, 0, math.pi / 2, math.pi / 2, math.pi / 2,
                                       math.pi / 2])


def test_repeated_vertices_and_heights_are_ignored():
    a = sample_path([(0, 0, 1.6), (0, 0, 1.6), (2, 0, 1.6), (2, 2, 1.6)], 0.5)
    b = sample_path([(0, 0), (2, 0), (2, 2)], 0.5)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_invalid_paths_and_spacings_are_refused():
    with pytest.raises(ValueError, match="spacing"):
        sample_path([(0, 0), (1, 0)], 0.0)
    with pytest.raises(ValueError, match="two"):
        sample_path([(1, 1), (1, 1)], 0.5)


def test_isovists_along_a_corridor_see_the_rest_of_it():
    df = path_isovists(CORRIDOR, [(-15, 0), (15, 0)], 5.0, fov=math.pi)
    assert list(df.columns) == ["s", "x", "y", "direction", *METRIC_NAMES, "clearance",
                                "on_wall", "unbounded"]
    x = df["x"].to_numpy()
    assert df["area"].to_numpy() == pytest.approx(2 * (20 - x), rel=1e-12)
    # Two walls of length 20 - x, the far wall, and two field-of-view edges of 1 m.
    assert df["perimeter"].to_numpy() == pytest.approx(2 * (20 - x) + 4, rel=1e-12)
    assert df["r_min"].to_numpy() == pytest.approx(np.ones(7))
    assert not df["on_wall"].any() and not df["unbounded"].any()


def test_walking_the_other_way_sees_the_other_half():
    df = path_isovists(CORRIDOR, [(15, 0), (-15, 0)], 5.0, fov=math.pi)
    assert df["direction"].to_numpy() == pytest.approx(np.full(7, math.pi))
    assert df["area"].to_numpy() == pytest.approx(2 * (df["x"].to_numpy() + 20), rel=1e-12)


def test_turning_a_corner_turns_the_view():
    plan = Plan(rect(0, 0, 10, 10))
    df = path_isovists(plan, [(2, 1), (2, 5), (8, 5)], 2.0, fov=math.pi)
    # Up the first leg the view is the room above y, along the second leg the room right of x.
    expected = [10 * (10 - 1), 10 * (10 - 3), 10 * (10 - 2), 10 * (10 - 4), 10 * (10 - 6),
                10 * (10 - 8)]
    assert df["area"].to_numpy() == pytest.approx(expected, rel=1e-12)


def test_a_full_field_of_view_equals_the_isovist_field():
    plan = Plan(np.concatenate([rect(0, 0, 10, 8), rect(4, 3, 6, 5)]))
    path = [(1, 1), (9, 1), (9, 7), (2, 7)]
    df = path_isovists(plan, path, 0.7, max_distance=6.0)
    field = isovist_field(plan, df[["x", "y"]].to_numpy(), max_distance=6.0)
    pd.testing.assert_frame_equal(df.drop(columns=["s", "direction"]), field)


def test_a_path_on_a_wall_keeps_its_rows_with_flags():
    df = path_isovists(CORRIDOR, [(-25, 0), (-15, 0)], 5.0, fov=math.pi)
    assert df["on_wall"].tolist() == [False, True, False]
    assert np.isnan(df.loc[1, "area"])


# --- property tests ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(st.lists(st.tuples(st.floats(-50, 50), st.floats(-50, 50)), min_size=2, max_size=8),
       st.floats(0.05, 20))
@example(vertices=[(0.0, 0.0), (1e-9, 0.0), (1e-9, 8.3e-10)], spacing=20.0)  # 1.8e-9 m long
def test_samples_lie_on_the_path_at_their_arc_length(vertices, spacing):
    """A path shorter than 1e-9 spacing used to end at its first vertex, without its last."""
    v = np.array(vertices)
    piece = np.hypot(*np.diff(v, axis=0).T)  # hypot does not underflow on tiny steps
    if piece.sum() == 0:
        with pytest.raises(ValueError):
            sample_path(v, spacing)
        return
    s, pts, direction = sample_path(v, spacing)
    total = piece.sum()
    assert s[0] == 0 and s[-1] == pytest.approx(total, rel=1e-12, abs=1e-9)
    gaps = np.diff(s)
    assert gaps[:-1] == pytest.approx(np.full(max(len(gaps) - 1, 0), spacing), rel=1e-9,
                                      abs=1e-9)
    assert (gaps > 0).all() and (gaps <= spacing * (1 + 1e-9)).all()
    keep = piece > 0
    starts, ends = v[:-1][keep], v[1:][keep]
    cum = np.r_[0, np.cumsum(piece[keep])]
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(starts) - 1)
    frac = (s - cum[k]) / piece[keep][k]
    assert pts == pytest.approx(starts[k] + frac[:, None] * (ends[k] - starts[k]), abs=1e-9)
    d = ends[k] - starts[k]
    assert np.cos(direction) == pytest.approx(d[:, 0] / np.hypot(*d.T), abs=1e-12)
    assert np.sin(direction) == pytest.approx(d[:, 1] / np.hypot(*d.T), abs=1e-12)


@settings(max_examples=40, deadline=None)
@given(st.floats(-14, 14), st.floats(-0.9, 0.9), st.floats(0.5, 2 * math.pi))
def test_each_row_is_the_isovist_looking_along_the_tangent(x0, y0, fov):
    df = path_isovists(CORRIDOR, [(x0, y0), (x0 + 5, y0 + 0.05)], 1.3, fov=fov,
                       max_distance=30.0)
    for row in df.itertuples():
        iso = isovist(CORRIDOR, (row.x, row.y), direction=row.direction, fov=fov,
                      max_distance=30.0)
        assert row.area == iso.area
        assert row.drift_angle == iso.metrics["drift_angle"]


def test_path_options_reach_every_observer():
    """Closed forms: with no walls and a range of 2 m every observer sees the disc 4 pi; without
    the end point a 2.5 m path at spacing 1 has three observers; eps = 1e-5 puts a path 1e-6 m
    from a wall on it, and on_wall="raise" refuses it."""
    df = path_isovists(np.empty((0, 2, 2)), [(0, 0), (2.5, 0)], 1.0, max_distance=2.0)
    np.testing.assert_allclose(df["area"], 4 * math.pi, rtol=1e-12)
    assert len(df) == 4
    short = path_isovists(np.empty((0, 2, 2)), [(0, 0), (2.5, 0)], 1.0, max_distance=2.0,
                          endpoint=False)
    assert short["s"].tolist() == [0.0, 1.0, 2.0]
    walls = rect(0, 0, 10, 10)
    near = path_isovists(walls, [(2, 1e-6), (8, 1e-6)], 2.0, eps=1e-5)
    assert near["on_wall"].all()
    with pytest.raises(ValueError, match="from a wall"):
        path_isovists(walls, [(2, 1e-6), (8, 1e-6)], 2.0, eps=1e-5, on_wall="raise")
