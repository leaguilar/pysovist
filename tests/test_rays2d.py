"""Single 2D rays against wall segments, on scenes with closed-form answers."""

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pysovist import Plan, isovist
from pysovist.rays2d import NO_HIT, RayHits, cast_rays

from .geometry import polygon_walls, rect, regular_ngon

# rect(0, 0, 10, 10) lists the bottom, right, top and left walls, in that order.
BOTTOM, RIGHT, TOP, LEFT = range(4)
SQUARE = Plan(rect(0, 0, 10, 10))


def test_rays_into_a_square_room_hit_the_expected_walls():
    angles = [0.0, math.pi / 2, math.pi, -math.pi / 2, math.pi / 4]
    hits = cast_rays(SQUARE, (2.0, 3.0), angles)
    assert isinstance(hits, RayHits)
    assert hits.segment.tolist() == [[RIGHT, TOP, LEFT, BOTTOM, TOP]]
    assert hits.distance[0] == pytest.approx([8, 7, 2, 3, 7 * math.sqrt(2)], rel=1e-12)
    expected = [(10, 3), (2, 10), (0, 3), (2, 0), (9, 10)]
    assert hits.point[0] == pytest.approx(np.array(expected, float), abs=1e-12)
    assert hits.hit.all()


def test_range_limit_turns_far_walls_into_misses():
    hits = cast_rays(SQUARE, (2.0, 3.0), [0.0, math.pi], max_distance=5.0)
    assert hits.segment.tolist() == [[NO_HIT, LEFT]]
    assert hits.distance[0, 0] == math.inf
    assert np.isnan(hits.point[0, 0]).all()
    assert hits.depth[0] == pytest.approx([5.0, 2.0])


def test_a_wall_exactly_at_the_range_is_hit():
    hits = cast_rays(SQUARE, (2.0, 3.0), 0.0, max_distance=8.0)
    assert hits.segment.tolist() == [[RIGHT]]
    assert hits.distance[0, 0] == 8.0


def test_a_ray_that_meets_no_wall_misses_without_a_range():
    hits = cast_rays([[(3, -1), (3, 1)]], (0.0, 0.0), [0.0, math.pi, math.pi / 2])
    assert hits.segment.tolist() == [[0, NO_HIT, NO_HIT]]
    assert hits.distance[0].tolist() == [3.0, math.inf, math.inf]
    assert hits.depth[0].tolist() == [3.0, math.inf, math.inf]


def test_rays_parallel_to_a_wall_raise_no_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hits = cast_rays([[(0, 1), (10, 1 + 1e-300)]], (0.0, 0.0), [0.0, 1e-310])
    assert (hits.segment == NO_HIT).all()


def test_without_walls_every_ray_misses():
    hits = cast_rays(Plan(np.empty((0, 2, 2))), [(0, 0), (1, 1)], [0.0, 1.0, 2.0])
    assert (hits.segment == NO_HIT).all()
    assert np.isinf(hits.distance).all()
    assert np.isinf(hits.depth).all()
    hits = cast_rays(np.empty((0, 2, 2)), (0, 0), [0.0], max_distance=3.0)
    assert hits.depth.tolist() == [[3.0]]


def test_shapes_shared_and_per_origin_angles():
    origins = [(2, 3), (5, 5), (8, 1)]
    shared = cast_rays(SQUARE, origins, np.linspace(0, 6, 7))
    assert shared.distance.shape == (3, 7)
    assert shared.point.shape == (3, 7, 2)
    assert shared.angles.shape == (3, 7)
    # One direction per origin, as rows of a 2-D array.
    own = cast_rays(SQUARE, origins, np.array([[0.0], [math.pi / 2], [math.pi]]))
    assert own.segment.ravel().tolist() == [RIGHT, TOP, LEFT]
    assert own.distance.ravel() == pytest.approx([8, 5, 8])
    single = cast_rays(SQUARE, (5, 5), 0.0)
    assert single.distance.shape == (1, 1)


def test_per_origin_angles_need_one_row_per_origin():
    with pytest.raises(ValueError, match="row"):
        cast_rays(SQUARE, [(2, 3), (5, 5)], np.zeros((3, 4)))


def test_a_wall_along_the_ray_is_not_hit():
    walls = [[(2, 0), (4, 0)], [(6, -1), (6, 1)]]
    hits = cast_rays(walls, (0.0, 0.0), 0.0)
    assert hits.segment.tolist() == [[1]]
    assert hits.distance[0, 0] == pytest.approx(6.0)


def test_indices_refer_to_the_plan_segments_after_zero_length_walls_are_dropped():
    walls = [[(1, 1), (1, 1)], [(3, -1), (3, 1)]]
    plan = Plan(walls)
    hits = cast_rays(walls, (0.0, 0.0), 0.0)
    assert len(plan) == 1
    assert hits.segment.tolist() == [[0]]


@pytest.mark.parametrize("n", [3, 4, 7, 12])
def test_rays_through_the_corners_of_a_closed_polygon_never_escape(n):
    walls = regular_ngon(n, 3.0, center=(1.0, -2.0))
    angles = 2 * np.pi * np.arange(n) / n
    hits = cast_rays(walls, (1.0, -2.0), angles)
    assert hits.hit.all()
    assert hits.distance[0] == pytest.approx(np.full(n, 3.0), rel=1e-12)


def test_hits_per_wall_count_the_rays_that_end_on_each_wall():
    # 360 rays offset by half a degree miss the corners: each wall of the square gets 90.
    angles = np.radians(np.arange(360) + 0.5)
    hits = cast_rays(SQUARE, (5.0, 5.0), angles)
    counts = np.bincount(hits.segment[hits.hit], minlength=len(SQUARE))
    assert counts.tolist() == [90, 90, 90, 90]


def test_l_shaped_room_hides_the_far_wing():
    room = polygon_walls([(0, 0), (10, 0), (10, 4), (4, 4), (4, 9), (0, 9)])
    # From the end of the lower wing, the ray towards the upper wing ends on the wall
    # (10, 4)-(4, 4) that closes the lower wing, 3 sqrt(2) m away.
    hits = cast_rays(room, (9.0, 1.0), 3 * math.pi / 4)
    assert hits.segment.tolist() == [[2]]
    assert hits.point[0, 0] == pytest.approx([6.0, 4.0])
    assert hits.distance[0, 0] == pytest.approx(3 * math.sqrt(2))


# --- property tests ---------------------------------------------------------------------------

@st.composite
def scene_and_rays(draw):
    size = draw(st.floats(5, 20))
    n = draw(st.integers(0, 10))
    coords = st.floats(0.05 * size, 0.95 * size)
    inner = np.array([[[draw(coords), draw(coords)], [draw(coords), draw(coords)]]
                      for _ in range(n)]).reshape(-1, 2, 2)
    walls = np.concatenate([rect(0, 0, size, size), inner])
    origin = np.array([draw(coords), draw(coords)])
    angles = np.array(draw(st.lists(st.floats(-10, 10), min_size=1, max_size=20)))
    return walls, origin, angles


@settings(max_examples=150, deadline=None)
@given(scene_and_rays())
def test_ray_depth_equals_the_exact_isovist_depth(scene):
    walls, origin, angles = scene
    plan = Plan(walls)
    iso = isovist(plan, origin)
    assume(not iso.flags.on_wall and iso.clearance > 1e-6)
    # At the angle of a wall endpoint the depth jumps, and the ray takes the nearer side.
    v = plan.vertices - origin
    gap = np.abs(np.angle(np.exp(1j * (angles[:, None] - np.arctan2(v[:, 1], v[:, 0])))))
    angles = angles[gap.min(axis=1) > 1e-7]
    assume(len(angles))
    hits = cast_rays(plan, origin, angles)
    assert hits.hit.all()  # the outer square closes the room
    assert hits.distance[0] == pytest.approx(iso.depth(angles), rel=1e-9, abs=1e-9)


@settings(max_examples=150, deadline=None)
@given(scene_and_rays(), st.floats(0.5, 15))
def test_hit_points_lie_on_the_reported_wall_within_range(scene, max_distance):
    walls, origin, angles = scene
    plan = Plan(walls)
    hits = cast_rays(plan, origin, angles, max_distance=max_distance)
    idx = hits.segment[0]
    ok = idx != NO_HIT
    assert (hits.distance[0, ok] <= max_distance).all()
    assert np.isinf(hits.distance[0, ~ok]).all()
    u = np.c_[np.cos(angles), np.sin(angles)]
    assert hits.point[0, ok] == pytest.approx(origin + hits.distance[0, ok, None] * u[ok],
                                              abs=1e-9)
    a, b = plan.segments[idx[ok], 0], plan.segments[idx[ok], 1]
    p = hits.point[0, ok]
    d = b - a
    s = np.clip(np.einsum("ij,ij->i", p - a, d) / np.einsum("ij,ij->i", d, d), 0, 1)
    gap = np.linalg.norm(a + s[:, None] * d - p, axis=1)
    assert (gap < 1e-9 * max(1.0, float(np.abs(walls).max()))).all()
    assert hits.depth[0] == pytest.approx(np.minimum(hits.distance[0], max_distance))


def test_a_ray_from_an_origin_on_a_wall_leaves_through_it():
    """Closed form: from (0, 5) on the west wall of a 10 m square, the ray along +x meets the
    east wall at 10 m and the ray along -x leaves the room."""
    hits = cast_rays(rect(0, 0, 10, 10), (0.0, 5.0), [0.0, np.pi])
    np.testing.assert_array_equal(hits.distance, [[10.0, np.inf]])
    assert hits.segment.tolist() == [[1, NO_HIT]]


def test_a_ray_through_the_free_end_of_a_wall_hits_it():
    """Closed form: a wall from (2, -1) to (2, 1) seen from the origin. The ray at
    atan2(1, 2) passes exactly through its upper end and hits it at sqrt(5)."""
    hits = cast_rays([[(2.0, -1.0), (2.0, 1.0)]], (0.0, 0.0), [math.atan2(1.0, 2.0)])
    assert hits.segment.tolist() == [[0]]
    assert hits.distance[0, 0] == pytest.approx(math.sqrt(5), rel=1e-12)
