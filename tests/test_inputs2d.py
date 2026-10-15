"""Inputs outside the documented domain of the 2D functions.

Each input here used to return a number, an empty result or a flag that hides the mistake, or
failed with a message that did not name the argument. Each now raises a ``ValueError`` that
names the argument and says what to pass, or returns a documented result.
"""

import math

import numpy as np
import pandas as pd
import pytest

from pysovist import (
    Mesh,
    Plan,
    cast_rays,
    isovist,
    isovist_field,
    path_isovists,
    sample_path,
    sampled_metrics,
    segment_ray_hits,
    segment_visibility,
    segment_visibility_field,
    vertical_isovist,
)

from .geometry import rect

SQUARE = rect(0, 0, 10, 10)
NAN, INF = float("nan"), float("inf")


# --- plans ------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [[[(0, 0), (NAN, 0)], [(0, 0), (1, 1)]],
                                 [[(0, 0), (INF, 0)], [(0, 0), (1, 1)]]])
def test_plan_refuses_coordinates_that_are_not_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        Plan(bad)


@pytest.mark.parametrize("shape", [(3, 4), (3, 2), (4,), (3, 3, 2), (2, 2, 4)])
def test_plan_refuses_arrays_that_are_not_segments(shape):
    """Rows of x0, y0, x1, y1, or a list of points, used to be cut into wrong segments."""
    with pytest.raises(ValueError, match=r"\(n, 2, 2\)"):
        Plan(np.arange(np.prod(shape), dtype=float).reshape(shape))


def test_plan_takes_one_segment_and_drops_a_third_coordinate():
    np.testing.assert_array_equal(Plan([(0, 0), (1, 0)]).segments, [[[0, 0], [1, 0]]])
    np.testing.assert_array_equal(Plan([[(0, 0, 5), (1, 0, 5)]]).segments, [[[0, 0], [1, 0]]])
    assert len(Plan(np.empty((0, 2, 2)))) == 0 == len(Plan([]))
    # One list of segments per room: the rooms are joined.
    rooms = np.stack([SQUARE, SQUARE + 20.0])
    np.testing.assert_array_equal(Plan(rooms).segments, np.concatenate(rooms))


def test_empty_plan_has_no_bounds_and_needs_a_domain_for_a_grid():
    empty = Plan(np.empty((0, 2, 2)))
    with pytest.raises(ValueError, match="no segments"):
        _ = empty.bounds
    with pytest.raises(ValueError, match="domain"):
        empty.grid(1.0)
    assert empty.grid(1.0, domain=(0, 0, 2, 2)).shape == (4, 2)


@pytest.mark.parametrize("spacing", [0.0, -1.0, NAN, INF])
def test_grid_needs_a_positive_finite_spacing(spacing):
    with pytest.raises(ValueError, match="spacing"):
        Plan(SQUARE).grid(spacing)


@pytest.mark.parametrize("domain", [(10, 10, 0, 0), (0, 0, NAN, 10), (0, 0, 0, 10), (0, 0, 10)])
def test_grid_needs_a_proper_domain(domain):
    with pytest.raises(ValueError, match="domain"):
        Plan(SQUARE).grid(1.0, domain=domain)


def test_flood_grid_refuses_a_seed_that_is_not_finite():
    with pytest.raises(ValueError, match="seed"):
        Plan(SQUARE).flood_grid(1.0, (NAN, 5.0))


def test_flood_grid_needs_a_proper_domain():
    with pytest.raises(ValueError, match="domain"):
        Plan(SQUARE).flood_grid(1.0, (5.0, 5.0), domain=(10, 10, 0, 0))


# --- isovist ----------------------------------------------------------------------------------

@pytest.mark.parametrize("origin", [(NAN, 5.0), (INF, 5.0), (5.0,), (5.0, 5.0, 5.0, 5.0),
                                    [(5.0, 5.0), (6.0, 6.0)]])
def test_isovist_needs_one_finite_origin(origin):
    """A NaN origin used to come back flagged unbounded, and (5,) as the origin (5, 5)."""
    with pytest.raises(ValueError, match="origin"):
        isovist(SQUARE, origin)


def test_isovist_drops_a_third_coordinate_of_the_origin():
    a = isovist(SQUARE, (2.0, 3.0, 1.7)).metrics
    b = isovist(SQUARE, (2.0, 3.0)).metrics
    assert a == b


@pytest.mark.parametrize("r", [0.0, -1.0, NAN])
def test_isovist_needs_a_positive_range(r):
    """A range of -1 used to give an area of pi and a perimeter of -2 pi."""
    with pytest.raises(ValueError, match="max_distance"):
        isovist(SQUARE, (5.0, 5.0), max_distance=r)


@pytest.mark.parametrize("fov", [0.0, -1.0, NAN])
def test_isovist_needs_a_positive_field_of_view(fov):
    with pytest.raises(ValueError, match="fov"):
        isovist(SQUARE, (5.0, 5.0), fov=fov)


BOX = Mesh.from_plan(SQUARE, floor=0.0, ceiling=2.7, close=None)


def _wedges(iso):
    return iso.fov, iso.direction, iso.metrics, iso.theta0, iso.theta1, iso.kind, iso.p, iso.phi


def _visibility(v):
    return v.fov, v.direction, v.visible_length, v.visible_angle, v.intervals


# Every public 2D function that takes a field of view in radians, called from inside the
# square, with a summary of its result that two calls can be compared on.
FOV_CALLS = {
    "isovist": lambda **kw: _wedges(isovist(SQUARE, (4.0, 3.0), **kw)),
    "isovist_field": lambda **kw: isovist_field(SQUARE, [(4.0, 3.0), (7.0, 8.0)], **kw),
    "path_isovists": lambda **kw: path_isovists(SQUARE, [(2, 5), (8, 6)], 1.5, **kw),
    "segment_visibility": lambda **kw: _visibility(segment_visibility(SQUARE, (4.0, 3.0), **kw)),
    "segment_visibility_field": lambda **kw: segment_visibility_field(
        SQUARE, [(4.0, 3.0), (7.0, 8.0)], per_viewpoint=True, chunk=1, **kw),
    "segment_ray_hits": lambda **kw: segment_ray_hits(SQUARE, (4.0, 3.0), 36, **kw),
    "segment_ray_hits_ds": lambda **kw: segment_ray_hits(SQUARE, (4.0, 3.0), 36,
                                                         layout="decodingspaces", **kw),
    "vertical_isovist": lambda **kw: _wedges(vertical_isovist(BOX, (4.0, 3.0, 1.6), 0.3, **kw)),
}


def assert_same(a, b):
    """Equal to the last bit, tables, arrays and NaN included."""
    if isinstance(a, tuple) and a and isinstance(a[0], pd.DataFrame):
        for x, y in zip(a, b, strict=True):
            assert_same(x, y)
    elif isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    else:
        np.testing.assert_equal(a, b)


@pytest.mark.parametrize("name", FOV_CALLS)
@pytest.mark.parametrize("fov", [120.0, 360.0, 2 * math.pi + 1e-9, INF])
def test_a_field_of_view_above_a_full_turn_is_refused_as_degrees(name, fov):
    """fov=120, meant as degrees, used to return the full turn without a message."""
    with pytest.raises(ValueError, match=r"fov must be an angle in radians.*math\.radians"):
        FOV_CALLS[name](fov=fov)


@pytest.mark.parametrize("name", FOV_CALLS)
@pytest.mark.parametrize("fov", [2 * math.pi, math.radians(360), math.tau, 2 * math.pi + 1e-13])
def test_a_full_turn_up_to_rounding_is_the_full_turn(name, fov):
    """Reference: the call with the default field of view, the full turn, to the last bit.
    Closed form for the isovist: the 10 m square has area 100 and perimeter 40."""
    assert_same(FOV_CALLS[name](fov=fov), FOV_CALLS[name]())
    iso = isovist(SQUARE, (4.0, 3.0), fov=fov)
    assert iso.fov == 2 * math.pi
    assert iso.metrics["area"] == pytest.approx(100.0, rel=1e-12)
    assert iso.metrics["perimeter"] == pytest.approx(40.0, rel=1e-12)


THREE = [(4.0, 3.0), (7.0, 8.0), (2.0, 2.0)]  # observers inside the square

# Every public 2D function that takes a viewing direction in radians, with a field of view of
# one radian so that the direction matters.
DIRECTION_CALLS = {
    "isovist": lambda d: _wedges(isovist(SQUARE, (4.0, 3.0), fov=1.0, direction=d)),
    "isovist_field": lambda d: isovist_field(SQUARE, THREE, fov=1.0, direction=d),
    "segment_visibility": lambda d: _visibility(segment_visibility(SQUARE, (4.0, 3.0), fov=1.0,
                                                                   direction=d)),
    "segment_visibility_field": lambda d: segment_visibility_field(
        SQUARE, THREE, fov=1.0, direction=d, per_viewpoint=True, chunk=1),
    "segment_ray_hits": lambda d: segment_ray_hits(SQUARE, (4.0, 3.0), 36, fov=1.0, direction=d),
}


@pytest.mark.parametrize("name", DIRECTION_CALLS)
@pytest.mark.parametrize("vector", [(3.0, 4.0), (-1.0, 0.0), (0.0, -2.0), (-1e-3, -5.0),
                                    (-7.0, 1e-300)])
def test_a_direction_vector_gives_the_view_of_its_angle(name, vector):
    """Reference: the same call with the angle atan2(dy, dx), to the last bit, for a tuple, a
    list and an array."""
    ref = DIRECTION_CALLS[name](math.atan2(vector[1], vector[0]))
    for v in (vector, list(vector), np.array(vector)):
        assert_same(DIRECTION_CALLS[name](v), ref)


@pytest.mark.parametrize("name", [*DIRECTION_CALLS, "segment_ray_hits_ds"])
@pytest.mark.parametrize("direction", [45.0, -90.0, 360.0, 2 * math.pi + 1e-9,
                                       -2 * math.pi - 1e-9])
def test_a_direction_beyond_a_full_turn_is_refused_as_degrees(name, direction):
    """direction=45, meant as degrees, used to give the view along 45 radians."""
    call = DIRECTION_CALLS.get(name) or (lambda d: segment_ray_hits(
        SQUARE, (4.0, 3.0), 36, fov=1.0, direction=d, layout="decodingspaces"))
    with pytest.raises(ValueError, match=r"direction must be an angle in radians, "
                                         r"counter-clockwise from \+x, between -2 pi and 2 pi, "
                                         r"or a vector \(dx, dy\).*math\.radians"):
        call(direction)


@pytest.mark.parametrize("name", DIRECTION_CALLS)
@pytest.mark.parametrize("turns", [1, -1])
def test_a_direction_of_a_full_turn_is_taken(name, turns):
    """Closed form: a direction one full turn away looks the same way, so the visible areas,
    lengths and hits agree with those of direction 0 (to rounding, as the angles differ)."""
    for d in (turns * 2 * math.pi, turns * (2 * math.pi + 1e-13)):
        DIRECTION_CALLS[name](d)
    iso = isovist(SQUARE, (4.0, 3.0), fov=1.0, direction=turns * 2 * math.pi)
    ref = isovist(SQUARE, (4.0, 3.0), fov=1.0, direction=0.0)
    assert iso.area == pytest.approx(ref.area, rel=1e-12)


@pytest.mark.parametrize("name", [*DIRECTION_CALLS, "segment_ray_hits_ds"])
@pytest.mark.parametrize("vector", [(0.0, 0.0), (NAN, 1.0), (1.0, INF)])
def test_a_direction_vector_without_a_direction_is_refused(name, vector):
    """atan2(0, 0) is 0, so the zero vector used to look along +x in the even layout."""
    call = DIRECTION_CALLS.get(name) or (lambda d: segment_ray_hits(
        SQUARE, (4.0, 3.0), 36, fov=1.0, direction=d, layout="decodingspaces"))
    with pytest.raises(ValueError, match=r"direction"):
        call(vector)


@pytest.mark.parametrize("name", ["isovist", "segment_visibility", "segment_ray_hits"])
@pytest.mark.parametrize("direction", [np.array([0.1, 0.2, 0.3]).reshape(3, 1), np.zeros((2, 2)),
                                       np.array([0.5]), [[1.0, 0.0]]])
def test_a_direction_of_another_shape_is_refused(name, direction):
    """These used to fail with a TypeError that named no argument."""
    with pytest.raises(ValueError, match=r"direction must be one angle in radians or one vector "
                                         r"\(dx, dy\), got an array of shape"):
        DIRECTION_CALLS[name](direction)


@pytest.mark.parametrize("field, single", [(isovist_field, "isovist"),
                                           (segment_visibility_field, "segment_visibility")])
@pytest.mark.parametrize("direction", [np.array([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3],
                                       np.array([0.5]), np.zeros((3, 2))])
def test_a_direction_per_observer_is_refused_with_the_loop_to_write(field, single, direction):
    """isovist_field with one angle per observer used to fail with "TypeError: only
    0-dimensional arrays can be converted to Python scalars"."""
    with pytest.raises(ValueError, match=r"direction is one angle in radians or one vector "
                                         r"\(dx, dy\) for every observer, got an array of "
                                         rf"shape .*call {single} for each observer"):
        field(SQUARE, THREE, fov=1.0, direction=direction)


def test_vertical_isovist_refuses_an_elevation_in_degrees():
    """elevation=30, meant as degrees, used to centre the view on 30 radians."""
    with pytest.raises(ValueError, match=r"elevation must be an angle in radians.*"
                                         r"math\.radians\(30\.0\)"):
        vertical_isovist(BOX, (4.0, 3.0, 1.6), 0.3, elevation=30.0, fov=1.0)


@pytest.mark.parametrize("direction", [NAN, INF])
def test_isovist_needs_a_finite_direction(direction):
    """A NaN direction used to come back flagged unbounded."""
    with pytest.raises(ValueError, match="direction"):
        isovist(SQUARE, (5.0, 5.0), fov=1.0, direction=direction)


@pytest.mark.parametrize("eps", [0.0, -1.0, NAN])
def test_isovist_needs_a_positive_eps(eps):
    """A negative eps used to let an observer on a wall through, flagged unbounded."""
    with pytest.raises(ValueError, match="eps"):
        isovist(SQUARE, (0.0, 5.0), eps=eps)


def test_isovist_field_takes_one_point_and_refuses_bad_parallel_settings():
    one = isovist_field(SQUARE, (5.0, 5.0))
    assert len(one) == 1
    assert one["area"].iloc[0] == pytest.approx(100.0, rel=1e-12)  # closed form
    with pytest.raises(ValueError, match="chunk"):
        isovist_field(SQUARE, [(5.0, 5.0)] * 3, chunk=0, n_jobs=2)
    for n_jobs in (0, -2):
        with pytest.raises(ValueError, match="n_jobs"):
            isovist_field(SQUARE, [(5.0, 5.0)], n_jobs=n_jobs)


# --- sampled metrics ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 2, -3, 2.5, NAN])
def test_sampled_metrics_need_a_polygon(n):
    """A full turn needs three rays for a polygon. Two rays gave an area of 0 and 2.5 rays
    were spaced 2 pi / 2.5 apart."""
    iso = isovist(SQUARE, (5.0, 5.0))
    with pytest.raises(ValueError, match="n_rays"):
        sampled_metrics(iso, n)


def test_sampled_metrics_of_a_partial_view_need_two_rays():
    iso = isovist(SQUARE, (5.0, 5.0), fov=math.pi / 2, direction=0.0)
    with pytest.raises(ValueError, match="n_rays"):
        sampled_metrics(iso, 1)
    # Closed form: two rays at -45 and +45 degrees reach the corners (10, 0) and (10, 10), and
    # the triangle with the observer has area 25.
    assert sampled_metrics(iso, 2)["area"] == pytest.approx(25.0, rel=1e-12)
    assert sampled_metrics(iso, 3.0)["area"] == pytest.approx(25.0, rel=1e-12)


def test_sampled_metrics_of_a_flagged_isovist_follow_its_flags():
    on_wall = sampled_metrics(isovist(SQUARE, (0.0, 5.0)), 36)
    assert all(math.isnan(v) for v in on_wall.values())
    unbounded = sampled_metrics(isovist(SQUARE[:3], (5.0, 5.0)), 36)
    assert unbounded["area"] == unbounded["perimeter"] == INF
    assert math.isnan(unbounded["r_mean"])


# --- rays and obstacle visibility ---------------------------------------------------------------

@pytest.mark.parametrize("origins, angles, r, name", [
    ((NAN, 5.0), [0.0, 1.0], INF, "origins"), ((5.0,), [0.0, 1.0], INF, "origins"),
    ((5.0, 5.0), [0.0, NAN], INF, "angles"), ((5.0, 5.0), [0.0, 1.0], NAN, "max_distance"),
    ((5.0, 5.0), [0.0, 1.0], 0.0, "max_distance"), ((5.0, 5.0), [0.0, 1.0], -1.0, "max_distance")])
def test_cast_rays_refuses_rays_without_a_start_direction_or_range(origins, angles, r, name):
    """A NaN origin, angle or range used to give misses everywhere."""
    with pytest.raises(ValueError, match=name):
        cast_rays(SQUARE, origins, angles, max_distance=r)


@pytest.mark.parametrize("kw, name", [({"origin": (NAN, 5.0)}, "origin"),
                                      ({"max_distance": 0.0}, "max_distance"),
                                      ({"max_distance": NAN}, "max_distance"),
                                      ({"fov": 0.0}, "fov"), ({"fov": -1.0}, "fov"),
                                      ({"fov": NAN}, "fov"),
                                      ({"direction": NAN, "fov": 1.0}, "direction"),
                                      ({"eps": -1.0}, "eps")])
def test_segment_visibility_refuses_a_view_outside_the_domain(kw, name):
    """These used to return zero visible length for every wall."""
    args = {"origin": (5.0, 5.0)} | kw
    origin = args.pop("origin")
    with pytest.raises(ValueError, match=name):
        segment_visibility(SQUARE, origin, **args)


@pytest.mark.parametrize("kw, name", [({"n_rays": 0}, "n_rays"), ({"n_rays": -1}, "n_rays"),
                                      ({"n_rays": 2.5}, "n_rays"),
                                      ({"origin": (NAN, 5.0)}, "origin"),
                                      ({"fov": 0.0}, "fov"), ({"fov": NAN}, "fov"),
                                      ({"max_distance": NAN}, "max_distance"),
                                      ({"direction": NAN}, "direction")])
def test_segment_ray_hits_refuses_rays_outside_the_domain(kw, name):
    """These used to return counts: zeros for no rays, ten hits on one wall for a field of view
    of zero."""
    args = {"origin": (5.0, 5.0), "n_rays": 10} | kw
    origin, n = args.pop("origin"), args.pop("n_rays")
    with pytest.raises(ValueError, match=name):
        segment_ray_hits(SQUARE, origin, n, **args)


def test_segment_ray_hits_decodingspaces_layout_keeps_the_plugin_rule_for_any_count():
    """The plugin's rule: a Precision of 10 or less casts one ray per degree (361 rays from the
    default direction in a closed room, measured for Precision 360)."""
    hits = segment_ray_hits(SQUARE, (2.1, 3.3), 0, direction=math.pi / 2,
                            layout="decodingspaces")
    assert hits.sum() == 361


def test_segment_visibility_field_refuses_a_chunk_of_zero():
    """A chunk of zero used to drop every viewpoint and return NaN means."""
    with pytest.raises(ValueError, match="chunk"):
        segment_visibility_field(SQUARE, [(5.0, 5.0)], chunk=0)
    for n_jobs in (0, -2):
        with pytest.raises(ValueError, match="n_jobs"):
            segment_visibility_field(SQUARE, [(5.0, 5.0)], n_jobs=n_jobs)


def test_segment_visibility_field_takes_one_point():
    table = segment_visibility_field(SQUARE, (5.0, 5.0))
    np.testing.assert_allclose(table["visible_length_sum"], 10.0, rtol=1e-12)  # closed form


# --- paths ------------------------------------------------------------------------------------

@pytest.mark.parametrize("spacing", [INF, NAN, 0.0, -1.0])
def test_sample_path_needs_a_positive_finite_spacing(spacing):
    """An infinite spacing used to give the arc length NaN."""
    with pytest.raises(ValueError, match="spacing"):
        sample_path([(0, 0), (1, 0)], spacing)


def test_sample_path_refuses_vertices_that_are_not_finite():
    with pytest.raises(ValueError, match="finite"):
        sample_path([(0, 0), (NAN, 0), (2, 0)], 0.5)
    with pytest.raises(ValueError, match="finite"):
        path_isovists(SQUARE, [(1, 1), (2, INF)], 0.5)


@pytest.mark.parametrize("eps", [NAN, -1.0, INF])
def test_plan_needs_a_finite_non_negative_eps(eps):
    """A NaN eps used to drop every segment, so the room came back unbounded."""
    with pytest.raises(ValueError, match="eps"):
        Plan(SQUARE, eps=eps)


def test_isovist_field_checks_its_options_without_observers():
    """An unknown option or a bad value used to pass when there were no observers."""
    with pytest.raises(TypeError, match="foo"):
        isovist_field(SQUARE, np.empty((0, 2)), foo=1)
    with pytest.raises(ValueError, match="on_wall"):
        isovist_field(SQUARE, np.empty((0, 2)), on_wall="bad")
    with pytest.raises(ValueError, match="max_distance"):
        isovist_field(SQUARE, np.empty((0, 2)), max_distance=-1.0)


def test_observer_on_a_wall_is_named_in_plain_numbers():
    with pytest.raises(ValueError, match=r"observer \(0\.0, 5\.0\) is 0 m from a wall"):
        isovist(SQUARE, np.array([0.0, 5.0]), on_wall="raise")
    with pytest.raises(ValueError, match=r"observer \(0\.0, 5\.0\) is 0 m from a wall"):
        segment_visibility(SQUARE, np.array([0.0, 5.0]), on_wall="raise")


def test_isovist_refuses_an_unknown_on_wall_policy_and_takes_a_large_eps():
    with pytest.raises(ValueError, match="on_wall"):
        isovist(SQUARE, (5.0, 5.0), on_wall="zero")
    assert isovist(SQUARE, (5.0, 5.0), eps=6.0).flags.on_wall  # the walls are 5 m away


@pytest.mark.parametrize("direction", [(0.0, 0.0), (NAN, 1.0), (1.0,)])
def test_segment_ray_hits_refuses_a_direction_vector_without_a_direction(direction):
    with pytest.raises(ValueError, match="direction"):
        segment_ray_hits(SQUARE, (5.0, 5.0), 10, direction=direction, layout="decodingspaces")


def test_segment_ray_hits_takes_a_direction_vector_in_the_even_layout():
    """Closed form: four even rays from the centre along (0, 1) and its quarter turns hit the
    four walls once each, as with direction = pi / 2."""
    hits = segment_ray_hits(SQUARE, (5.0, 5.0), 4, direction=(0.0, 2.0))
    assert hits.tolist() == [1, 1, 1, 1]
    np.testing.assert_array_equal(hits, segment_ray_hits(SQUARE, (5.0, 5.0), 4,
                                                         direction=math.pi / 2))
