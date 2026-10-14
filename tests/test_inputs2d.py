"""Inputs outside the documented domain of the 2D functions.

Each input here used to return a number, an empty result or a flag that hides the mistake, or
failed with a message that did not name the argument. Each now raises a ``ValueError`` that
names the argument and says what to pass, or returns a documented result.
"""

import math

import numpy as np
import pytest

from pysovist import (
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


def test_isovist_field_of_view_above_a_full_turn_is_a_full_turn():
    """Closed form: the 10 m square from its centre has area 100 and perimeter 40."""
    iso = isovist(SQUARE, (5.0, 5.0), fov=7.0)
    assert iso.fov == 2 * math.pi == isovist(SQUARE, (5.0, 5.0), fov=INF).fov
    assert iso.metrics["area"] == pytest.approx(100.0, rel=1e-12)
    assert iso.metrics["perimeter"] == pytest.approx(40.0, rel=1e-12)


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
