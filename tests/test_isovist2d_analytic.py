"""Exact 2D isovists on scenes with closed-form answers."""

import math

import numpy as np
import pytest

from pysovist import Plan, isovist

from .geometry import polygon_walls, rect, regular_ngon

REL = 1e-9


def approx(x):
    return pytest.approx(x, rel=REL, abs=1e-9)


# --- convex rooms -----------------------------------------------------------------------------

def test_square_centre_area_perimeter_and_radials():
    iso = isovist(Plan(rect(0, 0, 10, 10)), (5, 5))
    assert iso.area == approx(100.0)
    assert iso.perimeter == approx(40.0)
    m = iso.metrics
    assert m["occlusivity"] == approx(0.0)
    assert m["r_min"] == approx(5.0)
    assert m["r_max"] == approx(5 * math.sqrt(2))
    r_mean = 20 / math.pi * math.log(1 + math.sqrt(2))
    assert m["r_mean"] == approx(r_mean)
    assert m["r_var"] == approx(100 / math.pi - r_mean**2)
    assert m["compactness"] == approx(math.pi / 4)
    assert m["drift"] == approx(0.0)


@pytest.mark.parametrize("origin", [(2, 3), (0.5, 9.5), (9.9, 0.1), (5, 1e-6)])
def test_square_any_interior_point_sees_the_whole_room(origin):
    iso = isovist(Plan(rect(0, 0, 10, 10)), origin)
    assert iso.area == approx(100.0)
    assert iso.perimeter == approx(40.0)
    assert iso.metrics["drift"] == approx(math.hypot(origin[0] - 5, origin[1] - 5))


@pytest.mark.parametrize("origin", [(0, 0), (-19.5, 0.9), (7.3, -0.4)])
def test_hallway_unlimited_range(origin):
    iso = isovist(Plan(rect(-20, -1, 20, 1)), origin)
    assert iso.area == approx(80.0)
    assert iso.perimeter == approx(84.0)


@pytest.mark.parametrize("n,rho", [(3, 2.0), (6, 1.5), (17, 4.0)])
def test_regular_polygon(n, rho):
    iso = isovist(Plan(regular_ngon(n, rho)), (0.1 * rho, -0.05 * rho))
    assert iso.area == approx(n / 2 * rho**2 * math.sin(2 * math.pi / n))
    assert iso.perimeter == approx(2 * n * rho * math.sin(math.pi / n))


# --- range limit ------------------------------------------------------------------------------

def test_hallway_range_limit_keeps_walls_without_endpoints_in_range():
    # Both long walls cross the range disc, and none of their endpoints lies within it.
    iso = isovist(Plan(rect(-20, -1, 20, 1)), (0, 0), max_distance=10)
    assert iso.area == approx(2 * (math.sqrt(99) + 100 * math.asin(0.1)))
    assert iso.perimeter == approx(4 * math.sqrt(99) + 40 * math.asin(0.1))
    assert iso.metrics["r_max"] == approx(10.0)


def test_closed_room_range_beyond_diameter_changes_nothing():
    plan = Plan(rect(0, 0, 10, 10))
    a = isovist(plan, (3, 4))
    b = isovist(plan, (3, 4), max_distance=1e3)
    assert b.area == approx(a.area)
    assert b.perimeter == approx(a.perimeter)


def test_empty_scene_with_range_is_a_disc():
    iso = isovist(Plan(np.empty((0, 2, 2))), (1, 2), max_distance=3)
    assert iso.area == approx(math.pi * 9)
    assert iso.perimeter == approx(2 * math.pi * 3)


# --- occlusion --------------------------------------------------------------------------------

def test_l_shape_occluding_edge():
    walls = polygon_walls([(0, 0), (10, 0), (10, 2), (2, 2), (2, 10), (0, 10)])
    iso = isovist(Plan(walls), (1, 6))
    assert iso.area == approx(20.5)
    assert iso.metrics["occlusivity"] == approx(math.sqrt(4.25))
    assert iso.perimeter == approx(22.5 + math.sqrt(4.25))


def test_room_with_pillar():
    walls = np.concatenate([rect(-5, -5, 5, 5), rect(-1, -1, 1, 1)])
    iso = isovist(Plan(walls), (3, 0))
    assert iso.area == approx(70.0)
    assert iso.metrics["occlusivity"] == approx(2 * math.sqrt(45))
    assert iso.perimeter == approx(34 + 2 * math.sqrt(45))


def test_open_room_with_range():
    # Square without its top wall: the opening shows a quarter disc of radius 40.
    walls = np.array([[(0, 10), (0, 0)], [(0, 0), (10, 0)], [(10, 0), (10, 10)]], dtype=float)
    iso = isovist(Plan(walls), (5, 5), max_distance=40)
    assert iso.area == approx(75 + 400 * math.pi)
    assert iso.metrics["occlusivity"] == approx(2 * (40 - math.sqrt(50)))
    assert not iso.flags.unbounded


def test_open_room_without_range_is_unbounded():
    walls = np.array([[(0, 10), (0, 0)], [(0, 0), (10, 0)], [(10, 0), (10, 10)]], dtype=float)
    iso = isovist(Plan(walls), (5, 5))
    assert iso.flags.unbounded
    assert math.isinf(iso.area)


# --- field of view ----------------------------------------------------------------------------

def test_quarter_field_of_view():
    iso = isovist(Plan(rect(0, 0, 10, 10)), (5, 5), fov=math.pi / 2, direction=0.0)
    assert iso.area == approx(25.0)
    # Boundary: the wall piece of length 10 plus two radial edges of length 5*sqrt(2).
    assert iso.perimeter == approx(10 + 10 * math.sqrt(2))


def test_field_of_view_wedges_add_up():
    plan = Plan(np.concatenate([rect(-5, -5, 5, 5), rect(-1, -1, 1, 1)]))
    whole = isovist(plan, (3, 0.5))
    parts = [isovist(plan, (3, 0.5), fov=math.pi / 3, direction=k * math.pi / 3 + math.pi / 6)
             for k in range(6)]
    assert sum(p.area for p in parts) == approx(whole.area)


# --- degenerate input -------------------------------------------------------------------------

def test_observer_on_a_wall_is_flagged():
    iso = isovist(Plan(rect(0, 0, 10, 10)), (5, 0))
    assert iso.flags.on_wall
    assert math.isnan(iso.area)


def test_observer_on_a_wall_can_raise():
    with pytest.raises(ValueError):
        isovist(Plan(rect(0, 0, 10, 10)), (0, 0), on_wall="raise")


def test_subdivided_and_duplicated_walls_give_the_same_isovist():
    base = rect(0, 0, 10, 10)
    pieces = []
    for a, b in base:
        t = np.linspace(0, 1, 5)[:, None]
        pts = a + t * (b - a)
        pieces.extend(np.stack([pts[:-1], pts[1:]], axis=1))
    walls = np.concatenate([np.array(pieces), base[:2]])
    iso = isovist(Plan(walls), (2, 7))
    assert iso.area == approx(100.0)
    assert iso.perimeter == approx(40.0)
    assert iso.metrics["occlusivity"] == approx(0.0)


def test_crossing_walls():
    # An X of two walls inside a room: from (5, 1) the lower triangle of the X is hidden.
    walls = np.concatenate([rect(0, 0, 10, 10),
                            np.array([[(3, 3), (7, 7)], [(3, 7), (7, 3)]], dtype=float)])
    iso = isovist(Plan(walls), (5, 1))
    # Visible: the room minus the region behind the X. The X's arms meet at (5, 5).
    # Shadow of the arm pair, bounded by the rays through (3, 3) and (7, 3) to the walls.
    shadow = _shadow_area_crossing_walls()
    assert iso.area == approx(100.0 - shadow)


def _shadow_area_crossing_walls():
    # Rays from (5, 1) through (3, 3) and (7, 3) have slopes -1 and +1 and hit x = 0 and x = 10
    # at y = 6. Region hidden behind the V formed by (3, 3)-(5, 5)-(7, 3):
    # polygon (3,3) (0,6) (0,10) (10,10) (10,6) (7,3) (5,5).
    poly = np.array([(3, 3), (0, 6), (0, 10), (10, 10), (10, 6), (7, 3), (5, 5)], dtype=float)
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_wall_seen_edge_on_is_well_conditioned():
    # A wall almost aligned with the line of sight gives a wedge of 5e-6 rad in which the
    # depth formula works near cos = 0. Scaling the scene must scale the result exactly.
    seg = np.concatenate([rect(0, 0, 5, 5), np.array([[[0.99999, 2.0], [1.0, 1.0]]])])
    o = np.array([1.0, 4.0])
    base = isovist(Plan(seg), o)
    for k in (3.0, 7.0, 0.01):
        scaled = isovist(Plan(seg * k), o * k)
        assert scaled.perimeter == approx(base.perimeter * k)
        assert scaled.metrics["occlusivity"] == approx(base.metrics["occlusivity"] * k)
        assert scaled.area == approx(base.area * k * k)
