"""A closing polygon of Mesh.from_plan must hold the plan.

Every edge of a closing polygon becomes a wall, the edge from the last vertex
back to the first included. A polygon that leaves walls outside therefore cuts
through the building: a chain of outline walls closed by a chord across the
interior walls off part of every room the chord crosses. from_plan refuses a
polygon with a wall that lies mostly (more than half its length) outside it, and
names the default, which is right for a plan that already includes its outline.
A wall that ends on the polygon, runs along it or reaches past it is accepted,
so a footprint can close a plan whose walls overshoot a corner.
"""

import numpy as np
import pytest

from pysovist import Plan
from pysovist.mesh import Mesh

from .geometry import polygon_walls

FOOTPRINT_L = np.array([(0, 0), (10, 0), (10, 4), (4, 4), (4, 9), (0, 9)], dtype=float)
# interior walls, one of them ending on the outline at (2, 0)
INNER = np.array([[(2.5, 2.0), (2.5, 3.5)], [(6.0, 1.0), (7.5, 1.0)], [(2.0, 0.0), (2.0, 1.5)]],
                 dtype=float)
CLOSED_L = Plan(np.concatenate([INNER, polygon_walls(FOOTPRINT_L)]))
# part of the outline, closed by a chord from (4, 9) back to (10, 0)
CHAIN = np.array([(10, 0), (10, 4), (4, 4), (4, 9)], dtype=float)


def test_a_chord_across_the_building_is_refused():
    with pytest.raises(ValueError, match=r"walls lie mostly outside the closing polygon"):
        Mesh.from_plan(CLOSED_L, close=CHAIN)


def test_the_refusal_names_the_default():
    with pytest.raises(ValueError, match=r"keep the default close='hull'"):
        Mesh.from_plan(CLOSED_L, close=CHAIN)


def test_the_refusal_counts_the_walls_mostly_outside():
    # Of the 9 walls, the four outline walls (10, 0)-(10, 4)-(4, 4)-(4, 9)-(0, 9) and the
    # interior wall (6, 1)-(7.5, 1) lie outside the square, and the outline walls along y = 0
    # and x = 0 run along it for 3 m of 10 m and 9 m. The wall at x = 2.5 lies one third
    # outside, and the wall at x = 2 inside.
    small = np.array([(0, 0), (3, 0), (3, 3), (0, 3)], dtype=float)
    with pytest.raises(ValueError, match=rf"^7 of {len(CLOSED_L)} walls lie mostly outside"):
        Mesh.from_plan(CLOSED_L, close=small)


def test_a_polygon_through_three_corners_is_refused():
    # The walls (10, 0)-(10, 4), (4, 9)-(0, 9) and (0, 9)-(0, 0) touch the triangle only at a
    # corner, and its edges from (0, 0) and from (10, 0) to (4, 9) cut the L.
    with pytest.raises(ValueError, match=rf"^3 of {len(CLOSED_L)} walls lie mostly outside"):
        Mesh.from_plan(CLOSED_L, close=np.array([(0, 0), (10, 0), (4, 9)], dtype=float))


@pytest.mark.parametrize("extra", [[(12.0, 0.0), (12.0, 4.0)], [(10.0, 2.0), (12.0, 2.0)]])
def test_a_wall_mostly_outside_is_refused(extra):
    # a fence beside the building, and a garden wall that starts on its outline
    with pytest.raises(ValueError, match=rf"^1 of {len(CLOSED_L) + 1} walls"):
        Mesh.from_plan(Plan(np.concatenate([CLOSED_L.segments, [extra]])), close=FOOTPRINT_L)


def test_a_wall_that_reaches_past_a_corner_is_accepted():
    # 1.5 m of 4.5 m outside, as an exterior wall drawn past the building's corner
    overshoot = np.array([[(10.0, -1.5), (10.0, 3.0)]])
    Mesh.from_plan(Plan(np.concatenate([CLOSED_L.segments, overshoot])), close=FOOTPRINT_L)


@pytest.mark.parametrize("ring", [FOOTPRINT_L, FOOTPRINT_L[::-1], np.roll(FOOTPRINT_L, 2, axis=0)])
def test_the_outline_of_a_closed_plan_is_accepted(ring):
    # Walls on the polygon's edges and walls ending on them lie on it, not outside.
    Mesh.from_plan(CLOSED_L, close=ring)


def test_a_footprint_around_interior_walls_is_accepted():
    Mesh.from_plan(Plan(INNER), close=FOOTPRINT_L)


def test_coordinates_rounded_to_six_decimals_are_accepted():
    c, s = np.cos(0.7), np.sin(0.7)
    # a footprint of about 40 m whose corners are not round numbers
    turn = np.array([[c, -s], [s, c]]) * 3.3
    ring = FOOTPRINT_L @ turn.T
    walls = np.round(CLOSED_L.segments @ turn.T, 6)
    Mesh.from_plan(Plan(walls), close=ring)


def test_closing_a_closed_plan_along_its_outline_keeps_the_volume():
    pytest.importorskip("open3d")
    from pysovist.volume3d import view_volume

    by_outline = Mesh.from_plan(CLOSED_L, floor=0.0, ceiling=2.7, close=FOOTPRINT_L)
    by_hull = Mesh.from_plan(CLOSED_L, floor=0.0, ceiling=2.7)
    for eye in ((1.5, 1.2, 1.6), (2.0, 6.5, 0.4), (8.0, 2.0, 2.2)):
        a = view_volume(by_outline, eye, n_rays=4096).volume
        b = view_volume(by_hull, eye, n_rays=4096).volume
        assert a == pytest.approx(b, rel=1e-9)
