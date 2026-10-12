"""Horizontal sections of triangle meshes, as floor plans.

A section of an extruded plan between its floor and its ceiling is the plan
itself, so its isovists equal those of the plan to the 1e-9 of the exact
kernel. A section of a convex hull is the convex polygon cut out by the
hull's half-spaces, computed independently with scipy.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.spatial import ConvexHull, HalfspaceIntersection

from pysovist import Mesh, Plan, isovist

from .geometry import polygon_walls, rect

H = 2.7
PILLAR_ROOM = Plan(np.concatenate([rect(0, 0, 10, 8), rect(4, 3, 6, 5)]))
L_ROOM = Plan(polygon_walls([(0, 0), (10, 0), (10, 4), (4, 4), (4, 9), (0, 9)]))


@pytest.mark.parametrize("z", [0.0, 0.3, 1.6, H - 1e-9])
@pytest.mark.parametrize("plan, eyes", [
    (PILLAR_ROOM, [(1.5, 1.2), (8.7, 6.1), (2.0, 6.5), (5.0, 1.0)]),
    (L_ROOM, [(2.0, 2.0), (8.0, 1.0), (1.0, 8.0), (3.9, 4.1)]),
])
def test_section_of_an_extruded_plan_is_the_plan(plan, eyes, z):
    section = Mesh.from_plan(plan, floor=0.0, ceiling=H, close=None).section(z)
    assert isinstance(section, Plan)
    for eye in eyes:
        assert isovist(section, eye).area == pytest.approx(isovist(plan, eye).area, rel=1e-9)


def test_section_of_a_hull_closed_prism_adds_the_hull_walls():
    plan = Plan([[(0, 0), (10, 0)], [(10, 0), (10, 6)], [(3, 2), (3, 5)]])
    section = Mesh.from_plan(plan, floor=-1.0, ceiling=2.0).section(0.5)
    closed = Plan(np.concatenate([plan.segments, polygon_walls([(0, 0), (10, 0), (10, 6)])]))
    for eye in [(6.0, 2.0), (8.0, 5.0)]:
        assert isovist(section, eye).area == pytest.approx(isovist(closed, eye).area, rel=1e-9)


@pytest.mark.parametrize("z", [-0.5, H, 3.0])
def test_section_outside_the_walls_is_empty(z):
    # A solid between z0 and z1 is cut for z0 <= z < z1: the ceiling height itself is outside.
    assert len(Mesh.from_plan(L_ROOM, floor=0.0, ceiling=H, close=None).section(z)) == 0


def sorted_segments(segments):
    """Segments with their endpoints in lexicographic order, so that direction does not count."""
    s = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    swap = (s[:, 0, 0] > s[:, 1, 0]) | ((s[:, 0, 0] == s[:, 1, 0]) & (s[:, 0, 1] > s[:, 1, 1]))
    s[swap] = s[swap, ::-1]
    return s


def test_section_of_one_tilted_triangle():
    mesh = Mesh([(0, 0, 0), (2, 0, 2), (0, 2, 2)], [(0, 1, 2)])
    np.testing.assert_allclose(sorted_segments(mesh.section(1.0).segments), [[(0, 1), (1, 0)]],
                               atol=1e-15)
    # A vertex on the plane counts as below it. With the other two above, the cut is a single
    # point and no wall remains. An edge on the plane with the third vertex below gives nothing.
    assert len(mesh.section(0.0)) == 0
    assert len(mesh.section(2.0)) == 0
    # With one vertex above and one below, the cut runs from the vertex on the plane.
    split = Mesh([(0, 0, 0), (2, 0, 2), (0, 2, 1)], [(0, 1, 2)]).section(1.0)
    np.testing.assert_allclose(sorted_segments(split.segments), [[(0, 2), (1, 0)]], atol=1e-15)


def test_section_of_an_empty_mesh_is_an_empty_plan():
    assert len(Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=int)).section(1.0)) == 0


@settings(max_examples=25, deadline=None)
@given(n=st.integers(8, 40), seed=st.integers(0, 2**32 - 1), frac=st.floats(0.2, 0.8))
def test_section_of_a_convex_hull_is_its_halfspace_polygon(n, seed, frac):
    pts = np.random.default_rng(seed).uniform(-1, 1, size=(n, 3))
    hull = ConvexHull(pts)
    lo, hi = pts[hull.vertices, 2].min(), pts[hull.vertices, 2].max()
    z = lo + frac * (hi - lo)
    # The cut polygon: the hull's half-spaces a.x + b.y + c.z + d <= 0 restricted to height z.
    eq = hull.equations
    half2d = np.c_[eq[:, :2], eq[:, 2] * z + eq[:, 3]]
    keep = np.linalg.norm(half2d[:, :2], axis=1) > 1e-9
    section = Mesh(pts, hull.simplices).section(z)
    centre = section.segments.reshape(-1, 2).mean(axis=0)
    depth = -(half2d[keep, :2] @ centre + half2d[keep, 2]) / np.linalg.norm(half2d[keep, :2],
                                                                              axis=1)
    assume(depth.min() > 1e-3)
    polygon = HalfspaceIntersection(half2d[keep], centre).intersections
    assert isovist(section, centre).area == pytest.approx(ConvexHull(polygon).volume, rel=1e-9)
