"""A prism room's view volume is its ceiling height times the exact 2D isovist area.

Inside a prism (vertical walls between a horizontal floor and ceiling of
height H), the segment from the eye to any point between floor and ceiling
crosses a wall exactly when its plan projection does. So every eye sees
V = H * A, whatever its height, with A the exact isovist area of the plan.
The 3D side is a 65536-direction Fibonacci estimate. Over the 24 eyes below
its residual is at most 2.4e-4 and falls to 6e-5 at 262144 directions,
so a relative tolerance of 1e-3 bounds the quadrature error with margin.
"""

import numpy as np
import pytest

pytest.importorskip("open3d")

from pysovist import Plan, isovist  # noqa: E402
from pysovist.mesh import Mesh  # noqa: E402
from pysovist.volume3d import view_volume  # noqa: E402

from .geometry import polygon_walls, rect  # noqa: E402

H = 2.7
N = 65536
REL = 1e-3
HEIGHTS = (0.3, 1.6, 2.4)

PILLAR_ROOM = Plan(np.concatenate([rect(0, 0, 10, 8), rect(4, 3, 6, 5)]))
L_ROOM = Plan(polygon_walls([(0, 0), (10, 0), (10, 4), (4, 4), (4, 9), (0, 9)]))


def residual(plan, eye_xy, z, n=N):
    mesh = Mesh.from_plan(plan, floor=0.0, ceiling=H)
    vv = view_volume(mesh, (*eye_xy, z), n_rays=n)
    assert vv.metrics["escape_fraction"] == 0.0
    return vv.volume / (H * isovist(plan, eye_xy).area) - 1


@pytest.mark.parametrize("z", HEIGHTS)
@pytest.mark.parametrize("eye", [(1.5, 1.2), (8.7, 6.1), (2.0, 6.5), (5.0, 1.0)])
def test_room_with_pillar(eye, z):
    assert abs(residual(PILLAR_ROOM, eye, z)) < REL


@pytest.mark.parametrize("z", HEIGHTS)
@pytest.mark.parametrize("eye", [(2.0, 2.0), (8.0, 1.0), (1.0, 8.0), (3.9, 4.1)])
def test_l_shaped_room(eye, z):
    assert abs(residual(L_ROOM, eye, z)) < REL


def test_residual_is_quadrature_error():
    eyes = [(8.0, 1.0), (1.0, 8.0), (3.9, 4.1)]
    worst = [max(abs(residual(L_ROOM, e, 1.6, n)) for e in eyes) for n in (4096, N, 4 * N)]
    assert worst[0] > worst[1] > worst[2]
    assert worst[2] < REL / 10


def test_extruded_plan_is_a_closed_prism():
    mesh = Mesh.from_plan(L_ROOM, floor=-0.5, ceiling=3.0)
    lo, hi = mesh.bounds
    np.testing.assert_allclose(lo, [0, 0, -0.5])
    np.testing.assert_allclose(hi, [10, 9, 3.0])
    hull_edges = 5  # the L's reflex corner lies inside its convex hull
    assert len(mesh.triangles) == 2 * (len(L_ROOM) + hull_edges) + 2 * (hull_edges - 2)
    # A closed plan needs no hull walls: the same volume without them.
    open_sides = Mesh.from_plan(L_ROOM, floor=-0.5, ceiling=3.0, close=None)
    a = view_volume(mesh, (2.0, 2.0, 1.0), n_rays=4096).volume
    b = view_volume(open_sides, (2.0, 2.0, 1.0), n_rays=4096).volume
    assert a == pytest.approx(b, rel=1e-6)


def test_open_plan_without_hull_walls_escapes():
    plan = Plan([[[0.0, 0.0], [10.0, 0.0]], [[10.0, 0.0], [10.0, 5.0]]])
    mesh = Mesh.from_plan(plan, close=None)
    assert view_volume(mesh, (5.0, 1.0, 1.0), n_rays=1024).flags.unbounded
    closed = Mesh.from_plan(plan, close="hull")
    assert not view_volume(closed, (5.0, 1.0, 1.0), n_rays=1024).flags.unbounded
    with pytest.raises(ValueError):
        Mesh.from_plan(plan, close="box")
    with pytest.raises(ValueError):
        Mesh.from_plan(plan, floor=2.0, ceiling=1.0)
