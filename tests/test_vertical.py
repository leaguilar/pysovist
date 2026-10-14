"""Exact isovists in a vertical plane through the eye, cut from a triangle mesh."""

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pysovist import Mesh, Plan, isovist
from pysovist.rays2d import cast_rays
from pysovist.vertical import plane_to_world, vertical_isovist, vertical_section

from .geometry import polygon_walls, rect

H = 2.7
BOX = Mesh.from_plan(rect(0, 0, 10, 6), floor=0.0, ceiling=H, close=None)
L_ROOM = Plan(polygon_walls([(0, 0), (10, 0), (10, 4), (4, 4), (4, 9), (0, 9)]))
PILLAR_ROOM = Plan(np.concatenate([rect(0, 0, 10, 8), rect(4, 3, 6, 5)]))


def merge(*meshes):
    verts, tris, n = [], [], 0
    for m in meshes:
        verts.append(m.vertices)
        tris.append(m.triangles + n)
        n += len(m.vertices)
    return Mesh(np.concatenate(verts), np.concatenate(tris))


# A table 1 m deep and 0.75 m high stands 2 m in front of the eye at (1, 3, 1.6).
TABLE_ROOM = merge(BOX, Mesh.from_plan(rect(3, 2, 4, 4), floor=0.0, ceiling=0.75, close=None))


def test_box_room_seen_from_inside_is_the_rectangle_of_its_depth_and_height():
    iso = vertical_isovist(BOX, (5.0, 3.0, 1.6), 0.0)
    assert iso.area == pytest.approx(10 * H, rel=1e-12)
    assert iso.perimeter == pytest.approx(2 * (10 + H), rel=1e-12)
    assert iso.origin.tolist() == [0.0, 1.6]
    across = vertical_isovist(BOX, (4.0, 3.0, 1.6), math.pi / 2)
    assert across.area == pytest.approx(6 * H, rel=1e-12)
    diagonal = vertical_isovist(BOX, (4.0, 3.0, 1.6), math.pi / 4)
    assert diagonal.area == pytest.approx(6 * math.sqrt(2) * H, rel=1e-12)


def test_the_section_is_in_plane_coordinates():
    plan = vertical_section(BOX, (4.0, 3.0, 1.6), 0.0)
    assert isinstance(plan, Plan)
    assert plan.bounds == pytest.approx((-4.0, 0.0, 6.0, H))
    back = vertical_section(BOX, (4.0, 3.0, 1.6), math.pi)
    assert back.bounds == pytest.approx((-6.0, 0.0, 4.0, H))


def test_elevation_and_field_of_view_select_part_of_the_plane():
    eye = (5.0, 3.0, 1.6)
    up = vertical_isovist(BOX, eye, 0.0, elevation=math.pi / 2, fov=math.pi)
    down = vertical_isovist(BOX, eye, 0.0, elevation=-math.pi / 2, fov=math.pi)
    ahead = vertical_isovist(BOX, eye, 0.0, fov=math.pi)
    assert up.area == pytest.approx(10 * (H - 1.6), rel=1e-12)
    assert down.area == pytest.approx(10 * 1.6, rel=1e-12)
    assert ahead.area == pytest.approx(5 * H, rel=1e-12)
    assert up.direction == math.pi / 2


def test_a_table_casts_a_shadow_behind_it():
    # The sight line over the far edge of the table top, (3, 0.75) in the plane, meets the
    # floor at u = 1.6 x 3 / 0.85. The shadow behind the table is the triangle below it.
    iso = vertical_isovist(TABLE_ROOM, (1.0, 3.0, 1.6), 0.0)
    u_floor = 1.6 * 3 / 0.85
    shadow = 0.5 * (u_floor - 3) * 0.75
    assert iso.area == pytest.approx(10 * H - 1 * 0.75 - shadow, rel=1e-12)


def test_rotating_the_building_as_in_the_tutorial_gives_the_same_isovist():
    # The tutorial turns the model by 90 degrees about the x-axis and cuts it horizontally:
    # (x, y, z) -> (x, z, -y) puts the plane y = 3 at height -3, in plan coordinates (x, z).
    v = TABLE_ROOM.vertices
    turned = Mesh(np.c_[v[:, 0], v[:, 2], -v[:, 1]], TABLE_ROOM.triangles)
    reference = isovist(turned.section(-3.0), (1.0, 1.6))
    iso = vertical_isovist(TABLE_ROOM, (1.0, 3.0, 1.6), 0.0)
    for key in ("area", "perimeter", "occlusivity", "r_mean", "r_max"):
        assert iso.metrics[key] == pytest.approx(reference.metrics[key], rel=1e-12), key


def test_plane_to_world_puts_the_isovist_back_in_the_vertical_plane():
    eye = np.array([4.0, 3.0, 1.6])
    azimuth = 0.3
    iso = vertical_isovist(BOX, eye, azimuth)
    ring = iso.polygon()
    world = plane_to_world(ring, eye, azimuth)
    assert world.shape == (len(ring), 3)
    normal = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
    assert (world - eye) @ normal == pytest.approx(np.zeros(len(ring)), abs=1e-12)
    assert world[:, 2] == pytest.approx(ring[:, 1])
    ahead = (world - eye)[:, :2] @ [math.cos(azimuth), math.sin(azimuth)]
    assert ahead == pytest.approx(ring[:, 0], abs=1e-12)


def test_an_open_scene_needs_a_range():
    floor = Mesh([(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)], [(0, 1, 2), (0, 2, 3)])
    assert vertical_isovist(floor, (0.0, 0.0, 1.0), 0.0).flags.unbounded
    iso = vertical_isovist(floor, (0.0, 0.0, 1.0), 0.0, max_distance=2.0)
    # The range disc of radius 2 minus the cap below the floor, 1 m under the eye.
    cap = 4 * math.acos(0.5) - 1 * math.sqrt(3)
    assert iso.area == pytest.approx(math.pi * 4 - cap, rel=1e-12)


def test_an_eye_on_a_surface_is_flagged():
    iso = vertical_isovist(BOX, (5.0, 3.0, 0.0), 0.0)
    assert iso.flags.on_wall
    with pytest.raises(ValueError):
        vertical_isovist(BOX, (5.0, 3.0, 0.0), 0.0, on_wall="raise")


def test_a_triangle_in_the_plane_gives_no_segment():
    flat = Mesh([(0, 0, 0), (4, 0, 0), (0, 0, 3)], [(0, 1, 2)])
    assert len(vertical_section(flat, (1.0, 0.0, 1.0), 0.0)) == 0


# --- property tests ---------------------------------------------------------------------------

@settings(max_examples=60, deadline=None)
@given(plan=st.sampled_from([L_ROOM, PILLAR_ROOM]), fx=st.floats(0.01, 0.99),
       fy=st.floats(0.01, 0.99), fz=st.floats(0.02, 0.98), azimuth=st.floats(-7, 7))
def test_extruded_plan_gives_the_rectangle_of_the_two_plan_rays(plan, fx, fy, fz, azimuth):
    # Full-height walls hide everything behind the first wall in each direction, so the
    # vertical isovist is the rectangle between the plan's ray ahead and its ray behind.
    xmin, ymin, xmax, ymax = plan.bounds
    eye2 = np.array([xmin + fx * (xmax - xmin), ymin + fy * (ymax - ymin)])
    plan_iso = isovist(plan, eye2)
    assume(not plan_iso.flags.on_wall and plan_iso.clearance > 1e-3)
    assume(not plan_iso.flags.unbounded)  # the eye stands inside the room
    rays = cast_rays(plan, eye2, [azimuth, azimuth + math.pi])
    assume(rays.hit.all())
    ahead, behind = rays.distance[0]
    mesh = Mesh.from_plan(plan, floor=0.0, ceiling=H, close=None)
    iso = vertical_isovist(mesh, (*eye2, fz * H), azimuth)
    assert iso.area == pytest.approx(H * (ahead + behind), rel=1e-9)
    assert iso.perimeter == pytest.approx(2 * (H + ahead + behind), rel=1e-9)
