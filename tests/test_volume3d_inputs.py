"""Inputs outside the documented domain of the 3D tools.

Each input below used to return a number, or crash the interpreter, without a
message. Now it raises a ValueError that says what is wrong and what to do, or
it returns a result the docstring defines.
"""

import math

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")
pytest.importorskip("numba")

from pysovist import (  # noqa: E402
    Mesh,
    PointCloud,
    angle_grid,
    equiangular,
    fibonacci_sphere,
    in_field_of_view,
    isovist,
    random_sphere,
    section_isovist_area,
    solid_angle_weights,
    vertical_isovist,
    vertical_section,
    view_volume,
    view_volume_field,
)
from pysovist.directions import check_field_of_view  # noqa: E402
from pysovist.raycast import build_ball_grid, build_disk_grid, cast_balls, cast_disks  # noqa: E402

from .geometry import rect  # noqa: E402


def cube(a=1.0):
    m = o3d.geometry.TriangleMesh.create_box(2 * a, 2 * a, 2 * a)
    m.translate((-a, -a, -a))
    return Mesh(np.asarray(m.vertices), np.asarray(m.triangles))


BOX = cube()
TWO = Mesh.combine([cube(), cube(3.0)])
CLOUD = PointCloud(np.array([[3.0, 0, 0], [0, 3.0, 0], [0, 0, -3.0]]), radius=0.5)
EMPTY_MESH = Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=int))
EMPTY_CLOUD = PointCloud(np.empty((0, 3)))
SQUARE = [[(0, 0), (10, 0)], [(10, 0), (10, 6)], [(10, 6), (0, 6)], [(0, 6), (0, 0)]]


# --- Mesh -------------------------------------------------------------------------------------

@pytest.mark.parametrize("v", [[(0, 0, np.nan), (1, 0, 0), (0, 1, 0)],
                               [(0, 0, np.inf), (1, 0, 0), (0, 1, 0)]])
def test_mesh_vertices_must_be_finite(v):
    with pytest.raises(ValueError, match="finite"):
        Mesh(v, [(0, 1, 2)])


@pytest.mark.parametrize("t", [[(0, 1, 2, 3)] * 3, [0, 1, 2, 0, 2, 3], [[(0, 1, 2)]]])
def test_mesh_triangles_must_have_three_columns(t):
    # Three quads of a square hold 12 indices, which used to become four wrong triangles.
    v = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], dtype=float)
    with pytest.raises(ValueError, match=r"shape \(m, 3\).*\(a, b, c\) and \(a, c, d\)"):
        Mesh(v, t)


def test_mesh_triangle_indices_must_be_whole_numbers():
    with pytest.raises(ValueError, match="whole numbers"):
        Mesh(np.eye(3), [(0.4, 1.6, 2.0)])
    assert Mesh(np.eye(3), np.array([(0.0, 1.0, 2.0)])).triangles.tolist() == [[0, 1, 2]]


def test_mesh_from_a_missing_or_empty_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Mesh.from_file(tmp_path / "missing.ply")
    pts = np.random.default_rng(0).uniform(size=(10, 3))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    o3d.io.write_point_cloud(str(tmp_path / "points.ply"), pcd)
    with pytest.raises(ValueError, match="no triangles"):
        Mesh.from_file(tmp_path / "points.ply")


@pytest.mark.parametrize("kw", [{"floor": np.nan}, {"ceiling": np.inf}, {"floor": -np.inf}])
def test_extrusion_heights_must_be_finite(kw):
    with pytest.raises(ValueError, match="finite"):
        Mesh.from_plan(SQUARE, **kw)


@pytest.mark.parametrize("close, eye", [
    ([(0, 0), (10, 0), (10, 4), (4, 4), (4, 9), (0, 9)], (7.0, 6.0, 1.2)),  # in the L's notch
    ("hull", (12.0, 3.0, 1.2)),  # beside the building
    ("hull", (5.0, 3.0, 3.5)),  # above the ceiling
])
def test_an_eye_outside_the_closed_domain_is_flagged_unbounded(close, eye):
    # A defined result: rays leave the domain, so the eye is flagged unbounded without a range
    # limit, and escape_fraction reports the share that leaves with one.
    plan = [[(0, 0), (10, 0)], [(10, 0), (10, 4)], [(10, 4), (4, 4)], [(4, 4), (4, 9)],
            [(4, 9), (0, 9)], [(0, 9), (0, 0)]]
    mesh = Mesh.from_plan(plan, floor=0.0, ceiling=2.7, close=close)
    vv = view_volume(mesh, eye, n_rays=4096)
    assert vv.flags.unbounded and math.isinf(vv.volume)
    ranged = view_volume(mesh, eye, n_rays=4096, max_distance=50.0)
    assert 0 < ranged.metrics["escape_fraction"] < 1 and not ranged.flags.unbounded
    inside = view_volume(mesh, (2.0, 2.0, 1.2), n_rays=4096)
    assert not inside.flags.unbounded and inside.metrics["escape_fraction"] == 0.0


def test_duplicate_and_zero_length_walls_change_no_view_volume():
    # A defined result: a repeated wall, a reversed copy and a wall of zero length add no
    # surface that a ray could meet first.
    extra = SQUARE + [[(3, 3), (3, 3)], [(0, 0), (10, 0)], [(10, 6), (10, 0)]]
    a = view_volume(Mesh.from_plan(SQUARE, ceiling=2.7), (4, 2, 1.5), n_rays=8192)
    b = view_volume(Mesh.from_plan(extra, ceiling=2.7), (4, 2, 1.5), n_rays=8192)
    assert a.volume == b.volume and not b.flags.on_wall


@pytest.mark.parametrize("ring, match", [
    ([(0, 0), (0, 0), (10, 6)], "three distinct corners"),
    ([(0, 0), (5, 3), (10, 6)], "three distinct corners"),
    ([(0, 0), (10, 0), (np.nan, 6)], "finite"),
])
def test_a_closing_polygon_needs_three_corners_and_an_area(ring, match):
    # These used to be refused for walls outside the polygon, which named the wrong fault.
    with pytest.raises(ValueError, match=match):
        Mesh.from_plan(SQUARE, close=ring)


@pytest.mark.parametrize("method", ["cast", "cast_hits"])
def test_mesh_rays_must_be_finite_and_non_zero(method):
    f = getattr(BOX, method)
    with pytest.raises(ValueError, match="direction"):
        f([(0, 0, 0)], [(0, 0, 0), (1, 0, 0)])
    with pytest.raises(ValueError, match="direction"):
        f([(0, 0, 0)], [(np.nan, 0, 1)])
    with pytest.raises(ValueError, match="origin"):
        f([(np.nan, 0, 0)], [(1, 0, 0)])
    # A negative near clip used to start the rays behind the eye: a depth of -1 m.
    with pytest.raises(ValueError, match="near_clip"):
        f([(0, 0, 0)], [(1, 0, 0)], near_clip=-5.0)
    with pytest.raises(ValueError, match="max_distance"):
        f([(0, 0, 0)], [(1, 0, 0)], max_distance=0.0)


@pytest.mark.parametrize("z", [np.nan, np.inf])
def test_mesh_section_height_must_be_finite(z):
    # A NaN height used to give an empty plan.
    with pytest.raises(ValueError, match="finite"):
        BOX.section(z)


def test_point_cloud_clearance_radius_must_be_positive_and_finite():
    with pytest.raises(ValueError, match="radius"):
        CLOUD.clearance([(0, 0, 0)], radius=np.nan)


def test_mesh_clearance_needs_finite_points():
    with pytest.raises(ValueError, match="origin"):
        BOX.clearance([(np.nan, 0, 0)])


# --- view volumes -----------------------------------------------------------------------------

@pytest.mark.parametrize("scene", [BOX, CLOUD])
@pytest.mark.parametrize("origin", [(np.nan, 0, 0), (0, 0, np.inf), (0, 0), (0, 0, 0, 1),
                                    [(0, 0, 0), (0.1, 0, 0)]])
def test_view_volume_takes_one_finite_eye(scene, origin):
    with pytest.raises(ValueError, match="one finite eye point"):
        view_volume(scene, origin, n_rays=64)


@pytest.mark.parametrize("scene", [BOX, CLOUD])
def test_view_volume_field_refuses_rows_that_are_not_finite(scene):
    with pytest.raises(ValueError, match="1 of 2 origins"):
        view_volume_field(scene, [(0, 0, 0), (np.nan, 0, 0)], n_rays=64)


def test_view_volume_field_of_no_origins_is_an_empty_table():
    df = view_volume_field(BOX, np.empty((0, 3)), n_rays=64)
    assert len(df) == 0 and "volume" in df.columns


@pytest.mark.parametrize("scene", [EMPTY_MESH, EMPTY_CLOUD])
def test_an_empty_scene_is_refused(scene):
    with pytest.raises(ValueError, match="empty"):
        view_volume(scene, (0, 0, 0), n_rays=64, max_distance=2.0)
    with pytest.raises(ValueError, match="empty"):
        view_volume_field(scene, [(0, 0, 0)], n_rays=64, max_distance=2.0)


@pytest.mark.parametrize("d", [[(np.nan, 0, 1), (0, 0, 1)], [(np.inf, 0, 0)], [(0, 0, 0)]])
def test_direction_arrays_must_be_finite_and_non_zero(d):
    with pytest.raises(ValueError, match="directions must be finite, non-zero"):
        view_volume(BOX, (0, 0, 0), directions=d)


@pytest.mark.parametrize("kw, match", [
    ({"near_clip": np.nan}, "near_clip"),
    ({"near_clip": 2.0, "max_distance": 1.0}, "near_clip"),
    ({"eps": -1.0}, "eps"),
    ({"eps": np.nan}, "eps"),
    ({"n_rays": 2.7}, "whole number"),
    ({"vfov": 1.0, "elevation": np.nan}, "elevation"),
    ({"hfov": 1.0, "direction": np.nan}, "direction must be a finite"),
    ({"hfov": 1.0, "direction": np.inf}, "direction must be a finite"),
])
def test_view_volume_refuses_inconsistent_options(kw, match):
    with pytest.raises(ValueError, match=match):
        view_volume(BOX, (0, 0, 0), **kw)
    with pytest.raises(ValueError, match=match):
        view_volume_field(BOX, [(0, 0, 0)], **kw)


@pytest.mark.parametrize("ground, match", [
    ([7], "object 7"), ([-1], "object -1"), ([0.7], "whole numbers"),
    (np.array([False]), "one entry per object"), (np.array([True, False, True]),
                                                  "one entry per object"),
])
def test_ground_must_name_objects_of_the_scene(ground, match):
    vv = view_volume(TWO, (0, 0, 0), n_rays=64, object_ids=True)
    assert vv.n_objects == 2
    with pytest.raises(ValueError, match=match):
        vv.factors(ground=ground)
    with pytest.raises(ValueError, match=match):
        view_volume_field(TWO, [(0, 0, 0)], n_rays=64, ground=ground)


def test_ground_of_a_point_cloud_names_its_points():
    vv = view_volume(CLOUD, (0, 0, 0), n_rays=4096, object_ids=True)
    assert vv.n_objects == 3
    with pytest.raises(ValueError, match="object 3"):
        vv.factors(ground=[3])
    assert vv.factors(ground=np.array([False, False, True]))["ground_factor"] > 0


@pytest.mark.parametrize("radius", [np.nan, np.inf, 0.0, -0.1])
def test_ball_radius_must_be_positive_and_finite(radius):
    # A radius of NaN used to crash the interpreter in the ray caster.
    with pytest.raises(ValueError, match="radius"):
        view_volume(CLOUD, (0, 0, 0), n_rays=64, radius=radius)
    with pytest.raises(ValueError, match="radius"):
        PointCloud([(0, 0, 0)], radius=radius)
    with pytest.raises(ValueError, match="radius"):
        CLOUD.with_radius(radius)
    with pytest.raises(ValueError, match="radius"):
        build_ball_grid([(0, 0, 0)], radius)


# --- point clouds -----------------------------------------------------------------------------

@pytest.mark.parametrize("pts", [[(0, 0, np.nan)], [(0, 0, np.inf)]])
def test_points_must_be_finite(pts):
    with pytest.raises(ValueError, match="finite"):
        PointCloud(pts)
    with pytest.raises(ValueError, match="finite"):
        build_ball_grid(pts, 0.1)


def test_points_must_have_three_columns():
    # Six plan points (x, y) used to become four points (x, y, z) of mixed coordinates.
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        PointCloud(np.arange(12.0).reshape(6, 2))


def test_spacing_needs_two_points():
    with pytest.raises(ValueError, match="two points"):
        PointCloud([(0, 0, 0)]).spacing()
    with pytest.raises(ValueError, match="two points"):
        EMPTY_CLOUD.spacing()
    with pytest.raises(ValueError, match="sample"):
        CLOUD.spacing(sample=0)


@pytest.mark.parametrize("region, z", [((1, 1, 0, 0), None), ((0, 0, 0, 1, 1, -1), None),
                                       ((0, 0, 1, 1), (1, 0))])
def test_crop_boxes_must_not_be_inverted(region, z):
    with pytest.raises(ValueError, match="lower"):
        CLOUD.crop(region, z=z)


@pytest.mark.parametrize("o", [(np.nan, 0, 0), (0, np.inf, 0)])
def test_ray_casters_need_finite_origins(o):
    grid = build_ball_grid([(3.0, 0, 0)], 0.5)
    with pytest.raises(ValueError, match="origin"):
        cast_balls(grid, [o], [(1, 0, 0)])
    disks = build_disk_grid([(3.0, 0)], [0.5])
    with pytest.raises(ValueError, match="origin"):
        cast_disks(disks, [o[:2]], [(1, 0)])
    with pytest.raises(ValueError, match="direction"):
        cast_balls(grid, [(0, 0, 0)], [(np.nan, 0, 0)])


# --- horizontal sections ----------------------------------------------------------------------

SECTION_CLOUD = PointCloud([(3.0, 0, 1.0)], radius=0.5)


@pytest.mark.parametrize("kw, match", [
    ({"max_distance": -2.0}, "max_distance"),  # used to give the area pi R^2 of R = -2
    ({"max_distance": 0.0}, "max_distance"),
    ({"max_distance": np.nan}, "max_distance"),
    ({"radius": -1.0}, "radius"),  # used to drop every disk
    ({"radius": np.nan}, "radius"),
    ({"eps": np.nan}, "eps"),
    ({"eps": -1.0}, "eps"),
    ({"n_rays": 3.5}, "whole number"),
])
def test_section_refuses_inconsistent_options(kw, match):
    with pytest.raises(ValueError, match=match):
        section_isovist_area(SECTION_CLOUD, (0, 0), 1.0, **{"n_rays": 36, **kw})


@pytest.mark.parametrize("origin, z", [((np.nan, 0), 1.0), ((0, 0), np.nan), ((0, np.inf), 1.0)])
def test_section_needs_a_finite_eye(origin, z):
    with pytest.raises(ValueError, match="finite"):
        section_isovist_area(SECTION_CLOUD, origin, z, n_rays=36)


def test_section_of_an_empty_cloud_is_refused():
    with pytest.raises(ValueError, match="empty"):
        section_isovist_area(EMPTY_CLOUD, (0, 0), 1.0, n_rays=36)


# --- directions -------------------------------------------------------------------------------

@pytest.mark.parametrize("f", [fibonacci_sphere, random_sphere, equiangular, solid_angle_weights])
def test_direction_counts_are_whole_numbers_of_at_least_one(f):
    with pytest.raises(ValueError, match="whole number"):
        f(2.5)
    with pytest.raises(ValueError, match="at least 1"):
        f(0)
    assert len(f(np.int64(3))) == 3 and len(f(3.0)) == 3


def test_angle_grid_counts_are_whole_numbers():
    with pytest.raises(ValueError, match="whole number"):
        angle_grid(2.5, 2)


@pytest.mark.parametrize("kw", [{"elevation": np.nan}, {"elevation": np.inf},
                                {"direction": np.nan, "hfov": 1.0}])
def test_field_of_view_angles_must_be_finite(kw):
    with pytest.raises(ValueError):
        in_field_of_view(fibonacci_sphere(10), **{"vfov": 1.0, **kw})
    with pytest.raises(ValueError):
        angle_grid(4, 2, **{"vfov": 1.0, **kw})
    if "direction" not in kw:
        with pytest.raises(ValueError, match="elevation"):
            check_field_of_view(1.0, 1.0, kw["elevation"])


@pytest.mark.parametrize("d", [[(1.0, 0, 0.5)], [(0.0, 0, 0)], [(np.nan, 0, 0)]])
def test_field_of_view_needs_unit_vectors(d):
    # (1, 0, 0.5) lies 26.6 degrees up, inside a field 60 degrees tall, but arcsin(0.5) put it
    # at 30 degrees, on the excluded upper edge.
    with pytest.raises(ValueError, match="unit vectors"):
        in_field_of_view(d, vfov=math.radians(60))
    u = np.array(d[0]) / np.linalg.norm(d[0]) if np.all(np.isfinite(d[0])) and any(d[0]) else None
    if u is not None:
        assert in_field_of_view([u], vfov=math.radians(60)).tolist() == [True]


# --- vertical sections ------------------------------------------------------------------------

ROOM = Mesh.from_plan(rect(0, 0, 10, 6), floor=0.0, ceiling=2.7, close=None)


@pytest.mark.parametrize("eye, azimuth", [((np.nan, 3, 1.6), 0.0), ((5, 3), 0.0),
                                          ((5, 3, 1.6), np.nan), ((5, 3, np.inf), 0.0)])
def test_vertical_isovist_needs_a_finite_eye_and_azimuth(eye, azimuth):
    with pytest.raises(ValueError, match="finite"):
        vertical_isovist(ROOM, eye, azimuth)
    with pytest.raises(ValueError, match="finite"):
        vertical_section(ROOM, eye, azimuth)


@pytest.mark.parametrize("kw, match", [
    ({"fov": 0.0}, "fov"), ({"fov": -1.0}, "fov"), ({"fov": 7.0}, "fov"),
    ({"max_distance": -1.0}, "max_distance"), ({"max_distance": 0.0}, "max_distance"),
    ({"eps": -1.0}, "eps"), ({"fov": 1.0, "elevation": np.nan}, "elevation"),
])
def test_vertical_isovist_refuses_inconsistent_options(kw, match):
    with pytest.raises(ValueError, match=match):
        vertical_isovist(ROOM, (5, 3, 1.6), 0.0, **kw)


# --- Unity outputs ---------------------------------------------------------------------------

def test_unity_volumes_need_their_columns(tmp_path):
    from pysovist.compat.unity import read_unity_volumes

    (tmp_path / "v.csv").write_text("x,y,z\n1,2,3\n")
    with pytest.raises(ValueError, match="volume"):
        read_unity_volumes(tmp_path / "v.csv")


# --- 2D isovist results -----------------------------------------------------------------------

def test_radial_needs_enough_rays():
    full = isovist(rect(-1, -1, 1, 1), (0, 0))
    part = isovist(rect(-1, -1, 1, 1), (0, 0), fov=1.0)
    with pytest.raises(ValueError, match="at least 1"):
        full.radial(0)
    # One ray cannot span a field of view: it used to return NaN.
    with pytest.raises(ValueError, match="at least 2"):
        part.radial(1)
    with pytest.raises(ValueError, match="whole number"):
        full.radial(2.5)
    assert full.radial(1)[1] == pytest.approx([1.0])


@pytest.mark.parametrize("eye, flag", [((1.0, 0.0), "on_wall"), ((0.0, 0.0), "unbounded")])
def test_depth_and_radial_of_a_flagged_isovist_are_nan(eye, flag):
    # A defined result: a flagged isovist has no wedges, and its depths are NaN. They used to
    # raise an IndexError.
    plan = rect(-1, -1, 1, 1) if flag == "on_wall" else rect(-1, -1, 1, 1)[:3]
    iso = isovist(plan, eye)
    assert getattr(iso.flags, flag)
    assert np.isnan(iso.depth([0.0, 1.0, 2.0])).all()
    ang, r = iso.radial(4)
    assert len(ang) == 4 and np.isnan(r).all()


def test_radial_offset_must_be_finite():
    with pytest.raises(ValueError, match="offset"):
        isovist(rect(-1, -1, 1, 1), (0, 0)).radial(4, offset=np.nan)


@pytest.mark.parametrize("kw", [{}, {"fov": 1.0}, {"fov": 1.0, "max_distance": 0.5},
                                {"max_distance": 0.5}])
def test_polygon_ring_repeats_its_first_point(kw):
    # A full turn and a field of view alike end where they start (a field of view at the
    # observer), so the ring is closed. A field of view used to stop at its last boundary point.
    ring = isovist(rect(-1, -1, 1, 1), (0.2, 0.1), **kw).polygon()
    np.testing.assert_allclose(ring[-1], ring[0], atol=1e-12)
    assert len(ring) >= 4
    if "fov" in kw:
        assert ring[0].tolist() == [0.2, 0.1] and ring[-1].tolist() == [0.2, 0.1]
    assert len(isovist(rect(-1, -1, 1, 1), (1.0, 0.0)).polygon()) == 0  # on a wall


@pytest.mark.parametrize("close", ["hull", None])
def test_a_plan_without_area_is_refused(close):
    # The floor and the ceiling cover the plan's hull. Walls on one line have none: an opaque
    # QhullError before.
    with pytest.raises(ValueError, match="area"):
        Mesh.from_plan([[(0, 0), (10, 0)], [(2, 0), (5, 0)]], close=close)


@pytest.mark.parametrize("step", [0.0, -0.1, np.nan])
def test_polygon_arc_step_must_be_positive(step):
    iso = isovist(rect(-1, -1, 1, 1), (0, 0), max_distance=0.5)
    with pytest.raises(ValueError, match="arc_step"):
        iso.polygon(step)
