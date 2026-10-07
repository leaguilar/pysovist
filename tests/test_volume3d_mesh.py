"""View volumes of triangle meshes against closed-form volumes.

Every enclosure below is star-shaped from its eye, so the view volume equals
the enclosure's volume. The Fibonacci rule integrates r^3 over the sphere.
Here r^3 is piecewise smooth with kinks along the edges seen from the eye, and
the error falls roughly as N^-1 (measured: 3.7e-4 at N = 1024 to 3.2e-6 at
N = 65536 on the off-centre cube, at most 1.4e-5 on random hulls).
"""

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

o3d = pytest.importorskip("open3d")
from scipy.spatial import ConvexHull  # noqa: E402

from pysovist.directions import fibonacci_sphere  # noqa: E402
from pysovist.mesh import Mesh  # noqa: E402
from pysovist.volume3d import VOLUME_METRIC_NAMES, view_volume, view_volume_field  # noqa: E402

N_MAX = 65536
REL_MAX = 1e-4  # relative error target at N_MAX on smooth-faced enclosures


def box_mesh(a, center=(0.0, 0.0, 0.0)):
    m = o3d.geometry.TriangleMesh.create_box(2 * a, 2 * a, 2 * a)
    m.translate(np.asarray(center) - a)
    return Mesh(np.asarray(m.vertices), np.asarray(m.triangles))


def rel_err(x, ref):
    return abs(x / ref - 1)


# --- convergence ------------------------------------------------------------------------------

def test_cube_any_interior_eye_converges_to_8a3():
    a = 1.5
    mesh = box_mesh(a)
    ns = [1024, 4096, 16384, N_MAX]
    errs = [rel_err(view_volume(mesh, (0.2, -0.3, 0.4), n_rays=n).volume, 8 * a**3) for n in ns]
    slope = np.polyfit(np.log(ns), np.log(errs), 1)[0]
    assert errs[-1] < errs[0] / 16
    assert slope < -0.75  # faster than the N^-1/2 of independent random directions
    assert errs[-1] < REL_MAX


@pytest.mark.parametrize("eye", [(0.0, 0.0, 0.0), (1.4, 1.4, -1.4), (-0.7, 0.1, 1.2)])
def test_cube_volume_is_the_same_from_every_interior_eye(eye):
    assert rel_err(view_volume(box_mesh(1.5), eye, n_rays=N_MAX).volume, 27.0) < REL_MAX


def test_sphere_mesh_matches_its_own_tetrahedra_volume():
    m = o3d.geometry.TriangleMesh.create_sphere(radius=2.0, resolution=80)
    mesh = Mesh(np.asarray(m.vertices), np.asarray(m.triangles))
    ref = mesh.volume
    assert ref == pytest.approx(4 / 3 * math.pi * 8, rel=1e-3)  # faceted, slightly smaller
    assert rel_err(view_volume(mesh, (0.3, 0.2, -0.4), n_rays=N_MAX).volume, ref) < REL_MAX


def test_square_pyramid():
    v = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 3.0]])
    mesh = Mesh(v, ConvexHull(v).simplices)
    assert rel_err(view_volume(mesh, (0.1, -0.1, 0.8), n_rays=N_MAX).volume, 4.0) < REL_MAX


@settings(max_examples=8, deadline=None)
@given(n=st.integers(8, 40), seed=st.integers(0, 2**32 - 1))
def test_random_convex_hull(n, seed):
    pts = np.random.default_rng(seed).uniform(-1, 1, size=(n, 3))
    hull = ConvexHull(pts)
    eye = pts[hull.vertices].mean(axis=0)
    # Keep the eye well inside: the quadrature error grows as the eye nears a face.
    depth = -(hull.equations[:, :3] @ eye + hull.equations[:, 3]).max()
    assume(hull.volume > 0.2 and depth > 0.1)
    vv = view_volume(Mesh(pts, hull.simplices), eye, n_rays=N_MAX)
    assert rel_err(vv.volume, hull.volume) < 2 * REL_MAX


# --- range limit ------------------------------------------------------------------------------

def test_cube_with_range_limit_from_the_centre():
    R, h = 1.2, 0.2
    ref = 4 / 3 * math.pi * R**3 - 6 * math.pi * h**2 * (3 * R - h) / 3
    assert ref == pytest.approx(6.38372, abs=1e-5)
    vv = view_volume(box_mesh(1.0), (0, 0, 0), n_rays=N_MAX, max_distance=R)
    assert rel_err(vv.volume, ref) < REL_MAX
    assert not vv.flags.unbounded
    assert vv.distances.max() == R


def test_floor_with_range_limit():
    v = np.array([[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0.0]])
    mesh = Mesh(v, [[0, 1, 2], [0, 2, 3]])
    vv = view_volume(mesh, (0.0, 0.0, 1.0), n_rays=N_MAX, max_distance=2.0)
    assert 32 * math.pi / 3 - 5 * math.pi / 3 == pytest.approx(28.2743, abs=1e-4)
    assert rel_err(vv.volume, 9 * math.pi) < REL_MAX
    assert rel_err(vv.metrics["volume_up"], 16 * math.pi / 3) < REL_MAX
    assert vv.metrics["escape_fraction"] == pytest.approx(0.5 + 0.5 * 0.5, abs=1e-3)


def test_open_box_is_unbounded_without_range():
    m = o3d.geometry.TriangleMesh.create_box(2, 2, 2)
    m.translate((-1, -1, -1))
    v, t = np.asarray(m.vertices), np.asarray(m.triangles)
    top = np.all(v[t][:, :, 2] > 0.5, axis=1)
    vv = view_volume(Mesh(v, t[~top]), (0, 0, 0), n_rays=4096)
    assert vv.flags.unbounded and math.isinf(vv.volume)
    # The open top is a square seen under 4 arctan(1 / sqrt(3)) = 2 pi / 3 steradians.
    assert vv.metrics["escape_fraction"] == pytest.approx(1 / 6, abs=2e-3)


# --- mesh model -------------------------------------------------------------------------------

def test_mesh_from_arrays_and_file(tmp_path):
    m = o3d.geometry.TriangleMesh.create_box(1, 2, 3)
    path = tmp_path / "box.ply"
    assert o3d.io.write_triangle_mesh(str(path), m)
    mesh = Mesh.from_file(path)
    assert len(mesh.triangles) == 12
    assert abs(mesh.volume) == pytest.approx(6.0, rel=1e-12)
    np.testing.assert_allclose(mesh.vertices.min(axis=0), [0, 0, 0])
    with pytest.raises(ValueError):
        Mesh(np.zeros((3, 2)), [[0, 1, 2]])


def test_clearance_and_eye_on_a_wall():
    mesh = box_mesh(1.0)
    np.testing.assert_allclose(mesh.clearance([[0, 0, 0], [0.9, 0.2, 0.0], [1.0, 0.3, 0.3]]),
                               [1.0, 0.1, 0.0], atol=1e-12)
    vv = view_volume(mesh, (1.0, 0.3, 0.3), n_rays=256)
    assert vv.flags.on_wall and np.isnan(vv.volume)
    assert view_volume(mesh, (1.0, 0.3, 0.3), n_rays=256, inside="zero").volume == 0.0


def test_near_clip_skips_surfaces_close_along_each_ray():
    box = o3d.geometry.TriangleMesh.create_box(2, 2, 2)
    box.translate((-1, -1, -1))
    plate = o3d.geometry.TriangleMesh.create_box(0.4, 0.4, 0.01)
    plate.translate((-0.2, -0.2, 0.1))
    both = box + plate
    mesh = Mesh(np.asarray(both.vertices), np.asarray(both.triangles))
    clipped = view_volume(mesh, (0, 0, 0), n_rays=16384, near_clip=0.4)  # plate within 0.31
    plain = view_volume(box_mesh(1.0), (0, 0, 0), n_rays=16384)
    assert clipped.volume == pytest.approx(plain.volume, rel=1e-6)
    assert clipped.near_clipped
    assert view_volume(mesh, (0, 0, 0), n_rays=16384).volume < plain.volume - 0.01


def test_field_on_a_mesh_matches_single_volumes():
    mesh = box_mesh(1.0)
    origins = np.array([[0.0, 0, 0], [0.5, -0.5, 0.2], [1.0, 0.0, 0.0]])
    df = view_volume_field(mesh, origins, n_rays=2048)
    assert df["on_wall"].tolist() == [False, False, True]
    for i, o in enumerate(origins[:2]):
        vv = view_volume(mesh, o, n_rays=2048)
        for k in VOLUME_METRIC_NAMES:
            assert df[k].iloc[i] == pytest.approx(vv.metrics[k], rel=1e-12)
    assert df[list(VOLUME_METRIC_NAMES)].iloc[2].isna().all()


def test_far_from_the_coordinate_origin():
    # Rays are cast in single precision about the mesh centre, so a georeferenced
    # scene keeps its accuracy.
    shift = np.array([2.6e6, 1.2e6, 450.0])
    mesh = box_mesh(1.0, center=shift)
    vv = view_volume(mesh, shift + 0.1, directions=fibonacci_sphere(N_MAX))
    assert rel_err(vv.volume, 8.0) < REL_MAX


def test_clearance_matches_dense_sampling_of_random_triangles():
    rng = np.random.default_rng(4)
    tri = rng.uniform(-1, 1, size=(40, 3, 3))
    mesh = Mesh(tri.reshape(-1, 3), np.arange(120).reshape(40, 3))
    q = rng.uniform(-2, 2, size=(30, 3))
    k = 200
    u, v = np.meshgrid(np.linspace(0, 1, k + 1), np.linspace(0, 1, k + 1))
    keep = u + v <= 1
    bary = np.c_[1 - u[keep] - v[keep], u[keep], v[keep]]
    samples = np.einsum("sk,tkd->tsd", bary, tri).reshape(-1, 3)
    dense = np.linalg.norm(q[:, None] - samples[None], axis=2).min(axis=1)
    exact = mesh.clearance(q)
    # The sampled minimum overshoots by at most one barycentric step of the longest edge.
    step = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max() / k
    assert np.all(exact <= dense + 1e-12)
    assert np.all(dense - exact <= step)
