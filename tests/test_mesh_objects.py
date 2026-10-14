"""Meshes built from several objects: object ids per triangle and ray casts that return them."""

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from pysovist.directions import fibonacci_sphere  # noqa: E402
from pysovist.mesh import Mesh  # noqa: E402

from .geometry import rect  # noqa: E402


def box(x0, y0, z0, x1, y1, z1):
    m = o3d.geometry.TriangleMesh.create_box(x1 - x0, y1 - y0, z1 - z0)
    m.translate((x0, y0, z0))
    return Mesh(np.asarray(m.vertices), np.asarray(m.triangles))


def test_a_plain_mesh_is_one_object():
    m = box(0, 0, 0, 1, 1, 1)
    assert m.object_ids.shape == (12,)
    assert np.all(m.object_ids == 0)
    assert m.n_objects == 1
    assert not m.object_ids.flags.writeable


def test_object_ids_are_validated():
    v = np.eye(3)
    with pytest.raises(ValueError):
        Mesh(v, [(0, 1, 2)], object_ids=[0, 1])
    with pytest.raises(ValueError):
        Mesh(v, [(0, 1, 2)], object_ids=[-1])
    m = Mesh(v, [(0, 1, 2)], object_ids=[4])
    assert m.object_ids.tolist() == [4] and m.n_objects == 5


def test_combine_concatenates_and_remembers_the_source():
    a, b, c = box(0, 0, 0, 1, 1, 1), box(3, 0, 0, 4, 1, 2), Mesh.from_plan(rect(-5, -5, 5, 5))
    m = Mesh.combine([a, b, c])
    assert len(m.vertices) == len(a.vertices) + len(b.vertices) + len(c.vertices)
    assert len(m.triangles) == len(a.triangles) + len(b.triangles) + len(c.triangles)
    np.testing.assert_array_equal(m.vertices[: len(a.vertices)], a.vertices)
    np.testing.assert_array_equal(m.triangles[len(a.triangles): len(a.triangles) + 12],
                                  b.triangles + len(a.vertices))
    expected = np.repeat([0, 1, 2], [len(a.triangles), len(b.triangles), len(c.triangles)])
    np.testing.assert_array_equal(m.object_ids, expected)
    assert m.n_objects == 3
    # Each triangle keeps its corners.
    np.testing.assert_array_equal(m.vertices[m.triangles[m.object_ids == 1]],
                                  b.vertices[b.triangles])


def test_combine_accepts_an_empty_object_and_rejects_nothing():
    empty = Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=int))
    m = Mesh.combine([empty, box(0, 0, 0, 1, 1, 1)])
    assert np.all(m.object_ids == 1) and m.n_objects == 2
    with pytest.raises(ValueError):
        Mesh.combine([])


@pytest.mark.parametrize("kw", [{}, {"max_distance": 2.5}, {"near_clip": 0.3}])
def test_cast_hits_returns_the_distances_of_cast_and_the_triangles_hit(kw):
    scene = Mesh.combine([box(0, 0, 0, 10, 8, 3), box(6, 3, 0, 7, 5, 2)])
    origins = np.array([[2.0, 2.0, 1.5], [8.5, 4.0, 1.0], [5.0, 7.0, 2.9]])
    d = fibonacci_sphere(3000)
    t, tri = scene.cast_hits(origins, d, **kw)
    np.testing.assert_array_equal(t, scene.cast(origins, d, **kw))
    assert tri.shape == t.shape and tri.dtype.kind == "i"
    assert np.array_equal(tri >= 0, np.isfinite(t))
    # The hit point lies on the triangle reported for it.
    i, j = np.nonzero(tri >= 0)
    p = origins[i] + d[j] * t[i, j][:, None]
    a, b, c = (scene.vertices[scene.triangles[tri[i, j], k]] for k in range(3))
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1)[:, None]
    assert np.abs(np.einsum("ij,ij->i", p - a, n)).max() < 1e-5
    # The block (object 1) lies 4 m from the first eye: its rays reach it unless the range stops
    # them first.
    ids = scene.object_ids[tri[0][tri[0] >= 0]]
    assert set(np.unique(ids)) == ({0} if kw.get("max_distance", np.inf) < 4 else {0, 1})


def test_cast_hits_on_an_empty_mesh():
    empty = Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=int))
    t, tri = empty.cast_hits(np.zeros((1, 3)), fibonacci_sphere(5))
    assert np.all(np.isinf(t)) and np.all(tri == -1)
