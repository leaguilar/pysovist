"""Closed-form and independent references for 3D names and options that had none.

Each test names its reference: a closed form, an integral evaluated by scipy, a
brute-force search, or the documented definition of an output built by hand.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import dblquad

o3d = pytest.importorskip("open3d")
pytest.importorskip("numba")

from pysovist import (  # noqa: E402
    Mesh,
    Plan,
    PointCloud,
    fibonacci_sphere,
    isovist,
    random_sphere,
    section_isovist_area,
    vertical_isovist,
    view_volume,
    view_volume_field,
)
from pysovist.compat.unity import read_unity_volumes  # noqa: E402
from pysovist.directions import GOLDEN_ANGLE, check_field_of_view  # noqa: E402
from pysovist.raycast import (  # noqa: E402
    build_ball_grid,
    cast_balls,
    first_hit_bruteforce,
    first_hit_disks_bruteforce,
)

from .geometry import rect  # noqa: E402


def cube(a=1.0):
    m = o3d.geometry.TriangleMesh.create_box(2 * a, 2 * a, 2 * a)
    m.translate((-a, -a, -a))
    return Mesh(np.asarray(m.vertices), np.asarray(m.triangles))


def ray_box_distance(o, d, a=1.0):
    """Distance from o inside [-a, a]^3 along unit d to the box surface (closed form)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (np.sign(d) * a - o) / d
    return np.where(d == 0, np.inf, t).min(axis=1)


# --- Mesh -------------------------------------------------------------------------------------

def test_mesh_volume_sign_follows_the_orientation():
    # Reference: a 1 x 2 x 3 box encloses 6 m3, positive with outward normals.
    m = o3d.geometry.TriangleMesh.create_box(1, 2, 3)
    v, t = np.asarray(m.vertices), np.asarray(m.triangles)
    assert Mesh(v, t).volume == pytest.approx(6.0, rel=1e-12)
    assert Mesh(v, t[:, ::-1]).volume == pytest.approx(-6.0, rel=1e-12)


@pytest.mark.parametrize("n_threads", [None, 1])
def test_mesh_cast_gives_the_ray_box_distances(n_threads):
    # Reference: the closed-form distance from an interior point to the faces of a cube.
    box = cube()
    o = np.array([0.2, -0.3, 0.4])
    d = fibonacci_sphere(500)
    exact = ray_box_distance(o, d)
    t = box.cast([o], d, n_threads=n_threads)[0]
    np.testing.assert_allclose(t, exact, rtol=2e-6)
    # Range limit: depths beyond it are misses.
    t = box.cast([o], d, max_distance=1.0, n_threads=n_threads)[0]
    np.testing.assert_array_equal(np.isinf(t), exact > 1.0 * (1 + 1e-6))
    # Near clip: a ray starts 1.2 m out, so the faces nearer than that are skipped and the ray
    # leaves the single box.
    t = box.cast([np.zeros(3)], d, near_clip=1.2, n_threads=n_threads)[0]
    exact0 = ray_box_distance(np.zeros(3), d)
    far = exact0 > 1.2 + 1e-5
    near = exact0 < 1.2 - 1e-5
    np.testing.assert_allclose(t[far], exact0[far], rtol=2e-6)
    assert np.all(np.isinf(t[near]))


def test_mesh_blocked_ignores_the_surfaces_near_clip_skips():
    # Reference: the eye stands 0.05 m from the face x = 1 (closed-form clearance).
    box = cube()
    eye = [(0.95, 0.0, 0.0)]
    assert box.blocked(eye, eps=0.1).tolist() == [True]
    assert box.blocked(eye, eps=0.01).tolist() == [False]
    assert box.blocked(eye, eps=0.1, near_clip=0.1).tolist() == [False]


# --- view volumes -----------------------------------------------------------------------------

def test_random_directions_estimate_the_cube_volume_without_bias():
    # Reference: 8 a^3. Independent random directions give a Monte Carlo estimate whose
    # standard error is (4 pi / 3) sd(r^3) / sqrt(N).
    box = cube()
    n = 20000
    for seed in range(5):
        vv = view_volume(box, (0.2, -0.3, 0.4), n_rays=n, directions="random", seed=seed)
        se = 4 * math.pi / 3 * np.std(vv.distances**3) / math.sqrt(n)
        assert abs(vv.volume - 8.0) < 4 * se


def test_depth_statistics_from_the_centre_of_a_cube():
    # Reference: scipy's integrals over one face of the cube, for the mean depth and the mean
    # square depth over the sphere. A face at distance a seen from the centre contributes
    # int r dOmega = a int dA / rho^2 and int r^2 dOmega = a int dA / rho, rho^2 = x^2 + y^2 + a^2.
    a = 1.0
    i1 = dblquad(lambda y, x: a / (x * x + y * y + a * a), -a, a, -a, a, epsabs=1e-12)[0]
    i2 = dblquad(lambda y, x: a / math.sqrt(x * x + y * y + a * a), -a, a, -a, a,
                 epsabs=1e-12)[0]
    mean, mean2 = 6 * i1 / (4 * math.pi), 6 * i2 / (4 * math.pi)
    vv = view_volume(cube(a), (0, 0, 0), n_rays=65536)
    m = vv.metrics
    assert m["r_mean"] == pytest.approx(mean, rel=1e-4)
    assert m["r_std"] == pytest.approx(math.sqrt(mean2 - mean * mean), rel=1e-3)
    assert a <= m["r_min"] < a * (1 + 1e-5)
    assert a * math.sqrt(3) * (1 - 2e-3) < m["r_max"] <= a * math.sqrt(3) * (1 + 1e-6)


def test_equivalent_radius_of_a_sphere_is_its_radius():
    # Reference: a faceted sphere of radius 2 encloses Mesh.volume, the sum of its tetrahedra.
    m = o3d.geometry.TriangleMesh.create_sphere(radius=2.0, resolution=80)
    mesh = Mesh(np.asarray(m.vertices), np.asarray(m.triangles))
    vv = view_volume(mesh, (0, 0, 0), n_rays=65536)
    assert vv.metrics["equivalent_radius"] == pytest.approx(
        (3 * mesh.volume / (4 * math.pi)) ** (1 / 3), rel=1e-4)
    assert vv.metrics["equivalent_radius"] == pytest.approx(2.0, rel=1e-3)


@pytest.mark.parametrize("n_jobs", [-1, 1, 3])
def test_field_gives_the_cube_volume_from_every_eye(n_jobs):
    # Reference: 8 a^3 from every interior eye.
    df = view_volume_field(cube(), [(0, 0, 0), (0.5, -0.5, 0.2), (-0.7, 0.1, 0.6)],
                           n_rays=65536, n_jobs=n_jobs)
    np.testing.assert_allclose(df["volume"], 8.0, rtol=1e-4)
    assert not df[["on_wall", "inside_occluder", "unbounded", "near_clipped"]].any().any()


# --- directions -------------------------------------------------------------------------------

def test_field_of_view_edges():
    # Reference: the edges lie vfov / 2 below and above the centre, cut at the poles.
    assert check_field_of_view(1.0, 1.0, 0.2) == pytest.approx((-0.3, 0.7), abs=1e-15)
    assert check_field_of_view(2.0, math.pi, 0.0) == pytest.approx((-math.pi / 2, math.pi / 2))
    lo, hi = check_field_of_view(1.0, math.pi / 2, math.pi / 4)
    assert (lo, hi) == pytest.approx((0.0, math.pi / 2), abs=1e-15)


def test_fibonacci_azimuths_step_by_the_golden_angle():
    # Reference: GOLDEN_ANGLE = pi (3 - sqrt 5), the azimuth step of the spiral.
    assert GOLDEN_ANGLE == pytest.approx(math.pi * (3 - math.sqrt(5)), rel=1e-15)
    d = fibonacci_sphere(50)
    step = np.mod(np.diff(np.arctan2(d[:, 1], d[:, 0])), 2 * math.pi)
    np.testing.assert_allclose(step, GOLDEN_ANGLE, atol=1e-12)


# --- point clouds -----------------------------------------------------------------------------

def test_point_cloud_bounds():
    pts = np.array([(1.0, -2.0, 0.5), (3.0, 4.0, -1.0), (0.0, 0.0, 2.0)])
    lo, hi = PointCloud(pts).bounds
    assert lo.tolist() == [0.0, -2.0, -1.0] and hi.tolist() == [3.0, 4.0, 2.0]


def test_read_laz(tmp_path):
    laspy = pytest.importorskip("laspy")
    pytest.importorskip("lazrs")
    pts = np.round(np.random.default_rng(1).uniform(-5, 5, size=(300, 3)), 3)
    las = laspy.create(point_format=3, file_version="1.2")
    las.header.scales = np.array([0.001, 0.001, 0.001])
    las.header.offsets = np.zeros(3)
    las.x, las.y, las.z = pts.T
    las.write(tmp_path / "scan.laz")
    # Reference: the coordinates written, to the file's resolution of 1 mm.
    np.testing.assert_allclose(PointCloud.read(tmp_path / "scan.laz").points, pts, atol=1e-9)


def test_a_coarsened_ball_grid_stays_exact():
    # Reference: the brute-force first hit. A grid capped at 1,000 cells coarsens its cells.
    rng = np.random.default_rng(2)
    pts = rng.uniform(0, 2, size=(3000, 3))
    grid = build_ball_grid(pts, 0.05, max_cells=1000)
    assert grid.n_cells <= 1000 * 1.1 and grid.cell_size > 0.1
    origins = rng.uniform(0, 2, size=(10, 3))
    d = random_sphere(200, seed=4)
    t = cast_balls(grid, origins, d)
    ref = first_hit_bruteforce(pts, 0.05, origins, d)
    np.testing.assert_array_equal(np.isinf(t), np.isinf(ref))
    np.testing.assert_allclose(t[np.isfinite(ref)], ref[np.isfinite(ref)], atol=1e-12)


def test_brute_force_disk_distance_is_the_closed_form():
    # Reference: a ray along +x from (0, 0.3) meets the unit disk at (4, 0) at 4 - sqrt(0.91).
    t = first_hit_disks_bruteforce([(4.0, 0.0)], [1.0], [(0.0, 0.3), (4.2, 0.0), (0.0, 2.0)],
                                   [(1.0, 0.0)])
    assert t[0, 0] == pytest.approx(4 - math.sqrt(0.91), abs=1e-14)
    assert t[1, 0] == 0.0 and math.isinf(t[2, 0])


@pytest.mark.parametrize("z, disk", [(1.0, 0.5), (1.3, 0.4)])
def test_section_clearance_and_eps(z, disk):
    # Reference: the plane at height z cuts the ball of radius 0.5 at (3, 0, 1) in a disk of
    # radius sqrt(0.25 - (z - 1)^2), so an eye at (2.4, 0) stands 0.6 - disk from its edge.
    cloud = PointCloud([(3.0, 0.0, 1.0)], radius=0.5)
    sec = section_isovist_area(cloud, (2.4, 0.0), z, n_rays=360, max_distance=10.0)
    assert sec.clearance == pytest.approx(0.6 - disk, abs=1e-12)
    assert not sec.flags.inside_occluder
    near = section_isovist_area(cloud, (2.4, 0.0), z, n_rays=360, max_distance=10.0,
                                eps=0.6 - disk + 0.01)
    assert near.flags.inside_occluder and math.isnan(near.area)


# --- vertical isovists and 2D results ---------------------------------------------------------

def test_vertical_isovist_eps():
    # Reference: an eye 1e-4 m above the floor of a 10 x 6 x 2.7 m box room stands on it for
    # eps = 1e-3 and sees the 10 x 2.7 m rectangle for eps = 1e-6.
    room = Mesh.from_plan(rect(0, 0, 10, 6), floor=0.0, ceiling=2.7, close=None)
    assert vertical_isovist(room, (5.0, 3.0, 1e-4), 0.0, eps=1e-3).flags.on_wall
    iso = vertical_isovist(room, (5.0, 3.0, 1e-4), 0.0, eps=1e-6)
    assert iso.area == pytest.approx(27.0, rel=1e-12)


def test_radial_over_a_field_spans_both_edges():
    # Reference: from the centre of a 2 x 2 square, the rays at -45, 0 and 45 degrees meet the
    # walls at sqrt 2, 1 and sqrt 2.
    iso = isovist(Plan(rect(-1, -1, 1, 1)), (0, 0), fov=math.pi / 2, direction=0.0)
    ang, r = iso.radial(3)
    np.testing.assert_allclose(ang, [-math.pi / 4, 0.0, math.pi / 4], atol=1e-15)
    np.testing.assert_allclose(r, [math.sqrt(2), 1.0, math.sqrt(2)], rtol=1e-12)


@pytest.mark.parametrize("step", [math.radians(0.5), 0.3, 1.0])
def test_polygon_samples_range_arcs_every_arc_step(step):
    # Reference: with no wall in range, the ring is the regular polygon inscribed in the range
    # circle, with ceil(2 pi / step) sides, of area (k / 2) sin(2 pi / k) R^2.
    iso = isovist(Plan(rect(-10, -10, 10, 10)), (0, 0), max_distance=2.0)
    ring = iso.polygon(step)
    x, y = ring[:, 0], ring[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    np.testing.assert_allclose(ring[0], ring[-1], atol=1e-12)  # the ring repeats its start
    k = len(ring) - 1
    assert k == math.ceil(2 * math.pi / step)
    assert area == pytest.approx(k / 2 * math.sin(2 * math.pi / k) * 4.0, rel=1e-12)


# --- Unity outputs ----------------------------------------------------------------------------

def test_unity_volumes_swap_the_height_into_z(tmp_path):
    # Reference: the Unity output convention, y is the eye height and z the plan y.
    pd.DataFrame({"x": [1.0, 2.0], "y": [1.7, 0.8], "z": [5.0, 6.0],
                  "volume": [10.0, 0.0]}).to_csv(tmp_path / "v.csv", index=False)
    df = read_unity_volumes(tmp_path / "v.csv")
    assert list(df.columns) == ["x", "y", "z", "volume"]
    assert df.to_numpy().tolist() == [[1.0, 5.0, 1.7, 10.0], [2.0, 6.0, 0.8, 0.0]]
