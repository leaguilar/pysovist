"""Directed view volumes, angle-grid directions, per-ray object ids and view factors.

The reference for a box room is the exact wedge volume

    V = 1/3 int_{field} r(omega)^3 d omega,

integrated in closed form over elevation for each azimuth and to 1e-13 by
adaptive quadrature over azimuth (breakpoints at the corners). With the
direction at azimuth phi and elevation theta, the depth is the nearest of a
wall at horizontal distance rho(phi), r = rho / cos(theta), the ceiling,
r = h_up / sin(theta), and the floor, r = h_down / sin(-theta). Their
elevation integrals of r^3 cos(theta) / 3 are rho^3 tan(theta) / 3 and
-h^3 / (6 sin^2(theta)).

The references for view factors are exact solid angles of triangles (Van
Oosterom and Strackee 1983).
"""

import math

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")
pytest.importorskip("numba")

from scipy.integrate import quad  # noqa: E402

from pysovist.directions import (  # noqa: E402
    angle_grid,
    fibonacci_sphere,
    in_field_of_view,
)
from pysovist.mesh import Mesh  # noqa: E402
from pysovist.pointcloud import PointCloud  # noqa: E402
from pysovist.volume3d import (  # noqa: E402
    FACTOR_NAMES,
    VOLUME_METRIC_NAMES,
    view_volume,
    view_volume_field,
)

BOX = (0.0, 0.0, 0.0, 10.0, 8.0, 3.0)
EYE = (3.0, 2.5, 1.2)
FIELDS = [  # hfov, vfov, direction, elevation (radians)
    (math.radians(120), math.radians(60), 0.3, 0.1),
    (math.radians(90), math.pi, 2.0, 0.0),
    (2 * math.pi, math.radians(40), 0.0, -0.4),
    (math.radians(30), math.radians(20), -2.5, 0.6),
    (math.radians(200), math.radians(70), 3.0, -0.2),  # straddles azimuth +-pi
]


# --- references -------------------------------------------------------------------------------

def box_mesh(b=BOX):
    m = o3d.geometry.TriangleMesh.create_box(b[3] - b[0], b[4] - b[1], b[5] - b[2])
    m.translate(b[:3])
    return Mesh(np.asarray(m.vertices), np.asarray(m.triangles))


def _rho(phi, b, e):
    c, s = math.cos(phi), math.sin(phi)
    out = math.inf
    if c > 0:
        out = min(out, (b[3] - e[0]) / c)
    if c < 0:
        out = min(out, (b[0] - e[0]) / c)
    if s > 0:
        out = min(out, (b[4] - e[1]) / s)
    if s < 0:
        out = min(out, (b[1] - e[1]) / s)
    return out


def _elevation_integral(phi, b, e, t0, t1):
    r, up, down = _rho(phi, b, e), b[5] - e[2], e[2] - b[2]
    t_up, t_down = math.atan2(up, r), -math.atan2(down, r)
    v = 0.0
    lo, hi = max(t0, t_down), min(t1, t_up)
    if hi > lo:
        v += r**3 / 3 * (math.tan(hi) - math.tan(lo))
    lo, hi = max(t0, t_up), t1
    if hi > lo:
        v += up**3 / 6 * (1 / math.sin(lo) ** 2 - 1 / math.sin(hi) ** 2)
    lo, hi = t0, min(t1, t_down)
    if hi > lo:
        v += down**3 / 6 * (1 / math.sin(hi) ** 2 - 1 / math.sin(lo) ** 2)
    return v


def box_wedge_volume(hfov, vfov, direction, elevation, b=BOX, e=EYE):
    """Exact view volume of an eye in a box, within a field of view."""
    a0, a1 = direction - hfov / 2, direction + hfov / 2
    corners = [math.atan2(y - e[1], x - e[0]) for x in (b[0], b[3]) for y in (b[1], b[4])]
    pts = sorted(a0 + (c - a0) % (2 * math.pi) for c in corners)
    pts = [p for p in pts if a0 < p < a1] or None
    t0, t1 = elevation - vfov / 2, elevation + vfov / 2
    return quad(_elevation_integral, a0, a1, args=(b, e, t0, t1), points=pts, limit=500,
                epsabs=0, epsrel=1e-13)[0]


def triangle_solid_angle(eye, a, b, c):
    a, b, c = (np.asarray(p, dtype=float) - eye for p in (a, b, c))
    la, lb, lc = (np.linalg.norm(p) for p in (a, b, c))
    num = abs(a @ np.cross(b, c))
    den = la * lb * lc + (a @ b) * lc + (a @ c) * lb + (b @ c) * la
    return 2 * math.atan2(num, den)


def rect_solid_angle(eye, p0, p1, p2, p3):
    return triangle_solid_angle(eye, p0, p1, p2) + triangle_solid_angle(eye, p0, p2, p3)


def quad_mesh(p0, p1, p2, p3):
    return Mesh(np.array([p0, p1, p2, p3], dtype=float), [(0, 1, 2), (0, 2, 3)])


# Ground square at z = 0 (object 0) and a wall above eye height at x = 4 (object 1). The wall
# lies above the eye and the ground below it, so neither hides the other.
GROUND = [(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)]
WALL = [(4, -2, 2), (4, 2, 2), (4, 2, 4), (4, -2, 4)]
FACTOR_EYE = np.array([0.0, 0.0, 1.5])


@pytest.fixture(scope="module")
def two_objects():
    return Mesh.combine([quad_mesh(*GROUND), quad_mesh(*WALL)])


# --- angle grid -------------------------------------------------------------------------------

@pytest.mark.parametrize("field", FIELDS)
def test_angle_grid_weights_are_exact_cell_solid_angles(field):
    hfov, vfov, direction, elevation = field
    d, w = angle_grid(36, 12, hfov=hfov, vfov=vfov, direction=direction, elevation=elevation)
    assert d.shape == (36 * 12, 3) and w.shape == (36 * 12,)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-15)
    lo, hi = elevation - vfov / 2, elevation + vfov / 2
    assert w.sum() == pytest.approx(hfov * (math.sin(hi) - math.sin(lo)), rel=1e-13)
    # Cell (i, j) spans one azimuth step and the elevations between rows j and j + 1.
    edges = lo + vfov * np.arange(13) / 12
    cell = hfov / 36 * (np.sin(edges[1:]) - np.sin(edges[:-1]))
    np.testing.assert_allclose(w.reshape(36, 12), np.broadcast_to(cell, (36, 12)), rtol=1e-12)
    # Every cell centre lies inside the field.
    assert in_field_of_view(d, hfov=hfov, vfov=vfov, direction=direction,
                            elevation=elevation).all()


def test_angle_grid_order_and_centres():
    d, _ = angle_grid(4, 2, hfov=math.pi, vfov=math.pi / 2, direction=0.0, elevation=0.0)
    g = d.reshape(4, 2, 3)
    az = np.arctan2(g[..., 1], g[..., 0])
    el = np.arcsin(g[..., 2])
    # Azimuth-major order: azimuths rise counter-clockwise, elevations rise within each azimuth.
    np.testing.assert_allclose(az[:, 0], np.radians([-67.5, -22.5, 22.5, 67.5]), atol=1e-14)
    np.testing.assert_allclose(el[0], np.radians([-22.5, 22.5]), atol=1e-14)


def test_full_angle_grid_covers_the_sphere():
    _, w = angle_grid(8, 4)
    assert w.sum() == pytest.approx(4 * math.pi, rel=1e-14)


@pytest.mark.parametrize("bad", [
    {"hfov": 0.0}, {"hfov": 7.0}, {"vfov": 0.0}, {"vfov": 3.5},
    {"vfov": math.pi, "elevation": 0.1}, {"vfov": 1.0, "elevation": 1.2},
])
def test_angle_grid_rejects_fields_outside_the_sphere(bad):
    with pytest.raises(ValueError):
        angle_grid(8, 4, **bad)
    with pytest.raises(ValueError):
        in_field_of_view(fibonacci_sphere(10), **bad)


def test_field_of_view_mask():
    d = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 1], [1, 0, -0.5], [-1, 1e-9, 0],
                  [0, 0, -1]])
    d = d / np.linalg.norm(d, axis=1)[:, None]
    assert in_field_of_view(d).all()
    front = in_field_of_view(d, hfov=math.pi / 2, vfov=math.pi / 2)
    assert front.tolist() == [True, False, False, False, True, False, False]
    back = in_field_of_view(d, hfov=math.pi / 2, vfov=math.pi / 2, direction=math.pi)
    assert back.tolist() == [False, False, True, False, False, True, False]
    # Lower edges belong to a field and upper edges to its neighbour, except the top pole.
    down = in_field_of_view(d, vfov=math.pi / 2, elevation=-math.pi / 4)
    assert down.tolist() == [False, False, False, False, True, False, True]
    up = in_field_of_view(d, vfov=math.pi / 2, elevation=math.pi / 4)
    assert up.tolist() == [True, True, True, True, False, True, False]
    left = in_field_of_view(d, hfov=math.pi / 2, direction=math.pi / 4)  # azimuths [0, pi/2)
    assert left.tolist() == [True, False, False, True, True, False, True]


# --- directed view volumes --------------------------------------------------------------------

def test_empty_scene_gives_the_range_sphere_sector_exactly():
    # Every ray runs to the range, so V = R^3 / 3 times the field's solid angle.
    far = Mesh(np.array([[100, 100, 100], [101, 100, 100], [100, 101, 100]], float), [(0, 1, 2)])
    hfov, vfov, direction, elevation = FIELDS[0]
    d, w = angle_grid(40, 20, hfov=hfov, vfov=vfov, direction=direction, elevation=elevation)
    vv = view_volume(far, EYE, directions=d, weights=w, max_distance=5.0)
    omega = hfov * (math.sin(elevation + vfov / 2) - math.sin(elevation - vfov / 2))
    assert vv.volume == pytest.approx(125.0 / 3 * omega, rel=1e-13)
    assert vv.metrics["escape_fraction"] == 1.0
    assert vv.metrics["r_mean"] == pytest.approx(5.0, rel=1e-15)
    assert vv.metrics["r_std"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("field", FIELDS)
def test_box_room_grid_volume_converges_to_the_exact_wedge(field):
    mesh = box_mesh()
    hfov, vfov, direction, elevation = field
    ref = box_wedge_volume(*field)
    d, w = angle_grid(768, 384, hfov=hfov, vfov=vfov, direction=direction, elevation=elevation)
    vv = view_volume(mesh, EYE, directions=d, weights=w)
    assert vv.volume == pytest.approx(ref, rel=2e-5)
    assert vv.metrics["escape_fraction"] == 0.0


@pytest.mark.parametrize("field", FIELDS)
def test_box_room_fibonacci_volume_converges_to_the_exact_wedge(field):
    hfov, vfov, direction, elevation = field
    ref = box_wedge_volume(*field)
    vv = view_volume(box_mesh(), EYE, n_rays=1 << 20, hfov=hfov, vfov=vfov,
                     direction=direction, elevation=elevation)
    assert vv.volume == pytest.approx(ref, rel=1e-4)
    # The rays are the Fibonacci directions inside the field, each carrying 4 pi / N.
    inside = in_field_of_view(fibonacci_sphere(1 << 20), hfov=hfov, vfov=vfov,
                              direction=direction, elevation=elevation)
    assert vv.n_rays == inside.sum()
    np.testing.assert_array_equal(vv.weights, np.full(vv.n_rays, 4 * math.pi / (1 << 20)))


def test_full_box_room_wedge_reference_is_the_box_volume():
    assert box_wedge_volume(2 * math.pi, math.pi, 0.0, 0.0) == pytest.approx(240.0, rel=1e-13)


def test_a_wedge_with_full_vertical_field_is_height_times_the_2d_wedge_area():
    from pysovist import Plan, isovist

    from .geometry import rect

    hfov, direction = math.radians(75), 0.7
    area = isovist(Plan(rect(0, 0, 10, 8)), EYE[:2], fov=hfov, direction=direction).area
    assert box_wedge_volume(hfov, math.pi, direction, 0.0) == pytest.approx(3.0 * area, rel=1e-12)


def test_octants_partition_the_full_view():
    mesh = box_mesh()
    full = view_volume(mesh, EYE, n_rays=4096)
    parts = [view_volume(mesh, EYE, n_rays=4096, hfov=math.pi / 2, vfov=math.pi / 2,
                         direction=math.pi / 4 + k * math.pi / 2, elevation=s * math.pi / 4)
             for k in range(4) for s in (-1, 1)]
    assert sum(p.n_rays for p in parts) == 4096
    assert sum(p.volume for p in parts) == pytest.approx(full.volume, rel=1e-12)
    assert sum(p.metrics["volume_up"] for p in parts) == pytest.approx(
        full.metrics["volume_up"], rel=1e-12)


def test_weighted_moments_over_the_field():
    mesh = box_mesh()
    hfov, vfov, direction, elevation = FIELDS[0]
    d, w = angle_grid(64, 32, hfov=hfov, vfov=vfov, direction=direction, elevation=elevation)
    vv = view_volume(mesh, EYE, directions=d, weights=w)
    r = vv.distances
    mean = (w * r).sum() / w.sum()
    assert vv.metrics["r_mean"] == pytest.approx(mean, rel=1e-13)
    assert vv.metrics["r_std"] == pytest.approx(
        math.sqrt((w * (r - mean) ** 2).sum() / w.sum()), rel=1e-12)
    assert vv.metrics["r_min"] == r.min() and vv.metrics["r_max"] == r.max()
    assert vv.metrics["equivalent_radius"] == pytest.approx(
        (3 * vv.volume / (4 * math.pi)) ** (1 / 3), rel=1e-14)
    assert vv.metrics["volume_up"] + vv.metrics["volume_down"] == pytest.approx(vv.volume,
                                                                                  rel=1e-13)


def test_explicit_default_field_is_bit_identical_to_the_plain_call():
    mesh = box_mesh()
    plain = view_volume(mesh, EYE, n_rays=2048)
    explicit = view_volume(mesh, EYE, n_rays=2048, hfov=2 * math.pi, vfov=math.pi,
                           direction=0.0, elevation=0.0, weights=None)
    for k in VOLUME_METRIC_NAMES:
        assert np.float64(plain.metrics[k]).tobytes() == np.float64(explicit.metrics[k]).tobytes()
    assert plain.object_ids is None


def test_equal_weights_reproduce_the_full_view_to_rounding():
    mesh = box_mesh()
    d = fibonacci_sphere(4096)
    plain = view_volume(mesh, EYE, directions=d)
    weighted = view_volume(mesh, EYE, directions=d, weights=np.full(4096, 4 * math.pi / 4096))
    for k in VOLUME_METRIC_NAMES:
        assert weighted.metrics[k] == pytest.approx(plain.metrics[k], rel=1e-12, abs=1e-15)


def test_escape_policies_in_a_directed_view():
    # Looking out of an open box (no ceiling): the upper rays escape.
    v = np.asarray(box_mesh().vertices)
    t = np.asarray(box_mesh().triangles)
    keep = ~np.all(np.isclose(v[t][:, :, 2], 3.0), axis=1)
    open_box = Mesh(v, t[keep])
    kw = {"n_rays": 8192, "vfov": math.pi / 2, "elevation": math.pi / 4}
    clip = view_volume(open_box, EYE, max_distance=50.0, **kw)
    zero = view_volume(open_box, EYE, escape="zero", **kw)
    nan = view_volume(open_box, EYE, escape="nan", **kw)
    inf = view_volume(open_box, EYE, **kw)
    assert 0 < clip.metrics["escape_fraction"] < 1
    assert clip.metrics["escape_fraction"] == zero.metrics["escape_fraction"]
    assert clip.volume > zero.volume > 0
    assert math.isnan(nan.volume) and not math.isnan(nan.metrics["escape_fraction"])
    assert math.isinf(inf.volume) and inf.flags.unbounded
    # The open sky subtends the solid angle of the ceiling's opening, seen from the eye.
    opening = rect_solid_angle(np.array(EYE), (0, 0, 3), (10, 0, 3), (10, 8, 3), (0, 8, 3))
    assert clip.metrics["escape_fraction"] == pytest.approx(opening / (2 * math.pi), abs=2e-3)


def test_directed_view_rejects_bad_arguments():
    mesh = box_mesh()
    with pytest.raises(ValueError):
        view_volume(mesh, EYE, hfov=7.0)
    with pytest.raises(ValueError):
        view_volume(mesh, EYE, vfov=math.pi, elevation=0.2)
    with pytest.raises(ValueError):
        view_volume(mesh, EYE, directions=fibonacci_sphere(10), weights=np.ones(9))
    with pytest.raises(ValueError):
        view_volume(mesh, EYE, directions=fibonacci_sphere(10), weights=-np.ones(10))
    with pytest.raises(ValueError, match="field of view"):
        view_volume(mesh, EYE, n_rays=8, hfov=0.01, vfov=0.01)


# --- per-ray object ids and view factors ------------------------------------------------------

def test_object_ids_of_two_objects(two_objects):
    vv = view_volume(two_objects, FACTOR_EYE, n_rays=20000, object_ids=True)
    ids = vv.object_ids
    assert ids.dtype.kind == "i" and ids.shape == (vv.n_rays,)
    hit = ids >= 0
    p = FACTOR_EYE + vv.directions[hit] * vv.distances[hit, None]
    assert np.all(np.abs(p[ids[hit] == 0, 2]) < 1e-5)
    assert np.all(np.abs(p[ids[hit] == 1, 0] - 4.0) < 1e-5)
    assert np.all(np.isinf(vv.distances[ids == -1]))
    assert np.array_equal(ids >= 0, vv.hit)
    counts = vv.hits_per_object()
    assert counts.tolist() == [int((ids == 0).sum()), int((ids == 1).sum())]
    assert vv.hits_per_object(minlength=4).tolist() == counts.tolist() + [0, 0]
    omega = vv.solid_angle_per_object()
    np.testing.assert_allclose(omega, counts * 4 * math.pi / 20000, rtol=1e-12)
    # The distances do not depend on whether object ids are asked for.
    plain = view_volume(two_objects, FACTOR_EYE, n_rays=20000)
    np.testing.assert_array_equal(plain.distances, vv.distances)
    for k in VOLUME_METRIC_NAMES:
        np.testing.assert_array_equal(plain.metrics[k], vv.metrics[k])


def test_factors_need_object_ids(two_objects):
    vv = view_volume(two_objects, FACTOR_EYE, n_rays=1000)
    with pytest.raises(ValueError, match="object_ids"):
        vv.factors(ground=[0])


@pytest.mark.parametrize("how", ["fibonacci", "grid"])
def test_view_factors_match_exact_solid_angles(two_objects, how):
    omega_ground = rect_solid_angle(FACTOR_EYE, *GROUND)
    omega_wall = rect_solid_angle(FACTOR_EYE, *WALL)
    if how == "grid":
        d, w = angle_grid(720, 360)
        vv = view_volume(two_objects, FACTOR_EYE, directions=d, weights=w, object_ids=True)
    else:
        vv = view_volume(two_objects, FACTOR_EYE, n_rays=1 << 18, object_ids=True)
    f = vv.factors(ground=[0])
    assert tuple(f) == FACTOR_NAMES
    assert f["ground_factor"] == pytest.approx(omega_ground / (4 * math.pi), abs=5e-4)
    assert f["object_factor"] == pytest.approx(omega_wall / (4 * math.pi), abs=5e-4)
    assert f["sky_factor"] == pytest.approx(1 - (omega_ground + omega_wall) / (4 * math.pi),
                                            abs=5e-4)
    assert sum(f.values()) == pytest.approx(1.0, abs=1e-14)
    assert f["sky_factor"] == pytest.approx(vv.metrics["escape_fraction"], abs=1e-14)
    # A boolean mask over the objects designates the same ground.
    assert vv.factors(ground=np.array([True, False])) == f
    swapped = vv.factors(ground=[1])
    assert swapped["ground_factor"] == pytest.approx(f["object_factor"], abs=1e-15)


def test_view_factors_of_the_lower_hemisphere(two_objects):
    # Looking down, the ground fills the share omega_ground / 2 pi of the field.
    vv = view_volume(two_objects, FACTOR_EYE, n_rays=1 << 18, vfov=math.pi / 2,
                     elevation=-math.pi / 4, object_ids=True)
    f = vv.factors(ground=[0])
    omega_ground = rect_solid_angle(FACTOR_EYE, *GROUND)
    assert f["ground_factor"] == pytest.approx(omega_ground / (2 * math.pi), abs=5e-4)
    assert f["object_factor"] == 0.0


def test_point_cloud_object_ids_are_the_balls_hit():
    centres = np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, -5.0]])
    pc = PointCloud(centres, radius=1.0)
    d = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], [-1, 0, 0]])
    vv = view_volume(pc, (0, 0, 0), directions=d, object_ids=True)
    assert vv.object_ids.tolist() == [0, 1, 2, -1, -1]
    np.testing.assert_allclose(vv.distances[:3], 4.0, rtol=1e-14)
    f = vv.factors(ground=[2])
    assert f == pytest.approx({"sky_factor": 0.4, "ground_factor": 0.2, "object_factor": 0.4},
                              rel=1e-14)


def test_point_cloud_directed_volume_and_sky():
    # Balls far beyond the range: every ray is clipped at R.
    pc = PointCloud(np.array([[500.0, 0, 0]]), radius=0.1)
    hfov, vfov, direction, elevation = FIELDS[3]
    d, w = angle_grid(30, 20, hfov=hfov, vfov=vfov, direction=direction, elevation=elevation)
    vv = view_volume(pc, (0, 0, 0), directions=d, weights=w, max_distance=4.0, object_ids=True)
    assert vv.volume == pytest.approx(64.0 / 3 * w.sum(), rel=1e-13)
    assert vv.factors(ground=[0])["sky_factor"] == 1.0


# --- fields -----------------------------------------------------------------------------------

def test_directed_field_rows_match_single_volumes(two_objects):
    origins = np.array([[0.0, 0, 1.5], [2.0, 1.0, 0.8], [-3.0, 4.0, 2.5]])
    kw = {"n_rays": 4096, "hfov": math.radians(150), "vfov": math.radians(100),
          "direction": 0.4, "elevation": -0.3, "max_distance": 60.0}
    df = view_volume_field(two_objects, origins, ground=[0], **kw)
    for col in FACTOR_NAMES:
        assert col in df.columns
    for i, o in enumerate(origins):
        vv = view_volume(two_objects, o, object_ids=True, **kw)
        for k in VOLUME_METRIC_NAMES:
            assert df[k].iloc[i] == pytest.approx(vv.metrics[k], rel=1e-12, abs=0)
        for k, v in vv.factors(ground=[0]).items():
            assert df[k].iloc[i] == pytest.approx(v, rel=1e-12, abs=1e-15)


def test_field_with_grid_weights_and_without_factors(two_objects):
    d, w = angle_grid(48, 24, hfov=math.pi, vfov=math.pi / 2, direction=1.0)
    origins = np.array([[0.0, 0, 1.5], [1.0, 1.0, 1.0]])
    df = view_volume_field(two_objects, origins, directions=d, weights=w, max_distance=30.0)
    assert not set(FACTOR_NAMES) & set(df.columns)
    for i, o in enumerate(origins):
        vv = view_volume(two_objects, o, directions=d, weights=w, max_distance=30.0)
        assert df["volume"].iloc[i] == pytest.approx(vv.volume, rel=1e-12)


def test_field_factors_are_nan_for_blocked_eyes(two_objects):
    df = view_volume_field(two_objects, np.array([[0.0, 0, 0.0], [0.0, 0, 1.0]]), n_rays=512,
                           ground=[0])
    assert df["on_wall"].tolist() == [True, False]
    assert all(math.isnan(df[k].iloc[0]) for k in FACTOR_NAMES)
    assert df[list(FACTOR_NAMES)].iloc[1].sum() == pytest.approx(1.0, abs=1e-14)


@pytest.mark.slow
def test_worked_example_of_the_documentation():
    # The street of the directed-volumes page: the estimate from 2^18 directions lies 0.05 %
    # above the one from 2^22 directions, 137,406 m3.
    def box(x0, y0, z0, x1, y1, z1):
        v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
        quads = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2),
                 (4, 6, 7, 5)]
        return Mesh(v, [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])

    ground = quad_mesh((-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0))
    scene = Mesh.combine([ground, box(-30, 6, 0, 30, 16, 12), box(-30, -16, 0, 30, -6, 9)])
    field = {"hfov": math.radians(120), "vfov": math.radians(60), "direction": 0.0,
             "elevation": math.radians(10), "max_distance": 100.0}
    coarse = view_volume(scene, (0.0, 0.0, 1.7), n_rays=1 << 18, **field)
    fine = view_volume(scene, (0.0, 0.0, 1.7), n_rays=1 << 22, **field)
    assert coarse.n_rays == 43028 and round(fine.volume) == 137406
    assert round(100 * (coarse.volume / fine.volume - 1), 2) == 0.05
