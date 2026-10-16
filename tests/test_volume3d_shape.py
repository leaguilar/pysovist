"""Shape metrics of view volumes against independent references.

Each test names its reference: a closed form, the exact 2D isovist of the plan of a prism room
(a route that shares no code with ray casting on the sphere), shadow faces built in the test, or
an integral evaluated by scipy.
"""

import math

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")
pytest.importorskip("numba")

from scipy import stats  # noqa: E402
from scipy.integrate import dblquad  # noqa: E402

from pysovist import (  # noqa: E402
    SHAPE_METRIC_NAMES,
    Mesh,
    Plan,
    PointCloud,
    angle_grid,
    fibonacci_sphere,
    isovist,
    view_volume,
    view_volume_field,
)
from pysovist.volume3d import FACTOR_NAMES, VOLUME_METRIC_NAMES  # noqa: E402

from .geometry import polygon_walls, rect  # noqa: E402

H = 3.0  # height of the prism rooms
PILLAR = np.concatenate([rect(-5, -5, 5, 5), rect(-1, -1, 1, 1)])
L_ROOM = polygon_walls([(0, 0), (10, 0), (10, 2), (2, 2), (2, 10), (0, 10)])
THIN = np.concatenate([rect(-5, -5, 5, 5), rect(-0.15, -0.15, 0.15, 0.15)])
PRISM_VIEWS = [  # plan, eye: each sees an occluding edge
    (PILLAR, (3.0, 0.0, 1.2)),
    (PILLAR, (2.7, 3.1, 1.7)),
    (L_ROOM, (1.0, 6.0, 1.2)),
    (THIN, (3.0, 0.0, 1.2)),
]


def far_triangle():
    """A scene whose only triangle lies 100 m away, beyond every range used here."""
    return Mesh([(100.0, 0.0, 0.0), (100.0, 1.0, 0.0), (100.0, 0.0, 1.0)], [(0, 1, 2)])


def angle_gap(a, b):
    """Absolute difference of two angles in radians, across the cut at +-pi."""
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def box_room(x1, y1, z1):
    """The closed box [0, x1] x [0, y1] x [0, z1], extruded from its plan."""
    return Mesh.from_plan(rect(0, 0, x1, y1), floor=0.0, ceiling=z1)


def cube(a=1.0):
    m = o3d.geometry.TriangleMesh.create_box(2 * a, 2 * a, 2 * a)
    m.translate((-a, -a, -a))
    return Mesh(np.asarray(m.vertices), np.asarray(m.triangles))


# --- the entry point --------------------------------------------------------------------------

def test_shape_metrics_follow_the_volume_metrics_and_leave_them_unchanged():
    # Reference: the documented key order, and the plain call, whose values must not move.
    scene, eye = far_triangle(), (0.0, 0.0, 0.0)
    plain = view_volume(scene, eye, n_rays=4096, max_distance=2.0)
    shaped = view_volume(scene, eye, n_rays=4096, max_distance=2.0, shape=True)
    assert tuple(plain.metrics) == VOLUME_METRIC_NAMES
    assert tuple(shaped.metrics) == VOLUME_METRIC_NAMES + SHAPE_METRIC_NAMES
    for k in VOLUME_METRIC_NAMES:
        assert shaped.metrics[k] == plain.metrics[k] or (
            math.isnan(shaped.metrics[k]) and math.isnan(plain.metrics[k]))


# --- closed forms -----------------------------------------------------------------------------

def test_range_ball_seen_from_its_centre():
    # Reference: every ray is clipped at R, so the view volume is the ball of radius R. Closed
    # forms: S = 4 pi R^2, sphericity 1, centroid at the eye, no occluding face, no convex
    # deficiency, elongation 1, and a constant depth with zero spread.
    R = 2.0
    m = view_volume(far_triangle(), (0.0, 0.0, 0.0), n_rays=16384, max_distance=R,
                    shape=True).metrics
    assert m["surface_area"] == pytest.approx(4 * math.pi * R**2, rel=1e-3)
    assert m["occluding_area"] == 0.0
    assert m["sphericity"] == pytest.approx(1.0, abs=1e-3)
    assert m["jaggedness"] == pytest.approx(36 * math.pi, rel=1e-3)
    assert m["drift"] < 1e-3 * R
    assert m["elongation"] == pytest.approx(1.0, abs=1e-3)
    assert m["elongation_12"] == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= m["convex_deficiency"] < 1e-12
    for k in ("r_var", "r_skew", "r_mad", "dispersion"):
        assert m[k] == 0.0


def test_box_seen_from_an_interior_eye():
    # Reference: a closed box is star-shaped from any interior eye, so the view volume is the
    # box. Closed forms for a 6 x 4 x 3 m box: S = 2 (24 + 18 + 12) = 108 m2, centroid at the
    # box centre (3, 2, 1.5), principal second moments V a^2 / 12 for the sides a = 6, 4, 3, so
    # elongation 6 / 3 and elongation_12 6 / 4, no occluding face and no convex deficiency.
    eye = np.array([1.5, 1.2, 1.1])
    m = view_volume(box_room(6.0, 4.0, 3.0), eye, n_rays=65536, shape=True).metrics
    c = np.array([3.0, 2.0, 1.5]) - eye
    assert m["surface_area"] == pytest.approx(108.0, rel=1e-2)
    assert m["occluding_area"] == 0.0
    assert m["drift"] == pytest.approx(np.linalg.norm(c), rel=1e-4)
    assert m["drift_horizontal"] == pytest.approx(math.hypot(c[0], c[1]), rel=1e-4)
    assert m["drift_vertical"] == pytest.approx(c[2], rel=1e-3)
    assert m["drift_azimuth"] == pytest.approx(math.atan2(c[1], c[0]), abs=1e-4)
    assert m["drift_elevation"] == pytest.approx(math.atan2(c[2], math.hypot(c[0], c[1])),
                                                 abs=1e-4)
    assert m["elongation"] == pytest.approx(2.0, rel=1e-3)
    assert m["elongation_12"] == pytest.approx(1.5, rel=1e-3)
    assert 0.0 <= m["convex_deficiency"] < 1e-3
    assert m["sphericity"] == pytest.approx(36 * math.pi * 72.0**2 / 108.0**3, rel=3e-2)
    assert m["sphericity"] * m["jaggedness"] == pytest.approx(36 * math.pi, rel=1e-12)


# --- radial statistics ------------------------------------------------------------------------

def _cube_face_integral(f, a=1.0):
    """Integral over the sphere of f(r) for the depth r seen from the centre of [-a, a]^3.

    A face at distance a contributes int f(rho) a / rho^3 dA, rho^2 = x^2 + y^2 + a^2.
    """
    def g(y, x):
        rho = math.sqrt(x * x + y * y + a * a)
        return f(rho) * a / rho**3

    return 6 * dblquad(g, -a, a, -a, a, epsabs=1e-12, epsrel=1e-12)[0]


@pytest.fixture(scope="module")
def cube_moments():
    """Mean, variance, skewness and mean absolute deviation of the depth over the sphere."""
    m1, m2, m3 = (_cube_face_integral(lambda r, k=k: r**k) / (4 * math.pi) for k in (1, 2, 3))
    var = m2 - m1 * m1
    mad = _cube_face_integral(lambda r: abs(r - m1)) / (4 * math.pi)
    return {"mean": m1, "var": var, "skew": (m3 - 3 * m1 * m2 + 2 * m1**3) / var**1.5,
            "mad": mad}


@pytest.mark.parametrize("how", ["fibonacci", "grid"])
def test_radial_statistics_from_the_centre_of_a_cube(cube_moments, how):
    # Reference: scipy's integrals of r, r^2, r^3 and |r - mean| over the faces of the cube,
    # for equal-area directions and for an angle grid that carries its cell solid angles.
    if how == "fibonacci":
        vv = view_volume(cube(), (0, 0, 0), n_rays=65536, shape=True)
    else:
        d, w = angle_grid(512, 256)
        vv = view_volume(cube(), (0, 0, 0), directions=d, weights=w, shape=True)
    m, ref = vv.metrics, cube_moments
    assert m["r_var"] == pytest.approx(ref["var"], rel=2e-3)
    assert m["r_skew"] == pytest.approx(ref["skew"], rel=2e-3)
    assert m["r_mad"] == pytest.approx(ref["mad"], rel=2e-3)
    assert m["dispersion"] == pytest.approx(math.sqrt(ref["var"]) / ref["mean"], rel=2e-3)


def test_radial_statistics_of_equal_weights_are_the_population_statistics():
    # Reference: numpy's variance and scipy's population skewness of the per-ray depths, an
    # independent implementation, and the r_std column, whose square r_var is.
    vv = view_volume(box_room(6.0, 4.0, 3.0), (1.5, 1.2, 1.1), n_rays=4096, shape=True)
    r, m = vv.distances, vv.metrics
    assert m["r_var"] == pytest.approx(np.var(r), rel=1e-12)
    assert m["r_var"] == pytest.approx(m["r_std"] ** 2, rel=1e-12)
    assert m["r_skew"] == pytest.approx(stats.skew(r, bias=True), rel=1e-9)
    assert m["r_mad"] == pytest.approx(np.mean(np.abs(r - r.mean())), rel=1e-12)
    assert m["dispersion"] == pytest.approx(np.std(r) / np.mean(r), rel=1e-12)


def test_weighted_r_var_is_the_square_of_the_weighted_r_std():
    # Reference: the r_std column of a directed view, weighted by solid angle.
    d, w = angle_grid(64, 32)
    vv = view_volume(box_room(6.0, 4.0, 3.0), (1.5, 1.2, 1.1), directions=d, weights=w,
                     shape=True)
    assert vv.metrics["r_var"] == pytest.approx(vv.metrics["r_std"] ** 2, rel=1e-12)


def test_a_ball_sampled_box_room_lies_between_its_erosions():
    # Reference: closed forms of the 6 x 4 x 3 m box, sampled every s = 4 cm by balls of radius
    # r = 3 cm, through the ball ray caster, which shares no code with the mesh caster. The balls
    # seal a surface that lies between the box eroded by r and the box eroded by
    # sqrt(r^2 - s^2 / 2), and both erosions keep the box centre. The surface through the ray
    # ends falls up to 1 % short, as for a mesh. Once neighbouring rays come closer than the
    # spacing of the balls, the grooves between them count as occluding faces: none at 16,384
    # rays, under 2 % of the surface at 65,536.
    s, r = 0.04, 0.03
    size = np.array([6.0, 4.0, 3.0])
    axes = [np.linspace(0, a, int(round(a / s)) + 1) for a in size]
    faces = []
    for k in range(3):
        i, j = [q for q in range(3) if q != k]
        u, v = (g.ravel() for g in np.meshgrid(axes[i], axes[j], indexing="ij"))
        for level in (0.0, size[k]):
            p = np.empty((len(u), 3))
            p[:, i], p[:, j], p[:, k] = u, v, level
            faces.append(p)
    cloud = PointCloud(np.unique(np.concatenate(faces), axis=0), radius=r)
    eye = np.array([1.5, 1.2, 1.1])

    def eroded(e):
        a, b, c = size - 2 * e
        return 2 * (a * b + b * c + c * a), a / c, a / b

    outer, inner = eroded(r), eroded(math.sqrt(r * r - s * s / 2))
    m = view_volume(cloud, eye, n_rays=65536, shape=True).metrics
    assert m["drift"] == pytest.approx(np.linalg.norm(size / 2 - eye), rel=1e-3)
    assert outer[0] * (1 - 1e-2) < m["surface_area"] < inner[0]
    assert inner[1] * (1 - 1e-3) < m["elongation"] < outer[1] * (1 + 1e-3)
    assert inner[2] * (1 - 1e-3) < m["elongation_12"] < outer[2] * (1 + 1e-3)
    assert 0 < m["occluding_area"] < 2e-2 * m["surface_area"]
    assert view_volume(cloud, eye, n_rays=16384, shape=True).metrics["occluding_area"] == 0.0


# --- prism rooms against the exact 2D isovist -------------------------------------------------

@pytest.mark.parametrize("plan, eye", PRISM_VIEWS)
def test_prism_room_against_the_exact_2d_isovist(plan, eye):
    # Reference: in a prism room of height H every eye between floor and ceiling sees the
    # extrusion of its exact 2D isovist (area A, perimeter P, occlusivity O, centroid c). So the
    # boundary is the floor, the ceiling and H times the 2D boundary, S = 2 A + H P, the occluding
    # faces are the occluding edges times H, the centroid lies at c and mid-height, and the hull
    # is the extruded 2D hull. The exact isovist shares no code with ray casting on the sphere.
    # The surface through the ray ends cuts the edges where walls meet floor and ceiling, which
    # costs 1.1 % at 65,536 rays and falls as N^(-1/2) (see the convergence test).
    iso = isovist(Plan(plan), eye[:2]).metrics
    m = view_volume(Mesh.from_plan(plan, 0.0, H), eye, n_rays=65536, shape=True).metrics
    assert m["surface_area"] == pytest.approx(2 * iso["area"] + H * iso["perimeter"], rel=1.5e-2)
    assert m["occluding_area"] == pytest.approx(H * iso["occlusivity"], rel=1e-2)
    assert m["drift_horizontal"] == pytest.approx(iso["drift"], rel=1e-3)
    assert angle_gap(m["drift_azimuth"], iso["drift_angle"]) < 1e-3
    assert m["drift_vertical"] == pytest.approx(H / 2 - eye[2], abs=1e-3)
    assert m["convex_deficiency"] == pytest.approx(iso["convex_deficiency"], abs=2e-3)


def test_l_room_seen_whole_from_the_square_of_both_arms():
    # Reference: from the square where the arms of the L overlap the eye sees both boxes, 4 m
    # wide arms of 10 m, so V = 64 H, and the hull of the L is the 10 x 10 square without the
    # 6 x 6 / 2 corner triangle, 82 H: convex deficiency (82 - 64) / 82. No wall plane passes
    # nearer to the eye than its clearance, so no edge is flagged and the occluding area is 0.
    plan = polygon_walls([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])
    m = view_volume(Mesh.from_plan(plan, 0.0, H), (2.0, 2.0, 1.5), n_rays=65536,
                    shape=True).metrics
    assert m["volume"] == pytest.approx(64 * H, rel=1e-4)
    assert m["convex_deficiency"] == pytest.approx(18 / 82, abs=1e-3)
    assert m["occluding_area"] == 0.0


def _first_wall_hit(origin, direction, walls):
    """Distance along a 2D ray to the first of the wall segments (closed form per segment)."""
    best = np.inf
    for (ax, ay), (bx, by) in walls:
        ex, ey = bx - ax, by - ay
        den = direction[0] * ey - direction[1] * ex
        if abs(den) < 1e-15:
            continue
        qx, qy = ax - origin[0], ay - origin[1]
        t = (qx * ey - qy * ex) / den
        u = (qx * direction[1] - qy * direction[0]) / den
        if t > 1e-12 and -1e-12 <= u <= 1 + 1e-12:
            best = min(best, t)
    return best


@pytest.mark.parametrize("eye", [(3.0, 0.4, 1.2), (-2.5, 3.0, 2.0)])
def test_pillar_shadow_faces(eye):
    # Reference: the shadow faces built here. A full-height pillar hides the room behind it
    # along the two vertical edges whose sides the eye cannot both see. Each edge casts a
    # vertical rectangle from the edge to the first wall behind it along the ray in plan,
    # H high. The silhouette corners are found by the side of each face the eye stands on.
    e = np.asarray(eye[:2])
    corners = [np.array(c, float) for c in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    normals = [np.array(v, float) for v in ((1, 0), (0, 1), (-1, 0), (0, -1))]
    faces = list(zip(corners, corners[1:] + corners[:1], strict=True))  # counter-clockwise
    seen = [np.dot(e - a, nrm) > 0 for (a, _), nrm in zip(faces, [(0, 1), (-1, 0), (0, -1),
                                                                 (1, 0)], strict=True)]
    del normals
    room = rect(-5, -5, 5, 5)
    total = 0.0
    for k, c in enumerate(corners):  # corner k joins face k - 1 and face k
        if seen[k - 1] != seen[k]:
            d = (c - e) / np.linalg.norm(c - e)
            total += H * (_first_wall_hit(e, d, room) - np.linalg.norm(c - e))
    m = view_volume(Mesh.from_plan(PILLAR, 0.0, H), eye, n_rays=65536, shape=True).metrics
    assert total > 0
    assert m["occluding_area"] == pytest.approx(total, rel=1e-2)


def test_cube_in_empty_space_with_a_range_limit():
    # Reference: closed forms. An eye 2 m in front of a face of the cube [-1, 1]^3 sees only
    # that face, whose four edges cast planar shadow faces out to the range sphere R. An edge
    # of half-length a at distance p from the eye spans the face int (R^2 - p^2 / cos^2 phi) / 2
    # dphi over |phi| <= atan(a / p), that is R^2 atan(a / p) - p a. The face covers the solid
    # angle 4 atan(a^2 / (d sqrt(d^2 + 2 a^2))) at distance d, and every other ray is clipped
    # at R. The centroid lies on the axis, below the eye.
    a, d, R = 1.0, 2.0, 6.0
    p = math.hypot(a, d)
    occluding = 4 * (R**2 * math.atan(a / p) - p * a)
    omega = 4 * math.atan(a * a / (d * math.sqrt(d * d + 2 * a * a)))
    volume = R**3 / 3 * (4 * math.pi - omega) + (2 * a) ** 2 * d / 3
    surface = (2 * a) ** 2 + R**2 * (4 * math.pi - omega) + occluding
    m = view_volume(cube(a), (0.0, 0.0, -a - d), n_rays=65536, max_distance=R,
                    shape=True).metrics
    assert m["volume"] == pytest.approx(volume, rel=1e-4)
    assert m["occluding_area"] == pytest.approx(occluding, rel=1e-2)
    assert m["surface_area"] == pytest.approx(surface, rel=1e-2)
    assert m["drift_elevation"] == pytest.approx(-math.pi / 2, abs=1e-3)
    assert m["drift_horizontal"] < 1e-3 * m["drift"]


# --- convergence with the ray count -----------------------------------------------------------

def test_occluding_area_converges_with_the_ray_count():
    # Reference: H times the exact 2D occlusivity of the prism rooms, as above.
    bounds = {4096: 0.10, 16384: 0.015, 65536: 0.006, 262144: 0.006}
    for plan, eye in PRISM_VIEWS:
        exact = H * isovist(Plan(plan), eye[:2]).metrics["occlusivity"]
        mesh = Mesh.from_plan(plan, 0.0, H)
        for n, bound in bounds.items():
            m = view_volume(mesh, eye, n_rays=n, shape=True).metrics
            assert abs(m["occluding_area"] / exact - 1) < bound, (eye, n)


def test_surface_area_converges_with_the_ray_count():
    # Reference: 2 A + H P of the prism rooms, from their exact 2D isovists, and 108 m2 for the
    # box. The surface through the ray ends cuts the edges where two surfaces meet, so it falls
    # short, and its error falls as N^(-1/2): measured slopes -0.48 to -0.51, and 0.29 % to
    # 0.62 % short at 262,144 rays.
    ns = np.array([4096, 16384, 65536, 262144])
    views = [(rect(0, 0, 6, 4), (1.5, 1.2, 1.1)), *PRISM_VIEWS]
    for plan, eye in views:
        iso = isovist(Plan(plan), eye[:2]).metrics
        exact = 2 * iso["area"] + H * iso["perimeter"]
        mesh = Mesh.from_plan(plan, 0.0, H)
        err = np.array([view_volume(mesh, eye, n_rays=n, shape=True)
                        .metrics["surface_area"] / exact - 1 for n in ns])
        slope = np.polyfit(np.log(ns), np.log(-err), 1)[0]
        assert np.all(err < 0), (eye, err)
        assert -0.6 < slope < -0.4, (eye, err)
        assert -err[-1] < 8e-3, (eye, err)


@pytest.mark.parametrize("n", [4096, 16384, 65536])
def test_a_corridor_has_no_occluding_face(n):
    # Reference: a convex room has no occluding face. Every surface of the 40 x 2 x 3 m corridor
    # lies at least the clearance (1 m) from the eye, and one plane at distance p changes 1/r by
    # at most |u_i - u_j| / p between two rays, so no edge is flagged, however grazing the far
    # walls are.
    m = view_volume(box_room(40.0, 2.0, 3.0), (1.0, 1.0, 1.6), n_rays=n, shape=True).metrics
    assert m["occluding_area"] == 0.0
    assert m["convex_deficiency"] < 1e-3


def test_a_distant_jump_appears_once_the_rays_resolve_it():
    # Reference: H times the 2D occlusivity of a 60 x 40 x 6 m hall with a 2 m pillar 19 m from
    # the eye. From the pillar to the wall behind it 1/r changes by 1/19 - 1/50, which two
    # neighbouring rays resolve once they are closer than that times the clearance, 1.6 m:
    # at 4,096 rays they are not, at 262,144 the occluding area lies within 2 %.
    plan, eye, h = np.concatenate([rect(-30, -20, 30, 20), rect(-1, -1, 1, 1)]), (20, 0, 1.6), 6
    exact = h * isovist(Plan(plan), eye[:2]).metrics["occlusivity"]
    mesh = Mesh.from_plan(plan, 0.0, h)
    assert view_volume(mesh, eye, n_rays=4096, shape=True).metrics["occluding_area"] == 0.0
    m = view_volume(mesh, eye, n_rays=262144, shape=True).metrics
    assert m["occluding_area"] == pytest.approx(exact, rel=2e-2)


# --- directed views ---------------------------------------------------------------------------

SECTOR_FIELDS = [  # hfov, vfov, direction, elevation (radians)
    (math.radians(120), math.radians(60), 0.3, 0.1),
    (math.radians(90), math.pi, 2.0, 0.0),
    (2 * math.pi, math.radians(40), 0.0, -0.4),
    (math.radians(200), math.radians(70), 3.0, -0.2),  # straddles azimuth +-pi
]


def sector(R, hfov, vfov, direction, elevation):
    """Volume, surface area and centroid of the part of the ball of radius R inside a field.

    The field's side faces are two vertical circular sectors of angle vfov, unless the field
    is a full turn, and two cones at the elevations e0 and e1 of its edges, of area
    R^2 hfov cos(e) / 2. The centroid is (1 / V) int R^4 / 4 u domega.
    """
    e0, e1 = elevation - vfov / 2, elevation + vfov / 2
    omega = hfov * (math.sin(e1) - math.sin(e0))
    sides = 0.0 if hfov >= 2 * math.pi else R**2 * vfov
    cones = R**2 * hfov * (math.cos(e0) + math.cos(e1)) / 2
    volume = R**3 * omega / 3
    p0, p1 = direction - hfov / 2, direction + hfov / 2
    c2 = (e1 - e0) / 2 + (math.sin(2 * e1) - math.sin(2 * e0)) / 4  # int cos^2 e de
    first = np.array([(math.sin(p1) - math.sin(p0)) * c2, (math.cos(p0) - math.cos(p1)) * c2,
                      hfov * (math.sin(e1) ** 2 - math.sin(e0) ** 2) / 2])
    return volume, R**2 * omega + sides + cones, R**4 / 4 * first / volume


@pytest.mark.parametrize("field", SECTOR_FIELDS)
@pytest.mark.parametrize("how", ["fibonacci", "grid"])
def test_directed_view_of_an_empty_scene_is_a_spherical_sector(field, how):
    # Reference: the closed forms of sector(). Every ray is clipped at R, so the view is the
    # part of the ball inside the field, closed by its side faces. The surface ends at the
    # outermost rays, half a ray spacing inside the field's edges.
    R = 2.0
    hfov, vfov, direction, elevation = field
    kw = dict(hfov=hfov, vfov=vfov, direction=direction, elevation=elevation)
    if how == "fibonacci":
        vv = view_volume(far_triangle(), (0, 0, 0), n_rays=1 << 16, max_distance=R, shape=True,
                         **kw)
    else:
        d, w = angle_grid(360, 180, **kw)
        vv = view_volume(far_triangle(), (0, 0, 0), directions=d, weights=w, max_distance=R,
                         shape=True, **kw)
    volume, surface, centroid = sector(R, *field)
    m = vv.metrics
    assert m["volume"] == pytest.approx(volume, rel=1e-3)
    assert m["surface_area"] == pytest.approx(surface, rel=2e-2)
    assert m["occluding_area"] == 0.0
    assert m["drift"] == pytest.approx(np.linalg.norm(centroid), rel=2e-3)
    assert m["drift_vertical"] == pytest.approx(centroid[2], abs=2e-3 * R)
    assert m["drift_horizontal"] == pytest.approx(math.hypot(*centroid[:2]), abs=2e-3 * R)
    if hfov < 2 * math.pi:  # a full turn puts the centroid on the vertical axis
        assert angle_gap(m["drift_azimuth"], math.atan2(centroid[1], centroid[0])) < 2e-3


def test_a_ground_plane_below_a_raised_view_is_not_occluding():
    # Reference: a single ground plane hides nothing. The view looks 10 degrees up with a 60
    # degree vertical field, so the nearest ground it sees lies 5 m away, but the threshold
    # takes the clearance, 1.7 m, the distance of the ground plane itself.
    ground = Mesh([(-200, -200, 0), (200, -200, 0), (200, 200, 0), (-200, 200, 0)],
                  [(0, 1, 2), (0, 2, 3)])
    vv = view_volume(ground, (0.0, 0.0, 1.7), n_rays=65536, max_distance=50.0, shape=True,
                     hfov=math.radians(120), vfov=math.radians(60), elevation=math.radians(10))
    assert vv.distances.min() > 4.9
    assert vv.metrics["occluding_area"] == 0.0


# --- many eyes --------------------------------------------------------------------------------

def test_field_rows_equal_single_eyes():
    # Reference: view_volume of each eye alone, for the full view and for a directed one with
    # view factors, whose columns come before the shape metrics.
    mesh = Mesh.from_plan(PILLAR, 0.0, H)
    eyes = [(3.0, 0.0, 1.2), (2.7, 3.1, 1.7), (-2.5, 3.0, 2.0)]
    for kw in ({}, dict(hfov=math.radians(150), vfov=math.radians(80), direction=2.5,
                        ground=[0])):
        df = view_volume_field(mesh, eyes, n_rays=4096, shape=True, **kw)
        assert tuple(df.columns[-len(SHAPE_METRIC_NAMES):]) == SHAPE_METRIC_NAMES
        if "ground" in kw:
            factors = df.columns[-len(SHAPE_METRIC_NAMES) - 3:-len(SHAPE_METRIC_NAMES)]
            assert tuple(factors) == FACTOR_NAMES
        single = {k: v for k, v in kw.items() if k != "ground"}
        for row, eye in zip(df.itertuples(), eyes, strict=True):
            m = view_volume(mesh, eye, n_rays=4096, shape=True, **single).metrics
            for k in SHAPE_METRIC_NAMES:
                assert getattr(row, k) == pytest.approx(m[k], rel=1e-12, abs=1e-12), (eye, k)


def test_field_without_shape_keeps_its_columns():
    # Reference: the columns of 0.2.1, which the default call must keep.
    df = view_volume_field(far_triangle(), [(0.0, 0.0, 0.0)], n_rays=256, max_distance=2.0)
    assert not set(SHAPE_METRIC_NAMES) & set(df.columns)


# --- policies ---------------------------------------------------------------------------------

@pytest.mark.parametrize("inside, value", [("nan", math.nan), ("zero", 0.0)])
def test_an_eye_on_a_wall_follows_the_inside_policy(inside, value):
    # Reference: the documented inside policy, NaN or 0 for every metric of a blocked eye.
    mesh, eye = box_room(6.0, 4.0, 3.0), (0.0, 2.0, 1.5)
    m = view_volume(mesh, eye, n_rays=1024, inside=inside, shape=True).metrics
    df = view_volume_field(mesh, [eye, (3.0, 2.0, 1.5)], n_rays=1024, inside=inside, shape=True)
    for k in SHAPE_METRIC_NAMES:
        assert m[k] == value or (math.isnan(value) and math.isnan(m[k]))
        assert df[k][0] == value or (math.isnan(value) and math.isnan(df[k][0]))
        assert np.isfinite(df[k][1])


def open_box():
    """The box [0, 6] x [0, 4] x [0, 3] without its top face."""
    m = o3d.geometry.TriangleMesh.create_box(6.0, 4.0, 3.0)
    v, t = np.asarray(m.vertices), np.asarray(m.triangles)
    top = (v[t][:, :, 2] == 3.0).all(axis=1)
    return Mesh(v, t[~top])


def test_escaping_rays_follow_the_escape_policy():
    # Reference: the documented escape policies. With "nan" the metrics are NaN as the volume
    # is, and with "clip" and no range the surface is infinite and the other metrics NaN.
    eye = (2.0, 1.5, 1.1)
    m = view_volume(open_box(), eye, n_rays=4096, escape="nan", shape=True).metrics
    assert all(math.isnan(m[k]) for k in SHAPE_METRIC_NAMES)
    m = view_volume(open_box(), eye, n_rays=4096, escape="clip", shape=True).metrics
    assert m["surface_area"] == math.inf
    assert all(math.isnan(m[k]) for k in SHAPE_METRIC_NAMES if k != "surface_area")


def test_rays_that_escape_with_zero_depth_end_at_the_eye():
    # Reference: closed form. With escape="zero" the rays through the open top end at the eye,
    # so the view is the box without the pyramid from the eye to the opening. Its four sides
    # are occluding faces, triangles from the eye to the rim edges, and the rest of the
    # boundary is the floor and the four walls, 84 m2.
    eye = np.array([2.0, 1.5, 1.1])
    rim = [((0, 0, 3), (6, 0, 3)), ((6, 0, 3), (6, 4, 3)), ((6, 4, 3), (0, 4, 3)),
           ((0, 4, 3), (0, 0, 3))]
    sides = sum(0.5 * np.linalg.norm(np.cross(np.subtract(a, eye), np.subtract(b, eye)))
                for a, b in rim)
    m = view_volume(open_box(), eye, n_rays=65536, escape="zero", shape=True).metrics
    assert m["volume"] == pytest.approx(72 - 24 * 1.9 / 3, rel=1e-3)
    assert m["occluding_area"] == pytest.approx(sides, rel=1e-2)
    assert m["surface_area"] == pytest.approx(84 + sides, rel=1e-2)


# --- inputs -----------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_jump_factor_must_be_positive(bad):
    with pytest.raises(ValueError, match="jump_factor"):
        view_volume(far_triangle(), (0, 0, 0), n_rays=256, max_distance=2.0, shape=True,
                    jump_factor=bad)
    with pytest.raises(ValueError, match="jump_factor"):
        view_volume_field(far_triangle(), [(0, 0, 0)], n_rays=256, max_distance=2.0, shape=True,
                          jump_factor=bad)


def test_weights_over_part_of_the_sphere_need_the_field_they_cover():
    # The weights of a 90 x 60 degree grid cover 1.45 sr: without the field the surface cannot
    # be closed along its edges, so the call says which field to give.
    field = dict(hfov=math.radians(90), vfov=math.radians(60))
    d, w = angle_grid(90, 60, **field)
    with pytest.raises(ValueError, match="field of view"):
        view_volume(far_triangle(), (0, 0, 0), directions=d, weights=w, max_distance=2.0,
                    shape=True)
    m = view_volume(far_triangle(), (0, 0, 0), directions=d, weights=w, max_distance=2.0,
                    shape=True, **field).metrics
    assert m["surface_area"] == pytest.approx(sector(2.0, field["hfov"], field["vfov"], 0, 0)[1],
                                              rel=3e-2)


def test_directions_that_leave_a_gap_are_refused():
    # Equal weights over directions that cover only the upper half of the sphere would close the
    # surface with one flat lid: the call names the gap instead.
    d = fibonacci_sphere(4096)
    with pytest.raises(ValueError, match="gap"):
        view_volume(far_triangle(), (0, 0, 0), directions=d[d[:, 2] > 0], max_distance=2.0,
                    shape=True)
