"""Parity of pysovist.compat.decodingspaces3d with the DeCodingSpaces Isovist3D component.

Reference data, measured headless with DeCodingSpaces 2021.10.11 in Rhino 7.37:

- ``data/decodingspaces3d/isovist3d_rays.jsonl``: three scenes (a closed 10 x 10 x 3 m
  room, the room with a 1 x 2 x 2 m block, and a 1 x 2 x 2 m block on a 10 x 10 m ground
  mesh), each at 12 x 6 rays over 360 x 180 degrees, 180 x 90 degrees and 180 x 90 degrees
  with a vertical offset of 30 degrees (every ray listed with its hit vector, length and
  surface id), and at 360 x 180 rays over 360 x 180 degrees (volume and hits per mesh).
  View direction (1, 0, 0), range 1000 m, no implicit ground plane.
- ``ROOM_VOLUMES`` and ``HOSPITAL_VOLUMES``: the closed room from (5, 5, 1.5), and the
  hospital plan of ``data/decodingspaces3d/hospital_plan.json`` extruded to 3 m between a
  floor and a ceiling that cover its bounding box grown by 1 m, eye at candidate C04 at
  1.7 m, full view, several ray counts.

The plugin computes in single precision: its directions and lengths carry relative errors
up to 1e-6, its volumes agree to 1e-5 on the simple scenes and to 1e-4 on the hospital,
whose rays cross 183 walls.

Rhino's ``Mesh.CreateFromBox`` builds six quads. The plugin's surface ids follow from
triangles, and the box tests below triangulate the quads as openNURBS does (the first
triangle of each quad in place, the second appended at the end of the face list), with
the bottom face listed last and split along the diagonal from (x1, y0) to (x0, y1).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("open3d")

from pysovist import Plan, isovist  # noqa: E402
from pysovist.compat.decodingspaces3d import Isovist3D, isovist3d, ray_layout  # noqa: E402
from pysovist.mesh import Mesh  # noqa: E402

DATA = Path(__file__).parent / "data" / "decodingspaces3d"
RECORDS = [json.loads(line) for line in (DATA / "isovist3d_rays.jsonl").read_text().splitlines()]
RAY_RECORDS = [r for r in RECORDS if "rays" in r]
WIDE_RECORDS = [r for r in RECORDS if "rays" not in r]

ROOM_VOLUMES = [  # HPr, VPr, HVA, VVA, volume
    (10, 10, 180.0, 90.0, 69.822921752929688),
    (10, 10, 360.0, 180.0, 187.90608215332031),
    (36, 18, 360.0, 180.0, 253.44671630859375),
    (360, 180, 360.0, 180.0, 299.0810546875),
    (720, 360, 360.0, 180.0, 299.74136352539063),
]
HOSPITAL_EYE = (2.1080, 6.2618, 1.7)
HOSPITAL_EXACT = 200.008993
HOSPITAL_VOLUMES = [  # HPr, VPr, volume (360 x 180 degrees)
    (10, 10, 61.123859405517578),
    (36, 18, 114.22423553466797),
    (360, 180, 193.41255187988281),
    (720, 360, 197.84764099121094),
]
HOSPITAL_1440 = 193.23367309570313


# --- scenes as Rhino builds them --------------------------------------------------------------

def rhino_box(x0, y0, z0, x1, y1, z1):
    corners = {(i, j, k): (x, y, z) for i, x in enumerate((x0, x1)) for j, y in enumerate((y0, y1))
               for k, z in enumerate((z0, z1))}
    keys = list(corners)
    v = np.array([corners[k] for k in keys])
    at = {k: n for n, k in enumerate(keys)}
    quads = [  # front, right, back, left, top, bottom: outward normals
        [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
        [(1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)],
        [(0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1)],
        [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        [(1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)],
    ]
    q = [[at[c] for c in quad] for quad in quads]
    first = [(a, b, c) for a, b, c, _ in q]
    second = [(a, c, d) for a, _, c, d in q]
    return Mesh(v, first + second)


def rhino_plane(x0, y0, x1, y1, z=0.0):
    v = np.array([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], dtype=float)
    return Mesh(v, [(0, 1, 2), (0, 2, 3)])


SCENES = {
    "room": ([rhino_box(0, 0, 0, 10, 10, 3)], []),
    "room_with_block": ([rhino_box(0, 0, 0, 10, 10, 3), rhino_box(7, 4, 0, 8, 6, 2)], []),
    "open_with_ground": ([rhino_box(6, 4, 0, 7, 6, 2)], [rhino_plane(0, 0, 10, 10)]),
}


def hospital_meshes():
    segs = np.array(json.loads((DATA / "hospital_plan.json").read_text()), dtype=float)
    v, t = [], []
    for (a, b) in segs:
        n = len(v)
        v += [(a[0], a[1], 0), (b[0], b[1], 0), (b[0], b[1], 3), (a[0], a[1], 3)]
        t += [(n, n + 1, n + 2), (n, n + 2, n + 3)]
    walls = Mesh(np.array(v, dtype=float), t)
    lo, hi = segs.reshape(-1, 2).min(axis=0) - 1, segs.reshape(-1, 2).max(axis=0) + 1
    return [walls, rhino_plane(lo[0], lo[1], hi[0], hi[1], 0.0),
            rhino_plane(lo[0], lo[1], hi[0], hi[1], 3.0)]


def run(record, **kw):
    obstacles, ground = SCENES[record["scene"]]
    return isovist3d(record["eye"], obstacles, record["VD"], ground=ground, view_range=1000.0,
                     horizontal_angle=record["HVA"], vertical_angle=record["VVA"],
                     horizontal_rays=record["HPr"], vertical_rays=record["VPr"],
                     vertical_offset=record["VAO"], ground_at_zero=False, **kw)


def label(r):
    return f"{r['scene']}-{r['HPr']}x{r['VPr']}-{r['HVA']:g}x{r['VVA']:g}+{r['VAO']:g}"


def on_floor_diagonal(record, res):
    """Rays that meet a box floor on the diagonal its two triangles share.

    Which of the two triangles reports such a hit is decided by rounding, in the plugin and
    in Embree alike.
    """
    p = np.asarray(record["eye"]) + res.rays
    floors = [m for m in SCENES[record["scene"]][0] if m.vertices[:, 2].min() == 0.0]
    tie = np.zeros(len(p), dtype=bool)
    for m in floors:
        (x0, y0, _), (x1, y1, _) = m.bounds
        on = np.abs(p[:, 2]) < 1e-6
        on &= np.abs((p[:, 0] - x1) * (y1 - y0) + (p[:, 1] - y0) * (x1 - x0)) < 1e-6
        tie |= on
    return tie


# --- ray layout -------------------------------------------------------------------------------

@pytest.mark.parametrize("hpr, vpr, hva, vva, n", [
    (12, 6, 360, 180, 52), (12, 6, 180, 90, 14), (10, 10, 180, 90, 18), (10, 10, 360, 180, 66),
    (360, 180, 360, 180, 361 * 91), (720, 360, 360, 180, 721 * 181),
])
def test_ray_count(hpr, vpr, hva, vva, n):
    d = ray_layout((1, 0, 0), horizontal_angle=hva, vertical_angle=vva, horizontal_rays=hpr,
                   vertical_rays=vpr)
    assert d.shape[0] * d.shape[1] == n


@pytest.mark.parametrize("record", RAY_RECORDS, ids=label)
def test_ray_directions_and_order(record):
    res = run(record)
    rays = np.array(record["rays"])
    unit = rays / np.linalg.norm(rays, axis=1)[:, None]
    np.testing.assert_allclose(res.directions, unit, atol=1e-6)


def test_layout_turns_with_the_view_direction():
    east = ray_layout((1, 0, 0), horizontal_angle=180, vertical_angle=90, horizontal_rays=12,
                      vertical_rays=6)
    north = ray_layout((0, 2, 0), horizontal_angle=180, vertical_angle=90, horizontal_rays=12,
                       vertical_rays=6)
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    np.testing.assert_allclose(north, east @ rot.T, atol=1e-15)
    # Azimuths fall from the left edge of the field to its right edge, elevations rise.
    az = np.degrees(np.arctan2(east[:, 0, 1], east[:, 0, 0]))
    np.testing.assert_allclose(az, [90, 60, 30, 0, -30, -60, -90], atol=1e-12)
    el = np.degrees(np.arcsin(east[0, :, 2]))
    np.testing.assert_allclose(el, [-45, 15], atol=1e-12)


def test_view_direction_is_required_and_horizontal():
    room = SCENES["room"][0]
    with pytest.raises(TypeError):
        isovist3d((5, 5, 1.5), room)  # the plugin fails without a view direction
    with pytest.raises(ValueError):
        isovist3d((5, 5, 1.5), room, (0, 0, 1))
    with pytest.raises(ValueError):
        isovist3d((5, 5, 1.5), room, (1, 0, 0.5))


# --- per-ray outputs --------------------------------------------------------------------------

@pytest.mark.parametrize("record", RAY_RECORDS, ids=label)
def test_ray_lengths_and_hit_vectors(record):
    res = run(record)
    assert isinstance(res, Isovist3D)
    lengths = np.array(record["lengths"])
    np.testing.assert_allclose(res.lengths, lengths, rtol=2e-6)
    err = np.linalg.norm(res.rays - np.array(record["rays"]), axis=1)
    assert np.all(err <= 2e-6 * lengths)
    # A ray that hits nothing has the full range as its length.
    miss = np.array(record["surface_ids"]) == -1
    assert np.all(res.lengths[miss & (res.lengths > 999)] == 1000.0)


@pytest.mark.parametrize("record", RAY_RECORDS, ids=label)
def test_surface_ids(record):
    res = run(record)
    measured = np.array(record["surface_ids"])
    tie = on_floor_diagonal(record, res)
    np.testing.assert_array_equal(res.surface_ids[~tie], measured[~tie])
    assert set(res.surface_ids[tie]) <= {0, -1} and set(measured[tie]) <= {0, -1}


@pytest.mark.parametrize("record", RAY_RECORDS, ids=label)
def test_hits_per_obstacle_mesh(record):
    res = run(record)
    tie = on_floor_diagonal(record, res)
    np.testing.assert_allclose(res.object_visibility, record["OV"], atol=tie.sum())
    # The hits per mesh are the counts of the surface ids.
    ids = res.surface_ids
    assert res.object_visibility.tolist() == [int((ids == k).sum())
                                              for k in range(len(record["OV"]))]


def test_ground_meshes_continue_the_ids_after_the_obstacles():
    record = next(r for r in RAY_RECORDS if r["scene"] == "open_with_ground")
    res = run(record)
    down = res.directions[:, 2] < 0
    assert set(res.surface_ids[down & (res.lengths < 100)]) == {1}
    assert len(res.object_visibility) == 1


# --- volumes ----------------------------------------------------------------------------------

@pytest.mark.parametrize("record", RECORDS, ids=label)
def test_volume(record):
    res = run(record)
    assert res.volume == pytest.approx(record["volume"][0], rel=1e-5)


@pytest.mark.parametrize("record", WIDE_RECORDS, ids=label)
def test_hits_per_mesh_at_360_x_180_rays(record):
    res = run(record)
    measured = record["OV"]
    if record["scene"] == "room":
        # 437 rays meet the floor on the diagonal that splits it, where single-precision
        # rounding decides the triangle.
        assert abs(res.object_visibility[0] - measured[0]) <= 437
    else:
        np.testing.assert_allclose(res.object_visibility, measured, atol=2)


@pytest.mark.parametrize("hpr, vpr, hva, vva, volume", ROOM_VOLUMES)
def test_closed_room_volumes(hpr, vpr, hva, vva, volume):
    res = isovist3d((5, 5, 1.5), [rhino_box(0, 0, 0, 10, 10, 3)], (1, 0, 0),
                    horizontal_angle=hva, vertical_angle=vva, horizontal_rays=hpr,
                    vertical_rays=vpr, ground_at_zero=False)
    assert res.volume == pytest.approx(volume, rel=1e-5)


def test_hospital_exact_volume_is_height_times_isovist_area():
    segs = np.array(json.loads((DATA / "hospital_plan.json").read_text()))
    area = isovist(Plan(segs), HOSPITAL_EYE[:2]).area
    assert 3.0 * area == pytest.approx(HOSPITAL_EXACT, abs=1e-6)


@pytest.mark.parametrize("hpr, vpr, volume", HOSPITAL_VOLUMES)
def test_hospital_volumes(hpr, vpr, volume):
    res = isovist3d(HOSPITAL_EYE, hospital_meshes(), (1, 0, 0), horizontal_angle=360,
                    vertical_angle=180, horizontal_rays=hpr, vertical_rays=vpr,
                    ground_at_zero=False)
    assert res.volume == pytest.approx(volume, rel=1e-4)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="the plugin returns 2.9 % less than its own layout and "
                   "volume rule give at 1,440 x 720 rays on the hospital plan")
def test_hospital_volume_at_1440_x_720():
    res = isovist3d(HOSPITAL_EYE, hospital_meshes(), (1, 0, 0), horizontal_angle=360,
                    vertical_angle=180, horizontal_rays=1440, vertical_rays=720,
                    ground_at_zero=False)
    assert res.volume == pytest.approx(HOSPITAL_1440, rel=1e-4)


def test_hospital_error_falls_with_the_ray_count_under_the_layout_rule():
    errors = []
    for hpr, vpr in ((360, 180), (720, 360), (1440, 720)):
        res = isovist3d(HOSPITAL_EYE, hospital_meshes(), (1, 0, 0), horizontal_angle=360,
                        vertical_angle=180, horizontal_rays=hpr, vertical_rays=vpr,
                        ground_at_zero=False)
        errors.append(res.volume / HOSPITAL_EXACT - 1)
    assert errors[0] < errors[1] < errors[2] < 0
    assert errors[2] == pytest.approx(-0.0049, abs=5e-4)


# --- implicit ground plane and isovist mesh ---------------------------------------------------

def test_implicit_ground_plane_is_surface_minus_one():
    block = rhino_box(6, 4, 0, 7, 6, 2)
    eye = (3.0, 5.0, 1.5)
    on = isovist3d(eye, [block], (1, 0, 0), horizontal_angle=180, vertical_angle=90,
                   horizontal_rays=12, vertical_rays=6, vertical_offset=30, ground_at_zero=True)
    off = isovist3d(eye, [block], (1, 0, 0), horizontal_angle=180, vertical_angle=90,
                    horizontal_rays=12, vertical_rays=6, vertical_offset=30,
                    ground_at_zero=False)
    down = on.directions[:, 2] < 0
    plane = down & (on.surface_ids == -1)
    np.testing.assert_allclose(on.lengths[plane], 1.5 / -on.directions[plane, 2], rtol=1e-12)
    assert np.all(off.lengths[plane] == 1000.0)
    # The block is hit before the plane, and the hits per obstacle do not count the plane.
    assert on.surface_ids.tolist().count(0) == 1 and on.object_visibility.tolist() == [1]
    assert on.volume < off.volume


def test_a_mesh_floor_at_zero_keeps_its_hits_against_the_implicit_plane():
    record = next(r for r in RAY_RECORDS if r["scene"] == "room_with_block" and r["HVA"] == 360)
    without = run(record)  # measured with ground_at_zero=False
    with_plane = isovist3d(record["eye"], SCENES["room_with_block"][0], record["VD"],
                           horizontal_angle=360, vertical_angle=180, horizontal_rays=12,
                           vertical_rays=6, ground_at_zero=True)
    np.testing.assert_array_equal(with_plane.surface_ids, without.surface_ids)
    np.testing.assert_array_equal(with_plane.lengths, without.lengths)


def test_isovist_mesh_encloses_the_volume():
    record = next(r for r in RAY_RECORDS if r["scene"] == "room_with_block" and r["VAO"] == 30)
    res = run(record, geometry=True)
    assert isinstance(res.mesh, Mesh)
    assert res.mesh.volume == pytest.approx(res.volume, rel=1e-12)
    full = run(next(r for r in RAY_RECORDS if r["scene"] == "room" and r["HVA"] == 360),
               geometry=True)
    assert full.mesh.volume == pytest.approx(full.volume, rel=1e-12)
    assert run(record).mesh is None


def test_volume_is_the_polyhedron_through_the_hit_points():
    # A 12 x 6 view of the closed room from its centre is a 12-sided prism of height 3 whose
    # corners lie 3 cos 30 degrees from the eye: 3 * (3 R^2) with R^2 = 6.75.
    record = next(r for r in RAY_RECORDS if r["scene"] == "room" and r["HVA"] == 360)
    assert run(record).volume == pytest.approx(3 * 3 * 6.75, rel=1e-12)
    assert math.isclose(record["volume"][0], 60.75, rel_tol=1e-6)
