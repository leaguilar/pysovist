"""Parity of pysovist.compat.decodingspaces3d with the DeCodingSpaces Isovist3D component.

Reference data, measured headless with DeCodingSpaces 2021.10.11 in Rhino 7.37 (range 1000 m,
view direction (1, 0, 0) unless stated):

- ``data/decodingspaces3d/isovist3d_rays.jsonl``: three scenes (a closed 10 x 10 x 3 m room, the
  room with a 1 x 2 x 2 m block, and a 1 x 2 x 2 m block on a 10 x 10 m ground mesh), each at
  12 x 6 rays over 360 x 180 degrees, 180 x 90 degrees and 180 x 90 degrees with a vertical
  offset of 30 degrees (every ray listed with its hit vector, length and surface id), and at
  360 x 180 rays over 360 x 180 degrees (volume and hits per mesh). No implicit ground plane.
- ``data/decodingspaces3d/isovist3d_probe.jsonl``: the vertex and face order of Rhino's box and
  plane meshes as the component receives them; every ray of a single triangle, a room without a
  floor, a room with its faces reversed or triangulated, two boxes on two ground meshes, layouts
  with partial fields and odd ray counts, a tilted view direction, a vertical offset over the
  pole and ray counts from 0 to 5,000 (``caps``); the closed room and the extruded hospital plan
  at full view from 720 x 360 to 2,880 x 1,440 rays (volume, number of rays, hits per mesh); the
  hospital at 20 vantage points, one per solution and all in one solution.
- ``data/decodingspaces3d/isovist3d_counts.jsonl``: the closed room at HPr = 1000 ... 1100 and
  VPr = HPr // 2, full view, with the counts at which the component fails.
- ``data/decodingspaces3d/hospital_plan.json``: the 183 walls of the hospital plan, extruded to
  3 m between a floor and a ceiling that cover its bounding box grown by 1 m, eye at candidate
  C04 at 1.7 m.

The component computes in single precision: its directions and lengths carry relative errors up
to 1e-6. The scenes below are built with the vertex and face order Rhino gives them (measured):
a box has 24 vertices and six quads, the bottom first, and a quad (a, b, c, d) is split into
(a, b, c) and (a, c, d), the first halves in place and the second halves after them.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("open3d")

from pysovist import Plan, isovist  # noqa: E402
from pysovist.compat.decodingspaces3d import (  # noqa: E402
    DEFAULTS,
    FAILING_COUNTS,
    Isovist3D,
    isovist3d,
    ray_layout,
)
from pysovist.mesh import Mesh  # noqa: E402

DATA = Path(__file__).parent / "data" / "decodingspaces3d"


def _lines(name):
    return [json.loads(line) for line in (DATA / name).read_text().splitlines()]


RECORDS = _lines("isovist3d_rays.jsonl")
RAY_RECORDS = [r for r in RECORDS if "rays" in r]
WIDE_RECORDS = [r for r in RECORDS if "rays" not in r]
PROBE = _lines("isovist3d_probe.jsonl")
COUNTS = _lines("isovist3d_counts.jsonl")
FACES = next(r for r in PROBE if r["name"] == "face_order")
TRIANGULATED = next(r for r in PROBE if r["name"] == "triangulated_face_order")

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

def split_quads(quads):
    """Rhino's triangulation: (a, b, c) of every quad in place, then (a, c, d) of every quad."""
    first = [(a, b, c) for a, b, c, _ in quads]
    second = [(a, c, d) for a, _, c, d in quads if c != d]
    return first + second


def rhino_box(x0, y0, z0, x1, y1, z1, faces=None):
    """Mesh.CreateFromBox with the vertex and face order measured on (0, 0, 0)-(10, 10, 3)."""
    v = np.array(FACES["box_vertices"], dtype=float)
    v = np.c_[np.where(v[:, 0] > 0, x1, x0), np.where(v[:, 1] > 0, y1, y0),
              np.where(v[:, 2] > 0, z1, z0)]
    return Mesh(v, split_quads(FACES["box_faces"] if faces is None else faces))


def rhino_plane(x0, y0, x1, y1, z=0.0):
    """Mesh.CreateFromPlane with one face: vertices (x0, y0), (x1, y0), (x0, y1), (x1, y1)."""
    assert FACES["plane_faces"] == [[0, 1, 3, 2]]
    v = np.array([(x0, y0, z), (x1, y0, z), (x0, y1, z), (x1, y1, z)], dtype=float)
    return Mesh(v, split_quads(FACES["plane_faces"]))


ROOM = rhino_box(0, 0, 0, 10, 10, 3)
_BOTTOM = [f for f in FACES["box_faces"]
           if all(FACES["box_vertices"][k][2] == 0.0 for k in f)]
SCENES = {
    "room": ([ROOM], []),
    "room_with_block": ([ROOM, rhino_box(7, 4, 0, 8, 6, 2)], []),
    "open_with_ground": ([rhino_box(6, 4, 0, 7, 6, 2)], [rhino_plane(0, 0, 10, 10)]),
    "one_triangle": ([Mesh([(8, -5, -5), (8, 5, -5), (8, 0, 8)], [(0, 1, 2)])], []),
    "room_without_floor": ([rhino_box(0, 0, 0, 10, 10, 3, faces=[
        f for f in FACES["box_faces"] if f not in _BOTTOM])], []),
    "room_isZero": ([ROOM], []),
    "room_reversed_faces": ([rhino_box(0, 0, 0, 10, 10, 3, faces=FACES["box_faces"][::-1])], []),
    "room_triangulated": ([Mesh(TRIANGULATED["vertices"],
                                [f[:3] for f in TRIANGULATED["faces"]])], []),
    "two_and_two": ([rhino_box(6, 4, 0, 7, 6, 2), rhino_box(0, 8, 0, 2, 9, 3)],
                    [rhino_plane(0, 0, 10, 5), rhino_plane(0, 5, 10, 10)]),
    "layout": ([ROOM], []),
    "layout_vva": ([ROOM], []),
    "tilted_vd": ([ROOM], []),
    "vao_over_pole": ([ROOM], []),
    "caps": ([ROOM], []),
}


def hospital_meshes():
    segs = np.array(json.loads((DATA / "hospital_plan.json").read_text()), dtype=float)
    v, quads = [], []
    for (a, b) in segs:
        n = len(v)
        v += [(a[0], a[1], 0), (b[0], b[1], 0), (b[0], b[1], 3), (a[0], a[1], 3)]
        quads.append((n, n + 1, n + 2, n + 3))
    walls = Mesh(np.array(v, dtype=float), split_quads(quads))
    lo, hi = segs.reshape(-1, 2).min(axis=0) - 1, segs.reshape(-1, 2).max(axis=0) + 1
    return [walls, rhino_plane(lo[0], lo[1], hi[0], hi[1], 0.0),
            rhino_plane(lo[0], lo[1], hi[0], hi[1], 3.0)]


HOSPITAL = hospital_meshes()
SCENES["hospital"] = (HOSPITAL, [])
SCENES["hospital_isZero"] = (HOSPITAL, [])


def run(record, **kw):
    obstacles, ground = SCENES[record.get("scene", record.get("name"))]
    kw.setdefault("ground_at_zero", record.get("isZero", False))
    return isovist3d(record["eye"], obstacles, record["VD"], ground=ground, view_range=1000.0,
                     horizontal_angle=record["HVA"], vertical_angle=record["VVA"],
                     horizontal_rays=record["HPr"], vertical_rays=record["VPr"],
                     vertical_offset=record["VAO"], **kw)


def label(r):
    return (f"{r.get('scene', r.get('name'))}-{r['HPr']}x{r['VPr']}-{r['HVA']:g}x{r['VVA']:g}"
            f"+{r['VAO']:g}{'-isZero' if r.get('isZero') else ''}")


def on_floor_diagonal(record, res):
    """Rays that meet a box floor on the diagonal its two triangles share.

    Which of the two triangles reports such a hit is decided by rounding, in the plugin and
    in Embree alike.
    """
    p = np.asarray(record["eye"]) + res.rays
    obstacles = SCENES[record.get("scene", record.get("name"))][0]
    floors = [m for m in obstacles if m.vertices[:, 2].min() == 0.0]
    tie = np.zeros(len(p), dtype=bool)
    for m in floors:
        (x0, y0, _), (x1, y1, _) = m.bounds
        on = np.abs(p[:, 2]) < 1e-6
        on &= np.abs((p[:, 0] - x1) * (y1 - y0) + (p[:, 1] - y0) * (x1 - x0)) < 1e-6
        tie |= on
    return tie


PROBE_RAYS = [r for r in PROBE if r.get("rays") and r["HPr"] >= 1]
LARGE = [r for r in PROBE + COUNTS
         if r.get("name") in ("room", "hospital", "hospital_isZero") and r.get("volume")]


# --- ray layout -------------------------------------------------------------------------------

@pytest.mark.parametrize("hpr, vpr, hva, vva, n", [
    (12, 6, 360, 180, 52), (12, 6, 180, 90, 14), (10, 10, 180, 90, 18), (10, 10, 360, 180, 66),
    (360, 180, 360, 180, 361 * 91), (720, 360, 360, 180, 721 * 181),
    (1, 6, 360, 180, 8), (12, 1, 360, 180, 13), (5000, 6, 360, 180, 20004),
    (12, 5000, 360, 180, 32513), (10, 6, 100, 90, 6), (7, 6, 180, 90, 8), (12, 7, 180, 100, 14),
])
def test_ray_count(hpr, vpr, hva, vva, n):
    # Reference: the measured number of rays of each layout (records and caps of the probe).
    d = ray_layout((1, 0, 0), horizontal_angle=hva, vertical_angle=vva, horizontal_rays=hpr,
                   vertical_rays=vpr)
    assert d.shape[0] * d.shape[1] == n


@pytest.mark.parametrize("record", RAY_RECORDS + PROBE_RAYS, ids=label)
def test_ray_directions_and_order(record):
    # Reference: the measured hit vectors, every ray in the component's order.
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
    # Reference: a quarter turn of the view direction turns every ray by a quarter turn.
    np.testing.assert_allclose(north, east @ rot.T, atol=1e-6)
    # Azimuths fall from the left edge of the field to its right edge, elevations rise.
    az = np.degrees(np.arctan2(east[:, 0, 1], east[:, 0, 0]))
    np.testing.assert_allclose(az, [90, 60, 30, 0, -30, -60, -90], atol=1e-5)
    el = np.degrees(np.arcsin(east[0, :, 2]))
    np.testing.assert_allclose(el, [-45, 15], atol=1e-5)


def test_a_tilted_view_direction_is_laid_flat():
    # Reference: the component's rays for the view direction (1, 0, 0.5) (record tilted_vd).
    record = next(r for r in PROBE if r["name"] == "tilted_vd")
    res = run(record)
    rays = np.array(record["rays"])
    np.testing.assert_allclose(res.directions, rays / np.linalg.norm(rays, axis=1)[:, None],
                               atol=1e-6)
    flat = ray_layout((1, 0, 0), horizontal_angle=180, vertical_angle=90, horizontal_rays=12,
                      vertical_rays=6)
    np.testing.assert_array_equal(res.directions, flat.reshape(-1, 3))


def test_view_direction_is_required_and_not_vertical():
    with pytest.raises(TypeError):
        isovist3d((5, 5, 1.5), [ROOM])  # the plugin fails without a view direction
    for vd in ((0, 0, 1), (0, 0, -2), (0, 0, 0), (np.nan, 0, 0)):
        with pytest.raises(ValueError, match="view_direction"):
            isovist3d((5, 5, 1.5), [ROOM], vd)


@pytest.mark.parametrize("hpr, vpr", [(0, 6), (-4, 6), (12, 0), (12, -1)])
def test_counts_below_one_are_refused(hpr, vpr):
    # The component casts one column for HPr = 0 (4 rays, volume 0) and fails for HPr = -4.
    with pytest.raises(ValueError, match="at least 1"):
        isovist3d((5, 5, 1.5), [ROOM], (1, 0, 0), horizontal_rays=hpr, vertical_rays=vpr)


@pytest.mark.parametrize("kw", [{"horizontal_rays": 10.5}, {"vertical_rays": 6.2},
                                {"horizontal_angle": 0.0}, {"vertical_angle": -10.0},
                                {"horizontal_angle": 400.0}, {"vertical_angle": 200.0},
                                {"vertical_offset": np.nan}, {"view_range": 0.0},
                                {"view_range": np.inf}, {"view_range": np.nan}])
def test_inputs_outside_the_component_domain_are_refused(kw):
    with pytest.raises(ValueError):
        isovist3d((5, 5, 1.5), [ROOM], (1, 0, 0), **kw)


@pytest.mark.parametrize("eye", [(5, 5), (5, 5, 1.5, 0), [(5, 5, 1.5), (4, 4, 1.5)],
                                 (np.nan, 5, 1.5)])
def test_one_finite_vantage_point(eye):
    with pytest.raises(ValueError, match="vantage point"):
        isovist3d(eye, [ROOM], (1, 0, 0))


@pytest.mark.parametrize("hpr, vpr", sorted(FAILING_COUNTS))
def test_counts_at_which_the_component_fails_are_refused(hpr, vpr):
    with pytest.raises(ValueError, match="fails"):
        isovist3d((5, 5, 1.5), [ROOM], (1, 0, 0), horizontal_angle=360, vertical_angle=180,
                  horizontal_rays=hpr, vertical_rays=vpr)


def test_the_failing_counts_are_those_measured():
    # Reference: the counts that stopped with an error or crashed Rhino, over 360 x 180 degrees.
    failed = {(r["HPr"], r["VPr"]) for r in PROBE + COUNTS
              if r.get("HPr", 0) > 0 and r.get("HVA", 360.0) == 360.0
              and r.get("VVA", 180.0) == 180.0 and (r.get("crashed") or r.get("messages"))}
    assert failed == set(FAILING_COUNTS)
    assert len(FAILING_COUNTS) == 65
    sweep = [r for r in COUNTS if 1000 <= r["HPr"] <= 1100]
    assert len(sweep) == 101 and sum((r["HPr"], r["VPr"]) in FAILING_COUNTS for r in sweep) == 64
    # Where it works, the room's volume moves between 297.24 and 299.90 m3.
    volumes = [r["volume"][0] for r in sweep if r.get("volume")]
    assert (round(min(volumes), 2), round(max(volumes), 2)) == (297.24, 299.90)
    # At other viewing angles the counts were not measured, and the rules apply.
    res = isovist3d((5, 5, 1.5), [ROOM], (1, 0, 0), horizontal_angle=180, vertical_angle=90,
                    horizontal_rays=1000, vertical_rays=500)
    assert res.grid_shape == (501, 126)


def test_defaults_are_the_component_defaults():
    # Reference: the component's input defaults, read from the plugin.
    assert DEFAULTS == {"view_range": 1000.0, "horizontal_angle": 180.0, "vertical_angle": 90.0,
                        "horizontal_rays": 10, "vertical_rays": 10, "vertical_offset": 0.0,
                        "geometry": False, "ground_at_zero": True}
    res = isovist3d((5, 5, 1.5), [ROOM], (1, 0, 0))
    assert res.grid_shape == (6, 3) and res.view_range == 1000.0 and res.mesh is None


# --- per-ray outputs --------------------------------------------------------------------------

@pytest.mark.parametrize("record", RAY_RECORDS + PROBE_RAYS, ids=label)
def test_ray_lengths_and_hit_vectors(record):
    # Reference: the measured length and hit vector of every ray.
    res = run(record)
    assert isinstance(res, Isovist3D)
    lengths = np.array(record["lengths"])
    np.testing.assert_allclose(res.lengths, lengths, rtol=2e-6)
    err = np.linalg.norm(res.rays - np.array(record["rays"]), axis=1)
    assert np.all(err <= 2e-6 * lengths)
    # A ray that hits nothing has the full range as its length.
    miss = np.array(record["surface_ids"]) == -1
    assert np.all(res.lengths[miss & (res.lengths > 999)] == 1000.0)


@pytest.mark.parametrize("record", RAY_RECORDS + PROBE_RAYS, ids=label)
def test_surface_ids(record):
    # Reference: the measured surface id of every ray.
    res = run(record)
    measured = np.array(record["surface_ids"])
    tie = on_floor_diagonal(record, res)
    np.testing.assert_array_equal(res.surface_ids[~tie], measured[~tie])
    assert set(res.surface_ids[tie]) <= {0, -1} and set(measured[tie]) <= {0, -1}


@pytest.mark.parametrize("record", RAY_RECORDS + PROBE_RAYS, ids=label)
def test_hits_per_obstacle_mesh(record):
    # Reference: the measured hits per obstacle mesh (output OV).
    res = run(record)
    tie = on_floor_diagonal(record, res)
    np.testing.assert_allclose(res.object_visibility, record["OV"], atol=tie.sum())
    # The hits per mesh are the counts of the surface ids.
    ids = res.surface_ids
    assert res.object_visibility.tolist() == [int((ids == k).sum())
                                              for k in range(len(record["OV"]))]


def test_only_the_first_triangle_of_the_scene_reports_minus_one():
    # Reference: the measured surface ids. The first triangle of the first obstacle mesh is half
    # of the room's floor (6 rays report -1); with the faces reversed it is half of a wall that
    # no ray meets (no -1); the first triangle of a block or of a ground mesh keeps its id.
    ids = {r["name"]: r["surface_ids"] for r in PROBE_RAYS if not r.get("isZero")}
    assert ids["room_isZero"].count(-1) == 6 and ids["room_triangulated"].count(-1) == 6
    assert ids["room_reversed_faces"].count(-1) == 0
    assert ids["one_triangle"] == [-1] * 14  # its single triangle is hit once, at 8.28 m
    room = SCENES["room"][0][0]
    hit_first = isovist3d((5, 5, 1.5), [room], (1, 0, 0), horizontal_angle=360,
                          vertical_angle=180, horizontal_rays=12, vertical_rays=6)
    p = np.array([5, 5, 1.5]) + hit_first.rays
    a, b, c = room.vertices[room.triangles[0]]
    inside = (p[:, 2] < 1e-6) & (p[:, 0] + p[:, 1] > 10 + 1e-6)  # (0,10), (10,10), (10,0)
    assert inside.sum() == 6
    assert {tuple(x) for x in (a, b, c)} == {(0, 10, 0), (10, 10, 0), (10, 0, 0)}
    assert np.all(hit_first.surface_ids[inside] == -1)


def test_ground_meshes_continue_the_ids_after_the_obstacles():
    record = next(r for r in RAY_RECORDS if r["scene"] == "open_with_ground")
    res = run(record)
    down = res.directions[:, 2] < 0
    assert set(res.surface_ids[down & (res.lengths < 100)]) == {1}
    assert len(res.object_visibility) == 1
    two = next(r for r in PROBE_RAYS if r["name"] == "two_and_two")
    assert set(run(two).surface_ids) == {-1, 2, 3} and run(two).object_visibility.tolist() == [0, 0]


@pytest.mark.parametrize("name", ["room_without_floor", "room_isZero"])
def test_ground_at_zero_changes_nothing(name):
    # Reference: the component gives the same outputs with isZero true and false; in a room
    # without a floor, the rays that look down escape with the full range either way.
    on = next(r for r in PROBE_RAYS if r["name"] == name and r["isZero"])
    off = next(r for r in PROBE_RAYS if r["name"] == name and not r["isZero"])
    assert on["lengths"] == off["lengths"] and on["surface_ids"] == off["surface_ids"]
    a, b = run(on, ground_at_zero=True), run(on, ground_at_zero=False)
    np.testing.assert_array_equal(a.lengths, b.lengths)
    np.testing.assert_array_equal(a.surface_ids, b.surface_ids)
    assert a.volume == b.volume
    if name == "room_without_floor":
        assert np.all(a.lengths[a.directions[:, 2] < -0.1] == 1000.0)


# --- volumes ----------------------------------------------------------------------------------

@pytest.mark.parametrize("record", RECORDS + PROBE_RAYS, ids=label)
def test_volume(record):
    # Reference: the measured volume (output Vol). A grid that encloses nothing (one azimuth or
    # one elevation, or a full turn of one or two steps) gives 0, the component a few 1e-6.
    res = run(record)
    assert res.volume == pytest.approx(record["volume"][0], rel=2e-6, abs=1e-5)


@pytest.mark.parametrize("record", WIDE_RECORDS, ids=label)
def test_hits_per_mesh_at_360_x_180_rays(record):
    # Reference: the measured hits per mesh. In the closed room, rays that meet the floor on the
    # diagonal its two triangles share are decided by rounding, here and in the component.
    res = run(record)
    tol = 5 if record["scene"] == "room" else 0
    np.testing.assert_allclose(res.object_visibility, record["OV"], atol=tol)


def test_volume_sums_unsigned_tetrahedra():
    # Reference: the measured volume over the pole (vertical offset of 30 degrees on a vertical
    # angle of 180 degrees): the cells beyond the pole count with their own size, 54.95 m3,
    # where a signed sum would give 52.70 m3.
    record = next(r for r in PROBE_RAYS if r["name"] == "vao_over_pole")
    res = run(record)
    assert res.volume == pytest.approx(54.949241638183594, rel=1e-6)
    p = res.rays.reshape(*res.grid_shape, 3)
    signed = sum(np.dot(p[i, j], np.cross(p[i + 1, j + 1], p[i + 1, j]))
                 + np.dot(p[i, j], np.cross(p[i, j + 1], p[i + 1, j + 1]))
                 for i in range(p.shape[0] - 1) for j in range(p.shape[1] - 1)) / 6
    assert abs(signed) == pytest.approx(52.69924, rel=1e-5)


@pytest.mark.parametrize("hpr, vpr, hva, vva, volume", ROOM_VOLUMES)
def test_closed_room_volumes(hpr, vpr, hva, vva, volume):
    # Reference: the measured volumes of the closed room from (5, 5, 1.5).
    res = isovist3d((5, 5, 1.5), [ROOM], (1, 0, 0), horizontal_angle=hva, vertical_angle=vva,
                    horizontal_rays=hpr, vertical_rays=vpr, ground_at_zero=False)
    assert res.volume == pytest.approx(volume, rel=5e-6)


def test_hospital_exact_volume_is_height_times_isovist_area():
    # Reference: the extruded-plan identity V = H A with the exact 2D isovist area.
    segs = np.array(json.loads((DATA / "hospital_plan.json").read_text()))
    area = isovist(Plan(segs), HOSPITAL_EYE[:2]).area
    assert 3.0 * area == pytest.approx(HOSPITAL_EXACT, abs=1e-6)


@pytest.mark.parametrize("hpr, vpr, volume", HOSPITAL_VOLUMES)
def test_hospital_volumes(hpr, vpr, volume):
    # Reference: the measured volumes of the hospital plan at C04.
    res = isovist3d(HOSPITAL_EYE, HOSPITAL, (1, 0, 0), horizontal_angle=360,
                    vertical_angle=180, horizontal_rays=hpr, vertical_rays=vpr,
                    ground_at_zero=False)
    assert res.volume == pytest.approx(volume, rel=5e-6)


def test_hospital_volume_at_1440_x_720():
    # Reference: the measured 193.23 m3, 3.4 % below the exact 200.01 m3. The component lists
    # 362 elevations and gives each azimuth 361 of them in turn, so its rays leave the grid.
    res = isovist3d(HOSPITAL_EYE, HOSPITAL, (1, 0, 0), horizontal_angle=360, vertical_angle=180,
                    horizontal_rays=1440, vertical_rays=720, ground_at_zero=False)
    assert res.volume == pytest.approx(HOSPITAL_1440, rel=1e-4)


# The counts whose volume the rules do not reproduce to 2e-4, with the measured difference.
LOOSE = {("room", 2880, 1440): 1e-3, ("hospital", 2880, 1440): 6e-4,
         ("room", 1013, 506): 1e-3}


def large_label(r):
    return f"{r['name']}-{r['HPr']}x{r['VPr']}{'-isZero' if r.get('isZero') else ''}"


@pytest.mark.slow
@pytest.mark.parametrize("record", LARGE, ids=large_label)
def test_large_counts(record):
    # Reference: the measured volume, number of rays and hits per mesh at every count from
    # 720 x 360 to 2,880 x 1,440 where the component returns a result.
    key = (record["name"].replace("_isZero", ""), record["HPr"], record["VPr"])
    res = run(record)
    assert res.volume == pytest.approx(record["volume"][0], rel=LOOSE.get(key, 2e-4))
    if key == ("room", 1003, 501):
        # The component returns one column of rays fewer than its layout, 1,003 of 1,004, and
        # the volume of the whole layout.
        assert len(res.lengths) == record["n_rays"] + 251
    else:
        assert len(res.lengths) == record["n_rays"]
    if key[0] == "hospital":
        # Every ray meets a mesh. Single-precision rounding moves a few rays at the edges.
        assert res.object_visibility.sum() == record["n_rays"]
        assert np.abs(res.object_visibility - record["OV"]).sum() <= 10


def test_fine_vertical_steps_give_each_azimuth_the_next_elevations():
    # 720 vertical steps over 180 degrees list the elevations -90 ... 90.5 (362), and ray k takes
    # elevation k mod 362. The second azimuth therefore starts at 90.5 degrees, beyond the pole.
    d = ray_layout((1, 0, 0), horizontal_angle=360, vertical_angle=180, horizontal_rays=1440,
                   vertical_rays=720)
    assert d.shape == (1441, 361, 3)
    el = np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))
    # Single-precision steps drift by up to 3e-4 degrees over 361 steps.
    np.testing.assert_allclose(el[0], -90 + 0.5 * np.arange(361), atol=5e-4)
    assert d[1, 0, 2] == pytest.approx(math.sin(math.radians(90.5)), abs=1e-6)
    az1 = math.radians(180 - 0.25)
    assert d[1, 0, 0] * math.cos(az1) + d[1, 0, 1] * math.sin(az1) < 0  # beyond the pole
    # Reference: the hits per mesh of the hospital at 1,440 x 720 rays, which this rule gives
    # exactly and the plain grid misses by 1,514 rays.
    res = isovist3d(HOSPITAL_EYE, HOSPITAL, (1, 0, 0), horizontal_angle=360, vertical_angle=180,
                    horizontal_rays=1440, vertical_rays=720)
    np.testing.assert_allclose(res.object_visibility, [216670, 142936, 160595], atol=4)


def test_coarse_counts_keep_the_grid():
    # At steps above 0.6545 degrees no elevation beyond the upper edge is listed: the rays lie
    # on the grid of azimuths and elevations.
    d = ray_layout((1, 0, 0), horizontal_angle=360, vertical_angle=180, horizontal_rays=1100,
                   vertical_rays=550)
    el = np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))
    np.testing.assert_allclose(el, np.broadcast_to(-90 + 360 / 550 * np.arange(276), el.shape),
                               atol=5e-4)


def test_volume_does_not_converge_with_the_ray_count():
    # Reference: the measured hospital volumes at 360 x 180, 720 x 360 and 1,440 x 720 rays.
    errors = []
    for hpr, vpr in ((360, 180), (720, 360), (1440, 720)):
        res = isovist3d(HOSPITAL_EYE, HOSPITAL, (1, 0, 0), horizontal_angle=360,
                        vertical_angle=180, horizontal_rays=hpr, vertical_rays=vpr)
        errors.append(res.volume / HOSPITAL_EXACT - 1)
    np.testing.assert_allclose(errors, [-0.0330, -0.0108, -0.0339], atol=5e-4)


def test_one_vantage_point_per_call_as_the_component_alone_gives():
    # Reference: the volumes of the hospital at 20 vantage points, each solved alone at
    # 360 x 180 rays. Solved together in one solution, the component returns 0 for 13 of them,
    # different ones from one run to the next.
    alone = [r for r in PROBE if r["name"] == "hospital_one_eye"]
    together = [r["volume"] for r in PROBE if r["name"].startswith("hospital_all_eyes")]
    assert len(alone) == 20 and [v.count(0.0) for v in together] == [13, 13]
    assert [v == 0.0 for v in together[0]] != [v == 0.0 for v in together[1]]
    for r in alone:
        res = isovist3d(r["eye"], HOSPITAL, (1, 0, 0), horizontal_angle=360, vertical_angle=180,
                        horizontal_rays=360, vertical_rays=180)
        assert res.volume == pytest.approx(r["volume"][0], rel=1e-4)


# --- isovist mesh -----------------------------------------------------------------------------

def test_isovist_mesh_encloses_the_volume():
    record = next(r for r in RAY_RECORDS if r["scene"] == "room_with_block" and r["VAO"] == 30)
    res = run(record, geometry=True)
    assert isinstance(res.mesh, Mesh)
    # Reference: the signed volume of the closed mesh, computed by Mesh.volume about another
    # point. The two sums differ by rounding.
    assert res.mesh.volume == pytest.approx(res.volume, rel=1e-7)
    full = run(next(r for r in RAY_RECORDS if r["scene"] == "room" and r["HVA"] == 360),
               geometry=True)
    assert full.mesh.volume == pytest.approx(full.volume, rel=1e-7)
    assert run(record).mesh is None


def test_volume_is_the_polyhedron_through_the_hit_points():
    # A 12 x 6 view of the closed room from its centre is a 12-sided prism of height 3 whose
    # corners lie 3 cos 30 degrees from the eye: 3 * (3 R^2) with R^2 = 6.75.
    record = next(r for r in RAY_RECORDS if r["scene"] == "room" and r["HVA"] == 360)
    assert run(record).volume == pytest.approx(3 * 3 * 6.75, rel=1e-6)
    assert math.isclose(record["volume"][0], 60.75, rel_tol=1e-6)
