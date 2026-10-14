"""The pysovist command line and its ODTP mode."""

import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pysovist
from pysovist import Plan, PointCloud, isovist_field, section_isovist_area
from pysovist.cli import _odtp_argv, _parser, main

from .geometry import polygon_walls, rect, room_walls, square_rule


def _has_numba() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


NEEDS_NUMBA = pytest.mark.skipif(not _has_numba(), reason="needs numba, from the pointcloud extra")

@pytest.fixture
def square_plan(tmp_path):
    walls = rect(0, 0, 10, 10)
    records = [{"start": [*a, 0.0], "end": [*b, 0.0]} for a, b in walls.tolist()]
    path = tmp_path / "walls.json"
    path.write_text(json.dumps(records))
    return path


@pytest.fixture
def points(tmp_path):
    path = tmp_path / "points.csv"
    df = pd.DataFrame({"x": [5.0, 2.0, 8.0], "y": [5.0, 3.0, 9.0], "z": [1.7, 1.7, 0.8]})
    df.to_csv(path, index=False)
    return path


def test_isovist_command_writes_one_row_per_point(square_plan, points, tmp_path):
    out = tmp_path / "out.csv"
    main(["isovist", "--plan", str(square_plan), "--points", str(points), "--out", str(out)])
    df = pd.read_csv(out)
    assert len(df) == 3
    assert df.area.to_numpy() == pytest.approx([100, 100, 100])
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["task"] == "isovist" and meta["pysovist_version"]


def test_volume_command_on_an_extruded_plan(square_plan, points, tmp_path):
    pytest.importorskip("open3d")
    out = tmp_path / "vol.csv"
    main(["volume", "--plan", str(square_plan), "--ceiling", "2.5", "--points", str(points),
          "--n-rays", "65536", "--out", str(out)])
    df = pd.read_csv(out)
    assert df.volume.to_numpy() == pytest.approx([250, 250, 250], rel=2e-3)


def test_odtp_mode_reads_settings_from_the_environment(square_plan, points, tmp_path, monkeypatch):
    inp, outp = tmp_path / "odtp-input", tmp_path / "odtp-output"
    inp.mkdir()
    outp.mkdir()
    (inp / "walls.json").write_text(square_plan.read_text())
    (inp / "points.csv").write_text(points.read_text())
    for k, v in {"ODTP_INPUT": str(inp), "ODTP_OUTPUT": str(outp), "TASK": "isovist",
                 "PLAN_FILE": "walls.json", "QUERY_FILE": "points.csv", "MAX_DISTANCE": "40",
                 "OUTPUT_PREFIX": "run1"}.items():
        monkeypatch.setenv(k, v)
    main(["odtp"])
    df = pd.read_csv(outp / "run1_isovist.csv")
    assert df.area.to_numpy() == pytest.approx([100, 100, 100])
    meta = json.loads((outp / "run1_isovist.json").read_text())
    assert meta["parameters"]["max_distance"] == 40.0
    assert len(meta["inputs"]["plan"]["sha256"]) == 64


def test_odtp_mode_rejects_an_unknown_task(monkeypatch, tmp_path):
    monkeypatch.setenv("ODTP_INPUT", str(tmp_path))
    monkeypatch.setenv("ODTP_OUTPUT", str(tmp_path))
    monkeypatch.setenv("TASK", "teleport")
    with pytest.raises(SystemExit):
        main(["odtp"])


def test_volume_command_keeps_a_single_thread_request(square_plan, points, tmp_path):
    pytest.importorskip("open3d")
    out = tmp_path / "vol.csv"
    main(["volume", "--plan", str(square_plan), "--points", str(points), "--n-rays", "1024",
          "--n-jobs", "1", "--out", str(out)])
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["parameters"]["n_jobs"] == 1


# --- section isovists of a point cloud --------------------------------------------------------

def assert_same_bits(a, b):
    """Equal float64 bit patterns, with any two NaNs counting as equal (0.0 and -0.0 differ)."""
    a = np.ascontiguousarray(a, dtype="<f8")
    b = np.ascontiguousarray(b, dtype="<f8")
    assert a.shape == b.shape
    assert ((a.view("<i8") == b.view("<i8")) | (np.isnan(a) & np.isnan(b))).all()


def write_pts(path, points):
    np.savetxt(path, points, fmt="%.17g")
    return path


@pytest.fixture
def pillar_room(tmp_path):
    """A 4 m square room with a 0.6 m square pillar, walls 0 to 2 m high, sampled every 2 cm."""
    walls = room_walls(4.0, 0.02, 0.0, 2.0)
    pillar = room_walls(0.6, 0.02, 0.0, 2.0) + [2.5, 1.5, 0.0]
    return write_pts(tmp_path / "room.pts", np.concatenate([walls, pillar]))


@pytest.fixture
def eyes(tmp_path):
    # Free eyes at three heights, an eye on the pillar's wall, and an eye outside the room, whose
    # rays escape on the side away from it.
    df = pd.DataFrame({"x": [1.0, 3.5, 1.0, 2.0, 2.5, 5.0],
                       "y": [1.0, 3.0, 3.2, 1.8, 1.8, 2.0],
                       "z": [1.2, 0.8, 1.7, 1.2, 1.0, 1.0]})
    path = tmp_path / "eyes.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("options, kw", [
    ([], {}),
    (["--n-rays", "720", "--radius", "0.03", "--max-distance", "3", "--escape", "zero",
      "--inside", "zero"],
     {"n_rays": 720, "radius": 0.03, "max_distance": 3.0, "escape": "zero", "inside": "zero"}),
])
@NEEDS_NUMBA
def test_section_command_equals_the_library_bit_for_bit(pillar_room, eyes, tmp_path, options, kw):
    out = tmp_path / "sections.csv"
    main(["section", "--cloud", str(pillar_room), "--points", str(eyes), "--out", str(out),
          *options])
    df = pd.read_csv(out, float_precision="round_trip")
    q = pd.read_csv(eyes).to_numpy()
    kw = dict(kw)
    cloud = PointCloud.read(pillar_room, radius=kw.pop("radius", 0.05))
    ref = [section_isovist_area(cloud, p[:2], p[2], **kw) for p in q]
    assert list(df.columns) == ["x", "y", "z", "area", "clearance", "inside_occluder",
                                "unbounded"]
    assert_same_bits(df[["x", "y", "z"]].to_numpy(), q)
    assert_same_bits(df.area.to_numpy(), [s.area for s in ref])
    assert_same_bits(df.clearance.to_numpy(), [s.clearance for s in ref])
    np.testing.assert_array_equal(df.inside_occluder.to_numpy(),
                                  [s.flags.inside_occluder for s in ref])
    np.testing.assert_array_equal(df.unbounded.to_numpy(), [s.flags.unbounded for s in ref])
    # The eyes cover a free section, an eye inside a ball and an eye whose rays escape.
    assert df.inside_occluder.sum() == 1
    assert df.unbounded.sum() == (0 if "max_distance" in kw else 1)


@NEEDS_NUMBA
def test_section_record_holds_version_parameters_and_input_checksums(pillar_room, eyes,
                                                                     tmp_path):
    out = tmp_path / "sections.csv"
    main(["section", "--cloud", str(pillar_room), "--points", str(eyes), "--eye-height", "1.2",
          "--out", str(out)])
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["task"] == "section"
    assert meta["pysovist_version"] == pysovist.__version__
    assert meta["parameters"] == {"n_rays": 3600, "radius": 0.05, "max_distance": math.inf,
                                  "escape": "clip", "inside": "nan", "eye_height": 1.2,
                                  "directions": "equiangular"}
    assert meta["inputs"] == {
        role: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for role, path in (("cloud", pillar_room), ("points", eyes))}
    assert meta["n_points"] == 6
    assert (pd.read_csv(out).z == 1.2).all()


@NEEDS_NUMBA
def test_section_command_on_a_sampled_square_room(tmp_path):
    side, s, radius, n = 10.0, 0.02, 0.03, 3600
    cloud = write_pts(tmp_path / "room.pts", room_walls(side, s))
    points = tmp_path / "centre.csv"
    # The centre of the room, on a sample row and halfway between two rows.
    pd.DataFrame({"x": [5.0, 5.0], "y": [5.0, 5.0], "z": [1.5, 1.51]}).to_csv(points, index=False)
    out = tmp_path / "sections.csv"
    main(["section", "--cloud", str(cloud), "--points", str(points), "--radius", str(radius),
          "--out", str(out)])
    df = pd.read_csv(out)
    inner = square_rule(side - 2 * radius, n)
    outer = square_rule(side - 2 * math.sqrt(radius**2 - s**2 / 2), n)
    assert ((inner <= df.area) & (df.area <= outer)).all()
    assert not df.inside_occluder.any() and not df.unbounded.any()


# Parameters of the other tasks, which an ODTP component passes with their defaults.
OTHER_TASKS = {"N_RAYS": "262144", "N_JOBS": "-1", "NEAR_CLIP": "0.0", "FLOOR": "0.0",
               "CEILING": "2.5", "PLAN_FILE": "", "CLOSE_FILE": "", "MESH_FILE": ""}


def test_odtp_section_maps_its_own_variables_only():
    env = {"ODTP_INPUT": "/in", "ODTP_OUTPUT": "/out", "TASK": "section", "QUERY_FILE": "q.csv",
           "POINTCLOUD_FILE": "c.ply", "EYE_HEIGHT": "1.7", "SECTION_RAYS": "720",
           "SPLAT_RADIUS": "0.03", "MAX_DISTANCE": "40", "ESCAPE": "zero", "INSIDE": "zero",
           "OUTPUT_PREFIX": "run", **OTHER_TASKS}
    argv = _odtp_argv(env)
    assert argv[:5] == ["section", "--points", str(Path("/in") / "q.csv"),
                        "--out", str(Path("/out") / "run_section.csv")]
    assert dict(zip(argv[5::2], argv[6::2], strict=True)) == {
        "--cloud": str(Path("/in") / "c.ply"), "--eye-height": "1.7", "--n-rays": "720",
        "--radius": "0.03", "--max-distance": "40", "--escape": "zero", "--inside": "zero"}


@NEEDS_NUMBA
def test_odtp_section_defaults_to_3600_rays(pillar_room, eyes, tmp_path, monkeypatch):
    inp, outp = tmp_path / "odtp-input", tmp_path / "odtp-output"
    inp.mkdir()
    outp.mkdir()
    shutil.copy(pillar_room, inp / "room.pts")
    shutil.copy(eyes, inp / "eyes.csv")
    for k, v in {"ODTP_INPUT": str(inp), "ODTP_OUTPUT": str(outp), "TASK": "section",
                 "POINTCLOUD_FILE": "room.pts", "QUERY_FILE": "eyes.csv", "EYE_HEIGHT": "1.2",
                 "OUTPUT_PREFIX": "run1", **OTHER_TASKS}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("SECTION_RAYS", raising=False)
    main(["odtp"])
    meta = json.loads((outp / "run1_section.json").read_text())
    assert meta["parameters"]["n_rays"] == 3600
    assert meta["parameters"]["eye_height"] == 1.2
    assert len(pd.read_csv(outp / "run1_section.csv")) == 6


@pytest.mark.parametrize("command", ["isovist", "volume"])
def test_a_closing_polygon_that_cuts_the_plan_is_refused(square_plan, points, tmp_path, command):
    # three corners of the square room: the closing edge runs diagonally through it, and the walls
    # along y = 10 and x = 0 touch the triangle only at a corner
    ring = tmp_path / "ring.json"
    ring.write_text(json.dumps([[0, 0], [10, 0], [10, 10]]))
    refused = r"walls lie mostly outside the closing polygon"
    with pytest.raises((SystemExit, ValueError), match=refused):
        main([command, "--plan", str(square_plan), "--close", str(ring), "--points", str(points),
              "--out", str(tmp_path / "out.csv")])


def test_the_outline_of_a_closed_plan_closes_the_isovist_command(square_plan, points, tmp_path):
    ring = tmp_path / "ring.json"
    ring.write_text(json.dumps([[0, 0], [10, 0], [10, 10], [0, 10]]))
    main(["isovist", "--plan", str(square_plan), "--close", str(ring), "--points", str(points),
          "--out", str(tmp_path / "out.csv")])
    assert pd.read_csv(tmp_path / "out.csv").area.tolist() == pytest.approx([100.0] * 3)


# --- every command against the library call it wraps --------------------------------------------
#
# Reference: the library function that the command wraps, called with the same arguments. The
# command only reads files and writes a CSV, so its table equals the function's bit for bit.

def read_table(path):
    return pd.read_csv(path, float_precision="round_trip")


def assert_same_table(path, ref):
    pd.testing.assert_frame_equal(read_table(path), ref.reset_index(drop=True), check_exact=True,
                                  check_dtype=False)


FOOTPRINT = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 9), (0, 9)]  # an L-shaped building
INNER = [[(2.5, 2.0), (2.5, 3.5)], [(6.0, 1.0), (7.5, 1.0)]]  # its interior walls


def write_plan(path, segments):
    """Wall segments as JSON records or, for a .csv path, as columns x0, y0, x1, y1."""
    s = np.asarray(segments, dtype=float)
    if path.suffix == ".csv":
        pd.DataFrame(s.reshape(-1, 4), columns=["x0", "y0", "x1", "y1"]).to_csv(path, index=False)
    else:
        path.write_text(json.dumps([{"start": a, "end": b} for a, b in s.tolist()]))
    return path


def write_points(path, **columns):
    pd.DataFrame(columns).to_csv(path, index=False)
    return path


@pytest.fixture
def building(tmp_path):
    """Interior walls, the footprint that closes them, and points in both wings of the L."""
    ring = tmp_path / "footprint.json"
    ring.write_text(json.dumps(FOOTPRINT))
    pts = write_points(tmp_path / "pts.csv", x=[1.5, 8.0, 2.0, 3.0], y=[1.2, 2.5, 7.5, 3.0],
                       z=[1.6, 0.4, 2.2, 1.0])
    return write_plan(tmp_path / "inner.json", INNER), ring, pts


@pytest.mark.parametrize("suffix", [".json", ".csv"])
@pytest.mark.parametrize("options, kw", [
    ([], {}),
    (["--max-distance", "3.5"], {"max_distance": 3.5}),
])
def test_isovist_command_equals_isovist_field_bit_for_bit(tmp_path, suffix, options, kw):
    walls = np.concatenate([rect(0, 0, 10, 10), rect(4, 4, 5, 6)])
    plan = write_plan(tmp_path / f"walls{suffix}", walls)
    g = np.linspace(0.3, 9.7, 18)
    xy = np.array([(x, y) for x in g for y in g if not (4 <= x <= 5 and 4 <= y <= 6)])
    pts = write_points(tmp_path / "pts.csv", x=xy[:, 0], y=xy[:, 1])
    for n_jobs in (1, 2):  # 300 observers: two processes share them
        out = tmp_path / f"iso{n_jobs}.csv"
        main(["isovist", "--plan", str(plan), "--points", str(pts), "--out", str(out),
              "--n-jobs", str(n_jobs), *options])
        assert_same_table(out, isovist_field(Plan(walls), xy, n_jobs=n_jobs, **kw))
    assert len(xy) > 256  # more observers than one chunk of isovist_field


def test_isovist_command_closes_the_plan_with_the_polygon_as_walls(building, tmp_path):
    inner, ring, pts = building
    out = tmp_path / "iso.csv"
    main(["isovist", "--plan", str(inner), "--close", str(ring), "--points", str(pts),
          "--out", str(out)])
    closed = Plan(np.concatenate([INNER, polygon_walls(FOOTPRINT)]))
    assert_same_table(out, isovist_field(closed, pd.read_csv(pts).to_numpy()))


@pytest.mark.parametrize("options, kw", [
    ([], {}),
    (["--n-rays", "2048", "--max-distance", "4", "--escape", "zero", "--inside", "zero",
      "--near-clip", "0.3", "--n-jobs", "2"],
     {"n_rays": 2048, "max_distance": 4.0, "escape": "zero", "inside": "zero", "near_clip": 0.3,
      "n_jobs": 2}),
])
def test_volume_command_on_a_plan_equals_view_volume_field_bit_for_bit(building, tmp_path,
                                                                       options, kw):
    pytest.importorskip("open3d")
    from pysovist import Mesh, view_volume_field

    inner, ring, pts = building
    out = tmp_path / "vol.csv"
    main(["volume", "--plan", str(inner), "--close", str(ring), "--floor", "0.1", "--ceiling",
          "2.7", "--points", str(pts), "--out", str(out), "--n-rays", "4096", *options])
    kw = {"n_rays": 4096, **kw}
    mesh = Mesh.from_plan(Plan(INNER), floor=0.1, ceiling=2.7, close=FOOTPRINT)
    assert_same_table(out, view_volume_field(mesh, pd.read_csv(pts).to_numpy(), **kw))


def test_volume_command_on_a_plan_defaults_to_its_convex_hull(building, tmp_path):
    pytest.importorskip("open3d")
    from scipy.spatial import ConvexHull

    from pysovist import Mesh, view_volume_field

    inner, _, _ = building
    pts = write_points(tmp_path / "pts.csv", x=[3.0, 5.0], y=[3.0, 2.0], z=[0.3, 9.0])
    out = tmp_path / "vol.csv"
    main(["volume", "--plan", str(inner), "--eye-height", "1.25", "--points", str(pts),
          "--n-rays", "16384", "--out", str(out)])
    eyes = [(3.0, 3.0, 1.25), (5.0, 2.0, 1.25)]
    hull = Mesh.from_plan(Plan(INNER))  # floor 0, ceiling 2.5, closed along the hull
    assert_same_table(out, view_volume_field(hull, eyes, n_rays=16384))
    # The extruded-plan identity: 2.5 m times the isovist area of the plan closed along the hull,
    # to the 2e-3 of the quadrature at 16,384 directions.
    p = np.asarray(INNER, float).reshape(-1, 2)
    closed = Plan(np.concatenate([INNER, polygon_walls(p[ConvexHull(p).vertices])]))
    area = isovist_field(closed, np.asarray(eyes)).area.to_numpy()
    assert read_table(out).volume.to_numpy() == pytest.approx(2.5 * area, rel=2e-3)


def box_mesh_file(path, lo, hi):
    """A closed box as an OBJ file, its faces turned inwards, written by hand."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    quads = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)]
    tris = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    path.write_text("".join(f"v {x!r} {y!r} {z!r}\n" for x, y, z in v)
                    + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in tris))
    return path


def test_volume_command_on_a_mesh_equals_view_volume_field_bit_for_bit(tmp_path):
    pytest.importorskip("open3d")
    from pysovist import Mesh, view_volume_field

    path = box_mesh_file(tmp_path / "box.obj", (0, 0, 0), (4, 3, 2.5))
    pts = write_points(tmp_path / "pts.csv", x=[1.0, 2.0, 3.5], y=[1.0, 1.5, 2.5],
                       z=[1.7, 0.2, 2.4])
    out = tmp_path / "vol.csv"
    main(["volume", "--mesh", str(path), "--points", str(pts), "--n-rays", "4096",
          "--max-distance", "50", "--out", str(out)])
    mesh = Mesh.from_file(path)
    assert len(mesh.triangles) == 12
    ref = view_volume_field(mesh, pd.read_csv(pts).to_numpy(), n_rays=4096, max_distance=50.0)
    assert_same_table(out, ref)
    # Reference: a closed box is star-shaped from any eye inside it, so every eye sees all of it.
    assert read_table(out).volume.to_numpy() == pytest.approx(np.full(3, 30.0), rel=2e-3)


@pytest.fixture
def closed_room_cloud(tmp_path):
    """A 4 m x 3 m room 2.5 m high, its floor, ceiling and walls sampled every 5 cm."""
    s, (lx, ly, h) = 0.05, (4.0, 3.0, 2.5)
    xs, ys, zs = (np.linspace(0, n, int(round(n / s)) + 1) for n in (lx, ly, h))
    pts = []
    for z in (0.0, h):
        gx, gy = np.meshgrid(xs, ys)
        pts.append(np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, z)])
    for x in (0.0, lx):
        gy, gz = np.meshgrid(ys, zs)
        pts.append(np.c_[np.full(gy.size, x), gy.ravel(), gz.ravel()])
    for y in (0.0, ly):
        gx, gz = np.meshgrid(xs, zs)
        pts.append(np.c_[gx.ravel(), np.full(gx.size, y), gz.ravel()])
    return write_pts(tmp_path / "room.pts", np.concatenate(pts))


@pytest.mark.parametrize("options, kw", [
    ([], {}),
    (["--radius", "0.04", "--max-distance", "2.5", "--escape", "nan", "--inside", "zero",
      "--near-clip", "0.2", "--n-jobs", "3"],
     {"radius": 0.04, "max_distance": 2.5, "escape": "nan", "inside": "zero", "near_clip": 0.2,
      "n_jobs": 3}),
])
@NEEDS_NUMBA
def test_volume_command_on_a_cloud_equals_view_volume_field_bit_for_bit(closed_room_cloud,
                                                                        tmp_path, options, kw):
    from pysovist import view_volume_field

    # free eyes, an eye inside a ball of the floor, and an eye 10 cm from a wall
    pts = write_points(tmp_path / "pts.csv", x=[1.0, 2.0, 3.0, 3.9], y=[1.0, 1.5, 2.0, 1.0],
                       z=[1.7, 0.01, 1.2, 1.5])
    out = tmp_path / "vol.csv"
    main(["volume", "--cloud", str(closed_room_cloud), "--points", str(pts), "--n-rays", "2048",
          "--out", str(out), *options])
    kw = {"n_rays": 2048, **kw}
    cloud = PointCloud.read(closed_room_cloud, radius=kw.pop("radius", 0.05))
    assert_same_table(out, view_volume_field(cloud, pd.read_csv(pts).to_numpy(), **kw))
    df = read_table(out)
    # Near clipping ignores the balls around the eye, so the eye in the floor sees past them.
    assert df.inside_occluder.tolist() == [False, not options, False, False]
    assert df.near_clipped.tolist() == [False, bool(options), False, bool(options)]


@NEEDS_NUMBA
def test_volume_command_eye_height_replaces_the_z_column(closed_room_cloud, tmp_path):
    from pysovist import view_volume_field

    pts = write_points(tmp_path / "pts.csv", x=[1.0, 2.0], y=[1.0, 1.5], z=[0.3, 9.0])
    out = tmp_path / "vol.csv"
    main(["volume", "--cloud", str(closed_room_cloud), "--points", str(pts), "--eye-height",
          "1.6", "--n-rays", "1024", "--out", str(out)])
    ref = view_volume_field(PointCloud.read(closed_room_cloud), [(1.0, 1.0, 1.6), (2.0, 1.5, 1.6)],
                            n_rays=1024)
    assert_same_table(out, ref)


def record(out):
    return json.loads(Path(out).with_suffix(".json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_isovist_record_holds_every_parameter_and_input(building, tmp_path):
    inner, ring, pts = building
    out = tmp_path / "iso.csv"
    main(["isovist", "--plan", str(inner), "--close", str(ring), "--points", str(pts),
          "--max-distance", "12", "--out", str(out)])
    meta = record(out)
    assert meta["task"] == "isovist" and meta["pysovist_version"] == pysovist.__version__
    assert meta["parameters"] == {"max_distance": 12.0, "n_jobs": 1}
    assert meta["inputs"] == {role: {"path": str(p), "sha256": sha(p)}
                              for role, p in (("plan", inner), ("close", ring), ("points", pts))}
    assert meta["n_points"] == 4 and meta["seconds"] >= 0


@NEEDS_NUMBA
def test_volume_record_holds_every_parameter_and_input(closed_room_cloud, tmp_path):
    pts = write_points(tmp_path / "pts.csv", x=[1.0], y=[1.0], z=[1.7])
    out = tmp_path / "vol.csv"
    main(["volume", "--cloud", str(closed_room_cloud), "--points", str(pts), "--n-rays", "1024",
          "--radius", "0.04", "--escape", "zero", "--out", str(out)])
    meta = record(out)
    assert meta["task"] == "volume" and meta["pysovist_version"] == pysovist.__version__
    assert meta["parameters"] == {
        "n_rays": 1024, "max_distance": math.inf, "escape": "zero", "inside": "nan",
        "near_clip": 0.0, "radius": 0.04, "floor": 0.0, "ceiling": 2.5, "eye_height": None,
        "directions": "fibonacci", "n_jobs": -1}
    assert meta["inputs"] == {role: {"path": str(p), "sha256": sha(p)}
                              for role, p in (("cloud", closed_room_cloud), ("points", pts))}
    assert meta["n_points"] == 1


# --- the ODTP mode --------------------------------------------------------------------------------
#
# Reference: the same command given as options. Every variable maps to one option.

def test_odtp_isovist_maps_its_variables_and_defaults():
    env = {"TASK": "isovist", "QUERY_FILE": "q.csv", "PLAN_FILE": "w.json",
           "CLOSE_FILE": "ring.json", "MAX_DISTANCE": "40", "N_JOBS": "4", "EYE_HEIGHT": "1.7",
           "POINTCLOUD_FILE": "", "MESH_FILE": "", "N_RAYS": "262144", "SPLAT_RADIUS": "0.05"}
    argv = _odtp_argv(env)
    inp, outp = Path("/odtp/odtp-input"), Path("/odtp/odtp-output")
    assert argv[:5] == ["isovist", "--points", str(inp / "q.csv"),
                        "--out", str(outp / "pysovist_isovist.csv")]
    assert dict(zip(argv[5::2], argv[6::2], strict=True)) == {
        "--max-distance": "40", "--plan": str(inp / "w.json"), "--close": str(inp / "ring.json"),
        "--n-jobs": "4"}


@pytest.mark.parametrize("source, var, extra", [
    ("--plan", "PLAN_FILE", {"--close": "ring.json", "--floor": "0.1", "--ceiling": "2.7"}),
    ("--mesh", "MESH_FILE", {}),
    ("--cloud", "POINTCLOUD_FILE", {"--radius": "0.03"}),
])
def test_odtp_volume_maps_the_variables_of_its_source(source, var, extra):
    env = {"ODTP_INPUT": "/in", "ODTP_OUTPUT": "/out", "TASK": "volume", "QUERY_FILE": "q.csv",
           "PLAN_FILE": "", "MESH_FILE": "", "POINTCLOUD_FILE": "", var: "scene.x",
           "CLOSE_FILE": "ring.json", "FLOOR": "0.1", "CEILING": "2.7", "SPLAT_RADIUS": "0.03",
           "EYE_HEIGHT": "1.6", "N_RAYS": "4096", "SECTION_RAYS": "720", "MAX_DISTANCE": "30",
           "ESCAPE": "zero", "INSIDE": "zero", "NEAR_CLIP": "0.2", "N_JOBS": "2",
           "OUTPUT_PREFIX": "run"}
    argv = _odtp_argv(env)
    assert argv[:5] == ["volume", "--points", str(Path("/in") / "q.csv"),
                        "--out", str(Path("/out") / "run_volume.csv")]
    paths = {"--close"}
    expected = {source: str(Path("/in") / "scene.x"), "--eye-height": "1.6", "--n-rays": "4096",
                "--max-distance": "30", "--escape": "zero", "--inside": "zero",
                "--near-clip": "0.2", "--n-jobs": "2",
                **{k: str(Path("/in") / v) if k in paths else v for k, v in extra.items()}}
    assert dict(zip(argv[5::2], argv[6::2], strict=True)) == expected


@NEEDS_NUMBA
def test_odtp_volume_equals_the_command_bit_for_bit(closed_room_cloud, tmp_path, monkeypatch):
    inp, outp = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    shutil.copy(closed_room_cloud, inp / "room.pts")
    write_points(inp / "eyes.csv", x=[1.0, 2.0, 3.0], y=[1.0, 1.5, 2.0], z=[1.7, 0.3, 1.2])
    for k, v in {"ODTP_INPUT": str(inp), "ODTP_OUTPUT": str(outp), "TASK": "volume",
                 "POINTCLOUD_FILE": "room.pts", "QUERY_FILE": "eyes.csv",
                 **OTHER_TASKS, "N_RAYS": "1024", "SPLAT_RADIUS": "0.04", "ESCAPE": "zero",
                 "MAX_DISTANCE": "3", "OUTPUT_PREFIX": "run"}.items():
        monkeypatch.setenv(k, v)
    main(["odtp"])
    direct = tmp_path / "direct.csv"
    main(["volume", "--cloud", str(inp / "room.pts"), "--points", str(inp / "eyes.csv"),
          "--n-rays", "1024", "--radius", "0.04", "--escape", "zero", "--max-distance", "3",
          "--out", str(direct)])
    assert (outp / "run_volume.csv").read_bytes() == direct.read_bytes()
    assert record(outp / "run_volume.csv")["parameters"] == record(direct)["parameters"]


# --- inputs that the commands refuse -------------------------------------------------------------
#
# Each of these inputs used to give a table without a message: rows of inf or NaN, the range disc,
# an empty table, or an input file overwritten. The command now stops before it writes anything
# and says what is wrong.

def refused(argv, match, capsys, out=None):
    """Run the command, expect it to stop with a message matching ``match``, and no output."""
    with pytest.raises(SystemExit) as e:
        main([str(a) for a in argv])
    msg = e.value.code if isinstance(e.value.code, str) else capsys.readouterr().err
    assert re.search(match, msg), msg
    if out is not None:
        assert not Path(out).exists() and not Path(out).with_suffix(".json").exists()
    return msg


@pytest.fixture
def square(tmp_path):
    """A closed 10 m square room and two points in it, with z."""
    plan = write_plan(tmp_path / "walls.json", rect(0, 0, 10, 10))
    pts = write_points(tmp_path / "pts.csv", x=[5.0, 2.0], y=[5.0, 3.0], z=[1.6, 1.2])
    return plan, pts


def test_an_output_ending_in_json_is_refused(square, tmp_path, capsys):
    # The record of the run is written next to the CSV with the suffix .json, so it would
    # overwrite the table itself.
    plan, pts = square
    out = tmp_path / "result.json"
    refused(["isovist", "--plan", plan, "--points", pts, "--out", out], r"\.csv", capsys, out)


@pytest.mark.parametrize("target", ["points", "plan"])
def test_an_output_that_would_overwrite_an_input_is_refused(square, tmp_path, capsys, target):
    plan, pts = square
    before = {p: p.read_bytes() for p in (plan, pts)}
    # --out pts.csv overwrites the points, and --out walls.csv writes its record to walls.json.
    out = pts if target == "points" else plan.with_suffix(".csv")
    refused(["isovist", "--plan", plan, "--points", pts, "--out", out], r"overwrite", capsys)
    assert {p: p.read_bytes() for p in (plan, pts)} == before


@pytest.mark.parametrize("command", [
    ["isovist", "--plan", "missing.json"],
    ["volume", "--mesh", "missing.ply"],
    ["volume", "--cloud", "missing.pts"],
    ["section", "--cloud", "missing.pts"],
])
def test_a_missing_input_file_is_refused_before_any_work(square, tmp_path, capsys, command):
    plan, pts = square
    out = tmp_path / "out.csv"
    command = [str(tmp_path / c) if c.startswith("missing") else c for c in command]
    refused([*command, "--points", pts, "--eye-height", "1.5", "--out", out]
            if command[0] != "isovist" else [*command, "--points", pts, "--out", out],
            r"missing\.\w+", capsys, out)


@pytest.mark.parametrize("content, match", [
    ("X,Y\n5,5\n", r"no column.*'x'.*columns are \['X', 'Y'\]"),
    ("5,5\n2,3\n", r"no column.*'x'"),  # no header row: the first row is read as the header
    ("", r"empty"),
    ("x,y\n", r"no points"),
    ("x,y\n5,5\n,3\n", r"row 2"),  # an empty cell
    ("x,y\n5,5\n2,3\ninf,3\n", r"row 3"),
    ("x,y\n5 m,5\n", r"number"),
])
def test_a_points_file_that_is_not_a_table_of_coordinates_is_refused(square, tmp_path, capsys,
                                                                      content, match):
    plan, _ = square
    pts = tmp_path / "bad.csv"
    pts.write_text(content)
    out = tmp_path / "out.csv"
    refused(["isovist", "--plan", plan, "--points", pts, "--out", out], match, capsys, out)


def test_a_non_finite_eye_height_in_the_z_column_is_refused(tmp_path, capsys, closed_room_cloud):
    pts = tmp_path / "pts.csv"
    pts.write_text("x,y,z\n1,1,1.7\n2,2,\n")
    out = tmp_path / "out.csv"
    for command in ("volume", "section"):
        refused([command, "--cloud", closed_room_cloud, "--points", pts, "--out", out], r"row 2",
                capsys, out)


@pytest.mark.parametrize("name, content, match", [
    ("walls.csv", "a,b\n0,0\n", r"no column.*'x0'"),
    ("walls.csv", "x0,y0,x1,y1\n0,0,10,0\n10,0,,10\n", r"row 2"),
    ("walls.json", '[{"start": [0, 0], "end": [10, NaN]}]', r"record 1"),
    ("walls.json", "[[[0, 0], [10, 0]]]", r'"start"'),
    ("walls.json", "[]", r"no wall"),
    ("walls.json", '[{"start": [1, 1], "end": [1, 1]}]', r"no wall"),
])
def test_a_plan_file_without_valid_walls_is_refused(square, tmp_path, capsys, name, content,
                                                    match):
    _, pts = square
    plan = tmp_path / name
    plan.write_text(content)
    out = tmp_path / "out.csv"
    refused(["isovist", "--plan", plan, "--points", pts, "--out", out], match, capsys, out)
    if not content.startswith("[[["):  # the same plan, extruded
        pytest.importorskip("open3d")
        refused(["volume", "--plan", plan, "--points", pts, "--out", out], match, capsys, out)


@pytest.mark.parametrize("ring, match", [
    ([[0, 0], [10, 0]], r"three vertices"),
    ([[0, 0], [10, 0], [10, None], [0, 10]], r"vertex 3"),
    ({"x": [0, 10, 10]}, r"\[\[x, y\]"),
])
def test_a_closing_polygon_that_is_not_a_polygon_is_refused(square, tmp_path, capsys, ring,
                                                            match):
    plan, pts = square
    path = tmp_path / "ring.json"
    path.write_text(json.dumps(ring))
    out = tmp_path / "out.csv"
    for command in ("isovist", "volume"):
        refused([command, "--plan", plan, "--close", path, "--points", pts, "--out", out], match,
                capsys, out)


@pytest.mark.parametrize("option, value", [
    ("--max-distance", "0"), ("--max-distance", "-1"), ("--max-distance", "nan"),
    ("--n-jobs", "0"), ("--n-jobs", "-2"), ("--n-jobs", "1.5"),
])
def test_an_isovist_option_outside_its_domain_is_refused(square, tmp_path, capsys, option,
                                                         value):
    # --max-distance 0 gave areas of 0, -1 gave an area of pi and a perimeter of -2 pi, nan gave
    # unbounded rows; --n-jobs 0 ran on one process while volume refused it.
    plan, pts = square
    out = tmp_path / "out.csv"
    refused(["isovist", "--plan", plan, "--points", pts, "--out", out, option, value],
            re.escape(option), capsys, out)


@pytest.mark.parametrize("option, value", [
    ("--max-distance", "0"), ("--max-distance", "nan"), ("--n-rays", "0"), ("--n-rays", "-5"),
    ("--radius", "0"), ("--radius", "nan"), ("--radius", "inf"), ("--near-clip", "-1"),
    ("--near-clip", "nan"), ("--eye-height", "nan"), ("--eye-height", "inf"), ("--n-jobs", "0"),
])
def test_a_cloud_option_outside_its_domain_is_refused(closed_room_cloud, tmp_path, capsys,
                                                      option, value):
    pts = write_points(tmp_path / "pts.csv", x=[1.0], y=[1.0], z=[1.6])
    out = tmp_path / "out.csv"
    for command in ("volume", "section"):
        if command == "section" and option in ("--near-clip", "--n-jobs"):
            continue  # not options of section
        refused([command, "--cloud", closed_room_cloud, "--points", pts, "--out", out, option,
                 value], re.escape(option), capsys, out)


@pytest.mark.parametrize("option, value", [
    ("--floor", "nan"), ("--ceiling", "inf"), ("--height", "nan"),
])
def test_a_height_that_is_not_a_number_is_refused(square, tmp_path, capsys, option, value):
    plan, pts = square
    out = tmp_path / "out.csv"
    command = "isovist" if option == "--height" else "volume"
    refused([command, "--plan", plan, "--points", pts, "--out", out, option, value],
            re.escape(option), capsys, out)


@pytest.mark.parametrize("source, option, value", [
    ("--plan", "--radius", "0.1"),
    ("--mesh", "--radius", "0.1"),
    ("--cloud", "--floor", "0.5"),
    ("--cloud", "--ceiling", "3.0"),
    ("--mesh", "--close", "ring.json"),
    ("--plan", "--ifc-classes", "furnished"),
    ("--cloud", "--ifc-storey", "Level 1"),
])
def test_a_volume_option_that_does_not_apply_to_the_scene_is_refused(tmp_path, capsys, source,
                                                                     option, value):
    # --height with --plan was refused already. The other options of one scene type were
    # ignored without a message for the others.
    scene = {"--plan": write_plan(tmp_path / "walls.json", rect(0, 0, 4, 3)),
             "--mesh": box_mesh_file(tmp_path / "box.obj", (0, 0, 0), (4, 3, 2.5)),
             "--cloud": write_pts(tmp_path / "room.pts", room_walls(4.0, 0.05, 0.0, 2.5))}[source]
    (tmp_path / "ring.json").write_text(json.dumps([[0, 0], [4, 0], [4, 3], [0, 3]]))
    if option == "--close":
        value = tmp_path / value
    pts = write_points(tmp_path / "pts.csv", x=[1.0], y=[1.0], z=[1.6])
    out = tmp_path / "out.csv"
    refused(["volume", source, scene, "--points", pts, "--out", out, option, value],
            re.escape(option) + ".*" + re.escape(source.lstrip("-")), capsys, out)


def test_an_isovist_ifc_option_without_ifc_is_refused(square, tmp_path, capsys):
    plan, pts = square
    out = tmp_path / "out.csv"
    for option, value in (("--ifc-classes", "furnished"), ("--ifc-storey", "Level 1")):
        refused(["isovist", "--plan", plan, "--points", pts, "--out", out, option, value],
                re.escape(option), capsys, out)


def test_a_mesh_file_without_triangles_is_refused(tmp_path, capsys):
    pytest.importorskip("open3d")
    pts = write_points(tmp_path / "pts.csv", x=[1.0], y=[1.0], z=[1.6])
    out = tmp_path / "out.csv"
    # open3d reads a file it cannot parse, or a file of another format, as an empty mesh
    for name, content in (("garbage.ply", "not a mesh"), ("points.obj", "v 0 0 0\nv 1 0 0\n")):
        (tmp_path / name).write_text(content)
        refused(["volume", "--mesh", tmp_path / name, "--points", pts, "--out", out],
                r"no triangle", capsys, out)


@pytest.mark.parametrize("content", ["", "0\n"])
def test_a_cloud_file_without_points_is_refused(tmp_path, capsys, content):
    cloud = tmp_path / "empty.pts"
    cloud.write_text(content)
    pts = write_points(tmp_path / "pts.csv", x=[1.0], y=[1.0], z=[1.6])
    out = tmp_path / "out.csv"
    for command in ("volume", "section"):
        refused([command, "--cloud", cloud, "--points", pts, "--out", out], r"no point", capsys,
                out)


@NEEDS_NUMBA
def test_a_section_plane_that_cuts_no_ball_is_refused(pillar_room, tmp_path, capsys):
    # The walls stand from 0 to 2 m and the balls reach 5 cm beyond: a plane at 2.5 m, or at
    # 1700 for an eye height given in millimetres, cuts nothing. It gave the range disc, or inf.
    pts = write_points(tmp_path / "pts.csv", x=[1.0, 1.0, 1.0], y=[1.0, 1.0, 1.0],
                       z=[1.7, 2.5, 1700.0])
    out = tmp_path / "out.csv"
    for options in ([], ["--max-distance", "40"]):
        msg = refused(["section", "--cloud", pillar_room, "--points", pts, "--out", out,
                       *options], r"rows? 2, 3", capsys, out)
        assert re.search(r"-0\.05 to 2\.05", msg), msg
    # The planes just inside the reach of the balls cut them.
    edge = write_points(tmp_path / "edge.csv", x=[1.0, 1.0], y=[1.0, 1.0], z=[-0.049, 2.049])
    main(["section", "--cloud", str(pillar_room), "--points", str(edge), "--out", str(out)])
    assert np.isfinite(read_table(out).clearance).all()


@pytest.mark.parametrize("z, match", [(-0.5, r"row 1"), (2.6, r"row 1"), (1700.0, r"row 1")])
def test_an_eye_outside_an_extruded_plan_in_height_is_refused(square, tmp_path, capsys, z,
                                                              match):
    pytest.importorskip("open3d")
    plan, _ = square
    pts = write_points(tmp_path / "pts.csv", x=[5.0, 2.0], y=[5.0, 3.0], z=[z, 1.2])
    out = tmp_path / "out.csv"
    for options in ([], ["--max-distance", "40"]):
        msg = refused(["volume", "--plan", plan, "--points", pts, "--n-rays", "1024", "--out",
                       out, *options], match, capsys, out)
        assert "between the floor (0" in msg, msg


def test_an_eye_outside_the_outline_of_an_extruded_plan_is_refused(building, tmp_path, capsys):
    pytest.importorskip("open3d")
    inner, ring, _ = building
    # (5, 7) lies in the notch of the L: inside the hull of the footprint, outside the footprint.
    # (5000, 5000) is a point given in millimetres.
    pts = write_points(tmp_path / "pts.csv", x=[1.5, 5.0, 5000.0], y=[1.2, 7.0, 5000.0],
                       z=[1.6, 1.6, 1.6])
    out = tmp_path / "out.csv"
    refused(["volume", "--plan", inner, "--close", ring, "--points", pts, "--out", out],
            r"rows 2, 3.*closing polygon", capsys, out)
    # Without --close the plan closes along the hull of its two walls, which holds none of them.
    refused(["volume", "--plan", inner, "--points", pts, "--out", out],
            r"rows 1, 2, 3.*convex hull", capsys, out)


def test_an_eye_on_the_floor_of_an_extruded_plan_is_flagged_not_refused(square, tmp_path):
    pytest.importorskip("open3d")
    plan, _ = square
    pts = write_points(tmp_path / "pts.csv", x=[5.0, 0.0], y=[5.0, 3.0], z=[0.0, 1.2])
    out = tmp_path / "out.csv"
    main(["volume", "--plan", str(plan), "--points", str(pts), "--n-rays", "1024", "--out",
          str(out)])
    df = read_table(out)
    # one eye on the floor and one on a wall: both on a surface of the prism
    assert df.on_wall.tolist() == [True, True] and df.volume.isna().all()


def test_odtp_mode_names_the_variables_it_needs(tmp_path, monkeypatch, capsys):
    base = {"ODTP_INPUT": str(tmp_path), "ODTP_OUTPUT": str(tmp_path), **OTHER_TASKS,
            "POINTCLOUD_FILE": ""}
    cases = [({"TASK": "isovist", "PLAN_FILE": "walls.json"}, r"QUERY_FILE"),
             ({"TASK": "isovist", "QUERY_FILE": "q.csv"}, r"PLAN_FILE"),
             ({"TASK": "volume", "QUERY_FILE": "q.csv"},
              r"one of POINTCLOUD_FILE, MESH_FILE, PLAN_FILE"),
             ({"TASK": "volume", "QUERY_FILE": "q.csv", "MESH_FILE": "m.ply",
               "PLAN_FILE": "w.json"}, r"one of POINTCLOUD_FILE, MESH_FILE, PLAN_FILE"),
             ({"TASK": "section", "QUERY_FILE": "q.csv"}, r"POINTCLOUD_FILE")]
    for env, match in cases:
        monkeypatch.delenv("QUERY_FILE", raising=False)
        for k, v in {**base, **env}.items():
            monkeypatch.setenv(k, v)
        refused(["odtp"], match, capsys)


@pytest.mark.parametrize("task, scene, foreign, own", [
    ("isovist", {"PLAN_FILE": "w.json"}, {"POINTCLOUD_FILE": "c.ply", "SPLAT_RADIUS": "0.03"},
     "--plan"),
    ("section", {"POINTCLOUD_FILE": "c.ply"}, {"PLAN_FILE": "w.json", "CLOSE_FILE": "r.json",
                                               "FLOOR": "0.1", "CEILING": "2.7"}, "--cloud"),
])
def test_odtp_ignores_the_files_of_other_tasks(task, scene, foreign, own):
    # A component passes every variable: the files of another task add no option.
    argv = _odtp_argv({"TASK": task, "QUERY_FILE": "q.csv", **scene, **foreign})
    assert set(argv[5::2]) == {own}
    parsed = _parser().parse_args(argv)  # the command accepts its options
    assert parsed.command == task
