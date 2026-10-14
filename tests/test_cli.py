"""The pysovist command line and its ODTP mode."""

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pysovist
from pysovist import PointCloud, section_isovist_area
from pysovist.cli import _odtp_argv, main

from .geometry import rect, room_walls, square_rule


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
    # Free eyes at three heights, an eye on the pillar's wall, and an eye above all walls.
    df = pd.DataFrame({"x": [1.0, 3.5, 1.0, 2.0, 2.5, 1.0],
                       "y": [1.0, 3.0, 3.2, 1.8, 1.8, 1.0],
                       "z": [1.2, 0.8, 1.7, 1.2, 1.0, 2.5]})
    path = tmp_path / "eyes.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("options, kw", [
    ([], {}),
    (["--n-rays", "720", "--radius", "0.03", "--max-distance", "3", "--escape", "zero",
      "--inside", "zero"],
     {"n_rays": 720, "radius": 0.03, "max_distance": 3.0, "escape": "zero", "inside": "zero"}),
])
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
    # The eyes cover a free section, an eye inside a ball and an eye whose rays all escape.
    assert df.inside_occluder.sum() == 1
    assert df.unbounded.sum() == (0 if "max_distance" in kw else 1)


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
