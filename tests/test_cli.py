"""The pysovist command line and its ODTP mode."""

import json

import pandas as pd
import pytest

from pysovist.cli import main

from .geometry import rect


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
