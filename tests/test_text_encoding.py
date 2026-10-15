"""Text files are read and written as UTF-8 on every platform.

Python's default text encoding is the platform's, cp1252 on Windows, so a plan
written as UTF-8 with a non-ASCII name would decode differently there. Each
check runs in a new interpreter with ``-X warn_default_encoding``, where every
read or write that leaves the encoding to the platform warns, and a warning
raised from pysovist is an error.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

PRELUDE = """
import sys, warnings
warnings.filterwarnings("error", category=EncodingWarning, module=r"pysovist(\\.|$)")
"""

WALLS = [{"start": [0, 0], "end": [10, 0], "name": "Küche"},
         {"start": [10, 0], "end": [10, 10], "name": "Flur"},
         {"start": [10, 10], "end": [0, 10], "name": "Büro"},
         {"start": [0, 10], "end": [0, 0], "name": "Wand"}]


def run(code: str, cwd: Path) -> subprocess.CompletedProcess:
    script = PRELUDE + textwrap.dedent(code)
    # the whole environment, because Python on Windows needs SYSTEMROOT to start
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONUTF8")}
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run([sys.executable, "-X", "warn_default_encoding", "-c", script], cwd=cwd,
                          env=env, capture_output=True, text=True, encoding="utf-8", timeout=300)


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "walls.json").write_text(json.dumps(WALLS, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "ring.json").write_text("[[0, 0], [10, 0], [10, 10], [0, 10]]", encoding="utf-8")
    (tmp_path / "points.csv").write_text("x,y\n5,5\n2,3\n", encoding="utf-8")
    (tmp_path / "cloud.pts").write_text("2\n1.0 2.0 3.0 0 0 0 0\n4.0 5.0 6.0 0 0 0 0\n",
                                        encoding="utf-8")
    return tmp_path


def test_a_plan_with_non_ascii_names_reads_as_utf8(folder):
    r = run("""
        from pysovist import Plan
        print(len(Plan.from_json("walls.json")))
    """, folder)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["4"]


def test_a_point_cloud_text_file_reads_as_utf8(folder):
    r = run("""
        from pysovist import PointCloud
        print(len(PointCloud.read("cloud.pts", radius=0.05)))
    """, folder)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["2"]


def test_the_command_line_reads_and_writes_utf8(folder):
    r = run("""
        from pysovist.cli import main
        main(["isovist", "--plan", "walls.json", "--close", "ring.json",
              "--points", "points.csv", "--out", "iso.csv"])
    """, folder)
    assert r.returncode == 0, r.stderr
    record = json.loads((folder / "iso.json").read_text(encoding="utf-8"))
    assert record["n_points"] == 2
