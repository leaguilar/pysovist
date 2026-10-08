"""Command line: isovists and view volumes for a list of points, written to CSV.

    pysovist isovist --plan walls.json --points points.csv --out isovists.csv
    pysovist volume  --cloud scan.las  --points points.csv --out volumes.csv
    pysovist odtp

Every run also writes a JSON file next to the CSV, with the suffix ``.json``, holding the
pysovist version, all parameters and the sha256 of every input, so a result can be traced
back to what produced it.

``pysovist odtp`` reads its settings from environment variables, as the Open Digital Twin
Platform passes them to a component, and works in ``ODTP_INPUT`` and ``ODTP_OUTPUT``
(default ``/odtp/odtp-input`` and ``/odtp/odtp-output``):

    TASK             isovist | volume
    PLAN_FILE        wall segments (JSON records with start/end, or CSV x0,y0,x1,y1)
    CLOSE_FILE       optional polygon [[x, y], ...] that closes the domain (JSON)
    POINTCLOUD_FILE  point cloud (.pts, .las, .laz, .e57, .ply, .pcd) for TASK=volume
    MESH_FILE        mesh (.ply, .obj, .stl, ...) for TASK=volume
    QUERY_FILE       CSV with columns x, y and, for volumes, z
    EYE_HEIGHT       eye height that replaces the z column
    MAX_DISTANCE     range limit in metres (default: none)
    N_RAYS           directions per eye for volumes (default 262144)
    SPLAT_RADIUS     ball radius of cloud points in metres (default 0.05)
    ESCAPE, INSIDE, NEAR_CLIP, FLOOR, CEILING, N_JOBS, OUTPUT_PREFIX
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _read_plan(path: Path):
    from .plan import Plan

    if Path(path).suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return Plan(df[["x0", "y0", "x1", "y1"]].to_numpy(float).reshape(-1, 2, 2))
    return Plan.from_json(path)


def _read_ring(path: Path | None):
    if path is None:
        return None
    ring = np.asarray(json.loads(Path(path).read_text()), dtype=float)[:, :2]
    return ring


def _read_points(path: Path, eye_height: float | None, dims: int) -> np.ndarray:
    df = pd.read_csv(path)
    xy = df[["x", "y"]].to_numpy(float)
    if dims == 2:
        return xy
    if eye_height is not None:
        return np.c_[xy, np.full(len(xy), eye_height)]
    if "z" not in df:
        raise SystemExit("the points file has no z column: give --eye-height")
    return np.c_[xy, df["z"].to_numpy(float)]


def _write(df: pd.DataFrame, out: Path, task: str, params: dict, inputs: dict, t0: float) -> None:
    from . import __version__

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    meta = {
        "task": task,
        "pysovist_version": __version__,
        "parameters": params,
        "inputs": {k: {"path": str(p), "sha256": _sha256(p)} for k, p in inputs.items() if p},
        "n_points": len(df),
        "seconds": round(time.time() - t0, 3),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def _isovist(a: argparse.Namespace) -> None:
    from .field import isovist_field
    from .plan import Plan

    t0 = time.time()
    plan = _read_plan(a.plan)
    ring = _read_ring(a.close)
    if ring is not None:
        closing = np.stack([ring, np.roll(ring, -1, axis=0)], axis=1)
        plan = Plan(np.concatenate([plan.segments, closing]))
    pts = _read_points(a.points, None, 2)
    df = isovist_field(plan, pts, max_distance=a.max_distance, n_jobs=a.n_jobs)
    _write(df, a.out, "isovist", {"max_distance": a.max_distance, "n_jobs": a.n_jobs},
           {"plan": a.plan, "close": a.close, "points": a.points}, t0)


def _volume(a: argparse.Namespace) -> None:
    from .volume3d import view_volume_field

    t0 = time.time()
    sources = [s for s in (a.cloud, a.mesh, a.plan) if s]
    if len(sources) != 1:
        raise SystemExit("give exactly one of --cloud, --mesh, --plan")
    kw = {}
    if a.cloud:
        from .pointcloud import PointCloud

        occ = PointCloud.read(a.cloud, radius=a.radius)
        kw["radius"] = a.radius
    elif a.mesh:
        from .mesh import Mesh

        occ = Mesh.from_file(a.mesh)
    else:
        from .mesh import Mesh

        ring = _read_ring(a.close)
        occ = Mesh.from_plan(_read_plan(a.plan), floor=a.floor, ceiling=a.ceiling,
                             close=ring if ring is not None else "hull")
    pts = _read_points(a.points, a.eye_height, 3)
    df = view_volume_field(occ, pts, n_rays=a.n_rays, max_distance=a.max_distance, escape=a.escape,
                           inside=a.inside, near_clip=a.near_clip,
                           n_jobs=a.n_jobs, **kw)
    params = {"n_rays": a.n_rays, "max_distance": a.max_distance, "escape": a.escape,
              "inside": a.inside, "near_clip": a.near_clip, "radius": a.radius,
              "floor": a.floor, "ceiling": a.ceiling, "eye_height": a.eye_height,
              "directions": "fibonacci", "n_jobs": a.n_jobs}
    _write(df, a.out, "volume", params,
           {"cloud": a.cloud, "mesh": a.mesh, "plan": a.plan, "close": a.close,
            "points": a.points}, t0)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pysovist", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    iso = sub.add_parser("isovist", help="exact 2D isovists from wall segments")
    iso.add_argument("--plan", type=Path, required=True)
    iso.add_argument("--close", type=Path, help="polygon JSON that closes the domain")
    iso.add_argument("--points", type=Path, required=True)
    iso.add_argument("--out", type=Path, required=True)
    iso.add_argument("--max-distance", type=float, default=math.inf)
    iso.add_argument("--n-jobs", type=int, default=1)
    iso.set_defaults(run=_isovist)

    vol = sub.add_parser("volume",
                         help="view volumes from a point cloud, a mesh or an extruded plan")
    vol.add_argument("--cloud", type=Path)
    vol.add_argument("--mesh", type=Path)
    vol.add_argument("--plan", type=Path)
    vol.add_argument("--close", type=Path, help="polygon JSON that closes an extruded plan")
    vol.add_argument("--floor", type=float, default=0.0)
    vol.add_argument("--ceiling", type=float, default=2.5)
    vol.add_argument("--points", type=Path, required=True)
    vol.add_argument("--eye-height", type=float)
    vol.add_argument("--out", type=Path, required=True)
    vol.add_argument("--n-rays", type=int, default=262144)
    vol.add_argument("--radius", type=float, default=0.05)
    vol.add_argument("--max-distance", type=float, default=math.inf)
    vol.add_argument("--escape", choices=["clip", "zero", "nan"], default="clip")
    vol.add_argument("--inside", choices=["nan", "zero"], default="nan")
    vol.add_argument("--near-clip", type=float, default=0.0)
    vol.add_argument("--n-jobs", type=int, default=-1)
    vol.set_defaults(run=_volume)

    od = sub.add_parser("odtp", help="run as an ODTP component, settings from the environment")
    od.set_defaults(run=None)
    return ap


def _odtp_argv(env) -> list[str]:
    inp = Path(env.get("ODTP_INPUT", "/odtp/odtp-input"))
    out = Path(env.get("ODTP_OUTPUT", "/odtp/odtp-output"))
    task = env.get("TASK", "")
    if task not in ("isovist", "volume"):
        raise SystemExit(f"TASK must be 'isovist' or 'volume', got {task!r}")
    prefix = env.get("OUTPUT_PREFIX", "pysovist")
    argv = [task, "--points", str(inp / env["QUERY_FILE"]),
            "--out", str(out / f"{prefix}_{task}.csv")]

    def opt(name, var, path=False):
        if env.get(var):
            argv.extend([name, str(inp / env[var]) if path else env[var]])

    opt("--plan", "PLAN_FILE", path=True)
    opt("--close", "CLOSE_FILE", path=True)
    opt("--max-distance", "MAX_DISTANCE")
    opt("--n-jobs", "N_JOBS")
    if task == "volume":
        opt("--cloud", "POINTCLOUD_FILE", path=True)
        opt("--mesh", "MESH_FILE", path=True)
        opt("--eye-height", "EYE_HEIGHT")
        opt("--n-rays", "N_RAYS")
        opt("--radius", "SPLAT_RADIUS")
        opt("--escape", "ESCAPE")
        opt("--inside", "INSIDE")
        opt("--near-clip", "NEAR_CLIP")
        opt("--floor", "FLOOR")
        opt("--ceiling", "CEILING")
    return argv


def main(argv: list[str] | None = None) -> None:
    """Run the ``pysovist`` command with the arguments ``argv``, by default ``sys.argv[1:]``."""
    ap = _parser()
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.command == "odtp":
        a = ap.parse_args(_odtp_argv(os.environ))
    a.run(a)


if __name__ == "__main__":
    main()
