"""Command line: isovists and view volumes for a list of points, written to CSV.

    pysovist isovist --plan walls.json --points points.csv --out isovists.csv
    pysovist isovist --ifc model.ifc --height 1.2 --points points.csv --out isovists.csv
    pysovist volume  --cloud scan.las  --points points.csv --out volumes.csv
    pysovist section --cloud scan.las  --points points.csv --out sections.csv
    pysovist odtp

Every run also writes a JSON file next to the CSV, with the suffix ``.json``, holding the
pysovist version, all parameters and the sha256 of every input, so a result can be traced
back to what produced it.

A command checks its inputs before it computes anything, and stops with a message that says
what is wrong when an input file is missing, empty or lacks a column, holds a coordinate that is
not a finite number, or when an option lies outside its range or does not apply to the chosen
scene. It also stops when the output would overwrite an input, when an IFC cut meets no element,
when a section plane cuts no ball of the cloud, and when an eye lies outside the prism of an
extruded plan or outside the vertical extent of an IFC model.

``pysovist odtp`` reads its settings from environment variables, as the Open Digital Twin
Platform passes them to a component, and works in ``ODTP_INPUT`` and ``ODTP_OUTPUT``
(default ``/odtp/odtp-input`` and ``/odtp/odtp-output``):

    TASK             isovist | volume | section
    PLAN_FILE        wall segments (JSON records with start/end, a JSON list of segments
                     [[x0, y0], [x1, y1]], or CSV x0,y0,x1,y1)
    CLOSE_FILE       optional polygon [[x, y], ...] that closes the domain (JSON)
    POINTCLOUD_FILE  point cloud (.pts, .las, .laz, .e57, .ply, .pcd) for TASK=volume or section
    MESH_FILE        mesh (.ply, .obj, .stl, ...) for TASK=volume
    QUERY_FILE       CSV with columns x, y and, for volumes and sections, z
    EYE_HEIGHT       eye height that replaces the z column
    MAX_DISTANCE     range limit in metres (default: none)
    N_RAYS           directions per eye for volumes (default 262144)
    SECTION_RAYS     equiangular rays per eye for sections (default 3600)
    SPLAT_RADIUS     ball radius of cloud points in metres (default 0.05)
    ESCAPE, INSIDE, NEAR_CLIP, FLOOR, CEILING, N_JOBS, OUTPUT_PREFIX

A variable applies only to the scene it belongs to: ``CLOSE_FILE``, ``FLOOR`` and ``CEILING`` to
``PLAN_FILE``, and ``SPLAT_RADIUS`` to ``POINTCLOUD_FILE``. The others are ignored, so a
component can pass every variable with its default.
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

DEFAULTS = {"isovist": {"ifc_classes": ["walls"]},
            "volume": {"ifc_classes": ["walls"], "floor": 0.0, "ceiling": 2.5, "radius": 0.05}}
# The options that apply to one scene of a command only, and that scene.
SCENE_OPTIONS = {"isovist": {"height": "ifc", "ifc_classes": "ifc", "ifc_storey": "ifc"},
                 "volume": {"ifc_classes": "ifc", "ifc_storey": "ifc", "close": "plan",
                            "floor": "plan", "ceiling": "plan", "radius": "cloud"}}
_RING = "a list of vertices [[x, y], ...] in order"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# --- option values ------------------------------------------------------------------------------

def _number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None


def _finite(text: str) -> float:
    v = _number(text)
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"must be a finite number in metres, got {text!r}")
    return v


def _positive(text: str) -> float:
    v = _finite(text)
    if not v > 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {text!r}")
    return v


def _non_negative(text: str) -> float:
    v = _finite(text)
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or positive, got {text!r}")
    return v


def _range(text: str) -> float:
    v = _number(text)
    if not v > 0:  # NaN fails too
        raise argparse.ArgumentTypeError(f"must be positive, or inf for no limit, got {text!r}")
    return v


def _count(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {text!r}")
    return v


def _jobs(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if v != -1 and v < 1:
        raise argparse.ArgumentTypeError(f"must be -1 (all cores) or at least 1, got {text!r}")
    return v


def _scene_options(a: argparse.Namespace, scene: str) -> None:
    """Refuse an option given for another scene, then set the defaults of the others."""
    for name, owner in SCENE_OPTIONS[a.command].items():
        if getattr(a, name) is not None and owner != scene:
            option = "--" + name.replace("_", "-")
            raise SystemExit(f"{option} does not apply to --{scene}: it applies to --{owner} only")
    for name, value in DEFAULTS[a.command].items():
        if getattr(a, name) is None:
            setattr(a, name, value)


# --- input files --------------------------------------------------------------------------------

def _check_files(out: Path, inputs: dict) -> None:
    """Every input exists, and neither the table nor its record overwrites one of them."""
    for role, path in inputs.items():
        if path is not None and not Path(path).is_file():
            raise SystemExit(f"--{role}: no such file: {path}")
    out = Path(out)
    if out.suffix.lower() == ".json":
        raise SystemExit(f"--out {out} ends in .json, the suffix of the record written next to "
                         "the table: give a name that ends in .csv")
    targets = {out.resolve(), out.with_suffix(".json").resolve()}
    for role, path in inputs.items():
        if path is not None and Path(path).resolve() in targets:
            raise SystemExit(f"--out {out} would overwrite the input --{role} {path} with the "
                             "table or its .json record: choose another name")


def _rows(index) -> str:
    """Rows of a table, counted from 1 for the first row after the header."""
    rows = [str(int(i) + 1) for i in index]
    more = f" and {len(rows) - 10} more" if len(rows) > 10 else ""
    return f"row{'s' if len(rows) > 1 else ''} {', '.join(rows[:10])}{more}"


def _read_table(path: Path, what: str) -> pd.DataFrame:
    try:
        # round_trip reads every decimal to the nearest float, as Python's float() does
        return pd.read_csv(path, float_precision="round_trip")
    except pd.errors.EmptyDataError:
        raise SystemExit(f"the {what} file {path} is empty") from None


def _columns(df: pd.DataFrame, names: list[str], path: Path, what: str) -> np.ndarray:
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise SystemExit(
            f"the {what} file {path} has no column {', '.join(map(repr, missing))}: its columns "
            f"are {[str(c) for c in df.columns]}. Its first row must name the columns "
            f"{', '.join(names)}.")
    try:
        values = df[names].to_numpy(float)
    except (TypeError, ValueError):
        raise SystemExit(f"the columns {', '.join(names)} of the {what} file {path} must hold "
                         "numbers only (in metres, without units)") from None
    bad = np.flatnonzero(~np.isfinite(values).all(axis=1))
    if len(bad):
        raise SystemExit(f"the {what} file {path} has an empty or non-finite coordinate in "
                         f"{_rows(bad)}")
    return values


def _read_points(path: Path, eye_height: float | None, dims: int) -> np.ndarray:
    df = _read_table(path, "points")
    names = ["x", "y"] if dims == 2 or eye_height is not None else ["x", "y", "z"]
    if names[-1] == "z" and "z" not in df.columns and {"x", "y"} <= set(df.columns):
        raise SystemExit("the points file has no z column: give --eye-height")
    pts = _columns(df, names, path, "points")
    if len(pts) == 0:
        raise SystemExit(f"the points file {path} holds no points")
    if names[-1] == "y" and dims == 3:
        return np.c_[pts, np.full(len(pts), eye_height)]
    return pts


def _read_plan(path: Path):
    from .plan import Plan, _plan_file_segments

    path = Path(path)
    if path.suffix.lower() == ".csv":
        segs = _columns(_read_table(path, "plan"), ["x0", "y0", "x1", "y1"], path, "plan")
        plan = Plan(segs.reshape(-1, 2, 2))
    else:
        try:
            segments = _plan_file_segments(path)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        plan = Plan(segments)
    if len(plan) == 0:
        raise SystemExit(f"the plan {path} holds no wall of positive length")
    return plan


def _check_closing(plan, ring, path) -> None:
    """Refuse a closing polygon that cuts the plan, with the advice of the command line."""
    from .mesh import Mesh

    try:
        Mesh.check_closing(plan, ring)
    except ValueError as e:
        msg = str(e).replace("needs no polygon: keep the default close='hull'", "needs no --close")
        raise SystemExit(f"{msg} (closing polygon {path})") from None


def _read_ring(path: Path | None):
    if path is None:
        return None
    try:
        ring = np.asarray(json.loads(Path(path).read_text(encoding="utf-8")), dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError):
        raise SystemExit(f"the closing polygon {path} must be {_RING}") from None
    if ring.ndim != 2 or ring.shape[1] < 2:
        raise SystemExit(f"the closing polygon {path} must be {_RING}")
    ring = ring[:, :2]
    bad = np.flatnonzero(~np.isfinite(ring).all(axis=1))
    if len(bad):
        raise SystemExit(f"vertex {bad[0] + 1} of the closing polygon {path} has a missing or "
                         "non-finite coordinate")
    if len(ring) < 3:
        raise SystemExit("a closing polygon needs at least three vertices")
    return ring


# --- scenes and eyes ----------------------------------------------------------------------------

def _outside(xy: np.ndarray, ring: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """True for each point outside the polygon ``ring`` (even-odd rule) by more than ``tol``."""
    a = np.asarray(ring, dtype=float)
    b = np.roll(a, -1, axis=0)
    d = b - a
    dd = np.maximum((d * d).sum(axis=1), np.finfo(float).tiny)
    out = np.zeros(len(xy), dtype=bool)
    for i in range(0, len(xy), 4096):
        q = xy[i:i + 4096, None, :]
        t = np.clip(((q - a) * d).sum(axis=2) / dd, 0.0, 1.0)
        near = (np.linalg.norm(q - (a + t[..., None] * d), axis=2) <= tol).any(axis=1)
        x, y = q[..., 0], q[..., 1]
        spans = (a[:, 1] > y) != (b[:, 1] > y)
        xc = a[:, 0] + (y - a[:, 1]) * d[:, 0] / np.where(d[:, 1] == 0, 1.0, d[:, 1])
        inside = (spans & (x < xc)).sum(axis=1) % 2 == 1
        out[i:i + 4096] = ~near & ~inside
    return out


def _check_prism(plan, ring, floor: float, ceiling: float, eyes: np.ndarray) -> None:
    """Every eye lies in the closed prism of ``Mesh.from_plan`` or on its surface."""
    from scipy.spatial import ConvexHull

    z = eyes[:, 2]
    bad = np.flatnonzero((z < floor) | (z > ceiling))
    if len(bad):
        raise SystemExit(
            f"the eyes in {_rows(bad)} of the points file lie outside the extruded plan in "
            f"height: an eye must stand between the floor ({floor:g} m) and the ceiling "
            f"({ceiling:g} m), in metres")
    if ring is None:
        p = plan.segments.reshape(-1, 2)
        outline, name = p[ConvexHull(p).vertices], "the convex hull of the plan"
    else:
        outline, name = ring, "the closing polygon"
    bad = np.flatnonzero(_outside(eyes[:, :2], outline))
    if len(bad):
        raise SystemExit(
            f"the eyes in {_rows(bad)} of the points file lie outside {name}, which closes the "
            "extruded plan, so their rays leave the building: give points inside it, in the "
            "frame and units of the plan")


def _check_ifc_eyes(a: argparse.Namespace, mesh, eyes: np.ndarray) -> None:
    """Every eye lies within the vertical extent of the elements read from the IFC model."""
    lo, hi = float(mesh.vertices[:, 2].min()), float(mesh.vertices[:, 2].max())
    bad = np.flatnonzero((eyes[:, 2] < lo) | (eyes[:, 2] > hi))
    if not len(bad):
        return
    msg = (f"the eyes in {_rows(bad)} of the points file lie outside the model in height: its "
           f"elements span z = {lo:g} to {hi:g} m. The z of an eye is absolute, in metres in the "
           "world frame of the file")
    if a.ifc_storey is not None:
        from .ifc import read_ifc

        elevation = read_ifc(a.ifc, classes=a.ifc_classes, storey=a.ifc_storey)[2]
        msg += (f", and the storey {a.ifc_storey!r} stands at z = {elevation:g} m: an eye "
                f"1.6 m above it has z = {elevation + 1.6:g}")
    raise SystemExit(msg)


def _ifc_plan(a: argparse.Namespace):
    from .plan import Plan

    plan = Plan.from_ifc(a.ifc, a.height, classes=a.ifc_classes, storey=a.ifc_storey)
    if len(plan):
        return plan
    from .ifc import read_ifc

    vertices, _, elevation = read_ifc(a.ifc, classes=a.ifc_classes, storey=a.ifc_storey)
    where = (f"the storey {a.ifc_storey!r}" if a.ifc_storey is not None
             else "the project origin")
    lo, hi = vertices[:, 2].min() - elevation, vertices[:, 2].max() - elevation
    raise SystemExit(f"the cut at --height {a.height:g} m above {where} meets no element of "
                     f"{a.ifc_classes}: they span {lo:g} to {hi:g} m above {where}. Give "
                     "--height in metres above it")


def _read_cloud(path: Path, radius: float):
    from .pointcloud import PointCloud

    try:
        cloud = PointCloud.read(path, radius=radius)
    except pd.errors.EmptyDataError:
        cloud = None
    except ValueError as e:  # the reader names the file and what it lacks
        raise SystemExit(str(e)) from None
    if cloud is None or len(cloud) == 0:
        raise SystemExit(f"the point cloud {path} holds no points")
    return cloud


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
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _isovist(a: argparse.Namespace) -> None:
    from .field import isovist_field
    from .plan import Plan

    t0 = time.time()
    if (a.plan is None) == (a.ifc is None):
        raise SystemExit("give exactly one of --plan, --ifc")
    _scene_options(a, "ifc" if a.ifc else "plan")
    inputs = {"plan": a.plan, "ifc": a.ifc, "close": a.close, "points": a.points}
    params = {"max_distance": a.max_distance, "n_jobs": a.n_jobs}
    if a.ifc:
        if a.height is None:
            raise SystemExit("give --height with --ifc")
        _check_files(a.out, inputs)
        plan = _ifc_plan(a)
        params.update(height=a.height, ifc_classes=a.ifc_classes, ifc_storey=a.ifc_storey)
    else:
        _check_files(a.out, inputs)
        plan = _read_plan(a.plan)
    ring = _read_ring(a.close)
    if ring is not None:
        _check_closing(plan, ring, a.close)
        closing = np.stack([ring, np.roll(ring, -1, axis=0)], axis=1)
        plan = Plan(np.concatenate([plan.segments, closing]))
    pts = _read_points(a.points, None, 2)
    df = isovist_field(plan, pts, max_distance=a.max_distance, n_jobs=a.n_jobs)
    _write(df, a.out, "isovist", params, inputs, t0)


def _volume(a: argparse.Namespace) -> None:
    from .volume3d import view_volume_field

    t0 = time.time()
    sources = {"cloud": a.cloud, "mesh": a.mesh, "plan": a.plan, "ifc": a.ifc}
    given = [k for k, v in sources.items() if v]
    if len(given) != 1:
        raise SystemExit("give exactly one of --cloud, --mesh, --plan, --ifc")
    _scene_options(a, given[0])
    inputs = {**sources, "close": a.close, "points": a.points}
    _check_files(a.out, inputs)
    kw = {}
    if a.cloud:
        occ = _read_cloud(a.cloud, a.radius)
        kw["radius"] = a.radius
    elif a.mesh:
        from .mesh import Mesh

        try:
            occ = Mesh.from_file(a.mesh)
        except ValueError as e:  # the reader names the file and what it lacks
            raise SystemExit(f"{e}. Give a triangle mesh in a format open3d reads, such as .ply, "
                             ".obj or .stl") from None
        if len(occ.triangles) == 0:
            raise SystemExit(f"the mesh {a.mesh} holds no triangle: open3d reads a file it cannot "
                             "parse as an empty mesh. Give a triangle mesh in a format open3d "
                             "reads, such as .ply, .obj or .stl")
    elif a.ifc:
        from .mesh import Mesh

        occ = Mesh.from_ifc(a.ifc, classes=a.ifc_classes, storey=a.ifc_storey)
    else:
        from .mesh import Mesh

        plan, ring = _read_plan(a.plan), _read_ring(a.close)
        if ring is not None:
            _check_closing(plan, ring, a.close)
        occ = Mesh.from_plan(plan, floor=a.floor, ceiling=a.ceiling,
                             close=ring if ring is not None else "hull")
    pts = _read_points(a.points, a.eye_height, 3)
    if a.plan:
        _check_prism(plan, ring, a.floor, a.ceiling, pts)
    elif a.ifc:
        _check_ifc_eyes(a, occ, pts)
    df = view_volume_field(occ, pts, n_rays=a.n_rays, max_distance=a.max_distance, escape=a.escape,
                           inside=a.inside, near_clip=a.near_clip,
                           n_jobs=a.n_jobs, **kw)
    params = {"n_rays": a.n_rays, "max_distance": a.max_distance, "escape": a.escape,
              "inside": a.inside, "near_clip": a.near_clip, "radius": a.radius,
              "floor": a.floor, "ceiling": a.ceiling, "eye_height": a.eye_height,
              "directions": "fibonacci", "n_jobs": a.n_jobs}
    if a.ifc:
        params.update(ifc_classes=a.ifc_classes, ifc_storey=a.ifc_storey)
    _write(df, a.out, "volume", params, inputs, t0)


def _section(a: argparse.Namespace) -> None:
    from .raycast import section_isovist_area

    t0 = time.time()
    inputs = {"cloud": a.cloud, "points": a.points}
    _check_files(a.out, inputs)
    cloud = _read_cloud(a.cloud, a.radius)
    pts = _read_points(a.points, a.eye_height, 3)
    # A plane cuts the balls whose centre lies closer than the radius (as in section_disks).
    z = np.sort(cloud.points[:, 2])
    cut = (np.searchsorted(z, pts[:, 2] + a.radius, "left")
           - np.searchsorted(z, pts[:, 2] - a.radius, "right"))
    bad = np.flatnonzero(cut == 0)
    if len(bad):
        raise SystemExit(
            f"the planes at the eye heights of {_rows(bad)} of the points file cut no ball of "
            f"the cloud: its balls reach from z = {z[0] - a.radius:g} to {z[-1] + a.radius:g} m. "
            "Give eye heights in metres, in the frame of the cloud")
    secs = [section_isovist_area(cloud, p[:2], p[2], n_rays=a.n_rays,
                                 max_distance=a.max_distance, escape=a.escape, inside=a.inside)
            for p in pts]
    df = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
                       "area": [s.area for s in secs],
                       "clearance": [s.clearance for s in secs],
                       "inside_occluder": [s.flags.inside_occluder for s in secs],
                       "unbounded": [s.flags.unbounded for s in secs]})
    params = {"n_rays": a.n_rays, "radius": a.radius, "max_distance": a.max_distance,
              "escape": a.escape, "inside": a.inside, "eye_height": a.eye_height,
              "directions": "equiangular"}
    _write(df, a.out, "section", params, inputs, t0)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pysovist", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    iso = sub.add_parser("isovist", help="exact 2D isovists from wall segments or an IFC model")
    iso.add_argument("--plan", type=Path)
    iso.add_argument("--ifc", type=Path, help="IFC model, cut at --height")
    iso.add_argument("--height", type=_finite,
                     help="height of the cut in metres above the storey (or the project origin)")
    _ifc_options(iso)
    iso.add_argument("--close", type=Path, help="polygon JSON that closes the domain")
    iso.add_argument("--points", type=Path, required=True)
    iso.add_argument("--out", type=Path, required=True)
    iso.add_argument("--max-distance", type=_range, default=math.inf)
    iso.add_argument("--n-jobs", type=_jobs, default=1)
    iso.set_defaults(run=_isovist)

    vol = sub.add_parser("volume",
                         help="view volumes from a point cloud, a mesh or an extruded plan")
    vol.add_argument("--cloud", type=Path)
    vol.add_argument("--mesh", type=Path)
    vol.add_argument("--plan", type=Path)
    vol.add_argument("--ifc", type=Path, help="IFC model")
    _ifc_options(vol)
    vol.add_argument("--close", type=Path, help="polygon JSON that closes an extruded plan")
    vol.add_argument("--floor", type=_finite, help="floor of an extruded plan in metres (0.0)")
    vol.add_argument("--ceiling", type=_finite, help="ceiling of an extruded plan in metres (2.5)")
    vol.add_argument("--points", type=Path, required=True)
    vol.add_argument("--eye-height", type=_finite,
                     help="absolute z of every eye in metres, in place of the z column")
    vol.add_argument("--out", type=Path, required=True)
    vol.add_argument("--n-rays", type=_count, default=262144)
    vol.add_argument("--radius", type=_positive, help="ball radius of cloud points in metres "
                                                      "(0.05)")
    vol.add_argument("--max-distance", type=_range, default=math.inf)
    vol.add_argument("--escape", choices=["clip", "zero", "nan"], default="clip")
    vol.add_argument("--inside", choices=["nan", "zero"], default="nan")
    vol.add_argument("--near-clip", type=_non_negative, default=0.0)
    vol.add_argument("--n-jobs", type=_jobs, default=-1)
    vol.set_defaults(run=_volume)

    sec = sub.add_parser("section", help="isovist areas of the horizontal section of a point "
                                         "cloud at eye height")
    sec.add_argument("--cloud", type=Path, required=True)
    sec.add_argument("--points", type=Path, required=True)
    sec.add_argument("--eye-height", type=_finite,
                     help="absolute z of every eye in metres, in place of the z column")
    sec.add_argument("--out", type=Path, required=True)
    sec.add_argument("--n-rays", type=_count, default=3600)
    sec.add_argument("--radius", type=_positive, default=0.05)
    sec.add_argument("--max-distance", type=_range, default=math.inf)
    sec.add_argument("--escape", choices=["clip", "zero", "nan"], default="clip")
    sec.add_argument("--inside", choices=["nan", "zero"], default="nan")
    sec.set_defaults(run=_section)

    od = sub.add_parser("odtp", help="run as an ODTP component, settings from the environment")
    od.set_defaults(run=None)
    return ap


def _ifc_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ifc-classes", nargs="+",
                        help="presets (walls, furnished) and IFC classes to read from --ifc "
                             "(walls)")
    parser.add_argument("--ifc-storey", help="name or GlobalId of the storey to read from --ifc")


def _odtp_argv(env) -> list[str]:
    inp = Path(env.get("ODTP_INPUT", "/odtp/odtp-input"))
    out = Path(env.get("ODTP_OUTPUT", "/odtp/odtp-output"))
    task = env.get("TASK", "")
    if task not in ("isovist", "volume", "section"):
        raise SystemExit(f"TASK must be 'isovist', 'volume' or 'section', got {task!r}")
    if not env.get("QUERY_FILE"):
        raise SystemExit("QUERY_FILE must name the CSV of points in ODTP_INPUT")
    scenes = {"isovist": ["PLAN_FILE"], "volume": ["POINTCLOUD_FILE", "MESH_FILE", "PLAN_FILE"],
              "section": ["POINTCLOUD_FILE"]}[task]
    given = [v for v in scenes if env.get(v)]
    if len(given) != 1:
        need = scenes[0] if len(scenes) == 1 else f"exactly one of {', '.join(scenes)}"
        raise SystemExit(f"TASK={task} needs {need}")
    prefix = env.get("OUTPUT_PREFIX", "pysovist")
    argv = [task, "--points", str(inp / env["QUERY_FILE"]),
            "--out", str(out / f"{prefix}_{task}.csv")]

    def opt(name, var, path=False):
        if env.get(var):
            argv.extend([name, str(inp / env[var]) if path else env[var]])

    opt("--max-distance", "MAX_DISTANCE")
    if task in ("isovist", "volume"):
        opt("--plan", "PLAN_FILE", path=True)
        opt("--n-jobs", "N_JOBS")
        if env.get("PLAN_FILE"):
            opt("--close", "CLOSE_FILE", path=True)
    if task == "volume":
        opt("--cloud", "POINTCLOUD_FILE", path=True)
        opt("--mesh", "MESH_FILE", path=True)
        opt("--eye-height", "EYE_HEIGHT")
        opt("--n-rays", "N_RAYS")
        opt("--escape", "ESCAPE")
        opt("--inside", "INSIDE")
        opt("--near-clip", "NEAR_CLIP")
        if env.get("PLAN_FILE"):
            opt("--floor", "FLOOR")
            opt("--ceiling", "CEILING")
        if env.get("POINTCLOUD_FILE"):
            opt("--radius", "SPLAT_RADIUS")
    if task == "section":
        opt("--cloud", "POINTCLOUD_FILE", path=True)
        opt("--eye-height", "EYE_HEIGHT")
        opt("--n-rays", "SECTION_RAYS")
        opt("--radius", "SPLAT_RADIUS")
        opt("--escape", "ESCAPE")
        opt("--inside", "INSIDE")
    return argv


def main(argv: list[str] | None = None) -> None:
    """Run the ``pysovist`` command with the arguments ``argv``, by default ``sys.argv[1:]``.

    A command whose input is inconsistent stops with ``SystemExit`` and a message that says
    what is wrong, before it writes anything.
    """
    ap = _parser()
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.command == "odtp":
        a = ap.parse_args(_odtp_argv(os.environ))
    a.run(a)


if __name__ == "__main__":
    main()
