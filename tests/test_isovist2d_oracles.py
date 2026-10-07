"""Exact 2D kernel against independent references and geometric invariants."""

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pysovist import Plan, isovist

from .geometry import rect

shapely = pytest.importorskip("shapely")
visilibity = pytest.importorskip("visilibity")

from shapely.geometry import Polygon  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

SETTINGS = settings(max_examples=150, deadline=None)


# --- scene strategies -------------------------------------------------------------------------

@st.composite
def room_with_pillars(draw):
    """Rectangular room with up to one rectangular pillar per grid cell (no overlaps)."""
    w = draw(st.floats(4, 30))
    h = draw(st.floats(4, 30))
    nx, ny = draw(st.integers(1, 3)), draw(st.integers(1, 3))
    pillars = []
    for i in range(nx):
        for j in range(ny):
            if not draw(st.booleans()):
                continue
            cw, ch = w / nx, h / ny
            fx0, fx1 = sorted(draw(st.lists(st.floats(0.15, 0.85), min_size=2, max_size=2)))
            fy0, fy1 = sorted(draw(st.lists(st.floats(0.15, 0.85), min_size=2, max_size=2)))
            assume(fx1 - fx0 > 0.05 and fy1 - fy0 > 0.05)
            pillars.append((i * cw + fx0 * cw, j * ch + fy0 * ch,
                            i * cw + fx1 * cw, j * ch + fy1 * ch))
    ox = draw(st.floats(0.01, 0.99)) * w
    oy = draw(st.floats(0.01, 0.99)) * h
    for x0, y0, x1, y1 in pillars:
        assume(not (x0 - 1e-3 <= ox <= x1 + 1e-3 and y0 - 1e-3 <= oy <= y1 + 1e-3))
    return w, h, pillars, (ox, oy)


@st.composite
def room_with_free_walls(draw):
    """Square room with random interior walls that may cross each other."""
    size = draw(st.floats(5, 20))
    n = draw(st.integers(1, 10))
    coords = st.floats(0.05 * size, 0.95 * size)
    walls = np.array([[[draw(coords), draw(coords)], [draw(coords), draw(coords)]]
                      for _ in range(n)])
    walls = walls[np.linalg.norm(walls[:, 1] - walls[:, 0], axis=1) > 1e-2]
    o = np.array([draw(coords), draw(coords)])
    return size, walls, o


def _scene(w, h, pillars):
    parts = [rect(0, 0, w, h)] + [rect(*p) for p in pillars]
    return np.concatenate(parts)


def _visilibity_area(w, h, pillars, o):
    vis = visilibity
    outer = vis.Polygon([vis.Point(0, 0), vis.Point(w, 0), vis.Point(w, h), vis.Point(0, h)])
    holes = [vis.Polygon([vis.Point(x0, y0), vis.Point(x0, y1),
                          vis.Point(x1, y1), vis.Point(x1, y0)])
             for x0, y0, x1, y1 in pillars]
    env = vis.Environment([outer, *holes])
    return vis.Visibility_Polygon(vis.Point(*o), env, 1e-9).area()


def _shadow_area(room, walls, o):
    """Area of the room hidden behind the walls, by polygon subtraction."""
    far = 1e4
    shadows = []
    for a, b in walls:
        da, db = a - o, b - o
        if abs(da[0] * db[1] - da[1] * db[0]) < 1e-12:
            continue  # wall seen edge-on hides nothing of positive area
        fa = a + far * da / np.linalg.norm(da)
        fb = b + far * db / np.linalg.norm(db)
        shadows.append(Polygon([a, b, fb, fa]).buffer(0))
    if not shadows:
        return 0.0
    return room.intersection(unary_union(shadows)).area


# --- oracles ----------------------------------------------------------------------------------

@SETTINGS
@given(room_with_pillars())
def test_matches_visilibity_in_rooms_with_pillars(scene):
    w, h, pillars, o = scene
    iso = isovist(Plan(_scene(w, h, pillars)), o)
    assume(not iso.flags.on_wall)
    assert iso.area == pytest.approx(_visilibity_area(w, h, pillars, o), rel=1e-7)


@SETTINGS
@given(room_with_free_walls())
def test_matches_shadow_subtraction_with_crossing_walls(scene):
    size, walls, o = scene
    plan = Plan(np.concatenate([rect(0, 0, size, size), walls]))
    iso = isovist(plan, o)
    assume(not iso.flags.on_wall and iso.clearance > 1e-4)
    room = Polygon([(0, 0), (size, 0), (size, size), (0, size)])
    expected = size * size - _shadow_area(room, walls, o)
    assert iso.area == pytest.approx(expected, rel=1e-7, abs=1e-7)


# --- invariants -------------------------------------------------------------------------------

@SETTINGS
@given(room_with_free_walls(), st.floats(-1e3, 1e3), st.floats(-1e3, 1e3),
       st.floats(0, 2 * math.pi))
def test_rigid_motion_invariance(scene, dx, dy, angle):
    size, walls, o = scene
    seg = np.concatenate([rect(0, 0, size, size), walls])
    base = isovist(Plan(seg), o)
    assume(not base.flags.on_wall)
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    shift = np.array([dx, dy])
    moved = isovist(Plan(seg @ rot.T + shift), o @ rot.T + shift)
    for key in ("area", "perimeter", "occlusivity", "r_mean", "r_min", "r_max", "drift"):
        assert moved.metrics[key] == pytest.approx(base.metrics[key], rel=1e-7, abs=1e-7)


@SETTINGS
@given(room_with_free_walls(), st.floats(0.01, 100))
def test_scaling(scene, k):
    size, walls, o = scene
    seg = np.concatenate([rect(0, 0, size, size), walls])
    base = isovist(Plan(seg), o)
    assume(not base.flags.on_wall)
    scaled = isovist(Plan(seg * k), o * k)
    assert scaled.area == pytest.approx(base.area * k * k, rel=1e-7)
    assert scaled.perimeter == pytest.approx(base.perimeter * k, rel=1e-7)
    assert scaled.metrics["compactness"] == pytest.approx(base.metrics["compactness"], rel=1e-7)


@SETTINGS
@given(room_with_free_walls(), st.floats(0.05, 0.95), st.floats(0.05, 0.95),
       st.floats(0.05, 0.95), st.floats(0.05, 0.95))
def test_adding_a_wall_never_increases_area(scene, x0, y0, x1, y1):
    size, walls, o = scene
    seg = np.concatenate([rect(0, 0, size, size), walls])
    extra = np.array([[[x0 * size, y0 * size], [x1 * size, y1 * size]]])
    before = isovist(Plan(seg), o)
    after = isovist(Plan(np.concatenate([seg, extra])), o)
    assume(not before.flags.on_wall and not after.flags.on_wall)
    assert after.area <= before.area * (1 + 1e-9)


@SETTINGS
@given(room_with_free_walls(), st.floats(0.1, 30), st.floats(0.1, 30))
def test_area_grows_with_range_and_stays_inside_the_disc(scene, r1, r2):
    size, walls, o = scene
    plan = Plan(np.concatenate([rect(0, 0, size, size), walls]))
    lo, hi = sorted((r1, r2))
    a = isovist(plan, o, max_distance=lo)
    b = isovist(plan, o, max_distance=hi)
    assume(not a.flags.on_wall)
    assert a.area <= b.area * (1 + 1e-9)
    assert a.area <= math.pi * lo * lo * (1 + 1e-9)


def test_angular_index_gives_the_same_isovists_as_direct_search(monkeypatch):
    import pysovist.isovist2d as kernel

    rng = np.random.default_rng(7)
    start = rng.uniform(0.5, 29.5, size=(800, 2))
    walls = np.stack([start, start + rng.normal(0, 0.8, size=(800, 2))], axis=1)
    walls = np.clip(walls, 0.1, 29.9)
    plan = Plan(np.concatenate([rect(0, 0, 30, 30), walls]))
    origins = rng.uniform(0.5, 29.5, size=(25, 2))
    results = {}
    for name, work in (("direct", 10**12), ("indexed", 0)):
        monkeypatch.setattr(kernel, "_DIRECT_WORK", work)
        results[name] = [isovist(plan, o, max_distance=12.0) for o in origins]
    for a, b in zip(results["direct"], results["indexed"], strict=True):
        if a.flags.on_wall:
            assert b.flags.on_wall
            continue
        assert b.area == pytest.approx(a.area, rel=1e-12)
        assert b.perimeter == pytest.approx(a.perimeter, rel=1e-12)
