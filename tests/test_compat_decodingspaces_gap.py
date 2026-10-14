"""DeCodingSpaces 2D layer against the plugin where the gap between the last and the first ray
faces a pillar.

Reference: ``data/decodingspaces2d/ds_probe_2d_gap.jsonl``, 37 runs of the DeCodingSpaces
"Isovist" component (release 2021.10.11, Rhino 7, headless): the 10 m square with a 1 m x 2 m
pillar from (6, 4) to (7, 6), seen from (2, 5), full view, range 1,000 m, direction
(sin a, cos a). Tag ``gap_edge_12``: Precision 12 for a = 256 to 284 degrees. Tag
``gap_edge_11``: Precision 11 for a = 279, 282, 289, 292, 295, 302, 305 and 308 degrees. In
every run the plugin casts no closing ray, and the gap from the last ray back to the first
bridges the pillar's silhouette. Each record holds the 17 Isovist Properties (IP), the hits per
line (OV), the Isovist Polygon's area and vertices, and every radial's end point.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pysovist.compat.decodingspaces import DS_PROPERTY_NAMES, ds_isovist
from pysovist.visibility2d import segment_ray_hits

DATA = Path(__file__).parent / "data" / "decodingspaces2d" / "ds_probe_2d_gap.jsonl"
RECORDS = [json.loads(line) for line in DATA.read_text().splitlines()]


def _bearing(r):
    return round(math.degrees(math.atan2(*r["direction"]))) % 360


IDS = [f"pr{r['precision']}-a{_bearing(r)}" for r in RECORDS]


def _run(r):
    return ds_isovist(np.array(r["segments"], float), r["viewpoint"],
                      view_direction=r["direction"], view_range=r["range"],
                      view_angle_deg=r["view_angle"], precision=r["precision"])


def test_records_cover_both_precisions():
    assert len(RECORDS) == 37
    assert sorted({r["tag"] for r in RECORDS}) == ["gap_edge_11", "gap_edge_12"]


@pytest.mark.parametrize("r", RECORDS, ids=IDS)
def test_rays_match_the_plugin_one_by_one(r):
    """No closing ray: the plugin casts exactly Precision rays (measured)."""
    res = _run(r)
    v = np.array(r["radial_ends"], float) - np.array(r["viewpoint"], float)
    length = np.hypot(v[:, 0], v[:, 1])
    u = v / length[:, None]
    assert len(res.angles) == len(u) == r["precision"]
    mine = np.c_[np.cos(res.angles), np.sin(res.angles)]
    turn = np.arctan2(mine[:, 0] * u[:, 1] - mine[:, 1] * u[:, 0], (mine * u).sum(axis=1))
    assert np.abs(turn).max() < 1e-12
    np.testing.assert_allclose(res.depths, length, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("r", RECORDS, ids=IDS)
def test_properties_match_the_plugin(r):
    got = _run(r).properties
    np.testing.assert_allclose([got[k] for k in DS_PROPERTY_NAMES], r["IP"], rtol=1e-9,
                               atol=1e-9)


@pytest.mark.parametrize("r", RECORDS, ids=IDS)
def test_polygon_and_hits_per_line_match_the_plugin(r):
    res = _run(r)
    ring = res.polygon
    x, y = ring[:, 0], ring[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert area == pytest.approx(r["poly_area"], rel=1e-9)
    segs = np.array(r["segments"], float)
    ov = np.array(r["OV"], int)
    np.testing.assert_array_equal(np.bincount(res.lines[res.lines >= 0], minlength=len(segs)),
                                  ov)
    hits = segment_ray_hits(segs, r["viewpoint"], r["precision"], max_distance=r["range"],
                            direction=r["direction"], layout="decodingspaces")
    np.testing.assert_array_equal(hits, ov)


def _silhouettes(res):
    """Summed length of the edges between the pillar's west face (line 7) and a wall."""
    lines = res.lines
    edges = np.linalg.norm(np.diff(res.points, axis=0), axis=1)
    k = np.nonzero((lines[:-1] != lines[1:]) & ((lines[:-1] == 7) | (lines[1:] == 7)))[0]
    return float(edges[k].sum())


@pytest.mark.parametrize("r", RECORDS, ids=IDS)
def test_the_edge_back_to_the_first_ray_is_never_summed(r):
    """The last and the first ray end on the pillar and on a wall (or the other way round),
    4.2 m to 6.4 m apart. The plugin's Occlusivity never holds that edge: it sums the edges
    between consecutive rays of the sweep only (measured)."""
    res = _run(r)
    lines = res.lines
    assert 7 in (lines[0], lines[-1]) and lines[0] != lines[-1]
    closing = float(np.linalg.norm(res.points[0] - res.points[-1]))
    assert 4.1 < closing < 6.5
    assert res.properties["Occlusivity"] == pytest.approx(r["IP"][6], rel=1e-9)
    assert abs(r["IP"][6] - (res.properties["Occlusivity"] + closing)) > 4.0


@pytest.mark.parametrize("precision, bearing, corner_edge", [
    (12, 269, False), (12, 270, True), (12, 271, True), (12, 272, True), (11, 302, True),
    (11, 295, False)])
def test_a_corner_counts_when_the_next_ray_ends_nearer_the_far_end_of_its_wall(
        precision, bearing, corner_edge):
    """The sweep passes from the east wall to the south wall, which meet at (10, 0). The plugin
    looks up the end of the south wall nearest to the new hit point. When that hit lies more
    than halfway along the wall, 5.11 m to 5.64 m from (10, 0), the nearest end is (0, 0), which
    the east wall does not touch, and the plugin counts the edge as occluding. At 4.996 m and
    4.84 m from the corner it does not (measured)."""
    r = next(r for r in RECORDS if r["precision"] == precision and _bearing(r) == bearing)
    res = _run(r)
    lines, pts = res.lines, res.points
    k = np.nonzero((lines[:-1] == 1) & (lines[1:] == 0))[0][0]
    edge = float(np.linalg.norm(pts[k + 1] - pts[k]))
    from_corner = float(np.linalg.norm(pts[k + 1] - (10.0, 0.0)))
    assert (from_corner > 5.0) == corner_edge
    assert r["IP"][6] == pytest.approx(_silhouettes(res) + (edge if corner_edge else 0.0),
                                       rel=1e-9)
    assert res.properties["Occlusivity"] == pytest.approx(r["IP"][6], rel=1e-9)
