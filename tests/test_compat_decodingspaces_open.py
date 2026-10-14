"""DeCodingSpaces 2D layer against the plugin, beyond the default full view.

Reference: ``data/decodingspaces2d/ds_probe_2d_open.jsonl``, 30 runs of the DeCodingSpaces
"Isovist" component (release 2021.10.11, Rhino 7, headless). Each record holds the inputs, the
17 Isovist Properties (IP), the hits per obstacle line (OV), the area and the vertices of the
Isovist Polygon output, and the end points of every radial in output order (every radial
starts at the viewpoint). The runs cover five view directions, view angles of 90, 180 and 359
degrees, a Precision of 100.5, three viewpoints for the sign of the Drift Angle, a T-junction
and two walls 1 mm apart.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pysovist.compat.decodingspaces import (
    DS_PROPERTY_NAMES,
    ds_isovist,
    ds_ray_angles,
)
from pysovist.visibility2d import segment_ray_hits

DATA = Path(__file__).parent / "data" / "decodingspaces2d" / "ds_probe_2d_open.jsonl"
RECORDS = [json.loads(line) for line in DATA.read_text().splitlines()]
IDS = [f"{i:02d}-{r['tag']}-d{r['direction'][0]}{r['direction'][1]}-va{r['view_angle']:g}"
       f"-pr{r['precision']:g}" for i, r in enumerate(RECORDS)]


def _run(r):
    return ds_isovist(np.array(r["segments"], float), r["viewpoint"],
                      view_direction=r["direction"], view_range=r["range"],
                      view_angle_deg=r["view_angle"], precision=r["precision"])


def _plugin_rays(r):
    """Plugin's radials as (lengths, unit directions), in output order."""
    v = np.array(r["radial_ends"], float) - np.array(r["viewpoint"], float)
    length = np.hypot(v[:, 0], v[:, 1])
    return length, v / np.where(length > 0, length, 1.0)[:, None]


def test_records_cover_the_settings_named_in_the_probe():
    assert len(RECORDS) == 30
    assert {r["tag"] for r in RECORDS} == {"dir", "fov", "frac", "sign", "tee", "gap"}


# --- the rays ---------------------------------------------------------------------------------

@pytest.mark.parametrize("r", RECORDS, ids=IDS)
def test_rays_match_the_plugin_one_by_one(r):
    """Ray count, order and direction equal the plugin's radials (measured reference)."""
    res = _run(r)
    length, u = _plugin_rays(r)
    partial = r["view_angle"] < 360
    if partial:
        # A partial view lists the viewpoint first, as a radial of length 0.
        assert length[0] == 0.0 and (length[1:] > 0).all()
        length, u = length[1:], u[1:]
    else:
        assert (length > 0).all()
    assert len(res.angles) == len(u)
    mine = np.c_[np.cos(res.angles), np.sin(res.angles)]
    turn = np.arctan2(mine[:, 0] * u[:, 1] - mine[:, 1] * u[:, 0], (mine * u).sum(axis=1))
    assert np.abs(turn).max() < 1e-12
    np.testing.assert_allclose(res.depths, length, rtol=1e-9, atol=1e-12)


def test_rays_sweep_clockwise_from_the_left_edge_of_the_view():
    """A 90-degree view along +x: first ray at +45 degrees, then clockwise (measured)."""
    r = next(r for r in RECORDS if r["tag"] == "fov" and r["direction"] == [1, 0]
             and r["view_angle"] == 90.0 and r["precision"] == 360)
    ang = ds_ray_angles(360, view_angle_deg=90.0, view_direction=(1.0, 0.0, 0.0))
    assert ang[0] == pytest.approx(math.pi / 4, abs=1e-15)
    np.testing.assert_allclose(np.diff(ang), -math.pi / 180, rtol=1e-9)
    assert len(ang) == 91 == len(r["radial_ends"]) - 1
    assert ang[-1] == pytest.approx(-math.pi / 4, abs=1e-12)


@pytest.mark.parametrize("direction, rays", [((0, 1), 15), ((1, 0), 15), ((1, 1), 15),
                                             ((0, 2), 15), ((0, -1), 16)])
def test_closing_ray_follows_the_rounding_of_the_summed_step(direction, rays):
    """At Precision 15 the plugin casts the ray that closes the turn only for direction (0, -1)
    (measured): it adds the step to the angle until the angle reaches the far edge, so the
    rounding of the sum decides whether the last ray is cast."""
    r = next(r for r in RECORDS if r["tag"] == "dir" and r["precision"] == 15
             and tuple(r["direction"]) == direction)
    assert len(r["radial_ends"]) == rays
    assert len(ds_ray_angles(15, view_direction=direction)) == rays


@pytest.mark.parametrize("va, direction, rays", [(90.0, (0, 1), 90), (90.0, (1, 0), 91),
                                                 (180.0, (0, 1), 180), (180.0, (1, 0), 181),
                                                 (359.0, (0, 1), 360), (359.0, (1, 0), 360)])
def test_far_edge_of_a_partial_view_follows_the_same_rounding(va, direction, rays):
    """At Precision 360 the ray on the far edge of a 90 or 180 degree view is cast for
    direction (1, 0) and not for (0, 1) (measured)."""
    r = next(r for r in RECORDS if r["tag"] == "fov" and r["precision"] == 360
             and r["view_angle"] == va and tuple(r["direction"]) == direction)
    assert len(r["radial_ends"]) == rays + 1  # plus the viewpoint
    assert len(ds_ray_angles(360, view_angle_deg=va, view_direction=direction)) == rays


def test_non_integer_precision_sets_the_step_without_rounding():
    """Precision 100.5 gives a step of 360 / 100.5 degrees and 101 rays (measured)."""
    r = next(r for r in RECORDS if r["tag"] == "frac" and r["precision"] == 100.5)
    ang = ds_ray_angles(100.5)
    assert len(ang) == len(r["radial_ends"]) == 101
    np.testing.assert_allclose(np.diff(ang), -2 * math.pi / 100.5, rtol=1e-9)


# --- the 17 properties, the polygon and the hits per line -------------------------------------

@pytest.mark.parametrize("r", RECORDS, ids=IDS)
def test_properties_match_the_plugin(r):
    got = _run(r).properties
    np.testing.assert_allclose([got[k] for k in DS_PROPERTY_NAMES], r["IP"], rtol=1e-9,
                               atol=1e-9)


@pytest.mark.parametrize("r", RECORDS, ids=IDS)
def test_polygon_matches_the_plugin(r):
    """Same area as the plugin's Isovist Polygon, and every vertex of that polygon (which
    merges collinear edges) is a vertex of the layer's ring (measured)."""
    ring = _run(r).polygon
    x, y = ring[:, 0], ring[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert area == pytest.approx(r["poly_area"], rel=1e-9)
    verts = np.array(r["poly_vertices"], float)
    gap = np.linalg.norm(verts[:, None, :] - ring[None, :, :], axis=-1).min(axis=1)
    assert gap.max() < 1e-9
    if r["view_angle"] < 360:
        np.testing.assert_array_equal(ring[0], r["viewpoint"])


@pytest.mark.parametrize("r", RECORDS, ids=IDS)
def test_hits_per_line_match_the_plugin(r):
    """Obstacle visibility (OV): the layer's hit lines and the obstacle-visibility module's
    plugin layout both give the plugin's counts (measured)."""
    segs = np.array(r["segments"], float)
    ov = np.array(r["OV"], int)
    lines = _run(r).lines
    np.testing.assert_array_equal(np.bincount(lines[lines >= 0], minlength=len(segs)), ov)
    dx, dy = r["direction"]
    for direction in (r["direction"], math.atan2(dy, dx)):
        hits = segment_ray_hits(segs, r["viewpoint"], r["precision"], max_distance=r["range"],
                                fov=math.radians(r["view_angle"]), direction=direction,
                                layout="decodingspaces")
        np.testing.assert_array_equal(hits, ov)


# --- conventions, each on the record that shows it ----------------------------------------------

def test_partial_view_counts_the_viewpoint_as_a_radial_of_length_zero():
    """Min Radial is 0 and Mean Radial averages the rays and the viewpoint (measured)."""
    r = next(r for r in RECORDS if r["tag"] == "fov" and r["direction"] == [1, 0]
             and r["view_angle"] == 90.0 and r["precision"] == 360)
    res = _run(r)
    p = res.properties
    assert p["Min Radial"] == 0.0 == r["IP"][7]
    assert p["Mean Radial"] == pytest.approx(res.depths.sum() / (len(res.depths) + 1), rel=1e-12)
    assert p["Mean Radial"] == pytest.approx(r["IP"][9], rel=1e-9)


def test_partial_view_occlusivity_is_the_edge_from_the_viewpoint_to_the_first_ray():
    """A 90-degree view along +x from (2, 3) in a 10 m square: Occlusivity is the length of the
    first ray, at +45 degrees to (9, 10), 7 sqrt(2) (closed form, measured)."""
    r = next(r for r in RECORDS if r["tag"] == "fov" and r["direction"] == [1, 0]
             and r["view_angle"] == 90.0 and r["precision"] == 360)
    occ = _run(r).properties["Occlusivity"]
    assert occ == pytest.approx(7 * math.sqrt(2), rel=1e-12)
    assert r["IP"][6] == pytest.approx(7 * math.sqrt(2), rel=1e-12)


@pytest.mark.parametrize("vp, sign", [((8.0, 3.0), -1), ((8.0, 7.0), 1), ((2.0, 7.0), 1),
                                      ((2.0, 3.0), 1)])
def test_drift_angle_carries_a_sign(vp, sign):
    """Drift Angle is the signed angle from the direction point to the drift point, seen from
    the viewpoint, counter-clockwise positive (measured)."""
    r = next(r for r in RECORDS if r["tag"] in ("sign", "dir") and r["precision"] == 360
             and tuple(r["viewpoint"]) == vp and r["direction"] == [0, 1])
    got = _run(r).properties["Drift Angle"]
    assert np.sign(got) == np.sign(r["IP"][16]) == sign
    assert got == pytest.approx(r["IP"][16], rel=1e-9)


def test_drift_angle_near_a_half_turn_keeps_its_sign():
    r = next(r for r in RECORDS if r["tag"] == "gap")
    got = _run(r).properties["Drift Angle"]
    assert got == pytest.approx(r["IP"][16], rel=1e-9)
    assert -math.pi < got < -3.14


@pytest.mark.parametrize("precision", [360, 2999])
def test_t_junction_counts_only_a_step_in_depth(precision):
    """An interior wall ending on the middle of another wall. At 360 rays the two rays on either
    side of the junction differ by 7 cm in depth and the edge between them counts. At 2,999
    rays they differ by 2 mm and it does not (measured)."""
    r = next(r for r in RECORDS if r["tag"] == "tee" and r["precision"] == precision)
    res = _run(r)
    lines, d = res.lines, res.depths
    at = np.nonzero((lines[:-1] == 3) & (lines[1:] == 4))[0]
    assert len(at) == 1
    step = abs(d[at[0] + 1] - d[at[0]])
    edge = np.linalg.norm(res.points[at[0] + 1] - res.points[at[0]])
    silhouette = r["IP"][6] - (edge if precision == 360 else 0.0)
    assert (step > 0.05) == (precision == 360)
    assert res.properties["Occlusivity"] == pytest.approx(r["IP"][6], rel=1e-9)
    # The rest of the plugin's Occlusivity is the edge from the interior wall to the top wall.
    top = np.nonzero((lines[:-1] == 4) & (lines[1:] == 2))[0]
    assert np.linalg.norm(res.points[top[0] + 1] - res.points[top[0]]) == pytest.approx(
        silhouette, rel=1e-9)


def test_walls_one_millimetre_apart_count_as_continuous():
    """Two walls 1 mm apart in depth, with a gap between their ends: the edge across the gap
    does not count (measured)."""
    r = next(r for r in RECORDS if r["tag"] == "gap")
    res = _run(r)
    lines = res.lines
    across = np.nonzero((lines[:-1] == 4) & (lines[1:] == 5))[0]
    assert len(across) == 1
    edges = np.linalg.norm(np.diff(res.points, axis=0), axis=1)
    silhouettes = np.nonzero(((lines[:-1] == 3) & (lines[1:] == 4))
                             | ((lines[:-1] == 5) & (lines[1:] == 1)))[0]
    assert len(silhouettes) == 2
    assert edges[silhouettes].sum() == pytest.approx(r["IP"][6], rel=1e-9)
    assert res.properties["Occlusivity"] == pytest.approx(r["IP"][6], rel=1e-9)
