"""DeCodingSpaces 2D layer against the plugin: directions with a negative x component, the
Occlusivity threshold, the edge back to the first ray, settings outside their range and list
matching.

Reference: ``data/decodingspaces2d/ds_probe_2d_more.jsonl``, 54 runs of the DeCodingSpaces
"Isovist" component (release 2021.10.11, Rhino 7, headless). 53 records hold one viewpoint
each, with the same fields as ``ds_probe_2d_open.jsonl``: the 17 Isovist Properties (IP), the
hits per line (OV), the Isovist Polygon's area and vertices, and every radial's end point (all
radials start at the viewpoint). Tags:

- ``negx``: 10 m square from (2, 3), directions (-1, 0), (-1, -1), (-1, 1) and (-2, 1), view
  angles 360 and 90 degrees, Precision 15 and 360.
- ``tee_steps``: the square with an interior wall from (0, 5) to (5, 5), seen from (3, 2) at
  Precision 600 to 2,400.
- ``gap_edge``: the square with a 1 m x 2 m pillar seen from (2, 5) at Precision 15, with the
  direction turned in steps of 15 degrees.
- ``outside``: view angles of 400, 0 and -90 degrees and ranges of 0 and -5 m.
- ``list_matching`` (the last record): one solution with three viewpoints and two directions.
  IP holds 3 x 17 values, one data branch per viewpoint.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pysovist.compat.decodingspaces import (
    DS_PROPERTY_NAMES,
    ds_isovist,
    ds_isovist_properties,
    ds_ray_angles,
)
from pysovist.visibility2d import segment_ray_hits

DATA = Path(__file__).parent / "data" / "decodingspaces2d" / "ds_probe_2d_more.jsonl"
ALL = [json.loads(line) for line in DATA.read_text().splitlines()]
RECORDS = [r for r in ALL if r["tag"] != "list_matching"]
LIST = next(r for r in ALL if r["tag"] == "list_matching")
OPEN = [json.loads(line) for line in
        (DATA.parent / "ds_probe_2d_open.jsonl").read_text().splitlines()]
IDS = [f"{i:02d}-{r['tag']}-d{r['direction'][0]:.2f},{r['direction'][1]:.2f}"
       f"-va{r['view_angle']:g}-pr{r['precision']:g}-R{r['range']:g}"
       for i, r in enumerate(RECORDS)]


def _run(r):
    return ds_isovist(np.array(r["segments"], float), r["viewpoint"],
                      view_direction=r["direction"], view_range=r["range"],
                      view_angle_deg=r["view_angle"], precision=r["precision"])


def _plugin_rays(r):
    v = np.array(r["radial_ends"], float) - np.array(r["viewpoint"], float)
    length = np.hypot(v[:, 0], v[:, 1])
    keep = length > 0  # a partial view lists the viewpoint first, as a radial of length 0
    return length[keep], v[keep] / length[keep, None]


def test_records_cover_the_four_conventions():
    assert len(ALL) == 54
    assert {r["tag"] for r in RECORDS} == {"negx", "tee_steps", "gap_edge", "outside"}


@pytest.mark.parametrize("r", RECORDS, ids=IDS)
def test_rays_match_the_plugin_one_by_one(r):
    """Ray count, order and direction equal the plugin's radials (measured reference)."""
    res = _run(r)
    length, u = _plugin_rays(r)
    assert len(res.angles) == len(u)
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
def test_polygon_matches_the_plugin(r):
    ring = _run(r).polygon
    x, y = ring[:, 0], ring[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert area == pytest.approx(r["poly_area"], rel=1e-9)
    verts = np.array(r["poly_vertices"], float)
    gap = np.linalg.norm(verts[:, None, :] - ring[None, :, :], axis=-1).min(axis=1)
    assert gap.max() < 1e-9


@pytest.mark.parametrize("r", RECORDS, ids=IDS)
def test_hits_per_line_match_the_plugin(r):
    segs = np.array(r["segments"], float)
    ov = np.array(r["OV"], int)
    lines = _run(r).lines
    np.testing.assert_array_equal(np.bincount(lines[lines >= 0], minlength=len(segs)), ov)
    va = r["view_angle"]
    fov = math.radians(va) if 0 < va <= 360 else 2 * math.pi  # the plugin's rule
    rng = r["range"] if r["range"] > 0 else 1000.0  # the plugin's rule
    # The component's direction vector gives its rays bit for bit. The same direction in
    # radians gives them too wherever the rounding of the bearing does not move the last ray.
    hits = segment_ray_hits(segs, r["viewpoint"], r["precision"], max_distance=rng, fov=fov,
                            direction=r["direction"], layout="decodingspaces")
    np.testing.assert_array_equal(hits, ov)


# --- 1. the bearing of a view direction -------------------------------------------------------

@pytest.mark.parametrize("direction, rays", [((-1, 0), 16), ((-2, 1), 16), ((-1, -1), 16),
                                             ((-1, 1), 15)])
def test_bearing_runs_from_0_to_2_pi(direction, rays):
    """The plugin measures the direction clockwise from +y between 0 and 2 pi: (-1, 0) has the
    bearing 3 pi / 2, not -pi / 2. At Precision 15 the summed step then casts the closing ray
    for (-1, 0), which a bearing of -pi / 2 would not (measured)."""
    r = next(r for r in RECORDS if r["tag"] == "negx" and tuple(r["direction"]) == direction
             and r["view_angle"] == 360.0 and r["precision"] == 15)
    assert len(r["radial_ends"]) == rays
    assert len(ds_ray_angles(15, view_direction=direction)) == rays


@pytest.mark.parametrize("direction", [(-1, 1), (-2, 1)])
def test_bearing_decides_the_far_edge_of_a_partial_view(direction):
    """A 90-degree view at Precision 360: the plugin casts 91 rays (measured)."""
    r = next(r for r in RECORDS if r["tag"] == "negx" and tuple(r["direction"]) == direction
             and r["view_angle"] == 90.0 and r["precision"] == 360)
    assert len(r["radial_ends"]) == 92  # the viewpoint and 91 rays
    assert len(ds_ray_angles(360, view_angle_deg=90.0, view_direction=direction)) == 91


# --- 2. the Occlusivity threshold at a T-junction ---------------------------------------------

@pytest.mark.parametrize("r", [r for r in RECORDS if r["tag"] == "tee_steps"],
                         ids=lambda r: f"pr{r['precision']}")
def test_t_junction_edge_counts_down_to_a_depth_step_of_11_mm(r):
    """At Precision 600 to 2,400 the two rays on either side of the junction differ by 4.4 cm
    down to 1.1 cm in depth, and the plugin counts the edge between them every time (measured).
    With the 2.2 mm step it leaves out at 2,999 rays, its threshold lies between 2.2 and 11 mm,
    and the module's 1 cm lies inside."""
    res = _run(r)
    lines, d = res.lines, res.depths
    k = np.nonzero((lines[:-1] == 3) & (lines[1:] == 4))[0][0]
    assert 0.011 < abs(d[k + 1] - d[k]) < 0.044
    edge = np.linalg.norm(res.points[k + 1] - res.points[k])
    top = np.nonzero((lines[:-1] == 4) & (lines[1:] == 2))[0][0]
    silhouette = np.linalg.norm(res.points[top + 1] - res.points[top])
    assert r["IP"][6] == pytest.approx(silhouette + edge, rel=1e-9)


# --- 3. the closing ray over the full circle of directions ------------------------------------

def _bearing_deg(r):
    return round(math.degrees(math.atan2(*r["direction"]))) % 360


@pytest.mark.parametrize("r", [r for r in RECORDS if r["tag"] == "gap_edge"],
                         ids=lambda r: f"bearing{_bearing_deg(r)}")
def test_closing_ray_over_the_full_circle_of_bearings(r):
    """At Precision 15 the plugin casts the closing ray (16 rays) for some of the 24 bearings
    and not for the others, and the summed step from the bearing in [0, 2 pi) gives the same
    choice for every one (measured)."""
    n = len(r["radial_ends"])
    assert n in (15, 16)
    assert len(ds_ray_angles(15, view_direction=r["direction"])) == n


# --- 4. settings outside their range and list matching ----------------------------------------

@pytest.mark.parametrize("va, rng", [(400.0, 1000.0), (0.0, 1000.0), (-90.0, 1000.0),
                                     (360.0, 0.0), (360.0, -5.0)])
def test_settings_outside_their_range_mean_the_defaults(va, rng):
    """A view angle outside (0, 360] gives the full view and a range of 0 or below gives
    1,000 m: the plugin returns the default run's properties (measured)."""
    r = next(r for r in RECORDS if r["tag"] == "outside" and r["view_angle"] == va
             and r["range"] == rng)
    default = ds_isovist(np.array(r["segments"], float), r["viewpoint"], precision=15)
    np.testing.assert_allclose([default.properties[k] for k in DS_PROPERTY_NAMES], r["IP"],
                               rtol=1e-9, atol=1e-9)
    # The plugin's own default run of the same scene (direction (0, 1), Precision 15).
    open_run = next(o for o in OPEN if o["tag"] == "dir" and o["direction"] == [0, 1]
                    and o["precision"] == 15)
    assert r["IP"] == open_run["IP"]


def test_a_direction_list_of_another_length_gives_every_viewpoint_the_first():
    """Three viewpoints and two directions in one solution: each viewpoint gets the first
    direction (measured, one data branch of 17 properties per viewpoint)."""
    assert LIST["paths"] == ["{0}", "{1}", "{2}"]
    df = ds_isovist_properties(np.array(LIST["segments"], float), LIST["viewpoints"],
                               view_direction=LIST["directions"], view_angle_deg=90.0,
                               precision=15)
    np.testing.assert_allclose(df[list(DS_PROPERTY_NAMES)].to_numpy().ravel(), LIST["IP"],
                               rtol=1e-9, atol=1e-9)
    # The radials come ray by ray, one per viewpoint in turn: the viewpoints first, then the
    # first ray of each, and so on.
    starts = [tuple(a) for a, _ in LIST["radials"]]
    assert starts == [tuple(v) for v in LIST["viewpoints"]] * 5
