"""DeCodingSpaces compatibility layer: ray layout, the 17 properties and parity with the plugin.

The reference values come from the DeCodingSpaces "Isovist" component (2D), release 2021.10.11,
run headless in Rhino 7. The four simple scenes and their 17 properties are embedded below. The
hospital-plan areas are read from the project tree when it is present and skipped otherwise.
"""

import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysovist import isovist, sampled_metrics
from pysovist.compat.decodingspaces import (
    DS_PROPERTY_NAMES,
    ds_angular_step,
    ds_isovist,
    ds_isovist_properties,
    ds_ray_angles,
)

from .geometry import rect

ONE_DEGREE = 0.017453292519943295  # the plugin's fallback step, as stored in its code

# --- reference values of the plugin -----------------------------------------------------------

SCENES = {
    'pillar': ([[[0, 0], [10, 0]], [[10, 0], [10, 10]], [[10, 10], [0, 10]], [[0, 10], [0, 0]],
                [[6, 4], [7, 4]], [[7, 4], [7, 6]], [[7, 6], [6, 6]], [[6, 6], [6, 4]]],
               (2.0, 5.0)),
    'rectangle_centre': ([[[0, 0], [20, 0]], [[20, 0], [20, 5]], [[20, 5], [0, 5]],
                          [[0, 5], [0, 0]]],
                         (10.0, 2.5)),
    'square_centre': ([[[0, 0], [10, 0]], [[10, 0], [10, 10]], [[10, 10], [0, 10]],
                       [[0, 10], [0, 0]]],
                      (5.0, 5.0)),
    'square_offset': ([[[0, 0], [10, 0]], [[10, 0], [10, 10]], [[10, 10], [0, 10]],
                       [[0, 10], [0, 0]]],
                      (2.0, 3.0)),
}

# (scene, precision, range, the 17 values of the plugin's Isovist Properties output)
ORACLE = [
    ('pillar', 2999, 1000.0, [
        87.97444956570402, 87.97444956570402, 46.207071638479164,
        0.5177850776461422, 0.8381955854277993, 0.8797463619433001,
        8.253040518376437, 2.000000274338571, 9.42995499861137,
        4.844801939737585, 5118.760307737032, 13607.30604004788,
        13046.165252264329, -5113.915505797295, 0.513767238597747,
        2.559185599393935, 0.5750323480247744,
    ]),
    ('pillar', 360, 1000.0, [
        87.43204355900242, 87.43204355900242, 45.95599885842178,
        0.5202308150712907, 0.8399660094383944, 0.8743583323585924,
        8.32200969516591, 2.0, 9.433427226896828,
        4.834941617729613, 612.6359855589495, 1623.8532190575763,
        1581.4965250509701, -607.8010439412199, 0.5125328792428805,
        2.534162033397747, 0.5700763445000709,
    ]),
    ('pillar', 2999, 4.0, [
        40.43848095669475, 40.43848095669475, 23.681392449006065,
        0.90612941177204, 0.9545283926541828, 0.9999999999999969,
        0.008465903210777684, 2.000000274338571, 4.000000000000001,
        3.505233114044568, 2016.8015293042927, 1759.0980151199226,
        -1423.406910367856, -2013.296296190248, 0.8763082785111418,
        0.6853079167698267, 0.20652921083222758,
    ]),
    ('rectangle_centre', 2999, 1000.0, [
        99.99927963191539, 99.99927963191539, 49.9695132061842,
        0.5032647336335218, 0.7569071649242586, 1.0,
        0.0, 2.5, 10.306794665290948,
        4.908454929287474, 7062.13787813234, 23188.535468763017,
        63251.91257574269, -7057.229423203053, 0.4762348614372939,
        3.483288390171424e-06, 0.09608904335624731,
    ]),
    ('rectangle_centre', 360, 1000.0, [
        99.99099694522573, 99.99099694522573, 49.97325493594902,
        0.5031476950180926, 0.7553332374756561, 1.0000000000000002,
        0.0, 2.5, 10.306136293499012,
        4.903145835112379, 849.4803685083226, 2790.8445248092553,
        7642.77937734428, -844.5772226732103, 0.47575014491174783,
        1.7763568394002505e-15, 0.09608871551736707,
    ]),
    ('rectangle_centre', 2999, 4.0, [
        37.2166658339842, 37.2166658339842, 23.290016363158436,
        0.8621994464026819, 0.9673526499299393, 0.999999999999999,
        0.01828401514148672, 2.5, 4.000000000000001,
        3.3852146211777114, 1753.0896515902514, 1154.7104421255751,
        -172.9841175510144, -1749.7044369690736, 0.8463036552944276,
        1.4358354625265692e-06, 0.0960885803799064,
    ]),
    ('square_centre', 2999, 1000.0, [
        99.99984942925695, 99.99984942925695, 39.98378632296449,
        0.7860340768035763, 0.9890068308594707, 0.9999999999999996,
        0.0, 5.0, 7.069216720586045,
        5.61079475433857, 1493.836388525917, 1043.077175802801,
        515.2758456087162, -1488.2255937715786, 0.7936939799850172,
        2.7335940240291734e-06, 0.11065749453322295,
    ]),
    ('square_centre', 360, 1000.0, [
        99.99999999999993, 99.99999999999993, 39.99999999999959,
        0.7853981633974639, 0.988560895807562, 0.9999999999999999,
        0.0, 5.0, 7.071067811865414,
        5.609533904258455, 179.7980273367594, 125.77904243471738,
        62.65515237507004, -174.18849343250093, 0.7933078925994074,
        3.2023728339893768e-15, 0.11065722117389552,
    ]),
    ('square_centre', 2999, 4.0, [
        50.26544568475403, 50.26544568475403, 25.132736632132826,
        0.999999634215209, 1.0000007315698123, 0.9999999999999979,
        0.0, 3.9999999999999987, 4.000000000000001,
        4.0, 6.80788758700146e-13, 4.754859106219649e-28,
        -1.4407099836339526e-43, 3.999999999999319, 0.9999999999999998,
        6.764165321960921e-15, 0.11065722117389601,
    ]),
    ('square_offset', 2999, 1000.0, [
        99.9997041118826, 99.9997041118826, 39.97974848333323,
        0.786191716707796, 0.7843760821520479, 0.9999999999999999,
        0.0, 2.000000274338571, 10.626371439626078,
        4.996738162816325, 7240.4521236216115, 20567.911759612838,
        24059.38641320954, -7235.455385458795, 0.47022054435094646,
        3.605547546634801, 1.5707938778393133,
    ]),
    ('square_offset', 360, 1000.0, [
        99.98422351568782, 99.98422351568782, 39.8437531324497,
        0.7914452209988729, 0.782977996290222, 1.0,
        0.0, 2.0, 10.600103946790579,
        4.991896605589791, 870.6278772456939, 2472.5090265892186,
        2916.9828930292074, -865.6359806401041, 0.47092902396501507,
        3.60517072333948, 1.57058671715853,
    ]),
    ('square_offset', 2999, 4.0, [
        36.972339118038505, 36.972339118038505, 22.875195386054735,
        0.8878859165821708, 0.9540808690545698, 0.9999999999999992,
        0.010986010089762487, 2.000000274338571, 4.000000000000001,
        3.3508587509025842, 1948.1678253519708, 1618.4922252892293,
        -765.9155489740331, -1944.8169666010683, 0.8377146877256458,
        0.8062442545434815, 0.345854741712438,
    ]),
]

ORACLE_IDS = [f"{s}-{n}-{r:g}" for s, n, r, _ in ORACLE]


def _project_root():
    """The project directory that holds the analysis data, or None."""
    env = os.environ.get("PYSOVIST_PROJECT")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / "analysis" / "agents" / "task" / "plan.json").exists():
            return parent
    return None


ROOT = _project_root()
AGENTS = ROOT / "analysis" / "agents" if ROOT else Path("missing")
needs_hospital = pytest.mark.skipif(ROOT is None,
                                    reason="hospital plan and plugin areas not available")


# --- ray-count rule and ray layout ------------------------------------------------------------

@pytest.mark.parametrize("pr", [11, 12, 15, 100, 360, 720, 2048, 2880, 2999, 10.5, 2999.5])
def test_step_inside_the_accepted_range(pr):
    assert ds_angular_step(pr) == 2 * math.pi / pr


@pytest.mark.parametrize("pr", [-5, 0, 1, 10, 3000, 3001, 3599, 3600, 36000])
def test_step_falls_back_to_one_degree(pr):
    assert ds_angular_step(pr) == ONE_DEGREE


@pytest.mark.parametrize("pr, n", [(360, 361), (15, 15), (100, 100), (2999, 3000), (3000, 361),
                                   (10, 361), (36000, 361)])
def test_full_view_sweeps_one_turn_clockwise_from_minus_y(pr, n):
    """Ray counts of the plugin at the default direction: 361, 15, 100 and 3,000 (measured,
    ``data/decodingspaces2d``). Precision outside (10, 3000) gives the one-degree rays of
    Precision 360 (the plugin's areas at 3,000 and above equal its 360-ray areas, see the
    hospital test below)."""
    ang = ds_ray_angles(pr)
    assert len(ang) == n
    first = np.array([math.cos(ang[0]), math.sin(ang[0])])
    np.testing.assert_allclose(first, [0.0, -1.0], atol=1e-15)
    np.testing.assert_allclose(np.diff(ang), -ds_angular_step(pr), rtol=1e-9)
    assert ang[0] - ang[-1] <= 2 * math.pi + 1e-12


def test_rays_start_opposite_the_view_direction():
    """Direction (1, 0): the first ray points along -x (measured, 361 rays)."""
    ang = ds_ray_angles(360, view_direction=(1.0, 0.0, 0.0))
    np.testing.assert_allclose([math.cos(ang[0]), math.sin(ang[0])], [-1.0, 0.0], atol=1e-15)
    assert len(ang) == 361


def test_partial_view_is_swept_from_the_left_edge_at_the_same_step():
    """A 90-degree view along +y at Precision 360: rays from 135 degrees down to 46 degrees;
    the summed step stops short of the right edge at 45 degrees (measured)."""
    ang = ds_ray_angles(360, view_angle_deg=90.0, view_direction=(0.0, 1.0, 0.0))
    assert len(ang) == 90
    assert ang[0] == pytest.approx(math.radians(135), abs=1e-15)
    assert ang[-1] == pytest.approx(math.radians(46), abs=1e-12)


@pytest.mark.parametrize("va", [0.0, -90.0, 400.0])
def test_view_angle_outside_the_turn_means_full_view(va):
    np.testing.assert_array_equal(ds_ray_angles(360, view_angle_deg=va), ds_ray_angles(360))


# --- the 17 properties against the plugin ---------------------------------------------------

@pytest.mark.parametrize("scene, pr, rng, ip", ORACLE, ids=ORACLE_IDS)
def test_properties_match_the_plugin(scene, pr, rng, ip):
    walls, vp = SCENES[scene]
    got = ds_isovist(np.array(walls, float), vp, view_range=rng, precision=pr).properties
    assert tuple(got) == DS_PROPERTY_NAMES
    np.testing.assert_allclose([got[k] for k in DS_PROPERTY_NAMES], ip, rtol=1e-9, atol=1e-9)


def test_property_names_follow_the_plugin():
    assert DS_PROPERTY_NAMES == (
        "Area", "Distance Weighted Area", "Perimeter", "Compactness", "Circularity",
        "Convex Deficiency", "Occlusivity", "Min Radial", "Max Radial", "Mean Radial",
        "Standard Deviation", "Variance", "Skewness", "Dispersion", "Elongation",
        "Drift Magnitude", "Drift Angle",
    )


# --- definitions on scenes with known answers -------------------------------------------------

def test_square_from_its_centre_at_one_degree():
    res = ds_isovist(rect(0, 0, 10, 10), (5, 5), precision=360)
    p = res.properties
    assert p["Area"] == pytest.approx(100.0, rel=1e-12)
    assert p["Distance Weighted Area"] == p["Area"]
    assert p["Perimeter"] == pytest.approx(40.0, rel=1e-12)
    assert p["Compactness"] == pytest.approx(math.pi / 4, rel=1e-12)
    assert p["Convex Deficiency"] == pytest.approx(1.0, rel=1e-12)
    assert p["Occlusivity"] == 0.0
    assert p["Min Radial"] == pytest.approx(5.0, rel=1e-12)
    assert p["Max Radial"] == pytest.approx(5 * math.sqrt(2), rel=1e-12)
    r = res.depths
    assert len(r) == 361
    m = r.mean()
    assert p["Mean Radial"] == pytest.approx(m, rel=1e-12)
    assert p["Standard Deviation"] == pytest.approx(np.abs(r - m).sum(), rel=1e-12)
    assert p["Variance"] == pytest.approx(((r - m) ** 2).sum(), rel=1e-12)
    assert p["Skewness"] == pytest.approx(((r - m) ** 3).sum(), rel=1e-9)
    assert p["Dispersion"] == pytest.approx(m - np.abs(r - m).sum(), rel=1e-12)
    assert p["Circularity"] == pytest.approx(math.pi * m * m / p["Area"], rel=1e-12)
    assert p["Elongation"] == pytest.approx(m / p["Max Radial"], rel=1e-12)


def test_open_view_is_the_inscribed_polygon_of_the_range_circle():
    n, rng = 2999, 4.0
    p = ds_isovist(np.empty((0, 2, 2)), (3.0, -2.0), view_range=rng, precision=n).properties
    assert p["Area"] == pytest.approx(n / 2 * rng**2 * math.sin(2 * math.pi / n), rel=1e-12)
    assert p["Perimeter"] == pytest.approx(2 * n * rng * math.sin(math.pi / n), rel=1e-12)
    assert p["Min Radial"] == p["Max Radial"] == p["Mean Radial"] == rng
    assert p["Standard Deviation"] == p["Variance"] == p["Skewness"] == 0.0
    assert p["Elongation"] == 1.0
    assert p["Occlusivity"] == 0.0
    assert p["Drift Magnitude"] == pytest.approx(0.0, abs=1e-12)


def test_polygon_area_is_the_ray_polygon_of_pysovist():
    walls, vp = SCENES["pillar"]
    for n in (12, 15, 360, 2999):
        res = ds_isovist(np.array(walls, float), vp, precision=n)
        ref = sampled_metrics(isovist(np.array(walls, float), vp, max_distance=1000.0), n,
                              offset=-math.pi / 2)
        assert res.properties["Area"] == pytest.approx(ref["area"], rel=1e-12)
        assert res.properties["Perimeter"] == pytest.approx(ref["perimeter"], rel=1e-12)
        ring = res.polygon
        x, y = ring[:, 0], ring[:, 1]
        shoelace = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        assert shoelace == pytest.approx(res.properties["Area"], rel=1e-12)


# --- occlusivity: which polygon edges count ---------------------------------------------------

def _single_wall():
    return np.array([[[-1.0, 2.0], [2.0, 2.0]]])


def test_occlusivity_counts_the_edge_from_a_miss_onto_a_hit_only():
    """One wall ahead, off-centre, range 3. The clockwise sweep meets the wall at its left end,
    2.24 m away, and leaves it at its right end, 2.83 m away. Only the edge from the last miss
    onto the first hit counts, as on the plugin's T-junction and partial-view records."""
    res = ds_isovist(_single_wall(), (0.0, 0.0), view_range=3.0, precision=360)
    lines, pts = res.lines, res.points
    enter = np.nonzero((lines[:-1] < 0) & (lines[1:] >= 0))[0]
    leave = np.nonzero((lines[:-1] >= 0) & (lines[1:] < 0))[0]
    assert len(enter) == len(leave) == 1
    assert pts[enter[0] + 1, 0] < 0 < pts[leave[0], 0]  # the sweep runs from -x to +x
    onto = np.linalg.norm(pts[enter[0] + 1] - pts[enter[0]])
    off = np.linalg.norm(pts[leave[0] + 1] - pts[leave[0]])
    assert onto > 0.7 and off < 0.2
    assert res.properties["Occlusivity"] == pytest.approx(onto, rel=1e-12)


def test_occlusivity_ignores_corners_and_counts_jumps_behind_a_pillar():
    walls, vp = SCENES["pillar"]
    res = ds_isovist(np.array(walls, float), vp, precision=2999)
    d, lines = res.depths, res.lines
    change = np.nonzero(lines[:-1] != lines[1:])[0]
    jumps = change[np.abs(np.diff(d))[change] > 1.0]
    assert len(jumps) == 2  # the two silhouette edges of the pillar
    edges = np.linalg.norm(np.diff(res.points, axis=0), axis=1)
    assert res.properties["Occlusivity"] == pytest.approx(edges[jumps].sum(), rel=1e-12)


# --- drift --------------------------------------------------------------------------------------

def test_drift_magnitude_is_the_centroid_offset_and_drift_angle_depends_on_position():
    walls = rect(0, 0, 10, 10)
    a = ds_isovist(walls, (2.0, 3.0), precision=360).properties
    b = ds_isovist(walls + 100.0, (102.0, 103.0), precision=360).properties
    assert a["Drift Magnitude"] == pytest.approx(b["Drift Magnitude"], rel=1e-9)
    assert a["Drift Angle"] != pytest.approx(b["Drift Angle"], rel=1e-3)
    assert a["Drift Angle"] == pytest.approx(math.pi / 2, abs=1e-3)


# --- many viewpoints, inputs as the component takes them -------------------------------------

def test_table_has_one_row_per_viewpoint_with_the_plugin_names():
    walls, _ = SCENES["pillar"]
    vps = np.array([[2.0, 5.0], [8.0, 8.0], [5.0, 2.0]])
    df = ds_isovist_properties(np.array(walls, float), vps, precision=360)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x", "y", *DS_PROPERTY_NAMES]
    np.testing.assert_array_equal(df[["x", "y"]].to_numpy(), vps)
    one = ds_isovist(np.array(walls, float), vps[1], precision=360).properties
    assert df.iloc[1]["Area"] == one["Area"]


def test_one_direction_per_viewpoint_when_the_counts_match():
    walls = rect(0, 0, 10, 10)
    vps = [(2.0, 3.0), (7.0, 6.0)]
    dirs = [(1.0, 0.0, 0.0), (0.0, -1.0, 0.0)]
    df = ds_isovist_properties(walls, vps, view_direction=dirs, view_angle_deg=90.0)
    for i in range(2):
        one = ds_isovist(walls, vps[i], view_direction=dirs[i], view_angle_deg=90.0).properties
        assert df.iloc[i]["Area"] == one["Area"]
    # Otherwise the first direction serves every viewpoint.
    df3 = ds_isovist_properties(walls, [*vps, (5.0, 5.0)], view_direction=dirs,
                                view_angle_deg=90.0)
    alt = ds_isovist(walls, vps[1], view_direction=dirs[0], view_angle_deg=90.0).properties
    assert df3.iloc[1]["Area"] == alt["Area"]


def test_non_positive_range_means_the_default_range():
    walls, vp = SCENES["square_offset"]
    a = ds_isovist(np.array(walls, float), vp, view_range=0.0).properties
    b = ds_isovist(np.array(walls, float), vp, view_range=1000.0).properties
    assert a == b


def test_observer_on_a_wall_gives_nan():
    p = ds_isovist(rect(0, 0, 10, 10), (0.0, 5.0)).properties
    assert all(math.isnan(v) for v in p.values())


# --- parity on the hospital plan --------------------------------------------------------------

@pytest.fixture(scope="module")
def hospital():
    import json
    walls = np.array(json.loads((AGENTS / "task" / "plan.json").read_text()), float)
    cand = pd.read_csv(AGENTS / "task" / "candidates.csv")
    return walls, cand


def _areas(walls, cand, pr):
    df = ds_isovist_properties(walls, cand[["x", "y"]].to_numpy(), view_range=40.0, precision=pr)
    return pd.Series(df["Area"].to_numpy(), index=cand["id"])


@needs_hospital
def test_hospital_areas_at_24_ray_counts(hospital):
    walls, cand = hospital
    files = sorted((AGENTS / "routes" / "route_outputs").glob("ds_*.csv"))
    counts = sorted(int(re.search(r"ds_(\d+)\.csv", f.name).group(1)) for f in files)
    assert len(counts) == 24 and {15, 17, 25, 35} <= set(counts)  # odd counts fix the start ray
    for pr in counts:
        ref = pd.read_csv(AGENTS / "routes" / "route_outputs" / f"ds_{pr}.csv").set_index("id")
        got = _areas(walls, cand, pr)
        np.testing.assert_allclose(got[ref.index], ref["ip_area"], rtol=1e-8, err_msg=f"{pr} rays")


@needs_hospital
def test_hospital_areas_across_the_fallback(hospital):
    walls, cand = hospital
    feas = AGENTS / "rhino" / "feasibility"
    ref = pd.concat([pd.read_csv(f) for f in sorted(feas.glob("sweep*.csv"))
                     + sorted(feas.glob("ds_areas*.csv"))])
    assert (ref["precision"] >= 3000).any()
    for pr, grp in ref.groupby("precision"):
        got = _areas(walls, cand, pr)
        np.testing.assert_allclose(got[grp["id"]], grp["area_m2"], rtol=1e-7,
                                   err_msg=f"{pr} rays")
    # At 3,000 rays and above the plugin returns its 360-ray areas.
    np.testing.assert_array_equal(_areas(walls, cand, 3000), _areas(walls, cand, 360))


# --- inputs outside the component's domain ------------------------------------------------------

SQUARE = rect(0, 0, 10, 10)


@pytest.mark.parametrize("vp", [(np.nan, 5.0), (np.inf, 5.0), (2.0, -np.inf)])
def test_viewpoint_must_be_finite(vp):
    with pytest.raises(ValueError, match="viewpoint"):
        ds_isovist(SQUARE, vp)
    with pytest.raises(ValueError, match="viewpoint"):
        ds_isovist_properties(SQUARE, [(5.0, 5.0), vp])


@pytest.mark.parametrize("d", [(0.0, 0.0), (0.0, 0.0, 1.0), (np.nan, 1.0), (np.inf, 1.0)])
def test_view_direction_must_be_a_finite_vector_in_the_plan(d):
    with pytest.raises(ValueError, match="view_direction"):
        ds_isovist(SQUARE, (2.0, 3.0), view_direction=d)


@pytest.mark.parametrize("kw, name", [({"view_range": np.nan}, "view_range"),
                                      ({"view_angle_deg": np.nan}, "view_angle_deg"),
                                      ({"precision": np.nan}, "precision")])
def test_nan_settings_are_refused(kw, name):
    with pytest.raises(ValueError, match=name):
        ds_isovist(SQUARE, (2.0, 3.0), **kw)


def test_nan_precision_has_no_step():
    with pytest.raises(ValueError, match="precision"):
        ds_angular_step(np.nan)


def test_infinite_range_is_refused_when_a_ray_meets_no_line():
    """Closed room: an infinite range changes nothing. Open room: a ray without a hit has no
    end, so the polygon does not exist."""
    a = ds_isovist(SQUARE, (2.0, 3.0), view_range=np.inf).properties
    b = ds_isovist(SQUARE, (2.0, 3.0), view_range=1000.0).properties
    assert a == b
    with pytest.raises(ValueError, match="view_range"):
        ds_isovist(SQUARE[:3], (2.0, 3.0), view_range=np.inf)


@pytest.mark.parametrize("va", [0.5, 1.0])
def test_view_with_one_ray_is_refused(va):
    """At one degree per ray a view of 1 degree or less holds one ray, and the polygon through
    it and the viewpoint has no area."""
    assert len(ds_ray_angles(360, view_angle_deg=va)) == 1
    with pytest.raises(ValueError, match="one ray"):
        ds_isovist(SQUARE, (2.0, 3.0), view_angle_deg=va)


def test_two_rays_give_the_triangle_with_the_viewpoint():
    """Closed form: a 2-degree view along +y from (5, 5) has rays at 91 and 90 degrees (and
    stops short of 89), ending on the wall y = 10. The triangle with the viewpoint has area
    25 tan(1 degree) / 2."""
    res = ds_isovist(SQUARE, (5.0, 5.0), view_angle_deg=2.0)
    assert len(res.angles) == 2
    assert res.properties["Area"] == pytest.approx(12.5 * math.tan(math.radians(1)), rel=1e-12)


def test_infinite_range_in_a_partial_view_that_meets_lines_only():
    """An open room seen through a 90-degree view onto its closed side: every ray hits, so an
    infinite range gives the rays of the default range."""
    walls = SQUARE[:3]  # the side x = 0 is open
    a = ds_isovist(walls, (2.0, 3.0), view_angle_deg=90.0, view_direction=(1, 0),
                   view_range=np.inf).properties
    b = ds_isovist(walls, (2.0, 3.0), view_angle_deg=90.0, view_direction=(1, 0)).properties
    assert a == b
