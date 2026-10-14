"""Obstacle visibility: the exact visible part of every wall, its field, and ray hit counts."""

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pysovist import Plan, isovist, visibility2d
from pysovist.results import SEGMENT
from pysovist.visibility2d import segment_ray_hits, segment_visibility, segment_visibility_field

from .geometry import rect

TWO_PI = 2 * math.pi
SETTINGS = settings(max_examples=120, deadline=None)


def approx(x):
    return pytest.approx(x, rel=1e-9, abs=1e-9)


def pillar_scene():
    """10 m square room with a 1 m x 2 m pillar, the plugin's reference scene."""
    return np.concatenate([rect(0, 0, 10, 10), rect(6, 4, 7, 6)])


# --- exact visibility: analytic scenes ----------------------------------------------------------

@pytest.mark.parametrize("origin", [(5, 5), (2, 3), (0.5, 9.5)])
def test_square_room_every_wall_fully_visible(origin):
    v = segment_visibility(rect(0, 0, 10, 10), origin)
    assert v.visible_length == approx([10, 10, 10, 10])
    assert v.visible_fraction == approx([1, 1, 1, 1])
    assert v.visible.tolist() == [True] * 4
    for iv in v.intervals:
        np.testing.assert_allclose(iv, [[0.0, 1.0]], atol=1e-12)
    assert v.visible_angle.sum() == approx(TWO_PI)


def test_square_from_centre_each_wall_fills_a_quarter_turn():
    v = segment_visibility(rect(0, 0, 10, 10), (5, 5))
    assert v.visible_angle == approx([math.pi / 2] * 4)


def test_pillar_hides_the_middle_of_the_far_wall():
    # The pillar's near face x = 6 spans y in [4, 6]. Seen from (2, 5), its corners cast rays
    # of slope -1/4 and +1/4 that meet the far wall x = 10 at y = 3 and y = 7.
    v = segment_visibility(pillar_scene(), (2, 5))
    assert v.visible_length == approx([10, 6, 10, 10, 0, 0, 0, 2])
    assert v.visible_fraction == approx([1, 0.6, 1, 1, 0, 0, 0, 1])
    assert v.visible.tolist() == [True, True, True, True, False, False, False, True]
    np.testing.assert_allclose(v.intervals[1], [[0.0, 0.3], [0.7, 1.0]], atol=1e-12)
    np.testing.assert_allclose(v.intervals[7], [[0.0, 1.0]], atol=1e-12)
    assert len(v.intervals[4]) == 0
    assert v.visible_angle[7] == approx(2 * math.atan(1 / 4))
    assert v.visible_angle[1] == approx(2 * (math.atan(5 / 8) - math.atan(1 / 4)))


def test_visible_pieces_as_coordinates():
    v = segment_visibility(pillar_scene(), (2, 5))
    index, pieces = v.pieces()
    right = pieces[index == 1]
    np.testing.assert_allclose(right, [[[10, 0], [10, 3]], [[10, 7], [10, 10]]], atol=1e-12)
    assert np.linalg.norm(pieces[:, 1] - pieces[:, 0], axis=1).sum() == approx(38.0)


def test_range_cuts_walls_at_the_circle():
    # Both long walls of the hallway cross the range circle at x = +-sqrt(99).
    v = segment_visibility(rect(-20, -1, 20, 1), (0, 0), max_distance=10)
    half = math.sqrt(99)
    assert v.visible_length == approx([2 * half, 0, 2 * half, 0])
    assert v.visible_fraction[0] == approx(2 * half / 40)
    lo, hi = (20 - half) / 40, (20 + half) / 40
    np.testing.assert_allclose(v.intervals[0], [[lo, hi]], atol=1e-12)
    np.testing.assert_allclose(v.intervals[2], [[lo, hi]], atol=1e-12)
    assert v.visible.tolist() == [True, False, True, False]


def test_range_shorter_than_every_wall_sees_nothing():
    v = segment_visibility(rect(0, 0, 10, 10), (5, 5), max_distance=4)
    assert v.visible_length == approx([0, 0, 0, 0])
    assert not v.visible.any()


def test_quarter_field_of_view_sees_one_wall():
    v = segment_visibility(rect(0, 0, 10, 10), (5, 5), fov=math.pi / 2, direction=0.0)
    assert v.visible_length == approx([0, 10, 0, 0])
    assert v.visible.tolist() == [False, True, False, False]


def test_quarter_field_of_view_across_a_corner_sees_two_halves():
    v = segment_visibility(rect(0, 0, 10, 10), (5, 5), fov=math.pi / 2, direction=math.pi / 4)
    assert v.visible_length == approx([0, 5, 5, 0])
    np.testing.assert_allclose(v.intervals[1], [[0.5, 1.0]], atol=1e-12)
    np.testing.assert_allclose(v.intervals[2], [[0.0, 0.5]], atol=1e-12)


def test_fields_of_view_add_up_to_the_full_turn():
    walls = np.concatenate([rect(-5, -5, 5, 5), rect(-1, -1, 1, 1)])
    whole = segment_visibility(walls, (3, 0.5))
    parts = [segment_visibility(walls, (3, 0.5), fov=math.pi / 3, direction=k * math.pi / 3 + 0.1)
             for k in range(6)]
    assert sum(p.visible_length for p in parts) == approx(whole.visible_length)
    assert sum(p.visible_angle for p in parts) == approx(whole.visible_angle)


def test_viewpoint_on_a_wall_is_flagged():
    v = segment_visibility(rect(0, 0, 10, 10), (5, 0))
    assert v.flags.on_wall
    assert np.isnan(v.visible_length).all()
    assert np.isnan(v.visible_fraction).all()
    assert not v.visible.any()
    assert all(len(iv) == 0 for iv in v.intervals)


def test_viewpoint_on_a_wall_can_raise():
    with pytest.raises(ValueError):
        segment_visibility(rect(0, 0, 10, 10), (0, 0), on_wall="raise")


def test_wall_split_into_touching_pieces_is_fully_visible():
    walls = np.concatenate([[[(0, 0), (4, 0)], [(4, 0), (10, 0)]], rect(0, 0, 10, 10)[1:]])
    v = segment_visibility(walls, (5, 5))
    assert v.visible_length == approx([4, 6, 10, 10, 10])
    for iv in v.intervals:
        np.testing.assert_allclose(iv, [[0.0, 1.0]], atol=1e-12)


def test_duplicated_wall_counts_once_for_the_first_copy():
    walls = np.concatenate([rect(0, 0, 10, 10), [[(10, 0), (0, 0)]]])  # bottom wall, reversed
    v = segment_visibility(walls, (3, 7))
    assert v.visible_length == approx([10, 10, 10, 10, 0])
    assert v.visible_length.sum() == approx(40.0)


def test_overlapping_collinear_walls_share_the_overlap_with_the_first():
    walls = np.concatenate([[[(0, 0), (6, 0)], [(4, 0), (10, 0)]], rect(0, 0, 10, 10)[1:]])
    v = segment_visibility(walls, (5, 5))
    assert v.visible_length[:2] == approx([6, 4])
    np.testing.assert_allclose(v.intervals[1], [[1 / 3, 1.0]], atol=1e-12)


def test_wall_seen_edge_on_has_zero_visible_length():
    walls = np.concatenate([rect(0, 0, 10, 10), [[(5, 6), (5, 8)]]])
    v = segment_visibility(walls, (5, 5))
    assert v.visible_length[4] == 0.0
    assert not v.visible[4]
    assert v.visible_length[:4] == approx([10, 10, 10, 10])


def test_open_scene_without_range_still_has_visible_walls():
    v = segment_visibility([[(-1, 2), (1, 2)]], (0, 0))
    assert v.visible_length == approx([2.0])
    assert v.visible_angle == approx([2 * math.atan(1 / 2)])
    room_without_top = rect(0, 0, 10, 10)[[0, 1, 3]]
    w = segment_visibility(room_without_top, (5, 5))
    assert w.visible_length == approx([10, 10, 10])


def test_zero_length_segments_keep_their_rows():
    walls = np.concatenate([rect(0, 0, 10, 10)[:2], [[(3, 3), (3, 3)]], rect(0, 0, 10, 10)[2:]])
    v = segment_visibility(walls, (5, 5))
    assert len(v.visible_length) == 5
    assert v.length[2] == 0.0
    assert v.visible_length == approx([10, 10, 0, 10, 10])
    assert v.visible_fraction[2] == 0.0
    assert not v.visible[2]


def test_plan_and_array_input_agree():
    walls = pillar_scene()
    a = segment_visibility(walls, (2.3, 6.1), max_distance=7)
    b = segment_visibility(Plan(walls), (2.3, 6.1), max_distance=7)
    np.testing.assert_array_equal(a.visible_length, b.visible_length)


def test_segment_parameters_follow_the_order_of_the_endpoints():
    # From (2, 4.5) the pillar's corners (6, 4) and (6, 6) shade y in [3.5, 7.5] of the far wall.
    # Written from (10, 0) to (10, 10), the wall is seen on [0, 0.35] and [0.75, 1]. Written the
    # other way round, on [0, 0.25] and [0.65, 1].
    room = rect(0, 0, 10, 10)
    pillar = rect(6, 4, 7, 6)
    flipped = room.copy()
    flipped[1] = flipped[1][::-1]
    a = segment_visibility(np.concatenate([room, pillar]), (2, 4.5))
    b = segment_visibility(np.concatenate([flipped, pillar]), (2, 4.5))
    np.testing.assert_allclose(a.intervals[1], [[0.0, 0.35], [0.75, 1.0]], atol=1e-12)
    np.testing.assert_allclose(b.intervals[1], [[0.0, 0.25], [0.65, 1.0]], atol=1e-12)


# --- exact visibility: properties ---------------------------------------------------------------

@st.composite
def room_with_free_walls(draw):
    """Square room with random interior walls that may cross each other."""
    size = draw(st.floats(5, 20))
    n = draw(st.integers(1, 8))
    coords = st.floats(0.05 * size, 0.95 * size)
    walls = np.array([[[draw(coords), draw(coords)], [draw(coords), draw(coords)]]
                      for _ in range(n)])
    walls = walls[np.linalg.norm(walls[:, 1] - walls[:, 0], axis=1) > 1e-2]
    o = np.array([draw(coords), draw(coords)])
    return np.concatenate([rect(0, 0, size, size), walls]), o


def _wall_part_of_perimeter(iso):
    """Length of the isovist boundary that lies on walls: the chords of its wall wedges."""
    seg = iso.kind == SEGMENT
    th0, th1, p, phi = iso.theta0[seg], iso.theta1[seg], iso.p[seg], iso.phi[seg]
    r0, r1 = p / np.cos(th0 - phi), p / np.cos(th1 - phi)
    p0 = r0[:, None] * np.c_[np.cos(th0), np.sin(th0)]
    p1 = r1[:, None] * np.c_[np.cos(th1), np.sin(th1)]
    return np.hypot(*(p1 - p0).T).sum(), (th1 - th0).sum()


@SETTINGS
@given(room_with_free_walls(), st.one_of(st.just(math.inf), st.floats(0.5, 30)),
       st.floats(0.1, TWO_PI), st.floats(-math.pi, math.pi))
def test_visible_length_is_the_wall_part_of_the_isovist_perimeter(scene, R, fov, direction):
    walls, o = scene
    iso = isovist(walls, o, max_distance=R, fov=fov, direction=direction)
    assume(not iso.flags.on_wall and iso.clearance > 1e-4)
    v = segment_visibility(walls, o, max_distance=R, fov=fov, direction=direction)
    on_walls, wall_angle = _wall_part_of_perimeter(iso)
    assert np.all(v.visible_length <= v.length * (1 + 1e-12))
    assert np.all(v.visible_length >= 0)
    assert v.visible_length.sum() == pytest.approx(on_walls, rel=1e-9, abs=1e-9)
    assert v.visible_length.sum() <= iso.perimeter * (1 + 1e-12)
    assert v.visible_angle.sum() == pytest.approx(wall_angle, rel=1e-9, abs=1e-12)


@SETTINGS
@given(room_with_free_walls(), st.one_of(st.just(math.inf), st.floats(0.5, 30)))
def test_intervals_are_sorted_disjoint_and_add_up(scene, R):
    walls, o = scene
    v = segment_visibility(walls, o, max_distance=R)
    assume(not v.flags.on_wall)
    for j, iv in enumerate(v.intervals):
        assert np.all(iv >= 0) and np.all(iv <= 1)
        assert np.all(iv[:, 1] > iv[:, 0])
        assert np.all(iv[1:, 0] > iv[:-1, 1])
        assert (iv[:, 1] - iv[:, 0]).sum() * v.length[j] == pytest.approx(
            v.visible_length[j], rel=1e-12, abs=1e-12)
    assert v.visible_fraction == pytest.approx(v.visible_length / v.length)
    assert np.array_equal(v.visible, v.visible_length > 0)


@SETTINGS
@given(room_with_free_walls(), st.floats(-1e3, 1e3), st.floats(-1e3, 1e3), st.floats(0.01, 100))
def test_moving_and_scaling_the_scene(scene, dx, dy, k):
    walls, o = scene
    base = segment_visibility(walls, o)
    assume(not base.flags.on_wall and base.clearance > 1e-4)
    shift = np.array([dx, dy])
    moved = segment_visibility(walls * k + shift, o * k + shift)
    assert moved.visible_length == pytest.approx(base.visible_length * k, rel=1e-7, abs=1e-7 * k)
    assert moved.visible_fraction == pytest.approx(base.visible_fraction, abs=1e-7)


try:
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union
except ImportError:  # the shadow oracle needs shapely, every other test runs without it
    LineString = None


def _shadow(a, b, o, far=1e4):
    """Region hidden behind the wall from a to b, seen from o (None when seen edge-on)."""
    da, db = a - o, b - o
    cross = da[0] * db[1] - da[1] * db[0]
    if abs(cross) < 1e-12:
        return None
    ta, tb = np.arctan2(da[1], da[0]), np.arctan2(db[1], db[0])
    span = np.mod(tb - ta, TWO_PI) if cross > 0 else -np.mod(ta - tb, TWO_PI)
    arc = ta + span * np.linspace(0, 1, 65)
    far_pts = o + far * np.c_[np.cos(arc), np.sin(arc)]
    return Polygon(np.vstack([a, far_pts, b])).buffer(0)


def _overlapping(walls, tol=1e-9):
    """True when two walls lie on one line and share a stretch of positive length."""
    a, d = walls[:, 0], walls[:, 1] - walls[:, 0]
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            w = a[j] - a[i]
            if abs(d[i, 0] * d[j, 1] - d[i, 1] * d[j, 0]) > tol or \
                    abs(d[i, 0] * w[1] - d[i, 1] * w[0]) > tol:
                continue
            dd = d[i] @ d[i]
            s = sorted([(a[j] - a[i]) @ d[i] / dd, (walls[j, 1] - a[i]) @ d[i] / dd])
            if min(1.0, s[1]) - max(0.0, s[0]) > tol:
                return True
    return False


@pytest.mark.skipif(LineString is None, reason="the shadow oracle needs shapely")
@SETTINGS
@given(room_with_free_walls())
def test_matches_shadow_subtraction(scene):
    # Independent oracle: the visible part of wall j is wall j minus the shadows of all others.
    # Overlapping walls hide each other in this oracle, so they are left to the analytic tests.
    walls, o = scene
    assume(not _overlapping(walls))
    v = segment_visibility(walls, o)
    assume(not v.flags.on_wall and v.clearance > 1e-3)
    shadows = [_shadow(a, b, o) for a, b in walls]
    for j, (a, b) in enumerate(walls):
        if shadows[j] is None:  # a wall seen edge-on shows no length
            assert v.visible_length[j] == 0.0
            continue
        others = [s for k, s in enumerate(shadows) if k != j and s is not None]
        line = LineString([a, b])
        seen = line.difference(unary_union(others)).length if others else line.length
        assert v.visible_length[j] == pytest.approx(seen, rel=1e-6, abs=1e-6)


# --- field over many viewpoints -----------------------------------------------------------------

def _path():
    return np.c_[np.linspace(1, 9, 17), np.full(17, 2.0)]


def test_field_sums_and_means_over_the_path():
    walls = pillar_scene()
    pts = _path()
    singles = np.array([segment_visibility(walls, p).visible_length for p in pts])
    df = segment_visibility_field(walls, pts)
    assert list(df.index) == list(range(len(walls)))
    assert {"x0", "y0", "x1", "y1", "length", "seen_by", "visible_length_sum",
            "visible_length_mean", "visible_fraction_mean"} <= set(df.columns)
    assert df["visible_length_sum"].to_numpy() == approx(singles.sum(axis=0))
    assert df["visible_length_mean"].to_numpy() == approx(singles.mean(axis=0))
    assert df["seen_by"].tolist() == (singles > 0).sum(axis=0).tolist()
    assert df["visible_fraction_mean"].to_numpy() == approx(singles.mean(axis=0) / df["length"])
    assert df.attrs["viewpoints"] == len(pts)
    assert df.attrs["on_wall"] == 0


def test_field_per_viewpoint_table_lists_every_visible_pair():
    walls = pillar_scene()
    pts = _path()
    df, pairs = segment_visibility_field(walls, pts, per_viewpoint=True)
    assert {"viewpoint", "x", "y", "segment", "visible_length", "visible_fraction",
            "visible_angle"} <= set(pairs.columns)
    assert (pairs["visible_length"] > 0).all()
    wide = pairs.pivot_table(index="viewpoint", columns="segment", values="visible_length",
                             fill_value=0.0, aggfunc="sum")
    for i, p in enumerate(pts):
        v = segment_visibility(walls, p)
        row = wide.loc[i].reindex(range(len(walls)), fill_value=0.0).to_numpy()
        assert row == approx(v.visible_length)
    assert pairs.groupby("segment")["visible_length"].sum().to_numpy() == approx(
        df.loc[df["seen_by"] > 0, "visible_length_sum"].to_numpy())


def test_field_leaves_out_viewpoints_on_walls():
    walls = rect(0, 0, 10, 10)
    pts = np.array([[5, 5], [5, 0], [2, 3]])
    df = segment_visibility_field(walls, pts)
    assert df.attrs["viewpoints"] == 2
    assert df.attrs["on_wall"] == 1
    assert df["seen_by"].tolist() == [2, 2, 2, 2]
    assert df["visible_length_mean"].to_numpy() == approx([10, 10, 10, 10])


def test_field_passes_options_to_every_viewpoint():
    walls = rect(-20, -1, 20, 1)
    df = segment_visibility_field(walls, [[0, 0], [5, 0]], max_distance=10)
    assert df["visible_length_sum"].to_numpy() == approx([4 * math.sqrt(99), 0, 4 * math.sqrt(99),
                                                          0])


def test_field_in_parallel_equals_serial():
    walls = pillar_scene()
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.2, 5.8, size=(60, 2))
    serial = segment_visibility_field(walls, pts, n_jobs=1, chunk=16, max_distance=6)
    parallel = segment_visibility_field(walls, pts, n_jobs=2, chunk=16, max_distance=6)
    np.testing.assert_array_equal(serial.to_numpy(), parallel.to_numpy())


# --- ray hits -----------------------------------------------------------------------------------

# Hits per input line of the DeCodingSpaces Isovist component (output OV), view direction
# (0, 1, 0) and view angle 360. Each entry: scene, Precision input, Range input, hits per line.
DS_SCENES = {
    "pillar": ([[[0, 0], [10, 0]], [[10, 0], [10, 10]], [[10, 10], [0, 10]], [[0, 10], [0, 0]],
                [[6, 4], [7, 4]], [[7, 4], [7, 6]], [[7, 6], [6, 6]], [[6, 6], [6, 4]]],
               (2.0, 5.0)),
    "rectangle_centre": ([[[0, 0], [20, 0]], [[20, 0], [20, 5]], [[20, 5], [0, 5]],
                          [[0, 5], [0, 0]]], (10.0, 2.5)),
    "square_centre": ([[[0, 0], [10, 0]], [[10, 0], [10, 10]], [[10, 10], [0, 10]],
                       [[0, 10], [0, 0]]], (5.0, 5.0)),
    "square_offset": ([[[0, 0], [10, 0]], [[10, 0], [10, 10]], [[10, 10], [0, 10]],
                       [[0, 10], [0, 0]]], (2.0, 3.0)),
}
DS_OV = [
    ("pillar", 2999, 1000.0, [666, 299, 665, 1136, 0, 0, 0, 234]),
    ("pillar", 360, 1000.0, [80, 36, 79, 137, 0, 0, 0, 29]),
    ("pillar", 2999, 4.0, [0, 0, 0, 1000, 0, 0, 0, 0]),
    ("rectangle_centre", 2999, 1000.0, [1266, 234, 1266, 234]),
    ("rectangle_centre", 360, 1000.0, [152, 29, 151, 29]),
    ("rectangle_centre", 2999, 4.0, [856, 0, 856, 0]),
    ("square_centre", 2999, 1000.0, [750, 750, 750, 750]),
    ("square_centre", 360, 1000.0, [91, 90, 90, 90]),
    ("square_centre", 2999, 4.0, [0, 0, 0, 0]),
    ("square_offset", 2999, 1000.0, [860, 514, 540, 1086]),
    ("square_offset", 360, 1000.0, [104, 62, 64, 131]),
    ("square_offset", 2999, 4.0, [626, 0, 0, 969]),
]


@pytest.mark.parametrize("binned", [False, True], ids=["direct", "binned"])
@pytest.mark.parametrize("scene,rays,rng,ov", DS_OV)
def test_ray_hits_match_decodingspaces(scene, rays, rng, ov, binned, monkeypatch):
    if binned:
        monkeypatch.setattr(visibility2d, "_DIRECT_WORK", 0)
    walls, vp = DS_SCENES[scene]
    hits = segment_ray_hits(walls, vp, rays, max_distance=rng, direction=math.pi / 2,
                            layout="decodingspaces")
    assert hits.tolist() == ov


def test_square_from_centre_at_2999_rays_gives_750_per_wall():
    walls, vp = DS_SCENES["square_centre"]
    hits = segment_ray_hits(walls, vp, 2999, max_distance=1000, direction=math.pi / 2,
                            layout="decodingspaces")
    assert hits.tolist() == [750, 750, 750, 750]


@pytest.mark.parametrize("rays,direction,cast", [
    (15, math.pi / 2, 15), (15, 0.0, 15), (15, -math.pi / 2, 16), (100, math.pi / 2, 100),
    (100.5, math.pi / 2, 101), (360, 0.0, 361), (2999, math.pi / 2, 3000),
    (3000, math.pi / 2, 361), (10, math.pi / 2, 361), (5, math.pi / 2, 361),
    (100_000, math.pi / 2, 361)])
def test_decodingspaces_ray_count_rule(rays, direction, cast):
    """Ray counts of the plugin's full view (measured, ``data/decodingspaces2d``): the closing
    ray is cast or not depending on the rounding of the summed step. Precision outside
    (10, 3000) casts the one-degree rays of Precision 360."""
    # In a closed room every ray hits exactly one wall, so the hits add up to the rays cast.
    hits = segment_ray_hits(rect(0, 0, 10, 10), (2.1, 3.3), rays, direction=direction,
                            layout="decodingspaces")
    assert hits.sum() == cast


def test_even_layout_four_rays_hit_four_walls():
    hits = segment_ray_hits(rect(0, 0, 10, 10), (5, 5), 4)
    assert hits.tolist() == [1, 1, 1, 1]


@pytest.mark.parametrize("n", [7, 360, 1001])
def test_even_layout_casts_n_rays(n):
    assert segment_ray_hits(rect(0, 0, 10, 10), (2.1, 3.3), n).sum() == n


def test_ray_through_a_shared_corner_counts_for_the_wall_that_starts_there():
    # Five rays from -45 to +45 degrees: the outer two pass exactly through the corners (10, 0)
    # and (10, 10), where walls 1 and 2 start.
    hits = segment_ray_hits(rect(0, 0, 10, 10), (5, 5), 5, fov=math.pi / 2, direction=0.0)
    assert hits.tolist() == [0, 4, 1, 0]


def test_rays_beyond_the_range_do_not_count():
    hits = segment_ray_hits(rect(0, 0, 10, 10), (5, 5), 360, max_distance=4)
    assert hits.tolist() == [0, 0, 0, 0]


def test_ray_hits_keep_rows_of_zero_length_segments():
    walls = np.concatenate([rect(0, 0, 10, 10)[:2], [[(3, 3), (3, 3)]], rect(0, 0, 10, 10)[2:]])
    hits = segment_ray_hits(walls, (5, 5), 4)
    assert hits.tolist() == [1, 1, 0, 1, 1]


def test_ray_hits_reject_an_unknown_layout():
    with pytest.raises(ValueError):
        segment_ray_hits(rect(0, 0, 10, 10), (5, 5), 4, layout="spiral")


@settings(max_examples=40, deadline=None)
@given(room_with_free_walls())
def test_ray_hits_converge_to_the_visible_angle(scene):
    # Each visible interval loses or gains at most one ray at each end.
    walls, o = scene
    v = segment_visibility(walls, o)
    assume(not v.flags.on_wall and v.clearance > 1e-3)
    n = 20_000
    hits = segment_ray_hits(walls, o, n)
    step = TWO_PI / n
    pieces = np.array([len(iv) for iv in v.intervals])
    assert np.all(np.abs(hits * step - v.visible_angle) <= (pieces + 1) * step + 1e-12)


def test_binned_candidate_search_gives_the_same_visibility(monkeypatch):
    rng = np.random.default_rng(3)
    inner = rng.uniform(1, 19, size=(60, 2, 2))
    walls = np.concatenate([rect(0, 0, 20, 20), inner])
    pts = rng.uniform(0.5, 19.5, size=(10, 2))
    direct = [segment_visibility(walls, p, max_distance=12) for p in pts]
    monkeypatch.setattr(visibility2d, "_DIRECT_WORK", 0)
    binned = [segment_visibility(walls, p, max_distance=12) for p in pts]
    for a, b in zip(direct, binned, strict=True):
        np.testing.assert_array_equal(a.visible_length, b.visible_length)
