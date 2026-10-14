"""Flood-filled analysis grids from a seed point (DeCodingSpaces' Analysis Grid)."""

from collections import deque

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pysovist import Plan

from .geometry import polygon_walls, rect, regular_ngon

L_ROOM = Plan(polygon_walls([(0, 0), (10, 0), (10, 4), (4, 4), (4, 9), (0, 9)]))


def as_set(points):
    return {tuple(p) for p in np.asarray(points).tolist()}


@pytest.mark.parametrize("seed", [(2.0, 2.0), (9.7, 0.2), (1.0, 8.6)])
def test_l_shaped_room_fills_its_sixty_cells_and_not_the_notch(seed):
    centres, cells = L_ROOM.flood_grid(1.0, seed)
    assert len(centres) == 60  # 10 x 4 + 4 x 5 square metres
    assert not ((centres[:, 0] > 4) & (centres[:, 1] > 4)).any()
    expected = [(x + 0.5, y + 0.5) for y in range(9) for x in range(10) if x < 4 or y < 4]
    assert centres.tolist() == [list(p) for p in expected]
    assert cells == pytest.approx(np.c_[centres - 0.5, centres + 0.5])


def test_the_fill_is_an_ordered_subset_of_the_plan_grid():
    for spacing in (0.3, 0.7, 1.3):
        centres, _ = L_ROOM.flood_grid(spacing, (1.0, 1.0))
        grid = L_ROOM.grid(spacing)
        keep = np.array([tuple(p) in as_set(centres) for p in grid.tolist()])
        assert np.array_equal(grid[keep], centres)  # bit-identical, in the order of grid


def test_open_plan_fills_the_bounding_rectangle_of_the_walls():
    # Two walls meeting at a corner and one free wall enclose nothing.
    plan = Plan([[(0, 0), (6, 0)], [(0, 0), (0, 4)], [(3, 2), (5, 2)]])
    centres, cells = plan.flood_grid(0.5, (5.0, 3.0))
    assert np.array_equal(centres, plan.grid(0.5))
    assert cells[:, :2].min(axis=0).tolist() == [0.0, 0.0]
    assert cells[:, 2:].max(axis=0).tolist() == [6.0, 4.0]


def test_a_domain_bounds_the_fill_like_the_grid():
    plan = Plan([[(0, 0), (6, 0)]])
    # The flood passes around the ends of the wall and fills the whole domain.
    domain = (-1.0, -2.0, 7.0, 3.0)
    for seed in [(1.0, 1.0), (1.0, -1.0)]:
        centres, _ = plan.flood_grid(0.5, seed, domain)
        assert np.array_equal(centres, plan.grid(0.5, domain))
    # In a domain no wider than the wall, the wall splits the grid in two.
    domain = (0.0, -2.0, 6.0, 3.0)
    grid = plan.grid(0.5, domain)
    centres, _ = plan.flood_grid(0.5, (1.0, 1.0), domain)
    assert np.array_equal(centres, grid[grid[:, 1] > 0])
    centres, _ = plan.flood_grid(0.5, (1.0, -1.0), domain)
    assert np.array_equal(centres, grid[grid[:, 1] < 0])


def test_a_closed_wall_keeps_the_fill_in_one_room_and_a_door_lets_it_through():
    room = rect(0, 0, 10, 4)
    closed = Plan(np.concatenate([room, [[(5, 0), (5, 4)]]]))
    centres, _ = closed.flood_grid(1.0, (2.0, 2.0))
    assert len(centres) == 20 and (centres[:, 0] < 5).all()
    door = Plan(np.concatenate([room, [[(5, 0), (5, 1.2)], [(5, 2.8), (5, 4)]]]))
    centres, _ = door.flood_grid(1.0, (2.0, 2.0))
    assert len(centres) == 40


def test_a_door_whose_jambs_touch_the_links_is_closed():
    # The jambs end exactly on the rows of centres at y = 1.5 and y = 2.5. A link that
    # touches the end of a wall is blocked, and no row of centres passes between the jambs.
    room = rect(0, 0, 10, 4)
    door = Plan(np.concatenate([room, [[(5, 0), (5, 1.5)], [(5, 2.5), (5, 4)]]]))
    centres, _ = door.flood_grid(1.0, (2.0, 2.0))
    assert len(centres) == 20


def test_cells_cut_by_a_wall_are_kept_whole_on_the_side_of_their_centre():
    # The wall y = x + 0.2 cuts the diagonal cells of a 4 m x 4 m room. Their centres lie
    # below it, so they belong to the lower fill, whole, and to no upper fill.
    plan = Plan(np.concatenate([rect(0, 0, 4, 4), [[(0, 0.2), (3.8, 4)]]]))
    lower, cells = plan.flood_grid(1.0, (3.0, 1.0))
    assert as_set(lower) == {(i + 0.5, j + 0.5) for i in range(4) for j in range(4) if j <= i}
    assert (cells[:, 2] - cells[:, 0] == 1.0).all()
    upper, _ = plan.flood_grid(1.0, (0.5, 3.5))
    assert as_set(upper) == {(i + 0.5, j + 0.5) for i in range(4) for j in range(4) if j > i}


def test_no_centre_of_the_fill_lies_on_a_wall():
    # The wall x = 2.5 runs through a column of centres: those cells are never entered.
    plan = Plan(np.concatenate([rect(0, 0, 5, 3), [[(2.5, 0.8), (2.5, 2.2)]]]))
    centres, _ = plan.flood_grid(1.0, (0.5, 0.5))
    assert len(centres) == 15 - 1
    assert (2.5, 1.5) not in as_set(centres)


def test_a_seed_outside_the_domain_or_behind_a_wall_is_refused():
    with pytest.raises(ValueError, match="outside"):
        L_ROOM.flood_grid(1.0, (11.0, 2.0))
    # A wall between the seed and the centre of its cell.
    plan = Plan(np.concatenate([rect(0, 0, 4, 4), [[(0.3, 0), (0.3, 4)]]]))
    with pytest.raises(ValueError, match="wall"):
        plan.flood_grid(1.0, (0.1, 0.5))
    with pytest.raises(ValueError, match="spacing"):
        L_ROOM.flood_grid(0.0, (1.0, 1.0))


def test_a_larger_eps_closes_gaps_narrower_than_twice_it():
    # Two walls leave a gap of 0.02 m on the row of centres y = 0.5.
    plan = Plan(np.concatenate([rect(0, 0, 4, 1), [[(2, 0), (2, 0.49)], [(2, 0.51), (2, 1)]]]))
    assert len(plan.flood_grid(1.0, (0.5, 0.5))[0]) == 4
    assert len(plan.flood_grid(1.0, (0.5, 0.5), eps=0.05)[0]) == 2


def test_an_almost_horizontal_wall_blocks_without_a_warning():
    import warnings

    # The wall rises by 1e-320 m over 4 m and runs along the row of centres at y = 0.
    plan = Plan(np.concatenate([rect(0, -0.5, 4, 1.5), [[(0, 0), (4, 1e-320)]]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        centres, _ = plan.flood_grid(1.0, (0.5, 1.2))
    assert centres[:, 1].tolist() == [1.0] * 4


# --- property tests ---------------------------------------------------------------------------

def _oracle_flood(walls, spacing, seed, domain, eps):
    """Breadth-first fill whose link test is the GEOS segment distance."""
    from shapely import distance, linestrings

    xmin, ymin, xmax, ymax = domain
    xs = np.arange(xmin + spacing / 2, xmax, spacing)
    ys = np.arange(ymin + spacing / 2, ymax, spacing)
    gx, gy = np.meshgrid(xs, ys)
    nx, ny = len(xs), len(ys)
    wl = linestrings(walls)

    def blocked(a, b):
        if len(walls) == 0:
            return np.zeros(len(a), bool)
        lines = linestrings(np.stack([a, b], axis=1))
        return (distance(lines[:, None], wl[None, :]) < eps).any(axis=1)

    i, j = int(np.floor((seed[0] - xmin) / spacing)), int(np.floor((seed[1] - ymin) / spacing))
    if not (0 <= i < nx and 0 <= j < ny):
        return None
    if blocked(np.array([seed]), np.array([[xs[i], ys[j]]]))[0]:
        return None
    right = blocked(np.c_[gx[:, :-1].ravel(), gy[:, :-1].ravel()],
                    np.c_[gx[:, 1:].ravel(), gy[:, 1:].ravel()]).reshape(ny, nx - 1)
    up = blocked(np.c_[gx[:-1].ravel(), gy[:-1].ravel()],
                 np.c_[gx[1:].ravel(), gy[1:].ravel()]).reshape(ny - 1, nx)
    seen = np.zeros((ny, nx), bool)
    seen[j, i] = True
    queue = deque([(j, i)])
    while queue:
        r, c = queue.popleft()
        steps = [(r, c + 1, c + 1 < nx and not right[r, c]),
                 (r, c - 1, c > 0 and not right[r, c - 1]),
                 (r + 1, c, r + 1 < ny and not up[r, c]),
                 (r - 1, c, r > 0 and not up[r - 1, c])]
        for rr, cc, ok in steps:
            if ok and not seen[rr, cc]:
                seen[rr, cc] = True
                queue.append((rr, cc))
    return np.c_[gx.ravel(), gy.ravel()][seen.ravel()]


@st.composite
def free_walls(draw):
    size = draw(st.floats(3, 12))
    n = draw(st.integers(0, 12))
    coords = st.floats(-0.2 * size, 1.2 * size)
    inner = np.array([[[draw(coords), draw(coords)], [draw(coords), draw(coords)]]
                      for _ in range(n)]).reshape(-1, 2, 2)
    walls = np.concatenate([rect(0, 0, size, size), inner])
    spacing = draw(st.floats(0.15, 1.5))
    seed = np.array([draw(st.floats(0, size)), draw(st.floats(0, size))])
    eps = draw(st.sampled_from([1e-9, 1e-3, 0.05]))
    return walls, spacing, seed, eps


@settings(max_examples=60, deadline=None)
@given(free_walls())
def test_fill_equals_a_breadth_first_fill_with_geos_distances(scene):
    pytest.importorskip("shapely")
    walls, spacing, seed, eps = scene
    plan = Plan(walls)
    expected = _oracle_flood(plan.segments, spacing, seed, plan.bounds, eps)
    if expected is None:
        with pytest.raises(ValueError):
            plan.flood_grid(spacing, seed, eps=eps)
        return
    centres, cells = plan.flood_grid(spacing, seed, eps=eps)
    assert np.array_equal(centres, expected)
    assert np.array_equal(cells, np.c_[centres - spacing / 2, centres + spacing / 2])
    # Any cell of the fill, used as the seed, fills the same cells.
    other = centres[len(centres) // 2]
    assert np.array_equal(plan.flood_grid(spacing, other, eps=eps)[0], centres)


@settings(max_examples=60, deadline=None)
@given(n=st.integers(3, 12), rho=st.floats(1, 10), turn=st.floats(0, 2 * np.pi),
       spacing=st.floats(0.05, 1.0))
def test_a_closed_polygon_never_leaks(n, rho, turn, spacing):
    pytest.importorskip("shapely")
    from shapely.geometry import Point, Polygon

    k = np.arange(n)
    ring = rho * np.c_[np.cos(turn + 2 * np.pi * k / n), np.sin(turn + 2 * np.pi * k / n)]
    plan = Plan(polygon_walls(ring))
    polygon = Polygon(ring)
    try:
        centres, _ = plan.flood_grid(spacing, (0.0, 0.0))
    except ValueError:
        assume(False)
    assert len(centres) > 0
    assert all(polygon.contains(Point(p)) for p in centres)


def test_regular_polygon_fill_holds_every_interior_centre():
    pytest.importorskip("shapely")
    plan = Plan(regular_ngon(8, 5.0))
    centres, _ = plan.flood_grid(0.25, (0.1, 0.1))
    from shapely.geometry import Point, Polygon

    k = np.arange(8)
    polygon = Polygon(5.0 * np.c_[np.cos(2 * np.pi * k / 8), np.sin(2 * np.pi * k / 8)])
    grid = plan.grid(0.25)
    inside = grid[[polygon.distance(Point(p)) == 0 and polygon.exterior.distance(Point(p)) > 1e-9
                   for p in grid]]
    assert as_set(centres) == as_set(inside)
