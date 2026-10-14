"""The plugin's 2D ray layout is implemented twice: in the compatibility layer and in the
obstacle-visibility module. Both must give the same rays for every setting the plugin accepts."""

import numpy as np
import pytest

from pysovist.compat.decodingspaces import ds_ray_angles
from pysovist.rays2d import cast_rays
from pysovist.visibility2d import segment_ray_hits


def _square():
    corners = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    return np.stack([corners, np.roll(corners, -1, axis=0)], axis=1)


@pytest.mark.parametrize("precision", [12, 15, 360, 2999, 3000, 36000])
def test_hit_counts_follow_the_same_rays(precision):
    """Counting per wall the rays of the compatibility layer gives the obstacle-visibility
    module's plugin counts."""
    walls = _square()
    origin = (2.0, 3.0)
    angles = ds_ray_angles(precision)
    hits = cast_rays(walls, np.array([origin]), angles)
    counts = np.bincount(hits.segment[0][hits.hit[0]], minlength=len(walls))
    expected = segment_ray_hits(walls, origin, precision, direction=np.pi / 2,
                                layout="decodingspaces")
    assert counts.sum() == expected.sum() == len(angles)
    np.testing.assert_array_equal(counts, expected)


@pytest.mark.parametrize("precision", [15, 100.5, 360])
@pytest.mark.parametrize("va", [90.0, 180.0, 359.0, 360.0])
@pytest.mark.parametrize("direction", [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (1.0, 1.0),
                                       (-1.0, 0.5)])
def test_hit_counts_follow_the_same_rays_in_any_view(precision, va, direction):
    """The same identity for other view directions, partial views and a non-integer
    Precision: the layer's rays, cast one by one, and the module's plugin layout give the same
    counts per wall."""
    walls = _square()
    origin = (2.0, 3.0)
    angles = ds_ray_angles(precision, view_angle_deg=va, view_direction=direction)
    hits = cast_rays(walls, np.array([origin]), angles)
    counts = np.bincount(hits.segment[0][hits.hit[0]], minlength=len(walls))
    expected = segment_ray_hits(walls, origin, precision, fov=np.radians(va),
                                direction=np.arctan2(direction[1], direction[0]),
                                layout="decodingspaces")
    assert counts.sum() == expected.sum() == len(angles)
    np.testing.assert_array_equal(counts, expected)
