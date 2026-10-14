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
