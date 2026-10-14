"""Methods of the 2D isovist result on flagged isovists and degenerate ray counts.

The ``Isovist`` class lives in the module of result types shared with the 3D code. The cases
marked ``xfail`` fail today and document the behaviour the 2D functions expect from it.
"""

import math

import numpy as np
import pytest

from pysovist import isovist

from .geometry import rect

SQUARE = rect(0, 0, 10, 10)
OWNER = "Isovist methods live in results.py (3D row): reported, not fixed here"


@pytest.mark.xfail(strict=True, raises=IndexError, reason=OWNER)
def test_depth_of_an_observer_on_a_wall_is_nan():
    """An isovist flagged on_wall has no wedges, so its depth is NaN in every direction, as
    its metrics are. Today the lookup indexes an empty wedge list."""
    iso = isovist(SQUARE, (0.0, 5.0))
    assert np.isnan(iso.depth([0.0, 1.0])).all()


@pytest.mark.xfail(strict=True, raises=IndexError, reason=OWNER)
def test_depth_of_an_unbounded_isovist_is_nan():
    iso = isovist(SQUARE[:3], (5.0, 5.0))
    assert np.isnan(iso.depth([0.0, 1.0])).all()


@pytest.mark.xfail(strict=True, reason=OWNER)
def test_radial_of_a_partial_view_needs_two_rays():
    """One ray across a field of view divides by n - 1 = 0 and returns NaN without a message."""
    iso = isovist(SQUARE, (5.0, 5.0), fov=math.pi / 2, direction=0.0)
    with pytest.raises(ValueError):
        iso.radial(1)


def test_radial_of_a_partial_view_runs_from_edge_to_edge():
    """Closed form: three rays over a quarter turn from the centre of the square point at -45,
    0 and +45 degrees and end at the corners (10, 0), (10, 10) and the wall point (10, 5)."""
    iso = isovist(SQUARE, (5.0, 5.0), fov=math.pi / 2, direction=0.0)
    ang, r = iso.radial(3)
    np.testing.assert_allclose(ang, [-math.pi / 4, 0.0, math.pi / 4], atol=1e-15)
    np.testing.assert_allclose(r, [5 * math.sqrt(2), 5.0, 5 * math.sqrt(2)], rtol=1e-12)
