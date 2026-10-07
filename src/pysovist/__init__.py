"""Exact isovists and view volumes from floor plans, meshes and raw point clouds.

Conventions: lengths in metres, angles in radians measured counter-clockwise
from the +x axis, z up. A plan is a set of line segments in the (x, y) plane.
"""

from .isovist2d import isovist
from .metrics2d import METRIC_NAMES
from .plan import Plan
from .results import Flags, Isovist

__version__ = "0.1.0.dev0"

__all__ = ["METRIC_NAMES", "Flags", "Isovist", "Plan", "isovist"]
