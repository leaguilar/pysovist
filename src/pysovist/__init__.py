"""Exact isovists and view volumes from floor plans, meshes and raw point clouds.

Conventions: lengths in metres, angles in radians measured counter-clockwise
from the +x axis, z up. A plan is a set of line segments in the (x, y) plane.
"""

from .directions import equiangular, fibonacci_sphere, random_sphere, solid_angle_weights
from .field import isovist_field
from .isovist2d import isovist
from .mesh import Mesh
from .metrics2d import METRIC_NAMES
from .plan import Plan
from .pointcloud import PointCloud
from .raycast import SectionIsovist, section_isovist_area
from .results import Flags, Isovist
from .sampling import sampled_metrics
from .volume3d import VOLUME_METRIC_NAMES, ViewVolume, view_volume, view_volume_field

__version__ = "0.1.0.dev0"

__all__ = [
    "METRIC_NAMES", "VOLUME_METRIC_NAMES", "Flags", "Isovist", "Mesh", "Plan", "PointCloud",
    "SectionIsovist", "ViewVolume", "equiangular", "fibonacci_sphere", "isovist", "isovist_field",
    "random_sphere", "sampled_metrics", "section_isovist_area", "solid_angle_weights",
    "view_volume", "view_volume_field",
]
