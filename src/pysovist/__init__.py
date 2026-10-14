"""Exact isovists and view volumes from floor plans, meshes and raw point clouds.

Conventions: lengths in metres, angles in radians measured counter-clockwise
from the +x axis, z up. A plan is a set of line segments in the (x, y) plane.
"""

from .directions import (
    angle_grid,
    equiangular,
    fibonacci_sphere,
    in_field_of_view,
    random_sphere,
    solid_angle_weights,
)
from .field import isovist_field
from .isovist2d import isovist
from .mesh import Mesh
from .metrics2d import METRIC_NAMES
from .paths import path_isovists, sample_path
from .plan import Plan
from .pointcloud import PointCloud
from .raycast import SectionIsovist, section_isovist_area
from .rays2d import NO_HIT, RayHits, cast_rays
from .results import Flags, Isovist
from .sampling import sampled_metrics
from .vertical import plane_to_world, vertical_isovist, vertical_section
from .visibility2d import (
    SegmentVisibility,
    segment_ray_hits,
    segment_visibility,
    segment_visibility_field,
)
from .volume3d import (
    FACTOR_NAMES,
    VOLUME_METRIC_NAMES,
    ViewVolume,
    view_volume,
    view_volume_field,
)

__version__ = "0.2.0"

__all__ = [
    "FACTOR_NAMES",
    "METRIC_NAMES",
    "NO_HIT",
    "VOLUME_METRIC_NAMES",
    "Flags",
    "Isovist",
    "Mesh",
    "Plan",
    "PointCloud",
    "RayHits",
    "SectionIsovist",
    "SegmentVisibility",
    "ViewVolume",
    "angle_grid",
    "cast_rays",
    "equiangular",
    "fibonacci_sphere",
    "in_field_of_view",
    "isovist",
    "isovist_field",
    "path_isovists",
    "plane_to_world",
    "random_sphere",
    "sample_path",
    "sampled_metrics",
    "section_isovist_area",
    "segment_ray_hits",
    "segment_visibility",
    "segment_visibility_field",
    "solid_angle_weights",
    "vertical_isovist",
    "vertical_section",
    "view_volume",
    "view_volume_field",
]
