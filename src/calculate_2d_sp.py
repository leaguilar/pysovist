# /--------------------------------------------------/
# **2D Visible Area Calculation: Wrapper Function**
# -----
# *pysovist-dev* under MIT License
# -----
# This module brings together the available 2D visibility
# algorithms and exposes higher-level workflows for
# single-point, batched, and boundary calculations.
# /--------------------------------------------------/

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from .m2d_corner import visibility_polygon_corner
    from .m2d_discretized import visibility_discretized
    from .m2d_segments_angle import visibility_area_np
except ImportError:  # pragma: no cover - fallback for direct script usage
    from m2d_corner import visibility_polygon_corner
    from m2d_discretized import visibility_discretized
    from m2d_segments_angle import visibility_area_np


VALID_METHODS = {"corner", "discretized", "segments_angle"}


def _as_segments(segments: np.ndarray) -> np.ndarray:
    segments_arr = np.asarray(segments, dtype=float)
    if segments_arr.ndim != 3 or segments_arr.shape[1:] != (2, 2):
        raise ValueError(
            f"Segments should be a [N,2,2] NumPy array; got {segments_arr.shape}."
        )
    return segments_arr


def _as_origin(origin: np.ndarray) -> np.ndarray:
    origin_arr = np.asarray(origin, dtype=float)
    if origin_arr.shape != (2,):
        raise ValueError(f"Origin should be a 2-long NumPy array; got {origin_arr.shape}.")
    return origin_arr


def _as_origins(origins: np.ndarray) -> np.ndarray:
    origins_arr = np.asarray(origins, dtype=float)
    if origins_arr.ndim != 2 or origins_arr.shape[1] != 2:
        raise ValueError(
            f"Origins should be a [N,2] NumPy array; got {origins_arr.shape}."
        )
    return origins_arr


def _normalize_view_dir(view_dir: float | np.ndarray | None) -> float | np.ndarray | None:
    if view_dir is None:
        return None
    if isinstance(view_dir, (float, int, np.floating, np.integer)):
        return float(view_dir)

    view_dir_arr = np.asarray(view_dir, dtype=float)
    if view_dir_arr.shape != (2,):
        raise ValueError(
            "View direction should either be a float value or a 2-long vector."
        )

    norm = np.linalg.norm(view_dir_arr)
    if norm == 0:
        raise ValueError("View direction cannot be all zeros.")

    return view_dir_arr / norm


def _validate_common_inputs(
    dist_max: float,
    num_rays: int,
    origin: np.ndarray,
    segments: np.ndarray,
    fov: float | None,
    view_dir: float | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float | np.ndarray | None]:
    if dist_max <= 0:
        raise ValueError("dist_max should be greater than zero.")
    if num_rays <= 0:
        raise ValueError("N should be greater than zero.")

    if fov is not None:
        fov = float(fov)
        if fov <= 0 or fov > 2 * np.pi:
            raise ValueError("FOV should be in the interval (0, 2π].")
        raise NotImplementedError("FOV-limited 2D visibility is not implemented yet.")

    return _as_origin(origin), _as_segments(segments), _normalize_view_dir(view_dir)


def _dispatch_method(
    method: str,
    segments: np.ndarray,
    origin: np.ndarray,
    dist_max: float,
    num_rays: int,
    *,
    return_pts: bool,
    extra_kwargs: dict[str, Any],
) -> float | tuple[float, np.ndarray]:
    kernel_kwargs = dict(extra_kwargs)
    kernel_kwargs["return_pts"] = return_pts

    if method == "corner":
        return visibility_polygon_corner(
            segments,
            origin,
            max_distance=dist_max,
            num_samples=num_rays,
            **kernel_kwargs,
        )
    if method == "discretized":
        return visibility_discretized(
            segments,
            origin,
            max_dist=dist_max,
            num_rays=num_rays,
            **kernel_kwargs,
        )
    return visibility_area_np(
        segments,
        origin,
        max_distance=dist_max,
        num_rays=num_rays,
        **kernel_kwargs,
    )


def _distance_stats(origin: np.ndarray, hit_points: np.ndarray) -> dict[str, float]:
    distances = np.linalg.norm(hit_points - origin, axis=1)
    return {
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
    }


def single_point(
    dist_max: float,
    N: int,
    origin: np.ndarray,
    segments: np.ndarray,
    FOV: float | None = None,
    view_dir: float | np.ndarray | None = None,
    **kwargs: Any,
) -> float | tuple[float, np.ndarray] | tuple[float, dict[str, float]] | tuple[
    float, np.ndarray, dict[str, float]
]:
    """
    Compute visibility for a single 2D vantage point.
    """
    method = kwargs.get("method", "segments_angle")
    if method not in VALID_METHODS:
        raise ValueError(f"Method should be one of {sorted(VALID_METHODS)}.")

    origin_arr, segments_arr, _ = _validate_common_inputs(
        dist_max, N, origin, segments, FOV, view_dir
    )
    return_pts = bool(kwargs.get("return_pts", False))
    return_stats = bool(kwargs.get("return_stats", False))

    kernel_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"method", "return_pts", "return_stats", "two_step"}
    }

    result = _dispatch_method(
        method,
        segments_arr,
        origin_arr,
        dist_max,
        N,
        return_pts=return_pts or return_stats,
        extra_kwargs=kernel_kwargs,
    )

    if return_pts or return_stats:
        area_value, hit_points = result
        hit_points_arr = np.asarray(hit_points, dtype=float)
        stats = _distance_stats(origin_arr, hit_points_arr) if return_stats else None

        if return_pts and return_stats:
            return area_value, hit_points_arr, stats
        if return_pts:
            return area_value, hit_points_arr
        return area_value, stats

    return result


def visibility_polygon(
    dist_max: float,
    N: int,
    origin: np.ndarray,
    segments: np.ndarray,
    FOV: float | None = None,
    view_dir: float | np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[float, np.ndarray]:
    """
    Convenience wrapper that always returns the hit points as well.
    """
    kwargs["return_pts"] = True
    return single_point(dist_max, N, origin, segments, FOV, view_dir, **kwargs)


def area_array(
    dist_max: float,
    N: int,
    origins: np.ndarray,
    segments: np.ndarray,
    FOV: float | None = None,
    view_dir: float | np.ndarray | None = None,
    **kwargs: Any,
) -> np.ndarray | tuple[np.ndarray, list[np.ndarray]] | tuple[
    np.ndarray, list[dict[str, float]]
] | tuple[np.ndarray, list[np.ndarray], list[dict[str, float]]]:
    """
    Compute visibility for an array of 2D origins.
    """
    origins_arr = _as_origins(origins)
    segments_arr = _as_segments(segments)
    return_pts = bool(kwargs.get("return_pts", False))
    return_stats = bool(kwargs.get("return_stats", False))
    two_step = bool(kwargs.get("two_step", False))

    areas: list[float] = []
    points: list[np.ndarray] = []
    stats_list: list[dict[str, float]] = []

    for origin_arr in origins_arr:
        if two_step and kwargs.get("method", "segments_angle") != "corner":
            single_point(
                dist_max,
                N,
                origin_arr,
                segments_arr,
                FOV,
                view_dir,
                method="corner",
            )

        result = single_point(
            dist_max,
            N,
            origin_arr,
            segments_arr,
            FOV,
            view_dir,
            **kwargs,
        )

        if return_pts and return_stats:
            area_value, hit_points, stats = result
            areas.append(area_value)
            points.append(hit_points)
            stats_list.append(stats)
        elif return_pts:
            area_value, hit_points = result
            areas.append(area_value)
            points.append(hit_points)
        elif return_stats:
            area_value, stats = result
            areas.append(area_value)
            stats_list.append(stats)
        else:
            areas.append(result)

    areas_arr = np.asarray(areas, dtype=float)
    if return_pts and return_stats:
        return areas_arr, points, stats_list
    if return_pts:
        return areas_arr, points
    if return_stats:
        return areas_arr, stats_list
    return areas_arr


def _sample_boundary_points(
    boundary_segments: np.ndarray,
    *,
    step: float | None = None,
    samples_per_segment: int = 1,
    include_endpoints: bool = False,
) -> np.ndarray:
    if step is not None and step <= 0:
        raise ValueError("step should be greater than zero.")
    if samples_per_segment <= 0:
        raise ValueError("samples_per_segment should be greater than zero.")

    origins: list[np.ndarray] = []
    for start, end in boundary_segments:
        direction = end - start
        length = float(np.linalg.norm(direction))

        if length == 0:
            origins.append(start.copy())
            continue

        if step is not None:
            count = max(int(np.ceil(length / step)), 1)
        else:
            count = samples_per_segment

        if include_endpoints:
            t_values = np.linspace(0.0, 1.0, count + 2)
        else:
            t_values = (np.arange(count, dtype=float) + 0.5) / count

        for t in np.atleast_1d(t_values):
            origins.append(start + (direction * t))

    if not origins:
        return np.empty((0, 2), dtype=float)

    origins_arr = np.asarray(origins, dtype=float)
    return np.unique(np.round(origins_arr, decimals=12), axis=0)


def boundary(
    dist_max: float,
    N: int,
    boundary_segments: np.ndarray,
    segments: np.ndarray | None = None,
    FOV: float | None = None,
    view_dir: float | np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[np.ndarray]] | tuple[
    np.ndarray, np.ndarray, list[dict[str, float]]
] | tuple[np.ndarray, np.ndarray, list[np.ndarray], list[dict[str, float]]]:
    """
    Sample origins along boundary segments and compute visibility for each sample.
    """
    boundary_segments_arr = _as_segments(boundary_segments)
    scene_segments = boundary_segments_arr if segments is None else _as_segments(segments)

    step = kwargs.pop("step", None)
    samples_per_segment = int(kwargs.pop("samples_per_segment", 1))
    include_endpoints = bool(kwargs.pop("include_endpoints", False))

    origins = _sample_boundary_points(
        boundary_segments_arr,
        step=step,
        samples_per_segment=samples_per_segment,
        include_endpoints=include_endpoints,
    )
    results = area_array(
        dist_max,
        N,
        origins,
        scene_segments,
        FOV,
        view_dir,
        **kwargs,
    )

    if isinstance(results, tuple):
        return (origins, *results)
    return origins, results


class area:
    single_point = staticmethod(single_point)
    visibility_polygon = staticmethod(visibility_polygon)
    area_array = staticmethod(area_array)
    boundary = staticmethod(boundary)
