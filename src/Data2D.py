# /------------------------------------------------------------/
# **2D Data Class**
# -----
# *pysovist-dev* under MIT License
# -----
# This module defines the dictionary-like data class used for
# 2D visibility calculations and their related workflows.
# /------------------------------------------------------------/

from __future__ import annotations

from collections.abc import Iterator, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

try:
    from .calculate_2d_sp import area_array, boundary, single_point
except ImportError:  # pragma: no cover - fallback for direct script usage
    from src.calculate_2d_sp import area_array, boundary, single_point


ArrayKind = Literal["auto", "table", "segments"]


@dataclass
class Data2D(MutableMapping[str, Any]):
    data: dict[str, Any] = field(default_factory=dict)
    array: np.ndarray | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    columns: list[str] | None = None

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def import_array(
        self,
        arr: np.ndarray,
        *,
        kind: ArrayKind = "auto",
        columns: Sequence[str] | None = None,
    ) -> np.ndarray:
        """
        Import a NumPy array into the data object.
        """
        array = np.asarray(arr, dtype=float)

        if kind == "auto":
            if array.ndim == 3 and array.shape[1:] == (2, 2):
                kind = "segments"
            elif array.ndim == 2:
                kind = "table"
            else:
                raise ValueError(
                    "Could not infer array kind. Use kind='table' or kind='segments'."
                )

        if kind == "table":
            if array.ndim != 2:
                raise ValueError("kind='table' requires a 2D array shaped [N,C].")
            if columns is not None and len(columns) != array.shape[1]:
                raise ValueError("columns length must match the number of table columns.")
            self.columns = list(columns) if columns is not None else None
        elif kind == "segments":
            if array.ndim != 3 or array.shape[1:] != (2, 2):
                raise ValueError("kind='segments' requires an array shaped [N,2,2].")
            self.columns = None
            self.data["segments"] = array
        else:
            raise ValueError("kind must be one of: 'auto', 'table', 'segments'.")

        self.array = array
        self.data["array"] = array
        return array

    def row_as_dict(
        self,
        index: int = 0,
        *,
        columns: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """
        Return a row from a 2D table as a dictionary.
        """
        if self.array is None:
            raise ValueError("No table imported.")
        if self.array.ndim != 2:
            raise ValueError("row_as_dict is only available for 2D table arrays.")

        row = self.array[index]
        column_names = list(columns) if columns is not None else self.columns
        if column_names is None:
            column_names = [f"col_{i}" for i in range(row.shape[0])]
        if len(column_names) != row.shape[0]:
            raise ValueError("columns length must match the number of values in the row.")

        return {
            name: (value.item() if isinstance(value, np.generic) else value)
            for name, value in zip(column_names, row)
        }

    def calculate_2d(
        self,
        dist_max: float,
        N: int,
        origin: np.ndarray,
        segments: np.ndarray | None = None,
        FOV: float | None = None,
        view_dir: float | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run a single-point 2D visibility workflow.
        """
        segments_arr = self._segments_or_default(segments)
        result = single_point(
            dist_max,
            N,
            origin,
            segments_arr,
            FOV=FOV,
            view_dir=view_dir,
            **kwargs,
        )
        self._store_result(
            workflow="single_point",
            result=result,
            dist_max=dist_max,
            N=N,
            FOV=FOV,
            method=kwargs.get("method", "segments_angle"),
        )
        return result

    def calculate_array(
        self,
        dist_max: float,
        N: int,
        origins: np.ndarray,
        segments: np.ndarray | None = None,
        FOV: float | None = None,
        view_dir: float | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run the batched 2D visibility workflow.
        """
        segments_arr = self._segments_or_default(segments)
        result = area_array(
            dist_max,
            N,
            origins,
            segments_arr,
            FOV=FOV,
            view_dir=view_dir,
            **kwargs,
        )
        self._store_result(
            workflow="area_array",
            result=result,
            dist_max=dist_max,
            N=N,
            FOV=FOV,
            method=kwargs.get("method", "segments_angle"),
        )
        return result

    def calculate_boundary(
        self,
        dist_max: float,
        N: int,
        boundary_segments: np.ndarray,
        segments: np.ndarray | None = None,
        FOV: float | None = None,
        view_dir: float | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run the boundary-sampling 2D visibility workflow.
        """
        scene_segments = self._segments_or_default(segments)
        result = boundary(
            dist_max,
            N,
            boundary_segments,
            segments=scene_segments,
            FOV=FOV,
            view_dir=view_dir,
            **kwargs,
        )
        self._store_result(
            workflow="boundary",
            result=result,
            dist_max=dist_max,
            N=N,
            FOV=FOV,
            method=kwargs.get("method", "segments_angle"),
        )
        return result

    def _segments_or_default(self, segments: np.ndarray | None) -> np.ndarray:
        if segments is not None:
            return np.asarray(segments, dtype=float)
        if self.array is None:
            raise ValueError("No segments provided and no array has been imported.")
        return np.asarray(self.array, dtype=float)

    def _store_result(self, *, workflow: str, result: Any, **metadata: Any) -> None:
        record = {"workflow": workflow, "result": result, **metadata}
        self.results.append(record)
        self.data["last_result"] = record

    area_array = calculate_array
    boundary = calculate_boundary
