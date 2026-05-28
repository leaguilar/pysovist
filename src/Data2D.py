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
from typing import Any, Literal, List, Optional
from pathlib import Path

import numpy as np

try:
    from .calculate_2d_sp import area_array, boundary, single_point
except ImportError:  # pragma: no cover - fallback for direct script usage
    from src.calculate_2d_sp import area_array, boundary, single_point


ArrayKind = Literal["auto", "table", "segments"]

##TODO: desired metrics
#Core Set
#1. Angular Choice
#2. Angular Integration
#3. Connectivity
#4. Intelligibility
#5. Mean Depth


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

    def import_json(self,path:Path|str,delimiter:Optional[List]=None) -> None:
        '''
        Import JSON files
        ---
        <u>Inputs</u>

        - **Path to JSON file** | *str, required*
        - **Delimiter**: start-end points of lines. By default `['start', 'end']` | *list of str (len: 2), optional*
        '''
        from io_src.import_json import import_json
        array = import_json(path,delimiter)
        self.data['array'] = array
        return

    def import_rhino(self,filepath:Path|str,layer_name:str='Default') -> None:
        '''
        Import lines from Rhino document
        ---
        <u>Inputs</u>

        - **Path to Rhino doc** | *str, required*
        - **Delimiter**: start-end points of lines. By default `['start', 'end']` | *list of str (len: 2), optional*
        '''
        from io_src.import_rhino import from_rhino
        array = from_rhino(filepath,layer_name)
        self.data['array'] = array
        return
    
    def import_csv() -> None:
        return



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
