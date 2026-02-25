# /------------------------------------------------------------/
# **2D Data Class**
# -----
# *pysovist-dev* under MIT License
# -----
# This script defines the dictionary-like data class to be
# used for area calculations. All functions regarding area
# are performed on the data object.
# -----
# What's in the file:
# 1. imports
# 2. data class
# /------------------------------------------------------------/

## 1. Imports
from __future__ import annotations
from collections.abc import MutableMapping, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence
import numpy as np
from src.calculate_2d_sp import single_point

## 2. Data Class
@dataclass
class Data2D(MutableMapping[str,Any]):
    data: dict[str,Any] = field(default_factory=dict)
    array: Optional[np.ndarray]=None
    results: Optional[list[str]] = None
    
    def __getitem__(self, key: str) -> Any: return self._data[key]
    def __setitem__(self, key: str, value: Any) -> None: self._data[key] = value
    def __delitem__(self, key: str) -> None: del self._data[key]
    def __iter__(self) -> Iterator[str]: return iter(self._data)
    def __len__(self) -> int: return len(self._data)
    
    # function to import line segments as array
    def import_array(self,arr: np.ndarray,*,kind: Kind = "auto",columns: Optional[Sequence[str]] = None) -> None:
        a = np.asarray(arr)
        if kind == "auto":
            if a.ndim == 2:
                kind = "table"
            else:
                kind = "cube"
        if kind == "table":
            if a.ndim != 2:
                raise ValueError("kind='table' requires a 2D array (rows, cols).")
            self.array = a
        elif kind == "cube":
            if a.ndim < 3:
                raise ValueError("kind='cube' requires an array with ndim >= 3.")
            self.array = a
        else:
            raise ValueError("kind must be one of: 'auto', 'table', 'vector', 'cube'.")
    

    def row_as_dict(self, index: int = 0) -> dict[str, Any]:
        if self.array is None:
            raise ValueError("No table imported.")
        row = self.array[index]
        cols = [f"col_{i}" for i in range(row.shape[0])]
        if len(cols) != row.shape[0]:
            raise ValueError("columns length must match number of columns in table.")
        return {c: (v.item() if isinstance(v, np.generic) else v) for c, v in zip(cols, row, strict=True)}

    def calculate_2d(dist_max:float, N:int, origin: np.ndarray, segments: np.ndarray, FOV:float|None, view_dir:float|None|np.ndarray,**kwargs):
        vol = single_point

        ## dict includes:
        # line segments
        # calculated area, stats for each point (initialized zero)
