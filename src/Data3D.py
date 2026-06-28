# /------------------------------------------------------------/
# **3D Data Class**
# -----
# *pysovist* under MIT License
# -----
# This module defines the dictionary-like data class used for
# 3D visibility calculations and their related workflows. Most
# methods in this script are an extension of the established
# 2-dimensional workflows from literature. For detailed 
# descriptions of the methods, see the documentation.
# /------------------------------------------------------------/

from __future__ import annotations

from collections.abc import Iterator, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, List, Optional, Tuple
from pathlib import Path
import numpy as np

@dataclass
class Data3D(MutableMapping[str, Any]):
    data: dict[str, Any] = field(default_factory=dict)
    plan: np.ndarray | None = None
    
    results: dict[str, Any] = field(default_factory=dict)
    columns: list[str] | None = None

# TODO: 3d metrics
# visibility volume
# polygon area
# radial lengths: stats
# visibility centroid --drift
# compactness - 36πV^2/S^3
# isotropy (circularity)
# occlusivity
# jaggedness
# vertical openness: average elevation angle
# sky visibility factor?
# angular width --> solid angle
# visibility entropy: -∑(i)p_i x log(p_i)
# visibility tensor: ∫r^2uu^TdΩ
# volumetric connectivity
