"""Checks shared by every function that takes a field of view or a direction in radians."""

from __future__ import annotations

import math

TWO_PI = 2 * math.pi
ANGLE_TOL = 1e-12  # angles closer than this are equal, so 2 pi plus this is still the full turn


def degrees_hint(value: float) -> str:
    """The advice for an angle that looks like degrees."""
    if not math.isfinite(value):
        return "An angle in degrees converts with math.radians."
    return f"An angle in degrees converts with math.radians, such as math.radians({value})."


def field_of_view(fov, name: str = "fov") -> float:
    """A field of view in radians, positive and at most a full turn.

    Values up to ``ANGLE_TOL`` above 2 pi are the full turn and come back as 2 pi.
    """
    f = float(fov)
    if not f > 0:
        raise ValueError(f"{name} must be a positive angle in radians (2 pi for the full turn), "
                         f"got {f}")
    if f > TWO_PI + ANGLE_TOL:
        raise ValueError(f"{name} must be an angle in radians of at most 2 pi, the full turn, "
                         f"got {f}. {degrees_hint(f)}")
    return min(f, TWO_PI)
