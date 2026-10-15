"""Checks shared by every function that takes a field of view or a direction in radians."""

from __future__ import annotations

import math

import numpy as np

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



def angle(value, name: str, measured: str, alternative: str = "") -> float:
    """A finite angle in radians of at most a full turn in size, unchanged.

    ``measured`` says from where the angle is measured and ``alternative`` names another form
    of the argument, for the messages. Values up to ``ANGLE_TOL`` beyond 2 pi in size pass.
    """
    a = float(value)
    if not math.isfinite(a):
        raise ValueError(f"{name} must be a finite angle in radians, {measured}{alternative}, "
                         f"got {a}")
    if abs(a) > TWO_PI + ANGLE_TOL:
        raise ValueError(f"{name} must be an angle in radians, {measured}, between -2 pi and "
                         f"2 pi{alternative}, got {a}. {degrees_hint(a)}")
    return a


def direction_angle(direction, name: str = "direction", loop: str | None = None) -> float:
    """A viewing direction in radians, counter-clockwise from +x, from an angle or a vector.

    An angle must be finite and between -2 pi and 2 pi (up to ``ANGLE_TOL``), and comes back
    unchanged. A vector ``(dx, dy)`` must be finite and not zero, and gives ``atan2(dy, dx)``.
    Any other shape raises. A field passes ``loop``, the function that takes one observer, so
    that the message for one direction per observer names the loop to write.
    """
    v = np.asarray(direction, dtype=float)
    if v.ndim == 0:
        return angle(v, name, "counter-clockwise from +x", ", or a vector (dx, dy)")
    if v.shape == (2,):
        if not (np.isfinite(v).all() and v.any()):
            raise ValueError(f"{name} as a vector (dx, dy) must be finite, with an x or y "
                             f"component, got {tuple(v.tolist())}")
        return float(np.arctan2(v[1], v[0]))
    if loop is not None:
        raise ValueError(f"{name} is one angle in radians or one vector (dx, dy) for every "
                         f"observer, got an array of shape {v.shape}. For a direction per "
                         f"observer, call {loop} for each observer in a loop")
    raise ValueError(f"{name} must be one angle in radians or one vector (dx, dy), got an array "
                     f"of shape {v.shape}")
