"""Checks shared by every function that takes a field of view or a direction in radians."""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2 * math.pi
ANGLE_TOL = 1e-12  # wedges narrower than this merge into their neighbours
# An input up to this relative amount above its limit is the limit: a float32 2 pi lies 2.8e-8
# above 2 pi and a float32 pi 2.8e-8 above pi, relative. No angle typed in degrees comes close.
FULL_TURN_TOL = 1e-6
DEGREES_FROM = 7.0  # a field of view from here on reads as degrees, below it as radians


def beyond(value: float, limit: float) -> bool:
    """Whether ``value`` lies above ``limit`` by more than the relative ``FULL_TURN_TOL``."""
    return value > limit * (1 + FULL_TURN_TOL)


def degrees_hint(value: float) -> str:
    """The advice for an angle that looks like degrees."""
    if not math.isfinite(value):
        return "An angle in degrees converts with math.radians."
    return f"An angle in degrees converts with math.radians, such as math.radians({value})."


def full_turn_hint(value: float) -> str:
    """The advice for a horizontal field of view above the full turn."""
    if value >= DEGREES_FROM or not math.isfinite(value):
        return degrees_hint(value)
    return "For the full turn, pass math.tau."


def field_of_view(fov, name: str = "fov") -> float:
    """A field of view in radians, positive and at most a full turn.

    Values up to a relative ``FULL_TURN_TOL`` above 2 pi are the full turn and come back as
    2 pi.
    """
    f = float(fov)
    if not f > 0:
        raise ValueError(f"{name} must be a positive angle in radians (2 pi for the full turn), "
                         f"got {f}")
    if beyond(f, TWO_PI):
        raise ValueError(f"{name} must be an angle in radians of at most 2 pi, the full turn, "
                         f"got {f}. {full_turn_hint(f)}")
    return min(f, TWO_PI)


def angle(value, name: str, measured: str, alternative: str = "") -> float:
    """A finite angle in radians of at most a full turn in size, unchanged.

    ``measured`` says from where the angle is measured and ``alternative`` names another form
    of the argument, for the messages. Values up to a relative ``FULL_TURN_TOL`` beyond 2 pi in
    size pass.
    """
    a = float(value)
    if not math.isfinite(a):
        raise ValueError(f"{name} must be a finite angle in radians, {measured}{alternative}, "
                         f"got {a}")
    if beyond(abs(a), TWO_PI):
        raise ValueError(f"{name} must be an angle in radians, {measured}, between -2 pi and "
                         f"2 pi{alternative}, got {a}. {degrees_hint(a)} An angle in radians "
                         f"of more than a turn wraps with math.remainder({a}, math.tau).")
    return a


def direction_angle(direction, name: str = "direction", loop: str | None = None) -> float:
    """A viewing direction in radians, counter-clockwise from +x, from an angle or a vector.

    An angle must be finite and between -2 pi and 2 pi (up to ``FULL_TURN_TOL``), and comes
    back unchanged. A vector ``(dx, dy)`` must be finite and not zero, and gives
    ``atan2(dy, dx)``. Any other shape raises. A field passes ``loop``, the function that takes
    one observer: a field takes one angle for every observer, since with two observers a
    vector and two angles look alike, and its message names the loop to write.
    """
    if direction is None:
        raise ValueError(f"{name} must be an angle in radians, counter-clockwise from +x"
                         f"{'' if loop else ', or a vector (dx, dy)'}, got None")
    v = np.asarray(direction, dtype=float)
    if loop is not None:
        if v.ndim:
            raise ValueError(f"{name} of a field is one angle in radians for every observer, got "
                             f"an array of shape {v.shape}. A vector (dx, dy) converts with "
                             f"math.atan2(dy, dx). For a direction per observer, call {loop} for "
                             f"each observer in a loop")
        return angle(v, name, "counter-clockwise from +x")
    if v.ndim == 0:
        return angle(v, name, "counter-clockwise from +x", ", or a vector (dx, dy)")
    if v.shape == (2,):
        if not (np.isfinite(v).all() and v.any()):
            raise ValueError(f"{name} as a vector (dx, dy) must be finite, with an x or y "
                             f"component, got {tuple(v.tolist())}")
        return float(np.arctan2(v[1], v[0]))
    raise ValueError(f"{name} must be one angle in radians or one vector (dx, dy), got an array "
                     f"of shape {v.shape}")
