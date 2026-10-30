"""Portable motion-law descriptors for planar coordinate drives.

A motion law is pure data: it evaluates ``value(t)``, ``velocity(t)`` and
``acceleration(t)`` and serializes to a JSON-ready dictionary, so a driven
mechanism survives a ``to_json`` / ``from_json`` round trip. Arbitrary Python
callables remain available through ``coordinate_drive`` but are reported as
non-portable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import CubicSpline


class MotionLaw:
    """Base class for portable scalar motion laws."""

    kind = "motion_law"

    def value(self, t: float) -> float:
        raise NotImplementedError

    def velocity(self, t: float) -> float:
        raise NotImplementedError

    def acceleration(self, t: float) -> float:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantSpeed(MotionLaw):
    """``value(t) = start + rate * t``."""

    rate: float
    start: float = 0.0
    kind = "constant_speed"

    def __post_init__(self):
        object.__setattr__(self, "rate", _finite(self.rate, "rate"))
        object.__setattr__(self, "start", _finite(self.start, "start"))

    def value(self, t: float) -> float:
        return self.start + self.rate * t

    def velocity(self, t: float) -> float:
        return self.rate

    def acceleration(self, t: float) -> float:
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "start": self.start, "rate": self.rate}


@dataclass(frozen=True)
class Polynomial(MotionLaw):
    """``value(t) = c0 + c1*t + c2*t**2 + ...`` with ascending coefficients."""

    coefficients: tuple[float, ...]
    kind = "polynomial"

    def __post_init__(self):
        coefficients = tuple(_finite(c, "coefficient") for c in self.coefficients)
        if not coefficients:
            raise ValueError("polynomial needs at least one coefficient")
        object.__setattr__(self, "coefficients", coefficients)

    def _eval(self, coefficients: np.ndarray, t: float) -> float:
        return float(np.polynomial.polynomial.polyval(t, coefficients)) if coefficients.size else 0.0

    def value(self, t: float) -> float:
        return self._eval(np.asarray(self.coefficients), t)

    def velocity(self, t: float) -> float:
        return self._eval(np.polynomial.polynomial.polyder(self.coefficients, 1), t)

    def acceleration(self, t: float) -> float:
        return self._eval(np.polynomial.polynomial.polyder(self.coefficients, 2), t)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "coefficients": list(self.coefficients)}


@dataclass(frozen=True)
class Tabulated(MotionLaw):
    """Cubic-spline interpolation of sampled values.

    Velocity and acceleration are the spline derivatives. Evaluating outside
    ``[times[0], times[-1]]`` raises instead of extrapolating.
    """

    times: tuple[float, ...]
    values: tuple[float, ...]
    kind = "tabulated"

    def __post_init__(self):
        times = tuple(_finite(t, "time") for t in self.times)
        values = tuple(_finite(v, "value") for v in self.values)
        if len(times) < 2 or len(times) != len(values):
            raise ValueError("tabulated law needs at least two (time, value) samples")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("tabulated times must be strictly increasing")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "_spline", CubicSpline(times, values))

    def _checked(self, t: float) -> float:
        if t < self.times[0] - 1e-12 or t > self.times[-1] + 1e-12:
            raise ValueError(
                f"tabulated law evaluated at t={t} outside "
                f"[{self.times[0]}, {self.times[-1]}]"
            )
        return t

    def value(self, t: float) -> float:
        return float(self._spline(self._checked(t)))

    def velocity(self, t: float) -> float:
        return float(self._spline(self._checked(t), 1))

    def acceleration(self, t: float) -> float:
        return float(self._spline(self._checked(t), 2))

    @property
    def domain(self) -> tuple[float, float]:
        return self.times[0], self.times[-1]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "times": list(self.times), "values": list(self.values)}


def law_from_dict(payload: Mapping[str, Any]) -> MotionLaw:
    """Rebuild a motion law from its ``to_dict`` payload."""
    kind = payload.get("kind")
    if kind == "constant_speed":
        return ConstantSpeed(rate=payload["rate"], start=payload.get("start", 0.0))
    if kind == "polynomial":
        return Polynomial(tuple(payload["coefficients"]))
    if kind == "tabulated":
        return Tabulated(tuple(payload["times"]), tuple(payload["values"]))
    raise ValueError(f"unknown motion law kind: {kind!r}")


def _finite(value: Any, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value
