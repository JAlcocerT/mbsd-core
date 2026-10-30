"""Applied-load descriptors for planar mechanisms.

Descriptor loads are pure data and serialize into portable mechanism payloads.
``CallbackLoad`` wraps an arbitrary Python function; it works in Python but is
marked opaque, so validation reports the model as non-portable and importers
reject it.

Forces are world-frame vectors applied at a body-local point. Torques are
counterclockwise-positive moments about the out-of-plane axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .kinematics import rot_mat_d


class Load:
    """Base class for applied planar loads."""

    kind = "load"
    portable = True
    name: str

    def generalized(self, ncoord: int, q: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


def _point_force(ncoord: int, q: np.ndarray, body: int, point: np.ndarray, force: np.ndarray):
    Q = np.zeros(ncoord)
    start = 3 * body
    Q[start : start + 2] = force
    Q[start + 2] = float((rot_mat_d(q[start + 2]) @ point) @ force)
    return Q


@dataclass(frozen=True)
class ConstantForce(Load):
    name: str
    body: int
    force: tuple[float, float]
    point: tuple[float, float] = (0.0, 0.0)
    kind = "constant_force"

    def generalized(self, ncoord, q, v, t):
        return _point_force(ncoord, q, self.body, np.asarray(self.point), np.asarray(self.force))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "body": self.body,
            "force": list(self.force),
            "point": list(self.point),
        }


@dataclass(frozen=True)
class ConstantTorque(Load):
    name: str
    body: int
    torque: float
    kind = "constant_torque"

    def generalized(self, ncoord, q, v, t):
        Q = np.zeros(ncoord)
        Q[3 * self.body + 2] = self.torque
        return Q

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "body": self.body, "torque": self.torque}


@dataclass(frozen=True)
class TabulatedForce(Load):
    """Piecewise-linear world-frame force history at a body-local point."""

    name: str
    body: int
    times: tuple[float, ...]
    fx: tuple[float, ...]
    fy: tuple[float, ...]
    point: tuple[float, float] = (0.0, 0.0)
    kind = "tabulated_force"

    def generalized(self, ncoord, q, v, t):
        _check_domain(self.name, self.times, t)
        force = np.array([np.interp(t, self.times, self.fx), np.interp(t, self.times, self.fy)])
        return _point_force(ncoord, q, self.body, np.asarray(self.point), force)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "body": self.body,
            "times": list(self.times),
            "fx": list(self.fx),
            "fy": list(self.fy),
            "point": list(self.point),
        }


@dataclass(frozen=True)
class TabulatedTorque(Load):
    """Piecewise-linear torque history on one body."""

    name: str
    body: int
    times: tuple[float, ...]
    values: tuple[float, ...]
    kind = "tabulated_torque"

    def generalized(self, ncoord, q, v, t):
        _check_domain(self.name, self.times, t)
        Q = np.zeros(ncoord)
        Q[3 * self.body + 2] = np.interp(t, self.times, self.values)
        return Q

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "body": self.body,
            "times": list(self.times),
            "values": list(self.values),
        }


@dataclass(frozen=True)
class CallbackLoad(Load):
    """Opaque Python callback ``fn(t, q, v) -> generalized force vector``."""

    name: str
    fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray]
    kind = "callback"
    portable = False

    def generalized(self, ncoord, q, v, t):
        Q = np.asarray(self.fn(t, q, v), dtype=float)
        if Q.shape != (ncoord,):
            raise ValueError(
                f"force callback {self.name!r} must return shape ({ncoord},), got {Q.shape}"
            )
        if not np.all(np.isfinite(Q)):
            raise ValueError(f"force callback {self.name!r} returned non-finite values")
        return Q

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "opaque": True,
            "callable": getattr(self.fn, "__name__", "callable"),
        }


def load_from_dict(payload: dict[str, Any]) -> Load:
    """Rebuild a portable load descriptor from its ``to_dict`` payload."""
    from ..errors import NotPortableError

    kind = payload.get("kind")
    if kind == "constant_force":
        return ConstantForce(
            payload["name"],
            int(payload["body"]),
            tuple(payload["force"]),
            tuple(payload.get("point", (0.0, 0.0))),
        )
    if kind == "constant_torque":
        return ConstantTorque(payload["name"], int(payload["body"]), float(payload["torque"]))
    if kind == "tabulated_force":
        return TabulatedForce(
            payload["name"],
            int(payload["body"]),
            tuple(payload["times"]),
            tuple(payload["fx"]),
            tuple(payload["fy"]),
            tuple(payload.get("point", (0.0, 0.0))),
        )
    if kind == "tabulated_torque":
        return TabulatedTorque(
            payload["name"],
            int(payload["body"]),
            tuple(payload["times"]),
            tuple(payload["values"]),
        )
    if kind == "callback":
        raise NotPortableError(
            f"load {payload.get('name')!r} is an opaque Python callback and cannot be imported",
            context={"load": payload.get("name")},
            remediation="Replace the callback with a constant or tabulated load descriptor.",
        )
    raise ValueError(f"unknown load kind: {kind!r}")


def _check_domain(name: str, times: tuple[float, ...], t: float) -> None:
    if t < times[0] - 1e-12 or t > times[-1] + 1e-12:
        raise ValueError(f"tabulated load {name!r} evaluated at t={t} outside [{times[0]}, {times[-1]}]")
