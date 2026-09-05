"""Experimental spatial dynamics preview helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .vocabulary import Pose3D, Quaternion


@dataclass(frozen=True)
class SpatialState:
    """Small free-body state used by the experimental 3D dynamics preview."""

    pose: Pose3D
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "linear_velocity", _as_vector(self.linear_velocity, "linear_velocity"))
        object.__setattr__(self, "angular_velocity", _as_vector(self.angular_velocity, "angular_velocity"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "pose": self.pose.as_dict(),
            "linear_velocity": _vector(self.linear_velocity),
            "angular_velocity": _vector(self.angular_velocity),
        }


def step_free_body(
    state: SpatialState,
    *,
    force: np.ndarray,
    torque: np.ndarray,
    mass: float,
    inertia: np.ndarray,
    dt: float,
) -> SpatialState:
    """Advance one unconstrained rigid body with semi-implicit Euler integration."""
    mass = _positive_scalar(mass, "mass")
    inertia = _as_vector(inertia, "inertia")
    if np.any(inertia <= 0.0):
        raise ValueError("inertia values must be positive")
    dt = _positive_scalar(dt, "dt")
    force = _as_vector(force, "force")
    torque = _as_vector(torque, "torque")

    linear_velocity = state.linear_velocity + (force / mass) * dt
    angular_velocity = state.angular_velocity + (torque / inertia) * dt
    translation = state.pose.translation + linear_velocity * dt
    rotation = _integrate_orientation(state.pose.rotation, angular_velocity, dt)
    return SpatialState(
        pose=Pose3D(translation=translation, rotation=rotation),
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
    )


def simulate_free_body(
    state: SpatialState,
    *,
    force: np.ndarray,
    torque: np.ndarray,
    mass: float,
    inertia: np.ndarray,
    t: np.ndarray,
) -> list[SpatialState]:
    """Return a free-body state history sampled at monotonically increasing times."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) == 0:
        raise ValueError("t must be a non-empty 1D array")
    if not np.all(np.isfinite(t)):
        raise ValueError("t must contain only finite values")
    if len(t) > 1 and not np.all(np.diff(t) > 0.0):
        raise ValueError("t must be strictly increasing")

    states = [state]
    current = state
    for dt in np.diff(t):
        current = step_free_body(
            current,
            force=force,
            torque=torque,
            mass=mass,
            inertia=inertia,
            dt=float(dt),
        )
        states.append(current)
    return states


def _integrate_orientation(rotation: Quaternion, angular_velocity: np.ndarray, dt: float) -> Quaternion:
    angle = float(np.linalg.norm(angular_velocity) * dt)
    if angle == 0.0:
        return rotation.normalized()
    delta = Quaternion.from_axis_angle(angular_velocity, angle)
    return _multiply(delta, rotation).normalized()


def _multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    aw, ax, ay, az = a.normalized().as_array()
    bw, bx, by, bz = b.normalized().as_array()
    return Quaternion(
        w=float(aw * bw - ax * bx - ay * by - az * bz),
        x=float(aw * bx + ax * bw + ay * bz - az * by),
        y=float(aw * by - ax * bz + ay * bw + az * bx),
        z=float(aw * bz + ax * by - ay * bx + az * bw),
    )


def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _positive_scalar(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]
