"""Small 3D data vocabulary used by experimental spatial MBSD work."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


ArrayLike3 = Iterable[float]


@dataclass(frozen=True)
class Quaternion:
    """Unit quaternion stored as ``[w, x, y, z]``."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion()

    @staticmethod
    def from_axis_angle(axis: ArrayLike3, angle: float) -> "Quaternion":
        axis_arr = _as_vector(axis, "axis", 3)
        norm = np.linalg.norm(axis_arr)
        if norm == 0.0:
            raise ValueError("axis must be non-zero")
        axis_arr = axis_arr / norm
        half_angle = 0.5 * _as_finite_scalar(angle, "angle")
        scale = float(np.sin(half_angle))
        return Quaternion(
            w=float(np.cos(half_angle)),
            x=float(scale * axis_arr[0]),
            y=float(scale * axis_arr[1]),
            z=float(scale * axis_arr[2]),
        ).normalized()

    def as_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=float)

    def normalized(self) -> "Quaternion":
        arr = self.as_array()
        if not np.all(np.isfinite(arr)):
            raise ValueError("quaternion components must be finite")
        norm = np.linalg.norm(arr)
        if norm == 0.0:
            raise ValueError("quaternion norm must be non-zero")
        arr = arr / norm
        return Quaternion(*(float(value) for value in arr))

    def to_rotation_matrix(self) -> np.ndarray:
        w, x, y, z = self.normalized().as_array()
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def rotate(self, vector: ArrayLike3) -> np.ndarray:
        return self.to_rotation_matrix() @ _as_vector(vector, "vector", 3)

    def as_dict(self) -> dict[str, float]:
        normalized = self.normalized()
        return {
            "w": normalized.w,
            "x": normalized.x,
            "y": normalized.y,
            "z": normalized.z,
        }


@dataclass(frozen=True)
class Pose3D:
    """Rigid 3D pose with position and orientation."""

    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: Quaternion = field(default_factory=Quaternion.identity)

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation", _as_vector(self.translation, "translation", 3))
        object.__setattr__(self, "rotation", self.rotation.normalized())

    @staticmethod
    def identity() -> "Pose3D":
        return Pose3D()

    def transform_point(self, local_point: ArrayLike3) -> np.ndarray:
        return self.translation + self.rotation.rotate(local_point)

    def as_dict(self) -> dict[str, Any]:
        return {
            "translation": _vector(self.translation),
            "rotation": self.rotation.as_dict(),
        }


@dataclass(frozen=True)
class SpatialBody:
    """Experimental 3D body descriptor."""

    name: str
    mass: float = 1.0
    inertia: np.ndarray = field(default_factory=lambda: np.ones(3))
    center_of_mass: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("body name must be non-empty")
        mass = _as_finite_scalar(self.mass, "mass")
        if mass <= 0.0:
            raise ValueError("body mass must be positive")
        inertia = _as_vector(self.inertia, "inertia", 3)
        if np.any(inertia <= 0.0):
            raise ValueError("body inertia values must be positive")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "inertia", inertia)
        object.__setattr__(self, "center_of_mass", _as_vector(self.center_of_mass, "center_of_mass", 3))

    def as_dict(self, index: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "mass": self.mass,
            "inertia": _vector(self.inertia),
            "center_of_mass": _vector(self.center_of_mass),
        }
        if index is not None:
            payload["index"] = index
        return payload


@dataclass(frozen=True)
class Frame3D:
    """Named frame attached to a 3D pose."""

    name: str
    pose: Pose3D = field(default_factory=Pose3D.identity)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("frame name must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pose": self.pose.as_dict(),
        }


@dataclass(frozen=True)
class SpatialModel:
    """Experimental container for 3D bodies and frames."""

    bodies: tuple[SpatialBody, ...] = ()
    frames: tuple[Frame3D, ...] = ()

    def with_body(self, body: SpatialBody) -> "SpatialModel":
        return SpatialModel(bodies=(*self.bodies, body), frames=self.frames)

    def with_frame(self, frame: Frame3D) -> "SpatialModel":
        return SpatialModel(bodies=self.bodies, frames=(*self.frames, frame))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "mbsd.spatial.model",
            "schema_version": 1,
            "status": "experimental",
            "dimension": 3,
            "bodies": [body.as_dict(index) for index, body in enumerate(self.bodies)],
            "frames": [frame.as_dict() for frame in self.frames],
        }


def _as_vector(value: ArrayLike3, name: str, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"{name} must be a finite vector with shape ({length},)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _vector(values: Any) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]
