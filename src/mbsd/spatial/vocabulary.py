"""Small 3D data vocabulary used by experimental spatial MBSD work."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ArrayLike3 = Iterable[float]


@dataclass(frozen=True)
class Quaternion:
    """Unit quaternion stored as ``[w, x, y, z]``."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __post_init__(self) -> None:
        values = self.as_array()
        if not np.all(np.isfinite(values)):
            raise ValueError("quaternion components must be finite")
        if np.linalg.norm(values) == 0.0:
            raise ValueError("quaternion norm must be non-zero")
        for name, value in zip(("w", "x", "y", "z"), values):
            object.__setattr__(self, name, float(value))

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
        if not isinstance(self.rotation, Quaternion):
            raise TypeError("rotation must be a Quaternion")
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
    """Experimental posed body with principal inertia about its center of mass."""

    name: str
    mass: float = 1.0
    inertia: np.ndarray = field(default_factory=lambda: np.ones(3))
    center_of_mass: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pose: Pose3D = field(default_factory=Pose3D.identity)

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
        object.__setattr__(
            self,
            "center_of_mass",
            _as_vector(self.center_of_mass, "center_of_mass", 3),
        )
        if not isinstance(self.pose, Pose3D):
            raise TypeError("body pose must be a Pose3D")

    def as_dict(self, index: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "mass": self.mass,
            "inertia": _vector(self.inertia),
            "center_of_mass": _vector(self.center_of_mass),
            "pose": self.pose.as_dict(),
        }
        if index is not None:
            payload["index"] = index
        return payload


@dataclass(frozen=True)
class Frame3D:
    """Named frame in world coordinates or local to a body index."""

    name: str
    pose: Pose3D = field(default_factory=Pose3D.identity)
    parent_body: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("frame name must be non-empty")
        if not isinstance(self.pose, Pose3D):
            raise TypeError("frame pose must be a Pose3D")
        _validate_body_reference(self.parent_body, "parent_body")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pose": self.pose.as_dict(),
            "parent_body": self.parent_body,
        }


@dataclass(frozen=True)
class SphericalJoint3D:
    """Data-only spherical joint sketch; ``None`` identifies the world frame."""

    name: str
    body_i: int | None
    body_j: int | None
    point_i: np.ndarray = field(default_factory=lambda: np.zeros(3))
    point_j: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        _validate_joint_name_and_bodies(self.name, self.body_i, self.body_j)
        object.__setattr__(self, "point_i", _as_vector(self.point_i, "point_i", 3))
        object.__setattr__(self, "point_j", _as_vector(self.point_j, "point_j", 3))

    def as_dict(self, index: int | None = None) -> dict[str, Any]:
        payload = {
            "kind": "spherical",
            "name": self.name,
            "body_i": self.body_i,
            "body_j": self.body_j,
            "point_i": _vector(self.point_i),
            "point_j": _vector(self.point_j),
        }
        if index is not None:
            payload["index"] = index
        return payload


@dataclass(frozen=True)
class FixedJoint3D:
    """Data-only fixed-joint sketch between body-local poses or the world frame."""

    name: str
    body_i: int | None
    body_j: int | None
    frame_i: Pose3D = field(default_factory=Pose3D.identity)
    frame_j: Pose3D = field(default_factory=Pose3D.identity)

    def __post_init__(self) -> None:
        _validate_joint_name_and_bodies(self.name, self.body_i, self.body_j)
        if not isinstance(self.frame_i, Pose3D) or not isinstance(self.frame_j, Pose3D):
            raise TypeError("fixed-joint frames must be Pose3D instances")

    def as_dict(self, index: int | None = None) -> dict[str, Any]:
        payload = {
            "kind": "fixed",
            "name": self.name,
            "body_i": self.body_i,
            "body_j": self.body_j,
            "frame_i": self.frame_i.as_dict(),
            "frame_j": self.frame_j.as_dict(),
        }
        if index is not None:
            payload["index"] = index
        return payload


SpatialJoint3D = SphericalJoint3D | FixedJoint3D


@dataclass(frozen=True)
class SpatialModel:
    """Experimental container for posed 3D bodies, frames, and joint sketches."""

    bodies: tuple[SpatialBody, ...] = ()
    frames: tuple[Frame3D, ...] = ()
    joints: tuple[SpatialJoint3D, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _json_ready(dict(self.metadata)))
        for frame in self.frames:
            _validate_model_body_reference(
                frame.parent_body,
                len(self.bodies),
                "frame parent_body",
            )
        for joint in self.joints:
            self._validate_joint_references(joint)

    def with_body(self, body: SpatialBody) -> "SpatialModel":
        if not isinstance(body, SpatialBody):
            raise TypeError("body must be a SpatialBody")
        return SpatialModel(
            bodies=(*self.bodies, body),
            frames=self.frames,
            joints=self.joints,
            metadata=self.metadata,
        )

    def with_frame(self, frame: Frame3D) -> "SpatialModel":
        if not isinstance(frame, Frame3D):
            raise TypeError("frame must be a Frame3D")
        return SpatialModel(
            bodies=self.bodies,
            frames=(*self.frames, frame),
            joints=self.joints,
            metadata=self.metadata,
        )

    def with_joint(self, joint: SpatialJoint3D) -> "SpatialModel":
        if not isinstance(joint, (SphericalJoint3D, FixedJoint3D)):
            raise TypeError("joint must be a SphericalJoint3D or FixedJoint3D")
        self._validate_joint_references(joint)
        return SpatialModel(
            bodies=self.bodies,
            frames=self.frames,
            joints=(*self.joints, joint),
            metadata=self.metadata,
        )

    def with_metadata(self, **metadata: Any) -> "SpatialModel":
        return SpatialModel(
            bodies=self.bodies,
            frames=self.frames,
            joints=self.joints,
            metadata={**self.metadata, **metadata},
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "mbsd.spatial.model",
            "schema_version": 1,
            "mbsd_version": _package_version(),
            "status": "experimental",
            "dimension": 3,
            "units": {
                "length": "m",
                "angle": "rad",
                "mass": "kg",
                "inertia": "kg*m^2",
            },
            "conventions": {
                "world_frame": "right_handed_xyz",
                "rotation": "active_body_to_world",
                "quaternion_order": ["w", "x", "y", "z"],
                "pose_translation": "world_xyz",
                "inertia": "principal_moments_about_com_in_body_frame",
                "joint_points": "body_local_xyz; world_xyz_when_body_is_null",
            },
            "metadata": dict(self.metadata),
            "bodies": [body.as_dict(index) for index, body in enumerate(self.bodies)],
            "frames": [frame.as_dict() for frame in self.frames],
            "joints": [joint.as_dict(index) for index, joint in enumerate(self.joints)],
        }

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=indent, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def _validate_joint_references(self, joint: SpatialJoint3D) -> None:
        _validate_model_body_reference(joint.body_i, len(self.bodies), "joint body_i")
        _validate_model_body_reference(joint.body_j, len(self.bodies), "joint body_j")


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


def _validate_body_reference(value: int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be None or a non-negative body index")


def _validate_joint_name_and_bodies(
    name: str,
    body_i: int | None,
    body_j: int | None,
) -> None:
    if not name:
        raise ValueError("joint name must be non-empty")
    _validate_body_reference(body_i, "body_i")
    _validate_body_reference(body_j, "body_j")
    if body_i is None and body_j is None:
        raise ValueError("a joint must reference at least one body")
    if body_i is not None and body_i == body_j:
        raise ValueError("a joint cannot connect a body to itself")


def _validate_model_body_reference(value: int | None, body_count: int, name: str) -> None:
    _validate_body_reference(value, name)
    if value is not None and int(value) >= body_count:
        raise IndexError(f"{name} index {value} is out of range for {body_count} bodies")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ValueError("metadata arrays must contain only finite values")
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value
    raise TypeError(f"metadata value of type {type(value).__name__} is not JSON serializable")


def _package_version() -> str:
    try:
        return version("mbsd")
    except PackageNotFoundError:
        return "0+local"
