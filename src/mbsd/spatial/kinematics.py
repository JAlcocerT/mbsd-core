"""Experimental spatial kinematics helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .vocabulary import Pose3D


@dataclass(frozen=True)
class SphericalJoint3D:
    """Point-coincidence joint between two spatial bodies."""

    body_i: int
    body_j: int
    point_i: np.ndarray
    point_j: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_i", _as_vector(self.point_i, "point_i"))
        object.__setattr__(self, "point_j", _as_vector(self.point_j, "point_j"))

    def residual(self, poses: list[Pose3D] | tuple[Pose3D, ...]) -> np.ndarray:
        return point_position(poses[self.body_i], self.point_i) - point_position(
            poses[self.body_j],
            self.point_j,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "spherical",
            "body_i": self.body_i,
            "body_j": self.body_j,
            "point_i": _vector(self.point_i),
            "point_j": _vector(self.point_j),
        }


def point_position(pose: Pose3D, local_point: np.ndarray) -> np.ndarray:
    """Return a local point expressed in world coordinates."""
    return pose.transform_point(_as_vector(local_point, "local_point"))


def spherical_joint_residual(
    pose_i: Pose3D,
    point_i: np.ndarray,
    pose_j: Pose3D,
    point_j: np.ndarray,
) -> np.ndarray:
    """Return the 3D point-coincidence residual for two posed local points."""
    return point_position(pose_i, point_i) - point_position(pose_j, point_j)


def max_spatial_residual(residuals: list[np.ndarray] | tuple[np.ndarray, ...]) -> float:
    """Return the maximum infinity-norm residual over spatial residual vectors."""
    if not residuals:
        return 0.0
    return float(max(np.linalg.norm(residual, ord=np.inf) for residual in residuals))


def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]
