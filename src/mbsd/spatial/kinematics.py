"""Experimental, residual-based spatial kinematics helpers."""

from __future__ import annotations

import numpy as np

from .vocabulary import Pose3D, Quaternion, SphericalJoint3D


def point_position(pose: Pose3D, local_point: np.ndarray) -> np.ndarray:
    """Return a local point expressed in world coordinates."""
    if not isinstance(pose, Pose3D):
        raise TypeError("pose must be a Pose3D")
    return pose.transform_point(_as_vector(local_point, "local_point"))


def compose_pose(parent: Pose3D, local: Pose3D) -> Pose3D:
    """Compose a parent-to-world pose with a local child pose."""
    _validate_pose(parent, "parent")
    _validate_pose(local, "local")
    return Pose3D(
        translation=parent.transform_point(local.translation),
        rotation=_multiply_quaternions(parent.rotation, local.rotation),
    )


def inverse_pose(pose: Pose3D) -> Pose3D:
    """Return the inverse transform of a rigid pose."""
    _validate_pose(pose, "pose")
    rotation = Quaternion(pose.rotation.w, -pose.rotation.x, -pose.rotation.y, -pose.rotation.z)
    return Pose3D(translation=rotation.rotate(-pose.translation), rotation=rotation)


def spherical_joint_residual(
    pose_i: Pose3D,
    point_i: np.ndarray,
    pose_j: Pose3D,
    point_j: np.ndarray,
) -> np.ndarray:
    """Return the 3D point-coincidence residual for two posed local points."""
    return point_position(pose_i, point_i) - point_position(pose_j, point_j)


def spherical_joint_descriptor_residual(
    joint: SphericalJoint3D,
    poses: list[Pose3D] | tuple[Pose3D, ...],
) -> np.ndarray:
    """Evaluate a vocabulary joint, including its optional world endpoint."""
    if not isinstance(joint, SphericalJoint3D):
        raise TypeError("joint must be a SphericalJoint3D")
    return joint.residual(poses)


def max_spatial_residual(residuals: list[np.ndarray] | tuple[np.ndarray, ...]) -> float:
    """Return the maximum infinity-norm residual over spatial residual vectors."""
    if not residuals:
        return 0.0
    vectors = [_as_vector(residual, "residual") for residual in residuals]
    return float(max(np.linalg.norm(residual, ord=np.inf) for residual in vectors))


def _as_vector(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_pose(value: Pose3D, name: str) -> None:
    if not isinstance(value, Pose3D):
        raise TypeError(f"{name} must be a Pose3D")


def _multiply_quaternions(a: Quaternion, b: Quaternion) -> Quaternion:
    aw, ax, ay, az = a.as_array()
    bw, bx, by, bz = b.as_array()
    return Quaternion(
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )
