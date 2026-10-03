"""Experimental, residual-based spatial kinematics helpers."""

from __future__ import annotations

import numpy as np

from .vocabulary import FixedJoint3D, Frame3D, Pose3D, Quaternion, SphericalJoint3D


def point_position(pose: Pose3D, local_point: np.ndarray) -> np.ndarray:
    """Return a local point expressed in world coordinates."""
    if not isinstance(pose, Pose3D):
        raise TypeError("pose must be a Pose3D")
    return pose.transform_point(_as_vector(local_point, "local_point"))


def point_velocity(
    pose: Pose3D,
    local_point: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity_world: np.ndarray,
) -> np.ndarray:
    """Return world velocity using world-expressed angular velocity."""
    offset_world = pose.rotation.rotate(_as_vector(local_point, "local_point"))
    return _as_vector(linear_velocity, "linear_velocity") + np.cross(
        _as_vector(angular_velocity_world, "angular_velocity_world"), offset_world
    )


def quaternion_rate_world(
    rotation: Quaternion, angular_velocity_world: np.ndarray
) -> np.ndarray:
    """Return ``[w,x,y,z]`` rate for active body-to-world rotation and world omega."""
    if not isinstance(rotation, Quaternion):
        raise TypeError("rotation must be a Quaternion")
    omega = _as_vector(angular_velocity_world, "angular_velocity_world")
    # Quaternion normalisation is undesirable for a pure angular-velocity
    # quaternion, so evaluate the Hamilton product directly.
    ow, ox, oy, oz = 0.0, *omega
    qw, qx, qy, qz = rotation.as_array()
    return 0.5 * np.array(
        [
            ow * qw - ox * qx - oy * qy - oz * qz,
            ow * qx + ox * qw + oy * qz - oz * qy,
            ow * qy - ox * qz + oy * qw + oz * qx,
            ow * qz + ox * qy - oy * qx + oz * qw,
        ]
    )


def compose_pose(parent: Pose3D, local: Pose3D) -> Pose3D:
    """Compose a parent-to-world pose with a local child pose."""
    _validate_pose(parent, "parent")
    _validate_pose(local, "local")
    return Pose3D(
        translation=parent.transform_point(local.translation),
        rotation=parent.rotation.compose(local.rotation),
    )


def inverse_pose(pose: Pose3D) -> Pose3D:
    """Return the inverse transform of a rigid pose."""
    _validate_pose(pose, "pose")
    rotation = pose.rotation.inverse()
    return Pose3D(translation=rotation.rotate(-pose.translation), rotation=rotation)


def resolve_frame_pose(
    frame: Frame3D, poses: list[Pose3D] | tuple[Pose3D, ...]
) -> Pose3D:
    """Resolve a world or body-local frame pose into world coordinates."""
    if not isinstance(frame, Frame3D):
        raise TypeError("frame must be a Frame3D")
    if frame.parent_body is None:
        return frame.pose
    return compose_pose(_body_pose(frame.parent_body, poses), frame.pose)


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


def fixed_joint_descriptor_residual(
    joint: FixedJoint3D,
    poses: list[Pose3D] | tuple[Pose3D, ...],
) -> np.ndarray:
    """Return 3 translation and 3 rotation residuals for a fixed joint."""
    if not isinstance(joint, FixedJoint3D):
        raise TypeError("joint must be a FixedJoint3D")
    frame_i = _joint_frame_pose(joint.body_i, joint.frame_i, poses)
    frame_j = _joint_frame_pose(joint.body_j, joint.frame_j, poses)
    relative_rotation = frame_j.rotation.inverse().compose(frame_i.rotation)
    vector = relative_rotation.as_array()[1:]
    if relative_rotation.w < 0.0:
        vector = -vector
    return np.concatenate((frame_i.translation - frame_j.translation, 2.0 * vector))


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


def _body_pose(body: int, poses: list[Pose3D] | tuple[Pose3D, ...]) -> Pose3D:
    if not isinstance(poses, (list, tuple)):
        raise TypeError("poses must be a list or tuple of Pose3D values")
    if body >= len(poses):
        raise IndexError(f"body index {body} is out of range for {len(poses)} poses")
    pose = poses[body]
    _validate_pose(pose, f"poses[{body}]")
    return pose


def _joint_frame_pose(
    body: int | None,
    frame: Pose3D,
    poses: list[Pose3D] | tuple[Pose3D, ...],
) -> Pose3D:
    return frame if body is None else compose_pose(_body_pose(body, poses), frame)
