"""Experimental, residual-based spatial kinematics helpers."""

from __future__ import annotations

from dataclasses import dataclass

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
    rotation_vector = _rotation_vector(relative_rotation)
    return np.concatenate((frame_i.translation - frame_j.translation, rotation_vector))


def joint_residual_jacobian(
    joint: SphericalJoint3D | FixedJoint3D,
    poses: list[Pose3D] | tuple[Pose3D, ...],
    *,
    step: float = 1e-7,
) -> np.ndarray:
    """Return a joint residual Jacobian over world pose increments.

    Each body contributes ``[dx, dy, dz, dRx, dRy, dRz]`` columns. Rotation
    increments are active, world-expressed, and applied before the body pose.
    Spherical- and fixed-joint blocks are analytic. Fixed-joint rotation uses
    the local SO(3) logarithm chart and rejects the ambiguous 180-degree branch.
    """
    if not isinstance(joint, (SphericalJoint3D, FixedJoint3D)):
        raise TypeError("joint must be a SphericalJoint3D or FixedJoint3D")
    if not isinstance(poses, (list, tuple)) or not all(
        isinstance(pose, Pose3D) for pose in poses
    ):
        raise TypeError("poses must be a list or tuple of Pose3D values")
    step = float(step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be a positive finite scalar")
    if isinstance(joint, SphericalJoint3D):
        return _spherical_joint_residual_jacobian(joint, poses)
    return _fixed_joint_residual_jacobian(joint, poses)


def _spherical_joint_residual_jacobian(
    joint: SphericalJoint3D,
    poses: list[Pose3D] | tuple[Pose3D, ...],
) -> np.ndarray:
    result = np.zeros((3, 6 * len(poses)))
    for sign, body, point in (
        (1.0, joint.body_i, joint.point_i),
        (-1.0, joint.body_j, joint.point_j),
    ):
        if body is None:
            continue
        pose = _body_pose(body, poses)
        offset_world = pose.rotation.rotate(point)
        start = 6 * body
        result[:, start : start + 3] += sign * np.eye(3)
        result[:, start + 3 : start + 6] += -sign * _skew(offset_world)
    return result


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _fixed_joint_residual_jacobian(
    joint: FixedJoint3D,
    poses: list[Pose3D] | tuple[Pose3D, ...],
) -> np.ndarray:
    frame_i = _joint_frame_pose(joint.body_i, joint.frame_i, poses)
    frame_j = _joint_frame_pose(joint.body_j, joint.frame_j, poses)
    relative_rotation = frame_j.rotation.inverse().compose(frame_i.rotation)
    rotation_vector = _rotation_vector(relative_rotation)
    rotation_block = (
        _left_jacobian_inverse(rotation_vector)
        @ frame_j.rotation.to_rotation_matrix().T
    )
    result = np.zeros((6, 6 * len(poses)))
    for sign, body, frame in (
        (1.0, joint.body_i, joint.frame_i),
        (-1.0, joint.body_j, joint.frame_j),
    ):
        if body is None:
            continue
        pose = _body_pose(body, poses)
        offset_world = pose.rotation.rotate(frame.translation)
        start = 6 * body
        result[:3, start : start + 3] += sign * np.eye(3)
        result[:3, start + 3 : start + 6] += -sign * _skew(offset_world)
        result[3:, start + 3 : start + 6] += sign * rotation_block
    return result


def _rotation_vector(rotation: Quaternion) -> np.ndarray:
    values = rotation.normalized().as_array()
    if values[0] < 0.0:
        values = -values
    if abs(values[0]) <= 1e-7:
        raise ValueError(
            "fixed-joint rotation residual is undefined near 180 degrees; "
            "initialize the solve inside the local SO(3) chart"
        )
    vector = values[1:]
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(float).eps:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(norm, values[0])
    return vector * (angle / norm)


def _left_jacobian_inverse(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    skew = _skew(rotation_vector)
    if angle < 1e-6:
        return np.eye(3) - 0.5 * skew + (skew @ skew) / 12.0
    coefficient = 1.0 / angle**2 - (1.0 + np.cos(angle)) / (
        2.0 * angle * np.sin(angle)
    )
    return np.eye(3) - 0.5 * skew + coefficient * (skew @ skew)


@dataclass(frozen=True)
class SpatialResidualSummary:
    """Unit-aware maximum residuals for spatial joint constraints."""

    max_translation_residual: float
    max_rotation_residual: float
    finite: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "max_translation_residual_m": self.max_translation_residual,
            "max_rotation_residual_rad": self.max_rotation_residual,
            "finite": self.finite,
        }


def spatial_residual_summary(
    residuals: list[np.ndarray] | tuple[np.ndarray, ...],
) -> SpatialResidualSummary:
    """Summarize spherical (3) and fixed-joint (6) residual vectors by unit."""
    translation: list[float] = []
    rotation: list[float] = []
    for residual in residuals:
        vector = _as_residual_vector(residual)
        if vector.size not in (3, 6):
            raise ValueError("spatial residuals must have shape (3,) or (6,)")
        translation.append(float(np.linalg.norm(vector[:3], ord=np.inf)))
        if vector.size == 6:
            rotation.append(float(np.linalg.norm(vector[3:], ord=np.inf)))
    return SpatialResidualSummary(
        max(translation, default=0.0), max(rotation, default=0.0), True
    )


def max_spatial_residual(residuals: list[np.ndarray] | tuple[np.ndarray, ...]) -> float:
    """Return a raw maximum; use ``spatial_residual_summary`` for unit safety."""
    if not residuals:
        return 0.0
    vectors = [_as_residual_vector(residual) for residual in residuals]
    return float(max(np.linalg.norm(residual, ord=np.inf) for residual in vectors))


def _as_residual_vector(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("residual must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError("residual must contain only finite values")
    return arr


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


def _perturb_pose(pose: Pose3D, coordinate: int, amount: float) -> Pose3D:
    if coordinate < 3:
        translation = np.array(pose.translation, copy=True)
        translation[coordinate] += amount
        return Pose3D(translation, pose.rotation)
    axis = np.zeros(3)
    axis[coordinate - 3] = 1.0
    increment = Quaternion.from_axis_angle(axis, amount)
    return Pose3D(pose.translation, increment.compose(pose.rotation))
