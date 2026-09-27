"""Experimental spatial modeling vocabulary for MBSD.

This namespace is intentionally experimental. It provides portable 3D data
structures and early kinematics helpers before the public 3D solver APIs are
stabilized.
"""

from .kinematics import (
    compose_pose,
    inverse_pose,
    max_spatial_residual,
    point_position,
    spherical_joint_descriptor_residual,
    spherical_joint_residual,
)
from .vocabulary import (
    FixedJoint3D,
    Frame3D,
    Pose3D,
    Quaternion,
    SpatialBody,
    SpatialModel,
    SphericalJoint3D,
)

__all__ = [
    "FixedJoint3D",
    "Frame3D",
    "Pose3D",
    "Quaternion",
    "SphericalJoint3D",
    "SpatialBody",
    "SpatialModel",
    "compose_pose",
    "inverse_pose",
    "max_spatial_residual",
    "point_position",
    "spherical_joint_descriptor_residual",
    "spherical_joint_residual",
]
