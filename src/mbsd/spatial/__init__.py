"""Experimental spatial modeling vocabulary for MBSD.

This namespace is intentionally experimental. It provides portable 3D data
structures and early kinematics helpers before the public 3D solver APIs are
stabilized.
"""

from .kinematics import (
    SphericalJoint3D,
    max_spatial_residual,
    point_position,
    spherical_joint_residual,
)
from .dynamics import SpatialState, simulate_free_body, step_free_body
from .vocabulary import Frame3D, Pose3D, Quaternion, SpatialBody, SpatialModel

__all__ = [
    "Frame3D",
    "Pose3D",
    "Quaternion",
    "SphericalJoint3D",
    "SpatialState",
    "SpatialBody",
    "SpatialModel",
    "max_spatial_residual",
    "point_position",
    "simulate_free_body",
    "spherical_joint_residual",
    "step_free_body",
]
