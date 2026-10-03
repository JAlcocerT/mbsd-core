"""Experimental spatial modeling vocabulary for MBSD.

This namespace is intentionally experimental. It provides portable 3D data
structures and early kinematics helpers before the public 3D solver APIs are
stabilized.
"""

from .kinematics import (
    compose_pose,
    fixed_joint_descriptor_residual,
    inverse_pose,
    max_spatial_residual,
    point_position,
    point_velocity,
    quaternion_rate_world,
    resolve_frame_pose,
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
    SPATIAL_MODEL_SCHEMA,
    SPATIAL_MODEL_SCHEMA_VERSION,
    SUPPORTED_SPATIAL_MODEL_SCHEMA_VERSIONS,
    load_spatial_model_payload,
    validate_spatial_model_payload,
)

__all__ = [
    "FixedJoint3D",
    "Frame3D",
    "Pose3D",
    "Quaternion",
    "SphericalJoint3D",
    "SpatialBody",
    "SpatialModel",
    "SPATIAL_MODEL_SCHEMA",
    "SPATIAL_MODEL_SCHEMA_VERSION",
    "SUPPORTED_SPATIAL_MODEL_SCHEMA_VERSIONS",
    "compose_pose",
    "fixed_joint_descriptor_residual",
    "inverse_pose",
    "load_spatial_model_payload",
    "max_spatial_residual",
    "point_position",
    "point_velocity",
    "quaternion_rate_world",
    "resolve_frame_pose",
    "spherical_joint_descriptor_residual",
    "spherical_joint_residual",
    "validate_spatial_model_payload",
]
