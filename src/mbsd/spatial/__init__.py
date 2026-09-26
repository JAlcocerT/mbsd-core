"""Experimental spatial modeling vocabulary for MBSD.

This namespace is intentionally experimental. It provides portable 3D data
structures before the public 3D solver APIs are stabilized.
"""

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
    "SpatialBody",
    "SpatialModel",
    "SphericalJoint3D",
]
