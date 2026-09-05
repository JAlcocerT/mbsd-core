"""Experimental spatial modeling vocabulary for MBSD.

This namespace is intentionally experimental. It provides portable 3D data
structures before the public 3D solver APIs are stabilized.
"""

from .vocabulary import Frame3D, Pose3D, Quaternion, SpatialBody, SpatialModel

__all__ = [
    "Frame3D",
    "Pose3D",
    "Quaternion",
    "SpatialBody",
    "SpatialModel",
]
