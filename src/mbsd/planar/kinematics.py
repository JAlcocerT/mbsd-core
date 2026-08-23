"""
2D kinematic calculations: rotation matrices, point position/velocity/acceleration.

Migrated from:
  CalculosCinematicos/CalculosCinematicos_2D/RotMat_2D.m
  CalculosCinematicos/CalculosCinematicos_2D/RotTet_2D.m
  CalculosCinematicos/CalculosCinematicos_2D/PosicionPunto_2D.m
  CalculosCinematicos/CalculosCinematicos_2D/VelocidadPunto_2D.m
"""

import numpy as np


def rot_mat(theta):
    """2D rotation matrix A(theta).

    Migrated from RotMat_2D.m
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def rot_mat_d(theta):
    """Derivative of rotation matrix dA/dtheta.

    Migrated from RotTet_2D.m
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s, -c],
                     [ c, -s]])


def point_position(qi, rp):
    """Global position of a local point rp on body i.

    Rp = Ri + A(theta_i) @ rp

    Migrated from PosicionPunto_2D.m

    Args:
        qi: body coordinates [x, y, theta]
        rp: local point position [rx, ry]
    """
    Ri = qi[:2]
    Ai = rot_mat(qi[2])
    return Ri + Ai @ rp


def point_velocity(qi, vi, rp):
    """Global velocity of a local point rp on body i.

    Vp = Vi + dtheta_i * dA/dtheta(theta_i) @ rp

    Migrated from VelocidadPunto_2D.m

    Args:
        qi: body coordinates [x, y, theta]
        vi: body velocities [vx, vy, dtheta]
        rp: local point position [rx, ry]
    """
    Vi = vi[:2]
    dtheta_i = vi[2]
    Atet_i = rot_mat_d(qi[2])
    return Vi + dtheta_i * Atet_i @ rp


def point_acceleration(qi, vi, ai, rp):
    """Global acceleration of a local point rp on body i.

    Ap = Ai_trans + ddtheta_i * dA/dtheta @ rp + dtheta_i^2 * d²A/dtheta² @ rp

    Note: d²A/dtheta² = -A(theta), so the centripetal term is -dtheta² * A @ rp

    Args:
        qi: body coordinates [x, y, theta]
        vi: body velocities [vx, vy, dtheta]
        ai: body accelerations [ax, ay, ddtheta]
        rp: local point position [rx, ry]
    """
    Ai_trans = ai[:2]
    dtheta_i = vi[2]
    ddtheta_i = ai[2]
    Ai = rot_mat(qi[2])
    Atet_i = rot_mat_d(qi[2])
    return Ai_trans + ddtheta_i * Atet_i @ rp - dtheta_i**2 * Ai @ rp
