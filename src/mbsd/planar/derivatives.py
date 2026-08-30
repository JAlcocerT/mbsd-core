"""
Time derivatives of constraint Jacobians and constraints.

dCq/dt  — needed for acceleration-level kinematics
dC/dt   — partial time derivative of constraints (from user/driving constraints)
d²C/dt² — second partial time derivative

Migrated from:
  Restricciones/Restricciones_2D/DerivadasRestricciones_2D/DtJac_rotula_2D.m
  Restricciones/Restricciones_2D/DerivadasRestricciones_2D/DtJac_prism_2D.m
  Restricciones/Restricciones_2D/DerivadasRestricciones_2D/DtJacobPares_2D.m
  Restricciones/DerivadasRestricciones/DtJacobiano.m
  Restricciones/DerivadasRestricciones/dtRestr.m
  Restricciones/DerivadasRestricciones/DtdtRestr.m
"""

import numpy as np
from .kinematics import rot_mat, rot_mat_d


def dt_jac_rotula(joint, qi, qj, vi, vj):
    """Time derivative of revolute joint Jacobian (2x6).

    Migrated from DtJac_rotula_2D.m

    Key insight: d/dt(dA/dtheta * r) = -dtheta * A * r
    so DCq(:,3) = -dtheta_i * Ai * ri  and  DCq(:,6) = dtheta_j * Aj * rj
    """
    DCq = np.zeros((2, 6))

    Ai = rot_mat(qi[2])
    Aj = rot_mat(qj[2])
    dtheta_i = vi[2]
    dtheta_j = vj[2]

    DCq[:2, 2] = -dtheta_i * Ai @ joint.ri
    DCq[:2, 5] = dtheta_j * Aj @ joint.rj

    return DCq


def dt_jac_prism(joint, qi, qj, vi, vj):
    """Time derivative of prismatic joint Jacobian (2x6).

    Migrated from DtJac_prism_2D.m
    """
    DCq = np.zeros((2, 6))

    Ri, Rj = qi[:2], qj[:2]
    Vi, Vj = vi[:2], vj[:2]
    Ai = rot_mat(qi[2])
    Aj = rot_mat(qj[2])
    Atet_i = rot_mat_d(qi[2])
    Atet_j = rot_mat_d(qj[2])
    dtheta_i = vi[2]
    dtheta_j = vj[2]

    diff = Ri + Ai @ joint.ri - Rj - Aj @ joint.rj
    n = Ai @ joint.hi
    n_theta = Atet_i @ joint.hi
    ri_theta = Atet_i @ joint.ri
    rj_theta = Atet_j @ joint.rj
    n_dot = dtheta_i * n_theta
    diff_dot = Vi + dtheta_i * ri_theta - Vj - dtheta_j * rj_theta

    # Row 0: d/dt of [0,0,1,0,0,-1] = all zeros
    # Row 1 is the time derivative of jac_prism()'s perpendicular-distance row.
    DCq[1, 0:2] = n_dot
    DCq[1, 3:5] = -n_dot

    # d/dt of (dAi*hi)^T*diff + (Ai*hi)^T*(dAi*ri)
    DCq[1, 2] = (
        dtheta_i * (-n) @ diff
        + n_theta @ diff_dot
        + n_dot @ ri_theta
        + n @ (dtheta_i * (-Ai @ joint.ri))
    )

    # d/dt of -(Ai*hi)^T*(dAj*rj)
    DCq[1, 5] = (
        -(n_dot @ rj_theta)
        - n @ (dtheta_j * (-Aj @ joint.rj))
    )

    return DCq


def dt_jacobian(mbody, q, v, t):
    """Assemble the time derivative of the full Jacobian dCq/dt (nrestr x ncoord).

    Migrated from DtJacobiano.m + DtJacobPares_2D.m
    """
    nrestr = mbody.nrestr
    ncoord = mbody.ncoord
    DCq = np.zeros((nrestr, ncoord))

    # Fixed body: zeros (rows 0-2 are constant)
    row = 3

    # Revolute joints
    for k, joint in enumerate(mbody.rev_joints):
        qi = mbody.get_qi(q, joint.i)
        qj = mbody.get_qi(q, joint.j)
        vi = mbody.get_vi(v, joint.i)
        vj = mbody.get_vi(v, joint.j)
        DCqr = dt_jac_rotula(joint, qi, qj, vi, vj)

        ci = 3 * joint.i
        cj = 3 * joint.j
        DCq[row:row + 2, ci:ci + 3] = DCqr[:, :3]
        DCq[row:row + 2, cj:cj + 3] = DCqr[:, 3:6]
        row += 2

    # Prismatic joints
    for k, joint in enumerate(mbody.prism_joints):
        qi = mbody.get_qi(q, joint.i)
        qj = mbody.get_qi(q, joint.j)
        vi = mbody.get_vi(v, joint.i)
        vj = mbody.get_vi(v, joint.j)
        DCqp = dt_jac_prism(joint, qi, qj, vi, vj)

        ci = 3 * joint.i
        cj = 3 * joint.j
        DCq[row:row + 2, ci:ci + 3] = DCqp[:, :3]
        DCq[row:row + 2, cj:cj + 3] = DCqp[:, 3:6]
        row += 2

    row += 3 * mbody.nl + mbody.npl + mbody.ng

    # User constraints
    for uc in mbody.user_constraints:
        DCqUs = uc.dt_jacobian_fn(mbody, q, v, t)
        DCqUs = np.atleast_2d(DCqUs)
        n = DCqUs.shape[0]
        DCq[row:row + n, :] = DCqUs
        row += n

    return DCq


def dt_constraints(mbody, q, t):
    """Partial time derivative of constraints dC/dt (nrestr,).

    For geometric pair constraints this is zero.
    Only user/driving constraints contribute.

    Migrated from dtRestr.m (default) + mechanism-specific overrides.
    """
    Ct = np.zeros(mbody.nrestr)

    row = _user_constraint_start_row(mbody)

    for uc in mbody.user_constraints:
        ct_u = uc.dt_constraint_fn(mbody, q, t)
        ct_u = np.atleast_1d(ct_u)
        Ct[row:row + len(ct_u)] = ct_u
        row += len(ct_u)

    return Ct


def dtdt_constraints(mbody, q, v, t):
    """Second partial time derivative d²C/dt² (nrestr,).

    Migrated from DtdtRestr.m + mechanism-specific overrides.
    """
    DCt = np.zeros(mbody.nrestr)

    row = _user_constraint_start_row(mbody)

    for uc in mbody.user_constraints:
        dct_u = uc.dtdt_constraint_fn(mbody, q, v, t)
        dct_u = np.atleast_1d(dct_u)
        DCt[row:row + len(dct_u)] = dct_u
        row += len(dct_u)

    return DCt


def _user_constraint_start_row(mbody):
    return 3 + 2 * mbody.nr + 2 * mbody.np_ + 3 * mbody.nl + mbody.npl + mbody.ng
