"""
2D constraint Jacobians: Cq = dC/dq.

Migrated from:
  Restricciones/Restricciones_2D/JacobianosRestricciones_2D/Jac_rotula_2D.m
  Restricciones/Restricciones_2D/JacobianosRestricciones_2D/Jac_prism_2D.m
  Restricciones/Restricciones_2D/JacobianosRestricciones_2D/JacobPares_2D.m
  Restricciones/Jacobianos_Restricciones/Jacobiano.m
"""

import numpy as np
from .kinematics import rot_mat, rot_mat_d


def jac_rotula(joint, qi, qj):
    """Jacobian of revolute joint constraint (2x6).

    Columns: [dC/dqi (2x3), dC/dqj (2x3)]

    Migrated from Jac_rotula_2D.m
    """
    Cq = np.zeros((2, 6))

    Atet_i = rot_mat_d(qi[2])
    Atet_j = rot_mat_d(qj[2])

    # dC/d(Ri) = I
    Cq[:2, :2] = np.eye(2)
    # dC/d(Rj) = -I
    Cq[:2, 3:5] = -np.eye(2)
    # dC/d(theta_i) = dA/dtheta_i * ri
    Cq[:2, 2] = Atet_i @ joint.ri
    # dC/d(theta_j) = -dA/dtheta_j * rj
    Cq[:2, 5] = -Atet_j @ joint.rj

    return Cq


def jac_prism(joint, qi, qj):
    """Jacobian of prismatic joint constraint (2x6).

    Migrated from Jac_prism_2D.m
    """
    Cq = np.zeros((2, 6))

    Ri = qi[:2]
    Rj = qj[:2]
    Ai = rot_mat(qi[2])
    Aj = rot_mat(qj[2])
    Atet_i = rot_mat_d(qi[2])
    Atet_j = rot_mat_d(qj[2])

    # Row 0: d(theta_i - theta_j)/dq
    Cq[0, 2] = 1.0
    Cq[0, 5] = -1.0

    # Row 1: d/dq of (Ai*hi)^T * (Ri + Ai*ri - Rj - Aj*rj)
    Ai_hi = Ai @ joint.hi
    diff = Ri + Ai @ joint.ri - Rj - Aj @ joint.rj

    Cq[1, 0:2] = Ai_hi
    Cq[1, 3:5] = -Ai_hi

    Cq[1, 2] = ((Atet_i @ joint.hi) @ diff
                + Ai_hi @ (Atet_i @ joint.ri))
    Cq[1, 5] = Ai_hi @ (-Atet_j @ joint.rj)

    return Cq


def jacobian(mbody, q, t):
    """Assemble the full constraint Jacobian Cq (nrestr x ncoord).

    Migrated from Jacobiano.m + JacobPares_2D.m

    Uses analytical Jacobians where available (revolute, prismatic),
    and finite differences for complex constraints (CAM, gear, point-on-line).
    """
    from .constraints import constraints

    nrestr = mbody.nrestr
    ncoord = mbody.ncoord
    Cq = np.zeros((nrestr, ncoord))

    # Fixed body
    Cq[0, 0] = 1.0
    Cq[1, 1] = 1.0
    Cq[2, 2] = 1.0

    row = 3

    # Revolute joints
    for k, joint in enumerate(mbody.rev_joints):
        qi = mbody.get_qi(q, joint.i)
        qj = mbody.get_qi(q, joint.j)
        Cqr = jac_rotula(joint, qi, qj)

        ci = 3 * joint.i
        cj = 3 * joint.j
        Cq[row:row + 2, ci:ci + 3] = Cqr[:, :3]
        Cq[row:row + 2, cj:cj + 3] = Cqr[:, 3:6]
        row += 2

    # Prismatic joints
    for k, joint in enumerate(mbody.prism_joints):
        qi = mbody.get_qi(q, joint.i)
        qj = mbody.get_qi(q, joint.j)
        Cqp = jac_prism(joint, qi, qj)

        ci = 3 * joint.i
        cj = 3 * joint.j
        Cq[row:row + 2, ci:ci + 3] = Cqp[:, :3]
        Cq[row:row + 2, cj:cj + 3] = Cqp[:, 3:6]
        row += 2

    # CAM (Leva) joints - use finite differences
    h = 1e-8
    for joint in mbody.leva_joints:
        C_base = constraints(mbody, q, t)
        for col in range(ncoord):
            q_pert = q.copy()
            q_pert[col] += h
            C_pert = constraints(mbody, q_pert, t)
            # Extract only the CAM constraint rows for this joint
            Cq[row:row + 3, col] = (C_pert[row:row + 3] - C_base[row:row + 3]) / h
        row += 3

    # Point-on-Line joints - use finite differences
    for joint in mbody.punto_linea_joints:
        C_base = constraints(mbody, q, t)
        for col in range(ncoord):
            q_pert = q.copy()
            q_pert[col] += h
            C_pert = constraints(mbody, q_pert, t)
            Cq[row, col] = (C_pert[row] - C_base[row]) / h
        row += 1

    # Gear joints - use finite differences
    for joint in mbody.gear_joints:
        C_base = constraints(mbody, q, t)
        for col in range(ncoord):
            q_pert = q.copy()
            q_pert[col] += h
            C_pert = constraints(mbody, q_pert, t)
            Cq[row, col] = (C_pert[row] - C_base[row]) / h
        row += 1

    # User constraints
    for uc in mbody.user_constraints:
        CqUs = uc.jacobian_fn(mbody, q, t)
        CqUs = np.atleast_2d(CqUs)
        n = CqUs.shape[0]
        Cq[row:row + n, :] = CqUs
        row += n

    return Cq
