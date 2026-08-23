"""
2D constraint equations C(q, t) = 0.

Migrated from:
  Restricciones/Restricciones_2D/Pares_2D/Rotula_2D.m
  Restricciones/Restricciones_2D/Pares_2D/Prism_2D.m
  Restricciones/Restricciones_2D/Pares_2D/RestrPares_2D.m
  Restricciones/Restricciones.m
  Restricciones/RestrPares.m
"""

import numpy as np
from .kinematics import rot_mat


def rotula(joint, qi, qj):
    """Revolute joint constraint: point on body i coincides with point on body j.

    C = Ri + Ai*ri - Rj - Aj*rj = 0   (2 equations)

    Migrated from Rotula_2D.m
    """
    Ai = rot_mat(qi[2])
    Aj = rot_mat(qj[2])
    return qi[:2] + Ai @ joint.ri - qj[:2] - Aj @ joint.rj


def prism(joint, qi, qj):
    """Prismatic joint constraint: bodies share angle and motion is along hi.

    C[0] = theta_i - theta_j = 0
    C[1] = (Ai*hi)^T * (Ri + Ai*ri - Rj - Aj*rj) = 0   (2 equations)

    Migrated from Prism_2D.m
    """
    Ai = rot_mat(qi[2])
    Aj = rot_mat(qj[2])

    c = np.zeros(2)
    c[0] = qi[2] - qj[2]
    c[1] = (Ai @ joint.hi) @ (qi[:2] + Ai @ joint.ri - qj[:2] - Aj @ joint.rj)
    return c


def punto_linea(joint, qi, qj):
    """Point-on-Line constraint: point on body i stays on line on body j.

    C = (Ai*hi)^T * (Ri + Ai*ri - Rj - Aj*rj) = 0   (1 equation)

    Migrated from PuntoLinea_2D.m

    Args:
        joint: PuntoLinea joint with ri, rj, hi attributes
        qi: body i coordinates [x, y, theta]
        qj: body j coordinates [x, y, theta]

    Returns:
        Scalar constraint value
    """
    Ai = rot_mat(qi[2])
    Aj = rot_mat(qj[2])

    # Relative position vector
    rel_pos = qi[:2] + Ai @ joint.ri - qj[:2] - Aj @ joint.rj

    # Line direction in global frame
    line_dir = Ai @ joint.hi

    # Perpendicular distance from point to line
    return np.dot(line_dir, rel_pos)


def _find_closest_surface_params(geom_i, geom_j, qi, qj, ri, rj, s_init_i=0, s_init_j=0):
    """Find surface parameters s_i, s_j that minimize the gap between surfaces.

    Uses a simple grid search followed by local refinement.
    Returns s_i, s_j that best satisfy contact.
    """
    from scipy.optimize import minimize

    Ai = rot_mat(qi[2])
    Aj = rot_mat(qj[2])
    Ri = qi[:2]
    Rj = qj[:2]

    def gap_squared(s):
        s_i, s_j = s[0], s[1]
        # Contact point on body i: global position of surface point
        pi = Ri + Ai @ (geom_i.position(s_i) + ri)
        # Contact point on body j: global position of surface point
        pj = Rj + Aj @ (geom_j.position(s_j) + rj)
        # Gap between surfaces
        gap = pi - pj
        return np.dot(gap, gap)

    # Simple optimization to find closest points
    result = minimize(gap_squared, [s_init_i, s_init_j], method='Nelder-Mead')
    return result.x[0], result.x[1]


def gear(joint, qi, qj):
    """Gear constraint: fixed transmission ratio between two rotating bodies.

    C = theta_i - tau * theta_j = 0   (1 equation)

    Enforces a fixed gear ratio: rotation of body i is proportional to rotation of body j.
    tau = R_j / R_i or tooth_j / tooth_i

    Args:
        joint: GearJoint with i, j (body indices), tau (transmission ratio)
        qi: body i coordinates [x, y, theta]
        qj: body j coordinates [x, y, theta]

    Returns:
        Scalar constraint value
    """
    return qi[2] - joint.tau * qj[2]


def leva(joint, qi, qj, s_i=None, s_j=None):
    """CAM (Leva) constraint: two parameterized surfaces in contact.

    Surfaces are defined in body-fixed frames and parameterized by s (arc length/angle).
    The constraint enforces:
      1-2. Contact point coincidence
      3. Normal contact condition (surfaces in proper contact, not sliding)

    C = [
      position_i(s_i) + A(theta_i) @ ri - position_j(s_j) - A(theta_j) @ rj,
      normal_i(s_i) · (A(theta_j) @ tangent_j(s_j))
    ]

    Returns 3 constraint equations (2 coincidence + 1 normal).

    Args:
        joint: LevaJoint with geom_i, geom_j, ri, rj
        qi: body i coordinates [x, y, theta]
        qj: body j coordinates [x, y, theta]
        s_i, s_j: surface parameters (if None, found by minimizing gap)

    Returns:
        Array of 3 constraint equations
    """
    if s_i is None or s_j is None:
        s_i, s_j = _find_closest_surface_params(
            joint.geom_i, joint.geom_j, qi, qj, joint.ri, joint.rj,
            s_init_i=s_i if s_i is not None else 0,
            s_init_j=s_j if s_j is not None else 0
        )

    Ai = rot_mat(qi[2])
    Aj = rot_mat(qj[2])

    # Surface points in local frames
    ui = joint.geom_i.position(s_i)
    uj = joint.geom_j.position(s_j)

    # Surface tangents (derivatives)
    upi = joint.geom_i.tangent(s_i)
    upj = joint.geom_j.tangent(s_j)

    # Normal to surface i (perpendicular to tangent)
    # n_i = perpendicular to u'_i = [-dy, dx] for tangent [dx, dy]
    ni = np.array([-upi[1], upi[0]])

    # Global normal direction on body i
    ni_global = Ai @ ni

    # Global tangent direction on body j
    upj_global = Aj @ upj

    c = np.zeros(3)

    # Contact point coincidence (2 equations)
    # R_i + A_i @ (u_i + r_i) = R_j + A_j @ (u_j + r_j)
    pi = qi[:2] + Ai @ (ui + joint.ri)
    pj = qj[:2] + Aj @ (uj + joint.rj)
    c[0:2] = pi - pj

    # Normal contact condition (1 equation)
    # The normal to surface i should be perpendicular to the tangent of surface j
    # This ensures surfaces are in contact, not sliding
    c[2] = np.dot(ni_global, upj_global)

    return c


def constraints(mbody, q, t):
    """Assemble all constraint equations C(q, t).

    Order (matches RestrPares_2D.m):
      1. Fixed body (3 eqs): q[0:3] = 0
      2. Revolute joints (2 eqs each)
      3. Prismatic joints (2 eqs each)
      4. CAM (Leva) joints (3 eqs each)
      5. Point-on-Line joints (1 eq each)
      6. Gear joints (1 eq each)
      7. User constraints (nrus eqs)

    Migrated from Restricciones.m + RestrPares_2D.m
    """
    C = np.zeros(mbody.nrestr)

    # Fixed body (body 0)
    C[0] = q[0]
    C[1] = q[1]
    C[2] = q[2]

    row = 3

    # Revolute joints
    for k, joint in enumerate(mbody.rev_joints):
        qi = mbody.get_qi(q, joint.i)
        qj = mbody.get_qi(q, joint.j)
        Rr = rotula(joint, qi, qj)
        C[row:row + 2] = Rr
        row += 2

    # Prismatic joints
    for k, joint in enumerate(mbody.prism_joints):
        qi = mbody.get_qi(q, joint.i)
        qj = mbody.get_qi(q, joint.j)
        Rp = prism(joint, qi, qj)
        C[row:row + 2] = Rp
        row += 2

    # CAM (Leva) joints
    for k, joint in enumerate(mbody.leva_joints):
        qi = mbody.get_qi(q, joint.i)
        qj = mbody.get_qi(q, joint.j)
        # Extract surface parameters from extra coordinates
        s_i = q[3 * mbody.nb + 2 * k]
        s_j = q[3 * mbody.nb + 2 * k + 1]
        Rl = leva(joint, qi, qj, s_i, s_j)
        C[row:row + 3] = Rl
        row += 3

    # Point-on-Line joints
    for k, joint in enumerate(mbody.punto_linea_joints):
        qi = mbody.get_qi(q, joint.i)
        qj = mbody.get_qi(q, joint.j)
        Rpl = punto_linea(joint, qi, qj)
        C[row] = Rpl
        row += 1

    # Gear joints
    for k, joint in enumerate(mbody.gear_joints):
        qi = mbody.get_qi(q, joint.i)
        qj = mbody.get_qi(q, joint.j)
        Rg = gear(joint, qi, qj)
        C[row] = Rg
        row += 1

    # User constraints
    for uc in mbody.user_constraints:
        Cu = uc.constraint_fn(mbody, q, t)
        Cu = np.atleast_1d(Cu)
        C[row:row + len(Cu)] = Cu
        row += len(Cu)

    return C
