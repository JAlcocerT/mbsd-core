"""
2D Dynamic forces: mass matrix, gravity, springs, damping.

Ported from MATLAB:
  Fuerzas/Fuerzas_2D/MatrizMasa_2D.m
  Fuerzas/Fuerzas_2D/FGravedad_2D.m
  Fuerzas/Fuerzas_2D/FMuelleAmortiguador_2D.m
  Fuerzas/Fuerzas_2D/FCentrifuga_2D.m
"""

import numpy as np
from .kinematics import rot_mat, rot_mat_d


def mass_matrix(mbody, q):
    """Compute the mass matrix in reference coordinates.

    For each body i with coordinates [x_i, y_i, θ_i], the mass matrix block is:

        M_i = [
            m        0       (dA/dθ·rG)_x
            0        m       (dA/dθ·rG)_y
            (rG'·dA'/dθ)_x  (rG'·dA'/dθ)_y    I
        ]

    where:
        m = mass of body i
        rG = center of gravity in local frame [rG_x, rG_y]
        I = moment of inertia about center of gravity
        dA/dθ = derivative of rotation matrix

    CG caveat (important): the off-diagonal coupling term `m·(dA/dθ)·rG` is
    θ-dependent. The mass matrix is therefore constant **only when rG = 0
    for every body** (each body's reference point coincides with its CG).
    For bodies with offset CG the matrix depends on q through θ — the
    "constant M in reference coordinates" property is conditional. The
    integrator handles this correctly because `mass_matrix(mbody, q)` is
    re-evaluated at each step.

    Args:
        mbody: MBody mechanism definition
        q: coordinate vector. Used to read each body's θ for the
           translation-rotation coupling block (matters when rG ≠ 0).

    Returns:
        M: (ncoord x ncoord) symmetric, positive-definite mass matrix
    """
    M = np.zeros((mbody.ncoord, mbody.ncoord))

    for i in range(mbody.nb):
        # Body's current angle — feeds the rG≠0 coupling block.
        theta_i = q[3 * i + 2]

        # Rotation matrix derivative at this angle
        Atet = rot_mat_d(theta_i)

        # Body properties
        m_i = mbody.bodies[i].mass
        rG = mbody.bodies[i].rG
        I_cg = mbody.bodies[i].inertia

        # Apply Steiner's theorem: I_origin = I_cg + m*||rG||^2
        I_i = I_cg + m_i * (rG[0]**2 + rG[1]**2)

        # Translational mass (top-left 2x2)
        M[3 * i, 3 * i] = m_i
        M[3 * i + 1, 3 * i + 1] = m_i

        # Rotational inertia (bottom-right, [2,2])
        M[3 * i + 2, 3 * i + 2] = I_i

        # Coupling terms: translation-rotation (off-diagonal blocks)
        # dA/dθ @ rG = [dA/dθ_00*rG_x + dA/dθ_01*rG_y,
        #              dA/dθ_10*rG_x + dA/dθ_11*rG_y]
        dA_rG = Atet @ rG
        M[3 * i, 3 * i + 2] = m_i * dA_rG[0]
        M[3 * i + 1, 3 * i + 2] = m_i * dA_rG[1]

        # Symmetric blocks
        M[3 * i + 2, 3 * i] = m_i * dA_rG[0]
        M[3 * i + 2, 3 * i + 1] = m_i * dA_rG[1]

    return M


def gravity_forces(mbody, q, g=None):
    """Compute generalized gravity forces.

    For each body i, gravity applies:
        - Translational force: Q[x, y] = m*g (acts on reference point)
        - Torque about reference point: Q[θ] = m·rG' @ (dA/dθ)' @ g

    The torque arises because gravity acts at the center of gravity,
    which is offset from the reference point by rG.

    Args:
        mbody: MBody mechanism definition
        q: coordinate vector (θ values used)
        g: gravity vector [gx, gy], default [0, -9.81]

    Returns:
        Q_grav: (ncoord,) vector of generalized forces
    """
    if g is None:
        g = mbody.g  # Usually [0, -9.81]

    # Ensure g is always a numpy array with shape (2,)
    if np.isscalar(g):
        g = np.array([0.0, g])

    Q_grav = np.zeros(mbody.ncoord)

    for i in range(mbody.nb):
        theta_i = q[3 * i + 2]
        Atet = rot_mat_d(theta_i)

        m_i = mbody.bodies[i].mass
        rG = mbody.bodies[i].rG

        # Force on translational DOF
        Q_grav[3 * i : 3 * i + 2] = m_i * g

        # Torque on rotational DOF: m·rG' @ (dA/dθ)' @ g
        # This is the moment of the gravity force about the reference point
        torque = m_i * rG @ Atet.T @ g
        Q_grav[3 * i + 2] = torque

    return Q_grav


class Spring:
    """A linear spring-damper element connecting two bodies.

    The spring connects point at local position `ri` on body i
    to point at local position `rj` on body j.
    """

    def __init__(self, i, j, k, c=0, l0=None, ri=None, rj=None):
        """Initialize spring element.

        Args:
            i, j: body indices
            k: spring stiffness
            c: damping coefficient (default 0)
            l0: natural length (if None, computed from initial config)
            ri: local position on body i [rx, ry]
            rj: local position on body j [rx, ry]
        """
        self.i = i
        self.j = j
        self.k = k
        self.c = c
        self.l0 = l0
        self.ri = np.array(ri) if ri is not None else np.zeros(2)
        self.rj = np.array(rj) if rj is not None else np.zeros(2)

    def __repr__(self):
        return f"Spring(i={self.i}, j={self.j}, k={self.k}, c={self.c}, l0={self.l0})"


def spring_damper_forces(mbody, q, v, springs):
    """Compute generalized spring-damper forces.

    For each spring connecting bodies i and j:
        1. Compute global positions of attachment points
        2. Compute spring length and rate of length change
        3. Compute spring force f_s = k*(L - L_0)
        4. Compute damping force f_d = c*dL/dt
        5. Transform to generalized forces via Jacobian

    Args:
        mbody: MBody mechanism definition
        q: coordinate vector
        v: velocity vector
        springs: list of Spring objects

    Returns:
        Q_spring: (ncoord,) vector of spring generalized forces
    """
    Q_spring = np.zeros(mbody.ncoord)

    if not springs:
        return Q_spring

    for spring in springs:
        i, j = spring.i, spring.j
        k = spring.k
        c = spring.c
        l0 = spring.l0
        ri = spring.ri
        rj = spring.rj

        # Extract coordinates
        qi = mbody.get_qi(q, i)
        qj = mbody.get_qi(q, j)
        vi = mbody.get_vi(v, i)
        vj = mbody.get_vi(v, j)

        xi, yi, thi = qi[0], qi[1], qi[2]
        xj, yj, thj = qj[0], qj[1], qj[2]

        dxi, dyi, dthi = vi[0], vi[1], vi[2]
        dxj, dyj, dthj = vj[0], vj[1], vj[2]

        # Rotation matrices and derivatives
        A_i = rot_mat(thi)
        A_j = rot_mat(thj)
        Atet_i = rot_mat_d(thi)
        Atet_j = rot_mat_d(thj)

        # Global positions of attachment points
        # R_pi = [xi, yi] + A_i @ ri
        # R_pj = [xj, yj] + A_j @ rj
        R_pi = np.array([xi, yi]) + A_i @ ri
        R_pj = np.array([xj, yj]) + A_j @ rj

        # Spring displacement vector
        R = R_pi - R_pj

        # Spring length
        L = np.linalg.norm(R)

        # If natural length not specified, set on first call
        if l0 is None:
            spring.l0 = L
            l0 = L

        # Avoid division by zero for very short springs
        if L < 1e-8:
            continue

        # Velocity of attachment points
        # V_pi = [dxi, dyi] + (dA/dθ_i @ ri) * dθ_i
        # V_pj = [dxj, dyj] + (dA/dθ_j @ rj) * dθ_j
        V_pi = np.array([dxi, dyi]) + Atet_i @ ri * dthi
        V_pj = np.array([dxj, dyj]) + Atet_j @ rj * dthj

        # Rate of length change
        # dL = R' @ (V_pi - V_pj) / L
        dR = V_pi - V_pj
        dL = np.dot(R, dR) / L

        # Spring and damping magnitudes
        f_spring = k * (L - l0)
        f_damping = c * dL
        f_total = f_spring + f_damping

        # Direction unit vector
        u = R / L

        # Jacobian for transformation to generalized coordinates
        # L = [I, dA/dθ_i @ ri, -I, -dA/dθ_j @ rj]
        # where I is 2x2 identity
        L = np.zeros((2, 6))
        L[0:2, 0:2] = np.eye(2)  # ∂R_pi/∂[xi, yi]
        L[0:2, 2:3] = (Atet_i @ ri).reshape(-1, 1)  # ∂R_pi/∂θ_i
        L[0:2, 3:5] = -np.eye(2)  # ∂R_pj/∂[xj, yj]
        # Note: ∂R_pj/∂θ_j = -Atet_j @ rj (negative because R = R_pi - R_pj)
        L[0:2, 5:6] = -(Atet_j @ rj).reshape(-1, 1)

        # Generalized force: Q = -L' @ (f_total * u)
        # The negative sign comes from the spring force opposing displacement
        Q_i = -L.T @ (f_total * u)

        # Add to global force vector
        Q_spring[3 * i : 3 * i + 3] += Q_i[0:3]
        Q_spring[3 * j : 3 * j + 3] += Q_i[3:6]

    return Q_spring


class GroundContact:
    """Elastic (penalty-based) wheel-ground contact with optional friction
    and a position-dependent terrain profile.

    Normal force (Hertz + velocity damping):
        delta = max(0, y_terrain(x_wheel) - y_lowest_point)   # penetration
        Fn    = K_hertz * delta^(3/2) + C_damp * v_pen * delta   (delta > 0)
        Fn    = 0                                               (otherwise)

    Tangential friction (regularized Coulomb):
        v_slip = v_x_wheel + R * omega_wheel                 # tangential velocity at contact
        Ft     = -mu * Fn * tanh(v_slip / v_lim)

    where `tanh(v_slip/v_lim)` is a smooth approximation of sign(v_slip), making
    the friction force continuous through zero slip — critical for ODE integrators.

    The friction produces a horizontal force AND a torque about the wheel center
    (r = (0, -R) below the center, so tau_friction = R * Ft_x).

    Assumptions (small-slope / circular wheel):
      - Wheel is treated as a circle of radius `contact_radius`; the lowest
        point is directly below the center at (x_wheel, y_wheel - R).
      - Ground normal is (0, 1) everywhere (valid for gentle terrain slopes).

    Ported and extended from:
        Simulon_Matlab_CDM/Fuerzas/Fuerzas_2D/FuerzasContacto_2D/FnormalElas.m
        Simulon_Matlab_CDM/Fuerzas/Fuerzas_2D/FuerzasContacto_2D/FtanElas.m
    """

    def __init__(self, body, contact_radius, y_ground=0.0,
                 K_hertz=5e4, C_damp=1e3,
                 mu=0.0, v_lim=0.1,
                 y_ground_fn=None,
                 r_contact_local=None):
        """Initialize a body-ground contact.

        Args:
            body: body index in contact with ground
            contact_radius: wheel radius (treated as circular)
            y_ground: constant ground height (used only if y_ground_fn is None)
            K_hertz: Hertz stiffness (N/m^1.5)
            C_damp: contact damping coefficient
            mu: Coulomb friction coefficient (0 = frictionless)
            v_lim: slip velocity scale for the tanh regularization
            y_ground_fn: callable f(x) -> y, terrain height at position x
                (if None, uses constant y_ground)
            r_contact_local: legacy, ignored (kept for API compatibility)
        """
        self.body = body
        self.contact_radius = contact_radius
        self.y_ground = y_ground
        self.K_hertz = K_hertz
        self.C_damp = C_damp
        self.mu = mu
        self.v_lim = v_lim
        self.y_ground_fn = y_ground_fn
        self.r_contact_local = (np.array(r_contact_local) if r_contact_local is not None
                                else np.zeros(2))

        # Diagnostic state (updated on every force call)
        self.last_penetration = 0.0
        self.last_pen_velocity = 0.0
        self.last_force = 0.0
        self.last_friction = 0.0
        self.last_slip = 0.0
        self.last_ground_y = y_ground

    def ground_height(self, x):
        """Ground height at position x, using y_ground_fn if provided."""
        return self.y_ground_fn(x) if self.y_ground_fn is not None else self.y_ground

    def __repr__(self):
        return (f"GroundContact(body={self.body}, R={self.contact_radius}, "
                f"K={self.K_hertz}, C={self.C_damp}, mu={self.mu})")


def ground_contact_forces(mbody, q, v, contacts):
    """Compute generalized forces from all wheel-ground penalty contacts.

    For each GroundContact:
      1. Sample terrain height at the wheel's x-position.
      2. Compute penetration and penetration velocity.
      3. Normal force: Fn = K*delta^(3/2) + C*v_pen*delta, clipped >= 0.
      4. Slip velocity at contact: v_slip = v_x + R * omega.
      5. Friction force: Ft = -mu * Fn * tanh(v_slip / v_lim).
      6. Assemble generalized force on the body:
         - Fx = Ft               (tangential friction)
         - Fy = Fn               (normal contact)
         - tau = R * Ft          (friction torque about wheel center)

    Args:
        mbody: MBody mechanism
        q: coordinate vector
        v: velocity vector
        contacts: list of GroundContact objects

    Returns:
        Q_contact: (ncoord,) generalized force vector
    """
    Q = np.zeros(mbody.ncoord)
    if not contacts:
        return Q

    for c in contacts:
        qi = mbody.get_qi(q, c.body)
        vi = mbody.get_vi(v, c.body)

        x_wheel = qi[0]
        y_wheel = qi[1]
        v_x = vi[0]
        v_y = vi[1]
        omega = vi[2]
        R = c.contact_radius

        # Sample terrain at wheel's x-position
        y_g = c.ground_height(x_wheel)

        # Penetration (+ means wheel is below ground)
        y_lowest = y_wheel - R
        penetration = y_g - y_lowest
        pen_velocity = -v_y  # rate at which penetration grows

        if penetration > 0:
            Fn = c.K_hertz * penetration**1.5 + c.C_damp * pen_velocity * penetration
            if Fn < 0:
                Fn = 0.0
        else:
            Fn = 0.0

        # Tangential friction -- only non-zero if in contact
        if Fn > 0 and c.mu > 0:
            v_slip = v_x + R * omega        # tangential velocity at bottom point
            Ft = -c.mu * Fn * np.tanh(v_slip / c.v_lim)
        else:
            v_slip = v_x + R * omega
            Ft = 0.0

        # Record diagnostic state
        c.last_penetration = penetration
        c.last_pen_velocity = pen_velocity
        c.last_force = Fn
        c.last_friction = Ft
        c.last_slip = v_slip
        c.last_ground_y = y_g

        # Apply force on the body's reference origin, plus torque from friction.
        # Normal force: +y direction at wheel center.
        # Friction:    +x direction at wheel center, torque = R * Ft about center
        # (contact point at r=(0,-R) below center, so torque = cross(r, F_fric)
        #  = -R * 0 - (-R)*Ft_x = R * Ft_x)
        Q[3 * c.body]     += Ft
        Q[3 * c.body + 1] += Fn
        Q[3 * c.body + 2] += R * Ft

    return Q


class SurfaceContact:
    """Penalty contact between two parameterized surfaces on different bodies,
    with optional Coulomb friction.

    Unlike GroundContact (circular wheel on flat y_ground(x) terrain),
    SurfaceContact handles full surface-to-surface contact via closest-point
    search. Appropriate for polygonal / non-circular wheels, or contact
    between two curved bodies.

    Physics:
        delta = -n_i . (P_j - P_i)               # signed penetration along terrain normal
        Fn = K_hertz * delta^(3/2) + C_damp * v_pen * delta   (delta > 0, else 0)
        Ft = -mu * Fn * tanh(v_slip / v_lim)      # regularized Coulomb

    Force on body j is applied at its actual contact point P_j -- this
    produces a natural torque about j's center from the lever arm
    (P_j - R_j), no need to add torque terms by hand.

    Ported from Simulon_Matlab_CDM/Fuerzas/Fuerzas_2D/FuerzasContacto_2D/
        FContElastico_2D.m + FnormalElas.m + FtanElas.m
    """

    def __init__(self, body_i, body_j, geom_i, geom_j,
                 K_hertz=5e4, C_damp=1e3,
                 mu=0.0, v_lim=0.1,
                 ri=None, rj=None,
                 s_init_i=0.0, s_init_j=0.0):
        """
        Args:
            body_i, body_j: body indices; convention is i = "ground"/terrain,
                j = "wheel"/moving body.
            geom_i, geom_j: Surface objects with .position(s) and .tangent(s).
            K_hertz, C_damp: Hertz penalty normal-force parameters.
            mu, v_lim: friction coefficient and slip regularization scale.
            ri, rj: body-local offsets of each surface's reference frame.
            s_init_i, s_init_j: initial guesses for the closest-point params.
        """
        self.i = body_i
        self.j = body_j
        self.geom_i = geom_i
        self.geom_j = geom_j
        self.K_hertz = K_hertz
        self.C_damp = C_damp
        self.mu = mu
        self.v_lim = v_lim
        self.ri = np.array(ri) if ri is not None else np.zeros(2)
        self.rj = np.array(rj) if rj is not None else np.zeros(2)

        # Cached surface-parameter state — re-used as initial guess each call
        self.last_s_i = s_init_i
        self.last_s_j = s_init_j

        # Diagnostics updated on every force call
        self.last_penetration = 0.0
        self.last_pen_velocity = 0.0
        self.last_force = 0.0
        self.last_friction = 0.0
        self.last_slip = 0.0
        self.last_contact_pt = np.zeros(2)
        self.last_normal = np.array([0.0, 1.0])

    def __repr__(self):
        return (f"SurfaceContact(i={self.i}, j={self.j}, "
                f"K={self.K_hertz}, C={self.C_damp}, mu={self.mu})")


def surface_contact_forces(mbody, q, v, surface_contacts):
    """Compute generalized forces from surface-to-surface penalty contacts.

    For each contact runs scipy.optimize.fsolve on a 2-equation tangent-contact
    system (ported from MATLAB MaxDistancia_2D):

        t_i . (P_j - P_i) = 0     # gap perpendicular to surface i's tangent
        t_i x t_j = 0              # tangents parallel (equivalent to t_i . n_j = 0)

    This finds the unique surface-param pair (s_i, s_j) where the surfaces are
    tangent, whether they're separated OR interpenetrating. Much more robust
    than a min-distance search (which gives spurious zero-gap solutions at
    surface crossings when the bodies overlap).

    Penetration is then the SIGNED normal distance:
        delta = -n_i . (P_j - P_i)

    Cost is ~10x a simple Hertz evaluation.

    Args:
        mbody: MBody mechanism
        q: coordinate vector
        v: velocity vector
        surface_contacts: list of SurfaceContact objects

    Returns:
        Q: (ncoord,) generalized force vector
    """
    from scipy.optimize import fsolve

    Q = np.zeros(mbody.ncoord)
    if not surface_contacts:
        return Q

    for c in surface_contacts:
        qi = mbody.get_qi(q, c.i)
        qj = mbody.get_qi(q, c.j)
        vi = mbody.get_vi(v, c.i)
        vj = mbody.get_vi(v, c.j)

        Ai = rot_mat(qi[2])
        Aj = rot_mat(qj[2])
        Ri = qi[:2]
        Rj = qj[:2]

        # Tangent-contact root-finding (MATLAB MaxDistancia_2D formulation)
        def residuals(s):
            s_i_, s_j_ = s[0], s[1]
            P_i_ = Ri + Ai @ (c.geom_i.position(s_i_) + c.ri)
            P_j_ = Rj + Aj @ (c.geom_j.position(s_j_) + c.rj)
            t_i_ = Ai @ c.geom_i.tangent(s_i_)
            t_j_ = Aj @ c.geom_j.tangent(s_j_)
            gap_ = P_j_ - P_i_
            # r1: gap perpendicular to t_i
            # r2: t_i parallel to t_j (2D cross-product = 0)
            return [t_i_[0]*gap_[0] + t_i_[1]*gap_[1],
                    t_i_[0]*t_j_[1] - t_i_[1]*t_j_[0]]

        try:
            sol, _, ier, _ = fsolve(residuals, [c.last_s_i, c.last_s_j],
                                    full_output=True, xtol=1e-9)
            if ier == 1:
                s_i_new, s_j_new = float(sol[0]), float(sol[1])
            else:
                # fsolve didn't converge -- keep last values, no force
                s_i_new, s_j_new = c.last_s_i, c.last_s_j
        except Exception:
            s_i_new, s_j_new = c.last_s_i, c.last_s_j

        c.last_s_i, c.last_s_j = s_i_new, s_j_new

        # Global contact points
        P_i = Ri + Ai @ (c.geom_i.position(s_i_new) + c.ri)
        P_j = Rj + Aj @ (c.geom_j.position(s_j_new) + c.rj)
        gap = P_j - P_i

        # Surface i tangent (world frame) and outward normal (CCW 90°)
        t_local = c.geom_i.tangent(s_i_new)
        t_world = Ai @ t_local
        t_hat = t_world / np.linalg.norm(t_world)
        n_hat = np.array([-t_hat[1], t_hat[0]])  # CCW 90°; outward if traced +s

        penetration = -np.dot(n_hat, gap)

        # Velocities at contact points (world frame)
        r_j_lever = P_j - Rj
        r_i_lever = P_i - Ri
        # omega x r in 2D: (omega * (-r_y, r_x))
        V_j = vj[:2] + vj[2] * np.array([-r_j_lever[1], r_j_lever[0]])
        V_i = vi[:2] + vi[2] * np.array([-r_i_lever[1], r_i_lever[0]])
        V_rel = V_j - V_i

        v_pen = -np.dot(n_hat, V_rel)
        v_slip = np.dot(t_hat, V_rel)

        # Forces
        if penetration > 0:
            Fn = c.K_hertz * penetration**1.5 + c.C_damp * v_pen * penetration
            if Fn < 0:
                Fn = 0.0
        else:
            Fn = 0.0

        Ft = (-c.mu * Fn * np.tanh(v_slip / c.v_lim)) if (Fn > 0 and c.mu > 0) else 0.0

        F_j = Fn * n_hat + Ft * t_hat  # world-frame force on body j

        # Diagnostics
        c.last_penetration = penetration
        c.last_pen_velocity = v_pen
        c.last_force = Fn
        c.last_friction = Ft
        c.last_slip = v_slip
        c.last_contact_pt = P_j.copy()
        c.last_normal = n_hat.copy()

        # Apply force to body j at contact point (natural torque via lever arm)
        Q[3 * c.j]     += F_j[0]
        Q[3 * c.j + 1] += F_j[1]
        Q[3 * c.j + 2] += r_j_lever[0] * F_j[1] - r_j_lever[1] * F_j[0]

        # Reaction on body i at P_i (Newton's 3rd law). If body i is ground,
        # its 3 position constraints absorb the reaction; applying it here is
        # harmless.
        Q[3 * c.i]     -= F_j[0]
        Q[3 * c.i + 1] -= F_j[1]
        Q[3 * c.i + 2] -= r_i_lever[0] * F_j[1] - r_i_lever[1] * F_j[0]

    return Q


def centrifugal_forces(mbody, q, omega, alpha=None):
    """Compute pseudo-forces for rotating/accelerating reference frame.

    When the reference frame rotates with angular velocity ω and
    angular acceleration α, all bodies experience:
        - Centrifugal force: -m·ω × (ω × r)
        - Coriolis force: -2m·ω × v
        - Euler force: -m·α × r

    Args:
        mbody: MBody mechanism definition
        q: coordinate vector
        omega: angular velocity (scalar in 2D)
        alpha: angular acceleration (scalar in 2D, optional)

    Returns:
        Q_centrifugal: (ncoord,) vector of pseudo-forces
    """
    if alpha is None:
        alpha = 0.0

    Q_centrifugal = np.zeros(mbody.ncoord)

    for i in range(mbody.nb):
        m_i = mbody.bodies[i].mass
        rG = mbody.bodies[i].rG

        # In 2D, ω = ω_z (scalar), represented as [0, 0, ω_z]
        # ω × r = [-ω_z * r_y, ω_z * r_x, 0]
        # ω × (ω × r) = [-ω_z² * r_x, -ω_z² * r_y, 0]
        # -ω × (ω × r) = [ω_z² * r_x, ω_z² * r_y, 0]

        # Centrifugal acceleration: ω² * r (radially outward)
        centrifugal = (omega**2) * rG

        # Euler acceleration: α × r = [-α_z * r_y, α_z * r_x]
        euler = alpha * np.array([-rG[1], rG[0]])

        # Total pseudo-acceleration (only translational in 2D)
        pseudo_accel = centrifugal + euler

        # Generalized force (translational only)
        Q_centrifugal[3 * i : 3 * i + 2] = m_i * pseudo_accel

        # Note: No rotational pseudo-torque in 2D (ω is perpendicular to plane)

    return Q_centrifugal


def potential_energy(mbody, q, g=None, springs=None):
    """Compute total potential energy (gravity + elastic).

    Args:
        mbody: MBody mechanism definition
        q: coordinate vector
        g: gravity vector, default mbody.g
        springs: list of Spring objects

    Returns:
        V: scalar potential energy
    """
    if g is None:
        g = mbody.g

    # Ensure g is always a numpy array with shape (2,)
    if np.isscalar(g):
        g = np.array([0.0, g])

    V = 0.0

    # Gravitational potential energy
    for i in range(mbody.nb):
        qi = mbody.get_qi(q, i)
        A_i = rot_mat(qi[2])
        r_G_global = np.array([qi[0], qi[1]]) + A_i @ mbody.bodies[i].rG
        V += -mbody.bodies[i].mass * np.dot(r_G_global, g)

    # Spring potential energy
    if springs:
        for spring in springs:
            i, j = spring.i, spring.j
            qi = mbody.get_qi(q, i)
            qj = mbody.get_qi(q, j)

            A_i = rot_mat(qi[2])
            A_j = rot_mat(qj[2])

            r_pi = np.array([qi[0], qi[1]]) + A_i @ spring.ri
            r_pj = np.array([qj[0], qj[1]]) + A_j @ spring.rj

            L = np.linalg.norm(r_pi - r_pj)

            l0 = spring.l0 if spring.l0 is not None else L
            V += 0.5 * spring.k * (L - l0) ** 2

    return V


def kinetic_energy(M, v):
    """Compute kinetic energy from mass matrix and velocities.

    T = 0.5 * v' @ M @ v

    Args:
        M: (ncoord x ncoord) mass matrix
        v: (ncoord,) velocity vector

    Returns:
        T: scalar kinetic energy
    """
    return 0.5 * np.dot(v, M @ v)


def total_energy(mbody, q, v, M, g=None, springs=None):
    """Compute total mechanical energy.

    E = T + V = kinetic + potential

    Args:
        mbody: MBody mechanism definition
        q: coordinate vector
        v: velocity vector
        M: mass matrix
        g: gravity vector
        springs: list of Spring objects

    Returns:
        E: scalar total energy
    """
    T = kinetic_energy(M, v)
    V = potential_energy(mbody, q, g, springs)
    return T + V
