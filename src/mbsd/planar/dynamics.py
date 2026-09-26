"""
2D Dynamics solver: equations of motion and time integration.

Solves the constrained dynamics equation:
    M(q) @ a + ∇V(q) = Q_ext(q, v, t) + C_q(q)^T @ λ(t)

subject to constraints:
    C(q, t) = 0
    C_q(q) @ a = -dC_q @ v - d²C/dt²
"""

import numpy as np
from scipy.integrate import solve_ivp

from .constraints import constraints
from .jacobians import jacobian
from .derivatives import dt_jacobian, dtdt_constraints
from .solver import solve_position, solve_velocity, solve_acceleration_with_lagrange

from .forces import (
    mass_matrix, gravity_forces, spring_damper_forces,
    centrifugal_forces, potential_energy, kinetic_energy,
    ground_contact_forces, surface_contact_forces,
)


def ensure_centroidal_dynamics(mbody) -> None:
    """Reject models whose unsupported offset-COM terms would corrupt dynamics."""
    from ..errors import MechanismSolveError

    for index, body in enumerate(mbody.bodies):
        if np.linalg.norm(body.rG, ord=np.inf) > 0.0:
            raise MechanismSolveError(
                "Dynamics currently requires each body reference point to "
                "coincide with its center of mass. "
                f"Body {index} ({body.name!r}) has center_of_mass={body.rG}."
            )


def assemble_system(mbody, q, v, t, springs=None, Q_user=None,
                    omega=0, alpha=0, contacts=None, surface_contacts=None):
    """Assemble all generalized forces and the mass matrix.

    Sums: gravity + spring-dampers + centrifugal + user + ground contacts
        + surface-to-surface contacts.

    Args:
        mbody: MBody mechanism definition
        q: current position vector
        v: current velocity vector
        t: current time
        springs: list of Spring objects (optional)
        Q_user: user-defined generalized forces (optional)
        omega: angular velocity of reference frame (for centrifugal forces)
        alpha: angular acceleration of reference frame
        contacts: list of GroundContact objects (simple circular-wheel vs y(x))
        surface_contacts: list of SurfaceContact objects (surface-to-surface)

    Returns:
        Q_total: (ncoord,) summed generalized forces
        M: (ncoord, ncoord) mass matrix
    """
    M = mass_matrix(mbody, q)

    Q_grav = gravity_forces(mbody, q)
    Q_spring = spring_damper_forces(mbody, q, v, springs)
    Q_centrifugal = centrifugal_forces(mbody, q, omega, alpha)
    Q_contact = ground_contact_forces(mbody, q, v, contacts) if contacts else np.zeros(mbody.ncoord)
    Q_surf = (surface_contact_forces(mbody, q, v, surface_contacts)
              if surface_contacts else np.zeros(mbody.ncoord))

    Q_total = Q_grav + Q_spring + Q_centrifugal + Q_contact + Q_surf
    if Q_user is not None:
        Q_total += Q_user

    return Q_total, M


def solve_inverse_dynamics(mbody, q0, tspan, springs=None, Q_user_fn=None,
                           omega=0, alpha=0, contacts=None, surface_contacts=None):
    """Inverse dynamics: given a kinematically-driven trajectory, compute the
    Lagrange multipliers (constraint reaction forces) at every time step.

    Ported from MATLAB `SimulacionDinamicaInversa.m`. No time integration --
    the mechanism must be kinematically fully determined at each t (i.e.
    `nrestr == ncoord`, typically because user constraints drive the
    remaining DOFs).

    At each time t:
      1. Position:  C(q, t) = 0       (Newton-Raphson)
      2. Velocity:  C_q . v = -C_t
      3. Saddle-point:
             [ M      C_q^T ] [ a ]   [ Q_total ]
             [ C_q    0     ] [ λ ] = [ γ       ]
         where γ = -(dC_q/dt . v + d^2C/dt^2).

    The Lagrange multipliers λ are the constraint reaction forces:
      * Revolute-joint λ pair → (Rx, Ry) bearing reaction at that joint
      * Prismatic-joint λ pair → (angular moment, normal reaction)
      * User-constraint λ    → generalized force the constraint applies;
        the DRIVING force/torque the user must supply is `-λ_user`.

    λ ordering follows `constraints.py` assembly:
        [0:3]                                    fix-ground (x, y, θ)
        [3 : 3+2·n_rev]                          revolute pairs (Rx, Ry each)
        [... : ... + 2·n_prism]                  prismatic pairs
        [... : ... + n_puntolinea]               point-on-line
        [... : ... + 3·n_leva]                   CAM contact
        [... : ... + n_gear]                     gear
        [... : ... + n_user]                     user constraints (last)

    Args:
        mbody: MBody mechanism with user constraints fully driving the motion.
        q0:    initial guess for the position solver at tspan[0].
        tspan: array of time points.
        springs, contacts, surface_contacts, Q_user_fn, omega, alpha:
            passive/applied forces that the driving constraint must overcome.

    Returns:
        q_all, v_all, a_all:  (ncoord × nsteps) kinematic trajectory
        lambda_all:           (nrestr × nsteps) Lagrange multipliers
        Q_total_all:          (ncoord × nsteps) summed passive/applied forces
            (useful for checking the force balance afterwards)
    """
    ensure_centroidal_dynamics(mbody)
    ncoord = mbody.ncoord
    nrestr = mbody.nrestr
    nsteps = len(tspan)

    q_all = np.zeros((ncoord, nsteps))
    v_all = np.zeros((ncoord, nsteps))
    a_all = np.zeros((ncoord, nsteps))
    lambda_all = np.zeros((nrestr, nsteps))
    Q_total_all = np.zeros((ncoord, nsteps))

    q_guess = q0.copy()

    for i, t in enumerate(tspan):
        mbody.t = t

        # 1. Position solve -- warm-started with previous step's q
        q = solve_position(mbody, q_guess, t)

        # 2. Velocity solve (purely kinematic)
        v = solve_velocity(mbody, q, t)

        # 3. Assemble passive/applied forces and mass matrix
        Q_user = Q_user_fn(t, q, v) if Q_user_fn is not None else None
        Q_total, M = assemble_system(mbody, q, v, t, springs, Q_user,
                                     omega, alpha,
                                     contacts=contacts,
                                     surface_contacts=surface_contacts)

        # 4. Saddle-point solve for (accelerations, Lagrange multipliers)
        a, lam = solve_acceleration_with_lagrange(mbody, q, v, t, M, Q_total)

        q_all[:, i] = q
        v_all[:, i] = v
        a_all[:, i] = a
        lambda_all[:, i] = lam
        Q_total_all[:, i] = Q_total

        q_guess = q  # warm-start next step

    return q_all, v_all, a_all, lambda_all, Q_total_all


def solve_dynamics_rk45(
    mbody,
    q0,
    v0,
    tspan,
    springs=None,
    Q_user_fn=None,
    omega=0,
    alpha=0,
    verbose=False,
    allow_underconstrained=False,
):
    """Integrate dynamics with SciPy RK45 and return sampled reactions.

    Args:
        mbody: MBody mechanism definition
        q0: initial position (ncoord,)
        v0: initial velocity (ncoord,)
        tspan: array of time values [t0, t1, ..., tN]
        springs: list of Spring objects
        Q_user_fn: function Q_user(t, q, v) for user forces
        omega: angular velocity of reference frame
        alpha: angular acceleration of reference frame
        verbose: print sampled energy progress
        allow_underconstrained: permit mechanisms with free coordinates

    Returns:
        q_all: positions (ncoord x nsteps)
        v_all: velocities (ncoord x nsteps)
        a_all: accelerations (ncoord x nsteps)
        t_all: time array
        lambda_all: Lagrange multipliers (nrestr x nsteps)
        reaction_all: generalized constraint reactions (ncoord x nsteps)
    """
    ensure_centroidal_dynamics(mbody)
    sol = solve_dynamics_scipy(
        mbody,
        np.asarray(q0, dtype=float),
        np.asarray(v0, dtype=float),
        np.asarray(tspan, dtype=float),
        springs=springs,
        Q_user_fn=Q_user_fn,
        method="RK45",
        omega=omega,
        alpha=alpha,
        allow_underconstrained=allow_underconstrained,
    )
    if not sol.success:
        from ..errors import MechanismSolveError

        raise MechanismSolveError(f"Dynamic integration failed: {sol.message}")

    q_all, v_all, t_all = extract_dynamics_solution(mbody, sol)
    nsteps = len(t_all)
    a_all = np.zeros_like(q_all)
    lambda_all = np.zeros((mbody.nrestr, nsteps))
    reaction_all = np.zeros_like(q_all)
    for index, t in enumerate(t_all):
        q = q_all[:, index]
        v = v_all[:, index]
        Q_user = Q_user_fn(t, q, v) if Q_user_fn is not None else None
        Q_total, mass = assemble_system(mbody, q, v, t, springs, Q_user, omega, alpha)
        acceleration, multipliers = solve_acceleration_with_lagrange(
            mbody, q, v, t, mass, Q_total
        )
        a_all[:, index] = acceleration
        lambda_all[:, index] = multipliers
        reaction_all[:, index] = jacobian(mbody, q, t).T @ multipliers
        if verbose and index % max(1, nsteps // 10) == 0:
            energy = kinetic_energy(mass, v) + potential_energy(mbody, q, springs=springs)
            print(f"  Step {index}/{nsteps}, t = {t:.3f}s, E = {energy:.3f} J")

    return q_all, v_all, a_all, t_all, lambda_all, reaction_all


def _solve_acceleration_baumgarte(mbody, q, v, t, M, Q_total,
                                  alpha_baumgarte, beta_baumgarte):
    """Variant of `solve_acceleration_with_lagrange` with Baumgarte stabilization.

    The acceleration-level constraint is modified from

        Cq @ a = -(dCq/dt @ v + d²C/dt²)

    to the Baumgarte form

        Cq @ a = -(dCq/dt @ v + d²C/dt²)
                 - 2·α·(Cq @ v + ∂C/∂t)
                 - β²·C

    so the constraint residual `C` obeys
        C̈ + 2αĊ + β²C = 0,
    a stable second-order ODE that exponentially pulls drifted positions back
    onto the manifold without re-projecting. Choosing α = β gives critical
    damping at characteristic frequency β; rule of thumb is `β ≈ 5–20` for
    integrators with `rtol ≈ 1e-9`.

    Called only when both `alpha_baumgarte` and `beta_baumgarte` are non-zero
    in `solve_dynamics_scipy`.
    """
    C = constraints(mbody, q, t)
    Cq = jacobian(mbody, q, t)
    DCq = dt_jacobian(mbody, q, v, t)
    DCt = dtdt_constraints(mbody, q, v, t)
    # ∂C/∂t (partial in t, holding q) for the velocity-level stabilizer.
    from .derivatives import dt_constraints as _dt_constraints
    Ct = _dt_constraints(mbody, q, t)

    ncoord = len(q)
    nc = Cq.shape[0]

    rhs_constraint = (
        -(DCq @ v + DCt)
        - 2.0 * alpha_baumgarte * (Cq @ v + Ct)
        - (beta_baumgarte ** 2) * C
    )

    A_block = np.zeros((ncoord + nc, ncoord + nc))
    A_block[:ncoord, :ncoord] = M
    A_block[:ncoord, ncoord:] = Cq.T
    A_block[ncoord:, :ncoord] = Cq

    b = np.zeros(ncoord + nc)
    b[:ncoord] = Q_total
    b[ncoord:] = rhs_constraint

    x = np.linalg.solve(A_block, b)
    a = x[:ncoord]
    lambda_t = x[ncoord:]
    return a, lambda_t


def solve_dynamics_scipy(mbody, q0, v0, t_eval, springs=None, Q_user_fn=None,
                         method='RK45', omega=0, alpha=0,
                         rtol=1e-9, atol=1e-11, project_position=False,
                         contacts=None, surface_contacts=None,
                         alpha_baumgarte=0.0, beta_baumgarte=0.0,
                         allow_underconstrained=False):
    """Solve dynamics using scipy.integrate.solve_ivp.

    Integrates the constrained dynamics equations:
        M(q) @ a = Q_total + C_q^T @ lambda
        C_q @ a = -(dC_q/dt @ v + d^2 C/dt^2)

    The ODE state is [q, v] and the integrator evolves dq/dt = v, dv/dt = a.

    Args:
        mbody: MBody mechanism definition
        q0: initial position (ncoord,) -- must satisfy C(q0, 0) = 0
        v0: initial velocity (ncoord,) -- must satisfy C_q @ v0 = -dC/dt at t=0
        t_eval: array of time values for solution
        springs: list of Spring objects
        Q_user_fn: function Q_user(t, q, v) for user forces
        method: 'RK45', 'RK23', 'DOP853', 'Radau', etc.
        omega: angular velocity of reference frame
        alpha: angular acceleration of reference frame
        rtol, atol: integrator tolerances (default 1e-9 / 1e-11 for crisp physics)
        project_position: if True, re-solve C(q, t)=0 at every RHS eval via Newton-Raphson.
            This can help bounded systems stay on the constraint manifold but can also
            inject energy because the projection moves q off the true trajectory.
            Default False: rely on C_q @ a = ... to keep q on manifold.
        alpha_baumgarte, beta_baumgarte: Baumgarte stabilization gains. When both
            are non-zero the acceleration-level constraint becomes
            `Cq a = … − 2α(Cq v + ∂C/∂t) − β² C`, making drift decay as
            C̈ + 2αĊ + β²C = 0 (critically damped at frequency β when α = β).
            Default 0 = stock behavior. See `validation/baumgarte_comparison.md`
            for tuning guidance and side-by-side drift comparison.
        allow_underconstrained: allow position projection to use the same
            minimum-norm underconstrained solve permitted by the public facade.

    Returns:
        sol: scipy OdeResult object with attributes t, y (solution at times)
    """
    ensure_centroidal_dynamics(mbody)
    if mbody.nrestr < mbody.ncoord and not allow_underconstrained:
        from ..errors import MechanismSolveError

        raise MechanismSolveError(
            "Underconstrained dynamics requires allow_underconstrained=True: "
            f"{mbody.nrestr} constraints for {mbody.ncoord} coordinates."
        )
    if mbody.nrestr > mbody.ncoord:
        from ..errors import MechanismSolveError

        raise MechanismSolveError(
            "Overconstrained dynamics model: "
            f"{mbody.nrestr} constraints for {mbody.ncoord} coordinates."
        )
    use_baumgarte = (alpha_baumgarte != 0.0) or (beta_baumgarte != 0.0)

    def dynamics_rhs(t, state):
        """Right-hand side for ODE solver.

        state = [q, v] (concatenated)
        Uses constrained dynamics with Lagrange multipliers.
        """
        ncoord = mbody.ncoord
        q = state[:ncoord]
        v = state[ncoord:]

        mbody.t = t

        # Optionally re-project onto the position constraint manifold.
        # For smooth dynamics with well-posed initial conditions, the
        # acceleration-level constraint C_q @ a = ... already keeps q on the
        # manifold to leading order. Re-solving position here can inject
        # spurious energy into the system.
        if project_position:
            q = solve_position(
                mbody,
                q,
                t,
                allow_underconstrained=allow_underconstrained,
            )

        # Compute user forces
        Q_user = None
        if Q_user_fn is not None:
            Q_user = Q_user_fn(t, q, v)

        # Assemble forces and mass matrix
        Q_total, M = assemble_system(mbody, q, v, t, springs, Q_user, omega, alpha,
                                     contacts=contacts,
                                     surface_contacts=surface_contacts)

        # Solve full constrained dynamics equation with Lagrange multipliers.
        if use_baumgarte:
            a, _ = _solve_acceleration_baumgarte(
                mbody, q, v, t, M, Q_total,
                alpha_baumgarte, beta_baumgarte,
            )
        else:
            a, _ = solve_acceleration_with_lagrange(mbody, q, v, t, M, Q_total)

        return np.concatenate([v, a])

    # Initial state
    y0 = np.concatenate([q0, v0])

    # Solve ODE with tight tolerances for crisp physics
    sol = solve_ivp(
        dynamics_rhs,
        (t_eval[0], t_eval[-1]),
        y0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=True,
        events=None
    )

    return sol


def extract_dynamics_solution(mbody, sol):
    """Extract positions, velocities from scipy OdeResult.

    Args:
        mbody: MBody mechanism
        sol: scipy OdeResult from solve_ivp

    Returns:
        q_all: positions (ncoord x nsteps)
        v_all: velocities (ncoord x nsteps)
        t_all: time array
    """
    ncoord = mbody.ncoord
    q_all = sol.y[:ncoord, :]
    v_all = sol.y[ncoord:, :]

    return q_all, v_all, sol.t
