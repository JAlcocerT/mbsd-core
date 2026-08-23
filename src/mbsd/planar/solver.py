"""
Position, velocity, and acceleration solvers for 2D kinematic analysis.

Migrated from:
  MetodosNumericos/NewtonRaphsonJacob.m
  Z_ModelosMecanismos/BielaManivelaCinematico_2D/SimulacionCinematica.m (solve lines)
"""

import numpy as np
from ..errors import MechanismSolveError
from .constraints import constraints
from .jacobians import jacobian
from .derivatives import dt_jacobian, dt_constraints, dtdt_constraints


def newton_raphson(mbody, q0, t, tol=1e-10, max_iter=50, allow_underconstrained=False):
    """Solve the position problem C(q, t) = 0 using Newton-Raphson.

    Uses the analytical Jacobian for fast convergence.
    Handles both square and underconstrained systems.

    Migrated from NewtonRaphsonJacob.m

    Args:
        mbody: mechanism definition
        q0: initial guess for coordinates
        t: current time
        tol: convergence tolerance
        max_iter: maximum iterations

    Returns:
        q: solution coordinates
    """
    q = q0.copy()
    for _ in range(max_iter):
        C = constraints(mbody, q, t)
        Cq = jacobian(mbody, q, t)
        if Cq.shape[0] < Cq.shape[1] and not allow_underconstrained:
            raise MechanismSolveError(
                f"Underconstrained position solve at t={t:.6g}: "
                f"{Cq.shape[0]} constraints for {Cq.shape[1]} coordinates. "
                "Pass allow_underconstrained=True only when a minimum-norm "
                "projection is intended."
            )
        if Cq.shape[0] > Cq.shape[1]:
            raise MechanismSolveError(
                f"Overconstrained position solve at t={t:.6g}: "
                f"{Cq.shape[0]} constraints for {Cq.shape[1]} coordinates."
            )
        if np.max(np.abs(C)) < tol:
            return q

        # Handle both square and explicitly allowed underconstrained systems.
        if Cq.shape[0] == Cq.shape[1]:
            # Square system: use direct solve
            try:
                dq = np.linalg.solve(Cq, C)
            except np.linalg.LinAlgError as exc:
                raise MechanismSolveError(
                    f"Singular position solve at t={t:.6g}; "
                    f"Jacobian shape={Cq.shape}."
                ) from exc
        elif Cq.shape[0] < Cq.shape[1]:
            # Underconstrained: use least-squares (minimum-norm solution)
            dq = np.linalg.lstsq(Cq, C, rcond=None)[0]
        else:
            raise AssertionError("unreachable constraint shape branch")

        q = q - dq

    raise MechanismSolveError(
        f"Newton-Raphson did not converge after {max_iter} iterations. "
        f"Residual: {np.max(np.abs(constraints(mbody, q, t))):.2e}"
    )


def solve_position(mbody, q0, t, allow_underconstrained=False):
    """Solve position problem: find q such that C(q, t) = 0.

    Args:
        mbody: mechanism definition
        q0: initial guess
        t: current time

    Returns:
        q: coordinates satisfying all constraints
    """
    return newton_raphson(
        mbody,
        q0,
        t,
        allow_underconstrained=allow_underconstrained,
    )


def solve_velocity(mbody, q, t, allow_underconstrained=False):
    """Solve velocity problem: v = -Cq^{-1} * Ct.

    From the constraint velocity equation: Cq * v + Ct = 0

    Handles both:
    - Square systems (nrestr = ncoord): uses direct solve
    - Rectangular systems (nrestr < ncoord): uses least-squares (minimum-norm solution)

    Migrated from SimulacionCinematica.m line 34: v(:,i) = -Cq\\Ct

    Args:
        mbody: mechanism definition
        q: current position (must satisfy constraints)
        t: current time

    Returns:
        v: velocity vector
    """
    Cq = jacobian(mbody, q, t)
    Ct = dt_constraints(mbody, q, t)

    # If square system, use direct solve for efficiency
    if Cq.shape[0] == Cq.shape[1]:
        try:
            return np.linalg.solve(Cq, -Ct)
        except np.linalg.LinAlgError as exc:
            raise MechanismSolveError(
                f"Singular velocity solve at t={t:.6g}; Jacobian shape={Cq.shape}."
            ) from exc
    # If underconstrained (more DOF than constraints), use least-squares
    # This gives the minimum-norm solution
    elif Cq.shape[0] < Cq.shape[1]:
        if not allow_underconstrained:
            raise MechanismSolveError(
                f"Underconstrained velocity solve at t={t:.6g}: "
                f"{Cq.shape[0]} constraints for {Cq.shape[1]} coordinates. "
                "Pass allow_underconstrained=True only when a minimum-norm "
                "velocity is intended."
            )
        return np.linalg.lstsq(Cq, -Ct, rcond=None)[0]
    # If overconstrained (more constraints than DOF), error
    else:
        raise MechanismSolveError(
            f"Overconstrained velocity solve at t={t:.6g}: "
            f"{Cq.shape[0]} constraints for {Cq.shape[1]} coordinates."
        )


def solve_acceleration(mbody, q, v, t, allow_underconstrained=False):
    """Solve acceleration problem: a = -Cq^{-1} * (dCq*v + d²C/dt²).

    From the constraint acceleration equation:
        Cq * a + dCq/dt * v + d²C/dt² = 0

    Handles both square and underconstrained systems.

    Migrated from SimulacionCinematica.m line 40: a(:,i) = -Cq\\(DCq*v(:,i)+DCt)

    Args:
        mbody: mechanism definition
        q: current position
        v: current velocity
        t: current time

    Returns:
        a: acceleration vector
    """
    Cq = jacobian(mbody, q, t)
    DCq = dt_jacobian(mbody, q, v, t)
    DCt = dtdt_constraints(mbody, q, v, t)

    rhs = -(DCq @ v + DCt)

    # Handle both square and underconstrained systems
    if Cq.shape[0] == Cq.shape[1]:
        # Square system: use direct solve
        try:
            return np.linalg.solve(Cq, rhs)
        except np.linalg.LinAlgError as exc:
            raise MechanismSolveError(
                f"Singular acceleration solve at t={t:.6g}; Jacobian shape={Cq.shape}."
            ) from exc
    elif Cq.shape[0] < Cq.shape[1]:
        if not allow_underconstrained:
            raise MechanismSolveError(
                f"Underconstrained acceleration solve at t={t:.6g}: "
                f"{Cq.shape[0]} constraints for {Cq.shape[1]} coordinates. "
                "Pass allow_underconstrained=True only when a minimum-norm "
                "acceleration is intended."
            )
        # Underconstrained: use least-squares (minimum-norm solution)
        return np.linalg.lstsq(Cq, rhs, rcond=None)[0]
    else:
        # Overconstrained: error
        raise MechanismSolveError(
            f"Overconstrained acceleration solve at t={t:.6g}: "
            f"{Cq.shape[0]} constraints for {Cq.shape[1]} coordinates."
        )


def solve_acceleration_with_lagrange(mbody, q, v, t, M, Q_total):
    """Solve accelerations AND Lagrange multipliers from constrained dynamics.

    Solves the coupled system:
        [M        C_q^T] [a]   [Q_total - ∇V(q)      ]
        [C_q      0    ] [λ] = [-dC_q/dt @ v - d²C/dt²]

    Where:
        - M: mass matrix (ncoord x ncoord)
        - C_q: constraint Jacobian (nc x ncoord)
        - λ: Lagrange multipliers (nc,)
        - a: accelerations (ncoord,)

    This solves the full constrained dynamics equation:
        M @ a + ∇V = Q_total + C_q^T @ λ
    subject to:
        C_q @ a = -dC_q/dt @ v - d²C/dt²

    Args:
        mbody: mechanism definition
        q: current position (ncoord,)
        v: current velocity (ncoord,)
        t: current time
        M: mass matrix (ncoord x ncoord)
        Q_total: total generalized forces (ncoord,)

    Returns:
        a: acceleration vector (ncoord,)
        lambda_t: Lagrange multipliers for constraint forces (nc,)
    """
    # Get constraint Jacobian and its derivatives
    Cq = jacobian(mbody, q, t)  # (nc x ncoord)
    DCq = dt_jacobian(mbody, q, v, t)  # (nc x ncoord)
    DCt = dtdt_constraints(mbody, q, v, t)  # (nc,)

    ncoord = len(q)
    nc = Cq.shape[0]  # number of constraints

    # Build the saddle-point system
    # [M    C_q^T] [a]   [Q_total]
    # [C_q   0  ] [λ] = [-rhs   ]
    # where rhs = DCq @ v + DCt

    rhs_constraint = -(DCq @ v + DCt)  # constraint RHS

    # Build full system matrix
    A_block = np.zeros((ncoord + nc, ncoord + nc))
    A_block[:ncoord, :ncoord] = M
    A_block[:ncoord, ncoord:] = Cq.T
    A_block[ncoord:, :ncoord] = Cq

    # Build RHS
    b = np.zeros(ncoord + nc)
    b[:ncoord] = Q_total
    b[ncoord:] = rhs_constraint

    # Solve the coupled system.
    try:
        x = np.linalg.solve(A_block, b)
        a = x[:ncoord]
        lambda_t = x[ncoord:]
        return a, lambda_t
    except np.linalg.LinAlgError as exc:
        raise MechanismSolveError(
            f"Singular constrained-dynamics solve at t={t:.6g}; "
            f"saddle-point matrix shape={A_block.shape}."
        ) from exc
