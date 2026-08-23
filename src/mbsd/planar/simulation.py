"""Kinematic simulation loop for planar mechanisms."""

import numpy as np

from .solver import solve_acceleration, solve_position, solve_velocity


def run_kinematic_simulation(mbody, q0, tspan, verbose=False):
    """Run position, velocity, and acceleration solves over a time array.

    For each time step:
      1. Solve position: ``C(q, t) = 0``.
      2. Solve velocity: ``Cq @ v = -Ct``.
      3. Solve acceleration: ``Cq @ a = -(dCq @ v + Ctt)``.

    Args:
        mbody: mechanism definition
        q0: initial coordinate vector
        tspan: strictly increasing time array
        verbose: print progress

    Returns:
        Tuple of ``(q_all, v_all, a_all)`` arrays with shape
        ``(ncoord, len(tspan))``.
    """
    nsteps = len(tspan)
    ncoord = mbody.ncoord

    q_all = np.zeros((ncoord, nsteps))
    v_all = np.zeros((ncoord, nsteps))
    a_all = np.zeros((ncoord, nsteps))

    q_guess = q0.copy()

    for i, t in enumerate(tspan):
        mbody.t = t

        q = solve_position(mbody, q_guess, t)
        q_all[:, i] = q

        v = solve_velocity(mbody, q, t)
        v_all[:, i] = v

        a = solve_acceleration(mbody, q, v, t)
        a_all[:, i] = a

        q_guess = q.copy()

        if verbose and (i % max(1, nsteps // 10) == 0):
            print(f"  Step {i}/{nsteps}, t = {t:.4f}")

    return q_all, v_all, a_all
