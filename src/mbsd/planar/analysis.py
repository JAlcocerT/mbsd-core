"""Public planar analysis primitives for downstream workflows.

These are the building blocks a sizing, FEM-handoff or synthesis workflow
needs beyond ``simulate`` / ``inverse_dynamics`` / ``joint_loads``: the mass
matrix, generalized forces, constraint residuals and Jacobians with their time
derivatives, constraint-row groups, and point kinematics. Every function takes
a ``PlanarMechanism`` (or its low-level ``MBody``) and plain coordinate
vectors ``q``, ``v``, ``a`` laid out as ``[x, y, theta]`` per body.

The internal modules behind them (``constraints``, ``jacobians``,
``derivatives``, ``forces``, ``kinematics``) carry no stability promise;
this namespace does.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import kinematics as _kinematics
from .constraints import constraints as _constraints
from .derivatives import dt_constraints as _dt_constraints
from .derivatives import dt_jacobian as _dt_jacobian
from .derivatives import dtdt_constraints as _dtdt_constraints
from .dynamics import assemble_system as _assemble_system
from .forces import mass_matrix as _mass_matrix
from .jacobians import jacobian as _jacobian
from .model import MBody
from .reactions import ConstraintGroup
from .reactions import constraint_groups as _constraint_groups

__all__ = [
    "ConstraintGroup",
    "acceleration_residual",
    "body_vector",
    "constraint_groups",
    "constraint_jacobian",
    "constraint_residual",
    "constraint_second_time_derivative",
    "constraint_time_derivative",
    "equation_of_motion_residual",
    "generalized_forces",
    "jacobian_rate",
    "mass_matrix",
    "point_acceleration",
    "point_position",
    "point_velocity",
    "rotation_matrix",
]


def mass_matrix(mechanism, q) -> np.ndarray:
    """Mass matrix ``M(q)`` (ncoord x ncoord), including offset-COM coupling."""
    return _mass_matrix(_model(mechanism), _state(mechanism, q, "q"))


def generalized_forces(mechanism, q, v, t: float = 0.0) -> np.ndarray:
    """Applied generalized forces ``Q(q, v, t)``: gravity, offset-COM inertial
    terms, registered springs and portable or callback loads."""
    model = _model(mechanism)
    q, v = _state(mechanism, q, "q"), _state(mechanism, v, "v")
    springs, q_user_fn = (
        mechanism._force_inputs() if hasattr(mechanism, "_force_inputs") else ([], None)
    )
    q_user = None if q_user_fn is None else q_user_fn(float(t), q, v)
    return _assemble_system(model, q, v, float(t), springs, q_user)[0]


def constraint_residual(mechanism, q, t: float = 0.0) -> np.ndarray:
    """Constraint vector ``C(q, t)``; zero when the mechanism is assembled."""
    return _constraints(_model(mechanism), _state(mechanism, q, "q"), float(t))


def constraint_jacobian(mechanism, q, t: float = 0.0) -> np.ndarray:
    """Constraint Jacobian ``C_q`` (nrestr x ncoord)."""
    return _jacobian(_model(mechanism), _state(mechanism, q, "q"), float(t))


def jacobian_rate(mechanism, q, v, t: float = 0.0) -> np.ndarray:
    """Time derivative of the Jacobian ``dC_q/dt`` along the velocity ``v``."""
    return _dt_jacobian(
        _model(mechanism), _state(mechanism, q, "q"), _state(mechanism, v, "v"), float(t)
    )


def constraint_time_derivative(mechanism, q, t: float = 0.0) -> np.ndarray:
    """Explicit time derivative ``C_t`` (non-zero only for drives)."""
    return _dt_constraints(_model(mechanism), _state(mechanism, q, "q"), float(t))


def constraint_second_time_derivative(mechanism, q, v, t: float = 0.0) -> np.ndarray:
    """Explicit second time derivative ``C_tt`` (non-zero only for drives)."""
    return _dtdt_constraints(
        _model(mechanism), _state(mechanism, q, "q"), _state(mechanism, v, "v"), float(t)
    )


def acceleration_residual(mechanism, q, v, a, t: float = 0.0) -> np.ndarray:
    """Acceleration-level residual ``C_q a + dC_q/dt v + C_tt``."""
    q, v, a = (_state(mechanism, value, name) for value, name in ((q, "q"), (v, "v"), (a, "a")))
    return (
        constraint_jacobian(mechanism, q, t) @ a
        + jacobian_rate(mechanism, q, v, t) @ v
        + constraint_second_time_derivative(mechanism, q, v, t)
    )


def equation_of_motion_residual(mechanism, q, v, a, multipliers, t: float = 0.0) -> np.ndarray:
    """Residual of ``M a + C_q.T λ - Q``; zero for a consistent dynamic state."""
    q, v, a = (_state(mechanism, value, name) for value, name in ((q, "q"), (v, "v"), (a, "a")))
    multipliers = np.asarray(multipliers, dtype=float)
    return (
        mass_matrix(mechanism, q) @ a
        + constraint_jacobian(mechanism, q, t).T @ multipliers
        - generalized_forces(mechanism, q, v, t)
    )


def constraint_groups(mechanism) -> list[ConstraintGroup]:
    """Named constraint-row groups in assembly order; ``group.rows`` slices
    ``C``, ``C_q`` and the multiplier vector."""
    return _constraint_groups(_model(mechanism))


def rotation_matrix(theta: float) -> np.ndarray:
    """Planar rotation matrix ``A(theta)`` mapping body-local to world vectors."""
    return _kinematics.rot_mat(float(theta))


def body_vector(q, body, world_vector) -> np.ndarray:
    """Express a world vector in a body's local frame, e.g. a pin force for FEM."""
    index = _body(body)
    return rotation_matrix(np.asarray(q, dtype=float)[3 * index + 2]).T @ np.asarray(
        world_vector, dtype=float
    )


def point_position(q, body, local_point=(0.0, 0.0)) -> np.ndarray:
    """World position of a body-local point."""
    index = _body(body)
    return _kinematics.point_position(_block(q, index), np.asarray(local_point, dtype=float))


def point_velocity(q, v, body, local_point=(0.0, 0.0)) -> np.ndarray:
    """World velocity of a body-local point."""
    index = _body(body)
    return _kinematics.point_velocity(
        _block(q, index), _block(v, index), np.asarray(local_point, dtype=float)
    )


def point_acceleration(q, v, a, body, local_point=(0.0, 0.0)) -> np.ndarray:
    """World acceleration of a body-local point."""
    index = _body(body)
    return _kinematics.point_acceleration(
        _block(q, index), _block(v, index), _block(a, index), np.asarray(local_point, dtype=float)
    )


def _model(mechanism: Any) -> MBody:
    model = getattr(mechanism, "model", mechanism)
    if not isinstance(model, MBody):
        raise TypeError("mechanism must be a PlanarMechanism or MBody")
    return model


def _state(mechanism, value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    ncoord = _model(mechanism).ncoord
    if array.shape != (ncoord,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape ({ncoord},)")
    return array


def _body(body) -> int:
    return int(getattr(body, "index", body))


def _block(value, index: int) -> np.ndarray:
    return np.asarray(value, dtype=float)[3 * index : 3 * index + 3]
