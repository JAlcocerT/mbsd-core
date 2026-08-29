"""Small declarative facade for planar mechanisms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .dynamics import extract_dynamics_solution, solve_dynamics_scipy
from .constraints import constraints
from .model import Body, MBody, PrismJoint, RevJoint, UserConstraint
from .simulation import run_kinematic_simulation
from .solver import solve_position, solve_velocity


ArrayLike2 = Iterable[float]


@dataclass(frozen=True)
class BodyHandle:
    """Stable public reference to a body in a planar mechanism."""

    index: int
    name: str

    def __int__(self) -> int:
        return self.index


@dataclass(frozen=True)
class KinematicResult:
    """Position, velocity, and acceleration history from a kinematic solve."""

    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray


@dataclass(frozen=True)
class DynamicsResult:
    """Position and velocity history from a forward-dynamics solve."""

    t: np.ndarray
    q: np.ndarray
    v: np.ndarray


@dataclass(frozen=True)
class ResultDiagnostics:
    """Small summary for checking whether a mechanism result is plausible."""

    coordinates: int
    constraints: int
    steps: int
    t_start: float
    t_end: float
    max_constraint_residual: float
    finite: bool

    @property
    def degrees_of_freedom(self) -> int:
        return self.coordinates - self.constraints

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "coordinates": self.coordinates,
            "constraints": self.constraints,
            "degrees_of_freedom": self.degrees_of_freedom,
            "steps": self.steps,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "max_constraint_residual": self.max_constraint_residual,
            "finite": self.finite,
        }


class Mechanism:
    """Factory namespace for supported mechanism formulations."""

    @staticmethod
    def planar(gravity: ArrayLike2 = (0.0, -9.81)) -> "PlanarMechanism":
        return PlanarMechanism(gravity=gravity)


class PlanarMechanism:
    """Declarative builder over the 2D reference-coordinate kernel.

    The underlying model is available through ``.model`` for lower-level
    workflows. Body coordinates are stored as ``[x, y, theta]``.
    """

    def __init__(self, gravity: ArrayLike2 = (0.0, -9.81)):
        self.model = MBody()
        self.model.g = _as_vector(gravity, "gravity", 2)

    @property
    def ncoord(self) -> int:
        return self.model.ncoord

    @property
    def nrestr(self) -> int:
        return self.model.nrestr

    def ground(self, name: str = "ground") -> BodyHandle:
        """Return body 0, creating a fixed ground body if needed."""
        if not self.model.bodies:
            self.model.bodies.append(Body(name=name, mass=0.0, inertia=0.0))
        return BodyHandle(0, self.model.bodies[0].name)

    def body(
        self,
        name: str,
        mass: float = 1.0,
        inertia: float = 1.0,
        center_of_mass: ArrayLike2 = (0.0, 0.0),
    ) -> BodyHandle:
        """Add a planar rigid body and return its handle."""
        if not name:
            raise ValueError("body name must be non-empty")
        mass = _as_finite_scalar(mass, "mass")
        inertia = _as_finite_scalar(inertia, "inertia")
        if mass <= 0.0:
            raise ValueError("body mass must be positive")
        if inertia <= 0.0:
            raise ValueError("body inertia must be positive")
        center_of_mass = _as_vector(center_of_mass, "center_of_mass", 2)
        if not self.model.bodies:
            self.ground()
        idx = len(self.model.bodies)
        self.model.bodies.append(
            Body(
                name=name,
                mass=mass,
                inertia=inertia,
                rG=center_of_mass,
            )
        )
        return BodyHandle(idx, name)

    def pin(
        self,
        body_a: int | BodyHandle,
        body_b: int | BodyHandle,
        point: ArrayLike2 | None = None,
        point_a: ArrayLike2 | None = None,
        point_b: ArrayLike2 | None = None,
    ) -> RevJoint:
        """Add a revolute joint between two local attachment points."""
        body_a_idx = self._idx(body_a)
        body_b_idx = self._idx(body_b)
        if point is not None:
            point_a = point if point_a is None else point_a
            point_b = point if point_b is None else point_b
        ri = _as_vector((0.0, 0.0) if point_a is None else point_a, "point_a", 2)
        rj = _as_vector((0.0, 0.0) if point_b is None else point_b, "point_b", 2)
        joint = RevJoint(i=body_a_idx, j=body_b_idx, ri=ri, rj=rj)
        self.model.rev_joints.append(joint)
        return joint

    def slider(
        self,
        rail_body: int | BodyHandle,
        slider_body: int | BodyHandle,
        axis: ArrayLike2 = (1.0, 0.0),
        point_rail: ArrayLike2 = (0.0, 0.0),
        point_slider: ArrayLike2 = (0.0, 0.0),
    ) -> PrismJoint:
        """Add a prismatic joint.

        ``axis`` is the allowed sliding direction in the rail body's frame.
        The legacy kernel stores the normal direction, so this facade converts
        ``axis`` into the perpendicular constraint vector.
        """
        rail_idx = self._idx(rail_body)
        slider_idx = self._idx(slider_body)
        axis_arr = _as_vector(axis, "axis", 2)
        norm = np.linalg.norm(axis_arr)
        if norm == 0.0:
            raise ValueError("slider axis must be non-zero")
        axis_arr = axis_arr / norm
        normal = np.array([-axis_arr[1], axis_arr[0]])
        joint = PrismJoint(
            i=rail_idx,
            j=slider_idx,
            ri=_as_vector(point_rail, "point_rail", 2),
            rj=_as_vector(point_slider, "point_slider", 2),
            hi=normal,
        )
        self.model.prism_joints.append(joint)
        return joint

    def motor(
        self,
        body: int | BodyHandle,
        omega: float,
        phase: float = 0.0,
    ) -> UserConstraint:
        """Drive a body's absolute angle as ``theta = phase + omega * t``."""
        idx = self._idx(body)
        omega = _as_finite_scalar(omega, "omega")
        phase = _as_finite_scalar(phase, "phase")
        col = 3 * idx + 2

        def constraint(mb: MBody, q: np.ndarray, t: float) -> np.ndarray:
            return np.array([q[col] - (phase + omega * t)])

        def jacobian(mb: MBody, q: np.ndarray, t: float) -> np.ndarray:
            J = np.zeros((1, mb.ncoord))
            J[0, col] = 1.0
            return J

        def dt_jacobian(mb: MBody, q: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
            return np.zeros((1, mb.ncoord))

        def dt_constraint(mb: MBody, q: np.ndarray, t: float) -> np.ndarray:
            return np.array([-omega])

        def dtdt_constraint(mb: MBody, q: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
            return np.array([0.0])

        drive = UserConstraint(
            constraint,
            jacobian,
            dt_jacobian,
            dt_constraint,
            dtdt_constraint,
            count=1,
        )
        self.model.user_constraints.append(drive)
        return drive

    def coordinate_drive(
        self,
        body: int | BodyHandle,
        coordinate: str,
        value: Callable[[float], float],
        velocity: Callable[[float], float],
        acceleration: Callable[[float], float],
    ) -> UserConstraint:
        """Drive one body coordinate with a scalar time function."""
        offsets = {"x": 0, "y": 1, "theta": 2}
        if coordinate not in offsets:
            raise ValueError("coordinate must be one of: x, y, theta")
        for fn_name, fn in {
            "value": value,
            "velocity": velocity,
            "acceleration": acceleration,
        }.items():
            if not callable(fn):
                raise TypeError(f"{fn_name} must be callable")
        col = 3 * self._idx(body) + offsets[coordinate]

        def constraint(mb: MBody, q: np.ndarray, t: float) -> np.ndarray:
            return np.array([q[col] - value(t)])

        def jacobian(mb: MBody, q: np.ndarray, t: float) -> np.ndarray:
            J = np.zeros((1, mb.ncoord))
            J[0, col] = 1.0
            return J

        def dt_jacobian(mb: MBody, q: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
            return np.zeros((1, mb.ncoord))

        def dt_constraint(mb: MBody, q: np.ndarray, t: float) -> np.ndarray:
            return np.array([-velocity(t)])

        def dtdt_constraint(mb: MBody, q: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
            return np.array([-acceleration(t)])

        drive = UserConstraint(
            constraint,
            jacobian,
            dt_jacobian,
            dt_constraint,
            dtdt_constraint,
            count=1,
        )
        self.model.user_constraints.append(drive)
        return drive

    def solve_position(
        self,
        q0: np.ndarray | None = None,
        t: float = 0.0,
        allow_underconstrained: bool = False,
    ) -> np.ndarray:
        if q0 is None:
            q0 = np.zeros(self.model.ncoord)
        return solve_position(
            self.model,
            _as_state(q0, "q0", self.model.ncoord),
            _as_finite_scalar(t, "t"),
            allow_underconstrained=allow_underconstrained,
        )

    def solve_velocity(
        self,
        q: np.ndarray,
        t: float = 0.0,
        allow_underconstrained: bool = False,
    ) -> np.ndarray:
        return solve_velocity(
            self.model,
            _as_state(q, "q", self.model.ncoord),
            _as_finite_scalar(t, "t"),
            allow_underconstrained=allow_underconstrained,
        )

    def constraint_residual(self, q: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Return the constraint residual vector ``C(q, t)``."""
        return constraints(
            self.model,
            _as_state(q, "q", self.model.ncoord),
            _as_finite_scalar(t, "t"),
        )

    def constraint_residuals(self, result: KinematicResult | DynamicsResult) -> np.ndarray:
        """Return constraint residuals for every result step.

        The returned array has shape ``(n_constraints, n_steps)`` so each
        column is ``C(q[:, i], t[i])``.
        """
        self._validate_result(result)
        residuals = np.zeros((self.model.nrestr, len(result.t)))
        for i, ti in enumerate(result.t):
            residuals[:, i] = self.constraint_residual(result.q[:, i], float(ti))
        return residuals

    def max_constraint_residual(self, result: KinematicResult | DynamicsResult) -> float:
        """Return the maximum infinity-norm constraint residual over a result."""
        residuals = self.constraint_residuals(result)
        if residuals.size == 0:
            return 0.0
        return float(np.max(np.abs(residuals)))

    def assert_constraints_satisfied(
        self,
        result: KinematicResult | DynamicsResult,
        tol: float = 1e-8,
    ) -> None:
        """Raise ``MechanismSolveError`` when a result violates constraints."""
        from ..errors import MechanismSolveError

        tol = _as_finite_scalar(tol, "tol")
        if tol <= 0.0:
            raise ValueError("tol must be positive")
        max_residual = self.max_constraint_residual(result)
        if max_residual > tol:
            raise MechanismSolveError(
                f"Constraint residual {max_residual:.3e} exceeds tolerance {tol:.3e}."
            )

    def diagnostics(self, result: KinematicResult | DynamicsResult) -> ResultDiagnostics:
        """Return a compact diagnostic summary for a solved result."""
        self._validate_result(result)
        return ResultDiagnostics(
            coordinates=self.model.ncoord,
            constraints=self.model.nrestr,
            steps=len(result.t),
            t_start=float(result.t[0]),
            t_end=float(result.t[-1]),
            max_constraint_residual=self.max_constraint_residual(result),
            finite=bool(np.all(np.isfinite(result.t)) and np.all(np.isfinite(result.q))),
        )

    def solve_kinematics(
        self,
        t: np.ndarray,
        q0: np.ndarray | None = None,
        verbose: bool = False,
    ) -> KinematicResult:
        """Run position, velocity, and acceleration solves over ``t``."""
        t = _as_time_array(t)
        q_initial = self.solve_position(q0, float(t[0]))
        q, v, a = run_kinematic_simulation(self.model, q_initial, t, verbose=verbose)
        return KinematicResult(t=t, q=q, v=v, a=a)

    def simulate(
        self,
        t: np.ndarray,
        q0: np.ndarray,
        v0: np.ndarray | None = None,
        **kwargs,
    ) -> DynamicsResult:
        """Run forward dynamics with the scipy-based constrained integrator."""
        t = _as_time_array(t)
        q0 = self.solve_position(
            _as_state(q0, "q0", self.model.ncoord),
            float(t[0]),
            allow_underconstrained=True,
        )
        if v0 is None:
            v0 = solve_velocity(
                self.model,
                q0,
                float(t[0]),
                allow_underconstrained=True,
            )
        else:
            v0 = _as_state(v0, "v0", self.model.ncoord)
        sol = solve_dynamics_scipy(self.model, q0, v0, t, **kwargs)
        if not sol.success:
            from ..errors import MechanismSolveError

            raise MechanismSolveError(f"Dynamic integration failed: {sol.message}")
        q, v, t_out = extract_dynamics_solution(self.model, sol)
        return DynamicsResult(t=t_out, q=q, v=v)

    def _idx(self, body: int | BodyHandle) -> int:
        idx = body.index if isinstance(body, BodyHandle) else int(body)
        if idx < 0 or idx >= len(self.model.bodies):
            raise IndexError(f"body index {idx} is out of range")
        return idx

    def _validate_result(self, result: KinematicResult | DynamicsResult) -> None:
        t = np.asarray(result.t, dtype=float)
        q = np.asarray(result.q, dtype=float)
        if t.ndim != 1 or len(t) == 0:
            raise ValueError("result.t must be a non-empty 1D array")
        if q.shape != (self.model.ncoord, len(t)):
            raise ValueError(
                f"result.q must have shape ({self.model.ncoord}, {len(t)}), got {q.shape}"
            )
        if not np.all(np.isfinite(t)):
            raise ValueError("result.t must contain only finite values")
        if not np.all(np.isfinite(q)):
            raise ValueError("result.q must contain only finite values")


def _as_vector(value: ArrayLike2, name: str, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"{name} must be a finite vector with shape ({length},)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_state(value: np.ndarray, name: str, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"{name} must be a finite vector with shape ({length},)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_time_array(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("t must be a non-empty 1D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("t must contain only finite values")
    if len(arr) > 1 and not np.all(np.diff(arr) > 0.0):
        raise ValueError("t must be strictly increasing")
    return arr


def _as_finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value
