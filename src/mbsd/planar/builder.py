"""Small declarative facade for planar mechanisms."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .dynamics import (
    assemble_system,
    ensure_centroidal_dynamics,
    extract_dynamics_solution,
    solve_dynamics_scipy,
    solve_inverse_dynamics,
)
from .forces import Spring, kinetic_energy, potential_energy
from .constraints import constraints
from .derivatives import dt_constraints
from .jacobians import jacobian
from .laws import ConstantSpeed, MotionLaw
from .loads import (
    CallbackLoad,
    ConstantForce,
    ConstantTorque,
    Load,
    TabulatedForce,
    TabulatedTorque,
)
from .model import Body, MBody, PrismJoint, RevJoint, UserConstraint
from .reactions import JointLoads, joint_loads
from .simulation import run_kinematic_simulation
from .solver import solve_acceleration_with_lagrange, solve_position, solve_velocity


DEFAULT_MAX_STEPS = 100_000


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
    """Position and velocity history from a forward-dynamics solve.

    ``multipliers`` holds the saddle-system Lagrange multipliers sampled at
    every output time (``nrestr x nsteps``); pass the result to
    ``joint_loads`` for physical reactions.
    """

    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    provenance: Mapping[str, Any] = field(default_factory=dict)
    multipliers: np.ndarray | None = None


@dataclass(frozen=True)
class InverseDynamicsResult:
    """Kinematically driven trajectory with the multipliers that sustain it."""

    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    multipliers: np.ndarray
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Issue:
    """One structured finding from ``validate`` with a stable ``code``."""

    code: str
    severity: str
    message: str
    remediation: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Pre-solve model report: ``ok`` is false when any issue is an error."""

    ok: bool
    portable: bool
    classification: str | None
    degrees_of_freedom: int | None
    issues: tuple[Issue, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "portable": self.portable,
            "classification": self.classification,
            "degrees_of_freedom": self.degrees_of_freedom,
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class Check:
    """One post-solve check: ``passed`` is ``None`` when the check was skipped."""

    name: str
    value: float | None
    threshold: float | None
    passed: bool | None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "note": self.note,
        }


@dataclass(frozen=True)
class CheckReport:
    """Post-solve verdict with the numbers and thresholds behind it."""

    passed: bool
    result_type: str
    checks: tuple[Check, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "result_type": self.result_type,
            "checks": [check.as_dict() for check in self.checks],
        }


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


@dataclass(frozen=True)
class ModelDiagnostics:
    """Static diagnostic summary for the current constraint system."""

    coordinates: int
    constraints: int
    nominal_degrees_of_freedom: int
    rank_degrees_of_freedom: int
    jacobian_rank: int
    classification: str
    underconstrained: bool
    overconstrained: bool
    rank_deficient: bool
    singular: bool

    @property
    def degrees_of_freedom(self) -> int:
        return self.rank_degrees_of_freedom

    def as_dict(self) -> dict[str, int | bool | str]:
        return {
            "coordinates": self.coordinates,
            "constraints": self.constraints,
            "degrees_of_freedom": self.degrees_of_freedom,
            "nominal_degrees_of_freedom": self.nominal_degrees_of_freedom,
            "rank_degrees_of_freedom": self.rank_degrees_of_freedom,
            "jacobian_rank": self.jacobian_rank,
            "classification": self.classification,
            "underconstrained": self.underconstrained,
            "overconstrained": self.overconstrained,
            "rank_deficient": self.rank_deficient,
            "singular": self.singular,
        }


@dataclass(frozen=True)
class ConfigurationDiagnostics:
    """Single-configuration health summary for a planar mechanism."""

    time: float
    coordinates: int
    constraints: int
    constraint_norm: float
    velocity_residual_norm: float | None
    nominal_degrees_of_freedom: int
    rank_degrees_of_freedom: int
    jacobian_rank: int
    classification: str
    underconstrained: bool
    overconstrained: bool
    rank_deficient: bool
    singular: bool
    finite: bool

    @property
    def degrees_of_freedom(self) -> int:
        return self.rank_degrees_of_freedom

    def as_dict(self) -> dict[str, float | int | bool | str | None]:
        return {
            "time": self.time,
            "coordinates": self.coordinates,
            "constraints": self.constraints,
            "constraint_norm": self.constraint_norm,
            "velocity_residual_norm": self.velocity_residual_norm,
            "degrees_of_freedom": self.degrees_of_freedom,
            "nominal_degrees_of_freedom": self.nominal_degrees_of_freedom,
            "rank_degrees_of_freedom": self.rank_degrees_of_freedom,
            "jacobian_rank": self.jacobian_rank,
            "classification": self.classification,
            "underconstrained": self.underconstrained,
            "overconstrained": self.overconstrained,
            "rank_deficient": self.rank_deficient,
            "singular": self.singular,
            "finite": self.finite,
        }


class Mechanism:
    """Factory namespace for supported mechanism formulations."""

    @staticmethod
    def planar(gravity: ArrayLike2 = (0.0, -9.81)) -> "PlanarMechanism":
        return PlanarMechanism(gravity=gravity)

    @staticmethod
    def spatial():
        """Return the experimental constrained spatial-kinematics builder."""
        from ..spatial import SpatialMechanism

        return SpatialMechanism()


class PlanarMechanism:
    """Declarative builder over the 2D reference-coordinate kernel.

    The underlying model is available through ``.model`` for lower-level
    workflows. Body coordinates are stored as ``[x, y, theta]``.
    """

    def __init__(self, gravity: ArrayLike2 = (0.0, -9.81)):
        self.model = MBody()
        self.model.g = _as_vector(gravity, "gravity", 2)
        self.springs: list[Spring] = []
        self.loads: list[Load] = []
        self._names: set[str] = set()

    def _claim_name(self, name: str | None, prefix: str) -> str:
        """Return a unique element name, generating ``prefix_N`` when omitted."""
        if name is None:
            index = 0
            while f"{prefix}_{index}" in self._names:
                index += 1
            name = f"{prefix}_{index}"
        if not isinstance(name, str) or not name:
            raise ValueError("element name must be a non-empty string")
        if name in self._names:
            raise ValueError(f"element name {name!r} is already used in this mechanism")
        self._names.add(name)
        return name

    def _ensure_nonempty_model(self) -> None:
        if not self.model.bodies:
            raise ValueError("mechanism must contain at least one body")

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
        name: str | None = None,
    ) -> RevJoint:
        """Add a revolute joint between two local attachment points.

        ``name`` identifies the joint in reactions and exports; it defaults to
        ``pin_N``.
        """
        body_a_idx = self._idx(body_a)
        body_b_idx = self._idx(body_b)
        if point is not None:
            point_a = point if point_a is None else point_a
            point_b = point if point_b is None else point_b
        ri = _as_vector((0.0, 0.0) if point_a is None else point_a, "point_a", 2)
        rj = _as_vector((0.0, 0.0) if point_b is None else point_b, "point_b", 2)
        joint = RevJoint(
            i=body_a_idx, j=body_b_idx, ri=ri, rj=rj, name=self._claim_name(name, "pin")
        )
        self.model.rev_joints.append(joint)
        return joint

    def slider(
        self,
        rail_body: int | BodyHandle,
        slider_body: int | BodyHandle,
        axis: ArrayLike2 = (1.0, 0.0),
        point_rail: ArrayLike2 = (0.0, 0.0),
        point_slider: ArrayLike2 = (0.0, 0.0),
        name: str | None = None,
    ) -> PrismJoint:
        """Add a prismatic joint.

        ``axis`` is the allowed sliding direction in the rail body's frame.
        The legacy kernel stores the normal direction, so this facade converts
        ``axis`` into the perpendicular constraint vector. ``name`` defaults to
        ``slider_N``.
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
            name=self._claim_name(name, "slider"),
        )
        self.model.prism_joints.append(joint)
        return joint

    def motor(
        self,
        body: int | BodyHandle,
        omega: float,
        phase: float = 0.0,
        name: str | None = None,
    ) -> UserConstraint:
        """Drive a body's absolute angle as ``theta = phase + omega * t``."""
        idx = self._idx(body)
        omega = _as_finite_scalar(omega, "omega")
        phase = _as_finite_scalar(phase, "phase")
        return self._add_drive(
            idx,
            "theta",
            ConstantSpeed(rate=omega, start=phase),
            name=self._claim_name(name, "motor"),
            metadata={"kind": "motor", "omega": omega, "phase": phase},
        )

    def coordinate_drive(
        self,
        body: int | BodyHandle,
        coordinate: str,
        value: Callable[[float], float] | None = None,
        velocity: Callable[[float], float] | None = None,
        acceleration: Callable[[float], float] | None = None,
        *,
        law: MotionLaw | None = None,
        name: str | None = None,
    ) -> UserConstraint:
        """Drive one body coordinate with a motion law or scalar time functions.

        Pass a portable ``law`` (``ConstantSpeed``, ``Polynomial``,
        ``Tabulated``) or the three callables ``value``, ``velocity`` and
        ``acceleration``. Callables work in Python but make the model
        non-portable: exports mark them opaque and importers reject them.
        """
        offsets = {"x": 0, "y": 1, "theta": 2}
        if coordinate not in offsets:
            raise ValueError("coordinate must be one of: x, y, theta")
        idx = self._idx(body)
        if law is not None:
            if not isinstance(law, MotionLaw):
                raise TypeError("law must be a MotionLaw such as ConstantSpeed or Tabulated")
            if any(fn is not None for fn in (value, velocity, acceleration)):
                raise ValueError("pass either law or value/velocity/acceleration, not both")
            return self._add_drive(
                idx,
                coordinate,
                law,
                name=self._claim_name(name, "drive"),
                metadata={"kind": "coordinate_drive"},
            )
        for fn_name, fn in {
            "value": value,
            "velocity": velocity,
            "acceleration": acceleration,
        }.items():
            if not callable(fn):
                raise TypeError(f"{fn_name} must be callable")
        return self._add_drive(
            idx,
            coordinate,
            _CallableLaw(value, velocity, acceleration),
            name=self._claim_name(name, "drive"),
            metadata={
                "kind": "coordinate_drive",
                "value": getattr(value, "__name__", "callable"),
                "velocity": getattr(velocity, "__name__", "callable"),
                "acceleration": getattr(acceleration, "__name__", "callable"),
            },
        )

    def _add_drive(
        self,
        body_idx: int,
        coordinate: str,
        law: Any,
        *,
        name: str,
        metadata: dict[str, Any],
    ) -> UserConstraint:
        col = 3 * body_idx + {"x": 0, "y": 1, "theta": 2}[coordinate]
        value, velocity, acceleration = law.value, law.velocity, law.acceleration
        portable = isinstance(law, MotionLaw)

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
            name=name,
        )
        drive.metadata = {
            **metadata,
            "name": name,
            "body": body_idx,
            "coordinate": coordinate,
            "portable": portable,
        }
        if portable:
            drive.metadata["law"] = law.to_dict()
        drive.law = law
        self.model.user_constraints.append(drive)
        return drive

    def spring(
        self,
        body_a: int | BodyHandle,
        body_b: int | BodyHandle,
        stiffness: float,
        natural_length: float,
        damping: float = 0.0,
        point_a: ArrayLike2 = (0.0, 0.0),
        point_b: ArrayLike2 = (0.0, 0.0),
        name: str | None = None,
    ) -> Spring:
        """Register a linear spring-damper; it applies to every solve and export."""
        stiffness = _as_finite_scalar(stiffness, "stiffness")
        damping = _as_finite_scalar(damping, "damping")
        natural_length = _as_finite_scalar(natural_length, "natural_length")
        if stiffness <= 0.0:
            raise ValueError("spring stiffness must be positive")
        if damping < 0.0:
            raise ValueError("spring damping must be non-negative")
        if natural_length < 0.0:
            raise ValueError("spring natural length must be non-negative")
        spring = Spring(
            self._idx(body_a),
            self._idx(body_b),
            stiffness,
            c=damping,
            l0=natural_length,
            ri=_as_vector(point_a, "point_a", 2),
            rj=_as_vector(point_b, "point_b", 2),
        )
        spring.name = self._claim_name(name, "spring")
        self.springs.append(spring)
        return spring

    def constant_force(
        self,
        body: int | BodyHandle,
        force: ArrayLike2,
        point: ArrayLike2 = (0.0, 0.0),
        name: str | None = None,
    ) -> ConstantForce:
        """Apply a constant world-frame force at a body-local point."""
        load = ConstantForce(
            self._claim_name(name, "force"),
            self._idx(body),
            _as_tuple(force, "force", 2),
            _as_tuple(point, "point", 2),
        )
        self.loads.append(load)
        return load

    def constant_torque(
        self,
        body: int | BodyHandle,
        torque: float,
        name: str | None = None,
    ) -> ConstantTorque:
        """Apply a constant counterclockwise torque to a body."""
        load = ConstantTorque(
            self._claim_name(name, "torque"),
            self._idx(body),
            _as_finite_scalar(torque, "torque"),
        )
        self.loads.append(load)
        return load

    def tabulated_force(
        self,
        body: int | BodyHandle,
        times: Iterable[float],
        fx: Iterable[float],
        fy: Iterable[float],
        point: ArrayLike2 = (0.0, 0.0),
        name: str | None = None,
    ) -> TabulatedForce:
        """Apply a piecewise-linear world-frame force history at a body-local point."""
        times_t = _as_samples(times, "times")
        load = TabulatedForce(
            self._claim_name(name, "force"),
            self._idx(body),
            times_t,
            _as_tuple(fx, "fx", len(times_t)),
            _as_tuple(fy, "fy", len(times_t)),
            _as_tuple(point, "point", 2),
        )
        self.loads.append(load)
        return load

    def tabulated_torque(
        self,
        body: int | BodyHandle,
        times: Iterable[float],
        values: Iterable[float],
        name: str | None = None,
    ) -> TabulatedTorque:
        """Apply a piecewise-linear torque history to a body."""
        times_t = _as_samples(times, "times")
        load = TabulatedTorque(
            self._claim_name(name, "torque"),
            self._idx(body),
            times_t,
            _as_tuple(values, "values", len(times_t)),
        )
        self.loads.append(load)
        return load

    def force(
        self,
        fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
        name: str | None = None,
    ) -> CallbackLoad:
        """Apply an arbitrary generalized force ``fn(t, q, v) -> (ncoord,)``.

        This is a Python-only escape hatch (for example penalty contact). The
        model becomes non-portable: exports mark the load opaque, ``validate``
        reports it, and importers reject it.
        """
        if not callable(fn):
            raise TypeError("fn must be callable")
        load = CallbackLoad(self._claim_name(name, "callback"), fn)
        self.loads.append(load)
        return load

    def _force_inputs(self, springs=None, Q_user_fn=None):
        """Combine registered springs/loads with per-call extras."""
        all_springs = list(self.springs) + list(springs or [])
        loads = list(self.loads)
        if not loads and Q_user_fn is None:
            return all_springs, None
        ncoord = self.model.ncoord

        def combined(t, q, v):
            Q = np.zeros(ncoord)
            for load in loads:
                Q += load.generalized(ncoord, q, v, t)
            if Q_user_fn is not None:
                Q += np.asarray(Q_user_fn(t, q, v), dtype=float)
            return Q

        return all_springs, combined

    def solve_position(
        self,
        q0: np.ndarray | None = None,
        t: float = 0.0,
        allow_underconstrained: bool = False,
    ) -> np.ndarray:
        self._ensure_nonempty_model()
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
        self._ensure_nonempty_model()
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
            finite=_result_arrays_are_finite(result),
        )

    def model_diagnostics(self, q: np.ndarray | None = None, t: float = 0.0) -> ModelDiagnostics:
        """Return coordinate, constraint, DOF, and Jacobian-rank diagnostics."""
        self._ensure_nonempty_model()
        if q is None:
            q = np.zeros(self.model.ncoord)
        q = _as_state(q, "q", self.model.ncoord)
        Cq = jacobian(self.model, q, _as_finite_scalar(t, "t"))
        rank = int(np.linalg.matrix_rank(Cq))
        underconstrained = self.model.nrestr < self.model.ncoord
        overconstrained = self.model.nrestr > self.model.ncoord
        rank_deficient = rank < min(Cq.shape)
        if overconstrained:
            classification = "overconstrained"
        elif underconstrained:
            classification = "underconstrained"
        elif rank_deficient:
            classification = "rank_deficient"
        else:
            classification = "fully_constrained"
        return ModelDiagnostics(
            coordinates=self.model.ncoord,
            constraints=self.model.nrestr,
            nominal_degrees_of_freedom=self.model.ncoord - self.model.nrestr,
            rank_degrees_of_freedom=self.model.ncoord - rank,
            jacobian_rank=rank,
            classification=classification,
            underconstrained=underconstrained,
            overconstrained=overconstrained,
            rank_deficient=rank_deficient,
            singular=rank_deficient,
        )

    def configuration_diagnostics(
        self,
        q: np.ndarray,
        t: float = 0.0,
        v: np.ndarray | None = None,
    ) -> ConfigurationDiagnostics:
        """Return residual and Jacobian health metrics for one configuration."""
        self._ensure_nonempty_model()
        t = _as_finite_scalar(t, "t")
        q = _as_state(q, "q", self.model.ncoord)
        C = constraints(self.model, q, t)
        Cq = jacobian(self.model, q, t)
        rank = int(np.linalg.matrix_rank(Cq))
        underconstrained = self.model.nrestr < self.model.ncoord
        overconstrained = self.model.nrestr > self.model.ncoord
        rank_deficient = rank < min(Cq.shape)
        if overconstrained:
            classification = "overconstrained"
        elif underconstrained:
            classification = "underconstrained"
        elif rank_deficient:
            classification = "rank_deficient"
        else:
            classification = "fully_constrained"
        velocity_residual_norm = None
        finite = bool(np.all(np.isfinite(C)) and np.all(np.isfinite(Cq)))
        if v is not None:
            v = _as_state(v, "v", self.model.ncoord)
            velocity_residual = Cq @ v + dt_constraints(self.model, q, t)
            velocity_residual_norm = float(np.linalg.norm(velocity_residual, ord=np.inf))
            finite = finite and bool(np.all(np.isfinite(velocity_residual)))
        return ConfigurationDiagnostics(
            time=t,
            coordinates=self.model.ncoord,
            constraints=self.model.nrestr,
            constraint_norm=float(np.linalg.norm(C, ord=np.inf)) if C.size else 0.0,
            velocity_residual_norm=velocity_residual_norm,
            nominal_degrees_of_freedom=self.model.ncoord - self.model.nrestr,
            rank_degrees_of_freedom=self.model.ncoord - rank,
            jacobian_rank=rank,
            classification=classification,
            underconstrained=underconstrained,
            overconstrained=overconstrained,
            rank_deficient=rank_deficient,
            singular=rank_deficient,
            finite=finite,
        )

    def to_dict(
        self,
        *,
        springs: Iterable[Spring] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-ready mechanism export payload."""
        self._ensure_nonempty_model()
        from .export import mechanism_to_dict

        return mechanism_to_dict(self, springs=springs, metadata=metadata)

    def to_json(
        self,
        path: str | Path,
        *,
        indent: int = 2,
        springs: Iterable[Spring] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write a mechanism export JSON file and return its path."""
        self._ensure_nonempty_model()
        from .export import mechanism_to_json

        return mechanism_to_json(
            self,
            path,
            indent=indent,
            springs=springs,
            metadata=metadata,
        )

    def result_to_dict(
        self,
        result: KinematicResult | DynamicsResult,
        *,
        include_diagnostics: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-ready result export payload."""
        from .export import result_to_dict

        return result_to_dict(
            self,
            result,
            include_diagnostics=include_diagnostics,
            metadata=metadata,
        )

    def result_to_json(
        self,
        result: KinematicResult | DynamicsResult,
        path: str | Path,
        *,
        indent: int = 2,
        include_diagnostics: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write a result export JSON file and return its path."""
        from .export import result_to_json

        return result_to_json(
            self,
            result,
            path,
            indent=indent,
            include_diagnostics=include_diagnostics,
            metadata=metadata,
        )

    def result_to_csv(self, result: KinematicResult | DynamicsResult, path: str | Path) -> Path:
        """Write a wide trajectory CSV file and return its path."""
        from .export import result_to_csv

        return result_to_csv(self, result, path)

    def point_trace_to_dict(
        self,
        result: KinematicResult | DynamicsResult,
        body: int | BodyHandle,
        point: ArrayLike2 = (0.0, 0.0),
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-ready trace for a body-local point."""
        from .export import point_trace_to_dict

        return point_trace_to_dict(self, result, body, point, name=name, metadata=metadata)

    def point_trace_to_json(
        self,
        result: KinematicResult | DynamicsResult,
        body: int | BodyHandle,
        point: ArrayLike2,
        path: str | Path,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        indent: int = 2,
    ) -> Path:
        """Write a body-local point trace as JSON."""
        from .export import point_trace_to_json

        return point_trace_to_json(
            self,
            result,
            body,
            point,
            path,
            name=name,
            metadata=metadata,
            indent=indent,
        )

    def point_trace_to_csv(
        self,
        result: KinematicResult | DynamicsResult,
        body: int | BodyHandle,
        point: ArrayLike2,
        path: str | Path,
        *,
        name: str | None = None,
    ) -> Path:
        """Write a body-local point trace as an SI-unit CSV."""
        from .export import point_trace_to_csv

        return point_trace_to_csv(self, result, body, point, path, name=name)

    def solve_kinematics(
        self,
        t: np.ndarray,
        q0: np.ndarray | None = None,
        verbose: bool = False,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> KinematicResult:
        """Run position, velocity, and acceleration solves over ``t``."""
        self._ensure_nonempty_model()
        t = _as_time_array(t)
        _check_steps(t, max_steps)
        q_initial = self.solve_position(q0, float(t[0]))
        q, v, a = run_kinematic_simulation(self.model, q_initial, t, verbose=verbose)
        return KinematicResult(t=t, q=q, v=v, a=a)

    def simulate(
        self,
        t: np.ndarray,
        q0: np.ndarray,
        v0: np.ndarray | None = None,
        allow_underconstrained: bool = False,
        velocity_tol: float = 1e-8,
        **kwargs,
    ) -> DynamicsResult:
        """Run forward dynamics with the scipy-based constrained integrator.

        Registered springs and loads always apply; ``springs`` and
        ``Q_user_fn`` passed here are added on top. ``max_steps`` bounds the
        number of output samples and ``max_evaluations`` bounds right-hand-side
        evaluations; both fail with ``code="resource_limit"`` before or during
        the solve. The result carries Lagrange multipliers at every sample.
        """
        self._ensure_nonempty_model()
        from ..errors import MechanismSolveError

        max_steps = int(kwargs.pop("max_steps", DEFAULT_MAX_STEPS))
        max_evaluations = kwargs.pop("max_evaluations", None)
        call_Q_user_fn = kwargs.pop("Q_user_fn", None)
        springs, Q_user_fn = self._force_inputs(kwargs.pop("springs", None), call_Q_user_fn)
        t = _as_time_array(t)
        _check_steps(t, max_steps)
        if t.size < 2:
            raise ValueError("dynamic integration requires at least two time values")
        velocity_tol = _as_finite_scalar(velocity_tol, "velocity_tol")
        if velocity_tol <= 0.0:
            raise ValueError("velocity_tol must be positive")
        ensure_centroidal_dynamics(self.model)
        q0 = self.solve_position(
            _as_state(q0, "q0", self.model.ncoord),
            float(t[0]),
            allow_underconstrained=allow_underconstrained,
        )
        if v0 is None:
            v0 = solve_velocity(
                self.model,
                q0,
                float(t[0]),
                allow_underconstrained=allow_underconstrained,
            )
        else:
            v0 = _as_state(v0, "v0", self.model.ncoord)
            velocity_residual = jacobian(self.model, q0, float(t[0])) @ v0 + dt_constraints(
                self.model,
                q0,
                float(t[0]),
            )
            if np.linalg.norm(velocity_residual, ord=np.inf) > velocity_tol:
                raise MechanismSolveError(
                    "Initial velocity violates velocity-level constraints: "
                    f"{np.linalg.norm(velocity_residual, ord=np.inf):.3e} "
                    f"> {velocity_tol:.3e}."
                )
        integration_options = dict(kwargs)
        integration_options.setdefault("velocity_tol", velocity_tol)
        sol = solve_dynamics_scipy(
            self.model,
            q0,
            v0,
            t,
            springs=springs,
            Q_user_fn=Q_user_fn,
            allow_underconstrained=allow_underconstrained,
            max_evaluations=max_evaluations,
            **integration_options,
        )
        if not sol.success:
            raise MechanismSolveError(
                f"Dynamic integration failed: {sol.message}",
                code="integration_failed",
                remediation="Check initial conditions, tolerances, or try method='Radau'.",
            )
        q, v, t_out = extract_dynamics_solution(self.model, sol)
        multipliers = self._sampled_multipliers(
            t_out,
            q,
            v,
            springs,
            Q_user_fn,
            {
                key: integration_options[key]
                for key in ("omega", "alpha", "contacts", "surface_contacts")
                if key in integration_options
            },
        )
        from .. import __version__ as mbsd_version

        provenance = {
            "mbsd_version": mbsd_version,
            "solver": "scipy.solve_ivp",
            "method": integration_options.get("method", "RK45"),
            "rtol": integration_options.get("rtol", 1e-9),
            "atol": integration_options.get("atol", 1e-11),
            "position_tol": integration_options.get("position_tol", 1e-8),
            "velocity_tol": integration_options["velocity_tol"],
            "project_position": integration_options.get("project_position", False),
            "alpha_baumgarte": integration_options.get("alpha_baumgarte", 0.0),
            "beta_baumgarte": integration_options.get("beta_baumgarte", 0.0),
            "allow_underconstrained": allow_underconstrained,
            "max_steps": max_steps,
            "max_evaluations": max_evaluations,
            "portable_forces": all(load.portable for load in self.loads),
            "call_Q_user_fn": call_Q_user_fn is not None,
        }
        return DynamicsResult(
            t=t_out, q=q, v=v, provenance=provenance, multipliers=multipliers
        )

    def _sampled_multipliers(self, t, q, v, springs, Q_user_fn, extra):
        """Re-solve the saddle system at each sample to recover multipliers."""
        multipliers = np.zeros((self.model.nrestr, len(t)))
        if self.model.nrestr == 0:
            return multipliers
        for step, ti in enumerate(t):
            qk, vk = q[:, step], v[:, step]
            Q_user = Q_user_fn(ti, qk, vk) if Q_user_fn is not None else None
            Q_total, M = assemble_system(self.model, qk, vk, ti, springs, Q_user, **extra)
            _, lam = solve_acceleration_with_lagrange(self.model, qk, vk, ti, M, Q_total)
            multipliers[:, step] = lam
        return multipliers

    def inverse_dynamics(
        self,
        t: np.ndarray,
        q0: np.ndarray | None = None,
        *,
        springs: Iterable[Spring] | None = None,
        Q_user_fn: Callable | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> InverseDynamicsResult:
        """Solve a fully driven mechanism and the multipliers that sustain it.

        Every degree of freedom must be prescribed by drives
        (``nrestr == ncoord``). Pass the result to ``joint_loads`` for joint
        reactions and drive efforts.
        """
        self._ensure_nonempty_model()
        from ..errors import MechanismSolveError
        from .. import __version__ as mbsd_version

        t = _as_time_array(t)
        _check_steps(t, max_steps)
        if self.model.nrestr != self.model.ncoord:
            raise MechanismSolveError(
                "Inverse dynamics requires a kinematically driven mechanism: "
                f"{self.model.nrestr} constraints for {self.model.ncoord} coordinates.",
                code="not_kinematically_driven",
                context={"constraints": self.model.nrestr, "coordinates": self.model.ncoord},
                remediation="Add drives (motor or coordinate_drive) until the DOF count is zero.",
            )
        ensure_centroidal_dynamics(self.model)
        springs, Q_user_fn = self._force_inputs(springs, Q_user_fn)
        q_guess = np.zeros(self.model.ncoord) if q0 is None else _as_state(q0, "q0", self.model.ncoord)
        q, v, a, multipliers, _ = solve_inverse_dynamics(
            self.model, q_guess, t, springs=springs, Q_user_fn=Q_user_fn
        )
        provenance = {
            "mbsd_version": mbsd_version,
            "solver": "saddle_point_inverse_dynamics",
            "max_steps": max_steps,
            "portable_forces": all(load.portable for load in self.loads),
        }
        return InverseDynamicsResult(
            t=t, q=q, v=v, a=a, multipliers=multipliers, provenance=provenance
        )

    def joint_loads(self, result: InverseDynamicsResult | DynamicsResult) -> JointLoads:
        """Return named joint reactions, drive efforts, and the frame load.

        Joint forces are world-frame; moments are about the joint's world point
        (the body-j attachment point). Drive effort is the force or torque the
        drive applies to its coordinate. ``frame`` is the total load the
        mechanism puts on ground, with drive actuators reacting against ground.
        """
        from ..errors import MechanismSolveError

        self._validate_result(result)
        multipliers = getattr(result, "multipliers", None)
        if multipliers is None:
            raise MechanismSolveError(
                "Result has no Lagrange multipliers.",
                code="missing_multipliers",
                remediation="Use inverse_dynamics() or simulate(); kinematic results carry no loads.",
            )
        multipliers = np.asarray(multipliers, dtype=float)
        if multipliers.shape != (self.model.nrestr, len(result.t)):
            raise ValueError(
                f"result.multipliers must have shape ({self.model.nrestr}, {len(result.t)})"
            )
        return joint_loads(self.model, result.t, result.q, result.v, multipliers)

    def joint_loads_to_dict(
        self,
        result: InverseDynamicsResult | DynamicsResult,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a versioned JSON-ready joint-load history payload."""
        from .export import joint_loads_to_dict

        return joint_loads_to_dict(self, result, metadata=metadata)

    def joint_loads_to_json(
        self,
        result: InverseDynamicsResult | DynamicsResult,
        path: str | Path,
        *,
        indent: int = 2,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write the joint-load history payload as JSON and return its path."""
        from .export import joint_loads_to_json

        return joint_loads_to_json(self, result, path, indent=indent, metadata=metadata)

    def capabilities(self) -> dict[str, Any]:
        """Return the static planar capability manifest."""
        return planar_capabilities()

    def validate(
        self,
        q0: np.ndarray | None = None,
        t: np.ndarray | float = 0.0,
    ) -> ValidationReport:
        """Check the model before solving and report structured issues.

        ``t`` may be a time grid; tabulated drives and loads are then checked
        against it. Nothing is integrated.
        """
        issues: list[Issue] = []
        classification = None
        dof = None
        if not self.model.bodies:
            issues.append(
                Issue("empty_model", "error", "Mechanism has no bodies.", "Add ground() and body().")
            )
            return ValidationReport(False, True, None, None, tuple(issues))

        times = np.atleast_1d(np.asarray(t, dtype=float))
        t0, t1 = float(times[0]), float(times[-1])
        for index, body in enumerate(self.model.bodies):
            if np.linalg.norm(body.rG, ord=np.inf) > 0.0:
                issues.append(
                    Issue(
                        "offset_center_of_mass",
                        "error",
                        f"Body {body.name!r} has an offset center of mass; planar dynamics "
                        "and inverse dynamics reject it (kinematics is unaffected).",
                        "Place the body reference point at its center of mass.",
                    )
                )
        try:
            q = self.solve_position(
                q0, t0, allow_underconstrained=self.model.nrestr < self.model.ncoord
            )
            diagnostics = self.model_diagnostics(q, t0)
            classification = diagnostics.classification
            dof = diagnostics.degrees_of_freedom
            if diagnostics.rank_deficient:
                issues.append(
                    Issue(
                        "rank_deficient",
                        "error",
                        f"Constraint Jacobian rank {diagnostics.jacobian_rank} is below "
                        f"{min(self.model.nrestr, self.model.ncoord)}; the model has "
                        "redundant or singular constraints at this configuration.",
                        "Remove redundant joints or start from a non-singular q0.",
                    )
                )
            if diagnostics.overconstrained:
                issues.append(
                    Issue(
                        "overconstrained",
                        "error",
                        f"{self.model.nrestr} constraints for {self.model.ncoord} coordinates.",
                        "Remove redundant joints or drives.",
                    )
                )
            elif diagnostics.underconstrained:
                issues.append(
                    Issue(
                        "underconstrained",
                        "info",
                        f"{dof} free degree(s) of freedom: suitable for simulate(), "
                        "not for solve_kinematics() or inverse_dynamics().",
                        "Add drives to prescribe the free coordinates if needed.",
                    )
                )
        except Exception as exc:  # noqa: BLE001 - reported as structured data
            issues.append(
                Issue(
                    "position_solve_failed",
                    "error",
                    f"Assembly failed at t={t0}: {exc}",
                    "Provide a q0 closer to an assembled configuration.",
                )
            )

        for drive in self.model.user_constraints:
            metadata = drive.metadata or {}
            label = drive.name or metadata.get("kind", "user_constraint")
            if metadata.get("kind") not in {"motor", "coordinate_drive"}:
                issues.append(
                    Issue(
                        "opaque_constraint",
                        "warning",
                        f"User constraint {label!r} is raw Python and is not portable.",
                        "Use pin, slider, motor, or coordinate_drive(law=...).",
                    )
                )
            elif not metadata.get("portable", False):
                issues.append(
                    Issue(
                        "opaque_drive",
                        "warning",
                        f"Drive {label!r} uses Python callables and is not portable.",
                        "Pass law=ConstantSpeed/Polynomial/Tabulated instead.",
                    )
                )
            domain = getattr(getattr(drive, "law", None), "domain", None)
            if domain is not None and (t0 < domain[0] or t1 > domain[1]):
                issues.append(
                    Issue(
                        "time_outside_table",
                        "error",
                        f"Drive {label!r} is tabulated on {list(domain)} but t spans [{t0}, {t1}].",
                        "Extend the table or shorten the time grid.",
                    )
                )
        for load in self.loads:
            if not load.portable:
                issues.append(
                    Issue(
                        "opaque_load",
                        "warning",
                        f"Load {load.name!r} is a Python callback and is not portable.",
                        "Use constant or tabulated load descriptors where possible.",
                    )
                )
            times_attr = getattr(load, "times", None)
            if times_attr is not None and (t0 < times_attr[0] or t1 > times_attr[-1]):
                issues.append(
                    Issue(
                        "time_outside_table",
                        "error",
                        f"Load {load.name!r} is tabulated on [{times_attr[0]}, {times_attr[-1]}] "
                        f"but t spans [{t0}, {t1}].",
                        "Extend the table or shorten the time grid.",
                    )
                )
        for kind, items in (
            ("cam", self.model.leva_joints),
            ("point_on_line", self.model.punto_linea_joints),
            ("gear", self.model.gear_joints),
        ):
            if items:
                issues.append(
                    Issue(
                        "low_level_joint",
                        "warning",
                        f"{len(items)} low-level {kind} joint(s) are not part of the public "
                        "builder and are not exported.",
                        None,
                    )
                )
        if len(times) > DEFAULT_MAX_STEPS:
            issues.append(
                Issue(
                    "resource_limit",
                    "error",
                    f"{len(times)} time samples exceed the default max_steps={DEFAULT_MAX_STEPS}.",
                    "Use fewer samples or pass max_steps explicitly.",
                )
            )
        portable = not any(
            issue.code in {"opaque_constraint", "opaque_drive", "opaque_load", "low_level_joint"}
            for issue in issues
        )
        ok = not any(issue.severity == "error" for issue in issues)
        return ValidationReport(ok, portable, classification, dof, tuple(issues))

    def check(
        self,
        result: KinematicResult | DynamicsResult | InverseDynamicsResult,
        *,
        position_tol: float = 1e-8,
        velocity_tol: float = 1e-6,
        energy_tol: float = 1e-6,
        balance_tol: float = 1e-8,
        springs: Iterable[Spring] | None = None,
        Q_user_fn: Callable | None = None,
    ) -> CheckReport:
        """Return a pass/fail verdict on a solved result.

        Checks finiteness and position/velocity constraint residuals for every
        result. Forward dynamics adds relative energy drift when the model is
        conservative (otherwise the check is skipped with a note). Inverse
        dynamics adds the equation-of-motion balance
        ``|M a + C_q.T λ - Q|``. Thresholds are part of the report. Pass the
        same per-call ``springs`` / ``Q_user_fn`` used for the solve.
        """
        self._validate_result(result)
        checks: list[Check] = []
        finite = _result_arrays_are_finite(result)
        checks.append(Check("finite", None, None, finite))
        position = self.max_constraint_residual(result)
        checks.append(Check("position_residual", position, position_tol, position <= position_tol))
        velocity = 0.0
        for step, ti in enumerate(result.t):
            residual = jacobian(self.model, result.q[:, step], float(ti)) @ result.v[
                :, step
            ] + dt_constraints(self.model, result.q[:, step], float(ti))
            if residual.size:
                velocity = max(velocity, float(np.max(np.abs(residual))))
        checks.append(Check("velocity_residual", velocity, velocity_tol, velocity <= velocity_tol))

        if isinstance(result, InverseDynamicsResult):
            result_type = "inverse_dynamic"
            springs, Q_user_fn = self._force_inputs(springs, Q_user_fn)
            worst, scale = 0.0, 1.0
            for step, ti in enumerate(result.t):
                qk, vk = result.q[:, step], result.v[:, step]
                Q_user = Q_user_fn(ti, qk, vk) if Q_user_fn is not None else None
                Q, M = assemble_system(self.model, qk, vk, ti, springs, Q_user)
                Cq = jacobian(self.model, qk, float(ti))
                lhs = M @ result.a[:, step] + Cq.T @ result.multipliers[:, step]
                worst = max(worst, float(np.max(np.abs(lhs - Q))))
                scale = max(scale, float(np.max(np.abs(Q))), float(np.max(np.abs(lhs))))
            relative = worst / scale
            checks.append(
                Check("equation_of_motion_balance", relative, balance_tol, relative <= balance_tol)
            )
        elif isinstance(result, DynamicsResult):
            result_type = "dynamic"
            all_springs = list(self.springs) + list(springs or [])
            reason = self._nonconservative_reason(all_springs, Q_user_fn is not None)
            if reason is None and result.provenance.get("call_Q_user_fn") and Q_user_fn is None:
                reason = "skipped: the solve used a per-call Q_user_fn; pass it to check()"
            if reason:
                checks.append(Check("energy_drift", None, energy_tol, None, reason))
            else:
                energies = []
                for step in range(len(result.t)):
                    qk, vk = result.q[:, step], result.v[:, step]
                    M = assemble_system(self.model, qk, vk, result.t[step], all_springs)[1]
                    energies.append(
                        kinetic_energy(M, vk)
                        + potential_energy(self.model, qk, springs=all_springs)
                    )
                energies = np.asarray(energies)
                scale = max(1.0, float(np.max(np.abs(energies))))
                drift = float(np.max(np.abs(energies - energies[0]))) / scale
                checks.append(Check("energy_drift", drift, energy_tol, drift <= energy_tol))
        else:
            result_type = "kinematic"

        passed = all(check.passed is not False for check in checks)
        return CheckReport(passed, result_type, tuple(checks))

    def _nonconservative_reason(self, springs, has_user_force: bool) -> str | None:
        if self.model.user_constraints:
            return "skipped: drives or user constraints do work on the system"
        if self.loads or has_user_force:
            return "skipped: applied loads are not treated as potentials"
        if any(spring.c != 0.0 for spring in springs):
            return "skipped: damped springs dissipate energy"
        return None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanarMechanism":
        """Rebuild a mechanism from a ``to_dict`` payload (schema v1)."""
        from .export import mechanism_from_dict

        return mechanism_from_dict(payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "PlanarMechanism":
        """Rebuild a mechanism from a ``to_json`` file."""
        from .export import mechanism_from_json

        return mechanism_from_json(path)

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
        if not hasattr(result, "v"):
            raise ValueError("result.v is required")
        v = np.asarray(result.v, dtype=float)
        if v.shape != (self.model.ncoord, len(t)):
            raise ValueError(
                f"result.v must have shape ({self.model.ncoord}, {len(t)}), got {v.shape}"
            )
        if hasattr(result, "a"):
            a = np.asarray(result.a, dtype=float)
            if a.shape != (self.model.ncoord, len(t)):
                raise ValueError(
                    f"result.a must have shape ({self.model.ncoord}, {len(t)}), got {a.shape}"
                )
        if not np.all(np.isfinite(t)):
            raise ValueError("result.t must contain only finite values")
        if not np.all(np.isfinite(q)):
            raise ValueError("result.q must contain only finite values")
        if not np.all(np.isfinite(v)):
            raise ValueError("result.v must contain only finite values")
        if hasattr(result, "a") and not np.all(np.isfinite(result.a)):
            raise ValueError("result.a must contain only finite values")
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


def _result_arrays_are_finite(result: KinematicResult | DynamicsResult) -> bool:
    arrays = [result.t, result.q, result.v]
    if hasattr(result, "a"):
        arrays.append(result.a)
    return bool(all(np.all(np.isfinite(array)) for array in arrays))


def _as_finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class _CallableLaw:
    """Adapter giving three Python callables the motion-law interface."""

    def __init__(self, value, velocity, acceleration):
        self.value = value
        self.velocity = velocity
        self.acceleration = acceleration


def _as_tuple(value: Iterable[float], name: str, length: int) -> tuple[float, ...]:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"{name} must be a finite vector with shape ({length},)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return tuple(float(x) for x in arr)


def _as_samples(value: Iterable[float], name: str) -> tuple[float, ...]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least two samples")
    if not np.all(np.isfinite(arr)) or not np.all(np.diff(arr) > 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return tuple(float(x) for x in arr)


def _check_steps(t: np.ndarray, max_steps: int) -> None:
    from ..errors import MechanismSolveError

    if len(t) > max_steps:
        raise MechanismSolveError(
            f"{len(t)} time samples exceed max_steps={max_steps}.",
            code="resource_limit",
            context={"steps": len(t), "max_steps": max_steps},
            remediation="Use fewer samples or raise max_steps deliberately.",
        )


def planar_capabilities() -> dict[str, Any]:
    """Static description of what the planar builder supports."""
    return {
        "dimension": 2,
        "status": "stable_candidate",
        "coordinates": ["x", "y", "theta"],
        "joints": ["pin", "slider"],
        "drives": {
            "kinds": ["motor", "coordinate_drive"],
            "portable_laws": ["constant_speed", "polynomial", "tabulated"],
            "callables": "non_portable",
        },
        "forces": {
            "gravity": True,
            "portable": ["spring_damper", "constant_force", "constant_torque",
                         "tabulated_force", "tabulated_torque"],
            "callback": "non_portable",
        },
        "solvers": {
            "kinematics": True,
            "forward_dynamics": True,
            "inverse_dynamics": True,
            "joint_reactions": True,
        },
        "guardrails": ["validate", "check"],
        "exports": ["mechanism_json", "result_json", "result_csv", "point_trace",
                    "joint_loads_json"],
        "imports": ["mechanism_json"],
        "limits": {
            "offset_center_of_mass_dynamics": False,
            "contact": "callback_only",
            "flexible_bodies": False,
            "default_max_steps": DEFAULT_MAX_STEPS,
        },
    }
