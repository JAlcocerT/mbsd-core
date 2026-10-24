"""Restricted fixed-orientation spatial translational dynamics.

This module deliberately evolves only world-expressed translations.  Body
orientations are constant, torques are unsupported, and every public payload
names that capability boundary explicitly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from ..errors import MechanismSolveError
from .vocabulary import FixedJoint3D, Pose3D

if TYPE_CHECKING:
    from .builder import SpatialBodyHandle, SpatialMechanism


DYNAMIC_RESULT_SCHEMA = "mbsd.spatial.fixed_orientation_dynamic_result"
DYNAMIC_RESULT_SCHEMA_VERSION = 1
ForceLaw = Callable[[float, tuple[Pose3D, ...], np.ndarray], Iterable[float]]


def _readonly(value: Any) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class SpatialForce:
    """World-expressed force applied at a body's coincident reference point/COM."""

    body: int
    vector: np.ndarray | ForceLaw
    name: str = "force"
    frame: str = "world"
    application_point: str = "center_of_mass"

    def __post_init__(self) -> None:
        if isinstance(self.body, bool) or int(self.body) < 0:
            raise ValueError("force body must be a non-negative integer")
        object.__setattr__(self, "body", int(self.body))
        if not self.name:
            raise ValueError("force name must be non-empty")
        if self.frame != "world":
            raise ValueError("fixed-orientation dynamics supports only world-frame forces")
        if self.application_point != "center_of_mass":
            raise ValueError("forces must be applied at the center of mass")
        if not callable(self.vector):
            object.__setattr__(self, "vector", _vector3(self.vector, "force vector"))

    def evaluate(
        self, t: float, poses: tuple[Pose3D, ...], velocities: np.ndarray
    ) -> np.ndarray:
        value = self.vector(t, poses, velocities) if callable(self.vector) else self.vector
        return _vector3(value, f"force {self.name!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "body": self.body,
            "frame": self.frame,
            "application_point": self.application_point,
            "units": "N",
            "kind": "runtime_law" if callable(self.vector) else "constant",
            "vector": None if callable(self.vector) else self.vector.tolist(),
        }


@dataclass(frozen=True)
class SpatialDynamicsResult:
    """Result of fixed-orientation, translational-only spatial dynamics."""

    t: np.ndarray
    poses: tuple[tuple[Pose3D, ...], ...]
    linear_velocity: np.ndarray
    linear_acceleration: np.ndarray
    applied_force: np.ndarray
    multipliers: np.ndarray
    reaction_force: np.ndarray
    position_residuals: np.ndarray
    velocity_residuals: np.ndarray
    acceleration_residuals: np.ndarray
    success: bool
    message: str
    model_id: str
    excitation_id: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "t",
            "linear_velocity",
            "linear_acceleration",
            "applied_force",
            "multipliers",
            "reaction_force",
            "position_residuals",
            "velocity_residuals",
            "acceleration_residuals",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    def body_poses(self, body: int | SpatialBodyHandle) -> tuple[Pose3D, ...]:
        index = body.index if hasattr(body, "index") else int(body)
        if index < 0 or not self.poses or index >= len(self.poses[0]):
            raise IndexError(f"body index {index} is out of range")
        return tuple(step[index] for step in self.poses)

    def body_linear_velocity(self, body: int | SpatialBodyHandle) -> np.ndarray:
        index = body.index if hasattr(body, "index") else int(body)
        return self.linear_velocity[:, index, :]


@dataclass(frozen=True)
class SpatialDynamicsDiagnostics:
    steps: int
    max_position_residual_m: float
    max_velocity_residual_m_per_s: float
    max_acceleration_residual_m_per_s2: float
    max_force_balance_residual_n: float
    relative_energy_change: float
    relative_momentum_change: float
    finite: bool
    success: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def fixed_orientation_capabilities() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "solver_kind": "fixed_orientation_translational_dynamics",
            "translation_dynamics": True,
            "orientation_dynamics": False,
            "torque_inputs": False,
            "force_frame": "world",
            "velocity_frame": "world",
            "fixed_orientations": True,
            "joint_types": ("spherical", "fixed"),
        }
    )


def simulate_fixed_orientation(
    mechanism: SpatialMechanism,
    t: np.ndarray,
    *,
    poses0: Iterable[Pose3D] | None = None,
    linear_velocity0: np.ndarray | None = None,
    forces: Iterable[SpatialForce] = (),
    method: str = "DOP853",
    rtol: float = 1e-9,
    atol: float = 1e-11,
    constraint_tol: float = 1e-8,
    project_samples: bool = True,
) -> SpatialDynamicsResult:
    """Integrate constrained translations while holding all rotations fixed."""
    times = _time_array(t)
    if len(times) < 2:
        raise ValueError("t must contain at least two values for dynamics")
    for name, value in (("rtol", rtol), ("atol", atol), ("constraint_tol", constraint_tol)):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite scalar")
    if method not in {"RK45", "DOP853", "Radau"}:
        raise ValueError("method must be one of: RK45, DOP853, Radau")
    if mechanism.nbody == 0:
        raise ValueError("spatial mechanism must contain at least one body")

    force_values = tuple(forces)
    for index, force in enumerate(force_values):
        if not isinstance(force, SpatialForce):
            raise TypeError(f"forces[{index}] must be a SpatialForce")
        if force.body >= mechanism.nbody:
            raise IndexError(f"force body index {force.body} is out of range")

    if any(np.linalg.norm(body.center_of_mass, ord=np.inf) > 0.0 for body in mechanism._bodies):
        raise MechanismSolveError(
            "Fixed-orientation dynamics requires every body reference point to "
            "coincide with its center of mass."
        )
    if poses0 is None:
        template = mechanism._initial_poses()
        if mechanism._ground is not None or mechanism._joints or mechanism._drives:
            template = mechanism.solve_position(template, float(times[0]), tol=constraint_tol)
    else:
        template = mechanism._validate_poses(poses0)
    _validate_fixed_orientation_configuration(mechanism, template, constraint_tol)
    q0 = _translations(template)
    jacobian = _constraint_jacobian(mechanism)
    constraint_count = jacobian.shape[0]
    rank = int(np.linalg.matrix_rank(jacobian)) if jacobian.size else 0
    if rank < constraint_count:
        raise MechanismSolveError(
            "Fixed-orientation dynamics constraints are redundant or rank-deficient: "
            f"rank {rank} for {constraint_count} constraint rows."
        )
    if constraint_count > 3 * mechanism.nbody:
        raise MechanismSolveError("Fixed-orientation dynamics model is overconstrained.")

    position0 = _constraint_residual(mechanism, q0, template, float(times[0]))
    if _norm(position0) > constraint_tol:
        raise MechanismSolveError(
            "Initial position violates fixed-orientation translation constraints: "
            f"{_norm(position0):.3e} > {constraint_tol:.3e}."
        )
    if linear_velocity0 is None:
        v0 = _minimum_norm_velocity(mechanism, jacobian, float(times[0]))
    else:
        velocity = np.asarray(linear_velocity0, dtype=float)
        if velocity.shape == (mechanism.nbody, 3):
            velocity = velocity.reshape(-1)
        v0 = _state(velocity, 3 * mechanism.nbody, "linear_velocity0")
    velocity0 = jacobian @ v0 + _constraint_time_derivative(mechanism, float(times[0]))
    if _norm(velocity0) > constraint_tol:
        raise MechanismSolveError(
            "Initial velocity violates fixed-orientation translation constraints: "
            f"{_norm(velocity0):.3e} > {constraint_tol:.3e}."
        )

    mass = np.repeat([body.mass for body in mechanism._bodies], 3)
    mass_matrix = np.diag(mass)

    def acceleration(ti: float, q: np.ndarray, v: np.ndarray):
        poses = _poses(q, template)
        applied = _force_vector(force_values, ti, poses, v.reshape(mechanism.nbody, 3))
        gamma = _constraint_acceleration_rhs(mechanism, ti)
        if constraint_count == 0:
            return applied / mass, np.zeros(0), applied
        system = np.block(
            [
                [mass_matrix, jacobian.T],
                [jacobian, np.zeros((constraint_count, constraint_count))],
            ]
        )
        rhs = np.concatenate((applied, gamma))
        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError as error:
            raise MechanismSolveError(
                "Fixed-orientation dynamics acceleration solve is singular."
            ) from error
        return solution[: 3 * mechanism.nbody], solution[3 * mechanism.nbody :], applied

    def rhs(ti: float, state: np.ndarray) -> np.ndarray:
        q = state[: 3 * mechanism.nbody]
        v = state[3 * mechanism.nbody :]
        a, _, _ = acceleration(ti, q, v)
        return np.concatenate((v, a))

    solution = solve_ivp(
        rhs,
        (float(times[0]), float(times[-1])),
        np.concatenate((q0, v0)),
        t_eval=times,
        method=method,
        rtol=float(rtol),
        atol=float(atol),
    )
    if not solution.success:
        raise MechanismSolveError(f"Fixed-orientation dynamics failed: {solution.message}")
    q_history = solution.y[: 3 * mechanism.nbody].T.copy()
    v_history = solution.y[3 * mechanism.nbody :].T.copy()
    if project_samples and constraint_count:
        pinv = np.linalg.pinv(jacobian)
        for index, ti in enumerate(times):
            q_history[index] -= pinv @ _constraint_residual(
                mechanism, q_history[index], template, float(ti)
            )
            velocity_error = (
                jacobian @ v_history[index]
                + _constraint_time_derivative(mechanism, float(ti))
            )
            v_history[index] -= pinv @ velocity_error

    pose_history = tuple(_poses(q, template) for q in q_history)
    accelerations = np.zeros_like(v_history)
    applied_forces = np.zeros_like(v_history)
    multipliers = np.zeros((len(times), constraint_count))
    reactions = np.zeros_like(v_history)
    position_residuals = np.zeros((constraint_count, len(times)))
    velocity_residuals = np.zeros_like(position_residuals)
    acceleration_residuals = np.zeros_like(position_residuals)
    for index, ti in enumerate(times):
        a, lam, applied = acceleration(float(ti), q_history[index], v_history[index])
        accelerations[index] = a
        applied_forces[index] = applied
        multipliers[index] = lam
        reactions[index] = (-jacobian.T @ lam) if constraint_count else 0.0
        position_residuals[:, index] = _constraint_residual(
            mechanism, q_history[index], template, float(ti)
        )
        velocity_residuals[:, index] = (
            jacobian @ v_history[index]
            + _constraint_time_derivative(mechanism, float(ti))
        )
        acceleration_residuals[:, index] = (
            jacobian @ a - _constraint_acceleration_rhs(mechanism, float(ti))
        )

    excitation_id = _excitation_id(mechanism, times, force_values, applied_forces)
    provenance = MappingProxyType(
        {
            "mbsd_version": _package_version(),
            "solver": "scipy.solve_ivp",
            "method": method,
            "rtol": float(rtol),
            "atol": float(atol),
            "constraint_tolerance": float(constraint_tol),
            "sample_projection": bool(project_samples),
            "multiplier_equation": "M*a + J.T*lambda = applied_force",
            "physical_reaction": "-J.T*lambda",
            "drive_acceleration": "supplied_or_adaptive_velocity_derivative",
            "force_descriptors": tuple(
                MappingProxyType(force.as_dict()) for force in force_values
            ),
        }
    )
    return SpatialDynamicsResult(
        t=times,
        poses=pose_history,
        linear_velocity=v_history.reshape(len(times), mechanism.nbody, 3),
        linear_acceleration=accelerations.reshape(len(times), mechanism.nbody, 3),
        applied_force=applied_forces.reshape(len(times), mechanism.nbody, 3),
        multipliers=multipliers,
        reaction_force=reactions.reshape(len(times), mechanism.nbody, 3),
        position_residuals=position_residuals,
        velocity_residuals=velocity_residuals,
        acceleration_residuals=acceleration_residuals,
        success=True,
        message="converged",
        model_id=mechanism.model_id(),
        excitation_id=excitation_id,
        provenance=provenance,
        capabilities=fixed_orientation_capabilities(),
    )


def dynamics_diagnostics(
    mechanism: SpatialMechanism, result: SpatialDynamicsResult
) -> SpatialDynamicsDiagnostics:
    _validate_result(mechanism, result)
    masses = np.asarray([body.mass for body in mechanism._bodies])
    kinetic = 0.5 * np.sum(
        masses[None, :, None] * result.linear_velocity**2, axis=(1, 2)
    )
    momentum = np.sum(masses[None, :, None] * result.linear_velocity, axis=1)
    energy_scale = max(1.0, abs(float(kinetic[0])))
    momentum_scale = max(1.0, float(np.linalg.norm(momentum[0])))
    force_balance = (
        masses[None, :, None] * result.linear_acceleration
        - result.applied_force
        - result.reaction_force
    )
    return SpatialDynamicsDiagnostics(
        steps=len(result.t),
        max_position_residual_m=_norm(result.position_residuals),
        max_velocity_residual_m_per_s=_norm(result.velocity_residuals),
        max_acceleration_residual_m_per_s2=_norm(result.acceleration_residuals),
        max_force_balance_residual_n=_norm(force_balance),
        relative_energy_change=float(np.max(np.abs(kinetic - kinetic[0])) / energy_scale),
        relative_momentum_change=float(
            np.max(np.linalg.norm(momentum - momentum[0], axis=1)) / momentum_scale
        ),
        finite=True,
        success=bool(result.success),
    )


def dynamic_result_to_dict(
    mechanism: SpatialMechanism, result: SpatialDynamicsResult
) -> dict[str, Any]:
    diagnostics = dynamics_diagnostics(mechanism, result)
    return {
        "schema": DYNAMIC_RESULT_SCHEMA,
        "schema_version": DYNAMIC_RESULT_SCHEMA_VERSION,
        "mbsd_version": _package_version(),
        "status": "experimental",
        "dimension": 3,
        "solver_kind": "fixed_orientation_translational_dynamics",
        "model_id": result.model_id,
        "excitation_id": result.excitation_id,
        "units": {
            "length": "m",
            "time": "s",
            "linear_velocity": "m/s",
            "linear_acceleration": "m/s^2",
            "force": "N",
            "multiplier": "N",
        },
        "conventions": {
            "world_frame": "right_handed_xyz",
            "rotation": "active_body_to_world_constant",
            "array_layout": "time_body_component",
            "force_frame": "world",
            "reaction_force": "physical_-J.T_lambda",
        },
        "capabilities": _json_ready(result.capabilities),
        "solver_status": {"success": result.success, "message": result.message},
        "provenance": _json_ready(result.provenance),
        "diagnostics": diagnostics.as_dict(),
        "time": result.t.tolist(),
        "constraint_count": result.position_residuals.shape[0],
        "position_residuals": result.position_residuals.tolist(),
        "velocity_residuals": result.velocity_residuals.tolist(),
        "acceleration_residuals": result.acceleration_residuals.tolist(),
        "multipliers": result.multipliers.tolist(),
        "bodies": [
            {
                "index": body_index,
                "name": body.name,
                "poses": [step[body_index].as_dict() for step in result.poses],
                "linear_velocity": result.linear_velocity[:, body_index].tolist(),
                "linear_acceleration": result.linear_acceleration[:, body_index].tolist(),
                "applied_force": result.applied_force[:, body_index].tolist(),
                "reaction_force": result.reaction_force[:, body_index].tolist(),
            }
            for body_index, body in enumerate(mechanism._bodies)
        ],
    }


def dynamic_result_to_json(
    mechanism: SpatialMechanism,
    result: SpatialDynamicsResult,
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    payload = dynamic_result_to_dict(mechanism, result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, allow_nan=False) + "\n", encoding="utf-8")
    return path


def validate_spatial_dynamic_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the portable fixed-orientation dynamics result contract."""
    if not isinstance(payload, Mapping):
        raise TypeError("spatial dynamic result payload must be a mapping")
    if payload.get("schema") != DYNAMIC_RESULT_SCHEMA:
        raise ValueError(f"schema must be {DYNAMIC_RESULT_SCHEMA!r}")
    if payload.get("schema_version") != DYNAMIC_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported spatial dynamic result schema version")
    if payload.get("solver_kind") != "fixed_orientation_translational_dynamics":
        raise ValueError("solver_kind must identify fixed-orientation translational dynamics")
    if payload.get("status") != "experimental" or payload.get("dimension") != 3:
        raise ValueError("dynamic results require experimental status and dimension 3")
    for name in ("model_id", "excitation_id", "mbsd_version"):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise ValueError(f"{name} must be a non-empty string")
    expected_units = {
        "length": "m",
        "time": "s",
        "linear_velocity": "m/s",
        "linear_acceleration": "m/s^2",
        "force": "N",
        "multiplier": "N",
    }
    if payload.get("units") != expected_units:
        raise ValueError("units must match the fixed-orientation dynamics contract")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, Mapping):
        raise ValueError("capabilities must be a mapping")
    if capabilities.get("orientation_dynamics") is not False or capabilities.get("torque_inputs") is not False:
        raise ValueError("capabilities must reject orientation dynamics and torque inputs")
    solver_status = payload.get("solver_status")
    if (
        not isinstance(solver_status, Mapping)
        or not isinstance(solver_status.get("success"), bool)
        or not isinstance(solver_status.get("message"), str)
    ):
        raise ValueError("solver_status must contain boolean success and string message")
    if not isinstance(payload.get("provenance"), Mapping):
        raise ValueError("provenance must be a mapping")
    times = np.asarray(payload.get("time"), dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)):
        raise ValueError("time must be a finite one-dimensional array with at least two values")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("time must be strictly increasing")
    constraint_count = payload.get("constraint_count")
    if isinstance(constraint_count, bool) or not isinstance(constraint_count, int) or constraint_count < 0:
        raise ValueError("constraint_count must be a non-negative integer")
    bodies = payload.get("bodies")
    if not isinstance(bodies, list) or not bodies:
        raise ValueError("bodies must be a non-empty list")
    for index, body in enumerate(bodies):
        if not isinstance(body, Mapping) or body.get("index") != index:
            raise ValueError(f"bodies[{index}] has invalid identity fields")
        if len(body.get("poses", ())) != len(times):
            raise ValueError(f"bodies[{index}].poses must match time length")
        for pose_index, pose in enumerate(body["poses"]):
            if not isinstance(pose, Mapping):
                raise ValueError(f"bodies[{index}].poses[{pose_index}] must be a mapping")
            translation = np.asarray(pose.get("translation"), dtype=float)
            rotation = pose.get("rotation")
            if translation.shape != (3,) or not np.all(np.isfinite(translation)):
                raise ValueError(f"bodies[{index}].poses[{pose_index}] has invalid translation")
            if not isinstance(rotation, Mapping):
                raise ValueError(f"bodies[{index}].poses[{pose_index}] has invalid rotation")
            try:
                quaternion = np.asarray(
                    [rotation[name] for name in ("w", "x", "y", "z")], dtype=float
                )
            except KeyError as error:
                raise ValueError(
                    f"bodies[{index}].poses[{pose_index}] rotation is incomplete"
                ) from error
            if not np.all(np.isfinite(quaternion)) or not np.isclose(
                np.linalg.norm(quaternion), 1.0, atol=1e-10, rtol=1e-10
            ):
                raise ValueError(f"bodies[{index}].poses[{pose_index}] has invalid quaternion")
        for name in ("linear_velocity", "linear_acceleration", "applied_force", "reaction_force"):
            values = np.asarray(body.get(name), dtype=float)
            if values.shape != (len(times), 3) or not np.all(np.isfinite(values)):
                raise ValueError(f"bodies[{index}].{name} must be finite with time_by_3 shape")
    for name in ("position_residuals", "velocity_residuals", "acceleration_residuals"):
        raw_values = payload.get(name)
        if constraint_count == 0 and raw_values == []:
            values = np.empty((0, len(times)))
        else:
            values = np.asarray(raw_values, dtype=float)
        if values.shape != (constraint_count, len(times)) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite with constraint_by_time shape")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("diagnostics must be a mapping")
    expected_maxima = {
        "max_position_residual_m": _norm(
            np.asarray(payload["position_residuals"], dtype=float)
        ),
        "max_velocity_residual_m_per_s": _norm(
            np.asarray(payload["velocity_residuals"], dtype=float)
        ),
        "max_acceleration_residual_m_per_s2": _norm(
            np.asarray(payload["acceleration_residuals"], dtype=float)
        ),
    }
    for name, expected in expected_maxima.items():
        if not np.isclose(diagnostics.get(name, np.nan), expected, atol=1e-15, rtol=1e-12):
            raise ValueError(f"diagnostics.{name} does not match residual arrays")
    if diagnostics.get("finite") is not True or not isinstance(diagnostics.get("success"), bool):
        raise ValueError("diagnostics finite/success fields are invalid")
    if diagnostics["success"] is not solver_status["success"]:
        raise ValueError("diagnostics.success must match solver_status.success")
    multipliers = np.asarray(payload.get("multipliers"), dtype=float)
    expected_multiplier_shape = (len(times), constraint_count)
    if constraint_count == 0 and payload.get("multipliers") == []:
        multipliers = np.empty(expected_multiplier_shape)
    if multipliers.shape != expected_multiplier_shape or not np.all(np.isfinite(multipliers)):
        raise ValueError("multipliers must be finite with time_by_constraint shape")
    try:
        return json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("spatial dynamic result must contain finite JSON values") from error


def load_spatial_dynamic_result_payload(path: str | Path) -> dict[str, Any]:
    return validate_spatial_dynamic_result_payload(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _validate_result(mechanism: SpatialMechanism, result: SpatialDynamicsResult) -> None:
    if not isinstance(result, SpatialDynamicsResult):
        raise TypeError("result must be a SpatialDynamicsResult")
    if result.model_id != mechanism.model_id():
        raise ValueError("result model_id does not match this mechanism")
    expected = (len(result.t), mechanism.nbody, 3)
    if len(result.poses) != len(result.t) or any(
        len(step) != mechanism.nbody for step in result.poses
    ):
        raise ValueError("result pose history must match time and body counts")
    for name in ("linear_velocity", "linear_acceleration", "applied_force", "reaction_force"):
        if getattr(result, name).shape != expected:
            raise ValueError(f"result {name} must have shape {expected}")
    arrays = [
        result.t,
        result.linear_velocity,
        result.linear_acceleration,
        result.applied_force,
        result.multipliers,
        result.reaction_force,
        result.position_residuals,
        result.velocity_residuals,
        result.acceleration_residuals,
    ]
    arrays.extend(pose.translation for step in result.poses for pose in step)
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("dynamic result arrays must contain only finite values")


def _constraint_residual(mechanism, q, template, t):
    poses = _poses(q, template)
    values: list[float] = []
    if mechanism._ground is not None:
        values.extend(poses[mechanism._ground].translation)
    for joint in mechanism._joints:
        values.extend(joint.residual(poses)[:3])
    for drive in mechanism._drives:
        values.append(drive.residual(poses, t))
    return np.asarray(values, dtype=float)


def _constraint_jacobian(mechanism):
    rows: list[np.ndarray] = []
    ncoord = 3 * mechanism.nbody
    if mechanism._ground is not None:
        block = np.zeros((3, ncoord))
        start = 3 * mechanism._ground
        block[:, start : start + 3] = np.eye(3)
        rows.append(block)
    for joint in mechanism._joints:
        block = np.zeros((3, ncoord))
        if joint.body_i is not None:
            block[:, 3 * joint.body_i : 3 * joint.body_i + 3] = np.eye(3)
        if joint.body_j is not None:
            block[:, 3 * joint.body_j : 3 * joint.body_j + 3] = -np.eye(3)
        rows.append(block)
    for drive in mechanism._drives:
        row = np.zeros((1, ncoord))
        row[0, 3 * drive.body + {"x": 0, "y": 1, "z": 2}[drive.coordinate]] = 1.0
        rows.append(row)
    return np.vstack(rows) if rows else np.zeros((0, ncoord))


def _constraint_time_derivative(mechanism, t):
    count = (3 if mechanism._ground is not None else 0) + 3 * len(mechanism._joints) + len(mechanism._drives)
    values = np.zeros(count)
    offset = count - len(mechanism._drives)
    for index, drive in enumerate(mechanism._drives):
        drive.validate_derivative(t)
        values[offset + index] = -float(drive.velocity(t))
    return values


def _constraint_acceleration_rhs(mechanism, t):
    count = (3 if mechanism._ground is not None else 0) + 3 * len(mechanism._joints) + len(mechanism._drives)
    values = np.zeros(count)
    offset = count - len(mechanism._drives)
    for index, drive in enumerate(mechanism._drives):
        values[offset + index] = drive.acceleration_value(t)
    return values


def _validate_fixed_orientation_configuration(mechanism, poses, tolerance):
    if mechanism._ground is not None:
        ground = poses[mechanism._ground]
        if np.linalg.norm(ground.rotation.as_array() - np.array([1.0, 0.0, 0.0, 0.0])) > tolerance:
            raise MechanismSolveError("ground orientation must be identity in fixed-orientation dynamics")
    for joint in mechanism._joints:
        if isinstance(joint, FixedJoint3D) and _norm(joint.residual(poses)[3:]) > tolerance:
            raise MechanismSolveError(
                f"fixed joint {joint.name!r} has an inconsistent initial orientation"
            )


def _minimum_norm_velocity(mechanism, jacobian, t):
    if jacobian.size == 0:
        return np.zeros(3 * mechanism.nbody)
    value, *_ = np.linalg.lstsq(jacobian, -_constraint_time_derivative(mechanism, t), rcond=None)
    return value


def _force_vector(forces, t, poses, velocities):
    values = np.zeros(3 * len(poses))
    for force in forces:
        values[3 * force.body : 3 * force.body + 3] += force.evaluate(t, poses, velocities)
    return values


def _excitation_id(mechanism, times, forces, applied_forces):
    payload = {
        "kinematic_excitation_id": mechanism.excitation_id(times),
        "force_descriptors": [force.as_dict() for force in forces],
        "applied_force_samples": applied_forces.tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _translations(poses):
    return np.concatenate([pose.translation for pose in poses])


def _poses(q, template):
    return tuple(
        Pose3D(q[3 * index : 3 * index + 3], pose.rotation)
        for index, pose in enumerate(template)
    )


def _state(value, length, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape ({length},)")
    return np.array(array, copy=True)


def _vector3(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return _readonly(array)


def _time_array(value):
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("t must be a non-empty finite one-dimensional array")
    if len(array) > 1 and np.any(np.diff(array) <= 0.0):
        raise ValueError("t must be strictly increasing")
    return np.array(array, copy=True)


def _norm(value):
    array = np.asarray(value, dtype=float)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _package_version():
    try:
        return version("mbsd")
    except PackageNotFoundError:
        return "0+local"
