"""Experimental unconstrained rigid-body translation and rotation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING
import json

import numpy as np
from scipy.integrate import solve_ivp

from ..errors import MechanismSolveError
from .vocabulary import Pose3D, Quaternion

if TYPE_CHECKING:
    from .builder import SpatialMechanism


VectorLaw = Callable[[float, Pose3D, np.ndarray, np.ndarray], Iterable[float]]
RIGID_BODY_RESULT_SCHEMA = "mbsd.spatial.free_rigid_body_dynamic_result"
RIGID_BODY_RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RigidBodyLoad:
    """Force, torque, and body-fixed application point for a free rigid body."""

    body: int
    force: Iterable[float] | VectorLaw = (0.0, 0.0, 0.0)
    torque: Iterable[float] | VectorLaw = (0.0, 0.0, 0.0)
    application_point_from_com_body: Iterable[float] = (0.0, 0.0, 0.0)
    force_frame: str = "world"
    torque_frame: str = "body"
    name: str = "load"

    def __post_init__(self) -> None:
        if isinstance(self.body, bool) or int(self.body) < 0:
            raise ValueError("load body must be a non-negative integer")
        object.__setattr__(self, "body", int(self.body))
        if not self.name:
            raise ValueError("load name must be non-empty")
        if self.force_frame not in {"world", "body"}:
            raise ValueError("force_frame must be 'world' or 'body'")
        if self.torque_frame not in {"world", "body"}:
            raise ValueError("torque_frame must be 'world' or 'body'")
        object.__setattr__(self, "application_point_from_com_body", _vector(self.application_point_from_com_body, "application_point_from_com_body"))
        if not callable(self.force):
            object.__setattr__(self, "force", _vector(self.force, "force"))
        if not callable(self.torque):
            object.__setattr__(self, "torque", _vector(self.torque, "torque"))

    def evaluate(self, t: float, pose: Pose3D, velocity: np.ndarray, omega_body: np.ndarray):
        rotation = pose.rotation.to_rotation_matrix()
        force = self.force(t, pose, velocity, omega_body) if callable(self.force) else self.force
        torque = self.torque(t, pose, velocity, omega_body) if callable(self.torque) else self.torque
        force = _vector(force, f"load {self.name!r} force")
        torque = _vector(torque, f"load {self.name!r} torque")
        force_world = force if self.force_frame == "world" else rotation @ force
        force_body = rotation.T @ force_world
        torque_body = torque if self.torque_frame == "body" else rotation.T @ torque
        torque_body = torque_body + np.cross(self.application_point_from_com_body, force_body)
        return force_world, torque_body

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "body": self.body,
            "force_frame": self.force_frame, "torque_frame": self.torque_frame,
            "application_point_from_com_body": self.application_point_from_com_body.tolist(),
            "force_kind": "runtime_law" if callable(self.force) else "constant",
            "torque_kind": "runtime_law" if callable(self.torque) else "constant",
            "force": None if callable(self.force) else self.force.tolist(),
            "torque": None if callable(self.torque) else self.torque.tolist(),
        }


@dataclass(frozen=True)
class SpatialRigidBodyDynamicsResult:
    t: np.ndarray
    poses: tuple[Pose3D, ...]
    linear_velocity: np.ndarray
    angular_velocity_body: np.ndarray
    linear_acceleration: np.ndarray
    angular_acceleration_body: np.ndarray
    applied_force_world: np.ndarray
    applied_torque_body: np.ndarray
    inertia_tensor_body: np.ndarray
    success: bool
    message: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "t", "linear_velocity", "angular_velocity_body", "linear_acceleration",
            "angular_acceleration_body", "applied_force_world", "applied_torque_body",
            "inertia_tensor_body",
        ):
            array = np.asarray(getattr(self, name), dtype=float)
            object.__setattr__(self, name, np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape))


@dataclass(frozen=True)
class RigidBodyDynamicsDiagnostics:
    steps: int
    max_quaternion_norm_error: float
    max_rotation_orthogonality_error: float
    relative_energy_change: float
    relative_linear_momentum_change: float
    relative_angular_momentum_change: float
    finite: bool
    success: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def free_body_capabilities() -> Mapping[str, Any]:
    return MappingProxyType({
        "solver_kind": "unconstrained_free_rigid_body_dynamics",
        "translation_dynamics": True,
        "orientation_dynamics": True,
        "torque_inputs": True,
        "force_at_point": True,
        "constraints": False,
        "multiple_bodies": False,
        "orientation": "active_body_to_world_quaternion_wxyz",
        "angular_velocity_frame": "body",
        "inertia_frame": "body",
    })


def simulate_free_body(
    mechanism: SpatialMechanism,
    t: np.ndarray,
    *,
    linear_velocity0: Iterable[float] = (0.0, 0.0, 0.0),
    angular_velocity0_body: Iterable[float] = (0.0, 0.0, 0.0),
    loads: Iterable[RigidBodyLoad] = (),
    inertia_tensor_body: np.ndarray | None = None,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> SpatialRigidBodyDynamicsResult:
    """Integrate one unconstrained rigid body using body-frame Euler equations."""
    if mechanism.nbody != 1 or mechanism._ground is not None or mechanism._joints or mechanism._drives:
        raise MechanismSolveError("free-body dynamics requires exactly one body and no ground, joints, or drives")
    times = np.asarray(t, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("t must be a finite, strictly increasing array with at least two samples")
    if method not in {"RK45", "DOP853", "Radau"}:
        raise ValueError("method must be one of: RK45, DOP853, Radau")
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite scalar")
    body = mechanism._bodies[0]
    load_values = tuple(loads)
    if any(not isinstance(load, RigidBodyLoad) or load.body != 0 for load in load_values):
        raise ValueError("every load must be a RigidBodyLoad for the mechanism's sole body")
    pose0 = body.pose
    rotation0 = pose0.rotation.to_rotation_matrix()
    com0 = pose0.translation + rotation0 @ body.center_of_mass
    velocity0 = _vector(linear_velocity0, "linear_velocity0")
    omega0 = _vector(angular_velocity0_body, "angular_velocity0_body")
    inertia = _inertia_tensor(
        np.diag(body.inertia) if inertia_tensor_body is None else inertia_tensor_body
    )

    def evaluate(ti, state):
        com, quaternion, velocity, omega = state[:3], state[3:7], state[7:10], state[10:13]
        quaternion = _normalized_quaternion(quaternion)
        pose = _reference_pose(com, quaternion, body.center_of_mass)
        force = np.zeros(3)
        torque = np.zeros(3)
        for load in load_values:
            force_i, torque_i = load.evaluate(ti, pose, velocity, omega)
            force += force_i
            torque += torque_i
        angular_acceleration = np.linalg.solve(
            inertia, torque - np.cross(omega, inertia @ omega)
        )
        quaternion_rate = 0.5 * _quaternion_product(quaternion, np.array([0.0, *omega]))
        return force / body.mass, angular_acceleration, quaternion_rate, force, torque

    def rhs(ti, state):
        linear_acceleration, angular_acceleration, quaternion_rate, _, _ = evaluate(ti, state)
        return np.concatenate((state[7:10], quaternion_rate, linear_acceleration, angular_acceleration))

    initial = np.concatenate((com0, pose0.rotation.as_array(), velocity0, omega0))
    solution = solve_ivp(rhs, (float(times[0]), float(times[-1])), initial, t_eval=times, method=method, rtol=float(rtol), atol=float(atol))
    if not solution.success:
        raise MechanismSolveError(f"Free rigid-body dynamics failed: {solution.message}")
    states = solution.y.T
    poses: list[Pose3D] = []
    linear_acceleration = np.zeros((len(times), 3))
    angular_acceleration = np.zeros((len(times), 3))
    force_history = np.zeros((len(times), 3))
    torque_history = np.zeros((len(times), 3))
    for index, (ti, state) in enumerate(zip(times, states)):
        quaternion = _normalized_quaternion(state[3:7])
        states[index, 3:7] = quaternion
        poses.append(_reference_pose(state[:3], quaternion, body.center_of_mass))
        linear_acceleration[index], angular_acceleration[index], _, force_history[index], torque_history[index] = evaluate(float(ti), state)
    return SpatialRigidBodyDynamicsResult(
        t=times, poses=tuple(poses), linear_velocity=states[:, 7:10],
        angular_velocity_body=states[:, 10:13], linear_acceleration=linear_acceleration,
        angular_acceleration_body=angular_acceleration, applied_force_world=force_history,
        applied_torque_body=torque_history, success=True, message="converged",
        inertia_tensor_body=inertia,
        provenance=MappingProxyType({
            "solver": "scipy.solve_ivp", "method": method,
            "rtol": float(rtol), "atol": float(atol),
            "quaternion_normalization": "rhs_and_output_samples",
            "angular_velocity_frame": "body", "inertia_frame": "body",
        }),
        capabilities=free_body_capabilities(),
    )


def rigid_body_diagnostics(mechanism: SpatialMechanism, result: SpatialRigidBodyDynamicsResult) -> RigidBodyDynamicsDiagnostics:
    if mechanism.nbody != 1 or not isinstance(result, SpatialRigidBodyDynamicsResult):
        raise ValueError("result must belong to a one-body spatial mechanism")
    body = mechanism._bodies[0]
    rotations = np.asarray([pose.rotation.to_rotation_matrix() for pose in result.poses])
    quaternions = np.asarray([pose.rotation.as_array() for pose in result.poses])
    translational = 0.5 * body.mass * np.sum(result.linear_velocity**2, axis=1)
    rotational = 0.5 * np.einsum(
        "ti,ij,tj->t", result.angular_velocity_body,
        result.inertia_tensor_body, result.angular_velocity_body,
    )
    energy = translational + rotational
    linear_momentum = body.mass * result.linear_velocity
    angular_momentum = np.einsum(
        "tij,jk,tk->ti", rotations, result.inertia_tensor_body,
        result.angular_velocity_body,
    )
    return RigidBodyDynamicsDiagnostics(
        steps=len(result.t),
        max_quaternion_norm_error=float(np.max(np.abs(np.linalg.norm(quaternions, axis=1) - 1.0))),
        max_rotation_orthogonality_error=float(np.max(np.abs(rotations @ np.swapaxes(rotations, 1, 2) - np.eye(3)))),
        relative_energy_change=_relative_change(energy),
        relative_linear_momentum_change=_relative_vector_change(linear_momentum),
        relative_angular_momentum_change=_relative_vector_change(angular_momentum),
        finite=all(np.all(np.isfinite(value)) for value in (result.t, result.linear_velocity, result.angular_velocity_body)),
        success=result.success,
    )


def rigid_body_result_to_dict(mechanism: SpatialMechanism, result: SpatialRigidBodyDynamicsResult) -> dict[str, Any]:
    body = mechanism._bodies[0]
    return {
        "schema": RIGID_BODY_RESULT_SCHEMA, "schema_version": RIGID_BODY_RESULT_SCHEMA_VERSION,
        "status": "experimental", "solver_kind": "unconstrained_free_rigid_body_dynamics",
        "units": {"length": "m", "time": "s", "linear_velocity": "m/s", "angular_velocity": "rad/s", "force": "N", "torque": "N*m", "inertia": "kg*m^2"},
        "conventions": {"world_frame": "right_handed_xyz", "quaternion_order": ["w", "x", "y", "z"], "orientation": "active_body_to_world", "angular_velocity_frame": "body", "inertia_frame": "body"},
        "capabilities": dict(result.capabilities), "solver_status": {"success": result.success, "message": result.message},
        "provenance": dict(result.provenance), "diagnostics": rigid_body_diagnostics(mechanism, result).as_dict(),
        "time": result.t.tolist(), "body": {"name": body.name, "mass": body.mass, "inertia_tensor_body": result.inertia_tensor_body.tolist(), "center_of_mass": body.center_of_mass.tolist(), "poses": [pose.as_dict() for pose in result.poses], "linear_velocity": result.linear_velocity.tolist(), "angular_velocity_body": result.angular_velocity_body.tolist(), "linear_acceleration": result.linear_acceleration.tolist(), "angular_acceleration_body": result.angular_acceleration_body.tolist(), "applied_force_world": result.applied_force_world.tolist(), "applied_torque_body": result.applied_torque_body.tolist()},
    }


def rigid_body_result_to_json(mechanism, result, path: str | Path, *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rigid_body_result_to_dict(mechanism, result), indent=indent, allow_nan=False) + "\n", encoding="utf-8")
    return path


def validate_rigid_body_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the portable unconstrained free-rigid-body result contract."""
    if not isinstance(payload, Mapping):
        raise TypeError("free rigid-body result payload must be a mapping")
    if payload.get("schema") != RIGID_BODY_RESULT_SCHEMA or payload.get("schema_version") != 1:
        raise ValueError("unsupported free rigid-body result schema")
    if payload.get("status") != "experimental" or payload.get("solver_kind") != "unconstrained_free_rigid_body_dynamics":
        raise ValueError("invalid free rigid-body solver identity")
    expected_conventions = {"world_frame": "right_handed_xyz", "quaternion_order": ["w", "x", "y", "z"], "orientation": "active_body_to_world", "angular_velocity_frame": "body", "inertia_frame": "body"}
    if payload.get("conventions") != expected_conventions:
        raise ValueError("free rigid-body conventions are invalid")
    capabilities = payload.get("capabilities")
    if capabilities != dict(free_body_capabilities()):
        raise ValueError("free rigid-body capabilities are invalid")
    times = np.asarray(payload.get("time"), dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("time must be finite, increasing, and contain at least two samples")
    body = payload.get("body")
    if not isinstance(body, Mapping) or not isinstance(body.get("name"), str) or not body["name"]:
        raise ValueError("body identity is invalid")
    mass = body.get("mass")
    if not isinstance(mass, (int, float)) or not np.isfinite(mass) or mass <= 0:
        raise ValueError("body mass must be positive and finite")
    _inertia_tensor(body.get("inertia_tensor_body"))
    center = np.asarray(body.get("center_of_mass"), dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("body center_of_mass must be a finite vector")
    poses = body.get("poses")
    if not isinstance(poses, list) or len(poses) != len(times):
        raise ValueError("body poses must match time length")
    for pose in poses:
        if not isinstance(pose, Mapping):
            raise ValueError("each body pose must be a mapping")
        translation = np.asarray(pose.get("translation"), dtype=float)
        rotation = pose.get("rotation")
        if translation.shape != (3,) or not np.all(np.isfinite(translation)) or not isinstance(rotation, Mapping):
            raise ValueError("body pose is invalid")
        try:
            quaternion = np.asarray([rotation[key] for key in ("w", "x", "y", "z")], dtype=float)
        except KeyError as error:
            raise ValueError("body pose quaternion is incomplete") from error
        if not np.all(np.isfinite(quaternion)) or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-10, rtol=1e-10):
            raise ValueError("body pose quaternion norm must be 1")
    for name in ("linear_velocity", "angular_velocity_body", "linear_acceleration", "angular_acceleration_body", "applied_force_world", "applied_torque_body"):
        values = np.asarray(body.get(name), dtype=float)
        if values.shape != (len(times), 3) or not np.all(np.isfinite(values)):
            raise ValueError(f"body.{name} must be finite with time_by_3 shape")
    diagnostics = payload.get("diagnostics")
    status = payload.get("solver_status")
    if not isinstance(diagnostics, Mapping) or diagnostics.get("steps") != len(times):
        raise ValueError("diagnostics.steps must match time length")
    if not isinstance(status, Mapping) or diagnostics.get("success") is not status.get("success"):
        raise ValueError("diagnostics and solver status must agree")
    if diagnostics.get("finite") is not True:
        raise ValueError("diagnostics must report finite arrays")
    try:
        return json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("free rigid-body result must contain finite JSON values") from error


def load_rigid_body_result_payload(path: str | Path) -> dict[str, Any]:
    return validate_rigid_body_result_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def _reference_pose(com, quaternion, center_of_mass):
    rotation = Quaternion(*quaternion)
    return Pose3D(np.asarray(com) - rotation.rotate(center_of_mass), rotation)


def _normalized_quaternion(value):
    value = np.asarray(value, dtype=float)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm == 0:
        raise MechanismSolveError("quaternion integration produced an invalid orientation")
    return value / norm


def _quaternion_product(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return np.array(array, copy=True)


def _inertia_tensor(value):
    tensor = np.asarray(value, dtype=float)
    if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
        raise ValueError("inertia_tensor_body must be a finite 3 by 3 matrix")
    if not np.allclose(tensor, tensor.T, atol=1e-12, rtol=1e-12):
        raise ValueError("inertia_tensor_body must be symmetric")
    eigenvalues = np.linalg.eigvalsh(tensor)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("inertia_tensor_body must be positive definite")
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(eigenvalues))) * 32.0
    if any(eigenvalues[index] > np.sum(eigenvalues) - eigenvalues[index] + tolerance for index in range(3)):
        raise ValueError("inertia_tensor_body principal values must satisfy triangle inequalities")
    return np.array(tensor, copy=True)


def _relative_change(values):
    return float(np.max(np.abs(values - values[0])) / max(1.0, abs(float(values[0]))))


def _relative_vector_change(values):
    return float(np.max(np.linalg.norm(values - values[0], axis=1)) / max(1.0, float(np.linalg.norm(values[0]))))
