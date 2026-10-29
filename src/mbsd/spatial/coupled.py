"""Experimental unconstrained multi-body six-DOF rigid-body dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from ..errors import MechanismSolveError
from .dynamics import _package_version
from .rotation import RigidBodyLoad, _inertia_tensor, _normalized_quaternion, _quaternion_product, _vector
from .vocabulary import Pose3D, Quaternion

if TYPE_CHECKING:
    from .builder import SpatialMechanism


COUPLED_RESULT_SCHEMA = "mbsd.spatial.coupled_rigid_body_dynamic_result"
COUPLED_RESULT_SCHEMA_VERSION = 2
PairForceLaw = Callable[[float, tuple[Pose3D, ...], np.ndarray, np.ndarray], Iterable[float]]


@dataclass(frozen=True)
class RigidBodyPairInteraction:
    """Equal-and-opposite world force applied to two body-fixed points."""

    body_i: int
    body_j: int
    force_on_i_world: Iterable[float] | PairForceLaw
    point_i_from_com_body: Iterable[float] = (0.0, 0.0, 0.0)
    point_j_from_com_body: Iterable[float] = (0.0, 0.0, 0.0)
    name: str = "interaction"

    def __post_init__(self) -> None:
        for name in ("body_i", "body_j"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        if self.body_i == self.body_j:
            raise ValueError("pair interaction requires two different bodies")
        if not self.name:
            raise ValueError("interaction name must be non-empty")
        object.__setattr__(self, "point_i_from_com_body", _vector(self.point_i_from_com_body, "point_i_from_com_body"))
        object.__setattr__(self, "point_j_from_com_body", _vector(self.point_j_from_com_body, "point_j_from_com_body"))
        if not callable(self.force_on_i_world):
            object.__setattr__(self, "force_on_i_world", _vector(self.force_on_i_world, "force_on_i_world"))

    def evaluate(self, t, poses, velocities, angular_velocities):
        value = self.force_on_i_world(t, poses, velocities, angular_velocities) if callable(self.force_on_i_world) else self.force_on_i_world
        return _vector(value, f"interaction {self.name!r} force")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "body_i": self.body_i, "body_j": self.body_j,
            "frame": "world", "force_pair": "equal_and_opposite",
            "point_i_from_com_body": self.point_i_from_com_body.tolist(),
            "point_j_from_com_body": self.point_j_from_com_body.tolist(),
            "kind": "runtime_law" if callable(self.force_on_i_world) else "constant",
            "force_on_i_world": None if callable(self.force_on_i_world) else self.force_on_i_world.tolist(),
        }


@dataclass(frozen=True)
class CoupledRigidBodyDynamicsResult:
    t: np.ndarray
    poses: tuple[tuple[Pose3D, ...], ...]
    center_of_mass_position: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity_body: np.ndarray
    linear_acceleration: np.ndarray
    angular_acceleration_body: np.ndarray
    applied_force_world: np.ndarray
    applied_torque_body: np.ndarray
    interaction_force_world: np.ndarray
    inertia_tensor_body: np.ndarray
    model_id: str
    excitation_id: str
    simulation_id: str
    success: bool
    message: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "t", "center_of_mass_position", "linear_velocity", "angular_velocity_body",
            "linear_acceleration", "angular_acceleration_body", "applied_force_world",
            "applied_torque_body", "interaction_force_world", "inertia_tensor_body",
        ):
            array = np.asarray(getattr(self, name), dtype=float)
            object.__setattr__(self, name, np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape))

    def body_poses(self, body: int) -> tuple[Pose3D, ...]:
        return tuple(step[int(body)] for step in self.poses)


@dataclass(frozen=True)
class CoupledDynamicsDiagnostics:
    steps: int
    bodies: int
    max_quaternion_norm_error: float
    max_rotation_orthogonality_error: float
    relative_kinetic_energy_change: float
    relative_linear_momentum_change: float
    relative_angular_momentum_change: float
    pair_force_balance_residual_n: float
    finite: bool
    success: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def coupled_capabilities() -> Mapping[str, Any]:
    return MappingProxyType({
        "solver_kind": "unconstrained_coupled_six_dof_rigid_body_dynamics",
        "translation_dynamics": True, "orientation_dynamics": True,
        "multiple_bodies": True, "pair_interactions": True, "gravity": True,
        "pair_force_histories": True,
        "force_at_point": True, "offset_center_of_mass": True,
        "constraints": False, "contact": False,
        "orientation": "active_body_to_world_quaternion_wxyz",
        "angular_velocity_frame": "body", "inertia_frame": "body",
    })


def simulate_coupled_rigid_bodies(
    mechanism: SpatialMechanism,
    t: np.ndarray,
    *,
    linear_velocity0: np.ndarray | None = None,
    angular_velocity0_body: np.ndarray | None = None,
    loads: Iterable[RigidBodyLoad] = (),
    interactions: Iterable[RigidBodyPairInteraction] = (),
    gravity_world: Iterable[float] = (0.0, 0.0, 0.0),
    inertia_tensors_body: np.ndarray | None = None,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> CoupledRigidBodyDynamicsResult:
    """Integrate unconstrained six-DOF motion for one or more rigid bodies."""
    if mechanism.nbody == 0 or mechanism._ground is not None or mechanism._joints or mechanism._drives:
        raise MechanismSolveError("coupled rigid-body dynamics requires bodies without ground, joints, or drives")
    times = np.asarray(t, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("t must be a finite, strictly increasing array with at least two samples")
    if method not in {"RK45", "DOP853", "Radau"}:
        raise ValueError("method must be one of: RK45, DOP853, Radau")
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite scalar")
    count = mechanism.nbody
    masses = np.asarray([body.mass for body in mechanism._bodies])
    gravity = _vector(gravity_world, "gravity_world")
    velocity0 = _matrix3(linear_velocity0, count, "linear_velocity0")
    omega0 = _matrix3(angular_velocity0_body, count, "angular_velocity0_body")
    if inertia_tensors_body is None:
        inertias = np.asarray([np.diag(body.inertia) for body in mechanism._bodies])
    else:
        raw = np.asarray(inertia_tensors_body, dtype=float)
        if raw.shape != (count, 3, 3):
            raise ValueError(f"inertia_tensors_body must have shape ({count}, 3, 3)")
        inertias = np.asarray([_inertia_tensor(value) for value in raw])
    load_values = tuple(loads)
    for load in load_values:
        if not isinstance(load, RigidBodyLoad) or load.body >= count:
            raise ValueError("loads must reference bodies in this mechanism")
    interaction_values = tuple(interactions)
    for interaction in interaction_values:
        if not isinstance(interaction, RigidBodyPairInteraction) or max(interaction.body_i, interaction.body_j) >= count:
            raise ValueError("interactions must reference bodies in this mechanism")
    com0 = np.asarray([
        body.pose.translation + body.pose.rotation.rotate(body.center_of_mass)
        for body in mechanism._bodies
    ])
    quaternion0 = np.asarray([body.pose.rotation.as_array() for body in mechanism._bodies])

    def unpack(state):
        blocks = state.reshape(count, 13)
        return blocks[:, :3], blocks[:, 3:7], blocks[:, 7:10], blocks[:, 10:13]

    def evaluate(ti, state):
        com, quaternion, velocity, omega = unpack(state)
        quaternion = np.asarray([_normalized_quaternion(value) for value in quaternion])
        rotations = np.asarray([Quaternion(*value).to_rotation_matrix() for value in quaternion])
        poses = tuple(Pose3D(com[index] - rotations[index] @ mechanism._bodies[index].center_of_mass, Quaternion(*quaternion[index])) for index in range(count))
        force = masses[:, None] * gravity
        torque = np.zeros((count, 3))
        for load in load_values:
            force_i, torque_i = load.evaluate(ti, poses[load.body], velocity[load.body], omega[load.body])
            force[load.body] += force_i
            torque[load.body] += torque_i
        interaction_forces = np.zeros((len(interaction_values), 3))
        for interaction_index, interaction in enumerate(interaction_values):
            value = interaction.evaluate(ti, poses, velocity, omega)
            interaction_forces[interaction_index] = value
            i, j = interaction.body_i, interaction.body_j
            force[i] += value
            force[j] -= value
            force_i_body = rotations[i].T @ value
            force_j_body = rotations[j].T @ (-value)
            torque[i] += np.cross(interaction.point_i_from_com_body, force_i_body)
            torque[j] += np.cross(interaction.point_j_from_com_body, force_j_body)
        linear_acceleration = force / masses[:, None]
        angular_acceleration = np.asarray([
            np.linalg.solve(inertias[index], torque[index] - np.cross(omega[index], inertias[index] @ omega[index]))
            for index in range(count)
        ])
        quaternion_rate = np.asarray([
            0.5 * _quaternion_product(quaternion[index], np.array([0.0, *omega[index]]))
            for index in range(count)
        ])
        return linear_acceleration, angular_acceleration, quaternion_rate, force, torque, poses, interaction_forces

    def rhs(ti, state):
        _, _, velocity, _ = unpack(state)
        linear_acceleration, angular_acceleration, quaternion_rate, _, _, _, _ = evaluate(ti, state)
        derivative = np.zeros((count, 13))
        derivative[:, :3] = velocity
        derivative[:, 3:7] = quaternion_rate
        derivative[:, 7:10] = linear_acceleration
        derivative[:, 10:13] = angular_acceleration
        return derivative.reshape(-1)

    initial = np.concatenate((com0, quaternion0, velocity0, omega0), axis=1).reshape(-1)
    solution = solve_ivp(rhs, (float(times[0]), float(times[-1])), initial, t_eval=times, method=method, rtol=float(rtol), atol=float(atol))
    if not solution.success:
        raise MechanismSolveError(f"Coupled rigid-body dynamics failed: {solution.message}")
    states = solution.y.T.reshape(len(times), count, 13)
    com_history = states[:, :, :3].copy()
    velocity_history = states[:, :, 7:10].copy()
    omega_history = states[:, :, 10:13].copy()
    linear_acceleration = np.zeros((len(times), count, 3))
    angular_acceleration = np.zeros_like(linear_acceleration)
    force_history = np.zeros_like(linear_acceleration)
    torque_history = np.zeros_like(linear_acceleration)
    pose_history: list[tuple[Pose3D, ...]] = []
    interaction_force_history = np.zeros((len(times), len(interaction_values), 3))
    for time_index, ti in enumerate(times):
        for body_index in range(count):
            states[time_index, body_index, 3:7] = _normalized_quaternion(states[time_index, body_index, 3:7])
        evaluated = evaluate(float(ti), states[time_index].reshape(-1))
        linear_acceleration[time_index], angular_acceleration[time_index], _, force_history[time_index], torque_history[time_index], poses, interaction_forces = evaluated
        pose_history.append(poses)
        interaction_force_history[time_index] = interaction_forces
    provenance = MappingProxyType({
        "solver": "scipy.solve_ivp", "method": method, "rtol": float(rtol), "atol": float(atol),
        "quaternion_normalization": "rhs_and_output_samples", "angular_velocity_frame": "body",
        "inertia_frame": "body", "gravity_world": gravity.tolist(),
        "load_count": len(load_values), "interaction_count": len(interaction_values),
        "load_descriptors": tuple(MappingProxyType(value.as_dict()) for value in load_values),
        "interaction_descriptors": tuple(MappingProxyType(value.as_dict()) for value in interaction_values),
        "pair_force_contract": "exported_force_on_i_and_exact_negative_on_j",
    })
    excitation_payload = {
        "time": times.tolist(), "gravity_world": gravity.tolist(),
        "loads": [value.as_dict() for value in load_values],
        "interactions": [value.as_dict() for value in interaction_values],
        "applied_force_samples": force_history.tolist(),
        "applied_torque_samples": torque_history.tolist(),
        "interaction_force_samples": interaction_force_history.tolist(),
    }
    excitation_id = hashlib.sha256(json.dumps(excitation_payload, sort_keys=True).encode()).hexdigest()[:16]
    simulation_payload = {
        "model_id": mechanism.model_id(), "excitation_id": excitation_id,
        "initial_center_of_mass_position": com0.tolist(),
        "initial_orientation_wxyz": quaternion0.tolist(),
        "initial_linear_velocity_world": velocity0.tolist(),
        "initial_angular_velocity_body": omega0.tolist(),
        "effective_inertia_tensor_body": inertias.tolist(),
        "time": times.tolist(), "method": method, "rtol": float(rtol), "atol": float(atol),
    }
    return CoupledRigidBodyDynamicsResult(
        t=times, poses=tuple(pose_history), center_of_mass_position=com_history,
        linear_velocity=velocity_history, angular_velocity_body=omega_history,
        linear_acceleration=linear_acceleration, angular_acceleration_body=angular_acceleration,
        applied_force_world=force_history, applied_torque_body=torque_history,
        interaction_force_world=interaction_force_history,
        inertia_tensor_body=inertias, success=True, message="converged",
        model_id=mechanism.model_id(),
        excitation_id=excitation_id,
        simulation_id=hashlib.sha256(json.dumps(simulation_payload, sort_keys=True).encode()).hexdigest()[:16],
        provenance=provenance, capabilities=coupled_capabilities(),
    )


def coupled_diagnostics(mechanism: SpatialMechanism, result: CoupledRigidBodyDynamicsResult) -> CoupledDynamicsDiagnostics:
    _validate_coupled_result(mechanism, result)
    masses = np.asarray([body.mass for body in mechanism._bodies])
    rotations = np.asarray([[pose.rotation.to_rotation_matrix() for pose in step] for step in result.poses])
    quaternions = np.asarray([[pose.rotation.as_array() for pose in step] for step in result.poses])
    translational = 0.5 * np.sum(masses[None, :, None] * result.linear_velocity**2, axis=(1, 2))
    rotational = 0.5 * np.einsum("tbi,bij,tbj->t", result.angular_velocity_body, result.inertia_tensor_body, result.angular_velocity_body)
    energy = translational + rotational
    momentum = np.sum(masses[None, :, None] * result.linear_velocity, axis=1)
    spin = np.einsum("tbij,bjk,tbk->tbi", rotations, result.inertia_tensor_body, result.angular_velocity_body)
    orbital = np.sum(np.cross(result.center_of_mass_position, masses[None, :, None] * result.linear_velocity), axis=1)
    angular_momentum = orbital + np.sum(spin, axis=1)
    return CoupledDynamicsDiagnostics(
        steps=len(result.t), bodies=mechanism.nbody,
        max_quaternion_norm_error=float(np.max(np.abs(np.linalg.norm(quaternions, axis=2) - 1.0))),
        max_rotation_orthogonality_error=float(np.max(np.abs(rotations @ np.swapaxes(rotations, 2, 3) - np.eye(3)))),
        relative_kinetic_energy_change=_relative_change(energy),
        relative_linear_momentum_change=_relative_vector_change(momentum),
        relative_angular_momentum_change=_relative_vector_change(angular_momentum),
        pair_force_balance_residual_n=_pair_force_imbalance(result.interaction_force_world),
        finite=all(np.all(np.isfinite(value)) for value in (result.t, result.center_of_mass_position, result.linear_velocity, result.angular_velocity_body)),
        success=result.success,
    )


def coupled_result_to_dict(mechanism: SpatialMechanism, result: CoupledRigidBodyDynamicsResult) -> dict[str, Any]:
    diagnostics = coupled_diagnostics(mechanism, result)
    return {
        "schema": COUPLED_RESULT_SCHEMA, "schema_version": COUPLED_RESULT_SCHEMA_VERSION,
        "mbsd_version": _package_version(), "model_id": result.model_id,
        "excitation_id": result.excitation_id,
        "simulation_id": result.simulation_id,
        "identity_scope": {
            "model_id": "mechanism topology, body properties, and authored poses",
            "excitation_id": "time grid, gravity, load and interaction descriptors, and sampled loads",
            "simulation_id": "model and excitation identities, complete initial state, effective inertia, time grid, and solver controls",
        },
        "status": "experimental", "solver_kind": "unconstrained_coupled_six_dof_rigid_body_dynamics",
        "units": {"length": "m", "time": "s", "linear_velocity": "m/s", "angular_velocity": "rad/s", "linear_acceleration": "m/s^2", "angular_acceleration": "rad/s^2", "force": "N", "torque": "N*m", "inertia": "kg*m^2"},
        "conventions": {"world_frame": "right_handed_xyz", "quaternion_order": ["w", "x", "y", "z"], "orientation": "active_body_to_world", "translation_state": "center_of_mass_world", "body_pose_translation": "body_reference_origin_world", "angular_velocity_frame": "body", "inertia_frame": "body"},
        "capabilities": dict(result.capabilities), "solver_status": {"success": result.success, "message": result.message},
        "provenance": _json_ready(result.provenance), "diagnostics": diagnostics.as_dict(), "time": result.t.tolist(),
        "interaction_force_world": result.interaction_force_world.tolist(),
        "bodies": [{
            "index": index, "name": body.name, "mass": body.mass,
            "center_of_mass_body": body.center_of_mass.tolist(),
            "inertia_tensor_body": result.inertia_tensor_body[index].tolist(),
            "poses": [step[index].as_dict() for step in result.poses],
            "center_of_mass_position": result.center_of_mass_position[:, index].tolist(),
            "linear_velocity": result.linear_velocity[:, index].tolist(),
            "angular_velocity_body": result.angular_velocity_body[:, index].tolist(),
            "linear_acceleration": result.linear_acceleration[:, index].tolist(),
            "angular_acceleration_body": result.angular_acceleration_body[:, index].tolist(),
            "applied_force_world": result.applied_force_world[:, index].tolist(),
            "applied_torque_body": result.applied_torque_body[:, index].tolist(),
        } for index, body in enumerate(mechanism._bodies)],
    }


def coupled_result_to_json(mechanism, result, path: str | Path, *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coupled_result_to_dict(mechanism, result), indent=indent, allow_nan=False) + "\n", encoding="utf-8")
    return path


def validate_coupled_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != COUPLED_RESULT_SCHEMA or payload.get("schema_version") != 2:
        raise ValueError("unsupported coupled rigid-body result schema")
    if payload.get("solver_kind") != "unconstrained_coupled_six_dof_rigid_body_dynamics" or payload.get("capabilities") != dict(coupled_capabilities()):
        raise ValueError("invalid coupled rigid-body solver contract")
    for name in ("mbsd_version", "model_id", "excitation_id", "simulation_id"):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise ValueError(f"{name} must be a non-empty string")
    expected_identity_scope = {
        "model_id": "mechanism topology, body properties, and authored poses",
        "excitation_id": "time grid, gravity, load and interaction descriptors, and sampled loads",
        "simulation_id": "model and excitation identities, complete initial state, effective inertia, time grid, and solver controls",
    }
    if payload.get("identity_scope") != expected_identity_scope:
        raise ValueError("coupled result identity scope is invalid")
    expected_units = {"length": "m", "time": "s", "linear_velocity": "m/s", "angular_velocity": "rad/s", "linear_acceleration": "m/s^2", "angular_acceleration": "rad/s^2", "force": "N", "torque": "N*m", "inertia": "kg*m^2"}
    expected_conventions = {"world_frame": "right_handed_xyz", "quaternion_order": ["w", "x", "y", "z"], "orientation": "active_body_to_world", "translation_state": "center_of_mass_world", "body_pose_translation": "body_reference_origin_world", "angular_velocity_frame": "body", "inertia_frame": "body"}
    if payload.get("units") != expected_units or payload.get("conventions") != expected_conventions:
        raise ValueError("coupled result units or conventions are invalid")
    provenance = payload.get("provenance")
    required_provenance = {"solver", "method", "rtol", "atol", "quaternion_normalization", "angular_velocity_frame", "inertia_frame", "gravity_world", "load_count", "interaction_count", "load_descriptors", "interaction_descriptors", "pair_force_contract"}
    if not isinstance(provenance, Mapping) or not required_provenance <= provenance.keys():
        raise ValueError("coupled result provenance is incomplete")
    if provenance.get("solver") != "scipy.solve_ivp" or provenance.get("method") not in {"RK45", "DOP853", "Radau"}:
        raise ValueError("coupled result solver provenance is invalid")
    if provenance.get("quaternion_normalization") != "rhs_and_output_samples" or provenance.get("angular_velocity_frame") != "body" or provenance.get("inertia_frame") != "body":
        raise ValueError("coupled result frame provenance is invalid")
    _vector(provenance.get("gravity_world"), "provenance.gravity_world")
    for name in ("rtol", "atol"):
        value = provenance.get(name)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"provenance.{name} must be positive and finite")
    for count_name, descriptors_name in (("load_count", "load_descriptors"), ("interaction_count", "interaction_descriptors")):
        count = provenance.get(count_name)
        descriptors = provenance.get(descriptors_name)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0 or not isinstance(descriptors, list) or len(descriptors) != count:
            raise ValueError(f"provenance.{count_name} does not match descriptors")
    times = np.asarray(payload.get("time"), dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("time must be finite, increasing, and contain at least two samples")
    bodies = payload.get("bodies")
    if not isinstance(bodies, list) or not bodies:
        raise ValueError("bodies must be a non-empty list")
    for descriptor in provenance["load_descriptors"]:
        if not isinstance(descriptor, Mapping) or descriptor.get("body") not in range(len(bodies)) or descriptor.get("force_frame") not in {"world", "body"} or descriptor.get("torque_frame") not in {"world", "body"}:
            raise ValueError("load descriptor is invalid")
    for descriptor in provenance["interaction_descriptors"]:
        if not isinstance(descriptor, Mapping) or descriptor.get("body_i") not in range(len(bodies)) or descriptor.get("body_j") not in range(len(bodies)) or descriptor.get("body_i") == descriptor.get("body_j") or descriptor.get("frame") != "world" or descriptor.get("force_pair") != "equal_and_opposite":
            raise ValueError("interaction descriptor is invalid")
    if provenance.get("pair_force_contract") != "exported_force_on_i_and_exact_negative_on_j":
        raise ValueError("pair force contract provenance is invalid")
    interaction_forces = np.asarray(payload.get("interaction_force_world"), dtype=float)
    expected_interaction_shape = (len(times), provenance["interaction_count"], 3)
    if interaction_forces.shape != expected_interaction_shape or not np.all(np.isfinite(interaction_forces)):
        raise ValueError("interaction_force_world has invalid shape or values")
    names: set[str] = set()
    masses: list[float] = []
    inertias: list[np.ndarray] = []
    com_positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    omegas: list[np.ndarray] = []
    rotations_by_body: list[np.ndarray] = []
    applied_forces: list[np.ndarray] = []
    applied_torques: list[np.ndarray] = []
    initial_quaternions: list[list[float]] = []
    for index, body in enumerate(bodies):
        if not isinstance(body, Mapping) or body.get("index") != index or not isinstance(body.get("name"), str) or not body["name"] or body["name"] in names:
            raise ValueError("body identities must be ordered, named, and unique")
        names.add(body["name"])
        mass = body.get("mass")
        if not isinstance(mass, (int, float)) or not np.isfinite(mass) or mass <= 0:
            raise ValueError("body mass must be positive and finite")
        masses.append(float(mass))
        inertias.append(_inertia_tensor(body.get("inertia_tensor_body")))
        center_body = _vector(body.get("center_of_mass_body"), "body.center_of_mass_body")
        arrays: dict[str, np.ndarray] = {}
        for name in ("center_of_mass_position", "linear_velocity", "angular_velocity_body", "linear_acceleration", "angular_acceleration_body", "applied_force_world", "applied_torque_body"):
            values = np.asarray(body.get(name), dtype=float)
            if values.shape != (len(times), 3) or not np.all(np.isfinite(values)):
                raise ValueError(f"body.{name} must be finite with time_by_3 shape")
            arrays[name] = values
        if not isinstance(body.get("poses"), list) or len(body["poses"]) != len(times):
            raise ValueError("body poses must match time length")
        body_rotations: list[np.ndarray] = []
        for time_index, pose in enumerate(body["poses"]):
            quaternion = pose.get("rotation") if isinstance(pose, Mapping) else None
            if not isinstance(quaternion, Mapping):
                raise ValueError("body pose rotation is invalid")
            vector = np.asarray([quaternion.get(key) for key in ("w", "x", "y", "z")], dtype=float)
            if not np.all(np.isfinite(vector)) or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-10, rtol=1e-10):
                raise ValueError("body pose quaternion norm must be 1")
            translation = np.asarray(pose.get("translation"), dtype=float)
            if translation.shape != (3,) or not np.all(np.isfinite(translation)):
                raise ValueError("body pose translation is invalid")
            rotation = Quaternion(*vector).to_rotation_matrix()
            body_rotations.append(rotation)
            reconstructed_com = translation + rotation @ center_body
            if not np.allclose(reconstructed_com, arrays["center_of_mass_position"][time_index], atol=1e-10, rtol=1e-10):
                raise ValueError("body pose and center-of-mass position are inconsistent")
        force_balance = mass * arrays["linear_acceleration"] - arrays["applied_force_world"]
        if not np.allclose(force_balance, 0.0, atol=1e-10, rtol=1e-10):
            raise ValueError("body translational force balance is invalid")
        rotational_balance = np.einsum("ij,tj->ti", inertias[-1], arrays["angular_acceleration_body"]) + np.cross(arrays["angular_velocity_body"], np.einsum("ij,tj->ti", inertias[-1], arrays["angular_velocity_body"])) - arrays["applied_torque_body"]
        if not np.allclose(rotational_balance, 0.0, atol=1e-10, rtol=1e-10):
            raise ValueError("body rotational torque balance is invalid")
        com_positions.append(arrays["center_of_mass_position"])
        velocities.append(arrays["linear_velocity"])
        omegas.append(arrays["angular_velocity_body"])
        rotations_by_body.append(np.asarray(body_rotations))
        applied_forces.append(arrays["applied_force_world"])
        applied_torques.append(arrays["applied_torque_body"])
        first_rotation = body["poses"][0]["rotation"]
        initial_quaternions.append([first_rotation[key] for key in ("w", "x", "y", "z")])
    diagnostics = payload.get("diagnostics")
    status = payload.get("solver_status")
    if not isinstance(diagnostics, Mapping) or diagnostics.get("steps") != len(times) or diagnostics.get("bodies") != len(bodies):
        raise ValueError("coupled diagnostics dimensions are invalid")
    if not isinstance(status, Mapping) or diagnostics.get("success") is not status.get("success") or diagnostics.get("finite") is not True:
        raise ValueError("coupled diagnostics and solver status are inconsistent")
    mass_array = np.asarray(masses)
    com_array = np.stack(com_positions, axis=1)
    velocity_array = np.stack(velocities, axis=1)
    omega_array = np.stack(omegas, axis=1)
    rotation_array = np.stack(rotations_by_body, axis=1)
    inertia_array = np.asarray(inertias)
    applied_force_array = np.stack(applied_forces, axis=1)
    applied_torque_array = np.stack(applied_torques, axis=1)
    excitation_payload = {
        "time": times.tolist(), "gravity_world": list(provenance["gravity_world"]),
        "loads": list(provenance["load_descriptors"]),
        "interactions": list(provenance["interaction_descriptors"]),
        "applied_force_samples": applied_force_array.tolist(),
        "applied_torque_samples": applied_torque_array.tolist(),
        "interaction_force_samples": interaction_forces.tolist(),
    }
    expected_excitation_id = hashlib.sha256(
        json.dumps(excitation_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    if payload["excitation_id"] != expected_excitation_id:
        raise ValueError("excitation_id does not match exported load histories")
    simulation_payload = {
        "model_id": payload["model_id"], "excitation_id": payload["excitation_id"],
        "initial_center_of_mass_position": com_array[0].tolist(),
        "initial_orientation_wxyz": initial_quaternions,
        "initial_linear_velocity_world": velocity_array[0].tolist(),
        "initial_angular_velocity_body": omega_array[0].tolist(),
        "effective_inertia_tensor_body": inertia_array.tolist(),
        "time": times.tolist(), "method": provenance["method"],
        "rtol": float(provenance["rtol"]), "atol": float(provenance["atol"]),
    }
    expected_simulation_id = hashlib.sha256(
        json.dumps(simulation_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    if payload["simulation_id"] != expected_simulation_id:
        raise ValueError("simulation_id does not match exported run configuration")
    kinetic = 0.5 * np.sum(mass_array[None, :, None] * velocity_array**2, axis=(1, 2)) + 0.5 * np.einsum("tbi,bij,tbj->t", omega_array, inertia_array, omega_array)
    momentum = np.sum(mass_array[None, :, None] * velocity_array, axis=1)
    spin = np.einsum("tbij,bjk,tbk->tbi", rotation_array, inertia_array, omega_array)
    angular_momentum = np.sum(np.cross(com_array, mass_array[None, :, None] * velocity_array), axis=1) + np.sum(spin, axis=1)
    expected_diagnostics = {
        "max_quaternion_norm_error": 0.0,
        "max_rotation_orthogonality_error": float(np.max(np.abs(rotation_array @ np.swapaxes(rotation_array, 2, 3) - np.eye(3)))),
        "relative_kinetic_energy_change": _relative_change(kinetic),
        "relative_linear_momentum_change": _relative_vector_change(momentum),
        "relative_angular_momentum_change": _relative_vector_change(angular_momentum),
        "pair_force_balance_residual_n": _pair_force_imbalance(interaction_forces),
    }
    for name, expected in expected_diagnostics.items():
        if not np.isclose(diagnostics.get(name, np.nan), expected, atol=1e-10, rtol=1e-10):
            raise ValueError(f"diagnostics.{name} does not match result arrays")
    try:
        return json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("coupled result must contain finite JSON values") from error


def load_coupled_result_payload(path: str | Path) -> dict[str, Any]:
    return validate_coupled_result_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def _validate_coupled_result(mechanism, result):
    if not isinstance(result, CoupledRigidBodyDynamicsResult):
        raise ValueError("result does not match this mechanism")
    if result.model_id != mechanism.model_id():
        raise ValueError("result model_id does not match this mechanism")
    steps = len(result.t)
    bodies = mechanism.nbody
    histories = {
        "center_of_mass_position": (steps, bodies, 3),
        "linear_velocity": (steps, bodies, 3),
        "angular_velocity_body": (steps, bodies, 3),
        "linear_acceleration": (steps, bodies, 3),
        "angular_acceleration_body": (steps, bodies, 3),
        "applied_force_world": (steps, bodies, 3),
        "applied_torque_body": (steps, bodies, 3),
        "interaction_force_world": (steps, result.interaction_force_world.shape[1], 3),
        "inertia_tensor_body": (bodies, 3, 3),
    }
    if np.asarray(result.t).shape != (steps,) or steps < 2 or not np.all(np.isfinite(result.t)):
        raise ValueError("coupled result time history is invalid")
    for name, shape in histories.items():
        values = np.asarray(getattr(result, name))
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"coupled result {name} is invalid")
    if len(result.poses) != steps or any(len(step) != bodies for step in result.poses):
        raise ValueError("coupled result pose history is invalid")
    for step in result.poses:
        for pose in step:
            if not isinstance(pose, Pose3D):
                raise ValueError("coupled result pose history is invalid")


def _pair_force_imbalance(interaction_forces):
    """Measure the net force after reconstructing both sides of every pair."""
    values = np.asarray(interaction_forces, dtype=float)
    if not values.size:
        return 0.0
    paired = np.stack((values, -values), axis=2)
    return float(np.max(np.abs(np.sum(paired, axis=(1, 2)))))


def _matrix3(value, count, name):
    if value is None:
        return np.zeros((count, 3))
    array = np.asarray(value, dtype=float)
    if array.shape != (count, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape ({count}, 3)")
    return np.array(array, copy=True)


def _relative_change(values):
    return float(np.max(np.abs(values - values[0])) / max(1.0, abs(float(values[0]))))


def _relative_vector_change(values):
    return float(np.max(np.linalg.norm(values - values[0], axis=1)) / max(1.0, float(np.linalg.norm(values[0]))))


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value
