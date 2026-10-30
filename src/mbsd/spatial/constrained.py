"""Experimental constrained spatial rigid-body dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from ..errors import MechanismSolveError
from .coupled import RigidBodyPairInteraction
from .dynamics import _package_version
from .rotation import RigidBodyLoad, _inertia_tensor, _normalized_quaternion, _quaternion_product, _vector
from .vocabulary import Pose3D, Quaternion

if TYPE_CHECKING:
    from .builder import SpatialMechanism


CONSTRAINED_RESULT_SCHEMA = "mbsd.spatial.constrained_rigid_body_dynamic_result"
CONSTRAINED_RESULT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ConstrainedRigidBodyDynamicsResult:
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
    reaction_force_world: np.ndarray
    reaction_torque_body: np.ndarray
    multipliers: np.ndarray
    position_residuals: np.ndarray
    velocity_residuals: np.ndarray
    acceleration_residuals: np.ndarray
    constraint_jacobian: np.ndarray
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
            "applied_torque_body", "interaction_force_world", "reaction_force_world", "reaction_torque_body",
            "multipliers", "position_residuals", "velocity_residuals",
            "acceleration_residuals", "constraint_jacobian",
            "inertia_tensor_body",
        ):
            array = np.asarray(getattr(self, name), dtype=float)
            object.__setattr__(self, name, np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape))


@dataclass(frozen=True)
class ConstrainedDynamicsDiagnostics:
    steps: int
    constraints: int
    jacobian_rank: int
    degrees_of_freedom: int
    max_position_residual: float
    max_velocity_residual: float
    max_acceleration_residual: float
    max_force_balance_residual: float
    max_torque_balance_residual: float
    finite: bool
    success: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def constrained_capabilities() -> Mapping[str, Any]:
    return MappingProxyType({
        "solver_kind": "constrained_spatial_rigid_body_dynamics",
        "translation_dynamics": True, "orientation_dynamics": True,
        "multiple_bodies": True, "constraints": True,
        "joint_types": ("spherical", "fixed", "revolute"),
        "reaction_forces": True, "reaction_moments": True,
        "position_projection": True, "velocity_projection": True,
        "simulation_identity": True, "strict_result_validation": True,
        "contact": False,
        "angular_velocity_frame": "body", "inertia_frame": "body",
    })


def simulate_constrained_rigid_bodies(
    mechanism: SpatialMechanism,
    t: np.ndarray,
    *,
    poses0: Iterable[Pose3D] | None = None,
    linear_velocity0: np.ndarray | None = None,
    angular_velocity0_body: np.ndarray | None = None,
    loads: Iterable[RigidBodyLoad] = (),
    interactions: Iterable[RigidBodyPairInteraction] = (),
    gravity_world: Iterable[float] = (0.0, 0.0, 0.0),
    inertia_tensors_body: np.ndarray | None = None,
    stabilization_frequency: float = 0.0,
    project_samples: bool = True,
    constraint_tolerance: float = 1e-8,
    method: str = "DOP853",
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> ConstrainedRigidBodyDynamicsResult:
    """Integrate spherical, fixed, and revolute constrained rigid bodies."""
    if mechanism.nbody == 0 or not (mechanism._ground is not None or mechanism._joints or mechanism._drives):
        raise MechanismSolveError("constrained dynamics requires bodies and at least one constraint")
    times = np.asarray(t, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("t must be finite, strictly increasing, and contain at least two samples")
    for name, value, allow_zero in (("rtol", rtol, False), ("atol", atol, False), ("constraint_tolerance", constraint_tolerance, False), ("stabilization_frequency", stabilization_frequency, True)):
        if not np.isscalar(value) or not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'} and finite")
    if method not in {"RK45", "DOP853", "Radau"}:
        raise ValueError("method must be one of: RK45, DOP853, Radau")
    count = mechanism.nbody
    masses = np.asarray([body.mass for body in mechanism._bodies])
    gravity = _vector(gravity_world, "gravity_world")
    inertias = np.asarray([np.diag(body.inertia) for body in mechanism._bodies]) if inertia_tensors_body is None else np.asarray([_inertia_tensor(value) for value in np.asarray(inertia_tensors_body, dtype=float)])
    if inertias.shape != (count, 3, 3):
        raise ValueError(f"inertia_tensors_body must have shape ({count}, 3, 3)")
    load_values = tuple(loads)
    interaction_values = tuple(interactions)
    if any(not isinstance(load, RigidBodyLoad) or load.body >= count for load in load_values):
        raise ValueError("loads must reference bodies in this mechanism")
    if any(not isinstance(value, RigidBodyPairInteraction) or max(value.body_i, value.body_j) >= count for value in interaction_values):
        raise ValueError("interactions must reference bodies in this mechanism")
    authored = mechanism._initial_poses() if poses0 is None else mechanism._validate_poses(poses0)
    poses = mechanism.solve_position(authored, float(times[0]), tol=constraint_tolerance)
    com0 = np.asarray([poses[index].translation + poses[index].rotation.rotate(mechanism._bodies[index].center_of_mass) for index in range(count)])
    quaternion0 = np.asarray([pose.rotation.as_array() for pose in poses])
    velocity0 = _matrix3(linear_velocity0, count, "linear_velocity0")
    omega0 = _matrix3(angular_velocity0_body, count, "angular_velocity0_body")
    u0 = np.concatenate((velocity0, omega0), axis=1).reshape(-1)
    jacobian0 = _generalized_jacobian(mechanism, poses, float(times[0]))
    ct0 = mechanism.constraint_time_derivative(poses, float(times[0]))
    if linear_velocity0 is None and angular_velocity0_body is None:
        u0 = np.linalg.lstsq(jacobian0, -ct0, rcond=None)[0]
    elif _norm(jacobian0 @ u0 + ct0) > constraint_tolerance:
        raise MechanismSolveError("Initial velocity violates spatial constraints")
    rows = jacobian0.shape[0]
    rank = int(np.linalg.matrix_rank(jacobian0))
    if rank < rows:
        raise MechanismSolveError(f"Constrained dynamics Jacobian is rank-deficient: rank {rank} for {rows} rows")
    mass_matrix = np.zeros((6 * count, 6 * count))
    for index in range(count):
        mass_matrix[6*index:6*index+3, 6*index:6*index+3] = masses[index] * np.eye(3)
        mass_matrix[6*index+3:6*index+6, 6*index+3:6*index+6] = inertias[index]

    def unpack(state):
        blocks = state.reshape(count, 13)
        return blocks[:, :3], blocks[:, 3:7], blocks[:, 7:10], blocks[:, 10:13]

    def state_poses(com, quaternion):
        return tuple(Pose3D(com[index] - Quaternion(*quaternion[index]).rotate(mechanism._bodies[index].center_of_mass), Quaternion(*quaternion[index])) for index in range(count))

    def loads_at(ti, pose_values, velocity, omega):
        rotations = np.asarray([pose.rotation.to_rotation_matrix() for pose in pose_values])
        force = masses[:, None] * gravity
        torque = np.zeros((count, 3))
        for load in load_values:
            f, tau = load.evaluate(ti, pose_values[load.body], velocity[load.body], omega[load.body])
            force[load.body] += f
            torque[load.body] += tau
        interaction_forces = np.zeros((len(interaction_values), 3))
        for interaction_index, interaction in enumerate(interaction_values):
            value = interaction.evaluate(ti, pose_values, velocity, omega)
            interaction_forces[interaction_index] = value
            i, j = interaction.body_i, interaction.body_j
            force[i] += value
            force[j] -= value
            torque[i] += np.cross(interaction.point_i_from_com_body, rotations[i].T @ value)
            torque[j] += np.cross(interaction.point_j_from_com_body, rotations[j].T @ (-value))
        return force, torque, interaction_forces

    def acceleration(ti, state):
        com, quaternion, velocity, omega = unpack(state)
        quaternion = np.asarray([_normalized_quaternion(value) for value in quaternion])
        pose_values = state_poses(com, quaternion)
        u = np.concatenate((velocity, omega), axis=1).reshape(-1)
        jacobian = _generalized_jacobian(mechanism, pose_values, ti)
        ct = mechanism.constraint_time_derivative(pose_values, ti)
        bias = _constraint_bias(mechanism, pose_values, ti, velocity, omega)
        position = mechanism.constraint_residual(pose_values, ti)
        velocity_residual = jacobian @ u + ct
        omega_stab = float(stabilization_frequency)
        gamma = -bias - 2.0 * omega_stab * velocity_residual - omega_stab**2 * position
        force, torque, interaction_forces = loads_at(ti, pose_values, velocity, omega)
        generalized = np.concatenate((force, torque - np.cross(omega, np.einsum("bij,bj->bi", inertias, omega))), axis=1).reshape(-1)
        system = np.block([[mass_matrix, jacobian.T], [jacobian, np.zeros((rows, rows))]])
        try:
            solved = np.linalg.solve(system, np.concatenate((generalized, gamma)))
        except np.linalg.LinAlgError as error:
            raise MechanismSolveError("Constrained spatial acceleration solve is singular") from error
        reaction = -jacobian.T @ solved[6*count:]
        return solved[:6*count], solved[6*count:], force, torque, interaction_forces, reaction, position, velocity_residual, jacobian, bias

    def rhs(ti, state):
        _, quaternion, velocity, omega = unpack(state)
        generalized_acceleration, *_ = acceleration(ti, state)
        acceleration_blocks = generalized_acceleration.reshape(count, 6)
        derivative = np.zeros((count, 13))
        derivative[:, :3] = velocity
        derivative[:, 3:7] = np.asarray([0.5 * _quaternion_product(_normalized_quaternion(quaternion[index]), np.array([0.0, *omega[index]])) for index in range(count)])
        derivative[:, 7:10] = acceleration_blocks[:, :3]
        derivative[:, 10:13] = acceleration_blocks[:, 3:]
        return derivative.reshape(-1)

    initial = np.concatenate((com0, quaternion0, u0.reshape(count, 6)[:, :3], u0.reshape(count, 6)[:, 3:]), axis=1).reshape(-1)
    solution = solve_ivp(rhs, (float(times[0]), float(times[-1])), initial, t_eval=times, method=method, rtol=float(rtol), atol=float(atol))
    if not solution.success:
        raise MechanismSolveError(f"Constrained spatial dynamics failed: {solution.message}")
    states = solution.y.T.reshape(len(times), count, 13)
    if project_samples:
        for time_index, ti in enumerate(times):
            quaternion = np.asarray([_normalized_quaternion(value) for value in states[time_index, :, 3:7]])
            pose_values = state_poses(states[time_index, :, :3], quaternion)
            projected = mechanism.solve_position(pose_values, float(ti), tol=constraint_tolerance)
            states[time_index, :, :3] = np.asarray([projected[index].translation + projected[index].rotation.rotate(mechanism._bodies[index].center_of_mass) for index in range(count)])
            states[time_index, :, 3:7] = np.asarray([pose.rotation.as_array() for pose in projected])
            jacobian = _generalized_jacobian(mechanism, projected, float(ti))
            u = states[time_index, :, 7:13].reshape(-1)
            error = jacobian @ u + mechanism.constraint_time_derivative(projected, float(ti))
            u -= np.linalg.pinv(jacobian) @ error
            states[time_index, :, 7:13] = u.reshape(count, 6)
    shape = (len(times), count, 3)
    linear_acceleration = np.zeros(shape); angular_acceleration = np.zeros(shape)
    applied_force = np.zeros(shape); applied_torque = np.zeros(shape)
    interaction_force = np.zeros((len(times), len(interaction_values), 3))
    reaction_force = np.zeros(shape); reaction_torque = np.zeros(shape)
    multipliers = np.zeros((len(times), rows)); position_residuals = np.zeros((rows, len(times)))
    velocity_residuals = np.zeros_like(position_residuals); acceleration_residuals = np.zeros_like(position_residuals)
    jacobian_history = np.zeros((len(times), rows, 6 * count))
    pose_history: list[tuple[Pose3D, ...]] = []
    for time_index, ti in enumerate(times):
        state = states[time_index].reshape(-1)
        values = acceleration(float(ti), state)
        generalized_acceleration, multipliers[time_index], applied_force[time_index], applied_torque[time_index], interaction_force[time_index], reaction, position, velocity_residual, jacobian, bias = values
        blocks = generalized_acceleration.reshape(count, 6)
        linear_acceleration[time_index] = blocks[:, :3]; angular_acceleration[time_index] = blocks[:, 3:]
        reaction_blocks = reaction.reshape(count, 6)
        reaction_force[time_index] = reaction_blocks[:, :3]; reaction_torque[time_index] = reaction_blocks[:, 3:]
        position_residuals[:, time_index] = position; velocity_residuals[:, time_index] = velocity_residual
        acceleration_residuals[:, time_index] = jacobian @ generalized_acceleration + bias
        jacobian_history[time_index] = jacobian
        pose_history.append(state_poses(states[time_index, :, :3], states[time_index, :, 3:7]))
    drive_samples = _drive_samples(mechanism, times)
    excitation_payload = {
        "time": times.tolist(), "gravity_world": gravity.tolist(),
        "loads": [value.as_dict() for value in load_values],
        "interactions": [value.as_dict() for value in interaction_values],
        "drives": drive_samples,
        "applied_force_samples": applied_force.tolist(),
        "applied_torque_samples": applied_torque.tolist(),
        "interaction_force_samples": interaction_force.tolist(),
    }
    excitation_id = _identity(excitation_payload)
    simulation_payload = {
        "model_id": mechanism.model_id(), "excitation_id": excitation_id,
        "initial_center_of_mass_position": states[0, :, :3].tolist(),
        "initial_orientation_wxyz": states[0, :, 3:7].tolist(),
        "initial_linear_velocity_world": states[0, :, 7:10].tolist(),
        "initial_angular_velocity_body": states[0, :, 10:13].tolist(),
        "effective_inertia_tensor_body": inertias.tolist(),
        "time": times.tolist(), "method": method, "rtol": float(rtol),
        "atol": float(atol), "constraint_tolerance": float(constraint_tolerance),
        "stabilization_frequency": float(stabilization_frequency),
        "sample_projection": bool(project_samples),
    }
    return ConstrainedRigidBodyDynamicsResult(
        t=times, poses=tuple(pose_history), center_of_mass_position=states[:, :, :3],
        linear_velocity=states[:, :, 7:10], angular_velocity_body=states[:, :, 10:13],
        linear_acceleration=linear_acceleration, angular_acceleration_body=angular_acceleration,
        applied_force_world=applied_force, applied_torque_body=applied_torque,
        interaction_force_world=interaction_force,
        reaction_force_world=reaction_force, reaction_torque_body=reaction_torque,
        multipliers=multipliers, position_residuals=position_residuals,
        velocity_residuals=velocity_residuals, acceleration_residuals=acceleration_residuals,
        constraint_jacobian=jacobian_history,
        inertia_tensor_body=inertias,
        model_id=mechanism.model_id(),
        excitation_id=excitation_id, simulation_id=_identity(simulation_payload),
        success=True, message="converged",
        provenance=MappingProxyType({"solver": "scipy.solve_ivp", "method": method, "rtol": float(rtol), "atol": float(atol), "constraint_tolerance": float(constraint_tolerance), "stabilization_frequency": float(stabilization_frequency), "sample_projection": bool(project_samples), "initial_position_projection": True, "multiplier_equation": "M*a + J.T*lambda = applied_generalized_force", "physical_reaction": "-J.T*lambda", "jacobian": "analytic_position_transformed_to_com_body_velocity_basis", "jacobian_rate": "speed_scaled_central_richardson_directional_finite_difference", "gravity_world": tuple(gravity.tolist()), "load_descriptors": tuple(MappingProxyType(value.as_dict()) for value in load_values), "interaction_descriptors": tuple(MappingProxyType(value.as_dict()) for value in interaction_values), "drive_samples": tuple(MappingProxyType(value) for value in drive_samples)}),
        capabilities=constrained_capabilities(),
    )


def constrained_diagnostics(mechanism: SpatialMechanism, result: ConstrainedRigidBodyDynamicsResult) -> ConstrainedDynamicsDiagnostics:
    _validate_constrained_result(mechanism, result)
    pose0 = result.poses[0]
    jacobian = _generalized_jacobian(mechanism, pose0, float(result.t[0]))
    rank = int(np.linalg.matrix_rank(jacobian))
    masses = np.asarray([body.mass for body in mechanism._bodies])
    force_balance = masses[None, :, None] * result.linear_acceleration - result.applied_force_world - result.reaction_force_world
    gyroscopic = np.cross(
        result.angular_velocity_body,
        np.einsum("bij,tbj->tbi", result.inertia_tensor_body, result.angular_velocity_body),
    )
    torque_balance = (
        np.einsum("bij,tbj->tbi", result.inertia_tensor_body, result.angular_acceleration_body)
        + gyroscopic - result.applied_torque_body - result.reaction_torque_body
    )
    return ConstrainedDynamicsDiagnostics(
        steps=len(result.t), constraints=jacobian.shape[0], jacobian_rank=rank,
        degrees_of_freedom=6 * mechanism.nbody - rank,
        max_position_residual=_norm(result.position_residuals),
        max_velocity_residual=_norm(result.velocity_residuals),
        max_acceleration_residual=_norm(result.acceleration_residuals),
        max_force_balance_residual=_norm(force_balance),
        max_torque_balance_residual=_norm(torque_balance),
        finite=all(np.all(np.isfinite(value)) for value in (
            result.t, result.center_of_mass_position, result.linear_velocity,
            result.angular_velocity_body, result.linear_acceleration,
            result.angular_acceleration_body,
        )), success=result.success,
    )


def constrained_result_to_dict(mechanism: SpatialMechanism, result: ConstrainedRigidBodyDynamicsResult) -> dict[str, Any]:
    diagnostics = constrained_diagnostics(mechanism, result)
    layout = _constraint_layout(mechanism, result.poses[0])
    return {
        "schema": CONSTRAINED_RESULT_SCHEMA, "schema_version": CONSTRAINED_RESULT_SCHEMA_VERSION,
        "mbsd_version": _package_version(), "status": "experimental",
        "solver_kind": "constrained_spatial_rigid_body_dynamics",
        "model_id": result.model_id, "excitation_id": result.excitation_id,
        "simulation_id": result.simulation_id,
        "identity_scope": _identity_scope(),
        "units": _constrained_units(),
        "conventions": _constrained_conventions(),
        "capabilities": _json_ready(result.capabilities),
        "solver_status": {"success": result.success, "message": result.message},
        "provenance": _json_ready(result.provenance), "diagnostics": diagnostics.as_dict(),
        "time": result.t.tolist(), "constraint_layout": layout,
        "multipliers": result.multipliers.tolist(),
        "constraint_jacobian": result.constraint_jacobian.tolist(),
        "interaction_force_world": result.interaction_force_world.tolist(),
        "position_residuals": result.position_residuals.tolist(),
        "velocity_residuals": result.velocity_residuals.tolist(),
        "acceleration_residuals": result.acceleration_residuals.tolist(),
        "bodies": [{
            "index": index, "id": f"body:{body.name}", "name": body.name, "mass": body.mass,
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
            "reaction_force_world": result.reaction_force_world[:, index].tolist(),
            "reaction_torque_body": result.reaction_torque_body[:, index].tolist(),
        } for index, body in enumerate(mechanism._bodies)],
    }


def constrained_result_to_json(mechanism, result, path: str | Path, *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(constrained_result_to_dict(mechanism, result), indent=indent, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _validate_constrained_result_payload_v1_legacy(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != CONSTRAINED_RESULT_SCHEMA or payload.get("schema_version") != 1:
        raise ValueError("unsupported constrained rigid-body result schema")
    if payload.get("solver_kind") != "constrained_spatial_rigid_body_dynamics":
        raise ValueError("invalid constrained solver kind")
    for name in ("mbsd_version", "model_id", "excitation_id"):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise ValueError(f"{name} must be a non-empty string")
    times = np.asarray(payload.get("time"), dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("time must be finite and strictly increasing")
    layout = payload.get("constraint_layout")
    if not isinstance(layout, list):
        raise ValueError("constraint_layout must be a list")
    rows = len(layout)
    for row, descriptor in enumerate(layout):
        if not isinstance(descriptor, Mapping) or descriptor.get("row") != row or descriptor.get("kind") not in {"ground", "spherical", "fixed", "revolute", "drive"} or descriptor.get("component") not in {"x", "y", "z", "rx", "ry", "rz"}:
            raise ValueError("constraint_layout contains an invalid row")
        if descriptor.get("residual_units") not in {"m", "rad"} or descriptor.get("multiplier_units") not in {"N", "N*m"}:
            raise ValueError("constraint_layout row units are invalid")
    for name, shape in (("multipliers", (len(times), rows)), ("position_residuals", (rows, len(times))), ("velocity_residuals", (rows, len(times))), ("acceleration_residuals", (rows, len(times)))):
        values = np.asarray(payload.get(name), dtype=float)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} has invalid shape or values")
    bodies = payload.get("bodies")
    if not isinstance(bodies, list) or not bodies:
        raise ValueError("bodies must be a non-empty list")
    for index, body in enumerate(bodies):
        if not isinstance(body, Mapping) or body.get("index") != index:
            raise ValueError("body identity is invalid")
        for name in ("center_of_mass_position", "linear_velocity", "angular_velocity_body", "linear_acceleration", "angular_acceleration_body", "applied_force_world", "applied_torque_body", "reaction_force_world", "reaction_torque_body"):
            values = np.asarray(body.get(name), dtype=float)
            if values.shape != (len(times), 3) or not np.all(np.isfinite(values)):
                raise ValueError(f"body.{name} has invalid shape or values")
    try:
        return json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("constrained result must contain finite JSON values") from error


def validate_constrained_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload, Mapping) and payload.get("schema_version") == 1:
        return _validate_constrained_result_payload_v1_legacy(payload)
    from .constrained_contract import validate_constrained_payload_v2

    return validate_constrained_payload_v2(
        payload,
        schema=CONSTRAINED_RESULT_SCHEMA,
        capabilities=_json_ready(constrained_capabilities()),
        units=_constrained_units(),
        conventions=_constrained_conventions(),
        identity_scope=_identity_scope(),
    )


def load_constrained_result_payload(path: str | Path) -> dict[str, Any]:
    return validate_constrained_result_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def _constraint_layout(mechanism, poses):
    layout: list[dict[str, Any]] = []
    if mechanism._ground is not None:
        for component, units in zip(("x", "y", "z", "rx", "ry", "rz"), ("m", "m", "m", "rad", "rad", "rad")):
            layout.append(_layout_row(layout, "ground", mechanism._bodies[mechanism._ground].name, component, mechanism._ground, None, units))
    for joint in mechanism._joints:
        kind = joint.as_dict()["kind"]
        components = ("x", "y", "z") if kind == "spherical" else (("x", "y", "z", "rx", "ry", "rz") if kind == "fixed" else ("x", "y", "z", "rx", "ry"))
        for component in components:
            units = "m" if component in {"x", "y", "z"} else "rad"
            layout.append(_layout_row(layout, kind, joint.name, component, joint.body_i, joint.body_j, units))
    for drive in mechanism._drives:
        layout.append(_layout_row(layout, "drive", drive.name, drive.coordinate, drive.body, None, "m"))
    if len(layout) != len(mechanism.constraint_residual(poses)):
        raise ValueError("constraint layout does not match residual rows")
    return layout


def _layout_row(layout, kind, name, component, body_i, body_j, residual_units):
    return {"row": len(layout), "id": f"constraint:{kind}:{name}:{component}", "kind": kind, "name": name, "component": component, "body_i": body_i, "body_j": body_j, "residual_units": residual_units, "multiplier_units": "N" if residual_units == "m" else "N*m"}


def _generalized_jacobian(mechanism, poses, t):
    pose_jacobian = mechanism.constraint_jacobian(poses, t)
    transform = np.zeros((6 * mechanism.nbody, 6 * mechanism.nbody))
    for index, pose in enumerate(poses):
        rotation = pose.rotation.to_rotation_matrix()
        com_offset = rotation @ mechanism._bodies[index].center_of_mass
        block = np.block([[np.eye(3), _skew(com_offset) @ rotation], [np.zeros((3, 3)), rotation]])
        transform[6*index:6*index+6, 6*index:6*index+6] = block
    return pose_jacobian @ transform


def _constraint_bias(mechanism, poses, t, velocity, omega):
    rate = max(1.0, _norm(velocity), _norm(omega))
    step = min(1e-4, max(1e-8, 1e-5 / rate))

    def advanced_poses(signed_step):
        advanced = []
        for index, pose in enumerate(poses):
            rotation = pose.rotation.to_rotation_matrix()
            omega_world = rotation @ omega[index]
            angle = np.linalg.norm(omega_world) * signed_step
            increment = Quaternion.from_axis_angle(omega_world, angle) if np.linalg.norm(omega_world) else Quaternion.identity()
            com = pose.translation + rotation @ mechanism._bodies[index].center_of_mass
            new_rotation = increment.compose(pose.rotation)
            new_com = com + velocity[index] * signed_step
            advanced.append(Pose3D(new_com - new_rotation.rotate(mechanism._bodies[index].center_of_mass), new_rotation))
        return tuple(advanced)

    u = np.concatenate((velocity, omega), axis=1).reshape(-1)
    def central(value):
        plus = _generalized_jacobian(
            mechanism, advanced_poses(value), t + value
        )
        minus = _generalized_jacobian(
            mechanism, advanced_poses(-value), t - value
        )
        return ((plus - minus) / (2.0 * value)) @ u

    coarse = central(step)
    fine = central(step / 2.0)
    jdot_u = (4.0 * fine - coarse) / 3.0
    return jdot_u + mechanism.constraint_second_time_derivative(poses, t)


def _matrix3(value, count, name):
    if value is None:
        return np.zeros((count, 3))
    array = np.asarray(value, dtype=float)
    if array.shape != (count, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape ({count}, 3)")
    return np.array(array, copy=True)


def _validate_constrained_result(mechanism, result):
    if not isinstance(result, ConstrainedRigidBodyDynamicsResult):
        raise TypeError("result must be a ConstrainedRigidBodyDynamicsResult")
    if result.model_id != mechanism.model_id():
        raise ValueError("result model_id does not match this mechanism")
    steps, bodies = len(result.t), mechanism.nbody
    rows = result.multipliers.shape[1] if result.multipliers.ndim == 2 else -1
    shapes = {
        "center_of_mass_position": (steps, bodies, 3),
        "linear_velocity": (steps, bodies, 3),
        "angular_velocity_body": (steps, bodies, 3),
        "linear_acceleration": (steps, bodies, 3),
        "angular_acceleration_body": (steps, bodies, 3),
        "applied_force_world": (steps, bodies, 3),
        "applied_torque_body": (steps, bodies, 3),
        "reaction_force_world": (steps, bodies, 3),
        "reaction_torque_body": (steps, bodies, 3),
        "multipliers": (steps, rows),
        "position_residuals": (rows, steps),
        "velocity_residuals": (rows, steps),
        "acceleration_residuals": (rows, steps),
        "constraint_jacobian": (steps, rows, 6 * bodies),
        "inertia_tensor_body": (bodies, 3, 3),
    }
    if result.interaction_force_world.ndim != 3:
        raise ValueError("constrained result interaction history is invalid")
    shapes["interaction_force_world"] = (steps, result.interaction_force_world.shape[1], 3)
    if np.asarray(result.t).shape != (steps,) or steps < 2 or not np.all(np.isfinite(result.t)) or np.any(np.diff(result.t) <= 0):
        raise ValueError("constrained result time history is invalid")
    for name, shape in shapes.items():
        values = np.asarray(getattr(result, name))
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"constrained result {name} is invalid")
    if len(result.poses) != steps or any(len(step) != bodies for step in result.poses):
        raise ValueError("constrained result pose history is invalid")
    if any(not isinstance(pose, Pose3D) for step in result.poses for pose in step):
        raise ValueError("constrained result pose history is invalid")


def _drive_samples(mechanism, times):
    return [{
        "name": drive.name, "body": drive.body, "coordinate": drive.coordinate,
        "value": [float(drive.value(float(time))) for time in times],
        "velocity": [float(drive.velocity(float(time))) for time in times],
        "acceleration": [float(drive.acceleration_value(float(time))) for time in times],
    } for drive in mechanism._drives]


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def _constrained_units():
    return {"length": "m", "angle": "rad", "time": "s", "linear_velocity": "m/s", "angular_velocity": "rad/s", "linear_acceleration": "m/s^2", "angular_acceleration": "rad/s^2", "force": "N", "torque": "N*m", "inertia": "kg*m^2"}


def _constrained_conventions():
    return {"world_frame": "right_handed_xyz", "orientation": "active_body_to_world_quaternion_wxyz", "linear_state": "center_of_mass_world", "angular_velocity_frame": "body", "reaction_force_frame": "world", "reaction_torque_frame": "body", "multiplier_equation": "M*a + J.T*lambda = applied_generalized_force", "physical_reaction": "-J.T*lambda"}


def _identity_scope():
    return {
        "model_id": "mechanism topology, body properties, authored poses, and constraint descriptors",
        "excitation_id": "time grid, gravity, sampled drives, loads, interactions, and sampled applied loads",
        "simulation_id": "model and excitation identities, projected initial state, effective inertia, time grid, and solver controls",
    }


def _skew(value):
    x, y, z = value
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _norm(value):
    array = np.asarray(value)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value
