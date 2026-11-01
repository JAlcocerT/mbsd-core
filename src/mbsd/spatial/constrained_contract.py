"""Strict portable validation for constrained spatial dynamics schemas v2 and v3.

Schema v3 adds parallel-axes and revolute-drive rows, results without
constraints (free bodies), and the ``drive_efforts`` block.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from .rotation import _inertia_tensor, _vector
from .vocabulary import Quaternion


V2_LAYOUT_KINDS = {"ground", "spherical", "fixed", "revolute", "drive"}
V3_LAYOUT_KINDS = V2_LAYOUT_KINDS | {"parallel_axes", "revolute_drive"}


def validate_constrained_payload_v2(
    payload: Mapping[str, Any], *, schema: str, capabilities: dict[str, Any],
    units: dict[str, str], conventions: dict[str, str], identity_scope: dict[str, str],
    schema_version: int = 2,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != schema or payload.get("schema_version") != schema_version:
        raise ValueError("unsupported constrained rigid-body result schema")
    if payload.get("status") != "experimental" or payload.get("solver_kind") != "constrained_spatial_rigid_body_dynamics":
        raise ValueError("invalid constrained solver identity")
    if payload.get("units") != units or payload.get("conventions") != conventions:
        raise ValueError("constrained result units or conventions are invalid")
    if payload.get("capabilities") != capabilities:
        raise ValueError("constrained result capabilities are invalid")
    if payload.get("identity_scope") != identity_scope:
        raise ValueError("constrained result identity scope is invalid")
    for name in ("mbsd_version", "model_id", "excitation_id", "simulation_id"):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise ValueError(f"{name} must be a non-empty string")

    provenance = _provenance(payload.get("provenance"), schema_version)
    times = np.asarray(payload.get("time"), dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("time must be finite and strictly increasing")
    layout = _layout(payload.get("constraint_layout"), schema_version)
    rows = len(layout)
    histories = _constraint_histories(payload, len(times), rows)
    bodies = _bodies(payload.get("bodies"), len(times))
    body_count = len(bodies)
    for descriptor in layout:
        for key in ("body_i", "body_j"):
            value = descriptor.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value not in range(body_count)):
                raise ValueError("constraint_layout body reference is invalid")

    jacobians = np.asarray(payload.get("constraint_jacobian"), dtype=float)
    if rows == 0 and jacobians.size == 0:
        jacobians = jacobians.reshape(len(times), 0, 6 * body_count)
    if jacobians.shape != (len(times), rows, 6 * body_count) or not np.all(np.isfinite(jacobians)):
        raise ValueError("constraint_jacobian has invalid shape or values")
    interactions = np.asarray(payload.get("interaction_force_world"), dtype=float)
    interaction_descriptors = provenance["interaction_descriptors"]
    if not interaction_descriptors and interactions.shape == (len(times), 0):
        interactions = interactions.reshape(len(times), 0, 3)
    if interactions.shape != (len(times), len(interaction_descriptors), 3) or not np.all(np.isfinite(interactions)):
        raise ValueError("interaction_force_world has invalid shape or values")

    reaction = np.stack([
        np.concatenate((body["arrays"]["reaction_force_world"], body["arrays"]["reaction_torque_body"]), axis=1)
        for body in bodies
    ], axis=1).reshape(len(times), -1)
    expected_reaction = -np.einsum("trc,tr->tc", jacobians, histories["multipliers"])
    if not np.allclose(reaction, expected_reaction, atol=1e-9, rtol=1e-9):
        raise ValueError("reaction histories do not match -J.T*lambda")

    excitation_payload = {
        "time": times.tolist(), "gravity_world": provenance["gravity_world"],
        "loads": provenance["load_descriptors"],
        "interactions": interaction_descriptors, "drives": provenance["drive_samples"],
        "applied_force_samples": np.stack([body["arrays"]["applied_force_world"] for body in bodies], axis=1).tolist(),
        "applied_torque_samples": np.stack([body["arrays"]["applied_torque_body"] for body in bodies], axis=1).tolist(),
        "interaction_force_samples": interactions.tolist(),
    }
    if payload["excitation_id"] != _identity(excitation_payload):
        raise ValueError("excitation_id does not match constrained excitation histories")
    simulation_payload = {
        "model_id": payload["model_id"], "excitation_id": payload["excitation_id"],
        "initial_center_of_mass_position": [body["arrays"]["center_of_mass_position"][0].tolist() for body in bodies],
        "initial_orientation_wxyz": [body["quaternions"][0] for body in bodies],
        "initial_linear_velocity_world": [body["arrays"]["linear_velocity"][0].tolist() for body in bodies],
        "initial_angular_velocity_body": [body["arrays"]["angular_velocity_body"][0].tolist() for body in bodies],
        "effective_inertia_tensor_body": [body["inertia"].tolist() for body in bodies],
        "time": times.tolist(), "method": provenance["method"],
        "rtol": float(provenance["rtol"]), "atol": float(provenance["atol"]),
        "constraint_tolerance": float(provenance["constraint_tolerance"]),
        "stabilization_frequency": float(provenance["stabilization_frequency"]),
        "sample_projection": provenance["sample_projection"],
    }
    if payload["simulation_id"] != _identity(simulation_payload):
        raise ValueError("simulation_id does not match constrained run configuration")
    if schema_version >= 3:
        _drive_efforts(payload.get("drive_efforts"), layout, histories["multipliers"], provenance["drive_samples"])

    status = payload.get("solver_status")
    if not isinstance(status, Mapping) or not isinstance(status.get("success"), bool) or not isinstance(status.get("message"), str):
        raise ValueError("solver_status is invalid")
    expected = _diagnostics(bodies, histories, jacobians, status["success"])
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("diagnostics must be a mapping")
    for name, value in expected.items():
        actual = diagnostics.get(name)
        if isinstance(value, bool):
            if actual is not value:
                raise ValueError(f"diagnostics.{name} is invalid")
        elif not isinstance(actual, (int, float)) or not np.isclose(actual, value, atol=1e-9, rtol=1e-9):
            raise ValueError(f"diagnostics.{name} does not match result arrays")
    try:
        return json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("constrained result must contain finite JSON values") from error


def _drive_efforts(value, layout, multipliers, drive_samples):
    drive_rows = [row for row in layout if row["kind"] in {"drive", "revolute_drive"}]
    if not isinstance(value, list) or len(value) != len(drive_rows):
        raise ValueError("drive_efforts must list every drive row")
    samples = {sample.get("name"): sample for sample in drive_samples if isinstance(sample, Mapping)}
    for item, row in zip(value, drive_rows):
        if not isinstance(item, Mapping) or item.get("name") != row["name"] or item.get("row") != row["row"]:
            raise ValueError("drive_efforts identity does not match constraint_layout")
        expected_units = "N*m" if row["kind"] == "revolute_drive" else "N"
        if item.get("kind") != row["kind"] or item.get("effort_units") != expected_units or item.get("power_units") != "W":
            raise ValueError("drive_efforts kind or units are invalid")
        effort = np.asarray(item.get("effort"), dtype=float)
        power = np.asarray(item.get("power"), dtype=float)
        if effort.shape != (multipliers.shape[0],) or not np.allclose(effort, -multipliers[:, row["row"]], atol=1e-12, rtol=1e-12):
            raise ValueError("drive_efforts effort does not match -lambda")
        velocity = np.asarray(samples.get(row["name"], {}).get("velocity"), dtype=float)
        if power.shape != effort.shape or velocity.shape != effort.shape or not np.allclose(power, effort * velocity, atol=1e-12, rtol=1e-12):
            raise ValueError("drive_efforts power does not match effort * drive velocity")


def _provenance(value, schema_version=2):
    required = {
        "solver", "method", "rtol", "atol", "constraint_tolerance",
        "stabilization_frequency", "sample_projection", "initial_position_projection",
        "multiplier_equation", "physical_reaction", "jacobian", "jacobian_rate",
        "gravity_world", "load_descriptors", "interaction_descriptors", "drive_samples",
    }
    if not isinstance(value, Mapping) or not required <= value.keys():
        raise ValueError("constrained result provenance is incomplete")
    if value["solver"] != "scipy.solve_ivp" or value["method"] not in {"RK45", "DOP853", "Radau"}:
        raise ValueError("constrained solver provenance is invalid")
    for name in ("rtol", "atol", "constraint_tolerance"):
        if not isinstance(value[name], (int, float)) or not np.isfinite(value[name]) or value[name] <= 0:
            raise ValueError(f"provenance.{name} must be positive and finite")
    if not isinstance(value["stabilization_frequency"], (int, float)) or not np.isfinite(value["stabilization_frequency"]) or value["stabilization_frequency"] < 0:
        raise ValueError("provenance.stabilization_frequency is invalid")
    if not isinstance(value["sample_projection"], bool) or (
        value["initial_position_projection"] is not True
        if schema_version < 3 else not isinstance(value["initial_position_projection"], bool)
    ):
        raise ValueError("constrained projection provenance is invalid")
    if value["multiplier_equation"] != "M*a + J.T*lambda = applied_generalized_force" or value["physical_reaction"] != "-J.T*lambda":
        raise ValueError("constrained reaction provenance is invalid")
    if value["jacobian_rate"] != "speed_scaled_central_richardson_directional_finite_difference":
        raise ValueError("constrained Jacobian-rate provenance is invalid")
    gravity = _vector(value["gravity_world"], "provenance.gravity_world").tolist()
    result = dict(value)
    result["gravity_world"] = gravity
    for name in ("load_descriptors", "interaction_descriptors", "drive_samples"):
        if not isinstance(value[name], list):
            raise ValueError(f"provenance.{name} must be a list")
    return result


def _layout(value, schema_version=2):
    if not isinstance(value, list) or (schema_version < 3 and not value):
        raise ValueError("constraint_layout must be a non-empty list")
    kinds = V3_LAYOUT_KINDS if schema_version >= 3 else V2_LAYOUT_KINDS
    for row, item in enumerate(value):
        if not isinstance(item, Mapping) or item.get("row") != row or item.get("kind") not in kinds:
            raise ValueError("constraint_layout contains an invalid row")
        component = item.get("component")
        if component not in {"x", "y", "z", "rx", "ry", "rz"} or not isinstance(item.get("name"), str) or not item["name"]:
            raise ValueError("constraint_layout identity is invalid")
        if item.get("id") != f"constraint:{item['kind']}:{item['name']}:{component}":
            raise ValueError("constraint_layout id is invalid")
        expected = ("m", "N") if component in {"x", "y", "z"} else ("rad", "N*m")
        if (item.get("residual_units"), item.get("multiplier_units")) != expected:
            raise ValueError("constraint_layout component units are inconsistent")
    return value


def _constraint_histories(payload, steps, rows):
    result = {}
    for name, shape in (
        ("multipliers", (steps, rows)), ("position_residuals", (rows, steps)),
        ("velocity_residuals", (rows, steps)), ("acceleration_residuals", (rows, steps)),
    ):
        values = np.asarray(payload.get(name), dtype=float)
        if rows == 0 and values.size == 0:
            values = values.reshape(shape)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} has invalid shape or values")
        result[name] = values
    return result


def _bodies(value, steps):
    if not isinstance(value, list) or not value:
        raise ValueError("bodies must be a non-empty list")
    result = []
    names = set()
    array_names = (
        "center_of_mass_position", "linear_velocity", "angular_velocity_body",
        "linear_acceleration", "angular_acceleration_body", "applied_force_world",
        "applied_torque_body", "reaction_force_world", "reaction_torque_body",
    )
    for index, body in enumerate(value):
        if not isinstance(body, Mapping) or body.get("index") != index or body.get("id") != f"body:{body.get('name')}":
            raise ValueError("body identity is invalid")
        name = body.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("body names must be non-empty and unique")
        names.add(name)
        mass = body.get("mass")
        if not isinstance(mass, (int, float)) or not np.isfinite(mass) or mass <= 0:
            raise ValueError("body mass must be positive and finite")
        center = _vector(body.get("center_of_mass_body"), "body.center_of_mass_body")
        inertia = _inertia_tensor(body.get("inertia_tensor_body"))
        arrays = {}
        for field in array_names:
            values = np.asarray(body.get(field), dtype=float)
            if values.shape != (steps, 3) or not np.all(np.isfinite(values)):
                raise ValueError(f"body.{field} has invalid shape or values")
            arrays[field] = values
        poses = body.get("poses")
        if not isinstance(poses, list) or len(poses) != steps:
            raise ValueError("body poses must match the time grid")
        quaternions = []
        for time_index, pose in enumerate(poses):
            if not isinstance(pose, Mapping) or not isinstance(pose.get("rotation"), Mapping):
                raise ValueError("body pose is invalid")
            translation = _vector(pose.get("translation"), "body pose translation")
            rotation = pose["rotation"]
            quaternion = np.asarray([rotation.get(key) for key in ("w", "x", "y", "z")], dtype=float)
            if not np.all(np.isfinite(quaternion)) or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-10, rtol=1e-10):
                raise ValueError("body pose quaternion norm must be 1")
            if not np.allclose(translation + Quaternion(*quaternion).to_rotation_matrix() @ center, arrays["center_of_mass_position"][time_index], atol=1e-10, rtol=1e-10):
                raise ValueError("body pose and center-of-mass position are inconsistent")
            quaternions.append(quaternion.tolist())
        force_error = float(mass) * arrays["linear_acceleration"] - arrays["applied_force_world"] - arrays["reaction_force_world"]
        if not np.allclose(force_error, 0.0, atol=1e-9, rtol=1e-9):
            raise ValueError("body translational force balance is invalid")
        torque_error = np.einsum("ij,tj->ti", inertia, arrays["angular_acceleration_body"]) + np.cross(arrays["angular_velocity_body"], np.einsum("ij,tj->ti", inertia, arrays["angular_velocity_body"])) - arrays["applied_torque_body"] - arrays["reaction_torque_body"]
        if not np.allclose(torque_error, 0.0, atol=1e-9, rtol=1e-9):
            raise ValueError("body rotational torque balance is invalid")
        result.append({"mass": float(mass), "inertia": inertia, "arrays": arrays, "quaternions": quaternions})
    return result


def _diagnostics(bodies, histories, jacobians, success):
    force_errors = []
    torque_errors = []
    for body in bodies:
        arrays = body["arrays"]
        force_errors.append(body["mass"] * arrays["linear_acceleration"] - arrays["applied_force_world"] - arrays["reaction_force_world"])
        torque_errors.append(np.einsum("ij,tj->ti", body["inertia"], arrays["angular_acceleration_body"]) + np.cross(arrays["angular_velocity_body"], np.einsum("ij,tj->ti", body["inertia"], arrays["angular_velocity_body"])) - arrays["applied_torque_body"] - arrays["reaction_torque_body"])
    rank = int(np.linalg.matrix_rank(jacobians[0])) if jacobians.shape[1] else 0
    return {
        "steps": histories["multipliers"].shape[0],
        "constraints": histories["multipliers"].shape[1],
        "jacobian_rank": rank,
        "degrees_of_freedom": 6 * len(bodies) - rank,
        "max_position_residual": _norm(histories["position_residuals"]),
        "max_velocity_residual": _norm(histories["velocity_residuals"]),
        "max_acceleration_residual": _norm(histories["acceleration_residuals"]),
        "max_force_balance_residual": _norm(force_errors),
        "max_torque_balance_residual": _norm(torque_errors),
        "finite": True, "success": success,
    }


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def _norm(value):
    array = np.asarray(value, dtype=float)
    return float(np.max(np.abs(array))) if array.size else 0.0
