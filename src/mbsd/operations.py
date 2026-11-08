"""Machine-readable discovery: capabilities and the operation manifest.

The manifest describes MBSD's public operations as plain data so any agent or
tool system can wrap them without importing a vendor framework.
``schemas/mbsd.operations-v1.json`` is a generated copy for non-Python
consumers; a test keeps it identical to ``operation_manifest()``.
"""

from __future__ import annotations

from typing import Any, Mapping

MANIFEST_SCHEMA_VERSION = 1

ERROR_CODES = {
    "solve_failed": "Generic singular, inconsistent, or divergent solve.",
    "integration_failed": "The time integrator reported failure.",
    "invalid_input": "An argument is malformed, non-finite, or out of range.",
    "unsupported_capability": "The request needs a capability this release does not provide.",
    "rank_deficient": "Constraints are redundant or singular; context names the rows and joints.",
    "inconsistent_initial_state": "Initial velocities violate the constraints; context names them.",
    "not_kinematically_driven": "Inverse dynamics needs drives for every degree of freedom.",
    "missing_multipliers": "The result carries no Lagrange multipliers (kinematic result).",
    "resource_limit": "max_steps or max_evaluations was exceeded.",
    "not_portable": "A Python callback cannot be serialized or imported.",
    "schema_mismatch": "The payload is not the expected schema family.",
    "unsupported_schema_version": "The payload schema version cannot be read.",
}

VALIDATION_CODES = {
    "empty_model": "error",
    "invalid_input": "error",
    "position_solve_failed": "error",
    "rank_deficient": "error",
    "redundant_constraints": "error",
    "overconstrained": "error",
    "inconsistent_initial_velocity": "error",
    "time_outside_table": "error",
    "resource_limit": "error",
    "underconstrained": "info",
    "opaque_constraint": "warning",
    "opaque_drive": "warning",
    "opaque_load": "warning",
    "low_level_joint": "warning",
}


def _op(name, call, summary, inputs, returns, errors=(), *, dimension=2, options=()):
    operation = {
        "name": name,
        "dimension": dimension,
        "call": call,
        "summary": summary,
        "inputs": inputs,
        "returns": returns,
        "errors": list(errors),
    }
    if options:
        operation["options"] = list(options)
    return operation


def _option(name, type_, default, meaning, *, units=None, minimum=None, enum=None):
    option = {"name": name, "type": type_, "default": default, "meaning": meaning}
    if units is not None:
        option["units"] = units
    if minimum is not None:
        option["minimum"] = minimum
    if enum is not None:
        option["enum"] = list(enum)
    return option


_GUARDS = (
    _option("max_steps", "int", 100_000, "Largest accepted number of output samples.", minimum=1),
    _option("max_evaluations", "int | None", None,
            "Cap on right-hand-side evaluations; None means no cap.", minimum=1),
)

PLANAR_SIMULATE_OPTIONS = (
    _option("method", "str", "RK45", "SciPy solve_ivp integrator.",
            enum=("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")),
    _option("rtol", "float", 1e-9, "Relative integration tolerance.", minimum=0.0),
    _option("atol", "float", 1e-11, "Absolute integration tolerance.", minimum=0.0),
    _option("allow_underconstrained", "bool", False,
            "Accept free degrees of freedom (needed for unforced dynamics)."),
    *_GUARDS,
)

SPATIAL_SIMULATE_OPTIONS = (
    _option("method", "str", "DOP853", "SciPy solve_ivp integrator.",
            enum=("RK45", "DOP853", "Radau")),
    _option("rtol", "float", 1e-9, "Relative integration tolerance.", minimum=0.0),
    _option("atol", "float", 1e-11, "Absolute integration tolerance.", minimum=0.0),
    _option("constraint_tolerance", "float", 1e-8,
            "Position and velocity projection tolerance; tighten it for drive-angle accuracy.",
            units="m or rad", minimum=0.0),
    _option("stabilization_frequency", "float", 0.0,
            "Baumgarte-style constraint stabilization frequency; 0 disables it.",
            units="rad/s", minimum=0.0),
    _option("project_samples", "bool", True,
            "Project every output sample back onto the position and velocity constraints."),
    _option("gravity_world", "list[float]", [0.0, 0.0, 0.0], "Gravity vector.", units="m/s^2"),
    *_GUARDS,
)


def operation_manifest() -> dict[str, Any]:
    """Return the framework-neutral operation manifest."""
    return {
        "schema": "mbsd.operations",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "workflow": ["build_or_load", "validate", "solve", "check", "export"],
        "operations": [
            _op(
                "capabilities",
                "mbsd.capabilities()",
                "Static description of supported dimensions, joints, drives, solvers and limits.",
                {},
                "dict",
                dimension=None,
            ),
            _op(
                "api_surface",
                "mbsd.api_surface()",
                "Every public symbol and builder method classified stable, experimental or "
                "deprecated, with the versioning, deprecation and schema policies.",
                {},
                "dict",
                dimension=None,
            ),
            _op(
                "load",
                "PlanarMechanism.from_json(path)",
                "Rebuild a planar mechanism from a portable mbsd.planar.mechanism payload.",
                {"path": "str"},
                "PlanarMechanism",
                ["schema_mismatch", "unsupported_schema_version", "not_portable"],
            ),
            _op(
                "validate",
                "mechanism.validate(q0=None, t=0.0)",
                "Pre-solve report: DOF, rank, portability, table domains, resource limits.",
                {"q0": "list[float] | None", "t": "float | list[float]"},
                "ValidationReport",
            ),
            _op(
                "solve_kinematics",
                "mechanism.solve_kinematics(t, q0=None, max_steps=100000)",
                "Position, velocity and acceleration of a fully driven mechanism.",
                {"t": "list[float]", "q0": "list[float] | None", "max_steps": "int"},
                "KinematicResult",
                ["solve_failed", "resource_limit"],
            ),
            _op(
                "simulate",
                "mechanism.simulate(t, q0, v0=None, max_steps=100000, max_evaluations=None)",
                "Forward constrained dynamics; the result carries sampled multipliers.",
                {
                    "t": "list[float]",
                    "q0": "list[float]",
                    "v0": "list[float] | None",
                    "allow_underconstrained": "bool",
                    "max_steps": "int",
                    "max_evaluations": "int | None",
                },
                "DynamicsResult",
                ["solve_failed", "integration_failed", "resource_limit"],
                options=PLANAR_SIMULATE_OPTIONS,
            ),
            _op(
                "inverse_dynamics",
                "mechanism.inverse_dynamics(t, q0=None, max_steps=100000)",
                "Multipliers that sustain a fully driven motion.",
                {"t": "list[float]", "q0": "list[float] | None", "max_steps": "int"},
                "InverseDynamicsResult",
                ["not_kinematically_driven", "solve_failed", "resource_limit"],
            ),
            _op(
                "joint_loads",
                "mechanism.joint_loads(result)",
                "Named joint reactions, drive efforts (incl. linear actuators and relative "
                "drives), gear and rack mesh efforts, and the frame load.",
                {"result": "DynamicsResult | InverseDynamicsResult"},
                "JointLoads",
                ["missing_multipliers"],
            ),
            _op(
                "check",
                "mechanism.check(result)",
                "Pass/fail verdict: residuals, energy drift or equation-of-motion balance.",
                {"result": "KinematicResult | DynamicsResult | InverseDynamicsResult"},
                "CheckReport",
            ),
            _op(
                "export_model",
                "mechanism.to_json(path)",
                "Write the portable mechanism payload (mbsd.planar.mechanism v3).",
                {"path": "str"},
                "path",
            ),
            _op(
                "export_result",
                "mechanism.result_to_json(result, path)",
                "Write the trajectory payload (mbsd.planar.result v1).",
                {"result": "result", "path": "str"},
                "path",
            ),
            _op(
                "export_joint_loads",
                "mechanism.joint_loads_to_json(result, path)",
                "Write the joint-load history payload (mbsd.planar.joint_loads v3).",
                {"result": "DynamicsResult | InverseDynamicsResult", "path": "str"},
                "path",
                ["missing_multipliers"],
            ),
            _op(
                "spatial_load",
                "SpatialMechanism.from_json(path)",
                "Rebuild a spatial mechanism, including law-described drives, from mbsd.spatial.model.",
                {"path": "str"},
                "SpatialMechanism",
                ["not_portable"],
                dimension=3,
            ),
            _op(
                "spatial_validate",
                "spatial_mechanism.validate(t=0.0, poses0=None, linear_velocity0=None, "
                "angular_velocity0_body=None)",
                "Pre-solve report: redundant constraints by joint name, DOF, initial-velocity "
                "consistency, portability, table domains, resource limits.",
                {
                    "t": "float | list[float]",
                    "linear_velocity0": "dict[body, vec3] | list[vec3] | None",
                    "angular_velocity0_body": "dict[body, vec3] | list[vec3] | None",
                },
                "ValidationReport",
                dimension=3,
            ),
            _op(
                "spatial_consistent_velocities",
                "spatial_mechanism.consistent_velocities(t=0.0, linear_velocity=None, "
                "angular_velocity_body=None)",
                "Complete partially prescribed initial velocities (minimum kinetic energy).",
                {
                    "t": "float",
                    "linear_velocity": "dict[body, vec3] | None",
                    "angular_velocity_body": "dict[body, vec3] | None",
                },
                "ConsistentVelocities",
                ["inconsistent_initial_state"],
                dimension=3,
            ),
            _op(
                "spatial_solve_kinematics",
                "spatial_mechanism.solve_kinematics(t, poses=None)",
                "Poses and velocities of a fully driven spatial mechanism.",
                {"t": "list[float]"},
                "SpatialKinematicResult",
                ["solve_failed"],
                dimension=3,
            ),
            _op(
                "spatial_simulate",
                "spatial_mechanism.simulate(t, linear_velocity0=None, angular_velocity0_body=None, "
                "loads=(), interactions=(), gravity_world=(0, 0, 0))",
                "Six-DOF rigid-body dynamics for free, coupled, jointed and driven bodies.",
                {
                    "t": "list[float]",
                    "linear_velocity0": "dict[body, vec3] | list[vec3] | None",
                    "angular_velocity0_body": "dict[body, vec3] | list[vec3] | None",
                },
                "ConstrainedRigidBodyDynamicsResult",
                [
                    "invalid_input", "rank_deficient", "inconsistent_initial_state",
                    "solve_failed", "resource_limit",
                ],
                dimension=3,
                options=SPATIAL_SIMULATE_OPTIONS,
            ),
            _op(
                "spatial_drive_efforts",
                "spatial_mechanism.drive_efforts(result)",
                "Per-drive force or torque and power histories.",
                {"result": "ConstrainedRigidBodyDynamicsResult"},
                "dict",
                dimension=3,
            ),
            _op(
                "spatial_check",
                "spatial_mechanism.check(result)",
                "Pass/fail verdict: residuals, force and torque balance, energy drift.",
                {"result": "ConstrainedRigidBodyDynamicsResult | SpatialKinematicResult"},
                "CheckReport",
                dimension=3,
            ),
            _op(
                "spatial_export_model",
                "spatial_mechanism.to_json(path)",
                "Write the spatial model payload (mbsd.spatial.model v4).",
                {"path": "str"},
                "path",
                dimension=3,
            ),
            _op(
                "spatial_export_result",
                "spatial_mechanism.constrained_result_to_json(result, path)",
                "Write the dynamics payload (mbsd.spatial.constrained_rigid_body_dynamic_result v3).",
                {"result": "ConstrainedRigidBodyDynamicsResult", "path": "str"},
                "path",
                dimension=3,
            ),
        ],
        "error_codes": dict(ERROR_CODES),
        "validation_codes": dict(VALIDATION_CODES),
        "safety": {
            "portable_payloads_execute_code": False,
            "callbacks": "python_only_and_reported_non_portable",
            "default_max_steps": 100_000,
        },
    }


def capabilities() -> dict[str, Any]:
    """Return planar and spatial capabilities as plain JSON-ready data."""
    from .planar.builder import planar_capabilities
    from .spatial import SpatialMechanism

    return {
        "schema": "mbsd.capabilities",
        "schema_version": 1,
        "planar": planar_capabilities(),
        "spatial": _plain(SpatialMechanism().capabilities),
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
