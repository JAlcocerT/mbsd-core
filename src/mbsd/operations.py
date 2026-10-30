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
    "not_kinematically_driven": "Inverse dynamics needs drives for every degree of freedom.",
    "missing_multipliers": "The result carries no Lagrange multipliers (kinematic result).",
    "resource_limit": "max_steps or max_evaluations was exceeded.",
    "not_portable": "A Python callback cannot be serialized or imported.",
    "schema_mismatch": "The payload is not the expected schema family.",
    "unsupported_schema_version": "The payload schema version cannot be read.",
}

VALIDATION_CODES = {
    "empty_model": "error",
    "offset_center_of_mass": "error",
    "position_solve_failed": "error",
    "rank_deficient": "error",
    "overconstrained": "error",
    "time_outside_table": "error",
    "resource_limit": "error",
    "underconstrained": "info",
    "opaque_constraint": "warning",
    "opaque_drive": "warning",
    "opaque_load": "warning",
    "low_level_joint": "warning",
}


def _op(name, call, summary, inputs, returns, errors=()):
    return {
        "name": name,
        "call": call,
        "summary": summary,
        "inputs": inputs,
        "returns": returns,
        "errors": list(errors),
    }


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
                "Static description of supported dimensions, joints, solvers and limits.",
                {},
                "dict",
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
                "Named joint reactions, drive efforts and the frame (shaking) load.",
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
                "Write the portable mechanism payload (mbsd.planar.mechanism v1).",
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
                "Write the joint-load history payload (mbsd.planar.joint_loads v1).",
                {"result": "DynamicsResult | InverseDynamicsResult", "path": "str"},
                "path",
                ["missing_multipliers"],
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
