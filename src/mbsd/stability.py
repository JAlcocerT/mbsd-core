"""The public API surface and its stability classification.

Every public symbol of every public namespace, and every public method of the
two builders, is classified here as ``stable``, ``experimental`` or
``deprecated``. Anything not listed (modules such as ``mbsd.planar.solver``,
names starting with ``_``) is ``internal`` and carries no promise.

- **stable**: covered by semantic versioning from 1.0; a breaking change needs
  a major release.
- **experimental**: importable and tested, but may change in a minor release.
  Experimental symbols live in ``mbsd.experimental`` (or are documented
  experimental builder methods).
- **deprecated**: still works and warns; removed in the release named in
  ``removal``.

``schemas/mbsd.api-surface-v1.json`` is a generated copy of
:func:`api_surface`; a test keeps it identical and checks this registry
against the real ``__all__`` lists and builder classes, so the frozen surface
cannot drift silently.
"""

from __future__ import annotations

from typing import Any

API_SURFACE_SCHEMA_VERSION = 1
STATUSES = ("stable", "experimental", "deprecated")
REMOVAL = "1.0.0"

_STABLE_TOP = (
    "ConstantSpeed", "Mechanism", "MechanismSolveError", "NotPortableError", "Polynomial",
    "Spring", "Tabulated", "api_surface", "capabilities", "operation_manifest",
)

_STABLE_PLANAR = (
    "AffineFit", "BodyHandle", "Check", "CheckReport", "ConfigurationDiagnostics",
    "ConstantSpeed", "DynamicsResult", "FourBar", "FourBarPose", "InverseDynamicsResult",
    "Issue", "JointLoads", "KinematicResult", "Mechanism", "MechanismSolveError",
    "ModelDiagnostics", "MotionLaw", "PlanarMechanism", "Polynomial", "ResultDiagnostics",
    "Spring", "Tabulated", "ValidationReport", "analysis", "joint_loads_to_dict",
    "joint_loads_to_json", "mechanism_from_dict", "mechanism_from_json", "mechanism_to_dict",
    "mechanism_to_json", "point_trace_to_csv", "point_trace_to_dict", "point_trace_to_json",
    "result_to_csv", "result_to_dict", "result_to_json",
)

_STABLE_ANALYSIS = (
    "ConstraintGroup", "acceleration_residual", "body_vector", "constraint_groups",
    "constraint_jacobian", "constraint_residual", "constraint_second_time_derivative",
    "constraint_time_derivative", "equation_of_motion_residual", "generalized_forces",
    "jacobian_rate", "mass_matrix", "point_acceleration", "point_position", "point_velocity",
    "rotation_matrix",
)

_STABLE_REPORTS = ("Check", "CheckReport", "Issue", "ValidationReport")
_STABLE_ERRORS = ("MechanismSolveError", "NotPortableError")

_STABLE_SPATIAL = (
    # vocabulary and portable model
    "FixedJoint3D", "Frame3D", "ParallelAxesJoint3D", "Pose3D", "Quaternion", "RevoluteJoint3D",
    "SpatialBody", "SpatialModel", "SphericalJoint3D", "SPATIAL_MODEL_SCHEMA",
    "SPATIAL_MODEL_SCHEMA_VERSION", "SUPPORTED_SPATIAL_MODEL_SCHEMA_VERSIONS",
    "load_spatial_model_payload", "spatial_model_from_payload", "validate_spatial_model_payload",
    # builder, kinematics and the one simulate()
    "ConsistentVelocities", "SpatialBodyHandle", "SpatialCoordinateDrive", "SpatialFrameHandle",
    "SpatialKinematicResult", "SpatialMechanism", "SpatialModelDiagnostics",
    "SpatialResultDiagnostics", "SpatialRevoluteDrive", "load_spatial_kinematic_result_payload",
    "validate_spatial_kinematic_result_payload", "RigidBodyLoad", "RigidBodyPairInteraction",
    "ConstrainedDynamicsDiagnostics", "ConstrainedRigidBodyDynamicsResult",
    "CONSTRAINED_RESULT_SCHEMA", "CONSTRAINED_RESULT_SCHEMA_VERSION",
    "SUPPORTED_CONSTRAINED_RESULT_SCHEMA_VERSIONS", "constrained_capabilities",
    "load_constrained_result_payload", "validate_constrained_result_payload",
    # transforms and point kinematics
    "SpatialResidualSummary", "compose_pose", "fixed_joint_descriptor_residual", "inverse_pose",
    "joint_residual_jacobian", "max_spatial_residual", "point_position", "point_velocity",
    "quaternion_rate_world", "resolve_frame_pose", "spherical_joint_descriptor_residual",
    "spherical_joint_residual", "spatial_residual_summary",
    # engine-neutral handoff
    "HANDOFF_SCHEMA", "HANDOFF_SCHEMA_VERSION", "build_transform_handoff",
    "transform_handoff_to_json", "validate_transform_handoff",
)

# Fixed-orientation translational dynamics: kept, but not part of the 1.0
# stable claim. Importable from ``mbsd.experimental``.
EXPERIMENTAL_SPATIAL = (
    "DYNAMIC_RESULT_SCHEMA", "DYNAMIC_RESULT_SCHEMA_VERSION", "SpatialDynamicsDiagnostics",
    "SpatialDynamicsResult", "SpatialForce", "constraint_groups",
    "load_spatial_dynamic_result_payload", "validate_spatial_dynamic_result_payload",
)

# Result types of the pre-0.9.7 free-body and coupled entry points, which
# simulate() replaced. Readers stay until 1.0 so old payloads can be migrated.
DEPRECATED_SPATIAL = (
    "COUPLED_RESULT_SCHEMA", "COUPLED_RESULT_SCHEMA_VERSION", "CoupledDynamicsDiagnostics",
    "CoupledRigidBodyDynamicsResult", "RIGID_BODY_RESULT_SCHEMA",
    "RIGID_BODY_RESULT_SCHEMA_VERSION", "RigidBodyDynamicsDiagnostics",
    "SpatialRigidBodyDynamicsResult", "coupled_capabilities", "free_body_capabilities",
    "load_coupled_result_payload", "load_rigid_body_result_payload",
    "validate_coupled_result_payload", "validate_rigid_body_result_payload",
)

_PLANAR_METHODS_STABLE = (
    "assert_constraints_satisfied", "body", "capabilities", "check", "configuration_diagnostics",
    "constant_force", "constant_torque", "constraint_residual", "constraint_residuals",
    "coordinate_drive", "diagnostics", "distance_drive", "force", "from_dict", "from_json",
    "gear", "ground", "inverse_dynamics", "joint_loads", "joint_loads_to_dict",
    "joint_loads_to_json", "max_constraint_residual", "model", "model_diagnostics", "motor",
    "ncoord", "nrestr", "pin", "point_on_line", "point_trace_to_csv", "point_trace_to_dict",
    "point_trace_to_json", "rack", "result_to_csv", "result_to_dict", "result_to_json",
    "loads", "simulate", "slider", "solve_kinematics", "solve_position", "solve_velocity",
    "spring", "springs", "tabulated_force", "tabulated_torque", "to_dict", "to_json", "validate",
)

_SPATIAL_METHODS_STABLE = (
    "body", "capabilities", "check", "consistent_velocities", "constrained_diagnostics",
    "constrained_result_to_dict", "constrained_result_to_json", "constraint_jacobian",
    "constraint_layout", "constraint_residual", "constraint_second_time_derivative",
    "constraint_time_derivative", "coordinate_drive", "diagnostics", "drive_efforts",
    "excitation_id", "fixed", "frame", "from_dict", "from_json", "from_spatial_model", "ground",
    "model_diagnostics", "model_id", "nbody", "ncoord", "pair_interaction", "parallel_axes",
    "portable", "redundant_constraints", "result_diagnostics", "result_to_dict",
    "result_to_json", "revolute", "revolute_drive", "rigid_body_load", "simulate",
    "solve_kinematics", "solve_position", "solve_velocity", "spherical", "to_dict", "to_json",
    "to_spatial_model", "validate",
)
_SPATIAL_METHODS_EXPERIMENTAL = (
    "dynamics_constraint_groups", "dynamics_diagnostics", "dynamics_result_to_dict",
    "dynamics_result_to_json", "force", "simulate_fixed_orientation",
)
_SPATIAL_METHODS_DEPRECATED = (
    "coupled_diagnostics", "coupled_result_to_dict", "coupled_result_to_json",
    "rigid_body_diagnostics", "rigid_body_result_to_dict", "rigid_body_result_to_json",
    "simulate_constrained_rigid_bodies", "simulate_coupled_rigid_bodies", "simulate_free_body",
)


def _entries(names, status, **extra):
    return {name: {"status": status, **extra} for name in names}


def _deprecated(names, replacement):
    return _entries(names, "deprecated", removal=REMOVAL, replacement=replacement)


SURFACE: dict[str, dict[str, dict[str, str]]] = {
    "mbsd": _entries(_STABLE_TOP, "stable"),
    "mbsd.errors": _entries(_STABLE_ERRORS, "stable"),
    "mbsd.reports": _entries(_STABLE_REPORTS, "stable"),
    "mbsd.planar": _entries(_STABLE_PLANAR, "stable"),
    "mbsd.planar.analysis": _entries(_STABLE_ANALYSIS, "stable"),
    "mbsd.spatial": {
        **_entries(_STABLE_SPATIAL, "stable"),
        **_deprecated(DEPRECATED_SPATIAL, "SpatialMechanism.simulate() and its constrained result"),
    },
    "mbsd.experimental": _entries(EXPERIMENTAL_SPATIAL, "experimental"),
    "mbsd.planar.PlanarMechanism": _entries(_PLANAR_METHODS_STABLE, "stable"),
    "mbsd.spatial.SpatialMechanism": {
        **_entries(_SPATIAL_METHODS_STABLE, "stable"),
        **_entries(_SPATIAL_METHODS_EXPERIMENTAL, "experimental"),
        **_deprecated(_SPATIAL_METHODS_DEPRECATED, "simulate(), constrained_result_to_dict()"),
    },
}

POLICIES = {
    "versioning": "semver_from_1.0: stable breaking changes need a major release",
    "deprecation": "deprecated symbols warn (DeprecationWarning) and are removed no earlier "
                   "than the next major release; 0.x deprecations are removed in 1.0.0",
    "experimental": "experimental symbols may change in a minor release and are excluded "
                    "from the stable claim",
    "schemas": "schema versions are independent of the package version; changes are "
               "additive within a version, anything else is a new version; readers reject "
               "unknown versions; every emitted version stays readable for at least one major "
               "release",
    "internal": "modules and names not listed here carry no promise",
}


def status(namespace: str, name: str) -> str:
    """Return ``stable``, ``experimental``, ``deprecated`` or ``internal``."""
    entry = SURFACE.get(namespace, {}).get(name)
    return "internal" if entry is None else entry["status"]


def api_surface() -> dict[str, Any]:
    """Return the classified public API surface as JSON-ready data."""
    return {
        "schema": "mbsd.api_surface",
        "schema_version": API_SURFACE_SCHEMA_VERSION,
        "statuses": list(STATUSES),
        "policies": dict(POLICIES),
        "namespaces": {
            namespace: {name: dict(entry) for name, entry in sorted(entries.items())}
            for namespace, entries in SURFACE.items()
        },
    }
