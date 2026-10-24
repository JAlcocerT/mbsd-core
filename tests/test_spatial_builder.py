import json

import numpy as np
import pytest

from mbsd import Mechanism, MechanismSolveError
from mbsd.spatial import (
    Pose3D,
    Quaternion,
    SpatialMechanism,
    load_spatial_dynamic_result_payload,
    load_spatial_kinematic_result_payload,
    validate_spatial_kinematic_result_payload,
)


def test_spatial_factory_handles_and_portable_model_conversion():
    mechanism = Mechanism.spatial()
    assert isinstance(mechanism, SpatialMechanism)
    ground = mechanism.ground()
    link = mechanism.body("link", pose=Pose3D(translation=(0.0, -1.0, 0.0)))
    tip = mechanism.frame(
        "tip", parent_body=link, pose=Pose3D(translation=(0.0, 1.0, 0.0))
    )
    mechanism.spherical(ground, link, point_j=(0.0, 1.0, 0.0), name="pivot")

    portable = mechanism.to_spatial_model()
    restored = SpatialMechanism.from_spatial_model(portable)

    assert int(ground) == 0
    assert int(link) == 1
    assert int(tip) == 0
    assert [body.name for body in portable.bodies] == ["ground", "link"]
    assert portable.frames[0].parent_body == 1
    assert portable.joints[0].name == "pivot"
    assert restored.to_spatial_model().as_dict() == portable.as_dict()


def test_portable_model_conversion_rejects_nonportable_drive_callbacks():
    mechanism = Mechanism.spatial()
    body = mechanism.ground()
    mechanism.coordinate_drive(
        body, "x", value=lambda _t: 0.0, velocity=lambda _t: 0.0
    )

    with pytest.raises(ValueError, match="cannot reconstruct Python callbacks"):
        SpatialMechanism.from_spatial_model(mechanism.to_spatial_model())


def test_spherical_pendulum_sequence_meets_position_and_velocity_tolerances():
    mechanism = Mechanism.spatial()
    ground = mechanism.ground()
    link = mechanism.body("pendulum", pose=Pose3D(translation=(0.0, -1.0, 0.0)))
    mechanism.spherical(ground, link, point_j=(0.0, 1.0, 0.0), name="pivot")
    mechanism.coordinate_drive(
        link,
        "x",
        value=lambda t: 0.2 * t,
        velocity=lambda _t: 0.2,
    )

    result = mechanism.solve_kinematics(np.linspace(0.0, 0.5, 6), tol=1e-8)
    diagnostics = mechanism.result_diagnostics(result)

    assert diagnostics.max_position_residual < 1e-8
    assert diagnostics.max_velocity_residual < 1e-8
    assert diagnostics.max_translation_residual < 1e-8
    assert diagnostics.max_rotation_residual < 1e-8
    assert diagnostics.max_linear_velocity_residual < 1e-8
    assert diagnostics.max_angular_velocity_residual < 1e-8
    assert diagnostics.max_trajectory_linear_velocity_error < 1e-5
    assert diagnostics.max_trajectory_angular_velocity_error < 2e-5
    assert diagnostics.trajectory_velocity_checked
    assert diagnostics.finite
    assert diagnostics.success
    np.testing.assert_allclose(
        [pose.translation[0] for pose in result.body_poses(link)],
        0.2 * result.t,
        atol=1e-8,
    )
    assert result.provenance["pose_increment_frame"] == "world"
    assert result.capabilities["spatial_dynamics"] is False


def test_two_body_spherical_joint_rank_matches_hand_derived_dof():
    mechanism = Mechanism.spatial()
    ground = mechanism.ground()
    first = mechanism.body("first")
    second = mechanism.body("second")
    mechanism.fixed(ground, first, name="first-fixed")
    mechanism.spherical(first, second, name="coupling")

    diagnostics = mechanism.model_diagnostics()

    assert diagnostics.coordinates == 18
    assert diagnostics.constraints == 15
    assert diagnostics.jacobian_rank == 15
    assert diagnostics.degrees_of_freedom == 3
    assert diagnostics.classification == "underconstrained"


def test_fixed_attachment_is_fully_constrained():
    mechanism = Mechanism.spatial()
    ground = mechanism.ground()
    link = mechanism.body(
        "fixed-link",
        pose=Pose3D(
            translation=(0.3, -0.2, 0.5),
            rotation=Quaternion.from_axis_angle((1.0, 1.0, 0.0), 0.4),
        ),
    )
    mechanism.fixed(
        ground,
        link,
        frame_i=Pose3D(
            translation=(0.3, -0.2, 0.5),
            rotation=Quaternion.from_axis_angle((1.0, 1.0, 0.0), 0.4),
        ),
        name="attachment",
    )

    solved = mechanism.solve_position(tol=1e-9)
    diagnostics = mechanism.model_diagnostics(solved)

    assert diagnostics.classification == "fully_constrained"
    assert diagnostics.degrees_of_freedom == 0
    assert np.max(np.abs(mechanism.constraint_residual(solved))) < 1e-9


def test_duplicate_spatial_constraints_report_rank_deficiency():
    mechanism = Mechanism.spatial()
    ground = mechanism.ground()
    link = mechanism.body("link")
    mechanism.spherical(ground, link, name="pivot-a")
    mechanism.spherical(ground, link, name="pivot-b")

    diagnostics = mechanism.model_diagnostics()

    assert diagnostics.rank_deficient
    assert diagnostics.classification == "rank_deficient"
    assert diagnostics.jacobian_rank == 9
    assert diagnostics.constraints == 12


def test_inconsistent_spatial_constraints_fail_actionably():
    mechanism = Mechanism.spatial()
    ground = mechanism.ground()
    mechanism.coordinate_drive(
        ground,
        "x",
        value=lambda _t: 1.0,
        velocity=lambda _t: 0.0,
    )

    with pytest.raises(MechanismSolveError, match="Spatial position solve failed.*residual"):
        mechanism.solve_position(tol=1e-10)


def test_inconsistent_spatial_velocity_constraints_fail_actionably():
    mechanism = Mechanism.spatial()
    ground = mechanism.ground()
    link = mechanism.body("link")
    mechanism.fixed(ground, link, name="attachment")
    mechanism.coordinate_drive(
        link, "x", value=lambda t: t, velocity=lambda _t: 1.0
    )
    mechanism.coordinate_drive(
        link, "x", value=lambda t: -t, velocity=lambda _t: -1.0
    )
    poses = mechanism.solve_position(tol=1e-9)

    with pytest.raises(MechanismSolveError, match="Spatial velocity solve failed"):
        mechanism.solve_velocity(poses, tol=1e-9)


def test_spatial_drive_rejects_velocity_inconsistent_with_value_derivative():
    mechanism = Mechanism.spatial()
    body = mechanism.body("body")
    mechanism.coordinate_drive(
        body, "x", value=lambda t: 0.2 * t, velocity=lambda _t: 0.0
    )
    poses = mechanism.solve_position(t=0.5)

    with pytest.raises(ValueError, match="velocity is inconsistent"):
        mechanism.solve_velocity(poses, t=0.5)


@pytest.mark.parametrize(
    ("frequency", "time"),
    [(1000.0, 0.0), (1.0, 1.0e9), (10.0, 1.0e8)],
)
def test_spatial_drive_derivative_guard_accepts_scaled_exact_laws(frequency, time):
    mechanism = Mechanism.spatial()
    body = mechanism.body("body")
    drive = mechanism.coordinate_drive(
        body,
        "x",
        value=lambda t: np.sin(frequency * t),
        velocity=lambda t: frequency * np.cos(frequency * t),
    )

    assert drive.validate_derivative(time) == "validated"


def test_spatial_result_arrays_are_read_only_and_validation_is_comprehensive():
    mechanism = Mechanism.spatial()
    ground = mechanism.ground()
    link = mechanism.body("link")
    mechanism.fixed(ground, link)
    result = mechanism.solve_kinematics(np.array([0.0, 0.1, 0.2]))

    with pytest.raises(ValueError, match="read-only"):
        result.residuals[0, 0] = np.nan
    with pytest.raises(ValueError, match="read-only"):
        result.t[0] = np.nan
    with pytest.raises(ValueError, match="read-only"):
        result.linear_velocity[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="read-only"):
        result.poses[0][0].translation[0] = np.nan

    diagnostics = mechanism.result_diagnostics(result)
    assert diagnostics.finite
    assert diagnostics.success


def test_fixed_orientation_dynamics_matches_constant_force_solution(tmp_path):
    mechanism = Mechanism.spatial()
    body = mechanism.body("mass", mass=2.0)
    force = mechanism.force(body, (2.0, 0.0, 0.0), name="thrust")
    times = np.linspace(0.0, 1.0, 21)

    result = mechanism.simulate_fixed_orientation(times, forces=(force,))
    diagnostics = mechanism.dynamics_diagnostics(result)
    payload = mechanism.dynamics_result_to_dict(result)
    path = mechanism.dynamics_result_to_json(result, tmp_path / "dynamic-result.json")

    np.testing.assert_allclose(
        [pose.translation[0] for pose in result.body_poses(body)],
        0.5 * times**2,
        atol=2e-9,
    )
    np.testing.assert_allclose(result.body_linear_velocity(body)[:, 0], times, atol=2e-9)
    assert diagnostics.max_force_balance_residual_n < 1e-12
    assert payload["solver_kind"] == "fixed_orientation_translational_dynamics"
    assert payload["capabilities"]["orientation_dynamics"] is False
    assert path.exists()
    assert load_spatial_dynamic_result_payload(path)["model_id"] == mechanism.model_id()


def test_fixed_orientation_dynamics_enforces_acceleration_drive_and_reaction():
    mechanism = Mechanism.spatial()
    body = mechanism.body("driven", mass=2.0)
    mechanism.coordinate_drive(
        body,
        "x",
        value=lambda t: t * t,
        velocity=lambda t: 2.0 * t,
        acceleration=lambda _t: 2.0,
    )

    result = mechanism.simulate_fixed_orientation(np.linspace(0.0, 1.0, 11))
    diagnostics = mechanism.dynamics_diagnostics(result)

    np.testing.assert_allclose(result.body_poses(body)[-1].translation[0], 1.0, atol=1e-10)
    np.testing.assert_allclose(result.body_linear_velocity(body)[-1, 0], 2.0, atol=1e-10)
    np.testing.assert_allclose(result.linear_acceleration[:, 0, 0], 2.0, atol=1e-10)
    np.testing.assert_allclose(result.reaction_force[:, 0, 0], 4.0, atol=1e-10)
    assert diagnostics.max_acceleration_residual_m_per_s2 < 1e-12


def test_fixed_orientation_dynamics_rejects_torque_like_force_contracts():
    mechanism = Mechanism.spatial()
    body = mechanism.body("body")

    with pytest.raises(ValueError, match="world-frame"):
        mechanism.force(body, (1.0, 0.0, 0.0), frame="body")
    with pytest.raises(ValueError, match="center of mass"):
        mechanism.force(
            body, (1.0, 0.0, 0.0), application_point="offset"
        )

    offset = Mechanism.spatial()
    offset.body("offset-com", center_of_mass=(0.1, 0.0, 0.0))
    with pytest.raises(MechanismSolveError, match="coincide with its center of mass"):
        offset.simulate_fixed_orientation(np.array([0.0, 0.1]))


def test_spatial_model_and_excitation_identity_have_distinct_scopes():
    first = Mechanism.spatial()
    first_body = first.body("body")
    first.coordinate_drive(
        first_body, "x", value=lambda t: t, velocity=lambda _t: 1.0
    )
    second = Mechanism.spatial()
    second_body = second.body("body")
    second.coordinate_drive(
        second_body, "x", value=lambda t: 2.0 * t, velocity=lambda _t: 2.0
    )
    times = np.array([0.0, 0.5, 1.0])

    assert first.model_id() == second.model_id()
    assert first.excitation_id(times) != second.excitation_id(times)
    assert first.to_dict()["model_id_scope"] == (
        "structure_excluding_runtime_drive_callbacks"
    )


def test_highly_underconstrained_spatial_projection_uses_minimum_norm_step():
    mechanism = Mechanism.spatial()
    body = mechanism.body("free")
    mechanism.coordinate_drive(
        body, "x", value=lambda t: t, velocity=lambda _t: 1.0
    )

    solved = mechanism.solve_position(t=0.5)

    np.testing.assert_allclose(solved[0].translation, [0.5, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(solved[0].rotation.as_array(), [1.0, 0.0, 0.0, 0.0])


def test_spatial_result_schema_round_trip_preserves_contract(tmp_path):
    mechanism = Mechanism.spatial()
    ground = mechanism.ground()
    link = mechanism.body("link")
    mechanism.fixed(ground, link, name="attachment")
    result = mechanism.solve_kinematics(np.array([0.0, 0.1]))

    path = mechanism.result_to_json(result, tmp_path / "spatial-result.json")
    payload = load_spatial_kinematic_result_payload(path)

    assert payload == json.loads(path.read_text(encoding="utf-8"))
    assert payload["model_id"] == mechanism.model_id()
    assert payload["excitation_id"] == mechanism.excitation_id(result.t)
    assert payload["solver_status"] == {"success": True, "message": "converged"}
    assert payload["units"]["angular_velocity"] == "rad/s"
    assert payload["conventions"]["velocity_frame"] == "world"
    assert payload["bodies"][1]["name"] == "link"
    assert len(payload["bodies"][1]["poses"]) == 2
    assert payload["diagnostics"]["trajectory_velocity_checked"] is False
    assert len(payload["residual_layout"]) == len(payload["residuals"])
    assert np.shape(payload["velocity_residuals"]) == np.shape(payload["residuals"])

    payload["mbsd_version"] = "0.8.0"
    assert validate_spatial_kinematic_result_payload(payload)["mbsd_version"] == "0.8.0"

    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="unsupported spatial kinematic result"):
        validate_spatial_kinematic_result_payload(payload)


def test_spatial_result_schema_rejects_shape_mismatch(tmp_path):
    mechanism = Mechanism.spatial()
    ground = mechanism.ground()
    link = mechanism.body("link")
    mechanism.fixed(ground, link)
    payload = mechanism.result_to_dict(
        mechanism.solve_kinematics(np.array([0.0, 0.1]))
    )
    payload["bodies"][1]["linear_velocity"] = [[0.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="linear_velocity must have shape"):
        validate_spatial_kinematic_result_payload(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["units"].update(length="mm"), "units must be"),
        (
            lambda value: value["bodies"][0]["poses"][0]["translation"].__setitem__(
                0, np.nan
            ),
            "translation must contain only finite values",
        ),
        (
            lambda value: value["bodies"][0]["poses"][0]["rotation"].update(w=0.0),
            "pose quaternion norm must be 1",
        ),
        (lambda value: value["diagnostics"].update(steps=99), "steps must match"),
        (
            lambda value: value["diagnostics"].update(max_position_residual=-123.0),
            "non-negative finite number",
        ),
        (
            lambda value: value["bodies"][0]["poses"][0]["rotation"].update(w=2.0),
            "pose quaternion norm must be 1",
        ),
        (
            lambda value: value["diagnostics"].update(max_position_residual=123.0),
            "does not match residuals",
        ),
        (
            lambda value: value["capabilities"].update(spatial_dynamics=True),
            "capabilities do not match",
        ),
    ],
)
def test_spatial_result_schema_rejects_corrupt_nested_data(mutate, message):
    mechanism = Mechanism.spatial()
    ground = mechanism.ground()
    link = mechanism.body("link")
    mechanism.fixed(ground, link)
    payload = mechanism.result_to_dict(
        mechanism.solve_kinematics(np.array([0.0, 0.1]))
    )
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        validate_spatial_kinematic_result_payload(payload)


def test_successful_spatial_result_cannot_exceed_recorded_solver_tolerance():
    mechanism = Mechanism.spatial()
    ground = mechanism.ground()
    link = mechanism.body("link")
    mechanism.fixed(ground, link)
    payload = mechanism.result_to_dict(
        mechanism.solve_kinematics(np.array([0.0, 0.1]))
    )
    payload["residuals"][0][0] = 1.0
    payload["diagnostics"]["max_position_residual"] = 1.0
    payload["diagnostics"]["max_translation_residual_m"] = 1.0

    with pytest.raises(ValueError, match="must not exceed provenance.tolerance"):
        validate_spatial_kinematic_result_payload(payload)


def test_spatial_builder_does_not_claim_dynamics():
    mechanism = Mechanism.spatial()

    assert not hasattr(mechanism, "simulate")
    assert mechanism.capabilities["spatial_dynamics"] is False
