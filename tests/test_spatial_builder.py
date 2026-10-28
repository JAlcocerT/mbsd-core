import json
import copy
from pathlib import Path

import numpy as np
import pytest

from mbsd import Mechanism, MechanismSolveError
from mbsd.spatial import (
    Pose3D,
    Quaternion,
    SpatialMechanism,
    load_spatial_dynamic_result_payload,
    load_rigid_body_result_payload,
    load_coupled_result_payload,
    validate_coupled_result_payload,
    load_spatial_kinematic_result_payload,
    validate_spatial_kinematic_result_payload,
    validate_spatial_dynamic_result_payload,
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


def test_fixed_orientation_preserves_authored_rotation_during_initial_projection():
    mechanism = Mechanism.spatial()
    ground = mechanism.ground()
    authored = Quaternion.from_axis_angle((0.0, 0.0, 1.0), 0.4)
    link = mechanism.body("link", pose=Pose3D(rotation=authored))
    mechanism.spherical(
        ground, link, point_i=(1.0, 0.0, 0.0), point_j=(0.0, 1.0, 0.0)
    )

    result = mechanism.simulate_fixed_orientation(np.array([0.0, 0.1]))

    assert abs(np.dot(result.poses[0][1].rotation.as_array(), authored.as_array())) > 1 - 1e-12


def test_dynamic_payload_exposes_constraint_rows_and_rejects_physical_corruption():
    mechanism = Mechanism.spatial()
    body = mechanism.body("driven", mass=2.0)
    mechanism.coordinate_drive(
        body, "x", value=lambda t: t * t, velocity=lambda t: 2 * t,
        acceleration=lambda _t: 2.0, name="prescribed-x",
    )
    payload = mechanism.dynamics_result_to_dict(
        mechanism.simulate_fixed_orientation(np.array([0.0, 0.1, 0.2]))
    )

    assert payload["schema_version"] == 2
    assert payload["constraint_layout"][0]["kind"] == "drive"
    assert payload["constraint_layout"][0]["name"] == "prescribed-x"
    assert payload["constraint_layout"][0]["component"] == "x"
    validate_spatial_dynamic_result_payload(payload)

    mutations = []
    missing_conventions = copy.deepcopy(payload)
    missing_conventions.pop("conventions")
    mutations.append(missing_conventions)
    changing_rotation = copy.deepcopy(payload)
    changing_rotation["bodies"][0]["poses"][1]["rotation"] = {"w": 0.0, "x": 0.0, "y": 0.0, "z": 1.0}
    mutations.append(changing_rotation)
    bad_reaction = copy.deepcopy(payload)
    bad_reaction["bodies"][0]["reaction_force"][1][0] += 10.0
    mutations.append(bad_reaction)
    bad_steps = copy.deepcopy(payload)
    bad_steps["diagnostics"]["steps"] = 99
    mutations.append(bad_steps)
    bad_layout = copy.deepcopy(payload)
    bad_layout["constraint_layout"][0]["body_i"] = 99
    mutations.append(bad_layout)
    bad_provenance = copy.deepcopy(payload)
    bad_provenance["provenance"]["stabilization"] = "hidden"
    mutations.append(bad_provenance)

    for corrupt in mutations:
        with pytest.raises(ValueError):
            validate_spatial_dynamic_result_payload(corrupt)

    groups = mechanism.dynamics_constraint_groups(
        mechanism.simulate_fixed_orientation(np.array([0.0, 0.1, 0.2]))
    )
    assert list(groups) == ["drive:prescribed-x"]
    assert groups["drive:prescribed-x"]["components"] == ["x"]
    assert groups["drive:prescribed-x"]["multipliers"].shape == (1, 3)

    with pytest.raises(ValueError, match="already in use"):
        mechanism.coordinate_drive(
            body, "y", value=lambda _t: 0.0, velocity=lambda _t: 0.0,
            name="prescribed-x",
        )


def test_fixed_orientation_free_particle_and_two_body_momentum_conservation():
    free = Mechanism.spatial()
    free.body("free", mass=2.0)
    result = free.simulate_fixed_orientation(
        np.linspace(0.0, 2.0, 21), linear_velocity0=np.array([[1.0, -2.0, 0.5]])
    )
    diagnostics = free.dynamics_diagnostics(result)
    assert diagnostics.relative_energy_change < 1e-13
    assert diagnostics.relative_momentum_change < 1e-13

    constrained = Mechanism.spatial()
    first = constrained.body("first", mass=2.0)
    second = constrained.body("second", mass=3.0)
    constrained.spherical(first, second, name="coupling")
    load = constrained.force(first, (10.0, 0.0, 0.0))
    result = constrained.simulate_fixed_orientation(np.array([0.0, 1.0]), forces=(load,))
    np.testing.assert_allclose(result.linear_acceleration[:, :, 0], 2.0, atol=1e-11)
    np.testing.assert_allclose(
        result.reaction_force[:, :, 0], [[-6.0, 6.0], [-6.0, 6.0]], atol=1e-11
    )


def test_fixed_orientation_matches_independent_sine_force_fixture_and_converges():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "spatial_translation_sine_force.json").read_text()
    )
    mechanism = Mechanism.spatial()
    body = mechanism.body("mass", mass=2.0)
    force = mechanism.force(
        body, lambda t, _poses, _velocity: (2.0 * np.sin(t), 0.0, 0.0), name="sine"
    )
    samples = np.asarray(fixture["time"])
    result = mechanism.simulate_fixed_orientation(
        samples, forces=(force,), rtol=1e-11, atol=1e-13
    )
    np.testing.assert_allclose(
        [pose.translation[0] for pose in result.body_poses(body)],
        fixture["position_x"], atol=fixture["absolute_tolerance"],
    )
    np.testing.assert_allclose(
        result.body_linear_velocity(body)[:, 0], fixture["velocity_x"],
        atol=fixture["absolute_tolerance"],
    )

    reference = mechanism.simulate_fixed_orientation(
        np.linspace(0.0, 4.0, 81), forces=(force,), rtol=1e-12, atol=1e-14
    )
    loose = mechanism.simulate_fixed_orientation(
        reference.t, forces=(force,), method="RK45", rtol=1e-4, atol=1e-6
    )
    tight = mechanism.simulate_fixed_orientation(
        reference.t, forces=(force,), method="RK45", rtol=1e-9, atol=1e-11
    )
    loose_error = np.max(np.abs(loose.linear_velocity - reference.linear_velocity))
    tight_error = np.max(np.abs(tight.linear_velocity - reference.linear_velocity))
    assert tight_error < loose_error
    assert tight_error < 1e-8


def test_fixed_orientation_rejects_bad_initial_states_rank_and_runtime_force():
    mechanism = Mechanism.spatial()
    ground = mechanism.ground()
    link = mechanism.body("link")
    mechanism.spherical(ground, link, name="pivot")
    bad_poses = (Pose3D.identity(), Pose3D(translation=(1.0, 0.0, 0.0)))
    with pytest.raises(MechanismSolveError, match="Initial position"):
        mechanism.simulate_fixed_orientation(np.array([0.0, 0.1]), poses0=bad_poses)
    with pytest.raises(MechanismSolveError, match="Initial velocity"):
        mechanism.simulate_fixed_orientation(
            np.array([0.0, 0.1]), linear_velocity0=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )

    redundant = Mechanism.spatial()
    first = redundant.body("first")
    second = redundant.body("second")
    redundant.spherical(first, second, name="a")
    redundant.spherical(first, second, name="b")
    with pytest.raises(MechanismSolveError, match="rank-deficient"):
        redundant.simulate_fixed_orientation(np.array([0.0, 0.1]))

    invalid = Mechanism.spatial()
    body = invalid.body("body")
    force = invalid.force(
        body, lambda _t, _poses, _velocity: (np.nan, 0.0, 0.0), name="invalid"
    )
    with pytest.raises(ValueError, match="finite vector"):
        invalid.simulate_fixed_orientation(np.array([0.0, 0.1]), forces=(force,))


def test_free_body_constant_principal_torque_matches_analytic_rotation(tmp_path):
    mechanism = Mechanism.spatial()
    body = mechanism.body("rotor", mass=2.0, inertia=(2.0, 3.0, 4.0))
    load = mechanism.rigid_body_load(body, torque=(2.0, 0.0, 0.0), name="motor")
    times = np.linspace(0.0, 1.0, 51)

    result = mechanism.simulate_free_body(times, loads=(load,))
    diagnostics = mechanism.rigid_body_diagnostics(result)
    payload = mechanism.rigid_body_result_to_dict(result)
    path = mechanism.rigid_body_result_to_json(result, tmp_path / "rotor.json")

    np.testing.assert_allclose(result.angular_velocity_body[:, 0], times, atol=2e-9)
    final = result.poses[-1].rotation.to_rotation_matrix()
    expected = Quaternion.from_axis_angle((1.0, 0.0, 0.0), 0.5).to_rotation_matrix()
    np.testing.assert_allclose(final, expected, atol=2e-9)
    assert diagnostics.max_quaternion_norm_error < 1e-14
    assert diagnostics.max_rotation_orthogonality_error < 1e-14
    assert payload["conventions"]["angular_velocity_frame"] == "body"
    assert load_rigid_body_result_payload(path)["body"]["name"] == "rotor"


def test_free_asymmetric_body_conserves_energy_and_world_angular_momentum():
    mechanism = Mechanism.spatial()
    mechanism.body("top", inertia=(2.0, 3.0, 4.0))
    result = mechanism.simulate_free_body(
        np.linspace(0.0, 5.0, 201), angular_velocity0_body=(0.7, 1.1, -0.4),
        rtol=1e-11, atol=1e-13,
    )
    diagnostics = mechanism.rigid_body_diagnostics(result)

    assert diagnostics.relative_energy_change < 2e-10
    assert diagnostics.relative_angular_momentum_change < 2e-10


@pytest.mark.parametrize("axis", range(3))
def test_torque_free_principal_axis_rotation_is_uniform(axis):
    mechanism = Mechanism.spatial()
    mechanism.body("rotor", inertia=(2.0, 3.0, 4.0))
    omega = np.zeros(3)
    omega[axis] = 1.25
    result = mechanism.simulate_free_body(
        np.linspace(0.0, 2.0, 41), angular_velocity0_body=omega,
        rtol=1e-11, atol=1e-13,
    )
    np.testing.assert_allclose(
        result.angular_velocity_body, np.tile(omega, (len(result.t), 1)), atol=2e-10
    )
    diagnostics = mechanism.rigid_body_diagnostics(result)
    assert diagnostics.relative_energy_change < 1e-11
    assert diagnostics.relative_angular_momentum_change < 2e-10


def test_free_body_force_at_point_and_invalid_model_boundaries():
    mechanism = Mechanism.spatial()
    body = mechanism.body("body", mass=2.0, inertia=(1.0, 2.0, 3.0))
    load = mechanism.rigid_body_load(
        body, force=(0.0, 2.0, 0.0),
        application_point_from_com_body=(1.0, 0.0, 0.0),
        force_frame="body",
    )
    result = mechanism.simulate_free_body(np.array([0.0, 0.01]), loads=(load,))
    np.testing.assert_allclose(result.linear_acceleration[0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.angular_acceleration_body[0], [0.0, 0.0, 2.0 / 3.0], atol=1e-12)

    constrained = Mechanism.spatial()
    constrained.ground()
    constrained.body("other")
    with pytest.raises(MechanismSolveError, match="exactly one body"):
        constrained.simulate_free_body(np.array([0.0, 0.1]))


def test_free_body_accepts_symmetric_body_inertia_and_converges_with_tolerance():
    mechanism = Mechanism.spatial()
    body = mechanism.body("body", inertia=(2.0, 3.0, 4.0))
    tensor = np.array([[2.5, 0.2, 0.0], [0.2, 3.0, 0.1], [0.0, 0.1, 3.5]])
    load = mechanism.rigid_body_load(body, torque=(0.4, -0.2, 0.3))
    times = np.linspace(0.0, 2.0, 81)
    loose = mechanism.simulate_free_body(
        times, loads=(load,), inertia_tensor_body=tensor, rtol=1e-5, atol=1e-7
    )
    tight = mechanism.simulate_free_body(
        times, loads=(load,), inertia_tensor_body=tensor, rtol=1e-11, atol=1e-13
    )
    reference = mechanism.simulate_free_body(
        times, loads=(load,), inertia_tensor_body=tensor, rtol=1e-12, atol=1e-14
    )
    loose_error = np.max(np.abs(loose.angular_velocity_body - reference.angular_velocity_body))
    tight_error = np.max(np.abs(tight.angular_velocity_body - reference.angular_velocity_body))

    np.testing.assert_allclose(tight.inertia_tensor_body, tensor)
    assert tight_error < loose_error
    assert tight_error < 1e-10


def test_free_body_rejects_invalid_inertia_and_nonfinite_runtime_load():
    mechanism = Mechanism.spatial()
    body = mechanism.body("body")
    with pytest.raises(ValueError, match="symmetric"):
        mechanism.simulate_free_body(
            np.array([0.0, 0.1]), inertia_tensor_body=np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        )
    load = mechanism.rigid_body_load(
        body, force=lambda _t, _pose, _velocity, _omega: (np.nan, 0.0, 0.0)
    )
    with pytest.raises(ValueError, match="finite vector"):
        mechanism.simulate_free_body(np.array([0.0, 0.1]), loads=(load,))


def test_coupled_gravity_free_fall_preserves_arbitrary_orientations():
    mechanism = Mechanism.spatial()
    rotations = (
        Quaternion.from_axis_angle((1.0, 0.0, 0.0), 0.3),
        Quaternion.from_axis_angle((0.0, 1.0, 1.0), -0.5),
    )
    mechanism.body("a", mass=2.0, pose=Pose3D((1.0, 2.0, 3.0), rotations[0]))
    mechanism.body("b", mass=5.0, pose=Pose3D((-2.0, 0.5, 1.0), rotations[1]))
    times = np.linspace(0.0, 1.0, 21)
    result = mechanism.simulate_coupled_rigid_bodies(
        times, gravity_world=(0.0, 0.0, -9.81)
    )

    np.testing.assert_allclose(
        result.center_of_mass_position[:, :, 2],
        np.asarray([[3.0, 1.0]]) - 4.905 * times[:, None] ** 2,
        atol=2e-9,
    )
    for body_index, rotation in enumerate(rotations):
        for pose in result.body_poses(body_index):
            np.testing.assert_allclose(
                pose.rotation.to_rotation_matrix(), rotation.to_rotation_matrix(), atol=2e-10
            )


def test_coupled_off_center_force_and_pure_couple_have_expected_effects():
    mechanism = Mechanism.spatial()
    first = mechanism.body("forced", mass=2.0, inertia=(1.0, 2.0, 3.0))
    second = mechanism.body("couple", mass=4.0, inertia=(2.0, 3.0, 4.0))
    off_center = mechanism.rigid_body_load(
        first, force=(0.0, 2.0, 0.0),
        application_point_from_com_body=(1.0, 0.0, 0.0), force_frame="body",
    )
    couple = mechanism.rigid_body_load(second, torque=(0.0, 0.0, 4.0))
    result = mechanism.simulate_coupled_rigid_bodies(
        np.array([0.0, 0.01]), loads=(off_center, couple)
    )

    np.testing.assert_allclose(result.linear_acceleration[0, 0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(result.angular_acceleration_body[0, 0], [0.0, 0.0, 2.0 / 3.0])
    np.testing.assert_allclose(result.linear_acceleration[:, 1], 0.0, atol=1e-14)
    np.testing.assert_allclose(result.angular_acceleration_body[0, 1], [0.0, 0.0, 1.0])


def test_coupled_offset_com_pose_tracks_rotating_reference_origin():
    mechanism = Mechanism.spatial()
    body = mechanism.body(
        "offset", inertia=(2.0, 3.0, 4.0), center_of_mass=(0.5, 0.0, 0.0),
        pose=Pose3D(translation=(1.0, 2.0, 3.0)),
    )
    load = mechanism.rigid_body_load(body, torque=(0.0, 0.0, 4.0))
    result = mechanism.simulate_coupled_rigid_bodies(
        np.linspace(0.0, 1.0, 21), loads=(load,)
    )

    np.testing.assert_allclose(
        result.center_of_mass_position,
        np.tile([1.5, 2.0, 3.0], (len(result.t), 1, 1)), atol=2e-10,
    )
    for index, pose in enumerate(result.body_poses(0)):
        reconstructed = pose.translation + pose.rotation.rotate((0.5, 0.0, 0.0))
        np.testing.assert_allclose(reconstructed, result.center_of_mass_position[index, 0], atol=2e-10)


def test_coupled_isolated_system_conserves_total_momenta_and_energy():
    mechanism = Mechanism.spatial()
    mechanism.body("a", mass=2.0, inertia=(2.0, 3.0, 4.0), pose=Pose3D((-1.0, 0.0, 0.0)))
    mechanism.body("b", mass=3.0, inertia=(1.5, 2.0, 2.5), pose=Pose3D((2.0, 1.0, 0.0)))
    result = mechanism.simulate_coupled_rigid_bodies(
        np.linspace(0.0, 3.0, 101),
        linear_velocity0=np.array([[0.4, -0.2, 0.1], [-0.3, 0.5, -0.2]]),
        angular_velocity0_body=np.array([[0.7, 1.1, -0.4], [0.2, -0.3, 0.5]]),
        rtol=1e-11, atol=1e-13,
    )
    diagnostics = mechanism.coupled_diagnostics(result)

    assert diagnostics.relative_kinetic_energy_change < 3e-10
    assert diagnostics.relative_linear_momentum_change < 1e-12
    assert diagnostics.relative_angular_momentum_change < 3e-10


def test_coupled_pair_interaction_is_equal_opposite_and_exports_complete_state(tmp_path):
    mechanism = Mechanism.spatial()
    first = mechanism.body("first", mass=2.0, pose=Pose3D((-1.0, 0.0, 0.0)))
    second = mechanism.body("second", mass=3.0, pose=Pose3D((1.0, 0.0, 0.0)))
    interaction = mechanism.pair_interaction(
        first, second, (6.0, 0.0, 0.0), name="internal-pair"
    )
    result = mechanism.simulate_coupled_rigid_bodies(
        np.linspace(0.0, 1.0, 11), interactions=(interaction,)
    )
    path = mechanism.coupled_result_to_json(result, tmp_path / "coupled.json")
    payload = load_coupled_result_payload(path)

    np.testing.assert_allclose(result.linear_acceleration[:, 0, 0], 3.0, atol=1e-12)
    np.testing.assert_allclose(result.linear_acceleration[:, 1, 0], -2.0, atol=1e-12)
    total_momentum = np.sum(
        np.array([2.0, 3.0])[None, :, None] * result.linear_velocity, axis=1
    )
    np.testing.assert_allclose(total_momentum, 0.0, atol=2e-10)
    assert payload["bodies"][0]["center_of_mass_position"]
    assert payload["model_id"] == mechanism.model_id()
    assert len(payload["excitation_id"]) == 16
    assert payload["provenance"]["interaction_descriptors"][0]["name"] == "internal-pair"

    corrupt_force = copy.deepcopy(payload)
    corrupt_force["bodies"][0]["applied_force_world"][0][0] += 1.0
    with pytest.raises(ValueError, match="force balance"):
        validate_coupled_result_payload(corrupt_force)
    corrupt_torque = copy.deepcopy(payload)
    corrupt_torque["bodies"][0]["applied_torque_body"][0][2] += 1.0
    with pytest.raises(ValueError, match="torque balance"):
        validate_coupled_result_payload(corrupt_torque)
    corrupt_pose = copy.deepcopy(payload)
    corrupt_pose["bodies"][0]["poses"][0]["translation"][0] += 1.0
    with pytest.raises(ValueError, match="center-of-mass"):
        validate_coupled_result_payload(corrupt_pose)
    corrupt_provenance = copy.deepcopy(payload)
    corrupt_provenance["provenance"]["interaction_descriptors"][0]["body_j"] = 99
    with pytest.raises(ValueError, match="interaction descriptor"):
        validate_coupled_result_payload(corrupt_provenance)


def test_coupled_dynamics_is_invariant_under_world_frame_transform():
    base = Mechanism.spatial()
    base_body = base.body("body", mass=2.0, inertia=(2.0, 3.0, 4.0), pose=Pose3D((1.0, -2.0, 0.5)))
    base_load = base.rigid_body_load(
        base_body, force=(1.0, 2.0, -0.5), torque=(0.3, -0.2, 0.4), torque_frame="world"
    )
    times = np.linspace(0.0, 0.5, 31)
    velocity = np.array([[0.4, -0.1, 0.2]])
    base_result = base.simulate_coupled_rigid_bodies(
        times, linear_velocity0=velocity, loads=(base_load,), gravity_world=(0.0, 0.0, -9.81)
    )

    frame_rotation = Quaternion.from_axis_angle((1.0, 1.0, -0.5), 0.7)
    matrix = frame_rotation.to_rotation_matrix()
    shift = np.array([3.0, -1.0, 2.0])
    transformed = Mechanism.spatial()
    transformed_body = transformed.body(
        "body", mass=2.0, inertia=(2.0, 3.0, 4.0),
        pose=Pose3D(matrix @ np.array([1.0, -2.0, 0.5]) + shift, frame_rotation),
    )
    transformed_load = transformed.rigid_body_load(
        transformed_body, force=matrix @ np.array([1.0, 2.0, -0.5]),
        torque=matrix @ np.array([0.3, -0.2, 0.4]), torque_frame="world",
    )
    transformed_result = transformed.simulate_coupled_rigid_bodies(
        times, linear_velocity0=np.array([matrix @ velocity[0]]), loads=(transformed_load,),
        gravity_world=matrix @ np.array([0.0, 0.0, -9.81]),
    )

    expected_position = np.einsum("ij,tbj->tbi", matrix, base_result.center_of_mass_position) + shift
    expected_velocity = np.einsum("ij,tbj->tbi", matrix, base_result.linear_velocity)
    np.testing.assert_allclose(transformed_result.center_of_mass_position, expected_position, atol=2e-9)
    np.testing.assert_allclose(transformed_result.linear_velocity, expected_velocity, atol=2e-9)
    np.testing.assert_allclose(
        transformed_result.angular_velocity_body,
        base_result.angular_velocity_body, atol=2e-9,
    )
    expected_force = np.einsum("ij,tbj->tbi", matrix, base_result.applied_force_world)
    np.testing.assert_allclose(
        transformed_result.applied_force_world, expected_force, atol=2e-9
    )
    for base_pose, transformed_pose in zip(
        base_result.body_poses(0), transformed_result.body_poses(0)
    ):
        np.testing.assert_allclose(
            transformed_pose.rotation.to_rotation_matrix(),
            matrix @ base_pose.rotation.to_rotation_matrix(), atol=2e-9,
        )


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
