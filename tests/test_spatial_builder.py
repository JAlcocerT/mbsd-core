import json

import numpy as np
import pytest

from mbsd import Mechanism, MechanismSolveError
from mbsd.spatial import (
    Pose3D,
    Quaternion,
    SpatialMechanism,
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
        link, "x", value=lambda _t: 0.0, velocity=lambda _t: 1.0
    )
    mechanism.coordinate_drive(
        link, "x", value=lambda _t: 0.0, velocity=lambda _t: -1.0
    )
    poses = mechanism.solve_position(tol=1e-9)

    with pytest.raises(MechanismSolveError, match="Spatial velocity solve failed"):
        mechanism.solve_velocity(poses, tol=1e-9)


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
    assert payload["solver_status"] == {"success": True, "message": "converged"}
    assert payload["units"]["angular_velocity"] == "rad/s"
    assert payload["conventions"]["velocity_frame"] == "world"
    assert payload["bodies"][1]["name"] == "link"
    assert len(payload["bodies"][1]["poses"]) == 2

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


def test_spatial_builder_does_not_claim_dynamics():
    mechanism = Mechanism.spatial()

    assert not hasattr(mechanism, "simulate")
    assert mechanism.capabilities["spatial_dynamics"] is False
