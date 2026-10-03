import json
from pathlib import Path

import numpy as np
import pytest

import mbsd
from mbsd.spatial import (
    compose_pose,
    fixed_joint_descriptor_residual,
    FixedJoint3D,
    Frame3D,
    inverse_pose,
    joint_residual_jacobian,
    max_spatial_residual,
    Pose3D,
    point_position,
    point_velocity,
    Quaternion,
    quaternion_rate_world,
    resolve_frame_pose,
    spherical_joint_descriptor_residual,
    spherical_joint_residual,
    SpatialBody,
    SpatialModel,
    SphericalJoint3D,
    load_spatial_model_payload,
    validate_spatial_model_payload,
)


def test_quaternion_axis_angle_rotates_point():
    q = Quaternion.from_axis_angle((0.0, 0.0, 1.0), np.pi / 2.0)

    np.testing.assert_allclose(q.rotate((1.0, 0.0, 0.0)), [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        q.to_rotation_matrix() @ q.to_rotation_matrix().T, np.eye(3), atol=1e-12
    )


def test_randomized_transform_composition_and_inverse_identities():
    rng = np.random.default_rng(20261003)
    for _ in range(100):
        axis_a = rng.normal(size=3)
        axis_b = rng.normal(size=3)
        pose_a = Pose3D(
            rng.normal(size=3),
            Quaternion.from_axis_angle(axis_a, rng.uniform(-np.pi, np.pi)),
        )
        pose_b = Pose3D(
            rng.normal(size=3),
            Quaternion.from_axis_angle(axis_b, rng.uniform(-np.pi, np.pi)),
        )
        point = rng.normal(size=3)
        composed = compose_pose(pose_a, pose_b)

        np.testing.assert_allclose(
            composed.transform_point(point),
            pose_a.transform_point(pose_b.transform_point(point)),
            atol=2e-12,
        )
        identity = compose_pose(inverse_pose(composed), composed)
        np.testing.assert_allclose(identity.translation, 0.0, atol=2e-12)
        np.testing.assert_allclose(identity.rotation.to_rotation_matrix(), np.eye(3), atol=2e-12)


@pytest.mark.parametrize("angle", [np.pi - 1e-12, np.pi, np.pi + 1e-12])
def test_near_pi_quaternions_remain_proper_rotations(angle):
    quaternion = Quaternion.from_axis_angle((1.0, -2.0, 3.0), angle)
    matrix = quaternion.to_rotation_matrix()

    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=2e-12)
    np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=2e-12)
    np.testing.assert_allclose(
        quaternion.inverse().compose(quaternion).as_array(),
        [1, 0, 0, 0],
        atol=2e-15,
    )


def test_direct_quaternion_construction_normalizes_value():
    quaternion = Quaternion(2.0, 0.0, 0.0, 0.0)

    np.testing.assert_allclose(quaternion.as_array(), [1.0, 0.0, 0.0, 0.0])
    assert quaternion.as_dict()["w"] == 1.0


def test_pose_transforms_local_point():
    pose = Pose3D(
        translation=np.array([1.0, 2.0, 3.0]),
        rotation=Quaternion.from_axis_angle((0.0, 0.0, 1.0), np.pi / 2.0),
    )

    np.testing.assert_allclose(pose.transform_point((1.0, 0.0, 0.0)), [1.0, 3.0, 3.0], atol=1e-12)


def test_spatial_model_exports_posed_bodies_frames_and_joint_sketches(tmp_path):
    body_pose = Pose3D(
        translation=np.array([0.0, 0.0, -1.0]),
        rotation=Quaternion.from_axis_angle((0.0, 1.0, 0.0), 0.25),
    )
    model = SpatialModel(metadata={"case": "pendulum-sketch"})
    model = model.with_body(
        SpatialBody(
            "link",
            mass=2.0,
            inertia=(1.0, 2.0, 3.0),
            center_of_mass=(0.0, 0.0, -0.5),
            pose=body_pose,
        )
    )
    model = model.with_frame(Frame3D("tip", Pose3D.identity(), parent_body=0))
    model = model.with_joint(
        SphericalJoint3D(
            "world-pivot",
            body_i=None,
            body_j=0,
            point_i=(0.0, 0.0, 0.0),
            point_j=(0.0, 0.0, 1.0),
        )
    )
    model = model.with_body(SpatialBody("payload", mass=1.0, inertia=(0.2, 0.3, 0.4)))
    model = model.with_joint(FixedJoint3D("payload-mount", body_i=0, body_j=1))

    payload = model.as_dict()

    assert {
        "schema",
        "schema_version",
        "mbsd_version",
        "status",
        "capabilities",
        "dimension",
        "units",
        "conventions",
        "metadata",
        "bodies",
        "frames",
        "joints",
    } <= payload.keys()
    assert payload["schema"] == "mbsd.spatial.model"
    assert payload["schema_version"] == 2
    assert payload["mbsd_version"] == mbsd.__version__
    assert payload["status"] == "experimental"
    assert payload["capabilities"]["joint_residual_jacobian"] == "finite_difference"
    assert payload["capabilities"]["general_spatial_solver"] is False
    assert payload["units"]["inertia"] == "kg*m^2"
    assert payload["conventions"]["world_frame"] == "right_handed_xyz"
    assert payload["conventions"]["quaternion_order"] == ["w", "x", "y", "z"]
    assert payload["conventions"]["body_pose"] == "body_to_world"
    assert payload["conventions"]["frame_pose"].startswith("parent_body_local")
    assert payload["metadata"]["case"] == "pendulum-sketch"
    assert payload["bodies"][0]["name"] == "link"
    assert payload["bodies"][0]["pose"]["translation"] == [0.0, 0.0, -1.0]
    assert payload["frames"][0]["pose"]["rotation"]["w"] == 1.0
    assert payload["frames"][0]["parent_body"] == 0
    assert payload["joints"][0]["kind"] == "spherical"
    assert payload["joints"][0]["body_i"] is None
    assert payload["joints"][1]["kind"] == "fixed"

    path = model.to_json(tmp_path / "spatial-model.json")
    assert json.loads(path.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("version", [1, 2])
def test_spatial_schema_golden_payloads_remain_readable(version):
    fixture = Path(__file__).parent / "fixtures" / f"spatial_model_v{version}.json"

    payload = load_spatial_model_payload(fixture)

    assert payload["schema_version"] == version


def test_spatial_schema_rejects_unknown_versions():
    payload = SpatialModel().as_dict()
    payload["schema_version"] = 99

    with pytest.raises(ValueError, match="unsupported spatial model schema_version 99"):
        validate_spatial_model_payload(payload)


def test_export_consumer_reconstructs_body_local_frame_world_position():
    body_pose = Pose3D(
        translation=(1.0, 2.0, 3.0),
        rotation=Quaternion.from_axis_angle((0.0, 0.0, 1.0), np.pi / 2.0),
    )
    model = SpatialModel(
        bodies=(SpatialBody("link", inertia=(1.0, 1.0, 1.0), pose=body_pose),),
        frames=(Frame3D("tip", Pose3D(translation=(0.5, 0.0, 0.0)), parent_body=0),),
    )

    payload = model.as_dict()
    body = payload["bodies"][0]["pose"]
    frame = payload["frames"][0]
    rotation = Quaternion(**body["rotation"])
    world_position = np.asarray(body["translation"]) + rotation.rotate(
        frame["pose"]["translation"]
    )

    np.testing.assert_allclose(world_position, [1.0, 2.5, 3.0], atol=1e-12)


def test_spatial_vocabulary_validates_inputs():
    with pytest.raises(ValueError, match="axis must be non-zero"):
        Quaternion.from_axis_angle((0.0, 0.0, 0.0), 1.0)

    with pytest.raises(ValueError, match="mass must be positive"):
        SpatialBody("bad", mass=0.0)

    with pytest.raises(ValueError, match="quaternion components must be finite"):
        Quaternion(w=np.nan)

    with pytest.raises(ValueError, match="quaternion norm must be non-zero"):
        Quaternion(0.0, 0.0, 0.0, 0.0)

    with pytest.raises(TypeError, match="rotation must be a Quaternion"):
        Pose3D(rotation=np.eye(3))

    with pytest.raises(ValueError, match="inertia values must be positive"):
        SpatialBody("bad-inertia", inertia=(1.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="triangle inequalities"):
        SpatialBody("impossible-inertia", inertia=(1.0, 1.0, 3.0))

    with pytest.raises(ValueError, match="cannot connect a body to itself"):
        SphericalJoint3D("bad-joint", body_i=0, body_j=0)

    model = SpatialModel().with_body(SpatialBody("only-body"))
    with pytest.raises(IndexError, match="out of range"):
        model.with_joint(FixedJoint3D("bad-reference", body_i=0, body_j=1))

    with pytest.raises(IndexError, match="frame parent_body"):
        SpatialModel().with_frame(Frame3D("orphan", parent_body=0))

    with pytest.raises(TypeError, match="not JSON serializable"):
        SpatialModel(metadata={"callback": lambda: None})

    with pytest.raises(TypeError, match=r"bodies\[0\]"):
        SpatialModel(bodies=[object()])

    with pytest.raises(ValueError, match="body names must be unique"):
        SpatialModel(bodies=[SpatialBody("duplicate"), SpatialBody("duplicate")])


def test_spatial_values_do_not_alias_inputs_or_allow_direct_mutation():
    source = np.array([1.0, 2.0, 3.0])
    pose = Pose3D(translation=source)
    source[0] = 99.0

    np.testing.assert_allclose(pose.translation, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="read-only"):
        pose.translation[1] = 88.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        pose.translation.setflags(write=True)

    model = SpatialModel(metadata={"owner": "original", "tags": ["test"]})
    with pytest.raises(TypeError):
        model.metadata["owner"] = "mutated"
    with pytest.raises(AttributeError):
        model.metadata["tags"].append("mutated")
    assert model.as_dict()["metadata"] == {"owner": "original", "tags": ["test"]}


def test_spatial_model_normalizes_collections_to_tuples():
    model = SpatialModel(bodies=[SpatialBody("link")], frames=[], joints=[])

    assert isinstance(model.bodies, tuple)
    assert isinstance(model.frames, tuple)
    assert isinstance(model.joints, tuple)


def test_spatial_point_position_pose_composition_and_inverse():
    parent = Pose3D(
        translation=(1.0, 2.0, 0.0),
        rotation=Quaternion.from_axis_angle((0.0, 0.0, 1.0), np.pi / 2.0),
    )
    local = Pose3D(translation=(0.5, 0.0, 0.0))

    world = compose_pose(parent, local)

    np.testing.assert_allclose(point_position(world, (0.0, 0.0, 0.0)), [1.0, 2.5, 0.0])
    identity = compose_pose(inverse_pose(world), world)
    np.testing.assert_allclose(identity.translation, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(identity.rotation.as_array(), [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_spatial_point_velocity_and_quaternion_rate_match_finite_difference():
    pose = Pose3D(
        translation=(1.0, 2.0, 3.0),
        rotation=Quaternion.from_axis_angle((0.0, 0.0, 1.0), 0.4),
    )
    local_point = np.array([0.5, -0.2, 0.1])
    linear_velocity = np.array([0.2, -0.3, 0.4])
    omega = np.array([0.1, 0.2, -0.4])
    dt = 1e-7
    delta = Quaternion.from_axis_angle(omega, np.linalg.norm(omega) * dt)
    advanced = Pose3D(
        translation=pose.translation + linear_velocity * dt,
        rotation=delta.compose(pose.rotation),
    )
    finite_difference = (
        point_position(advanced, local_point) - point_position(pose, local_point)
    ) / dt

    np.testing.assert_allclose(
        point_velocity(pose, local_point, linear_velocity, omega),
        finite_difference,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        quaternion_rate_world(Quaternion.identity(), omega),
        [0.0, 0.05, 0.1, -0.2],
    )


def test_resolved_frame_and_fixed_joint_residuals():
    body_pose = Pose3D(
        translation=(1.0, 2.0, 0.0),
        rotation=Quaternion.from_axis_angle((0.0, 0.0, 1.0), np.pi / 2.0),
    )
    local_frame = Frame3D("tip", Pose3D(translation=(1.0, 0.0, 0.0)), parent_body=0)
    resolved = resolve_frame_pose(local_frame, [body_pose])
    joint = FixedJoint3D(
        "world-fixed",
        body_i=None,
        body_j=0,
        frame_i=resolved,
        frame_j=local_frame.pose,
    )

    np.testing.assert_allclose(resolved.translation, [1.0, 3.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(fixed_joint_descriptor_residual(joint, [body_pose]), 0.0)
    np.testing.assert_allclose(joint.residual([body_pose]), 0.0)
    assert np.linalg.matrix_rank(joint_residual_jacobian(joint, [body_pose])) == 6

    translated = Pose3D(
        translation=body_pose.translation + np.array([1e-4, 0.0, 0.0]),
        rotation=body_pose.rotation,
    )
    np.testing.assert_allclose(
        joint.residual([translated]),
        [-1e-4, 0.0, 0.0, 0.0, 0.0, 0.0],
        atol=1e-12,
    )


def test_spherical_joint_residual_uses_canonical_vocabulary_descriptor():
    pose_i = Pose3D(translation=(1.0, 0.0, 0.0))
    pose_j = Pose3D(translation=(0.0, 1.0, 0.0))
    poses = [pose_i, pose_j]
    joint = SphericalJoint3D(
        "coincident-points",
        body_i=0,
        body_j=1,
        point_i=(0.0, 1.0, 0.0),
        point_j=(1.0, 0.0, 0.0),
    )

    residual = spherical_joint_descriptor_residual(joint, poses)

    np.testing.assert_allclose(residual, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(joint.residual(poses), residual)
    np.testing.assert_allclose(
        spherical_joint_residual(pose_i, joint.point_i, pose_j, joint.point_j),
        residual,
    )
    assert max_spatial_residual([residual]) == 0.0


def test_spherical_joint_residual_supports_world_endpoint():
    joint = SphericalJoint3D(
        "world-pivot",
        body_i=None,
        body_j=0,
        point_i=(1.0, 2.0, 3.0),
        point_j=(0.0, 0.0, 0.0),
    )

    residual = joint.residual([Pose3D(translation=(1.0, 2.0, 3.0))])

    np.testing.assert_allclose(residual, np.zeros(3), atol=1e-12)


def test_spherical_joint_residual_and_jacobian_respond_to_known_perturbation():
    joint = SphericalJoint3D(
        "two-body",
        body_i=0,
        body_j=1,
        point_i=(0.0, 1.0, 0.0),
        point_j=(0.0, 1.0, 0.0),
    )
    poses = [Pose3D.identity(), Pose3D.identity()]
    jacobian = joint_residual_jacobian(joint, poses)

    np.testing.assert_allclose(jacobian[:, :3], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(jacobian[:, 6:9], -np.eye(3), atol=1e-9)
    expected_rotation_i = np.array(
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    np.testing.assert_allclose(jacobian[:, 3:6], expected_rotation_i, atol=1e-9)
    np.testing.assert_allclose(jacobian[:, 9:12], -expected_rotation_i, atol=1e-9)

    perturbed = [Pose3D(translation=(1e-4, 0.0, 0.0)), poses[1]]
    np.testing.assert_allclose(joint.residual(perturbed), [1e-4, 0.0, 0.0])


def test_spatial_kinematics_reject_invalid_inputs():
    with pytest.raises(TypeError, match="pose must be a Pose3D"):
        point_position(np.eye(4), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite vector"):
        point_position(Pose3D.identity(), (0.0, 0.0))
    with pytest.raises(ValueError, match="finite values"):
        max_spatial_residual([np.array([0.0, np.nan, 0.0])])
    with pytest.raises(IndexError, match="out of range"):
        SphericalJoint3D("orphan", 0, 1).residual([Pose3D.identity()])
