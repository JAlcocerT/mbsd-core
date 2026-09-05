import numpy as np
import pytest

from mbsd.spatial import (
    Frame3D,
    Pose3D,
    Quaternion,
    SpatialBody,
    SpatialModel,
    SphericalJoint3D,
    max_spatial_residual,
    point_position,
    spherical_joint_residual,
)


def test_quaternion_axis_angle_rotates_point():
    q = Quaternion.from_axis_angle((0.0, 0.0, 1.0), np.pi / 2.0)

    np.testing.assert_allclose(q.rotate((1.0, 0.0, 0.0)), [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(q.to_rotation_matrix() @ q.to_rotation_matrix().T, np.eye(3), atol=1e-12)


def test_pose_transforms_local_point():
    pose = Pose3D(
        translation=np.array([1.0, 2.0, 3.0]),
        rotation=Quaternion.from_axis_angle((0.0, 0.0, 1.0), np.pi / 2.0),
    )

    np.testing.assert_allclose(pose.transform_point((1.0, 0.0, 0.0)), [1.0, 3.0, 3.0], atol=1e-12)


def test_spatial_model_exports_json_ready_dict():
    model = SpatialModel().with_body(SpatialBody("link", mass=2.0, inertia=(1.0, 2.0, 3.0)))
    model = model.with_frame(Frame3D("world", Pose3D.identity()))

    payload = model.as_dict()

    assert payload["schema"] == "mbsd.spatial.model"
    assert payload["status"] == "experimental"
    assert payload["bodies"][0]["name"] == "link"
    assert payload["frames"][0]["pose"]["rotation"]["w"] == 1.0


def test_spatial_vocabulary_validates_inputs():
    with pytest.raises(ValueError, match="axis must be non-zero"):
        Quaternion.from_axis_angle((0.0, 0.0, 0.0), 1.0)

    with pytest.raises(ValueError, match="mass must be positive"):
        SpatialBody("bad", mass=0.0)


def test_spatial_point_position_and_spherical_residual():
    pose_i = Pose3D(translation=np.array([1.0, 0.0, 0.0]))
    pose_j = Pose3D(translation=np.array([0.0, 1.0, 0.0]))

    np.testing.assert_allclose(point_position(pose_i, np.array([0.0, 1.0, 0.0])), [1.0, 1.0, 0.0])
    np.testing.assert_allclose(
        spherical_joint_residual(
            pose_i,
            np.array([0.0, 1.0, 0.0]),
            pose_j,
            np.array([1.0, 0.0, 0.0]),
        ),
        [0.0, 0.0, 0.0],
    )


def test_spherical_joint_reports_residual_and_metadata():
    joint = SphericalJoint3D(0, 1, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    poses = [
        Pose3D(translation=np.array([1.0, 0.0, 0.0])),
        Pose3D(translation=np.array([0.0, 1.0, 0.0])),
    ]

    np.testing.assert_allclose(joint.residual(poses), [0.0, 0.0, 0.0])
    assert max_spatial_residual([joint.residual(poses)]) == 0.0
    assert joint.as_dict()["kind"] == "spherical"
