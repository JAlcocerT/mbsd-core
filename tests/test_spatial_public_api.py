import json

import numpy as np
import pytest

import mbsd
from mbsd.spatial import (
    FixedJoint3D,
    Frame3D,
    Pose3D,
    Quaternion,
    SpatialBody,
    SpatialModel,
    SphericalJoint3D,
)


def test_quaternion_axis_angle_rotates_point():
    q = Quaternion.from_axis_angle((0.0, 0.0, 1.0), np.pi / 2.0)

    np.testing.assert_allclose(q.rotate((1.0, 0.0, 0.0)), [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        q.to_rotation_matrix() @ q.to_rotation_matrix().T, np.eye(3), atol=1e-12
    )


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
        "dimension",
        "units",
        "conventions",
        "metadata",
        "bodies",
        "frames",
        "joints",
    } <= payload.keys()
    assert payload["schema"] == "mbsd.spatial.model"
    assert payload["schema_version"] == 1
    assert payload["mbsd_version"] == mbsd.__version__
    assert payload["status"] == "experimental"
    assert payload["units"]["inertia"] == "kg*m^2"
    assert payload["conventions"]["world_frame"] == "right_handed_xyz"
    assert payload["conventions"]["quaternion_order"] == ["w", "x", "y", "z"]
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

    with pytest.raises(ValueError, match="cannot connect a body to itself"):
        SphericalJoint3D("bad-joint", body_i=0, body_j=0)

    model = SpatialModel().with_body(SpatialBody("only-body"))
    with pytest.raises(IndexError, match="out of range"):
        model.with_joint(FixedJoint3D("bad-reference", body_i=0, body_j=1))

    with pytest.raises(IndexError, match="frame parent_body"):
        SpatialModel().with_frame(Frame3D("orphan", parent_body=0))

    with pytest.raises(TypeError, match="not JSON serializable"):
        SpatialModel(metadata={"callback": lambda: None})
