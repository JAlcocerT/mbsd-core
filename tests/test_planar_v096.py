"""v0.9.6: planar reactions, portable descriptors, guardrails, and import."""

import json
from pathlib import Path

import numpy as np
import pytest

import mbsd
from mbsd import (
    ConstantSpeed,
    Mechanism,
    MechanismSolveError,
    NotPortableError,
    Polynomial,
    Tabulated,
)
from mbsd.planar import PlanarMechanism

M, L, G = 2.0, 1.0, 9.81
ROD_Q0 = np.array([0.0, 0.0, 0.0, L / 2, 0.0, 0.0])
SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"


def _rod(gravity=(0.0, -G)):
    mechanism = Mechanism.planar(gravity=gravity)
    ground = mechanism.ground()
    rod = mechanism.body("rod", mass=M, inertia=M * L**2 / 12)
    mechanism.pin(ground, rod, point_b=(-L / 2, 0.0), name="hub")
    return mechanism, rod


def _slider_crank(omega=100.0):
    R, rod_length = 0.05, 0.2
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    ground = mechanism.ground()
    crank = mechanism.body("crank", mass=0.5, inertia=1e-4)
    rod = mechanism.body("rod", mass=0.8, inertia=3e-3)
    piston = mechanism.body("piston", mass=0.4, inertia=1e-4)
    mechanism.pin(ground, crank, point_b=(-R / 2, 0.0), name="main")
    mechanism.pin(crank, rod, point_a=(R / 2, 0.0), point_b=(-rod_length / 2, 0.0), name="crankpin")
    mechanism.pin(rod, piston, point_a=(rod_length / 2, 0.0), name="wristpin")
    mechanism.slider(ground, piston, axis=(1.0, 0.0), name="bore")
    mechanism.motor(crank, omega=omega, name="engine")
    q0 = np.array([0, 0, 0, R / 2, 0, 0, R + rod_length / 2, 0, 0, R + rod_length, 0, 0], float)
    return mechanism, q0


# --- reactions -------------------------------------------------------------


def test_static_rod_motor_torque_pin_force_and_frame_load():
    mechanism, rod = _rod()
    mechanism.motor(rod, omega=0.0, name="actuator")
    result = mechanism.inverse_dynamics(np.linspace(0.0, 1.0, 5), q0=ROD_Q0)
    loads = mechanism.joint_loads(result)

    np.testing.assert_allclose(loads.drives["actuator"]["effort"], M * G * L / 2)
    np.testing.assert_allclose(loads.joints["hub"]["force_on_j"][:, 0], [0.0, M * G], atol=1e-12)
    np.testing.assert_allclose(loads.joints["hub"]["force_on_i"][:, 0], [0.0, -M * G], atol=1e-12)
    np.testing.assert_allclose(loads.joints["hub"]["moment_on_j"], 0.0, atol=1e-12)
    np.testing.assert_allclose(loads.frame["force"][:, 0], [0.0, -M * G], atol=1e-12)
    np.testing.assert_allclose(loads.frame["moment"][0], -M * G * L / 2, atol=1e-12)
    assert mechanism.check(result).passed


def test_spinning_rod_pin_carries_centripetal_force():
    omega = 3.0
    mechanism, rod = _rod(gravity=(0.0, 0.0))
    mechanism.motor(rod, omega=omega, name="spin")
    result = mechanism.inverse_dynamics(np.linspace(0.0, 1.0, 11), q0=ROD_Q0)
    loads = mechanism.joint_loads(result)

    np.testing.assert_allclose(
        np.linalg.norm(loads.joints["hub"]["force_on_j"], axis=0), M * omega**2 * L / 2
    )
    np.testing.assert_allclose(loads.drives["spin"]["effort"], 0.0, atol=1e-12)


def test_released_pendulum_pin_force_is_quarter_weight():
    mechanism, _ = _rod()
    result = mechanism.simulate(np.linspace(0.0, 0.5, 20), ROD_Q0, allow_underconstrained=True)
    assert result.multipliers.shape == (mechanism.nrestr, len(result.t))
    loads = mechanism.joint_loads(result)

    np.testing.assert_allclose(loads.joints["hub"]["force_on_j"][:, 0], [0.0, M * G / 4], atol=1e-9)
    report = mechanism.check(result)
    assert report.passed
    assert {check.name for check in report.checks} >= {"energy_drift", "position_residual"}


def test_slider_crank_frame_load_is_shaking_force():
    mechanism, q0 = _slider_crank()
    t = np.linspace(0.0, 2 * np.pi / 100.0, 181)
    result = mechanism.inverse_dynamics(t, q0=q0)
    loads = mechanism.joint_loads(result)

    masses = [0.5, 0.8, 0.4]
    momentum_rate = sum(m * result.a[3 * (k + 1) : 3 * (k + 1) + 2] for k, m in enumerate(masses))
    np.testing.assert_allclose(loads.frame["force"], -momentum_rate, atol=1e-9)
    for pin in ("main", "crankpin", "wristpin"):
        np.testing.assert_allclose(loads.joints[pin]["moment_on_j"], 0.0, atol=1e-9)
    assert abs(np.mean(loads.drives["engine"]["power"][:-1])) < 1e-6
    assert loads.max_action_reaction_residual < 1e-6
    assert mechanism.check(result).passed


def test_joint_loads_payload_matches_schema_contract(tmp_path):
    mechanism, q0 = _slider_crank()
    result = mechanism.inverse_dynamics(np.linspace(0.0, 0.01, 11), q0=q0)
    payload = mechanism.joint_loads_to_dict(result, metadata={"study": "unit"})
    schema = json.loads((SCHEMAS / "mbsd.planar.joint-loads-v1.schema.json").read_text())

    assert set(schema["required"]) <= payload.keys()
    assert payload["schema"] == "mbsd.planar.joint_loads"
    assert payload["result_type"] == "inverse_dynamic"
    assert [joint["name"] for joint in payload["joints"]] == ["main", "crankpin", "wristpin", "bore"]
    assert payload["drives"][0]["effort_unit"] == "N*m"
    for key, value in schema["properties"]["conventions"]["properties"].items():
        assert payload["conventions"][key] == value["const"]
    path = mechanism.joint_loads_to_json(result, tmp_path / "loads.json")
    assert json.loads(path.read_text())["time"] == payload["time"]


def test_joint_loads_reject_kinematic_results():
    mechanism, q0 = _slider_crank()
    result = mechanism.solve_kinematics(np.linspace(0.0, 0.01, 3), q0=q0)
    with pytest.raises(MechanismSolveError) as excinfo:
        mechanism.joint_loads(result)
    assert excinfo.value.code == "missing_multipliers"


def test_inverse_dynamics_requires_full_drive():
    mechanism, _ = _rod()
    with pytest.raises(MechanismSolveError) as excinfo:
        mechanism.inverse_dynamics(np.linspace(0.0, 1.0, 3), q0=ROD_Q0)
    assert excinfo.value.code == "not_kinematically_driven"
    assert excinfo.value.as_dict()["context"] == {"constraints": 5, "coordinates": 6}


# --- names -----------------------------------------------------------------


def test_joints_and_drives_get_unique_names():
    mechanism = Mechanism.planar()
    ground = mechanism.ground()
    a = mechanism.body("a")
    b = mechanism.body("b")
    assert mechanism.pin(ground, a).name == "pin_0"
    assert mechanism.pin(a, b).name == "pin_1"
    assert mechanism.slider(ground, b, name="rail").name == "rail"
    assert mechanism.motor(a, omega=1.0).name == "motor_0"
    with pytest.raises(ValueError, match="already used"):
        mechanism.pin(ground, b, name="rail")
    payload = mechanism.to_dict()
    assert [joint["name"] for joint in payload["joints"]["revolute"]] == ["pin_0", "pin_1"]
    assert payload["user_constraints"][0]["name"] == "motor_0"


# --- motion laws and loads -------------------------------------------------


def test_motion_law_derivatives_and_domain():
    law = Polynomial((1.0, 2.0, 3.0))
    assert law.value(2.0) == 17.0
    assert law.velocity(2.0) == 14.0
    assert law.acceleration(2.0) == 6.0
    table = Tabulated((0.0, 1.0, 2.0), (0.0, 1.0, 4.0))
    assert table.value(1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="outside"):
        table.value(2.5)
    assert ConstantSpeed(rate=2.0, start=1.0).value(3.0) == 7.0


def test_law_drive_matches_callable_drive():
    def build(use_law):
        mechanism = Mechanism.planar(gravity=(0.0, 0.0))
        ground = mechanism.ground()
        slider = mechanism.body("slider")
        mechanism.slider(ground, slider)
        if use_law:
            mechanism.coordinate_drive(slider, "x", law=Polynomial((0.0, 0.0, 0.5)))
        else:
            mechanism.coordinate_drive(
                slider, "x", lambda t: 0.5 * t**2, lambda t: t, lambda t: 1.0
            )
        return mechanism

    t = np.linspace(0.0, 1.0, 11)
    portable, opaque = build(True), build(False)
    np.testing.assert_allclose(portable.solve_kinematics(t).q, opaque.solve_kinematics(t).q)
    assert portable.to_dict()["portable"] is True
    assert opaque.to_dict()["portable"] is False
    assert opaque.to_dict()["user_constraints"][0]["opaque"] is True
    with pytest.raises(ValueError, match="either law"):
        portable.coordinate_drive(1, "y", lambda t: t, law=ConstantSpeed(1.0))


def test_constant_and_tabulated_loads_accelerate_free_body():
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    mechanism.ground()
    body = mechanism.body("puck", mass=2.0, inertia=0.5)
    mechanism.constant_force(body, (4.0, 0.0))
    mechanism.tabulated_torque(body, (0.0, 1.0), (1.0, 1.0))
    result = mechanism.simulate(np.linspace(0.0, 1.0, 11), np.zeros(6), allow_underconstrained=True)

    np.testing.assert_allclose(result.q[3, -1], 0.5 * 2.0 * 1.0**2, rtol=1e-7)
    np.testing.assert_allclose(result.q[5, -1], 0.5 * 2.0 * 1.0**2, rtol=1e-7)


def test_force_offset_point_generates_moment():
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    mechanism.ground()
    body = mechanism.body("bar", mass=1.0, inertia=1.0)
    load = mechanism.constant_force(body, (0.0, 2.0), point=(0.5, 0.0))
    Q = load.generalized(mechanism.ncoord, np.zeros(6), np.zeros(6), 0.0)
    np.testing.assert_allclose(Q[3:], [0.0, 2.0, 1.0])


def test_callback_load_is_reported_and_rejected_on_import():
    mechanism, _ = _rod()
    mechanism.force(lambda t, q, v: np.zeros(6), name="contact")
    report = mechanism.validate(ROD_Q0)
    assert report.portable is False
    assert "opaque_load" in {issue.code for issue in report.issues}
    payload = mechanism.to_dict()
    assert payload["portable"] is False
    with pytest.raises(NotPortableError) as excinfo:
        PlanarMechanism.from_dict(payload)
    assert excinfo.value.code == "not_portable"


# --- import / round trip ---------------------------------------------------


def test_json_round_trip_gives_identical_solves(tmp_path):
    mechanism, q0 = _slider_crank()
    mechanism.spring(2, 0, stiffness=500.0, natural_length=0.1, point_a=(0.0, 0.0), name="return")
    mechanism.constant_force(3, (-20.0, 0.0), name="gas")
    path = mechanism.to_json(tmp_path / "engine.json")
    rebuilt = PlanarMechanism.from_json(path)

    t = np.linspace(0.0, 0.02, 21)
    original = mechanism.inverse_dynamics(t, q0=q0)
    copy = rebuilt.inverse_dynamics(t, q0=q0)
    np.testing.assert_array_equal(copy.q, original.q)
    np.testing.assert_array_equal(copy.multipliers, original.multipliers)
    assert rebuilt.to_dict() == mechanism.to_dict()


def test_tabulated_drive_round_trip_and_domain_validation(tmp_path):
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    ground = mechanism.ground()
    slider = mechanism.body("slider")
    mechanism.slider(ground, slider)
    mechanism.coordinate_drive(slider, "x", law=Tabulated((0.0, 0.5, 1.0), (0.0, 0.2, 0.1)))
    rebuilt = PlanarMechanism.from_json(mechanism.to_json(tmp_path / "m.json"))
    t = np.linspace(0.0, 1.0, 11)
    np.testing.assert_array_equal(rebuilt.solve_kinematics(t).q, mechanism.solve_kinematics(t).q)

    report = mechanism.validate(t=np.linspace(0.0, 2.0, 5))
    assert not report.ok
    assert "time_outside_table" in {issue.code for issue in report.issues}


def test_from_dict_reads_v095_motor_payload_and_rejects_callables():
    legacy = {
        "schema": "mbsd.planar.mechanism",
        "schema_version": 1,
        "gravity": [0.0, 0.0],
        "bodies": [
            {"index": 0, "name": "ground", "mass": 0.0, "inertia": 0.0, "center_of_mass": [0, 0]},
            {"index": 1, "name": "crank", "mass": 1.0, "inertia": 0.1, "center_of_mass": [0, 0]},
        ],
        "joints": {
            "revolute": [
                {"index": 0, "body_i": 0, "body_j": 1, "point_i": [0, 0], "point_j": [0, 0]}
            ],
            "prismatic": [],
        },
        "user_constraints": [
            {"index": 0, "kind": "motor", "count": 1, "body": 1, "coordinate": "theta",
             "omega": 2.0, "phase": 0.0}
        ],
        "forces": {"gravity": [0.0, 0.0], "springs": [], "custom": []},
    }
    mechanism = PlanarMechanism.from_dict(legacy)
    result = mechanism.solve_kinematics(np.array([0.0, 1.0]))
    assert result.q[5, -1] == pytest.approx(2.0)
    assert mechanism.model.rev_joints[0].name == "pin_0"

    legacy["user_constraints"][0] = {"index": 0, "kind": "coordinate_drive", "count": 1}
    with pytest.raises(NotPortableError):
        PlanarMechanism.from_dict(legacy)
    legacy["schema_version"] = 99
    with pytest.raises(MechanismSolveError) as excinfo:
        PlanarMechanism.from_dict(legacy)
    assert excinfo.value.code == "unsupported_schema_version"


# --- guardrails ------------------------------------------------------------


def test_validate_reports_structured_issues():
    assert Mechanism.planar().validate().issues[0].code == "empty_model"

    mechanism, q0 = _slider_crank()
    report = mechanism.validate(q0)
    assert report.ok and report.portable
    assert report.classification == "fully_constrained"
    assert report.degrees_of_freedom == 0

    free, _ = _rod()
    report = free.validate(ROD_Q0)
    assert report.ok
    assert [issue.code for issue in report.issues] == ["underconstrained"]

    offset = Mechanism.planar()
    offset.ground()
    offset.body("lump", center_of_mass=(0.1, 0.0))
    codes = {issue.code for issue in offset.validate().issues}
    assert "offset_center_of_mass" in codes
    json.dumps(offset.validate().as_dict())


def test_check_fails_on_corrupted_result():
    mechanism, q0 = _slider_crank()
    result = mechanism.solve_kinematics(np.linspace(0.0, 0.01, 5), q0=q0)
    assert mechanism.check(result).passed
    q = result.q.copy()
    q[3, 2] += 1e-3
    corrupted = type(result)(t=result.t, q=q, v=result.v, a=result.a)
    report = mechanism.check(corrupted)
    assert not report.passed
    failed = [check.name for check in report.checks if check.passed is False]
    assert "position_residual" in failed


def test_check_skips_energy_for_damped_models():
    mechanism, _ = _rod(gravity=(0.0, 0.0))
    mechanism.spring(0, 1, stiffness=10.0, natural_length=0.0, damping=0.5,
                     point_b=(L / 2, 0.0))
    result = mechanism.simulate(np.linspace(0.0, 0.1, 5), ROD_Q0, allow_underconstrained=True)
    energy = next(check for check in mechanism.check(result).checks if check.name == "energy_drift")
    assert energy.passed is None
    assert "dissipate" in energy.note


def test_resource_limits_fail_with_structured_codes():
    mechanism, q0 = _slider_crank()
    with pytest.raises(MechanismSolveError) as excinfo:
        mechanism.solve_kinematics(np.linspace(0.0, 1.0, 11), q0=q0, max_steps=10)
    assert excinfo.value.code == "resource_limit"

    pendulum, _ = _rod()
    with pytest.raises(MechanismSolveError) as excinfo:
        pendulum.simulate(
            np.linspace(0.0, 1.0, 11), ROD_Q0, allow_underconstrained=True, max_evaluations=5
        )
    assert excinfo.value.code == "resource_limit"


# --- discovery -------------------------------------------------------------


def test_operation_manifest_file_matches_runtime_and_api():
    manifest = mbsd.operation_manifest()
    on_disk = json.loads((SCHEMAS / "mbsd.operations-v1.json").read_text())
    assert on_disk == json.loads(json.dumps(manifest))
    for operation in manifest["operations"]:
        call = operation["call"].split("(")[0]
        owner, _, attribute = call.rpartition(".")
        target = mbsd if owner == "mbsd" else PlanarMechanism
        assert hasattr(target, attribute), operation["call"]


def test_capabilities_are_plain_json():
    capabilities = mbsd.capabilities()
    json.dumps(capabilities)
    assert capabilities["planar"]["solvers"]["inverse_dynamics"] is True
    assert capabilities["planar"]["limits"]["offset_center_of_mass_dynamics"] is False
    assert "revolute" in capabilities["spatial"]["joint_types"]
