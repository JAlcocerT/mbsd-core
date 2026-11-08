"""v0.9.7: one spatial simulate(), parallel_axes, revolute and distance drives,
spatial guardrails, planar point-on-line, offset COM, and public analysis."""

import json
import math
import warnings

import numpy as np
import pytest

import mbsd
from mbsd import ConstantSpeed, Mechanism, MechanismSolveError, NotPortableError, Polynomial
from mbsd.planar import PlanarMechanism, analysis
from mbsd.planar.derivatives import dt_jacobian
from mbsd.planar.jacobians import jacobian
from mbsd.spatial import (
    ParallelAxesJoint3D,
    Pose3D,
    Quaternion,
    SpatialBody,
    SpatialMechanism,
    SpatialModel,
    joint_residual_jacobian,
    validate_constrained_result_payload,
    validate_spatial_model_payload,
)

ROT = Quaternion.from_axis_angle
Z_TO_X = ROT((0.0, 1.0, 0.0), math.pi / 2)
Z_TO_Y = ROT((1.0, 0.0, 0.0), -math.pi / 2)
OMEGA = 4.0 * math.pi


def cardan(beta, *, drive=None, redundant=False):
    """Input shaft along x, output shaft at angle beta in the xy plane."""
    m = Mechanism.spatial()
    ground = m.ground("world")
    shaft_in = m.body("shaft_in", mass=5.0, inertia=(0.04, 0.02, 0.02))
    m.body("cross", mass=0.3, inertia=(0.002, 0.002, 0.002))
    m.body("shaft_out", mass=2.0, inertia=(0.02, 0.02, 0.02))
    z_to_out = ROT((0.0, 0.0, 1.0), beta) * Z_TO_X
    bearing = m.revolute(
        ground, shaft_in, frame_i=Pose3D(rotation=Z_TO_X), frame_j=Pose3D(rotation=Z_TO_X),
        name="input_bearing",
    )
    m.revolute(
        shaft_in, 2, frame_i=Pose3D(rotation=Z_TO_Y), frame_j=Pose3D(rotation=Z_TO_Y), name="yoke_in"
    )
    m.revolute(2, 3, name="yoke_out")
    second_bearing = m.revolute if redundant else m.parallel_axes
    second_bearing(
        ground, 3, frame_i=Pose3D(rotation=z_to_out), frame_j=Pose3D(rotation=z_to_out),
        name="output_bearing",
    )
    if drive is not None:
        m.revolute_drive(bearing, law=drive, name="motor")
    return m


def output_speed(result, body, axis):
    return np.array([
        step[body].rotation.to_rotation_matrix() @ result.angular_velocity_body[k, body]
        for k, step in enumerate(result.poses)
    ]) @ np.asarray(axis)


# --- spatial: parallel_axes and the Cardan family ----------------------------------


def test_parallel_axes_jacobian_is_analytic():
    poses = (
        Pose3D(translation=(0.1, 0.2, 0.3), rotation=ROT((1.0, 2.0, 0.5), 0.7)),
        Pose3D(translation=(-0.2, 0.1, 0.4), rotation=ROT((0.3, -1.0, 2.0), 1.1)),
    )
    joint = ParallelAxesJoint3D("pa", 0, 1, Pose3D(rotation=ROT((0, 1, 0), 0.4)), Pose3D(rotation=ROT((1, 0, 0), -0.3)))
    analytic = joint_residual_jacobian(joint, poses)
    numeric = np.zeros_like(analytic)
    h = 1e-7
    for column in range(12):
        for sign in (1.0, -1.0):
            body, coordinate = divmod(column, 6)
            moved = list(poses)
            pose = poses[body]
            if coordinate < 3:
                translation = np.array(pose.translation)
                translation[coordinate] += sign * h
                moved[body] = Pose3D(translation, pose.rotation)
            else:
                axis = np.zeros(3)
                axis[coordinate - 3] = 1.0
                moved[body] = Pose3D(pose.translation, ROT(axis, sign * h) * pose.rotation)
            numeric[:, column] += sign * joint.residual(tuple(moved)) / (2 * h)
    assert analytic.shape == (2, 12)
    assert np.allclose(analytic, numeric, atol=1e-7)


@pytest.mark.parametrize("degrees", [15.0, 30.0, 45.0])
def test_driven_cardan_matches_closed_form_speed_ratio(degrees):
    beta = math.radians(degrees)
    mechanism = cardan(beta, drive=ConstantSpeed(rate=OMEGA))
    report = mechanism.validate()
    assert report.ok and report.degrees_of_freedom == 0
    t = np.linspace(0.0, 0.125, 9)
    result = mechanism.simulate(t)
    ratio = output_speed(result, 3, (math.cos(beta), math.sin(beta), 0.0)) / OMEGA
    closed = math.cos(beta) / (1.0 - math.sin(beta) ** 2 * np.cos(OMEGA * t) ** 2)
    assert np.max(np.abs(ratio - closed)) < 1e-6
    assert mechanism.check(result).passed
    torque = mechanism.drive_efforts(result)["motor"]
    assert torque["effort_units"] == "N*m"
    assert np.allclose(torque["power"], torque["effort"] * OMEGA)


def test_free_cardan_from_input_spin_only():
    beta = math.radians(30.0)
    mechanism = cardan(beta)
    report = mechanism.validate()
    assert report.ok and report.degrees_of_freedom == 1
    assert [issue.code for issue in report.issues] == ["underconstrained"]

    velocities = mechanism.consistent_velocities(angular_velocity_body={"shaft_in": (OMEGA, 0.0, 0.0)})
    assert velocities.unique
    hand = (OMEGA, OMEGA * math.tan(beta), 0.0)
    assert np.allclose(velocities.angular_velocity_body[2], hand, atol=1e-12)
    assert np.allclose(velocities.angular_velocity_body[3], hand, atol=1e-12)

    t = np.linspace(0.0, 0.06, 7)
    result = mechanism.simulate(
        t, linear_velocity0=velocities.linear_velocity,
        angular_velocity0_body=velocities.angular_velocity_body, stabilization_frequency=8.0,
    )
    win = result.angular_velocity_body[:, 1, 0]
    theta = np.array([2 * math.atan2(step[1].rotation.x, step[1].rotation.w) for step in result.poses])
    closed = math.cos(beta) / (1.0 - math.sin(beta) ** 2 * np.cos(theta) ** 2)
    ratio = output_speed(result, 3, (math.cos(beta), math.sin(beta), 0.0)) / win
    assert np.max(np.abs(ratio - closed)) < 1e-6
    verdict = mechanism.check(result)
    assert verdict.passed
    assert {check.name for check in verdict.checks} >= {"energy_drift", "force_balance", "torque_balance"}


def test_double_cardan_gives_constant_output_speed():
    beta, length = math.radians(25.0), 0.3
    direction = np.array([math.cos(beta), math.sin(beta), 0.0])
    z_to_d = ROT((0.0, 0.0, 1.0), beta) * Z_TO_X
    m = Mechanism.spatial()
    g = m.ground()
    m.body("shaft_in", mass=2.0, inertia=(0.02, 0.01, 0.01))
    m.body("cross1", mass=0.2, inertia=(0.001, 0.001, 0.001))
    m.body("mid", mass=1.0, inertia=(0.01, 0.01, 0.01))
    m.body("cross2", mass=0.2, inertia=(0.001, 0.001, 0.001), pose=Pose3D(translation=length * direction))
    m.body("shaft_out", mass=2.0, inertia=(0.02, 0.01, 0.01), pose=Pose3D(translation=length * direction))
    bearing = m.revolute(g, 1, frame_i=Pose3D(rotation=Z_TO_X), frame_j=Pose3D(rotation=Z_TO_X), name="input_bearing")
    m.revolute(1, 2, frame_i=Pose3D(rotation=Z_TO_Y), frame_j=Pose3D(rotation=Z_TO_Y), name="yoke_1a")
    m.revolute(2, 3, name="yoke_1b")
    m.parallel_axes(g, 3, frame_i=Pose3D(rotation=z_to_d), frame_j=Pose3D(rotation=z_to_d), name="mid_axis")
    m.revolute(3, 4, frame_i=Pose3D(translation=length * direction), name="yoke_2a")
    m.revolute(4, 5, frame_i=Pose3D(rotation=Z_TO_Y), frame_j=Pose3D(rotation=Z_TO_Y), name="yoke_2b")
    m.parallel_axes(g, 5, frame_i=Pose3D(rotation=Z_TO_X), frame_j=Pose3D(rotation=Z_TO_X), name="output_bearing")
    m.revolute_drive(bearing, law=ConstantSpeed(rate=OMEGA), name="motor")
    assert m.validate().ok
    result = m.simulate(np.linspace(0.0, 0.125, 7))
    assert np.max(np.abs(output_speed(result, 5, (1.0, 0.0, 0.0)) - OMEGA)) < 1e-6
    assert np.ptp(output_speed(result, 3, direction)) > 0.1


def test_redundant_bearing_is_named_before_and_during_solve():
    mechanism = cardan(math.radians(30.0), redundant=True)
    report = mechanism.validate()
    assert not report.ok
    redundant = [issue for issue in report.issues if issue.code == "redundant_constraints"]
    assert redundant and "output_bearing" in redundant[0].message
    assert "parallel_axes" in redundant[0].remediation
    assert {row["name"] for row in mechanism.redundant_constraints()} == {"output_bearing"}
    with pytest.raises(MechanismSolveError) as excinfo:
        mechanism.simulate(np.linspace(0.0, 0.01, 3))
    assert excinfo.value.code == "rank_deficient"
    assert excinfo.value.context["constraints"] == ["output_bearing"]


# --- spatial: drives -----------------------------------------------------------------


def _driven_disc(law):
    m = Mechanism.spatial()
    g = m.ground()
    disc = m.body("disc", mass=2.0, inertia=(0.5, 0.25, 0.25), pose=Pose3D(rotation=Z_TO_X))
    hinge = m.revolute(g, disc, frame_i=Pose3D(rotation=Z_TO_X), name="hinge")
    m.revolute_drive(hinge, law=law, name="motor")
    return m


def test_revolute_drive_reproduces_angle_and_inertial_torque():
    for law in (ConstantSpeed(rate=3.0), Polynomial((0.0, 0.0, 2.0))):
        mechanism = _driven_disc(law)
        t = np.linspace(0.0, 1.0, 21)
        result = mechanism.simulate(t, constraint_tolerance=1e-11)
        angle = np.unwrap([2 * math.atan2(step[1].rotation.x, step[1].rotation.w) for step in result.poses])
        assert np.max(np.abs(angle - [law.value(x) for x in t])) < 1e-9
        torque = mechanism.drive_efforts(result)["motor"]["effort"]
        assert np.allclose(torque, 0.25 * law.acceleration(0.0), atol=1e-9)


def test_revolute_drive_holds_gravity_torque():
    m = Mechanism.spatial()
    g = m.ground()
    arm = m.body("arm", mass=2.0, inertia=(0.01, 0.2, 0.2), pose=Pose3D(translation=(0.5, 0.0, 0.0)))
    hinge = m.revolute(g, arm, frame_j=Pose3D(translation=(-0.5, 0.0, 0.0)), name="hinge")
    m.revolute_drive(hinge, law=ConstantSpeed(rate=0.0), name="hold")
    result = m.simulate(np.linspace(0.0, 0.1, 3), gravity_world=(0.0, -9.81, 0.0))
    assert np.allclose(m.drive_efforts(result)["hold"]["effort"], 2.0 * 9.81 * 0.5, rtol=1e-9)


def test_spatial_coordinate_drive_rejects_rotation_with_structured_error():
    m = Mechanism.spatial()
    body = m.body("b")
    with pytest.raises(MechanismSolveError) as excinfo:
        m.coordinate_drive(body, "rz", law=ConstantSpeed(rate=1.0))
    assert excinfo.value.code == "unsupported_capability"
    assert "revolute_drive" in excinfo.value.remediation


# --- spatial: one simulate() -----------------------------------------------------------


def test_simulate_covers_free_and_coupled_bodies():
    t = np.linspace(0.0, 0.5, 11)
    free = Mechanism.spatial()
    free.body("top", mass=1.5, inertia=(0.3, 0.2, 0.1))
    spin = {"top": (1.0, 4.0, 0.2)}
    result = free.simulate(t, angular_velocity0_body=spin, gravity_world=(0.0, 0.0, -9.81))
    assert result.multipliers.shape == (11, 0)
    assert np.allclose(result.center_of_mass_position[-1, 0, 2], -0.5 * 9.81 * 0.25, atol=1e-9)
    assert free.check(result).passed
    validate_constrained_result_payload(free.constrained_result_to_dict(result))
    with pytest.warns(DeprecationWarning):
        legacy = free.simulate_free_body(t, angular_velocity0_body=spin[ "top"])
    assert np.allclose(legacy.angular_velocity_body, result.angular_velocity_body[:, 0], atol=1e-7)

    pair = Mechanism.spatial()
    a = pair.body("a", mass=1.0, inertia=(0.1, 0.1, 0.1))
    b = pair.body("b", mass=2.0, inertia=(0.1, 0.1, 0.1), pose=Pose3D(translation=(1.0, 0.0, 0.0)))
    spring = pair.pair_interaction(a, b, (1.0, 0.0, 0.0), name="tie")
    coupled = pair.simulate(t, interactions=(spring,))
    with pytest.warns(DeprecationWarning):
        legacy = pair.simulate_coupled_rigid_bodies(t, interactions=(spring,))
    assert np.allclose(legacy.center_of_mass_position, coupled.center_of_mass_position, atol=1e-8)


def test_old_constrained_entry_point_is_a_deprecated_alias():
    mechanism = _driven_disc(ConstantSpeed(rate=1.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mechanism.simulate_constrained_rigid_bodies(np.linspace(0.0, 0.1, 3))
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert mechanism.capabilities["dynamics_entry_point"] == "simulate"


def test_inconsistent_initial_velocity_is_reported_with_remediation():
    mechanism = cardan(math.radians(30.0))
    spin = {"shaft_in": (OMEGA, 0.0, 0.0)}
    report = mechanism.validate(angular_velocity0_body=spin)
    codes = [issue.code for issue in report.issues]
    assert "inconsistent_initial_velocity" in codes
    with pytest.raises(MechanismSolveError) as excinfo:
        mechanism.simulate(np.linspace(0.0, 0.01, 3), angular_velocity0_body=spin)
    assert excinfo.value.code == "inconsistent_initial_state"
    assert "consistent_velocities" in excinfo.value.remediation
    with pytest.raises(MechanismSolveError) as excinfo:
        mechanism.consistent_velocities(angular_velocity_body={"shaft_in": (OMEGA, 1.0, 0.0)})
    assert excinfo.value.code == "inconsistent_initial_state"


def test_spatial_resource_guards():
    mechanism = _driven_disc(ConstantSpeed(rate=1.0))
    with pytest.raises(MechanismSolveError) as excinfo:
        mechanism.simulate(np.linspace(0.0, 1.0, 11), max_steps=5)
    assert excinfo.value.code == "resource_limit"
    with pytest.raises(MechanismSolveError) as excinfo:
        mechanism.simulate(np.linspace(0.0, 1.0, 11), max_evaluations=3)
    assert excinfo.value.code == "resource_limit"
    assert any(issue.code == "resource_limit" for issue in mechanism.validate(np.zeros(5), max_steps=2).issues)


def test_spatial_check_fails_with_strict_thresholds():
    mechanism = _driven_disc(ConstantSpeed(rate=3.0))
    result = mechanism.simulate(np.linspace(0.0, 0.2, 5))
    verdict = mechanism.check(result, position_tol=0.0)
    assert not verdict.passed
    assert [check.name for check in verdict.checks if check.passed is False] == ["position_residual"]
    energy = [check for check in verdict.checks if check.name == "energy_drift"][0]
    assert energy.passed is None and "drives" in energy.note


def test_fixed_orientation_rejects_orientation_constraints():
    m = cardan(math.radians(20.0))
    with pytest.raises(MechanismSolveError) as excinfo:
        m.simulate_fixed_orientation(np.linspace(0.0, 0.1, 3))
    assert excinfo.value.code == "unsupported_capability"


# --- spatial: portability, schemas -----------------------------------------------------


def test_spatial_model_with_laws_round_trips(tmp_path):
    mechanism = cardan(math.radians(30.0), drive=ConstantSpeed(rate=OMEGA))
    assert mechanism.portable and mechanism.validate().portable
    payload = mechanism.to_dict()
    assert payload["schema_version"] == 4
    validate_spatial_model_payload({k: v for k, v in payload.items() if k not in {"model_id", "model_id_scope", "solver_capabilities"}})
    rebuilt = SpatialMechanism.from_json(mechanism.to_json(tmp_path / "cardan.json"))
    assert rebuilt.model_id() == mechanism.model_id()
    t = np.linspace(0.0, 0.05, 4)
    original, replay = mechanism.simulate(t), rebuilt.simulate(t)
    assert original.simulation_id == replay.simulation_id
    assert np.array_equal(original.multipliers, replay.multipliers)

    lift = Mechanism.spatial()
    lift.body("carriage", mass=3.0, inertia=(0.1, 0.1, 0.1))
    lift.coordinate_drive(0, "z", law=Polynomial((0.0, 0.2, 0.1)), name="lift")
    again = SpatialMechanism.from_dict(json.loads(json.dumps(lift.to_dict())))
    first, second = lift.simulate(t), again.simulate(t)
    assert np.array_equal(first.multipliers, second.multipliers)
    assert np.allclose(lift.drive_efforts(first)["lift"]["effort"], 3.0 * 0.2, atol=1e-9)


def test_callable_spatial_drive_is_not_portable():
    m = Mechanism.spatial()
    g = m.ground()
    body = m.body("slider")
    m.spherical(g, body, name="anchor")
    m.coordinate_drive(body, "x", value=lambda t: 0.0, velocity=lambda t: 0.0, name="python_drive")
    report = m.validate()
    assert not report.portable
    assert "opaque_drive" in [issue.code for issue in report.issues]
    with pytest.raises(NotPortableError):
        SpatialMechanism.from_dict(m.to_dict())


def test_constrained_result_v3_payload_is_strict_and_v2_stays_readable():
    mechanism = _driven_disc(ConstantSpeed(rate=2.0))
    result = mechanism.simulate(np.linspace(0.0, 0.1, 4))
    payload = mechanism.constrained_result_to_dict(result)
    assert payload["schema_version"] == 3
    assert [row["kind"] for row in payload["constraint_layout"]][-1] == "revolute_drive"
    validate_constrained_result_payload(payload)

    tampered = json.loads(json.dumps(payload))
    tampered["drive_efforts"][0]["effort"][1] += 1.0
    with pytest.raises(ValueError, match="drive_efforts"):
        validate_constrained_result_payload(tampered)

    plain = Mechanism.spatial()
    g = plain.ground()
    bob = plain.body("bob", pose=Pose3D(translation=(0.0, -1.0, 0.0)))
    plain.spherical(g, bob, point_j=(0.0, 1.0, 0.0), name="pivot")
    v2 = plain.constrained_result_to_dict(plain.simulate(np.linspace(0.0, 0.1, 4), gravity_world=(0, -9.81, 0)))
    v2["schema_version"] = 2
    v2.pop("drive_efforts")
    for key in ("orientation_primitives", "drive_kinds", "drive_coordinates", "redundant_constraints", "unconstrained_bodies"):
        v2["capabilities"].pop(key)
    validate_constrained_result_payload(v2)


def test_spatial_model_v3_cannot_carry_parallel_axes():
    model = SpatialModel().with_body(SpatialBody("a")).with_joint(ParallelAxesJoint3D("pa", None, 0))
    payload = model.as_dict()
    assert payload["joints"][0]["kind"] == "parallel_axes"
    validate_spatial_model_payload(payload)
    old = json.loads(json.dumps(payload))
    old["schema_version"] = 3
    old["conventions"].pop("parallel_axes_frames")
    with pytest.raises(ValueError, match="schema_version 4"):
        validate_spatial_model_payload(old)


def test_spatial_ergonomics():
    a, b = ROT((0, 0, 1), 0.3), ROT((1, 0, 0), 0.2)
    assert np.allclose((a * b).as_array(), a.compose(b).as_array())
    with pytest.raises(ValueError, match=r"I_x <= I_y \+ I_z.*\(I, I/2, I/2\)"):
        SpatialBody("disc", inertia=(400.0, 1.0, 1.0))
    m = cardan(math.radians(10.0))
    with pytest.raises(ValueError, match="ground body at index 0"):
        m.simulate(np.linspace(0.0, 0.1, 3), angular_velocity0_body=np.zeros((3, 3)))


# --- planar: point_on_line, distance_drive, user-constraint loads -----------------------


def _rocker_on_valve():
    """Rocker pivoted on ground; its tip rides on a valve stem that slides vertically."""
    m = Mechanism.planar(gravity=(0.0, -9.81))
    g = m.ground()
    rocker = m.body("rocker", mass=0.4, inertia=2e-3)
    valve = m.body("valve", mass=0.1, inertia=1e-5)
    m.pin(g, rocker, point_b=(-0.05, 0.0), name="pivot")
    m.slider(g, valve, axis=(0.0, 1.0), point_rail=(0.1, 0.0), name="guide")
    m.point_on_line(rocker, valve, point_a=(0.05, 0.0), direction_b=(1.0, 0.0), name="tip")
    m.coordinate_drive(rocker, "theta", law=Polynomial((0.0, 0.0, 1.0)), name="cam")
    m.constant_force(valve, (0.0, -50.0), name="spring_preload")
    q0 = np.array([0, 0, 0, 0.05, 0, 0, 0.1, 0, 0], dtype=float)
    return m, q0


def test_point_on_line_analytic_derivatives_match_finite_differences():
    m, q0 = _rocker_on_valve()
    q = m.solve_position(q0, 0.3)
    rng = np.random.default_rng(3)
    q_test = q + 0.05 * rng.standard_normal(q.size)
    v = rng.standard_normal(q.size)
    h = 1e-6
    numeric = np.column_stack([
        (m.constraint_residual(q_test + h * e, 0.3) - m.constraint_residual(q_test - h * e, 0.3)) / (2 * h)
        for e in np.eye(q.size)
    ])
    row = [group.rows for group in analysis.constraint_groups(m) if group.name == "tip"][0]
    assert np.allclose(jacobian(m.model, q_test, 0.3)[row], numeric[row], atol=1e-8)
    rate = (jacobian(m.model, q_test + h * v, 0.3) - jacobian(m.model, q_test - h * v, 0.3)) / (2 * h)
    assert np.allclose(dt_jacobian(m.model, q_test, v, 0.3)[row], rate[row], atol=1e-7)


def test_point_on_line_reaction_is_normal_and_round_trips():
    m, q0 = _rocker_on_valve()
    assert m.validate(q0).portable
    t = np.linspace(0.0, 0.4, 9)
    result = m.inverse_dynamics(t, q0=q0)
    assert m.check(result).passed
    tip = m.joint_loads(result).joints["tip"]
    assert tip["kind"] == "point_on_line"
    assert np.allclose(tip["force_on_j"][0], 0.0, atol=1e-9)
    assert np.allclose(tip["moment_on_j"], 0.0, atol=1e-9)
    # The rocker tip lifts the valve against its 50 N preload and weight.
    assert np.all(tip["force_on_j"][1] > 50.0)
    payload = m.to_dict()
    assert payload["schema_version"] >= 2
    assert payload["joints"]["point_on_line"][0]["name"] == "tip"
    replay = PlanarMechanism.from_dict(json.loads(json.dumps(payload))).inverse_dynamics(t, q0=q0)
    assert np.array_equal(replay.multipliers, result.multipliers)


def _excavator_boom():
    m = Mechanism.planar(gravity=(0.0, -9.81))
    g = m.ground()
    boom = m.body("boom", mass=120.0, inertia=40.0)
    m.pin(g, boom, point_b=(-1.0, 0.0), name="boom_pivot")
    base, attach = np.array([0.5, -0.5]), np.array([0.8, 0.0])
    length0 = float(np.linalg.norm(attach - base))
    m.distance_drive(g, boom, law=Polynomial((length0, 0.05)), point_a=base, point_b=(-0.2, 0.0), name="cylinder")
    return m, np.array([0, 0, 0, 1.0, 0, 0], dtype=float)


def test_distance_drive_is_portable_and_reports_actuator_force():
    m, q0 = _excavator_boom()
    report = m.validate(q0)
    assert report.ok and report.portable
    t = np.linspace(0.0, 2.0, 11)
    result = m.inverse_dynamics(t, q0=q0)
    assert m.check(result).passed
    loads = m.joint_loads(result)
    cylinder = loads.drives["cylinder"]
    assert cylinder["kind"] == "distance_drive"
    row = [group.rows for group in analysis.constraint_groups(m) if group.name == "cylinder"][0]
    assert np.allclose(cylinder["effort"], -result.multipliers[row][0], atol=1e-9)
    assert np.all(cylinder["effort"] > 0.0)  # pushing to lift the boom
    assert np.allclose(cylinder["power"], cylinder["effort"] * 0.05)

    # Frame load closes the free-body diagram of the whole mechanism.
    for k in range(len(t)):
        accel = analysis.point_acceleration(result.q[:, k], result.v[:, k], result.a[:, k], 1)
        expected = -(120.0 * accel - 120.0 * np.array([0.0, -9.81]))
        assert np.allclose(loads.frame["force"][:, k], expected, atol=1e-8)

    replay = PlanarMechanism.from_json(m.to_json(_tmp_json()))
    assert np.array_equal(replay.inverse_dynamics(t, q0=q0).multipliers, result.multipliers)
    payload = m.joint_loads_to_dict(result)
    assert payload["schema_version"] >= 2
    assert payload["drives"][0]["coordinate"] == "distance"


def _tmp_json():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp()) / "boom.json"


def test_user_constraints_appear_in_joint_loads_and_frame():
    from mbsd.planar.model import UserConstraint

    m = Mechanism.planar(gravity=(0.0, -9.81))
    m.ground()
    block = m.body("block", mass=3.0, inertia=0.1)
    m.slider(0, block, axis=(1.0, 0.0), name="rail")

    def c(mb, q, t):
        return np.array([q[3] - 0.5 * t])

    def cq(mb, q, t):
        row = np.zeros((1, mb.ncoord))
        row[0, 3] = 1.0
        return row

    raw = UserConstraint(c, cq, lambda mb, q, v, t: np.zeros((1, mb.ncoord)),
                         lambda mb, q, t: np.array([-0.5]), lambda mb, q, v, t: np.array([0.0]),
                         name="python_push")
    m.model.user_constraints.append(raw)
    result = m.inverse_dynamics(np.linspace(0.0, 1.0, 3), q0=np.zeros(6))
    loads = m.joint_loads(result)
    assert set(loads.constraints) == {"python_push"}
    assert loads.constraints["python_push"]["generalized_force"].shape == (6, 3)
    # The block's weight presses on the ground through the rail.
    assert np.allclose(loads.frame["force"][1], -3.0 * 9.81)
    assert np.allclose(loads.frame["force"][0], 0.0, atol=1e-12)
    assert m.joint_loads_to_dict(result)["constraints"][0]["name"] == "python_push"


# --- planar: offset centre of mass and analysis -------------------------------------------


def _pendulum(offset: bool):
    length, mass, inertia = 0.4, 2.0, 0.03
    m = Mechanism.planar(gravity=(0.0, -9.81))
    g = m.ground()
    if offset:
        arm = m.body("arm", mass=mass, inertia=inertia, center_of_mass=(length, 0.0))
        m.pin(g, arm, name="pivot")
        q0 = np.zeros(6)
    else:
        arm = m.body("arm", mass=mass, inertia=inertia)
        m.pin(g, arm, point_b=(-length, 0.0), name="pivot")
        q0 = np.array([0, 0, 0, length, 0, 0], dtype=float)
    return m, arm, q0


def test_offset_center_of_mass_matches_centroidal_model():
    t = np.linspace(0.0, 1.0, 51)
    results = []
    for offset in (True, False):
        m, _, q0 = _pendulum(offset)
        results.append(m.simulate(t, q0=q0, allow_underconstrained=True))
        assert m.check(results[-1]).passed
    assert np.max(np.abs(results[0].q[5] - results[1].q[5])) < 1e-7


def test_offset_center_of_mass_inverse_dynamics_torque_is_analytic():
    m, arm, q0 = _pendulum(True)
    m.coordinate_drive(arm, "theta", law=Polynomial((0.0, 1.0, 0.5)), name="motor")
    result = m.inverse_dynamics(np.linspace(0.0, 1.5, 16), q0=q0)
    theta, alpha = result.q[5], result.a[5]
    expected = (0.03 + 2.0 * 0.4**2) * alpha + 2.0 * 9.81 * 0.4 * np.cos(theta)
    assert np.allclose(m.joint_loads(result).drives["motor"]["effort"], expected, atol=1e-10)
    assert "offset_center_of_mass" not in {issue.code for issue in m.validate(q0).issues}
    assert mbsd.capabilities()["planar"]["limits"]["offset_center_of_mass_dynamics"] is True


def test_public_analysis_primitives_close_the_equations():
    m, arm, q0 = _pendulum(True)
    m.coordinate_drive(arm, "theta", law=ConstantSpeed(rate=2.0), name="motor")
    result = m.inverse_dynamics(np.linspace(0.0, 0.5, 6), q0=q0)
    for k, ti in enumerate(result.t):
        state = (result.q[:, k], result.v[:, k], result.a[:, k])
        assert np.max(np.abs(analysis.equation_of_motion_residual(m, *state, result.multipliers[:, k], ti))) < 1e-9
        assert np.max(np.abs(analysis.acceleration_residual(m, *state, ti))) < 1e-9
        assert np.max(np.abs(analysis.constraint_residual(m, state[0], ti))) < 1e-9
    q = result.q[:, -1]
    assert np.allclose(analysis.point_position(q, arm, (0.4, 0.0)), 0.4 * np.array([np.cos(q[5]), np.sin(q[5])]))
    assert np.allclose(analysis.body_vector(q, arm, analysis.rotation_matrix(q[5]) @ [1.0, 2.0]), [1.0, 2.0])
    assert analysis.mass_matrix(m, q).shape == (6, 6)
