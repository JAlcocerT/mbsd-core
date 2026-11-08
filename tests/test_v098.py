"""v0.9.8 candidate items F1-F5 (gaps found building the FreeCAD shorts).

F1 planar forward-dynamics accelerations, F2 tabulated laws at their own
domain ends, F3 portable gear / rack meshes, F4 relative drives, and F5
validation-corpus cases: closed forms, exact power / momentum balances and
Newton-Euler on the frame, all with the solver's own accelerations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mbsd import ConstantSpeed, Mechanism, Polynomial, Tabulated
from mbsd.planar import PlanarMechanism, analysis
from mbsd.spatial import Pose3D, Quaternion
from mbsd.spatial.builder import SpatialCoordinateDrive, _adaptive_derivative

SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"
GRAVITY = np.array([0.0, -9.81])


def _frame_newton_euler(mechanism, result, gravity=GRAVITY):
    """Expected frame force and moment: -sum m(a_G - g) and -dH/dt about the origin."""
    bodies = mechanism.model.bodies
    nsteps = result.q.shape[1]
    force = np.zeros((2, nsteps))
    moment = np.zeros(nsteps)
    for k in range(nsteps):
        q, v, a = result.q[:, k], result.v[:, k], result.a[:, k]
        for index in range(1, len(bodies)):
            body = bodies[index]
            com = np.asarray(body.rG, dtype=float)
            r = analysis.point_position(q, index, com)
            acc = analysis.point_acceleration(q, v, a, index, com)
            inertial = body.mass * (acc - gravity)
            force[:, k] -= inertial
            moment[k] -= r[0] * inertial[1] - r[1] * inertial[0] + body.inertia * a[3 * index + 2]
    return force, moment


def _kinetic_energy_rate(mechanism, result):
    """dKE/dt = sum(m v_G . a_G + I omega alpha) from the solved accelerations."""
    rate = np.zeros(result.q.shape[1])
    for k in range(result.q.shape[1]):
        q, v, a = result.q[:, k], result.v[:, k], result.a[:, k]
        for index, body in enumerate(mechanism.model.bodies[1:], start=1):
            vel = analysis.point_velocity(q, v, index, body.rG)
            acc = analysis.point_acceleration(q, v, a, index, body.rG)
            rate[k] += body.mass * vel @ acc + body.inertia * v[3 * index + 2] * a[3 * index + 2]
    return rate


# --- F1: DynamicsResult.a -------------------------------------------------------------


def _pendulum():
    m = Mechanism.planar(gravity=tuple(GRAVITY))
    g = m.ground()
    arm = m.body("arm", mass=2.0, inertia=0.05, center_of_mass=(0.4, 0.0))
    m.pin(g, arm, name="pivot")
    return m, np.zeros(6)


def test_forward_dynamics_returns_solved_accelerations():
    m, q0 = _pendulum()
    t = np.linspace(0.0, 1.0, 201)
    result = m.simulate(t, q0, allow_underconstrained=True, rtol=1e-11, atol=1e-13)
    assert result.a is not None and result.a.shape == result.q.shape
    # The angle obeys I_pivot * theta'' = -m g r cos(theta).
    theta = result.q[5]
    expected = -(2.0 * 9.81 * 0.4 * np.cos(theta)) / (0.05 + 2.0 * 0.4**2)
    assert np.max(np.abs(result.a[5] - expected)) < 1e-9
    # Exact power balance: dKE/dt equals the gravity power.
    gravity_power = np.array(
        [2.0 * GRAVITY @ analysis.point_velocity(result.q[:, k], result.v[:, k], 1, (0.4, 0.0))
         for k in range(len(t))]
    )
    assert np.max(np.abs(_kinetic_energy_rate(m, result) - gravity_power)) < 1e-9
    payload = m.result_to_dict(result)
    assert np.allclose(payload["coordinates"]["a"], result.a)


def test_newton_euler_on_frame_closes_for_forward_dynamics():
    """Short #10: frame load = -sum m(a - g) and -dH/dt, checking every reaction at once."""
    m, q0 = _pendulum()
    t = np.linspace(0.0, 0.8, 81)
    result = m.simulate(t, q0, allow_underconstrained=True, rtol=1e-11, atol=1e-13)
    loads = m.joint_loads(result)
    force, moment = _frame_newton_euler(m, result)
    assert np.max(np.abs(loads.frame["force"] - force)) < 1e-9
    assert np.max(np.abs(loads.frame["moment"] - moment)) < 1e-9


# --- F2: tabulated laws at their domain ends -------------------------------------------


def _tabulated_lift():
    times = np.linspace(0.0, 2.0, 41)
    law = Tabulated(tuple(times), tuple(0.1 * np.sin(times)))
    m = Mechanism.spatial()
    m.body("carriage", mass=3.0, inertia=(0.1, 0.1, 0.1))
    drive = m.coordinate_drive(0, "z", law=law, name="lift")
    return m, drive, law


def test_spatial_tabulated_drive_runs_over_its_whole_domain():
    m, drive, law = _tabulated_lift()
    for t in (0.0, 2.0):
        assert drive.validate_derivative(t) == "validated"
        assert drive.acceleration_value(t) == pytest.approx(law.acceleration(t))
    result = m.simulate(np.linspace(0.0, 2.0, 21))
    effort = m.drive_efforts(result)["lift"]["effort"]
    assert np.allclose(effort, [3.0 * law.acceleration(x) for x in result.t], atol=1e-6)


def test_one_sided_derivative_still_catches_inconsistent_velocity():
    times = np.linspace(0.0, 1.0, 11)
    good = Tabulated(tuple(times), tuple(times**2))
    # Value from the table (so the check knows its domain), wrong velocity.
    drive = SpatialCoordinateDrive(0, "x", good.value, lambda t: 5.0, name="bad", law=good)
    for t in (0.0, 1.0):
        with pytest.raises(ValueError, match="inconsistent"):
            drive.validate_derivative(t)
    for t, slope in ((0.0, 0.0), (1.0, 2.0)):
        numeric, _, converged = _adaptive_derivative(good.value, t, "value", good.domain)
        assert converged and numeric == pytest.approx(slope, abs=1e-8)


# --- F3: gear and rack meshes -----------------------------------------------------------


def _gear_pair(ratio=-15.0 / 45.0):
    m = Mechanism.planar(gravity=(0.0, 0.0))
    g = m.ground()
    a = m.body("pinion", mass=1.0, inertia=0.02)
    b = m.body("wheel", mass=2.0, inertia=0.3)
    m.pin(g, a, name="axle_a")
    m.pin(g, b, point_a=(0.3, 0.0), name="axle_b")
    m.gear(a, b, ratio, name="mesh")
    m.coordinate_drive(a, "theta", law=Polynomial((0.0, 0.0, 2.0)), name="motor")
    return m, np.array([0, 0, 0, 0, 0, 0, 0.3, 0, 0], dtype=float)


def test_gear_mesh_is_solved_portable_and_reports_torques(tmp_path):
    m, q0 = _gear_pair()
    report = m.validate(q0)
    assert report.ok and report.portable
    t = np.linspace(0.0, 1.0, 11)
    result = m.inverse_dynamics(t, q0=q0)
    assert m.check(result).passed
    assert np.max(np.abs(result.q[8] + result.q[5] / 3.0)) < 1e-12
    loads = m.joint_loads(result)
    mesh = loads.meshes["mesh"]
    alpha_a, alpha_b = 4.0, -4.0 / 3.0
    assert np.allclose(mesh["effort_on_b"], 0.3 * alpha_b, atol=1e-12)
    # Power balance: the motor feeds both inertias.
    assert np.allclose(loads.drives["motor"]["effort"], 0.02 * alpha_a + 0.3 * alpha_b * (-1 / 3), atol=1e-12)
    # The mesh reacts on its ground carrier: frame moment = -dH/dt.
    force, moment = _frame_newton_euler(m, result, gravity=np.zeros(2))
    assert np.allclose(loads.frame["moment"], moment, atol=1e-12)
    assert np.allclose(loads.frame["force"], force, atol=1e-12)

    payload = m.to_dict()
    assert payload["schema_version"] == 3
    assert payload["user_constraints"][0]["kind"] == "gear"
    replay = PlanarMechanism.from_json(m.to_json(tmp_path / "gears.json"))
    assert np.array_equal(replay.inverse_dynamics(t, q0=q0).multipliers, result.multipliers)


def test_reverser_on_the_frame_cancels_frame_torque():
    """Short #5: counter-rotating sets cancel the frame torque; same-direction ones double it."""
    torques = {}
    for label, ratio in (("opposite", -1.0), ("same", 1.0)):
        m = Mechanism.planar(gravity=(0.0, 0.0))
        g = m.ground()
        outer = m.body("outer", mass=1.0, inertia=0.4)
        inner = m.body("inner", mass=1.0, inertia=0.4)
        m.pin(g, outer, name="sleeve")
        m.pin(g, inner, name="shaft")
        m.gear(outer, inner, ratio, name="reverser")
        m.coordinate_drive(outer, "theta", law=Polynomial((0.0, 0.0, 1.5)), name="crank")
        result = m.inverse_dynamics(np.linspace(0.0, 1.0, 6), q0=np.zeros(9))
        loads = m.joint_loads(result)
        _, moment = _frame_newton_euler(m, result, gravity=np.zeros(2))
        assert np.allclose(loads.frame["moment"], moment, atol=1e-12)
        torques[label] = np.max(np.abs(loads.frame["moment"]))
    assert torques["opposite"] < 1e-12
    assert torques["same"] == pytest.approx(2 * 0.4 * 3.0)


def test_gear_with_moving_carrier_is_an_open_differential():
    m = Mechanism.planar(gravity=(0.0, 0.0))
    g = m.ground()
    case = m.body("case", mass=1.0, inertia=0.05)
    left = m.body("left", mass=1.0, inertia=0.01)
    right = m.body("right", mass=1.0, inertia=0.01)
    for body, name in ((case, "case_bearing"), (left, "left_bearing"), (right, "right_bearing")):
        m.pin(g, body, name=name)
    m.gear(left, right, -1.0, carrier=case, name="spiders")
    m.coordinate_drive(case, "theta", law=ConstantSpeed(rate=10.0), name="ring")
    m.coordinate_drive(left, "theta", law=Polynomial((0.0, 12.0)), name="left_road")
    result = m.inverse_dynamics(np.linspace(0.0, 0.5, 6), q0=np.zeros(12))
    case_angle, left_angle, right_angle = result.q[5], result.q[8], result.q[11]
    assert np.allclose(left_angle + right_angle, 2.0 * case_angle, atol=1e-12)
    assert np.allclose(right_angle, (20.0 - 12.0) * result.t, atol=1e-12)
    mesh = m.joint_loads(result).meshes["spiders"]
    assert np.allclose(mesh["effort_on_a"] + mesh["effort_on_b"] + mesh["effort_on_carrier"], 0.0, atol=1e-12)


def _rack_and_pinion():
    m = Mechanism.planar(gravity=tuple(GRAVITY))
    g = m.ground()
    pinion = m.body("pinion", mass=1.0, inertia=0.01)
    rack = m.body("rack", mass=5.0, inertia=0.1)
    m.pin(g, pinion, name="axle")
    m.slider(g, rack, axis=(1.0, 0.0), point_rail=(0.0, -0.05), name="guide")
    m.rack(pinion, rack, 0.05, name="teeth")
    m.coordinate_drive(pinion, "theta", law=Polynomial((0.0, 0.0, 3.0)), name="motor")
    return m, np.array([0, 0, 0, 0, 0, 0, 0, -0.05, 0], dtype=float)


def test_rack_mesh_reports_axial_force_and_closes_the_frame(tmp_path):
    m, q0 = _rack_and_pinion()
    assert m.validate(q0).ok
    t = np.linspace(0.0, 0.5, 11)
    result = m.inverse_dynamics(t, q0=q0)
    assert m.check(result).passed
    assert np.max(np.abs(result.q[6] - 0.05 * result.q[5])) < 1e-12
    loads = m.joint_loads(result)
    rack_accel = 0.05 * 6.0
    assert np.allclose(loads.meshes["teeth"]["effort_on_b"], 5.0 * rack_accel, atol=1e-10)
    assert np.allclose(loads.drives["motor"]["effort"], 0.01 * 6.0 + 0.05 * 5.0 * rack_accel, atol=1e-10)
    force, moment = _frame_newton_euler(m, result)
    assert np.allclose(loads.frame["force"], force, atol=1e-9)
    assert np.allclose(loads.frame["moment"], moment, atol=1e-9)
    replay = PlanarMechanism.from_dict(json.loads(json.dumps(m.to_dict())))
    assert np.array_equal(replay.inverse_dynamics(t, q0=q0).multipliers, result.multipliers)


def test_joint_loads_v3_payload_matches_schema_contract():
    m, q0 = _rack_and_pinion()
    payload = m.joint_loads_to_dict(m.inverse_dynamics(np.linspace(0.0, 0.1, 3), q0=q0))
    schema = json.loads((SCHEMAS / "mbsd.planar.joint-loads-v3.schema.json").read_text())
    assert payload["schema_version"] == schema["properties"]["schema_version"]["const"] == 3
    assert set(schema["required"]) <= payload.keys()
    for key, value in schema["properties"]["conventions"]["properties"].items():
        assert payload["conventions"][key] == value["const"]
    mesh = payload["meshes"][0]
    assert set(schema["$defs"]["mesh"]["required"]) <= mesh.keys()
    assert mesh["kind"] == "rack" and mesh["effort_units"] == ["N*m", "N", "N*m"]


def test_gear_and_rack_reject_degenerate_input():
    m = Mechanism.planar()
    g = m.ground()
    a, b = m.body("a"), m.body("b")
    with pytest.raises(ValueError, match="three different bodies"):
        m.gear(a, g, -1.0)
    with pytest.raises(ValueError, match="non-zero"):
        m.gear(a, b, 0.0)
    with pytest.raises(ValueError, match="moving pinion"):
        m.rack(g, b, 0.05)
    with pytest.raises(ValueError, match="non-zero"):
        m.rack(a, b, 0.05, axis=(0.0, 0.0))


# --- F4: relative drives ----------------------------------------------------------------


def _coaxial_pair(relative: bool):
    m = Mechanism.planar(gravity=(0.0, 0.0))
    g = m.ground()
    stator = m.body("stator", mass=1.0, inertia=0.5)
    rotor = m.body("rotor", mass=1.0, inertia=0.1)
    m.pin(g, stator, name="bearing")
    m.pin(stator, rotor, name="shaft")
    law = Polynomial((0.0, 0.0, 4.0))
    if relative:
        m.coordinate_drive(rotor, "theta", law=law, relative_to=stator, name="motor")
    else:
        m.coordinate_drive(rotor, "theta", law=law, name="motor")
    return m


def test_relative_drive_keeps_the_reaction_inside_the_system(tmp_path):
    """Short #4: an internal motor conserves angular momentum; an absolute one cannot."""
    m = _coaxial_pair(relative=True)
    t = np.linspace(0.0, 1.0, 51)
    result = m.simulate(t, np.zeros(9), v0=np.zeros(9), allow_underconstrained=True, rtol=1e-11, atol=1e-13)
    momentum = 0.5 * result.v[5] + 0.1 * result.v[8]
    assert np.max(np.abs(momentum)) < 1e-10
    assert np.allclose(result.v[8] - result.v[5], 8.0 * t, atol=1e-8)
    loads = m.joint_loads(result)
    assert np.max(np.abs(loads.frame["moment"])) < 1e-10
    motor = loads.drives["motor"]
    assert motor["relative_to"] == 1
    relative_rate = result.v[8] - result.v[5]
    assert np.allclose(motor["power"], motor["effort"] * relative_rate)
    assert np.allclose(motor["power"], _kinetic_energy_rate(m, result), atol=1e-9)

    payload = m.joint_loads_to_dict(result)
    assert payload["drives"][0]["relative_to"] == {"index": 1, "name": "stator"}
    replay = PlanarMechanism.from_json(m.to_json(tmp_path / "pair.json"))
    assert replay.to_dict()["user_constraints"][0]["relative_to"] == 1

    # The same motor mounted absolutely reacts on ground: the frame feels it.
    absolute = _coaxial_pair(relative=False)
    grounded = absolute.simulate(t, np.zeros(9), v0=np.zeros(9), allow_underconstrained=True)
    assert np.max(np.abs(absolute.joint_loads(grounded).frame["moment"])) > 0.1

    with pytest.raises(ValueError, match="theta drives only"):
        m.coordinate_drive(2, "x", law=ConstantSpeed(rate=0.0), relative_to=1)


def test_relative_motor_matches_coordinate_drive():
    m = Mechanism.planar(gravity=(0.0, 0.0))
    g = m.ground()
    arm = m.body("arm", mass=1.0, inertia=0.2)
    tool = m.body("tool", mass=0.5, inertia=0.05)
    m.pin(g, arm, name="base")
    m.pin(arm, tool, point_a=(0.5, 0.0), name="wrist")
    m.motor(arm, omega=1.0, name="shoulder")
    m.motor(tool, omega=2.0, relative_to=arm, name="wrist_motor")
    q0 = np.array([0, 0, 0, 0, 0, 0, 0.5, 0, 0], dtype=float)
    result = m.inverse_dynamics(np.linspace(0.0, 1.0, 11), q0=q0)
    assert np.allclose(result.q[8] - result.q[5], 2.0 * result.t, atol=1e-12)
    assert m.to_dict()["user_constraints"][1]["relative_to"] == 1


# --- F5: corpus cases from the shorts -------------------------------------------------


def test_offset_slider_crank_matches_closed_form_and_power_balance():
    """Shorts #1 and #8: kinematics against the closed form, exact power balance."""
    r, l, e = 0.1, 0.35, 0.02
    m = Mechanism.planar(gravity=tuple(GRAVITY))
    g = m.ground()
    crank = m.body("crank", mass=1.0, inertia=0.004, center_of_mass=(r / 2, 0.0))
    rod = m.body("rod", mass=0.8, inertia=0.01, center_of_mass=(l / 2, 0.0))
    ram = m.body("ram", mass=2.0, inertia=0.01)
    m.pin(g, crank, name="main")
    m.pin(crank, rod, point_a=(r, 0.0), name="crankpin")
    m.pin(rod, ram, point_a=(l, 0.0), name="wristpin")
    m.slider(g, ram, axis=(1.0, 0.0), point_rail=(0.0, e), name="bore")
    m.motor(crank, omega=12.0, name="engine")
    theta0 = 0.0
    x0 = r + math.sqrt(l**2 - (e - 0.0) ** 2)
    phi0 = math.asin(e / l)
    q0 = np.array([0, 0, 0, 0, 0, theta0, r, 0, phi0, x0, e, 0], dtype=float)
    t = np.linspace(0.0, 0.6, 61)
    result = m.inverse_dynamics(t, q0=q0)
    assert m.check(result).passed
    theta = result.q[5]
    closed = r * np.cos(theta) + np.sqrt(l**2 - (e - r * np.sin(theta)) ** 2)
    assert np.max(np.abs(result.q[9] - closed)) < 1e-9
    loads = m.joint_loads(result)
    gravity_power = np.zeros_like(t)
    for k in range(len(t)):
        for index, body in enumerate(m.model.bodies[1:], start=1):
            vel = analysis.point_velocity(result.q[:, k], result.v[:, k], index, body.rG)
            gravity_power[k] += body.mass * GRAVITY @ vel
    balance = loads.drives["engine"]["power"] + gravity_power - _kinetic_energy_rate(m, result)
    assert np.max(np.abs(balance)) < 1e-9
    force, moment = _frame_newton_euler(m, result)
    assert np.allclose(loads.frame["force"], force, atol=1e-8)
    assert np.allclose(loads.frame["moment"], moment, atol=1e-8)


def test_ground_pin_force_is_the_force_on_the_wall():
    """Short #10 sign convention: force_on_i of a ground pin is what the wall feels."""
    m = Mechanism.planar(gravity=tuple(GRAVITY))
    g = m.ground()
    shelf = m.body("shelf", mass=10.0, inertia=0.5, center_of_mass=(0.3, 0.0))
    m.pin(g, shelf, name="hinge")
    m.motor(shelf, omega=0.0, name="hold")
    loads = m.joint_loads(m.inverse_dynamics(np.linspace(0.0, 0.1, 3), q0=np.zeros(6)))
    hinge = loads.joints["hinge"]
    assert hinge["body_i"] == 0
    assert np.allclose(hinge["force_on_i"][1], -10.0 * 9.81)  # the wall is pulled down
    assert np.allclose(hinge["force_on_j"][1], 10.0 * 9.81)   # the shelf is held up


def test_mirrored_hinges_report_equal_torques_about_their_own_axes():
    """Short #6: torques are axial vectors; under a mirror the drive axis flips too."""
    law = Polynomial((0.0, 0.0, 2.0))
    m = Mechanism.spatial()
    g = m.ground()
    a = m.body("a", mass=2.0, inertia=(0.01, 0.2, 0.2), pose=Pose3D(translation=(0.5, 0.0, 0.0)))
    b = m.body("b", mass=2.0, inertia=(0.01, 0.2, 0.2), pose=Pose3D(translation=(-0.5, 0.0, 0.0)))
    flip = Quaternion.from_axis_angle((0.0, 1.0, 0.0), math.pi)
    hinge_a = m.revolute(g, a, frame_j=Pose3D(translation=(-0.5, 0.0, 0.0)), name="hinge_a")
    hinge_b = m.revolute(
        g, b,
        frame_i=Pose3D(rotation=flip),
        frame_j=Pose3D(translation=(0.5, 0.0, 0.0), rotation=flip),
        name="hinge_b",
    )
    m.revolute_drive(hinge_a, law=law, name="servo_a")
    m.revolute_drive(hinge_b, law=law, name="servo_b")
    result = m.simulate(np.linspace(0.0, 0.5, 11), gravity_world=(0.0, -9.81, 0.0), constraint_tolerance=1e-11)
    efforts = m.drive_efforts(result)
    tau_a, tau_b = efforts["servo_a"]["effort"], efforts["servo_b"]["effort"]
    # Same law about mirrored axes: equal torques about each drive's own axis,
    # which is tau_a + tau_b = 0 about the common world +z axis.
    assert np.max(np.abs(np.asarray(tau_a) - np.asarray(tau_b))) < 1e-9
    for k in (0, 5, 10):
        xa = result.poses[k][1].translation
        xb = result.poses[k][2].translation
        assert np.allclose(xb, xa * np.array([-1.0, 1.0, 1.0]), atol=1e-9)
