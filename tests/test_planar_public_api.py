import csv
import json
from importlib.metadata import version

import numpy as np
import pytest

import mbsd
from mbsd import Mechanism, MechanismSolveError, Spring
from mbsd.planar.constraints import constraints
from mbsd.planar.derivatives import dt_constraints, dt_jacobian, dtdt_constraints
from mbsd.planar.jacobians import jacobian
from mbsd.planar.kinematics import rot_mat
from mbsd.planar.model import Body, GearJoint, MBody, PrismJoint, UserConstraint
from mbsd.planar.synthesis import (
    FourBar,
    affine_fit,
    freudenstein_3pt,
    rocker_angles,
)


def test_rotation_matrix_is_proper():
    A = rot_mat(0.37)
    np.testing.assert_allclose(A @ A.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(A), 1.0, atol=1e-12)


def test_top_level_public_api_is_small():
    assert sorted(mbsd.__all__) == ["Mechanism", "MechanismSolveError", "Spring"]


def test_package_version_matches_installed_metadata():
    assert mbsd.__version__ == version("mbsd")


def test_driven_slider_tracks_prescribed_x_motion():
    omega = 2.0 * np.pi
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    ground = mechanism.ground()
    slider = mechanism.body("slider", mass=1.0, inertia=0.01)

    mechanism.slider(ground, slider, axis=(1.0, 0.0))
    mechanism.coordinate_drive(
        slider,
        "x",
        value=lambda t: np.sin(omega * t),
        velocity=lambda t: omega * np.cos(omega * t),
        acceleration=lambda t: -(omega**2) * np.sin(omega * t),
    )

    t = np.linspace(0.0, 1.0, 61)
    result = mechanism.solve_kinematics(t)

    np.testing.assert_allclose(result.q[3, :], np.sin(omega * t), atol=1e-9)
    np.testing.assert_allclose(result.q[4, :], 0.0, atol=1e-9)
    np.testing.assert_allclose(result.q[5, :], 0.0, atol=1e-9)
    assert mechanism.max_constraint_residual(result) < 1e-9
    mechanism.assert_constraints_satisfied(result, tol=1e-9)

    residuals = mechanism.constraint_residuals(result)
    diagnostics = mechanism.diagnostics(result)

    assert residuals.shape == (mechanism.nrestr, len(t))
    assert diagnostics.coordinates == mechanism.ncoord
    assert diagnostics.constraints == mechanism.nrestr
    assert diagnostics.steps == len(t)
    assert diagnostics.finite
    assert diagnostics.max_constraint_residual < 1e-9
    assert diagnostics.as_dict()["degrees_of_freedom"] == 0


def test_mass_spring_dynamics_runs_from_public_api():
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    ground = mechanism.ground()
    mass = mechanism.body("mass", mass=1.0, inertia=0.01)
    mechanism.slider(ground, mass, axis=(1.0, 0.0))

    spring = Spring(
        i=int(ground),
        j=int(mass),
        k=10.0,
        c=0.5,
        l0=1.0,
        ri=np.array([0.0, 0.0]),
        rj=np.array([0.0, 0.0]),
    )

    q0 = np.zeros(mechanism.ncoord)
    q0[3] = 1.5
    v0 = np.zeros(mechanism.ncoord)
    t = np.linspace(0.0, 0.15, 25)

    result = mechanism.simulate(
        t,
        q0=q0,
        v0=v0,
        springs=[spring],
        allow_underconstrained=True,
    )

    assert result.q.shape == (mechanism.ncoord, len(t))
    assert np.all(np.isfinite(result.q))
    assert result.q[3, -1] < result.q[3, 0]
    np.testing.assert_allclose(result.q[4, :], 0.0, atol=1e-9)
    np.testing.assert_allclose(result.q[5, :], 0.0, atol=1e-9)
    assert mechanism.max_constraint_residual(result) < 1e-8


def test_undamped_mass_spring_energy_is_reasonably_conserved():
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    ground = mechanism.ground()
    mass = mechanism.body("mass", mass=1.0, inertia=0.01)
    mechanism.slider(ground, mass, axis=(1.0, 0.0))

    spring = Spring(
        i=int(ground),
        j=int(mass),
        k=10.0,
        c=0.0,
        l0=1.0,
        ri=np.array([0.0, 0.0]),
        rj=np.array([0.0, 0.0]),
    )

    q0 = np.zeros(mechanism.ncoord)
    q0[3] = 1.5
    v0 = np.zeros(mechanism.ncoord)
    t = np.linspace(0.0, 1.0, 101)

    result = mechanism.simulate(
        t,
        q0=q0,
        v0=v0,
        springs=[spring],
        allow_underconstrained=True,
    )
    x = result.q[3, :]
    vx = result.v[3, :]
    energy = 0.5 * vx**2 + 0.5 * spring.k * (x - spring.l0) ** 2

    assert np.ptp(energy) / energy[0] < 0.02


def test_public_model_jacobian_matches_finite_difference():
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    ground = mechanism.ground()
    link = mechanism.body("link", mass=1.0, inertia=0.01)
    mechanism.pin(ground, link, point=(0.0, 0.0))
    mechanism.motor(link, omega=3.0)

    t = 0.2
    q = mechanism.solve_position(t=t)
    Cq = jacobian(mechanism.model, q, t)

    h = 1e-7
    C0 = constraints(mechanism.model, q, t)
    Cq_fd = np.zeros_like(Cq)
    for col in range(mechanism.ncoord):
        q_pert = q.copy()
        q_pert[col] += h
        Cq_fd[:, col] = (constraints(mechanism.model, q_pert, t) - C0) / h

    np.testing.assert_allclose(Cq, Cq_fd, atol=1e-6)


def test_offset_prismatic_dt_jacobian_matches_finite_difference():
    mbody = MBody()
    mbody.bodies = [
        Body("ground"),
        Body("slider", mass=1.0, inertia=1.0),
    ]
    mbody.prism_joints.append(
        PrismJoint(
            i=0,
            j=1,
            ri=np.array([0.2, -0.1]),
            rj=np.array([-0.3, 0.4]),
            hi=np.array([0.6, 0.8]),
        )
    )
    q = np.array([0.1, -0.2, 0.3, 0.8, 0.4, -0.1])
    v = np.array([0.2, 0.3, 0.4, -0.1, 0.2, -0.3])

    h = 1e-6
    DCq = dt_jacobian(mbody, q, v, 0.0)
    DCq_fd = (jacobian(mbody, q + h * v, 0.0) - jacobian(mbody, q - h * v, 0.0)) / (
        2.0 * h
    )

    np.testing.assert_allclose(DCq, DCq_fd, atol=1e-8)


def test_offset_slider_acceleration_residual_stays_small():
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    ground = mechanism.ground()
    slider = mechanism.body("slider", mass=1.0, inertia=0.01)
    mechanism.slider(
        ground,
        slider,
        axis=(1.0, 0.0),
        point_rail=(0.1, 0.2),
        point_slider=(-0.3, 0.4),
    )
    mechanism.coordinate_drive(
        slider,
        "x",
        value=lambda t: 0.5 + 0.2 * np.sin(t),
        velocity=lambda t: 0.2 * np.cos(t),
        acceleration=lambda t: -0.2 * np.sin(t),
    )
    q0 = np.zeros(mechanism.ncoord)
    q0[3] = 0.5
    q0[4] = -0.2

    result = mechanism.solve_kinematics(np.linspace(0.0, 0.5, 21), q0=q0)
    max_acceleration_residual = 0.0
    for step, ti in enumerate(result.t):
        residual = (
            jacobian(mechanism.model, result.q[:, step], float(ti)) @ result.a[:, step]
            + dt_jacobian(mechanism.model, result.q[:, step], result.v[:, step], float(ti))
            @ result.v[:, step]
            + dtdt_constraints(mechanism.model, result.q[:, step], result.v[:, step], float(ti))
        )
        max_acceleration_residual = max(
            max_acceleration_residual,
            float(np.linalg.norm(residual, ord=np.inf)),
        )

    assert max_acceleration_residual < 1e-8


def test_bad_inputs_raise_clear_errors():
    mechanism = Mechanism.planar()
    ground = mechanism.ground()

    with pytest.raises(ValueError, match="mass must be positive"):
        mechanism.body("bad", mass=0.0)

    body = mechanism.body("body")

    with pytest.raises(IndexError, match="out of range"):
        mechanism.pin(ground, 99)

    with pytest.raises(ValueError, match="axis must be non-zero"):
        mechanism.slider(ground, body, axis=(0.0, 0.0))

    with pytest.raises(ValueError, match="strictly increasing"):
        mechanism.solve_kinematics(np.array([0.0, 0.0]))

    with pytest.raises(ValueError, match="shape"):
        mechanism.simulate(np.array([0.0, 0.1]), q0=np.zeros(mechanism.ncoord + 1))


def test_underconstrained_dynamics_requires_explicit_opt_in():
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    ground = mechanism.ground()
    mass = mechanism.body("mass", mass=1.0, inertia=0.01)
    mechanism.slider(ground, mass, axis=(1.0, 0.0))

    spring = Spring(
        i=int(ground),
        j=int(mass),
        k=10.0,
        c=0.0,
        l0=1.0,
        ri=np.array([0.0, 0.0]),
        rj=np.array([0.0, 0.0]),
    )
    q0 = np.zeros(mechanism.ncoord)
    q0[3] = 1.5

    with pytest.raises(MechanismSolveError, match="Underconstrained position solve"):
        mechanism.simulate(np.array([0.0, 0.1]), q0=q0, springs=[spring])


def test_dynamics_rejects_offset_center_of_mass_for_now():
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    ground = mechanism.ground()
    body = mechanism.body("offset", mass=1.0, inertia=0.01, center_of_mass=(0.2, 0.0))
    mechanism.slider(ground, body, axis=(1.0, 0.0))

    q0 = np.zeros(mechanism.ncoord)
    q0[3] = 1.0

    with pytest.raises(MechanismSolveError, match="center of mass"):
        mechanism.simulate(
            np.array([0.0, 0.1]),
            q0=q0,
            allow_underconstrained=True,
        )


def test_dynamics_rejects_inconsistent_initial_velocity():
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    ground = mechanism.ground()
    slider = mechanism.body("slider", mass=1.0, inertia=0.01)
    mechanism.slider(ground, slider, axis=(1.0, 0.0))

    q0 = np.zeros(mechanism.ncoord)
    q0[3] = 1.0
    v0 = np.zeros(mechanism.ncoord)
    v0[4] = 1.0

    with pytest.raises(MechanismSolveError, match="Initial velocity violates"):
        mechanism.simulate(
            np.array([0.0, 0.1]),
            q0=q0,
            v0=v0,
            allow_underconstrained=True,
        )


def test_model_diagnostics_reports_rank_and_dof():
    mechanism = Mechanism.planar()
    ground = mechanism.ground()
    slider = mechanism.body("slider")
    mechanism.slider(ground, slider, axis=(1.0, 0.0))
    q0 = np.zeros(mechanism.ncoord)
    diagnostics = mechanism.model_diagnostics(q0)

    assert diagnostics.coordinates == mechanism.ncoord
    assert diagnostics.constraints == mechanism.nrestr
    assert diagnostics.degrees_of_freedom == 1
    assert diagnostics.nominal_degrees_of_freedom == 1
    assert diagnostics.rank_degrees_of_freedom == 1
    assert diagnostics.jacobian_rank == mechanism.nrestr
    assert not diagnostics.singular
    assert diagnostics.as_dict()["nominal_degrees_of_freedom"] == 1
    assert diagnostics.as_dict()["rank_degrees_of_freedom"] == 1
    assert diagnostics.as_dict()["jacobian_rank"] == mechanism.nrestr


def test_user_constraint_derivative_rows_follow_lower_level_constraints():
    mbody = MBody()
    mbody.bodies = [
        Body("ground"),
        Body("gear_a", mass=1.0, inertia=1.0),
    ]
    mbody.gear_joints.append(GearJoint(i=0, j=1, tau=2.0))

    def constraint(_mb, q, _t):
        return np.array([q[3] - 1.0])

    def jacobian_fn(mb, _q, _t):
        J = np.zeros((1, mb.ncoord))
        J[0, 3] = 1.0
        return J

    def dt_jacobian_fn(mb, _q, _v, _t):
        J = np.zeros((1, mb.ncoord))
        J[0, 4] = 7.0
        return J

    def dt_constraint_fn(_mb, _q, _t):
        return np.array([3.0])

    def dtdt_constraint_fn(_mb, _q, _v, _t):
        return np.array([5.0])

    mbody.user_constraints.append(
        UserConstraint(
            constraint,
            jacobian_fn,
            dt_jacobian_fn,
            dt_constraint_fn,
            dtdt_constraint_fn,
            count=1,
        )
    )
    q = np.zeros(mbody.ncoord)
    v = np.zeros(mbody.ncoord)

    np.testing.assert_allclose(dt_constraints(mbody, q, 0.0), [0.0, 0.0, 0.0, 0.0, 3.0])
    np.testing.assert_allclose(
        dtdt_constraints(mbody, q, v, 0.0),
        [0.0, 0.0, 0.0, 0.0, 5.0],
    )
    DCq = dt_jacobian(mbody, q, v, 0.0)
    assert DCq[3, 4] == 0.0
    assert DCq[4, 4] == 7.0


def test_constraint_assertion_reports_bad_result():
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    ground = mechanism.ground()
    slider = mechanism.body("slider")
    mechanism.slider(ground, slider, axis=(1.0, 0.0))
    mechanism.coordinate_drive(
        slider,
        "x",
        value=lambda t: t,
        velocity=lambda _t: 1.0,
        acceleration=lambda _t: 0.0,
    )

    result = mechanism.solve_kinematics(np.linspace(0.0, 0.2, 5))
    bad_q = result.q.copy()
    bad_q[4, 2] = 1.0
    bad = type(result)(t=result.t, q=bad_q, v=result.v, a=result.a)

    with pytest.raises(MechanismSolveError, match="exceeds tolerance"):
        mechanism.assert_constraints_satisfied(bad)

    with pytest.raises(ValueError, match="tol must be positive"):
        mechanism.assert_constraints_satisfied(result, tol=0.0)


def test_diagnostics_rejects_mismatched_result_shape():
    mechanism = Mechanism.planar()
    mechanism.ground()
    bad = type("BadResult", (), {})()
    bad.t = np.array([0.0, 0.1])
    bad.q = np.zeros((mechanism.ncoord + 1, 2))

    with pytest.raises(ValueError, match="result.q must have shape"):
        mechanism.diagnostics(bad)


def test_diagnostics_validate_velocity_and_acceleration_arrays():
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    ground = mechanism.ground()
    slider = mechanism.body("slider")
    mechanism.slider(ground, slider, axis=(1.0, 0.0))
    mechanism.coordinate_drive(
        slider,
        "x",
        value=lambda t: t,
        velocity=lambda _t: 1.0,
        acceleration=lambda _t: 0.0,
    )
    result = mechanism.solve_kinematics(np.linspace(0.0, 0.2, 5))

    bad_v = result.v.copy()
    bad_v[0, 0] = np.nan
    with pytest.raises(ValueError, match="result.v must contain only finite values"):
        mechanism.diagnostics(type(result)(t=result.t, q=result.q, v=bad_v, a=result.a))

    bad_a = result.a.copy()
    bad_a[0, 0] = np.inf
    with pytest.raises(ValueError, match="result.a must contain only finite values"):
        mechanism.diagnostics(type(result)(t=result.t, q=result.q, v=result.v, a=bad_a))


def test_planar_mechanism_and_result_exports_are_json_ready(tmp_path):
    mechanism = Mechanism.planar(gravity=(0.0, 0.0))
    ground = mechanism.ground()
    slider = mechanism.body("slider", mass=1.0, inertia=0.01)
    mechanism.slider(
        ground,
        slider,
        axis=(1.0, 0.0),
        point_rail=(0.1, 0.2),
        point_slider=(-0.2, 0.3),
    )
    mechanism.coordinate_drive(
        slider,
        "x",
        value=lambda t: 0.5 + t,
        velocity=lambda _t: 1.0,
        acceleration=lambda _t: 0.0,
    )
    q0 = np.zeros(mechanism.ncoord)
    q0[3] = 0.5
    q0[4] = -0.1
    result = mechanism.solve_kinematics(np.linspace(0.0, 0.2, 3), q0=q0)

    mechanism_payload = mechanism.to_dict()
    result_payload = mechanism.result_to_dict(result)

    assert mechanism_payload["schema"] == "mbsd.planar.mechanism"
    assert mechanism_payload["counts"]["bodies"] == 2
    assert mechanism_payload["joints"]["prismatic"][0]["axis"] == [1.0, -0.0]
    assert mechanism_payload["user_constraints"][0]["kind"] == "coordinate_drive"
    assert result_payload["schema"] == "mbsd.planar.result"
    assert result_payload["result_type"] == "kinematic"
    assert result_payload["diagnostics"]["max_constraint_residual"] < 1e-9
    assert result_payload["body_poses"][1]["name"] == "slider"

    mechanism_json = mechanism.to_json(tmp_path / "mechanism.json")
    result_json = mechanism.result_to_json(result, tmp_path / "result.json")
    trajectory_csv = mechanism.result_to_csv(result, tmp_path / "trajectory.csv")

    assert json.loads(mechanism_json.read_text(encoding="utf-8"))["schema_version"] == 1
    assert json.loads(result_json.read_text(encoding="utf-8"))["coordinates"]["a"]
    with trajectory_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][:4] == ["time", "body0_ground_x", "body0_ground_y", "body0_ground_theta"]
    assert len(rows) == len(result.t) + 1


def test_slider_axis_is_normalized():
    mechanism = Mechanism.planar()
    ground = mechanism.ground()
    slider = mechanism.body("slider")
    joint = mechanism.slider(ground, slider, axis=(2.0, 0.0))

    np.testing.assert_allclose(joint.hi, np.array([0.0, 1.0]))


def test_direct_underconstrained_position_solve_requires_explicit_opt_in():
    mechanism = Mechanism.planar()
    ground = mechanism.ground()
    slider = mechanism.body("slider")
    mechanism.slider(ground, slider, axis=(1.0, 0.0))
    q0 = np.zeros(mechanism.ncoord)

    with pytest.raises(MechanismSolveError, match="Underconstrained position solve"):
        mechanism.solve_position(q0)

    q = mechanism.solve_position(q0, allow_underconstrained=True)
    assert q.shape == (mechanism.ncoord,)


def test_four_bar_synthesis_recovers_known_precision_points():
    known = FourBar(ground=1.0, crank=0.3, coupler=0.85, rocker=0.7)
    theta = np.radians([70.0, 130.0, 190.0])
    psi = rocker_angles(known, theta)

    synthesized = freudenstein_3pt(zip(theta, psi), ground=known.ground)
    np.testing.assert_allclose(synthesized.lengths, known.lengths, atol=1e-10)


def test_four_bar_synthesis_helpers_validate_inputs():
    with pytest.raises(ValueError, match="positive"):
        FourBar(ground=1.0, crank=0.0, coupler=0.8, rocker=0.7)

    with pytest.raises(ValueError, match="exactly three"):
        freudenstein_3pt([(0.0, 0.0), (1.0, 1.0)])

    with pytest.raises(ValueError, match="cannot assemble"):
        FourBar(ground=1.0, crank=0.1, coupler=0.1, rocker=0.1).assemble(0.0)


def test_affine_fit_reports_rms_error():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    fit = affine_fit(x, y)

    np.testing.assert_allclose(fit.transform(x), y, atol=1e-12)
    assert fit.rms_error < 1e-12
