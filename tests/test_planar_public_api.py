import numpy as np
import pytest

import mbsd
from mbsd import Mechanism, MechanismSolveError, Spring
from mbsd.planar.constraints import constraints
from mbsd.planar.jacobians import jacobian
from mbsd.planar.kinematics import rot_mat


def test_rotation_matrix_is_proper():
    A = rot_mat(0.37)
    np.testing.assert_allclose(A @ A.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(A), 1.0, atol=1e-12)


def test_top_level_public_api_is_small():
    assert sorted(mbsd.__all__) == ["Mechanism", "MechanismSolveError", "Spring"]


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

    result = mechanism.simulate(t, q0=q0, v0=v0, springs=[spring])

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

    result = mechanism.simulate(t, q0=q0, v0=v0, springs=[spring])
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
