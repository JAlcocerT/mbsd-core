"""The MBSD validation corpus (v1).

Each case builds a model, solves it with explicit settings, and returns:

- ``errors``: deviations from an **independent** reference (closed form,
  separately integrated ODE, conservation law, Newton-Euler balance) with the
  bound each must stay under;
- ``values``: physical results recorded in ``expected-v1.json``, compared on
  every run so a numerical change cannot pass unexplained;
- ``check``: the expected ``check()`` verdict (``True``, ``False``) or ``None``.

Construction identities (a drive reproducing its own law) are kept apart from
independent checks: cases whose reference is only the model's own definition
say so in ``reference``. Run ``python corpus/run.py`` to print the corpus and
``python corpus/run.py --write`` to regenerate the expected file after an
explained change.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from mbsd import ConstantSpeed, Mechanism, MechanismSolveError, Polynomial
from mbsd.planar import PlanarMechanism, analysis
from mbsd.spatial import Pose3D, Quaternion, SpatialMechanism, compose_pose, inverse_pose

CORPUS_VERSION = 1
FIXTURES = Path(__file__).resolve().parent / "fixtures"
G2 = np.array([0.0, -9.81])
ROT = Quaternion.from_axis_angle
Z_TO_X = ROT((0.0, 1.0, 0.0), math.pi / 2)
Z_TO_Y = ROT((1.0, 0.0, 0.0), -math.pi / 2)


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    dimension: int | None
    reference: str
    settings: dict[str, Any]
    run: Callable[[], dict[str, Any]]
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


CASES: dict[str, Case] = {}


def case(id, family, dimension, reference, settings=None, tags=()):
    def register(fn):
        CASES[id] = Case(id, family, dimension, reference, dict(settings or {}), fn,
                         (fn.__doc__ or "").strip().splitlines()[0], tuple(tags))
        return fn
    return register


def outcome(errors=None, values=None, check=None):
    return {"errors": errors or {}, "values": values or {}, "check": check}


# --- planar helpers --------------------------------------------------------------------


def _planar_newton_euler(m, result, gravity=G2):
    """Max deviation of the frame load from -sum m(a_G - g) and -dH/dt."""
    loads = m.joint_loads(result)
    bodies = m.model.bodies
    force_err = moment_err = 0.0
    for k in range(result.q.shape[1]):
        q, v, a = result.q[:, k], result.v[:, k], result.a[:, k]
        force, moment = np.zeros(2), 0.0
        for i in range(1, len(bodies)):
            body = bodies[i]
            r = analysis.point_position(q, i, body.rG)
            inertial = body.mass * (analysis.point_acceleration(q, v, a, i, body.rG) - gravity)
            force -= inertial
            moment -= r[0] * inertial[1] - r[1] * inertial[0] + body.inertia * a[3 * i + 2]
        force_err = max(force_err, float(np.max(np.abs(loads.frame["force"][:, k] - force))))
        moment_err = max(moment_err, abs(float(loads.frame["moment"][k] - moment)))
    return force_err, moment_err, loads


def _planar_power(m, result, gravity=G2):
    """Per sample: dKE/dt and gravity power from the solved accelerations."""
    ke_rate, gravity_power = np.zeros(result.q.shape[1]), np.zeros(result.q.shape[1])
    for k in range(result.q.shape[1]):
        q, v, a = result.q[:, k], result.v[:, k], result.a[:, k]
        for i, body in enumerate(m.model.bodies[1:], start=1):
            vel = analysis.point_velocity(q, v, i, body.rG)
            acc = analysis.point_acceleration(q, v, a, i, body.rG)
            ke_rate[k] += body.mass * vel @ acc + body.inertia * v[3 * i + 2] * a[3 * i + 2]
            gravity_power[k] += body.mass * gravity @ vel
    return ke_rate, gravity_power


def _slider_crank(r=0.1, length=0.35, offset=0.02):
    m = Mechanism.planar(gravity=tuple(G2))
    g = m.ground()
    crank = m.body("crank", mass=1.0, inertia=0.004, center_of_mass=(r / 2, 0.0))
    rod = m.body("rod", mass=0.8, inertia=0.01, center_of_mass=(length / 2, 0.0))
    ram = m.body("ram", mass=2.0, inertia=0.01)
    m.pin(g, crank, name="main")
    m.pin(crank, rod, point_a=(r, 0.0), name="crankpin")
    m.pin(rod, ram, point_a=(length, 0.0), name="wristpin")
    m.slider(g, ram, axis=(1.0, 0.0), point_rail=(0.0, offset), name="bore")
    m.motor(crank, omega=12.0, name="engine")
    q0 = np.array([0, 0, 0, 0, 0, 0, r, 0, math.asin(offset / length),
                   r + math.sqrt(length**2 - offset**2), offset, 0], dtype=float)
    return m, q0


# --- planar cases ----------------------------------------------------------------------


@case("planar.slider_crank_reactions", "planar_reactions", 2,
      "closed-form offset slider-crank position; Newton-Euler on the frame; power balance",
      {"solver": "inverse_dynamics", "samples": 61, "duration_s": 0.6})
def planar_slider_crank_reactions():
    """Offset slider-crank with offset COMs: ram position, frame load and power."""
    r, length, offset = 0.1, 0.35, 0.02
    m, q0 = _slider_crank(r, length, offset)
    result = m.inverse_dynamics(np.linspace(0.0, 0.6, 61), q0=q0)
    theta = result.q[5]
    closed = r * np.cos(theta) + np.sqrt(length**2 - (offset - r * np.sin(theta)) ** 2)
    force_err, moment_err, loads = _planar_newton_euler(m, result)
    ke_rate, gravity_power = _planar_power(m, result)
    power = loads.drives["engine"]["power"]
    return outcome(
        errors={"ram_position_m": (float(np.max(np.abs(result.q[9] - closed))), 1e-9),
                "frame_force_N": (force_err, 1e-8), "frame_moment_Nm": (moment_err, 1e-8),
                "power_balance_W": (float(np.max(np.abs(power + gravity_power - ke_rate))), 1e-9)},
        values={"peak_engine_torque_Nm": float(np.max(np.abs(loads.drives["engine"]["effort"]))),
                "peak_main_pin_force_N": float(np.max(np.linalg.norm(loads.joints["main"]["force_on_j"], axis=0)))},
        check=m.check(result).passed,
    )


@case("planar.offset_com_pendulum", "planar_forward_dynamics", 2,
      "analytic angular acceleration of a compound pendulum; energy conservation",
      {"solver": "simulate", "method": "RK45", "rtol": 1e-11, "atol": 1e-13, "duration_s": 1.0})
def planar_offset_com_pendulum():
    """Compound pendulum with an offset COM, released horizontally."""
    m = Mechanism.planar(gravity=tuple(G2))
    g = m.ground()
    m.body("arm", mass=2.0, inertia=0.05, center_of_mass=(0.4, 0.0))
    m.pin(g, 1, name="pivot")
    result = m.simulate(np.linspace(0.0, 1.0, 201), np.zeros(6), allow_underconstrained=True, rtol=1e-11, atol=1e-13)
    theta = result.q[5]
    pivot_inertia = 0.05 + 2.0 * 0.4**2
    expected = -(2.0 * 9.81 * 0.4 * np.cos(theta)) / pivot_inertia
    energy = 0.5 * pivot_inertia * result.v[5] ** 2 + 2.0 * 9.81 * 0.4 * np.sin(theta)
    return outcome(
        errors={"alpha_rad_s2": (float(np.max(np.abs(result.a[5] - expected))), 1e-9),
                "energy_drift_J": (float(np.max(np.abs(energy - energy[0]))), 1e-8)},
        values={"theta_end_rad": float(theta[-1])},
        check=m.check(result).passed,
    )


@case("planar.distance_drive_boom", "planar_reactions", 2,
      "Newton-Euler on the frame including the actuator base reaction; multiplier identity",
      {"solver": "inverse_dynamics", "samples": 21, "duration_s": 2.0})
def planar_distance_drive_boom():
    """Boom lifted by a portable hydraulic cylinder (distance_drive)."""
    m = Mechanism.planar(gravity=tuple(G2))
    g = m.ground()
    boom = m.body("boom", mass=120.0, inertia=40.0, center_of_mass=(0.3, 0.05))
    m.pin(g, boom, point_b=(-1.0, 0.0), name="boom_pivot")
    base, attach = np.array([0.5, -0.5]), np.array([0.8, 0.0])
    m.distance_drive(g, boom, law=Polynomial((float(np.linalg.norm(attach - base)), 0.05)),
                     point_a=base, point_b=(-0.2, 0.0), name="cylinder")
    result = m.inverse_dynamics(np.linspace(0.0, 2.0, 21), q0=np.array([0, 0, 0, 1.0, 0, 0], dtype=float))
    force_err, moment_err, loads = _planar_newton_euler(m, result)
    ke_rate, gravity_power = _planar_power(m, result)
    cylinder = loads.drives["cylinder"]
    return outcome(
        errors={"frame_force_N": (force_err, 1e-7), "frame_moment_Nm": (moment_err, 1e-7),
                "power_balance_W": (float(np.max(np.abs(cylinder["power"] + gravity_power - ke_rate))), 1e-7)},
        values={"peak_cylinder_force_N": float(np.max(cylinder["effort"]))},
        check=m.check(result).passed,
    )


@case("planar.point_on_line_follower", "planar_reactions", 2,
      "normal reaction of the line only; Newton-Euler on the frame",
      {"solver": "inverse_dynamics", "samples": 9, "duration_s": 0.4})
def planar_point_on_line_follower():
    """Rocker tip riding on a valve stem (point_on_line) against a preload."""
    m = Mechanism.planar(gravity=tuple(G2))
    g = m.ground()
    rocker = m.body("rocker", mass=0.4, inertia=2e-3)
    valve = m.body("valve", mass=0.1, inertia=1e-5)
    m.pin(g, rocker, point_b=(-0.05, 0.0), name="pivot")
    m.slider(g, valve, axis=(0.0, 1.0), point_rail=(0.1, 0.0), name="guide")
    m.point_on_line(rocker, valve, point_a=(0.05, 0.0), direction_b=(1.0, 0.0), name="tip")
    m.coordinate_drive(rocker, "theta", law=Polynomial((0.0, 0.0, 1.0)), name="cam")
    m.constant_force(valve, (0.0, -50.0), name="spring_preload")
    result = m.inverse_dynamics(np.linspace(0.0, 0.4, 9), q0=np.array([0, 0, 0, 0.05, 0, 0, 0.1, 0, 0], dtype=float))
    loads = m.joint_loads(result)
    tip = loads.joints["tip"]
    return outcome(
        errors={"tangential_tip_force_N": (float(np.max(np.abs(tip["force_on_j"][0]))), 1e-9),
                "tip_moment_Nm": (float(np.max(np.abs(tip["moment_on_j"]))), 1e-9)},
        values={"peak_tip_force_N": float(np.max(tip["force_on_j"][1]))},
        check=m.check(result).passed,
    )


@case("planar.gear_train_power", "planar_meshes", 2,
      "gear ratio kinematics; motor power = dKE/dt; frame moment = -dH/dt",
      {"solver": "inverse_dynamics", "samples": 11, "duration_s": 1.0})
def planar_gear_train_power():
    """Pinion driving a wheel through an external mesh (gear)."""
    m = Mechanism.planar(gravity=(0.0, 0.0))
    g = m.ground()
    a, b = m.body("pinion", mass=1.0, inertia=0.02), m.body("wheel", mass=2.0, inertia=0.3)
    m.pin(g, a, name="axle_a")
    m.pin(g, b, point_a=(0.3, 0.0), name="axle_b")
    m.gear(a, b, -15.0 / 45.0, name="mesh")
    m.coordinate_drive(a, "theta", law=Polynomial((0.0, 0.0, 2.0)), name="motor")
    result = m.inverse_dynamics(np.linspace(0.0, 1.0, 11), q0=np.array([0, 0, 0, 0, 0, 0, 0.3, 0, 0], dtype=float))
    force_err, moment_err, loads = _planar_newton_euler(m, result, gravity=np.zeros(2))
    ke_rate, _ = _planar_power(m, result, gravity=np.zeros(2))
    return outcome(
        errors={"ratio_rad": (float(np.max(np.abs(result.q[8] + result.q[5] / 3.0))), 1e-12),
                "power_balance_W": (float(np.max(np.abs(loads.drives["motor"]["power"] - ke_rate))), 1e-12),
                "frame_moment_Nm": (moment_err, 1e-12)},
        values={"motor_torque_Nm": float(loads.drives["motor"]["effort"][-1]),
                "wheel_tooth_torque_Nm": float(loads.meshes["mesh"]["effort_on_b"][-1])},
        check=m.check(result).passed,
    )


@case("planar.reverser_frame_torque", "planar_meshes", 2,
      "equal and opposite inertia torques cancel on the frame (short #5)",
      {"solver": "inverse_dynamics", "samples": 6, "duration_s": 1.0})
def planar_reverser_frame_torque():
    """Counter-rotating paddle sets through a reverser on the frame."""
    peaks = {}
    for label, ratio in (("opposite", -1.0), ("same", 1.0)):
        m = Mechanism.planar(gravity=(0.0, 0.0))
        g = m.ground()
        outer, inner = m.body("outer", mass=1.0, inertia=0.4), m.body("inner", mass=1.0, inertia=0.4)
        m.pin(g, outer, name="sleeve")
        m.pin(g, inner, name="shaft")
        m.gear(outer, inner, ratio, name="reverser")
        m.coordinate_drive(outer, "theta", law=Polynomial((0.0, 0.0, 1.5)), name="crank")
        result = m.inverse_dynamics(np.linspace(0.0, 1.0, 6), q0=np.zeros(9))
        peaks[label] = float(np.max(np.abs(m.joint_loads(result).frame["moment"])))
    return outcome(errors={"opposite_frame_torque_Nm": (peaks["opposite"], 1e-12),
                           "same_minus_2I_alpha_Nm": (abs(peaks["same"] - 2 * 0.4 * 3.0), 1e-12)},
                   values={"same_frame_torque_Nm": peaks["same"]})


@case("planar.rack_pinion", "planar_meshes", 2,
      "rack kinematics x = r theta; tooth force m a; Newton-Euler on the frame",
      {"solver": "inverse_dynamics", "samples": 11, "duration_s": 0.5})
def planar_rack_pinion():
    """Pinion accelerating a guided rack under gravity."""
    m = Mechanism.planar(gravity=tuple(G2))
    g = m.ground()
    pinion, rack = m.body("pinion", mass=1.0, inertia=0.01), m.body("rack", mass=5.0, inertia=0.1)
    m.pin(g, pinion, name="axle")
    m.slider(g, rack, axis=(1.0, 0.0), point_rail=(0.0, -0.05), name="guide")
    m.rack(pinion, rack, 0.05, name="teeth")
    m.coordinate_drive(pinion, "theta", law=Polynomial((0.0, 0.0, 3.0)), name="motor")
    result = m.inverse_dynamics(np.linspace(0.0, 0.5, 11), q0=np.array([0, 0, 0, 0, 0, 0, 0, -0.05, 0], dtype=float))
    force_err, moment_err, loads = _planar_newton_euler(m, result)
    return outcome(
        errors={"rack_position_m": (float(np.max(np.abs(result.q[6] - 0.05 * result.q[5]))), 1e-12),
                "tooth_force_N": (float(np.max(np.abs(loads.meshes["teeth"]["effort_on_b"] - 5.0 * 0.3))), 1e-10),
                "frame_force_N": (force_err, 1e-9), "frame_moment_Nm": (moment_err, 1e-9)},
        values={"motor_torque_Nm": float(loads.drives["motor"]["effort"][-1])},
        check=m.check(result).passed,
    )


@case("planar.relative_drive_momentum", "planar_forward_dynamics", 2,
      "angular-momentum conservation with an internal motor; power = dKE/dt",
      {"solver": "simulate", "rtol": 1e-11, "atol": 1e-13, "duration_s": 1.0})
def planar_relative_drive_momentum():
    """Reaction wheel driven relative to a free platform (short #4)."""
    m = Mechanism.planar(gravity=(0.0, 0.0))
    g = m.ground()
    stator, rotor = m.body("stator", mass=1.0, inertia=0.5), m.body("rotor", mass=1.0, inertia=0.1)
    m.pin(g, stator, name="bearing")
    m.pin(stator, rotor, name="shaft")
    m.coordinate_drive(rotor, "theta", law=Polynomial((0.0, 0.0, 4.0)), relative_to=stator, name="motor")
    t = np.linspace(0.0, 1.0, 51)
    result = m.simulate(t, np.zeros(9), v0=np.zeros(9), allow_underconstrained=True, rtol=1e-11, atol=1e-13)
    loads = m.joint_loads(result)
    ke_rate, _ = _planar_power(m, result, gravity=np.zeros(2))
    return outcome(
        errors={"momentum_Nms": (float(np.max(np.abs(0.5 * result.v[5] + 0.1 * result.v[8]))), 1e-10),
                "frame_moment_Nm": (float(np.max(np.abs(loads.frame["moment"]))), 1e-10),
                "power_balance_W": (float(np.max(np.abs(loads.drives["motor"]["power"] - ke_rate))), 1e-9)},
        values={"stator_rate_end_rad_s": float(result.v[5, -1])},
        check=m.check(result).passed,
    )


@case("planar.quick_return_ratio", "planar_kinematics", 2,
      "crank-and-slotted-lever time ratio (pi + 2 asin(r/d)) / (pi - 2 asin(r/d)) (short #8)",
      {"solver": "solve_kinematics", "samples": 4801, "turns": 4})
def planar_quick_return_ratio():
    """Crank-and-slotted-lever quick return: forward / return time ratio."""
    r, d = 0.08, 0.2  # crank radius, crank centre to lever pivot distance
    omega = 2.0 * math.pi
    m = Mechanism.planar(gravity=(0.0, 0.0))
    g = m.ground()
    crank, lever = m.body("crank", mass=1.0, inertia=0.01), m.body("lever", mass=1.0, inertia=0.05)
    m.pin(g, crank, point_a=(0.0, d), name="crank_axle")
    m.pin(g, lever, name="lever_pivot")
    m.point_on_line(crank, lever, point_a=(r, 0.0), direction_b=(1.0, 0.0), name="slot")
    m.motor(crank, omega=omega, name="motor")
    turns = 4
    t = np.linspace(0.0, turns, 4801)
    q0 = np.array([0, 0, 0, 0, d, 0, 0, 0, math.atan2(d, r)], dtype=float)
    result = m.solve_kinematics(t, q0=q0)
    rate = result.v[8]
    # Time events by interpolation, never by sample counts (short #8): the
    # turning points are the roots of a spline through the lever rate.
    crossings = CubicSpline(t, rate).roots(extrapolate=False)
    strokes = np.diff(crossings)
    strokes_a, strokes_b = strokes[0::2], strokes[1::2]
    n = min(len(strokes_a), len(strokes_b))
    slow, fast = max(strokes_a[:n].sum(), strokes_b[:n].sum()), min(strokes_a[:n].sum(), strokes_b[:n].sum())
    alpha = math.asin(r / d)
    closed = (math.pi + 2 * alpha) / (math.pi - 2 * alpha)
    return outcome(errors={"time_ratio": (abs(slow / fast - closed), 1e-6)},
                   values={"time_ratio": float(slow / fast)},
                   check=m.check(result).passed)


def _double_pendulum():
    m = Mechanism.planar(gravity=tuple(G2))
    g = m.ground()
    a = m.body("upper", mass=1.0, inertia=0.02, center_of_mass=(0.25, 0.0))
    b = m.body("lower", mass=0.7, inertia=0.01, center_of_mass=(0.2, 0.0))
    m.pin(g, a, name="shoulder")
    m.pin(a, b, point_a=(0.5, 0.0), name="elbow")
    return m, np.array([0, 0, 0, 0, 0, 0, 0.5, 0, 0], dtype=float)


@case("planar.baumgarte_drift", "planar_stabilization", 2,
      "constraint drift of a chaotic double pendulum with and without Baumgarte (short #4)",
      {"solver": "simulate", "method": "RK45", "rtol": 1e-7, "atol": 1e-9, "duration_s": 3.0,
       "alpha_baumgarte": 60.0, "beta_baumgarte": 60.0})
def planar_baumgarte_drift():
    """Double pendulum: Baumgarte stabilization keeps the constraint drift small."""
    m, q0 = _double_pendulum()
    t = np.linspace(0.0, 3.0, 301)
    common = dict(allow_underconstrained=True, rtol=1e-7, atol=1e-9)
    plain = m.simulate(t, q0, **common)
    stable = m.simulate(t, q0, alpha_baumgarte=60.0, beta_baumgarte=60.0, **common)
    return outcome(errors={"stabilized_drift_m": (m.max_constraint_residual(stable), 1e-7)},
                   values={"unstabilized_drift_m": m.max_constraint_residual(plain)},
                   check=m.check(stable).passed)


@case("planar.check_fails_on_loose_tolerance", "check_verdicts", 2,
      "check() must reject a result whose integration tolerance is too loose",
      {"solver": "simulate", "method": "RK23", "rtol": 1e-2, "atol": 1e-2, "duration_s": 3.0})
def planar_check_fails_on_loose_tolerance():
    """Failing verdict: the same double pendulum integrated far too loosely."""
    m, q0 = _double_pendulum()
    result = m.simulate(np.linspace(0.0, 3.0, 301), q0, allow_underconstrained=True, method="RK23", rtol=1e-2, atol=1e-2)
    report = m.check(result)
    failed = sorted(c.name for c in report.checks if c.passed is False)
    return outcome(values={"failed_checks": float(len(failed))}, check=report.passed)


# --- spatial helpers -------------------------------------------------------------------


def _relative_angle(pose0, pose, axis_world):
    rel = pose.rotation * pose0.rotation.inverse()
    vector = np.array([rel.x, rel.y, rel.z])
    return 2.0 * math.atan2(float(vector @ axis_world), rel.w)


def _axial_speed(result, body, axis):
    return np.array([
        step[body].rotation.to_rotation_matrix() @ result.angular_velocity_body[k, body]
        for k, step in enumerate(result.poses)
    ]) @ np.asarray(axis)


# --- spatial cases ---------------------------------------------------------------------


@case("spatial.hinge_multi_revolution", "spatial_drives", 3,
      "construction identity: a revolute drive reproduces its law through exact half turns",
      {"solver": "simulate", "constraint_tolerance": 1e-11, "samples": 25, "turns": 3})
def spatial_hinge_multi_revolution():
    """Hinge driven forward and backward through whole and half revolutions."""
    errors, values = {}, {}
    for label, rate in (("positive", 3 * math.pi), ("negative", -3 * math.pi)):
        m = Mechanism.spatial()
        g = m.ground()
        disc = m.body("disc", mass=2.0, inertia=(0.5, 0.25, 0.25), pose=Pose3D(rotation=Z_TO_X))
        hinge = m.revolute(g, disc, frame_i=Pose3D(rotation=Z_TO_X), name="hinge")
        m.revolute_drive(hinge, law=ConstantSpeed(rate=rate), name="motor")
        t = np.linspace(0.0, 2.0, 25)  # samples land on exact multiples of pi
        result = m.simulate(t, constraint_tolerance=1e-11)
        angle = np.unwrap([_relative_angle(result.poses[0][1], step[1], np.array([1.0, 0.0, 0.0])) for step in result.poses])
        errors[f"{label}_angle_rad"] = (float(np.max(np.abs(angle - rate * t))), 1e-8)
        values[f"{label}_final_angle_rad"] = float(angle[-1])
    return outcome(errors=errors, values=values)


@case("spatial.compound_pendulum_arbitrary_axis", "spatial_constrained_dynamics", 3,
      "separately integrated 1-DOF ODE about the hinge axis (full inertia tensor, eccentric COM)",
      {"solver": "simulate", "rtol": 1e-11, "atol": 1e-13, "constraint_tolerance": 1e-11,
       "reference_solver": "scipy DOP853 rtol 1e-12"})
def spatial_compound_pendulum_arbitrary_axis():
    """Pendulum on a world-oriented hinge with non-identity frames on both bodies."""
    axis_rotation = ROT((1.0, 2.0, 3.0), 0.7)
    hinge_world = Pose3D(translation=(0.1, 0.2, -0.3), rotation=axis_rotation)
    body_pose = Pose3D(translation=(0.6, -0.1, 0.2), rotation=ROT((0.0, 1.0, 1.0), 0.4))
    com_local = np.array([0.05, -0.02, 0.03])
    inertia_com = np.array([[0.30, 0.02, -0.01], [0.02, 0.20, 0.03], [-0.01, 0.03, 0.25]])
    mass = 3.0
    m = Mechanism.spatial()
    g = m.ground()
    arm = m.body("arm", mass=mass, inertia=(0.30, 0.20, 0.25), pose=body_pose, center_of_mass=com_local)
    m.revolute(g, arm, frame_i=hinge_world, frame_j=compose_pose(inverse_pose(body_pose), hinge_world), name="hinge")
    tensors = np.array([np.eye(3), inertia_com]) if m.nbody == 2 else np.array([inertia_com])
    gravity = np.array([0.0, 0.0, -9.81])
    t = np.linspace(0.0, 1.5, 76)
    result = m.simulate(t, gravity_world=tuple(gravity), inertia_tensors_body=tensors,
                        rtol=1e-11, atol=1e-13, constraint_tolerance=1e-11)
    n = axis_rotation.to_rotation_matrix() @ np.array([0.0, 0.0, 1.0])
    p = np.asarray(hinge_world.translation)
    body_rot = body_pose.rotation.to_rotation_matrix()
    d0 = np.asarray(body_pose.translation) + body_rot @ com_local - p
    n_body = body_rot.T @ n
    d_perp = d0 - (d0 @ n) * n
    inertia_axis = float(n_body @ inertia_com @ n_body + mass * d_perp @ d_perp)

    def rotate(vec, angle):
        return (vec * math.cos(angle) + np.cross(n, vec) * math.sin(angle)
                + n * (n @ vec) * (1 - math.cos(angle)))

    def rhs(_, y):
        torque = float(n @ np.cross(rotate(d0, y[0]), mass * gravity))
        return [y[1], torque / inertia_axis]

    reference = solve_ivp(rhs, (0.0, 1.5), [0.0, 0.0], t_eval=t, method="DOP853", rtol=1e-12, atol=1e-14)
    angle = np.unwrap([_relative_angle(result.poses[0][-1], step[-1], n) for step in result.poses])
    return outcome(errors={"angle_rad": (float(np.max(np.abs(angle - reference.y[0]))), 1e-7)},
                   values={"angle_end_rad": float(angle[-1])},
                   check=m.check(result).passed)


@case("spatial.three_joint_kinds_between_moving_bodies", "spatial_constrained_dynamics", 3,
      "energy conservation; check() residual and force/torque balance verdicts",
      {"solver": "simulate", "rtol": 1e-10, "atol": 1e-12, "duration_s": 1.0})
def spatial_three_joint_kinds():
    """Chain ground-spherical-A-revolute-B-fixed-C swinging under gravity."""
    m = Mechanism.spatial()
    g = m.ground()
    a = m.body("a", mass=1.0, inertia=(0.02, 0.02, 0.01), pose=Pose3D(translation=(0.3, 0.0, 0.0)))
    b = m.body("b", mass=0.8, inertia=(0.01, 0.015, 0.01), pose=Pose3D(translation=(0.7, 0.0, 0.1)))
    c = m.body("c", mass=0.3, inertia=(0.002, 0.002, 0.002), pose=Pose3D(translation=(0.9, 0.0, 0.1)))
    m.spherical(g, a, point_j=(-0.3, 0.0, 0.0), name="ball")
    m.revolute(a, b, frame_i=Pose3D(translation=(0.3, 0.0, 0.0), rotation=Z_TO_Y),
               frame_j=Pose3D(translation=(-0.1, 0.0, -0.1), rotation=Z_TO_Y), name="elbow")
    m.fixed(b, c, frame_i=Pose3D(translation=(0.2, 0.0, 0.0)), name="weld")
    gravity = np.array([0.0, 0.0, -9.81])
    t = np.linspace(0.0, 1.0, 51)
    result = m.simulate(t, gravity_world=tuple(gravity), rtol=1e-10, atol=1e-12)
    masses = np.array([0.0, 1.0, 0.8, 0.3])[-m.nbody:]
    inertias = result.inertia_tensor_body
    energy = []
    for k in range(len(t)):
        e = 0.0
        for i in range(m.nbody):
            if masses[i] == 0.0:
                continue
            v, w = result.linear_velocity[k, i], result.angular_velocity_body[k, i]
            e += 0.5 * masses[i] * v @ v + 0.5 * w @ inertias[i] @ w - masses[i] * gravity @ result.center_of_mass_position[k, i]
        energy.append(e)
    energy = np.array(energy)
    return outcome(errors={"energy_drift_J": (float(np.max(np.abs(energy - energy[0]))), 1e-6)},
                   values={"c_height_end_m": float(result.center_of_mass_position[-1, -1, 2])},
                   check=m.check(result).passed)


@case("spatial.fixed_pair_pure_couple", "spatial_constrained_dynamics", 3,
      "rigid composite under a pure couple: alpha = tau / I_total; weld reaction = I_b alpha",
      {"solver": "simulate", "rtol": 1e-11, "atol": 1e-13, "duration_s": 0.5})
def spatial_fixed_pair_pure_couple():
    """Two welded bodies, free in space, spun up by a couple on one of them."""
    m = Mechanism.spatial()
    a = m.body("a", mass=1.0, inertia=(0.1, 0.1, 0.2))
    b = m.body("b", mass=1.0, inertia=(0.2, 0.2, 0.3))
    m.fixed(a, b, name="weld")
    tau = 1.2
    load = m.rigid_body_load(a, torque=(0.0, 0.0, tau), name="couple")
    t = np.linspace(0.0, 0.5, 26)
    result = m.simulate(t, loads=(load,), rtol=1e-11, atol=1e-13)
    alpha = tau / 0.5
    ia, ib = m._idx(a), m._idx(b)
    return outcome(
        errors={"alpha_a_rad_s2": (float(np.max(np.abs(result.angular_acceleration_body[:, ia, 2] - alpha))), 1e-9),
                "weld_torque_on_b_Nm": (float(np.max(np.abs(result.reaction_torque_body[:, ib, 2] - 0.3 * alpha))), 1e-9),
                "com_drift_m": (float(np.max(np.abs(result.center_of_mass_position[:, ia]))), 1e-12)},
        values={"spin_end_rad_s": float(result.angular_velocity_body[-1, ia, 2])},
        check=m.check(result).passed,
    )


@case("spatial.offset_force_reaction", "spatial_reactions", 3,
      "static equilibrium: reaction force -F and reaction torque -(r x F) about the COM",
      {"solver": "simulate", "samples": 3})
def spatial_offset_force_reaction():
    """A welded body loaded by a force applied away from its COM."""
    m = Mechanism.spatial()
    g = m.ground()
    block = m.body("block", mass=2.0, inertia=(0.1, 0.2, 0.3), pose=Pose3D(translation=(0.4, 0.1, 0.0)))
    m.fixed(g, block, frame_i=Pose3D(translation=(0.4, 0.1, 0.0)), name="mount")
    force, arm = np.array([3.0, -2.0, 5.0]), np.array([0.1, 0.25, -0.05])
    load = m.rigid_body_load(block, force=tuple(force), application_point_from_com_body=tuple(arm), name="push")
    result = m.simulate(np.linspace(0.0, 0.1, 3), loads=(load,))
    i = m._idx(block)
    return outcome(
        errors={"reaction_force_N": (float(np.max(np.abs(result.reaction_force_world[:, i] + force))), 1e-9),
                "reaction_torque_Nm": (float(np.max(np.abs(result.reaction_torque_body[:, i] + np.cross(arm, force)))), 1e-9)},
        values={"reaction_force_z_N": float(result.reaction_force_world[0, i, 2])},
        check=m.check(result).passed,
    )


@case("spatial.coordinate_drive_effort", "spatial_drives", 3,
      "drive force = m (z'' + g) for a time-varying lift with explicit acceleration",
      {"solver": "simulate", "samples": 11})
def spatial_coordinate_drive_effort():
    """Carriage lifted by a polynomial coordinate drive against gravity."""
    m = Mechanism.spatial()
    m.body("carriage", mass=3.0, inertia=(0.1, 0.1, 0.1))
    law = Polynomial((0.0, 0.2, 0.3, -0.1))
    m.coordinate_drive(0, "z", law=law, name="lift")
    t = np.linspace(0.0, 1.0, 11)
    result = m.simulate(t, gravity_world=(0.0, 0.0, -9.81))
    effort = np.asarray(m.drive_efforts(result)["lift"]["effort"])
    expected = 3.0 * (np.array([law.acceleration(x) for x in t]) + 9.81)
    return outcome(errors={"effort_N": (float(np.max(np.abs(effort - expected))), 1e-8)},
                   values={"effort_end_N": float(effort[-1])},
                   check=m.check(result).passed)


@case("spatial.revolute_drive_effort", "spatial_drives", 3,
      "hinge torque = I alpha + m g r cos(theta) for an accelerating arm",
      {"solver": "simulate", "constraint_tolerance": 1e-11, "samples": 21})
def spatial_revolute_drive_effort():
    """Arm swung up by an accelerating revolute drive against gravity."""
    m = Mechanism.spatial()
    g = m.ground()
    arm = m.body("arm", mass=2.0, inertia=(0.01, 0.2, 0.2), pose=Pose3D(translation=(0.5, 0.0, 0.0)))
    hinge = m.revolute(g, arm, frame_j=Pose3D(translation=(-0.5, 0.0, 0.0)), name="hinge")
    law = Polynomial((0.0, 0.0, 1.5))
    m.revolute_drive(hinge, law=law, name="motor")
    t = np.linspace(0.0, 1.0, 21)
    result = m.simulate(t, gravity_world=(0.0, -9.81, 0.0), constraint_tolerance=1e-11)
    torque = np.asarray(m.drive_efforts(result)["motor"]["effort"])
    theta = np.array([law.value(x) for x in t])
    expected = (0.2 + 2.0 * 0.25) * 3.0 + 2.0 * 9.81 * 0.5 * np.cos(theta)
    return outcome(errors={"torque_Nm": (float(np.max(np.abs(torque - expected))), 1e-7)},
                   values={"torque_end_Nm": float(torque[-1])},
                   check=m.check(result).passed)


@case("spatial.fast_asymmetric_top", "spatial_free_dynamics", 3,
      "energy and world angular momentum conservation at high spin",
      {"solver": "simulate", "rtol": 1e-11, "atol": 1e-13, "spin_rad_s": 300.0})
def spatial_fast_asymmetric_top():
    """Torque-free asymmetric body spinning fast near its major axis."""
    m = Mechanism.spatial()
    m.body("top", mass=1.0, inertia=(0.3, 0.2, 0.1))
    inertia = np.diag([0.3, 0.2, 0.1])
    w0 = np.array([300.0, 2.0, 1.0])
    t = np.linspace(0.0, 0.2, 101)
    result = m.simulate(t, angular_velocity0_body={"top": tuple(w0)}, rtol=1e-11, atol=1e-13)
    energy, momentum = [], []
    for k, step in enumerate(result.poses):
        w = result.angular_velocity_body[k, -1]
        energy.append(0.5 * w @ inertia @ w)
        momentum.append(step[-1].rotation.to_rotation_matrix() @ (inertia @ w))
    energy, momentum = np.array(energy), np.array(momentum)
    return outcome(errors={"energy_rel": (float(np.max(np.abs(energy / energy[0] - 1))), 1e-9),
                           "momentum_rel": (float(np.max(np.linalg.norm(momentum - momentum[0], axis=1)) / np.linalg.norm(momentum[0])), 1e-9)},
                   values={"spin_x_end_rad_s": float(result.angular_velocity_body[-1, -1, 0])},
                   check=m.check(result).passed)


def _cardan(beta, omega, redundant=False):
    m = Mechanism.spatial()
    ground = m.ground("frame")
    shaft_in = m.body("shaft_in", mass=5.0, inertia=(0.04, 0.02, 0.02))
    cross = m.body("cross", mass=0.3, inertia=(0.002, 0.002, 0.002))
    shaft_out = m.body("shaft_out", mass=2.0, inertia=(0.02, 0.02, 0.02))
    z_to_out = ROT((0.0, 0.0, 1.0), beta) * Z_TO_X
    bearing = m.revolute(ground, shaft_in, frame_i=Pose3D(rotation=Z_TO_X), frame_j=Pose3D(rotation=Z_TO_X), name="input_bearing")
    m.revolute(shaft_in, cross, frame_i=Pose3D(rotation=Z_TO_Y), frame_j=Pose3D(rotation=Z_TO_Y), name="yoke_in")
    m.revolute(cross, shaft_out, name="yoke_out")
    (m.revolute if redundant else m.parallel_axes)(
        ground, shaft_out, frame_i=Pose3D(rotation=z_to_out), frame_j=Pose3D(rotation=z_to_out), name="output_bearing")
    m.revolute_drive(bearing, law=ConstantSpeed(rate=omega), name="motor")
    return m


@case("spatial.cardan_speed_ratio", "spatial_constrained_dynamics", 3,
      "closed-form Cardan ratio cos(b) / (1 - sin^2(b) cos^2(theta)) at b = 15, 30, 45 deg",
      {"solver": "solve_kinematics", "samples": 33})
def spatial_cardan_speed_ratio():
    """Single Cardan shaft through parallel_axes, three shaft angles."""
    omega, errors, values = 4.0 * math.pi, {}, {}
    for deg in (15, 30, 45):
        beta = math.radians(deg)
        m = _cardan(beta, omega)
        t = np.linspace(0.0, 0.5, 33)
        result = m.solve_kinematics(t)
        axis_out = np.array([math.cos(beta), math.sin(beta), 0.0])
        ratio = result.angular_velocity[:, 3] @ axis_out / omega  # kinematic results are world-frame
        theta = omega * t
        closed = math.cos(beta) / (1.0 - math.sin(beta) ** 2 * np.cos(theta) ** 2)
        errors[f"ratio_beta_{deg}"] = (float(np.max(np.abs(np.abs(ratio) - closed))), 1e-6)
        values[f"ratio_range_beta_{deg}"] = float(np.ptp(np.abs(ratio)))
    return outcome(errors=errors, values=values)


@case("spatial.redundant_bearing_rejected", "spatial_guardrails", 3,
      "validate() names the redundant bearing before any integration; simulate() raises rank_deficient",
      {"solver": "validate"})
def spatial_redundant_bearing_rejected():
    """Cardan shaft with a second revolute bearing: redundant, rejected by name."""
    m = _cardan(math.radians(30), 4.0 * math.pi, redundant=True)
    report = m.validate()
    named = any("output_bearing" in (issue.message or "") or "output_bearing" in json.dumps(issue.as_dict(), default=str)
                for issue in report.issues if issue.code == "redundant_constraints")
    try:
        m.simulate(np.linspace(0.0, 0.01, 2))
        code = "none"
    except MechanismSolveError as exc:
        code = exc.code
    return outcome(errors={"not_named": (0.0 if named else 1.0, 0.0),
                           "wrong_error_code": (0.0 if code == "rank_deficient" else 1.0, 0.0),
                           "validate_ok": (1.0 if report.ok else 0.0, 0.0)})


def _spatial_pendulum(length):
    m = Mechanism.spatial()
    g = m.ground()
    bob = m.body("bob", mass=1.0, inertia=(1e-6 * length**2,) * 3, pose=Pose3D(translation=(length * math.sin(0.3), -length * math.cos(0.3), 0.0)))
    m.revolute(g, bob, frame_j=Pose3D(translation=(-length * math.sin(0.3), length * math.cos(0.3), 0.0)), name="pivot")
    return m


@case("spatial.scale_invariance", "spatial_conditioning", 3,
      "dimensionless pendulum motion identical at 1 mm and 1 km scale (t scaled by sqrt(L/g))",
      {"solver": "simulate", "rtol": 1e-10, "atol_scaled": 1e-12, "lengths_m": [1e-3, 1.0, 1e3]})
def spatial_scale_invariance():
    """The same pendulum at three length scales, compared in dimensionless time."""
    tau = np.linspace(0.0, 6.0, 61)
    angles = {}
    for length in (1e-3, 1.0, 1e3):
        m = _spatial_pendulum(length)
        t = tau * math.sqrt(length / 9.81)
        result = m.simulate(t, gravity_world=(0.0, -9.81, 0.0), rtol=1e-10, atol=1e-12 * length,
                            constraint_tolerance=1e-10 * length)
        com = result.center_of_mass_position[:, -1]
        angles[length] = np.arctan2(com[:, 0], -com[:, 1])
    return outcome(errors={"mm_vs_m_rad": (float(np.max(np.abs(angles[1e-3] - angles[1.0]))), 1e-6),
                           "km_vs_m_rad": (float(np.max(np.abs(angles[1e3] - angles[1.0]))), 1e-6)},
                   values={"angle_end_rad": float(angles[1.0][-1])})


@case("spatial.tolerance_convergence", "spatial_convergence", 3,
      "error against the 1-DOF reference ODE falls as rtol tightens; projection on and off",
      {"solver": "simulate", "rtols": [1e-5, 1e-8, 1e-11]})
def spatial_tolerance_convergence():
    """Large-swing pendulum at three tolerances, with and without sample projection."""
    length, theta0 = 1.0, 1.2
    m = Mechanism.spatial()
    g = m.ground()
    m.body("bob", mass=1.0, inertia=(1e-4,) * 3, pose=Pose3D(translation=(length * math.sin(theta0), -length * math.cos(theta0), 0.0)))
    m.revolute(g, 1, frame_j=Pose3D(translation=(-length * math.sin(theta0), length * math.cos(theta0), 0.0)), name="pivot")
    t = np.linspace(0.0, 3.0, 61)
    inertia = 1.0 * length**2 + 1e-4
    reference = solve_ivp(lambda _, y: [y[1], -9.81 * length * math.sin(y[0]) / inertia], (0, 3.0), [theta0, 0.0],
                          t_eval=t, method="DOP853", rtol=1e-13, atol=1e-14).y[0]
    errors_by = {}
    for rtol in (1e-5, 1e-8, 1e-11):
        for project in (True, False):
            result = m.simulate(t, gravity_world=(0.0, -9.81, 0.0), rtol=rtol, atol=rtol * 1e-2, project_samples=project)
            com = result.center_of_mass_position[:, -1]
            errors_by[(rtol, project)] = float(np.max(np.abs(np.arctan2(com[:, 0], -com[:, 1]) - reference)))
    monotone = all(errors_by[(1e-5, p)] > errors_by[(1e-8, p)] > errors_by[(1e-11, p)] for p in (True, False))
    # Truncation-level errors differ in their last digits across CPUs and BLAS
    # builds, so they are bounded (error <= max(10 rtol, 1e-8)), not recorded.
    errors = {f"error_rtol_{r:g}_{'proj' if p else 'noproj'}": (e, max(10.0 * r, 1e-8))
              for (r, p), e in errors_by.items()}
    errors["not_monotone"] = (0.0 if monotone else 1.0, 0.0)
    return outcome(errors=errors)


@case("spatial.check_fails_without_projection", "check_verdicts", 3,
      "check() must reject a loosely integrated, unprojected constrained result",
      {"solver": "simulate", "method": "RK45", "rtol": 1e-3, "atol": 1e-3, "project_samples": False})
def spatial_check_fails_without_projection():
    """Failing verdict: a pendulum integrated loosely with projection disabled."""
    m = _spatial_pendulum(1.0)
    result = m.simulate(np.linspace(0.0, 5.0, 51), gravity_world=(0.0, -9.81, 0.0), method="RK45",
                        rtol=1e-3, atol=1e-3, project_samples=False)
    report = m.check(result)
    return outcome(values={"failed_checks": float(sum(c.passed is False for c in report.checks))}, check=report.passed)


@case("spatial.cross_mechanism_result_rejected", "spatial_guardrails", 3,
      "a result is bound to its model identity: check() of another model's result raises",
      {"solver": "simulate"})
def spatial_cross_mechanism_result_rejected():
    """Identity guard: one mechanism's result handed to another mechanism."""
    a, b = _spatial_pendulum(1.0), _spatial_pendulum(2.0)
    result = a.simulate(np.linspace(0.0, 0.1, 3), gravity_world=(0.0, -9.81, 0.0))
    try:
        b.check(result)
        rejected = False
    except (MechanismSolveError, ValueError):
        rejected = True
    return outcome(errors={"accepted_foreign_result": (0.0 if rejected else 1.0, 0.0)})


# --- schema fixtures from released versions --------------------------------------------


def _fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@case("schemas.planar_mechanism_fixtures", "schema_fixtures", 2,
      "payloads written by mbsd 0.9.6 (v1), 0.9.7 (v2) and 0.9.8 (v3) re-solve to what those releases computed",
      {"solver": "inverse_dynamics", "fixtures": ["planar-mechanism-v1", "planar-mechanism-v2", "planar-mechanism-v3"]})
def schemas_planar_mechanism_fixtures():
    """Planar mechanism v1-v3 fixtures: read, re-solve, compare with the producer."""
    errors = {}
    for version in (1, 2, 3):
        record = _fixture(f"planar-mechanism-v{version}")
        m = PlanarMechanism.from_dict(record["payload"])
        computed = record["computed"]
        result = m.inverse_dynamics(np.array(computed["t"]), q0=np.array(computed["q0"]))
        loads = m.joint_loads(result)
        errors[f"v{version}_engine_torque_Nm"] = (float(np.max(np.abs(loads.drives["engine"]["effort"] - computed["engine_torque"]))), 1e-9)
        errors[f"v{version}_main_force_N"] = (float(np.max(np.abs(loads.joints["main"]["force_on_j"] - np.array(computed["main_force_on_j"])))), 1e-9)
    return outcome(errors=errors)


@case("schemas.spatial_fixtures", "schema_fixtures", 3,
      "spatial model v3 (0.9.6) and v4 (0.9.7) re-simulate to what those releases computed; old results validate",
      {"solver": "simulate", "fixtures": ["spatial-model-v3", "spatial-model-v4",
                                          "spatial-constrained-result-v2", "spatial-constrained-result-v3"]})
def schemas_spatial_fixtures():
    """Spatial model and constrained-result fixtures from released versions."""
    from mbsd.spatial import validate_constrained_result_payload

    errors = {}
    for version in (3, 4):
        record = _fixture(f"spatial-model-v{version}")
        m = SpatialMechanism.from_dict(record["payload"])
        result = m.simulate(np.array(record["computed"]["t"]), gravity_world=(0.0, -9.81, 0.0))
        com_y = result.center_of_mass_position[:, 1, 1]
        errors[f"model_v{version}_com_y_m"] = (float(np.max(np.abs(com_y - record["computed"]["com_y"]))), 1e-9)
    for version in (2, 3):
        payload = _fixture(f"spatial-constrained-result-v{version}")["payload"]
        try:
            validate_constrained_result_payload(payload)
            errors[f"result_v{version}_rejected"] = (0.0, 0.0)
        except ValueError:
            errors[f"result_v{version}_rejected"] = (1.0, 0.0)
    return outcome(errors=errors)


@case("schemas.joint_loads_fixtures", "schema_fixtures", 2,
      "joint-loads v1-v3 fixtures carry the conventions their published JSON Schema pins",
      {"fixtures": ["planar-joint-loads-v1", "planar-joint-loads-v2", "planar-joint-loads-v3"]})
def schemas_joint_loads_fixtures():
    """Joint-load payloads from 0.9.6, 0.9.7 and 0.9.8 against their JSON Schemas."""
    schemas = Path(__file__).resolve().parents[1] / "schemas"
    errors = {}
    for version in (1, 2, 3):
        payload = _fixture(f"planar-joint-loads-v{version}")["payload"]
        schema = json.loads((schemas / f"mbsd.planar.joint-loads-v{version}.schema.json").read_text())
        missing = set(schema["required"]) - payload.keys()
        mismatched = [k for k, v in schema["properties"]["conventions"]["properties"].items()
                      if payload["conventions"].get(k) != v["const"]]
        errors[f"v{version}_contract_violations"] = (float(len(missing) + len(mismatched)), 0.0)
    return outcome(errors=errors)
