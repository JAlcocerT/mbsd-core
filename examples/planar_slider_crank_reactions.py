"""Slider-crank joint reactions, drive torque, and shaking force via inverse dynamics."""

import numpy as np

from mbsd import Mechanism


def main() -> None:
    crank_radius, rod_length, omega = 0.05, 0.2, 100.0
    mechanism = Mechanism.planar(gravity=(0.0, -9.81))
    ground = mechanism.ground()
    crank = mechanism.body("crank", mass=0.5, inertia=1e-4)
    rod = mechanism.body("rod", mass=0.8, inertia=3e-3)
    piston = mechanism.body("piston", mass=0.4, inertia=1e-4)

    mechanism.pin(ground, crank, point_b=(-crank_radius / 2, 0.0), name="main_bearing")
    mechanism.pin(
        crank,
        rod,
        point_a=(crank_radius / 2, 0.0),
        point_b=(-rod_length / 2, 0.0),
        name="crank_pin",
    )
    mechanism.pin(rod, piston, point_a=(rod_length / 2, 0.0), name="wrist_pin")
    mechanism.slider(ground, piston, axis=(1.0, 0.0), name="bore")
    mechanism.motor(crank, omega=omega, name="engine")
    mechanism.constant_force(piston, (-50.0, 0.0), name="gas_load")

    q0 = np.array(
        [0, 0, 0, crank_radius / 2, 0, 0, crank_radius + rod_length / 2, 0, 0,
         crank_radius + rod_length, 0, 0],
        dtype=float,
    )
    report = mechanism.validate(q0)
    print("Validation:", "ok" if report.ok else "FAILED", "| portable:", report.portable)

    t = np.linspace(0.0, 2.0 * np.pi / omega, 361)
    result = mechanism.inverse_dynamics(t, q0=q0)
    verdict = mechanism.check(result)
    print("Check:", "passed" if verdict.passed else "FAILED")

    loads = mechanism.joint_loads(result)
    crank_pin = np.linalg.norm(loads.joints["crank_pin"]["force_on_j"], axis=0)
    torque = loads.drives["engine"]["effort"]
    shaking = loads.frame["force"]
    print("Slider-crank reactions over one revolution")
    print(f"  peak crank-pin force:   {crank_pin.max(): .2f} N")
    print(f"  peak / RMS drive torque: {np.abs(torque).max(): .3f} / "
          f"{np.sqrt(np.mean(torque**2)): .3f} N*m")
    print(f"  peak shaking force x/y: {np.abs(shaking[0]).max(): .2f} / "
          f"{np.abs(shaking[1]).max(): .2f} N")


if __name__ == "__main__":
    main()
