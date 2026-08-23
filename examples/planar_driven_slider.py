"""Driven planar slider using the public MBSD API."""

import numpy as np

from mbsd import Mechanism


def main() -> None:
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

    result = mechanism.solve_kinematics(np.linspace(0.0, 1.0, 101))

    x = result.q[3, :]
    y = result.q[4, :]
    theta = result.q[5, :]

    print("Driven slider")
    print(f"  final x:        {x[-1]: .6f}")
    print(f"  max |y|:        {np.max(np.abs(y)):.3e}")
    print(f"  max |theta|:    {np.max(np.abs(theta)):.3e}")


if __name__ == "__main__":
    main()
