"""Planar mass-spring-damper dynamics using the public MBSD API."""

import numpy as np

from mbsd import Mechanism, Spring


def main() -> None:
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
    t = np.linspace(0.0, 1.0, 101)

    result = mechanism.simulate(t, q0=q0, v0=v0, springs=[spring])

    x = result.q[3, :]
    print("Mass-spring-damper")
    print(f"  initial x:      {x[0]: .6f}")
    print(f"  final x:        {x[-1]: .6f}")
    print(f"  min/max x:      {np.min(x): .6f} / {np.max(x): .6f}")


if __name__ == "__main__":
    main()
