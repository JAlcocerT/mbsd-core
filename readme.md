# MBSD Core

Readable Python tools for planar multibody mechanism kinematics and dynamics.

Runnable companion examples and generated gallery assets live in
[MBSD Examples](https://github.com/JAlcocerT/mbsd-examples).

MBSD is a small, inspectable mechanism framework. It is aimed at engineers,
students, and researchers who want to script mechanisms directly in Python:
declare bodies, joints, drives, springs, and forces, then solve the kinematics
or constrained dynamics with transparent equations.

The public API is intentionally focused on **2D planar mechanisms**. Plotting,
CAD/export, gallery, browser-demo, larger synthesis workflows, and 3D work
will be released separately after the core API is stable.

## Install

```sh
uv sync --extra dev
```

Plain pip also works:

```sh
pip install -e .[dev]
```

Plotting is optional:

```sh
pip install -e .[plot]
```

## Quick Start

```python
import numpy as np

from mbsd import Mechanism

m = Mechanism.planar(gravity=(0.0, 0.0))

ground = m.ground()
slider = m.body("slider", mass=1.0, inertia=0.01)

m.slider(ground, slider, axis=(1.0, 0.0))
m.coordinate_drive(
    slider,
    "x",
    value=lambda t: np.sin(2.0 * np.pi * t),
    velocity=lambda t: 2.0 * np.pi * np.cos(2.0 * np.pi * t),
    acceleration=lambda t: -(2.0 * np.pi) ** 2 * np.sin(2.0 * np.pi * t),
)

result = m.solve_kinematics(np.linspace(0.0, 1.0, 101))
m.assert_constraints_satisfied(result)
print(m.model_diagnostics(result.q[:, 0]).as_dict())
print(m.diagnostics(result).as_dict())
print(result.q[3, -1])  # slider x at final time
```

The local `v0.4.0-dev` branch includes portable export helpers for downstream
apps:

```python
m.to_json("mechanism.json")
m.result_to_json(result, "result.json")
m.result_to_csv(result, "trajectory.csv")
```

## What Works Now

- Planar rigid bodies with reference coordinates `[x, y, theta]`.
- Revolute joints.
- Prismatic joints.
- User-defined scalar constraints and drives.
- Point kinematics: position, velocity, acceleration.
- Position, velocity, and acceleration solves.
- Constrained dynamics with Lagrange multipliers.
- Gravity, springs, and damping helpers.
- Constraint residual, assertion, and diagnostics helpers for validating solved
  trajectories.
- Static model diagnostics with Jacobian rank, rank-based DOF, and nominal DOF.
- JSON/CSV export helpers on the local `v0.4.0-dev` branch.
- Preview four-bar synthesis helpers under `mbsd.planar.synthesis`.

Forward dynamics currently requires each body reference point to coincide with
its center of mass. Offset-COM kinematics are accepted, but offset-COM dynamics
raise a clear error until the corresponding inertial terms are implemented.

## Examples

```sh
uv run python examples/planar_driven_slider.py
uv run python examples/planar_mass_spring.py
uv run python examples/planar_slider_crank_analysis.py
```

## Tests

```sh
uv run pytest -q
```

## Makefile

```sh
make sync      # install with uv
make test      # run pytest
make examples  # run bundled examples
make check     # run tests and examples
```

## Positioning

This project is not trying to replace large industrial multibody engines.
Use Chrono, Exudyn, Simbody, Siconos, or a commercial tool when you need broad
multiphysics coverage, large-scale collision/contact stacks, or production
vehicle simulation.

MBSD's niche is smaller:

- mechanism-first modeling
- readable Python source
- direct NumPy/SciPy integration
- examples that expose the math
- a clean path from teaching model to custom engineering script

Think of it as a "PySpice for mechanisms": a compact declarative layer over a
solver kernel that stays close enough to the equations to inspect and modify.

## Repository Notes

This repository contains the installable framework for the weekly OSS release
series. The historical workbench still contains course material, generated
plots, animations, synthesis experiments, fluid mechanics work, CAD rendering,
and a 3D MBSD kernel. Release branches keep only the public package surface
under:

```text
src/mbsd/
```

Plotting, CAD/export, browser demos, gallery assets, larger synthesis
workflows, and 3D mechanisms are intentionally outside this core repository.

Longer-form docs, release planning, and website content live outside this core
package repository.

## Consulting

The project is connected to `multibodysystemsdynamics.com` for teams that need
help setting up a mechanism model, validating a solver, adding a custom joint,
or turning a one-off engineering script into a maintainable workflow.
