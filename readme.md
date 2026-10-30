# MBSD Core

Readable Python tools for planar multibody mechanism kinematics and dynamics.

Runnable companion examples and generated gallery assets live in
[MBSD Examples](https://github.com/JAlcocerT/mbsd-examples).

MBSD is a small, inspectable mechanism framework. It is aimed at engineers,
students, and researchers who want to script mechanisms directly in Python:
declare bodies, joints, drives, springs, and forces, then solve the kinematics
or constrained dynamics with transparent equations.

The stable solver API is intentionally focused on **2D planar mechanisms**.
Core also contains experimental spatial kinematics and explicitly named
fixed-orientation, free-body, unconstrained coupled six-DOF, and constrained
six-DOF solvers. Plotting, gallery, browser UI, larger synthesis workflows,
and CAD-specific integrations remain separate while the API stabilizes.
Portable JSON and CSV exports support downstream applications and neutral CAD
handoffs.

## Install

```bash
git clone https://github.com/JAlcocerT/mbsd-core.git
cd mbsd-core
git checkout v0.9.6
uv sync --extra dev
```

Plain pip and the optional plotting extra also work from the cloned repository:

```bash
python -m pip install -e ".[dev,plot]"
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

Since `0.4.0`, MBSD includes portable export helpers for downstream applications:

```python
m.to_json("mechanism.json")
m.result_to_json(result, "result.json")
m.result_to_csv(result, "trajectory.csv")
m.point_trace_to_json(result, slider, (0.2, 0.0), "slider-point.json")
m.point_trace_to_csv(result, slider, (0.2, 0.0), "slider-point.csv")
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
- Versioned JSON and CSV mechanism, result, and body-point export helpers.
- Experimental `mbsd.spatial` vocabulary for portable 3D model descriptions.
- Per-configuration solver diagnostics with explicit model classification.
- Experimental spatial transforms, point velocities, and joint residuals in
  `0.7.0`.
- Experimental `Mechanism.spatial()` position/velocity solves, rank diagnostics,
  result provenance, and versioned kinematic-result exports in `0.8.0`.
- Experimental fixed-orientation spatial translational dynamics, force and
  reaction histories, three-level residuals, and versioned exports in `0.9.1`.
- Experimental unconstrained free-body rotation with quaternion integration,
  torque response, principal inertia, and conservation diagnostics in `0.9.2`.
- Experimental unconstrained multi-body six-DOF dynamics with gravity, offset
  centers of mass, pair interactions, and complete state export in `0.9.3`.
- Constrained spatial dynamics with spherical, fixed, and revolute joints,
  physical reaction loads, projection, and stabilization controls in `0.9.4`.
- Strict constrained-result identity and validation plus engine-neutral
  FreeCAD/Blender/browser transform handoffs in `0.9.5`.
- Planar inverse dynamics, named joint reactions, drive efforts, frame
  (shaking) loads, portable motion laws and load descriptors, model import,
  and `validate` / `check` guardrails in `0.9.6`.
- Preview four-bar synthesis helpers under `mbsd.planar.synthesis`.

The restricted planar and fixed-orientation preview paths retain their stated
center-of-mass limitations. The unconstrained coupled and constrained six-DOF
spatial solvers support offset centers of mass and export the distinction
between center-of-mass state and body reference-origin pose.

## Planar Reactions and Guardrails

MBSD `0.9.6` turns planar Lagrange multipliers into physical loads and adds
pre- and post-solve guardrails that agents and tools can read as data.

```python
import numpy as np
from mbsd import Mechanism

m = Mechanism.planar(gravity=(0.0, -9.81))
ground = m.ground()
rod = m.body("rod", mass=2.0, inertia=2.0 / 12)
m.pin(ground, rod, point_b=(-0.5, 0.0), name="shoulder")
m.motor(rod, omega=1.0, name="actuator")

q0 = np.array([0, 0, 0, 0.5, 0, 0], dtype=float)
print(m.validate(q0).as_dict())               # DOF, rank, portability, issues

result = m.inverse_dynamics(np.linspace(0.0, 1.0, 101), q0=q0)
print(m.check(result).passed)                 # residuals + equation-of-motion balance
loads = m.joint_loads(result)
loads.joints["shoulder"]["force_on_j"]        # world-frame pin force on the rod
loads.drives["actuator"]["effort"]            # motor torque, N*m
loads.frame["force"]                          # load the mechanism puts on ground
m.joint_loads_to_json(result, "loads.json")   # mbsd.planar.joint_loads v1
```

- `simulate()` results also carry multipliers, so `joint_loads()` works on
  forward dynamics.
- Joint forces are world-frame; moments are about the joint's world point
  (the body-j attachment point). Drive effort is the force or torque a drive
  applies to its coordinate; `frame` assumes actuators react against ground.
- Portable inputs are data: `coordinate_drive(law=ConstantSpeed(...) |
  Polynomial(...) | Tabulated(...))`, `spring()`, `constant_force()`,
  `constant_torque()`, `tabulated_force()`, `tabulated_torque()`.
  `PlanarMechanism.from_json()` rebuilds them exactly.
- Python callables (`coordinate_drive(value, velocity, acceleration)`,
  `force(fn)`) still work but are exported as opaque. `validate()` reports
  the model as non-portable and `from_json()` raises `NotPortableError`.
- `mbsd.capabilities()` and `mbsd.operation_manifest()` describe the public
  operations, error codes, and validation codes as plain data
  (`schemas/mbsd.operations-v1.json`).
- `max_steps` and `max_evaluations` bound work; errors carry a stable `code`,
  `context`, and `remediation`.

## Experimental Spatial Vocabulary

MBSD `0.5.0` introduced data structures for 3D poses, bodies, frames, and fixed
or spherical joint sketches under `mbsd.spatial`. That original surface was a
model vocabulary only; the later solver milestones described below add
kinematics and dynamics while remaining experimental until `1.0`.

Spatial data uses a right-handed XYZ world frame, metres, kilograms, seconds,
and radians. Quaternions are `[w, x, y, z]` active rotations from body to world.
Body inertia values are principal moments about the center of mass in the body
frame. Joint points are body-local; a `None` body reference means the world and
its point is expressed in world coordinates.

## Experimental Spatial Kinematics

MBSD `0.7.0` adds point transforms and velocities, pose composition and
inversion, frame-pose resolution, spherical/fixed-joint residuals, and
finite-difference residual Jacobians. Angular velocity and Jacobian rotation
increments are world-expressed. Spatial schema v2 advertises these capabilities
and explicitly distinguishes residual helpers from solved spatial results.

MBSD `0.8.0` adds `Mechanism.spatial()` for small constrained position and
velocity problems with spherical and fixed joints. The builder uses stable body
and frame handles, reports Jacobian rank and degrees of freedom, exports
identity-linked results, and converts explicitly to the portable `SpatialModel`
description. It remains experimental and does not expose spatial dynamics,
rotational equations of motion, contact, or a broad joint catalog.

MBSD `0.8.1` hardens the same experimental kinematics surface: result readers
now validate complete pose, diagnostics, units, conventions, and capability
records. Coordinate drives verify their supplied velocity against the numerical
derivative of their value law. Results separate structural `model_id` from the
sampled runtime `excitation_id` and report linear/angular trajectory derivative
errors in their respective units. Rectangular position problems use a
rank-aware minimum-norm projection. This release does not add spatial dynamics
or change the schema version.

## Fixed-Orientation Spatial Translational Dynamics

MBSD `0.9.1` adds a deliberately restricted translational solver. Positions,
velocities, accelerations, and forces are world-expressed; orientations remain
constant. Forces act at each body's coincident reference point and center of
mass. Torque inputs, offset forces, changing quaternions, rotational equations,
contact, and general six-DOF dynamics are rejected or unavailable.

```python
import numpy as np

from mbsd import Mechanism

m = Mechanism.spatial()
payload = m.body("payload", mass=2.0)
thrust = m.force(payload, (2.0, 0.0, 0.0), name="thrust")
result = m.simulate_fixed_orientation(
    np.linspace(0.0, 1.0, 51),
    forces=(thrust,),
)
print(m.dynamics_diagnostics(result).as_dict())
m.dynamics_result_to_json(result, "fixed-orientation-result.json")
```

Coordinate drives accept `value`, `velocity`, and an optional `acceleration`.
Dynamics uses the supplied acceleration after checking it against velocity, or
derives it with the same adaptive multi-step policy used by the velocity guard.
The dynamic result records the integration tolerances, sample projection,
multiplier equation, physical reaction sign, solver kind, and unsupported
rotational capabilities.

Dynamic schema version 2 assigns every constraint row a stable kind, name,
component, body reference, and unit pair. Use
`mechanism.dynamics_constraint_groups(result)` to inspect residual and
multiplier histories by joint or drive without depending on internal row order.
The fixed-orientation constraints are affine in translation, so the solver
records `none_affine_translation_constraints` instead of applying hidden
stabilization; optional sample projection is recorded separately.

## Free-Body Rotational Dynamics

MBSD `0.9.2` adds an experimental one-body, unconstrained rotational solver.
Orientations are active body-to-world quaternions in `[w, x, y, z]` order;
angular velocity, principal inertia, and Euler's rotational equations use the
body frame. Forces may use world or body coordinates, torques may use either
frame, and a body-fixed application point contributes its moment about the
center of mass.

```python
import numpy as np

from mbsd import Mechanism

m = Mechanism.spatial()
rotor = m.body("rotor", inertia=(2.0, 3.0, 4.0))
motor = m.rigid_body_load(rotor, torque=(2.0, 0.0, 0.0))
result = m.simulate_free_body(np.linspace(0.0, 1.0, 101), loads=(motor,))
print(m.rigid_body_diagnostics(result).as_dict())
```

The `simulate_free_body()` solver deliberately remains a one-body API. Use
`simulate_coupled_rigid_bodies()` for unconstrained multi-body motion. Both
solvers reject ground bodies, joints, and coordinate drives. Use the separate
`simulate_constrained_rigid_bodies()` API added in `0.9.4` for supported
jointed mechanisms.

## Coupled Six-DOF Rigid Bodies

MBSD `0.9.3` adds a separate unconstrained multi-body solver. Each body evolves
its center-of-mass translation and body-to-world quaternion while retaining a
distinct body reference-origin pose for offset-COM geometry. Gravity, external
loads, torques, force-at-point moments, and equal-and-opposite pair interactions
share explicit body/world frame conventions.

```python
import numpy as np

from mbsd import Mechanism
from mbsd.spatial import Pose3D

m = Mechanism.spatial()
a = m.body("a", mass=2.0, pose=Pose3D((-1.0, 0.0, 0.0)))
b = m.body("b", mass=3.0, pose=Pose3D((1.0, 0.0, 0.0)))
pair = m.pair_interaction(a, b, (6.0, 0.0, 0.0), name="pair")
result = m.simulate_coupled_rigid_bodies(
    np.linspace(0.0, 1.0, 51),
    interactions=(pair,),
    gravity_world=(0.0, 0.0, -9.81),
)
print(m.coupled_diagnostics(result).as_dict())
```

Coupled-result schema version 2 binds in-memory results to their originating
mechanism and publishes three deliberately separate identities: `model_id`
for structure, `excitation_id` for sampled loads, and `simulation_id` for the
complete initial state, effective inertia, time grid, and solver controls.
Per-interaction force histories make the equal-and-opposite construction
portable and independently checkable by readers.

This solver remains unconstrained: ground bodies, joints, coordinate drives,
contacts, and reaction forces are rejected. Use
`simulate_constrained_rigid_bodies()` for supported jointed mechanisms.

## Constrained Rotational Dynamics

MBSD `0.9.4` adds `simulate_constrained_rigid_bodies()` for small spatial
mechanisms using spherical, fixed, and revolute joints. The solver assembles
the mass/Jacobian saddle system in center-of-mass linear and body-angular
coordinates, includes `Jdot*v`, and reports physical reactions as
`-J.T*lambda` in world-force/body-moment components.

Position and velocity projection are controlled independently from optional
Baumgarte stabilization. Every choice is recorded in result provenance.
Constraint rows carry stable names, components, body references, residual
units, and multiplier units.

```python
import numpy as np

from mbsd import Mechanism

m = Mechanism.spatial()
ground = m.ground()
rotor = m.body("rotor", inertia=(2.0, 3.0, 4.0))
m.revolute(ground, rotor, name="hinge")
motor = m.rigid_body_load(rotor, torque=(0.0, 0.0, 4.0))
result = m.simulate_constrained_rigid_bodies(
    np.linspace(0.0, 1.0, 51), loads=(motor,)
)
print(m.constrained_diagnostics(result).as_dict())
```

Contact, collision, friction, and flexible-body dynamics remain unsupported.

## Engine-Neutral Interoperability

MBSD `0.9.5` publishes spatial-model schema v3, constrained-result schema v2,
and transform-track handoff schema v1. `build_transform_handoff()` validates a
spatial result before converting poses into an explicit target unit, axis map,
quaternion order, time base, optional geometry references, and content-derived
artifact identity. Both spatial kinematic and rigid-body dynamic result
contracts can supply tracks.

```python
from mbsd.spatial import build_transform_handoff, transform_handoff_to_json

payload = m.constrained_result_to_dict(result)
freecad_tracks = build_transform_handoff(payload, target="freecad")
transform_handoff_to_json(freecad_tracks, "freecad-tracks.json")
```

The core does not import FreeCAD, Blender, browser, or PWA runtimes. Reference
consumers live in the companion examples repository and use only this neutral
contract.

Planar dynamics uses the saddle-multiplier convention
`M @ a + Cq.T @ lambda = Q`; exported generalized constraint reactions are the
physical `-Cq.T @ lambda`. `rotating_frame_pseudo_forces()` includes
centrifugal, Coriolis, and Euler terms. Its body positions, velocities, rotation
origin, gravity, constraints, user loads, and returned force components are all
expressed in the same rotating XYZ basis; the origin defaults to `(0, 0)`.
Angular frame acceleration also contributes the planar Euler torque `-I*alpha`.

Planar dynamics results record the integration method, tolerances, stabilization
and projection settings, package version, and underconstraint policy in their
`provenance` mapping and result JSON export.

## Examples

```sh
uv run python examples/planar_driven_slider.py
uv run python examples/planar_mass_spring.py
uv run python examples/planar_slider_crank_analysis.py
uv run python examples/planar_slider_crank_reactions.py
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

This repository contains the installable public framework. The historical
workbench still contains course material, generated
plots, animations, synthesis experiments, fluid mechanics work, CAD rendering,
and a 3D MBSD kernel. Release branches keep only the public package surface
under:

```text
src/mbsd/
```

Plotting, CAD-specific integrations, browser demos, gallery assets, larger
synthesis workflows, and solved 3D mechanisms are intentionally outside this
core repository. Experimental spatial description and export objects live
under `mbsd.spatial`.

Longer-form docs, release planning, and website content live outside this core
package repository.

## Export Contract

The handoff schemas available since `0.4.0` are:

- `mbsd.planar.mechanism`: bodies, joints, drives, gravity, explicit
  spring-damper descriptors, units, conventions, and caller metadata.
- `mbsd.planar.result`: time, body coordinates, velocities, optional
  accelerations, body poses, diagnostics, units, conventions, and metadata.
- `mbsd.planar.point_trace`: position, velocity, and optional acceleration for
  a named point expressed in a body's local frame.
- `mbsd.planar.joint_loads` (since `0.9.6`): named joint reactions, drive
  efforts and power, and the frame load over time.

Since `0.9.6`, mechanism payloads also carry joint and drive names, portable
motion laws, registered springs, load descriptors, and a top-level `portable`
flag, and `PlanarMechanism.from_json()` reads them back. Dynamic and
inverse-dynamic result payloads include the sampled multipliers.

Every JSON payload contains `schema`, `schema_version`, and `mbsd_version`.
Lengths use metres, angles use radians, time uses seconds, and rotations are
counterclockwise-positive in an inertial XY frame. Array-valued result data is
stored component-by-time. CSV headers carry their SI units.

Springs can be registered with `m.spring(...)` (preferred since `0.9.6`, so
every solve and export uses them) or supplied explicitly when exporting:

```python
from mbsd import Spring

spring = Spring(int(ground), int(slider), k=20.0, c=0.5, l0=0.4)
m.to_json("mechanism.json", springs=[spring], metadata={"consumer": "cad"})
```

A spring must have an explicit natural length to be portable. Arbitrary Python
force callbacks are intentionally not serialized: `force(fn)` loads and
callable drives are exported as opaque entries, and importers reject them.
Exchange the supported force and motion-law descriptors instead.

## Consulting

The project is connected to `multibodysystemsdynamics.com` for teams that need
help setting up a mechanism model, validating a solver, adding a custom joint,
or turning a one-off engineering script into a maintainable workflow.
