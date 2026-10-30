# Changelog

## v0.9.6 - Planar Reactions, Portable Descriptors, and Guardrails

- Add `PlanarMechanism.inverse_dynamics()` for fully driven mechanisms and
  return sampled Lagrange multipliers on every `simulate()` result.
- Add `joint_loads()`: named joint reactions (world-frame force and moment
  about the joint point on both bodies), drive effort and power, and the
  frame (shaking) load on ground, with an action-reaction residual.
- Add the versioned `mbsd.planar.joint_loads` v1 payload and JSON Schema as
  the handoff for FEM and drive-sizing workflows.
- Name pins, sliders, motors, and drives (`pin_N`, `slider_N`, `motor_N`,
  `drive_N` by default); names are unique and exported.
- Add portable motion laws (`ConstantSpeed`, `Polynomial`, `Tabulated`) via
  `coordinate_drive(law=...)`; Python callables still work but are exported as
  opaque and non-portable.
- Add registered springs and load descriptors (`spring`, `constant_force`,
  `constant_torque`, `tabulated_force`, `tabulated_torque`) plus the
  non-portable `force(fn)` callback escape hatch.
- Add `PlanarMechanism.from_dict()` / `from_json()`; a `to_json` /
  `from_json` round trip reproduces solves exactly. Opaque entries raise
  `NotPortableError`.
- Add `validate()` (pre-solve structured issues with stable codes) and
  `check()` (pass/fail verdict on residuals, energy drift, or
  equation-of-motion balance, with thresholds).
- Add `mbsd.capabilities()` and `mbsd.operation_manifest()`, with a generated
  `schemas/mbsd.operations-v1.json` for non-Python consumers.
- Add `code`, `context`, and `remediation` to `MechanismSolveError`, and
  `max_steps` / `max_evaluations` resource guards (`code="resource_limit"`).
- Export `portable` flags in mechanism payloads and multipliers in dynamic
  result payloads; the mechanism schema stays at version 1 (additive fields).
- Planar offset-center-of-mass dynamics remain rejected; `validate()` now
  reports it before solving.

## v0.9.5 - Interoperability and Contract Hardening

- Replace the revolute joint's fixed-logarithm chart with transverse relative-
  quaternion constraints that cross 180 degrees and support multiple turns.
- Add constrained-result schema version 2 with mechanism binding, deterministic
  simulation identity, strict semantic validation, exported Jacobians,
  interaction histories, and checked force/torque reactions.
- Use sampled coordinate-drive histories in constrained excitation identity and
  record solver, projection, stabilization, and effective-inertia controls.
- Replace the fixed forward `Jdot*v` difference with a speed-scaled central
  Richardson directional derivative and explicit drive acceleration.
- Add spatial-model schema version 3 with stable entity IDs and an explicit
  revolute-joint contract while retaining readers for versions 1 and 2.
- Publish JSON Schemas and an engine-neutral transform-track handoff with
  explicit axes, units, quaternions, capabilities, optional geometry
  references, and artifact identity for kinematic and dynamic results.
- Keep external-engine dependencies out of core; companion examples provide
  dependency-free FreeCAD, Blender, and browser/PWA reference mappings.

## v0.9.4 - Constrained Rotational Dynamics

- Add coupled Lagrange-multiplier dynamics for spatial rigid bodies with
  position-, velocity-, and acceleration-level constraints.
- Transform analytic joint Jacobians into the center-of-mass/world-linear and
  body-angular velocity basis and include the full directional `Jdot*v` bias.
- Export physical reaction forces and body-frame reaction moments using the
  documented `-J.T*lambda` convention.
- Add independent position and velocity projection plus configurable,
  provenance-recorded Baumgarte stabilization.
- Promote spherical and fixed joints to constrained dynamics and add an
  analytic five-row revolute joint with a free local-Z hinge axis.
- Add a versioned constrained-result schema with mixed-unit row identities,
  multipliers, reactions, residuals, complete body state, and solver controls.
- Harden the coupled-result contract with originating-mechanism checks,
  deterministic simulation identities, explicit identity scopes, and exported
  per-interaction force histories; coupled payloads now use schema version 2.
- Replace ambiguous broad spatial-dynamics discovery with structured,
  solver-specific capability records.
- Validate a gravity-driven spherical/physical pendulum, fixed composite body,
  equilibrium reaction signs, analytic torque-driven hinge motion, and the
  revolute Jacobian against finite differences.
- Keep contact, collision, friction, and flexible bodies outside the framework.

## v0.9.3 - Coupled Six-DOF Rigid-Body Dynamics

- Add an unconstrained multi-body solver coupling center-of-mass translation,
  quaternion orientation, linear velocity, and body-frame angular velocity.
- Support per-body symmetric inertia tensors, offset centers of mass, gravity,
  world/body-frame loads, torques, and force-at-point moments.
- Add equal-and-opposite pair interactions with body-fixed application points
  for explicit multi-body force coupling.
- Export complete body state, accelerations, forces, torques, inertia, gravity,
  load descriptors, interaction descriptors, and solver provenance.
- Add coupled diagnostics for quaternion validity, rotation orthogonality,
  kinetic energy, total linear momentum, total inertial angular momentum, and
  internal-force balance.
- Validate arbitrary-orientation free fall, off-center force, pure couple,
  rotating offset-COM reference motion, isolated multi-body conservation,
  pair-force momentum balance, and world-frame invariance.
- Keep joints, constraints, reaction loads, contact, and collision outside this
  release; constrained rotational dynamics remains the `0.9.4` milestone.

## v0.9.2 - Free-Body Rotational Dynamics

- Add an explicitly unconstrained one-body solver with body-to-world quaternion
  integration and body-frame angular velocity.
- Implement principal-axis Euler equations, gyroscopic terms, world/body load
  frames, torques, force-at-point moments, and offset center-of-mass poses.
- Add rotational diagnostics for quaternion norm, rotation orthogonality,
  energy, linear momentum, and inertial angular momentum.
- Add a versioned free-rigid-body result export with explicit conventions and
  capability boundaries.
- Upgrade fixed-orientation result exports to schema version 2 with constraint
  row identities, masses, and a portable multiplier-to-reaction mapping.
- Harden the fixed-orientation reader against changing rotations, fabricated
  reactions, inconsistent diagnostics, missing conventions, and residuals
  exceeding the recorded tolerance.
- Preserve authored orientations while projecting initial translations in the
  fixed-orientation solver.
- Add free-particle, two-body reaction, constant-torque, force-at-point, and
  asymmetric-top conservation tests.
- Add stable named constraint rows, strict row/reference validation, and a
  public helper grouping residual and multiplier histories by constraint.
- Add an independent sine-force reference fixture, translational convergence
  checks, adversarial initial-state/load/rank tests, and all three torque-free
  principal-axis cases.
- Keep multi-body coupling, joints, contacts, and constrained rotational
  dynamics outside this release.

## v0.9.1 - Fixed-Orientation Spatial Translational Dynamics

- Add an explicitly restricted fixed-orientation translational dynamics solver
  using bounded SciPy integration and Lagrange-multiplier acceleration solves.
- Support validated world-frame forces applied at coincident body reference
  points and centers of mass; reject body-frame loads, offsets, and torques.
- Add supplied-or-adaptively-derived coordinate-drive acceleration laws.
- Export immutable dynamic results with applied forces, physical reactions,
  multipliers, and position-, velocity-, and acceleration-level residuals.
- Add a strict versioned dynamic-result JSON contract and capability flags that
  explicitly reject orientation dynamics and torque inputs.
- Make drive derivative validation adaptive across high-frequency and
  large-absolute-time laws.
- Make spatial in-memory results deeply read-only at their NumPy boundaries and
  validate residual, time, velocity, and pose-history consistency.
- Define planar rotating-frame positions, velocities, gravity, constraints,
  loads, origin, and returned forces in one rotating-coordinate basis; rename
  the complete helper while preserving the former compatibility alias.
- Validate the restricted solver against analytic constant-force and prescribed
  acceleration cases, plus an independently transformed rotating-frame path.
- Keep quaternion integration, torque response, coupled rotational equations,
  contact, and general six-DOF rigid-body dynamics outside this release.

## v0.9.0 - Empty Release Correction

- The published `v0.9.0` tag accidentally referenced the same source commit as
  `v0.8.1`; it did not contain the spatial dynamics changes announced in its
  GitHub release notes.
- Version `v0.9.1` is the first release containing those changes. The `v0.9.0`
  tag remains unchanged to preserve public release history.

## v0.8.1 - Spatial Kinematics Hardening

- Fully validate nested poses, quaternions, body identities, diagnostics,
  capabilities, units, and conventions in spatial kinematic-result payloads.
- Reject non-finite JSON values during result validation and writing.
- Reject coordinate-drive velocity laws that contradict the numerical
  derivative of their value law.
- Report separate finite-difference consistency errors for exported linear and
  angular velocity trajectories.
- Define `model_id` as structural identity and add a sampled `excitation_id`
  for runtime drive behavior.
- Solve rectangular constrained position systems with rank-aware minimum-norm
  world-frame projection and actionable rank/DOF failure details.
- Correct rotating-frame centrifugal, Coriolis, and Euler forces using world
  COM position/velocity and the documented pseudo-force signs.
- Report physical generalized constraint reactions as `-Cq.T @ lambda` and
  document the retained saddle-multiplier convention.
- Include configuration-dependent CAM and point-on-line Jacobian derivatives
  in acceleration-level constraints.
- Separate translation/rotation and linear/angular-velocity constraint metrics
  in spatial result diagnostics, with residual-row units exported explicitly.
- Reject non-unit exported quaternions, negative diagnostics, and diagnostic
  values that disagree with exported residual or trajectory arrays.
- Apply the rotating-frame Euler pseudo-torque `-I*alpha` for planar bodies.
- Reject non-unit quaternions in spatial-model payloads before vocabulary
  construction can normalize them.
- Reject successful spatial results whose position or velocity residual exceeds
  the solver tolerance recorded in provenance.
- Align the lower-level point-on-line constraint with its documented geometry:
  `hi` is the line direction in body j and the residual is perpendicular distance.
- Preserve the `0.8.0` schema version and accept valid `0.8.0` result files.
- Keep spatial dynamics outside the release; it remains planned for `0.9.0`.

## v0.8.0 - Constrained Spatial Kinematics

- Add the experimental `Mechanism.spatial()` builder with stable body and frame
  handles.
- Add canonical spherical and fixed-joint construction without duplicating the
  portable spatial vocabulary.
- Add position and velocity constraint solves using world-frame pose increments.
- Add rank, degree-of-freedom, singularity, position-residual, and
  velocity-residual diagnostics.
- Add `SpatialKinematicResult` with solver status, provenance, capability
  metadata, model identity, poses, and world-expressed velocities.
- Add versioned spatial kinematic-result JSON exports and a validating reader.
- Add explicit conversion between `SpatialMechanism` and portable
  `SpatialModel` descriptions.
- Add unit-aware translation/rotation residual summaries and accept complete
  fixed-joint residual vectors in the compatibility maximum helper.
- Replace the fixed-joint quaternion-vector approximation with an SO(3) log
  residual, an analytic Jacobian, and explicit 180-degree branch rejection.
- Validate schema-v2 spatial-model bodies, frames, joints, units, conventions,
  capabilities, references, and finite JSON values; retain header-only v1 reads.
- Reject one-sample planar dynamics requests at both public and lower-level APIs.
- Keep spatial dynamics, rotational equations of motion, contact, and a broad
  joint catalog outside this release.

## v0.7.0 - Spatial Kinematics Preview

- Add experimental spatial point-position, pose-composition, and pose-inverse
  helpers.
- Add point-coincidence residual evaluation to the canonical vocabulary
  `SphericalJoint3D`, including world endpoints.
- Add `max_spatial_residual()` for simple 3D kinematic validation summaries.
- Add public quaternion composition/inverse, point velocity, world-angular-
  velocity quaternion rate, resolved frame poses, and fixed-joint residuals.
- Add analytic spherical-joint and finite-difference fixed-joint residual
  Jacobians with explicit world-increment conventions and numerical checks.
- Record integration method, tolerances, stabilization, projection, package
  version, and underconstraint policy in planar dynamics result provenance.
- Publish spatial-model schema v2 for the clarified frame conventions, retain
  validated v1 reading, explicit capability metadata, and unknown-version
  rejection.
- Validate lower-level dynamics state shapes, times, tolerances, and initial
  position/velocity consistency before integration.
- Give empty planar diagnostics, solves, and exports a clear validation error.
- Record paired release tags, package compatibility, and schema support in
  `release-compatibility.json`.
- Keep the preview residual-based; no general 3D position solver is claimed yet.

## v0.6.0 - Solver and Contract Hardening

- Add `ConfigurationDiagnostics` for single-state residual and Jacobian health
  checks.
- Add `PlanarMechanism.configuration_diagnostics(q, t, v=None)` for downstream
  validation panels and solver debugging.
- Include velocity-level residuals when a velocity vector is supplied.
- Classify fully constrained, underconstrained, overconstrained, and
  rank-deficient configurations explicitly.
- Make `solve_dynamics_rk45()` perform genuine RK45 integration and preserve
  caller-supplied initial velocities.
- Reject unsupported offset-center-of-mass models at every dynamics entry
  point, including lower-level functions.
- Harden experimental spatial values, collection invariants, inertia checks,
  and frame-pose export conventions.
- Build and import-test distributions in CI.

## v0.5.0 - Experimental 3D Vocabulary

- Add experimental `mbsd.spatial` namespace.
- Add posed `SpatialBody` objects and world- or body-local `Frame3D` objects.
- Add data-only spherical and fixed joint sketches with explicit world-body
  references.
- Add versioned JSON spatial model exports with SI units, coordinate
  conventions, and caller metadata.
- Keep the 3D track vocabulary-only at this stage; no solved 3D kinematics or
  dynamics are claimed yet.

## v0.4.0 - Export Schema

- Add JSON-ready planar mechanism exports.
- Add JSON-ready planar result exports with diagnostics.
- Add wide CSV trajectory exports for solved results.
- Add `PlanarMechanism.to_dict()`, `to_json()`, `result_to_dict()`,
  `result_to_json()`, and `result_to_csv()`.
- Record metadata for built-in motor and coordinate-drive constraints so export
  consumers can identify them.
- Add explicit SI units, coordinate conventions, caller metadata, and portable
  linear spring-damper descriptors to mechanism exports.
- Add JSON and CSV traces for named body-local points used by CAD and browser
  handoffs.
- Reject non-finite and nonserializable export values with clear errors.
- Expose the installed package version through `mbsd.__version__`.
- Keep application UI code outside `mbsd-core`; the core exposes portable
  schemas for downstream consumers.

## v0.3.0 - Synthesis Preview

Prepared release candidate.

- Add `mbsd.planar.synthesis` with a small four-bar synthesis preview.
- Add `FourBar` and `FourBarPose` helpers for pure-geometry assembly.
- Add Grashof classification, rocker-angle sweeps, and affine fitting helpers.
- Add closed-form Freudenstein three-precision-point function synthesis.
- Add explicit underconstrained-dynamics opt-in to `simulate()`.
- Reject offset center-of-mass dynamics until the missing inertial terms are
  implemented.
- Add static model diagnostics with Jacobian rank and singularity status.
- Report both nominal and rank-based model degrees of freedom.
- Fix user-constraint derivative row alignment after lower-level constraint
  types.
- Fix offset prismatic-joint Jacobian time derivatives used by acceleration
  solves.
- Validate velocity and acceleration arrays in result diagnostics.
- Remove `mypy` from dev dependencies until type checking is wired into CI.
- Add tests for synthesis recovery, offset-slider acceleration derivatives,
  invalid geometry, and affine-fit behavior.
- Keep top-level `mbsd` exports unchanged.

## v0.2.0 - Validation

Prepared release candidate.

- Add result-wide constraint residual helpers.
- Add `PlanarMechanism.assert_constraints_satisfied()` for explicit validation
  failures.
- Add `PlanarMechanism.diagnostics()` and `ResultDiagnostics` for compact
  simulation summaries.
- Add tests for diagnostics, residual shapes, and failed constraint assertions.
- Preserve the small top-level API: `Mechanism`, `Spring`, and
  `MechanismSolveError`.
- Keep core runtime dependencies limited to NumPy and SciPy.

## v0.1.0 - Initial Core

Initial public core release candidate.

- Add installable `mbsd` Python package.
- Add planar mechanism builder API through `Mechanism.planar()`.
- Add planar rigid bodies, revolute joints, prismatic joints, coordinate drives,
  angular motors, kinematic solves, and constrained dynamics.
- Add examples for a driven slider, mass-spring mechanism, and slider-crank
  analysis.
- Add pytest smoke coverage for rotation matrices, slider tracking,
  mass-spring dynamics, and Jacobian finite-difference consistency.
- Keep longer-form documentation and website content outside the core package
  repository.
- Keep release planning outside the core package repository.
- Keep top-level exports intentionally small: `Mechanism`, `Spring`, and
  `MechanismSolveError`.
- Move Matplotlib out of core runtime dependencies and into the `plot` extra.
- Add explicit solver errors for singular, overconstrained, and accidental
  underconstrained solves.
- Add input validation and constraint-residual helpers.
- Add GitHub Actions CI for Python 3.10 through 3.13.
- Keep generated media, browser demo, CAD/rendering workflows, synthesis
  batches, and experimental 3D work out of the first core release.
