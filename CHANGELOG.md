# Changelog

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
