# Changelog

## v0.3.0 - Week 3 Synthesis Preview

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

## v0.2.0 - Week 2 Validation

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

## v0.1.0 - Week 1 Core

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
