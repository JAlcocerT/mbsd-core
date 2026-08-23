# Changelog

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
