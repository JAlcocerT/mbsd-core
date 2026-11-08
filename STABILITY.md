# MBSD Stability, Compatibility and Migration

This document is the human-readable side of the 1.0 contract. The
machine-readable side is `mbsd.api_surface()` (generated copy:
`schemas/mbsd.api-surface-v1.json`), which classifies every public symbol and
builder method. `tests/test_api_surface.py` keeps the two in step with the
code, so no public symbol can appear, disappear or change class silently.

## Stability classes

| Class | Meaning | Where |
|---|---|---|
| **stable** | Semantic versioning from 1.0: a breaking change needs a major release. | `mbsd`, `mbsd.errors`, `mbsd.reports`, `mbsd.planar`, `mbsd.planar.analysis`, `mbsd.spatial` (`__all__`), and the stable builder methods |
| **experimental** | Importable and tested, may change in a minor release, excluded from the stable claim. | `mbsd.experimental`; `SpatialMechanism.simulate_fixed_orientation()`, `force()`, `dynamics_*()` |
| **deprecated** | Works and emits `DeprecationWarning`; removed in **1.0.0**. | the pre-0.9.7 free-body / coupled result types in `mbsd.spatial`; `SpatialMechanism.simulate_free_body()`, `simulate_coupled_rigid_bodies()`, `simulate_constrained_rigid_bodies()`, `rigid_body_*()`, `coupled_*()` |
| **internal** | No promise. | everything else: `mbsd.planar.solver`, `.dynamics`, `.jacobians`, `.constraints`, `mbsd.spatial.builder`, names starting with `_`, the `model` attribute's kernel objects |

Check a symbol at runtime:

```python
import mbsd
from mbsd import stability

stability.status("mbsd.spatial", "SpatialMechanism")   # "stable"
stability.status("mbsd.experimental", "SpatialForce")  # "experimental"
mbsd.api_surface()["namespaces"]["mbsd.spatial.SpatialMechanism"]["simulate_free_body"]
# {'status': 'deprecated', 'removal': '1.0.0', 'replacement': 'simulate(), ...'}
```

## Policies

**Versioning.** From 1.0.0, stable APIs follow semantic versioning. Patch
releases fix bugs without changing the stable surface; minor releases add
features compatibly; anything that breaks a stable API, a stable schema
emission or a documented convention needs a major release.

**Deprecation.** A deprecated symbol keeps working, warns with
`DeprecationWarning` naming its replacement, and is removed no earlier than the
next major release. Everything deprecated during 0.x is removed in 1.0.0.

**Experimental.** Experimental symbols may change or disappear in a minor
release. Promoting one to stable is a minor-release change and is recorded in
the changelog.

**Schemas.** Schema versions are independent of the package version.

- Within a schema version, changes are additive only (new optional fields).
- New required fields, changed meanings, removed fields and new values in a
  closed enum need a new schema version.
- Readers must reject a version they do not know instead of guessing.
- Every schema version MBSD has emitted stays readable for at least one major
  release after it stops being emitted.
- Migration notes for every transition live in `schemas/README.md`; old
  payloads written by the releases themselves are kept as fixtures in
  `corpus/fixtures/` and re-solved by the validation corpus.

**Candidate changes.** During the v0.9.8 candidate period every change carries
a compatibility note in the changelog. Any change to the frozen surface or to
an emitted schema is an explicit candidate break: it is logged in
`CANDIDATE.md` and restarts the affected gates.

## Compatibility tables

### Package, Python and dependencies

| mbsd | Python | numpy | scipy | Qualified by |
|---|---|---|---|---|
| 0.9.8 | 3.10, 3.11, 3.12, 3.13 | >=2.0, <3 | >=1.13, <2 | `scripts/qualify_release.sh`: wheel on every Python, sdist, dependency floor (numpy 2.0 / scipy 1.13 on 3.10) and ceiling (newest on 3.13) |

### Schemas (emitted / readable)

| Schema | 0.9.6 | 0.9.7 | 0.9.8 |
|---|---|---|---|
| `mbsd.planar.mechanism` | 1 / 1 | 2 / 1-2 | **3** / 1-3 |
| `mbsd.planar.result` | 1 | 1 | 1 (dynamic results now carry `a`) |
| `mbsd.planar.point_trace` | 1 | 1 | 1 |
| `mbsd.planar.joint_loads` | 1 | 2 | **3** |
| `mbsd.spatial.model` | 3 / 1-3 | 4 / 1-4 | 4 / 1-4 |
| `mbsd.spatial.constrained_rigid_body_dynamic_result` | 2 | 3 / 1-3 | 3 / 1-3 |
| `mbsd.interop.transform_tracks` | 1 | 1 | 1 |
| `mbsd.operations` | 1 | 1 | 1 (adds `api_surface`) |
| `mbsd.capabilities` | 1 | 1 | 1 |
| `mbsd.api_surface` | - | - | **1** |
| `mbsd.validation_corpus` | - | - | **1** |

### Examples and adapters

| mbsd | mbsd-examples | Reference consumer | Other adapters |
|---|---|---|---|
| 0.9.8 | 0.9.8 (`mbsd>=0.9.8,<0.10`) | Blender 5.1 (`examples/adapters/blender_consumer.py`, run headless against golden fixtures) | FreeCAD, browser: record mapping only (`*_reference.py`), after 1.0 |

## Migration notes (0.x to 1.0)

**Spatial dynamics entry points (0.9.7).** Replace `simulate_free_body()`,
`simulate_coupled_rigid_bodies()` and `simulate_constrained_rigid_bodies()`
with `SpatialMechanism.simulate()`; it returns a
`ConstrainedRigidBodyDynamicsResult` for free, coupled, jointed and driven
models alike. Export it with `constrained_result_to_dict()`.

**Fixed-orientation dynamics (0.9.8).** `SpatialForce`,
`SpatialDynamicsResult` and their payload helpers moved to
`mbsd.experimental`. The `mbsd.spatial` names still work until 1.0 but warn.

**Legacy result readers (0.9.8).** `load_rigid_body_result_payload()`,
`load_coupled_result_payload()` and their validators are deprecated; read old
payloads before 1.0 and re-run the model through `simulate()`.

**Builder `to_dict()` capabilities (0.9.7).** `SpatialMechanism.to_dict()`
keeps the model's `capabilities` and puts the builder's in
`solver_capabilities`. In 0.9.6 it overwrote the model capabilities, so a
0.9.6 `SpatialMechanism.to_dict()` payload is not a valid spatial-model
payload; 0.9.6's portable emission was `to_spatial_model().as_dict()`, which
current readers accept (fixture `corpus/fixtures/spatial-model-v3.json`).

**Planar mechanism payloads.** v1 (0.9.6) and v2 (0.9.7) payloads are read
as-is and re-solve to what those releases computed (validation corpus). v3
(0.9.8) adds `gear`, `rack` and `relative_to`; a v2 reader rejects it.

**Joint loads.** v3 adds the required `meshes` block and `relative_to` on
drives; consumers of v2 keep working on v2 payloads but must accept `meshes`
to read v3.

**Raw `UserConstraint` gears.** Rebuild hand-written gear / rack / reverser
constraints with `gear()` and `rack()`: they become portable, named, and
their reactions reach the frame load (see the readme's sign conventions).

**Absolute motors on moving parts.** A motor mounted on a moving body should
use `relative_to=`; an absolute drive reacts against ground.
