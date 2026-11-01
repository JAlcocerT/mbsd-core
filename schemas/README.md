# MBSD Portable Schemas

These JSON Schemas describe the engine-neutral payloads emitted by MBSD
`0.9.7`. Runtime readers remain authoritative because they additionally check
cross-field physics invariants, identities, and force/reaction balances.

- `mbsd.spatial.model-v4.schema.json`: spatial bodies, frames, and stable joint
  descriptors, including revolute and parallel-axes joints and stable entity
  IDs. `model-v3` stays published for readers of older payloads.
- `mbsd.spatial.constrained-result-v3.schema.json`: six-DOF state, reaction,
  multiplier, drive-effort, identity, and provenance records from
  `simulate()`. `constrained-result-v2` stays published.
- `mbsd.interop.transform-tracks-v1.schema.json`: transform animation handoff
  consumed by the FreeCAD-, Blender-, browser-, and generic reference adapters.
- `mbsd.planar.joint-loads-v2.schema.json`: named planar joint reactions,
  drive efforts and power (including distance drives), raw user-constraint
  reactions, and the frame load over time; the handoff for FEM and
  drive-sizing workflows. `joint-loads-v1` stays published.
- `mbsd.operations-v1.json`: the operation manifest (not a JSON Schema). It is
  generated from `mbsd.operation_manifest()` and a test keeps them identical.

Schema changes are additive only within a version. New required fields, changed
meanings, removed fields, and new closed-enum values require a new schema
version. Consumers must reject unsupported versions before interpreting data.

## Migration notes

Spatial-model v4 adds the `parallel_axes` joint kind (`parallel_axis:
"local_z"`) and portable drive laws in `metadata.drives` /
`metadata.revolute_drives`. v3 readers must reject v4 rather than drop the
joint; MBSD reads v1-v4.

Constrained-result v3 adds `parallel_axes` and `revolute_drive` constraint
rows, results with no constraints (free bodies, empty layout), and a
`drive_efforts` block whose efforts equal `-lambda` of each drive row and whose
power equals effort times the sampled drive velocity. MBSD reads v1-v3.

Planar-mechanism v2 adds `joints.point_on_line` and `distance_drive` user
constraints; joint-loads v2 adds a drive `kind`, the `distance` coordinate,
`body_a`, the `constraints` block, and three convention keys. Every v1
convention keeps its value.

Spatial-model v3 adds stable `body:*`, `frame:*`, and `joint:*` identifiers and
makes `revolute` an owned joint kind. The runtime reader continues to accept v1
and v2; a v2 producer must not emit revolute joints. Consumers should retain the
IDs instead of treating array positions as permanent identity.

Constrained-result v2 adds `simulation_id`, an explicit identity scope, stable
body and constraint-row IDs, sampled drive evidence, the constraint Jacobian,
per-interaction force histories, and rotational torque-balance diagnostics. The
runtime reader retains legacy v1 support, but v1 cannot provide the same
cross-field identity and physics guarantees. New adapters should require v2
when those guarantees matter.

Transform-track v1 is an adapter boundary, not an engine API. Consumers must
check `capabilities`, then honor the declared source-to-target axis matrix,
length scale, quaternion order, rotation direction, and shared time grid. Its
`artifact_id` covers the complete handoff content. Core never imports FreeCAD,
Blender, or browser-engine dependencies.
