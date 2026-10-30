# MBSD Portable Schemas

These JSON Schemas describe the engine-neutral payloads emitted by MBSD
`0.9.6`. Runtime readers remain authoritative because they additionally check
cross-field physics invariants, identities, and force/reaction balances.

- `mbsd.spatial.model-v3.schema.json`: spatial bodies, frames, and stable joint
  descriptors, including revolute joints and stable entity IDs.
- `mbsd.spatial.constrained-result-v2.schema.json`: constrained six-DOF state,
  reaction, multiplier, identity, and provenance records.
- `mbsd.interop.transform-tracks-v1.schema.json`: transform animation handoff
  consumed by the FreeCAD-, Blender-, browser-, and generic reference adapters.
- `mbsd.planar.joint-loads-v1.schema.json`: named planar joint reactions,
  drive efforts and power, and the frame load over time; the handoff for FEM
  and drive-sizing workflows.
- `mbsd.operations-v1.json`: the operation manifest (not a JSON Schema). It is
  generated from `mbsd.operation_manifest()` and a test keeps them identical.

Schema changes are additive only within a version. New required fields, changed
meanings, removed fields, and new closed-enum values require a new schema
version. Consumers must reject unsupported versions before interpreting data.

## Migration notes

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
