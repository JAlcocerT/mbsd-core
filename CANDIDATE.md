# v0.9.8 Release Candidate Log

v0.9.8 is the 1.0 release candidate. Its public surface (`mbsd.api_surface()`)
and emitted schemas are frozen. This file records the candidate period: who
pinned the candidate, every reported incompatibility, and its resolution.

## Candidate period

| | |
|---|---|
| Candidate | mbsd `0.9.8`, mbsd-examples `0.9.8` |
| Opened | 2026-11-08 |
| Planned close | 2026-11-22 (two weeks), extended by any candidate break |
| Pinned by | `mbsd-examples` (`mbsd>=0.9.8,<0.10`, CI on core `v0.9.8`), the Blender reference consumer |

The period closes only when every entry below is resolved and no breaking
incompatibility is open. A change to the frozen surface or to an emitted
schema is a **candidate break**: it restarts the affected gates (API surface,
corpus, packaging qualification, reference consumer) and the two-week clock.

## Gates

| Gate | Evidence | Status |
|---|---|---|
| No open correctness issue in a stable capability | issues below | open period |
| No unexplained corpus regression | `tests/test_corpus.py` against `corpus/expected-v1.json` | passing |
| Stable API frozen | `tests/test_api_surface.py`, `schemas/mbsd.api-surface-v1.json` | passing |
| Schemas frozen | `release-compatibility.json`, fixtures `corpus/fixtures/` | passing |
| Python / dependency / packaging matrix | `scripts/qualify_release.sh`, CI `qualify` job | passing |
| Examples against the candidate tag | mbsd-examples CI on core `v0.9.8` | passing |
| Reference consumer | Blender 5.1 headless run on golden fixtures | passing |
| Docs agree with machine-readable claims | `tests/test_api_surface.py`, operation manifest test | passing |
| Feedback period ends without a breaking incompatibility | this log | open until 2026-11-22 |

## Reported incompatibilities

| # | Date | Reporter | Surface | Description | Breaking | Resolution |
|---|---|---|---|---|---|---|
| 1 | 2026-11-08 | corpus | `mbsd.spatial.model` v3 | A v3 payload from 0.9.6 `SpatialMechanism.to_dict()` fails validation: that method overwrote the model capabilities with the builder's (fixed in 0.9.7). | no (pre-candidate, documented) | Documented in `STABILITY.md`; 0.9.6's canonical `to_spatial_model().as_dict()` payload is the fixture and reads correctly. |
| 2 | 2026-11-08 | CI (Python 3.11-3.13) | validation corpus | `spatial.tolerance_convergence` recorded truncation-level integration errors as regression values (rtol 1e-6); they differ in the last digits across CPUs / BLAS builds (e.g. 1.7e-10 locally vs 4.4e-10 on CI). | no (test asset only; no API or schema change) | The errors are now bounded (`<= max(10 rtol, 1e-8)`) and must still fall monotonically; only that case was re-recorded. |
| 3 | 2026-11-08 | CI (mbsd-examples, Python 3.11) | golden interop fixture test | A sample at a half turn (w ~ 0) came out as `-q` instead of `q`, the same rotation: the sign there is set by round-off. | no (test only) | The reproducibility test compares quaternions up to sign, as the Blender consumer already does. |
