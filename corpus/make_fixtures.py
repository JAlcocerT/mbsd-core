"""Write schema fixtures with whichever mbsd is importable.

Run once per released version to freeze what that release emitted and
computed, e.g. against a worktree of the v0.9.6 tag:

    PYTHONPATH=/path/to/worktree-v0.9.6/src uv run python corpus/make_fixtures.py

Each fixture stores the payload plus the values that release computed from
it, so the corpus can check that today's readers reproduce them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import mbsd
from mbsd import Mechanism

OUT = Path(__file__).resolve().parent / "fixtures"
VERSION = mbsd.__version__
TIMES = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]


def _write(name: str, payload: dict, computed: dict) -> None:
    record = {
        "fixture": name,
        "produced_by": f"mbsd {VERSION}",
        "schema": payload["schema"],
        "schema_version": payload["schema_version"],
        "payload": payload,
        "computed": computed,
    }
    path = OUT / f"{name}.json"
    if path.exists():
        # Keep the oldest release that emitted this schema version.
        print("kept", path.name, "from", json.loads(path.read_text())["produced_by"])
        return
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", path.name)


def planar_slider_crank():
    m = Mechanism.planar(gravity=(0.0, -9.81))
    g = m.ground()
    crank = m.body("crank", mass=1.0, inertia=0.01)
    rod = m.body("rod", mass=0.5, inertia=0.02)
    piston = m.body("piston", mass=0.8, inertia=0.001)
    m.pin(g, crank, name="main")
    m.pin(crank, rod, point_a=(0.1, 0.0), name="crankpin")
    m.pin(rod, piston, point_a=(0.3, 0.0), name="wristpin")
    m.slider(g, piston, axis=(1.0, 0.0), name="bore")
    m.motor(crank, omega=20.0, name="engine")
    q0 = np.array([0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0.4, 0, 0], dtype=float)
    return m, q0


def planar_fixtures():
    m, q0 = planar_slider_crank()
    t = np.array(TIMES)
    result = m.inverse_dynamics(t, q0=q0)
    loads = m.joint_loads(result)
    payload = m.to_dict()
    tag = f"planar-mechanism-v{payload['schema_version']}"
    _write(tag, payload, {
        "q0": q0.tolist(), "t": TIMES,
        "engine_torque": loads.drives["engine"]["effort"].tolist(),
        "main_force_on_j": loads.joints["main"]["force_on_j"].tolist(),
        "piston_x": result.q[9].tolist(),
    })
    loads_payload = m.joint_loads_to_dict(result)
    _write(f"planar-joint-loads-v{loads_payload['schema_version']}", loads_payload, {})


def spatial_fixtures():
    from mbsd.spatial import Pose3D

    m = Mechanism.spatial()
    g = m.ground()
    arm = m.body("arm", mass=2.0, inertia=(0.01, 0.2, 0.2), pose=Pose3D(translation=(0.5, 0.0, 0.0)))
    if hasattr(m, "revolute"):
        m.revolute(g, arm, frame_j=Pose3D(translation=(-0.5, 0.0, 0.0)), name="hinge")
    else:
        m.spherical(g, arm, point_j=(-0.5, 0.0, 0.0), name="hinge")
    # 0.9.6 SpatialMechanism.to_dict() replaced the model capabilities with the
    # builder's, so it never produced a valid spatial-model payload (fixed in
    # 0.9.7). Its canonical portable emission is to_spatial_model().as_dict().
    payload = m.to_dict() if hasattr(m, "from_dict") else m.to_spatial_model().as_dict()
    t = np.array(TIMES)
    simulate = getattr(m, "simulate", None) or m.simulate_constrained_rigid_bodies
    result = simulate(t, gravity_world=(0.0, -9.81, 0.0))
    _write(f"spatial-model-v{payload['schema_version']}", payload, {
        "t": TIMES,
        "com_y": [float(x) for x in result.center_of_mass_position[:, 1, 1]],
    })
    result_payload = m.constrained_result_to_dict(result)
    _write(f"spatial-constrained-result-v{result_payload['schema_version']}", result_payload, {})


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    print(f"mbsd {VERSION} from {Path(mbsd.__file__).parent}", file=sys.stderr)
    planar_fixtures()
    spatial_fixtures()
