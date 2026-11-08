"""Deterministic output: the same inputs give bitwise-identical results.

Within one platform and one numpy / scipy / BLAS build, repeated solves (in
the same process or a fresh one) produce identical arrays and identities.
Across platforms results may differ in the last bits, which is why the
validation corpus compares with tolerances (``corpus/run.py``).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = r"""
import hashlib, sys
import numpy as np
from mbsd import ConstantSpeed, Mechanism, Polynomial
from mbsd.spatial import Pose3D

def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
    return h.hexdigest()

p = Mechanism.planar()
g = p.ground()
a = p.body("upper", mass=1.0, inertia=0.02, center_of_mass=(0.25, 0.0))
b = p.body("lower", mass=0.7, inertia=0.01, center_of_mass=(0.2, 0.0))
p.pin(g, a, name="shoulder")
p.pin(a, b, point_a=(0.5, 0.0), name="elbow")
r = p.simulate(np.linspace(0.0, 1.0, 51), np.array([0, 0, 0, 0, 0, 0, 0.5, 0, 0.0]), allow_underconstrained=True)
planar = digest(r.q, r.v, r.a, r.multipliers)

s = Mechanism.spatial()
g = s.ground()
arm = s.body("arm", mass=2.0, inertia=(0.01, 0.2, 0.2), pose=Pose3D(translation=(0.5, 0.0, 0.0)))
hinge = s.revolute(g, arm, frame_j=Pose3D(translation=(-0.5, 0.0, 0.0)), name="hinge")
s.revolute_drive(hinge, law=Polynomial((0.0, 0.0, 1.5)), name="motor")
res = s.simulate(np.linspace(0.0, 0.5, 26), gravity_world=(0.0, -9.81, 0.0))
spatial = digest(res.center_of_mass_position, res.angular_velocity_body, res.multipliers)
print(planar, spatial, res.simulation_id)
"""


def _run_in_fresh_process() -> str:
    src = Path(__file__).resolve().parents[1] / "src"
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT], check=True, capture_output=True, text=True,
        env={"PYTHONPATH": str(src), "PYTHONHASHSEED": "random", "PATH": ""},
    )
    return completed.stdout.strip()


def test_repeated_solves_are_bitwise_identical_in_process():
    namespace: dict = {}
    outputs = []
    for _ in range(2):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(SCRIPT, "<determinism>", "exec"), namespace)
        outputs.append(buffer.getvalue().strip())
    assert outputs[0] == outputs[1]


def test_fresh_processes_produce_identical_digests():
    first, second = _run_in_fresh_process(), _run_in_fresh_process()
    assert first == second
    assert len(first.split()) == 3


def test_digest_helper_is_sensitive():
    a = np.array([1.0, 2.0])
    b = np.nextafter(a, 3.0)
    assert hashlib.sha256(a.tobytes()).hexdigest() != hashlib.sha256(b.tobytes()).hexdigest()
