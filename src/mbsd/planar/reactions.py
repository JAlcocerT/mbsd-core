"""Physical joint loads from planar Lagrange multipliers.

The saddle system is ``M a + C_q.T λ = Q``, so the generalized reaction of a
constraint group is ``-C_q[rows].T @ λ[rows]``. For each body touched by the
group, the translational entries are the world-frame reaction force and the
rotational entry is the moment about that body's reference point. This module
re-expresses those moments about the joint's world point, so a joint load
reads the same way a free-body diagram does.

Row order mirrors ``constraints.constraints``: ground fixture, revolute,
prismatic, CAM, point-on-line, gear, user constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .jacobians import jacobian
from .kinematics import rot_mat


@dataclass(frozen=True)
class ConstraintGroup:
    """Rows of the constraint vector owned by one joint or drive."""

    kind: str
    index: int
    name: str
    start: int
    count: int
    bodies: tuple[int, ...]
    metadata: dict = field(default_factory=dict)

    @property
    def rows(self) -> slice:
        return slice(self.start, self.start + self.count)


def constraint_groups(mbody) -> list[ConstraintGroup]:
    """Return the constraint groups in assembly order."""
    groups = [ConstraintGroup("ground_fixture", 0, "ground_fixture", 0, 3, (0,))]
    row = 3

    def add(kind, items, count, bodies_of, default_prefix):
        nonlocal row
        for index, item in enumerate(items):
            n = count(item)
            name = getattr(item, "name", "") or f"{default_prefix}_{index}"
            metadata = dict(getattr(item, "metadata", {}) or {})
            groups.append(ConstraintGroup(kind, index, name, row, n, bodies_of(item), metadata))
            row += n

    add("revolute", mbody.rev_joints, lambda j: 2, lambda j: (j.i, j.j), "revolute")
    add("prismatic", mbody.prism_joints, lambda j: 2, lambda j: (j.i, j.j), "prismatic")
    add("cam", mbody.leva_joints, lambda j: 3, lambda j: (j.i, j.j), "cam")
    add("point_on_line", mbody.punto_linea_joints, lambda j: 1, lambda j: (j.i, j.j), "point_on_line")
    add("gear", mbody.gear_joints, lambda j: 1, lambda j: (j.i, j.j), "gear")

    def user_bodies(uc):
        body = (getattr(uc, "metadata", {}) or {}).get("body")
        return () if body is None else (int(body),)

    def user_kind_groups():
        nonlocal row
        for index, uc in enumerate(mbody.user_constraints):
            metadata = dict(getattr(uc, "metadata", {}) or {})
            kind = "drive" if metadata.get("kind") in {"motor", "coordinate_drive"} else "user"
            name = getattr(uc, "name", "") or f"{kind}_{index}"
            groups.append(ConstraintGroup(kind, index, name, row, uc.count, user_bodies(uc), metadata))
            row += uc.count

    user_kind_groups()
    return groups


@dataclass(frozen=True)
class JointLoads:
    """Per-joint and per-drive load histories over a result.

    ``joints[name]`` holds, for both connected bodies, the world-frame force
    the joint applies to that body and the moment about the joint's world
    point. ``drives[name]`` holds the generalized effort (N or N*m) the drive
    applies to its body coordinate and the matching power. ``frame`` holds the
    total force and moment (about the world origin) the mechanism exerts on the
    ground body, assuming drive actuators react against ground.
    """

    t: np.ndarray
    joints: dict[str, dict[str, Any]]
    drives: dict[str, dict[str, Any]]
    frame: dict[str, np.ndarray]
    max_action_reaction_residual: float


_DRIVE_OFFSETS = {"x": 0, "y": 1, "theta": 2}


def joint_loads(mbody, t: np.ndarray, q: np.ndarray, v: np.ndarray, multipliers: np.ndarray) -> JointLoads:
    groups = constraint_groups(mbody)
    nsteps = len(t)
    joints: dict[str, dict[str, Any]] = {}
    drives: dict[str, dict[str, Any]] = {}
    frame_force = np.zeros((2, nsteps))
    frame_moment = np.zeros(nsteps)
    max_residual = 0.0

    for group in groups:
        if group.kind in {"revolute", "prismatic", "point_on_line", "gear", "cam"}:
            joint = _joint_object(mbody, group)
            joints[group.name] = {
                "kind": group.kind,
                "body_i": group.bodies[0],
                "body_j": group.bodies[1],
                "point": np.zeros((2, nsteps)),
                "force_on_i": np.zeros((2, nsteps)),
                "moment_on_i": np.zeros(nsteps),
                "force_on_j": np.zeros((2, nsteps)),
                "moment_on_j": np.zeros(nsteps),
            }
            joints[group.name]["_joint"] = joint
        elif group.kind == "drive":
            drives[group.name] = {
                "body": group.metadata.get("body"),
                "coordinate": group.metadata.get("coordinate"),
                "effort": np.zeros(nsteps),
                "power": np.zeros(nsteps),
            }

    for step in range(nsteps):
        qk = q[:, step]
        vk = v[:, step]
        Cq = jacobian(mbody, qk, float(t[step]))
        lam = multipliers[:, step]
        for group in groups:
            Q = -Cq[group.rows].T @ lam[group.rows]
            if group.name in joints:
                record = joints[group.name]
                point = _joint_point(record["_joint"], group.kind, qk)
                record["point"][:, step] = point
                totals = np.zeros(3)
                for side, body in (("i", group.bodies[0]), ("j", group.bodies[1])):
                    force, moment = _about_point(Q, qk, body, point)
                    record[f"force_on_{side}"][:, step] = force
                    record[f"moment_on_{side}"][step] = moment
                    totals += np.array([force[0], force[1], moment])
                    if body == 0:
                        frame_force[:, step] += force
                        frame_moment[step] += moment + _cross(point, force)
                max_residual = max(max_residual, float(np.max(np.abs(totals))))
            elif group.name in drives:
                record = drives[group.name]
                coordinate = record["coordinate"]
                if record["body"] is None or coordinate not in _DRIVE_OFFSETS:
                    continue
                col = 3 * int(record["body"]) + _DRIVE_OFFSETS[coordinate]
                effort = float(Q[col])
                record["effort"][step] = effort
                record["power"][step] = effort * vk[col]
                body = int(record["body"])
                if body != 0:
                    reference = qk[3 * body : 3 * body + 2]
                    if coordinate == "theta":
                        frame_moment[step] -= effort
                    else:
                        reaction = np.zeros(2)
                        reaction[_DRIVE_OFFSETS[coordinate]] = -effort
                        frame_force[:, step] += reaction
                        frame_moment[step] += _cross(reference, reaction)

    for record in joints.values():
        record.pop("_joint")
    return JointLoads(
        t=np.asarray(t, dtype=float),
        joints=joints,
        drives=drives,
        frame={"force": frame_force, "moment": frame_moment},
        max_action_reaction_residual=max_residual,
    )


def _joint_object(mbody, group: ConstraintGroup):
    collection = {
        "revolute": mbody.rev_joints,
        "prismatic": mbody.prism_joints,
        "cam": mbody.leva_joints,
        "point_on_line": mbody.punto_linea_joints,
        "gear": mbody.gear_joints,
    }[group.kind]
    return collection[group.index]


def _joint_point(joint, kind: str, q: np.ndarray) -> np.ndarray:
    """World point of the joint: body-j attachment point (body-j origin for gears)."""
    j = joint.j
    origin = q[3 * j : 3 * j + 2]
    local = getattr(joint, "rj", None)
    if kind == "gear" or local is None:
        return origin.copy()
    return origin + rot_mat(q[3 * j + 2]) @ np.asarray(local, dtype=float)


def _about_point(Q: np.ndarray, q: np.ndarray, body: int, point: np.ndarray):
    start = 3 * body
    force = Q[start : start + 2]
    moment_reference = Q[start + 2]
    lever = q[start : start + 2] - point
    return force.copy(), float(moment_reference + _cross(lever, force))


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])
