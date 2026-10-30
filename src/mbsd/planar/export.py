"""Portable export helpers for planar MBSD mechanisms and results."""

from __future__ import annotations

import csv
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .builder import (
    BodyHandle,
    DynamicsResult,
    InverseDynamicsResult,
    KinematicResult,
    PlanarMechanism,
)
from .forces import Spring
from .kinematics import point_acceleration, point_position, point_velocity
from .model import MBody


SCHEMA_VERSION = 1
JOINT_LOADS_SCHEMA_VERSION = 1


def _units() -> dict[str, str]:
    return {
        "length": "m",
        "angle": "rad",
        "time": "s",
        "mass": "kg",
        "inertia": "kg*m^2",
        "force": "N",
        "torque": "N*m",
        "linear_velocity": "m/s",
        "angular_velocity": "rad/s",
        "linear_acceleration": "m/s^2",
        "angular_acceleration": "rad/s^2",
        "spring_stiffness": "N/m",
        "damping": "N*s/m",
    }


def _conventions() -> dict[str, Any]:
    return {
        "reference_frame": "inertial_xy",
        "coordinates": ["x", "y", "theta"],
        "rotation": "counterclockwise_positive",
        "body_points": "body_local_xy",
        "array_layout": "component_by_time",
    }


def mechanism_to_dict(
    mechanism: PlanarMechanism | MBody,
    *,
    springs: Iterable[Spring] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready description of a planar mechanism."""
    mbody = _as_mbody(mechanism)
    registered_springs = list(getattr(mechanism, "springs", []))
    loads = list(getattr(mechanism, "loads", []))
    all_springs = registered_springs + list(() if springs is None else springs)
    user_constraints = [
        _user_constraint_to_dict(index, constraint)
        for index, constraint in enumerate(mbody.user_constraints)
    ]
    portable = (
        all(item.get("portable", False) for item in user_constraints)
        and all(load.portable for load in loads)
        and not (mbody.leva_joints or mbody.punto_linea_joints or mbody.gear_joints)
    )
    return {
        "schema": "mbsd.planar.mechanism",
        "portable": portable,
        "schema_version": SCHEMA_VERSION,
        "mbsd_version": _package_version(),
        "dimension": 2,
        "units": _units(),
        "conventions": _conventions(),
        "metadata": _metadata(metadata),
        "gravity": _vector(mbody.g),
        "counts": {
            "bodies": mbody.nb,
            "coordinates": mbody.ncoord,
            "constraints": mbody.nrestr,
            "revolute_joints": mbody.nr,
            "prismatic_joints": mbody.np_,
            "user_constraints": mbody.nrus,
        },
        "bodies": [
            {
                "index": index,
                "name": body.name,
                "mass": float(body.mass),
                "inertia": float(body.inertia),
                "center_of_mass": _vector(body.rG),
            }
            for index, body in enumerate(mbody.bodies)
        ],
        "joints": {
            "revolute": [
                {
                    "index": index,
                    "name": joint.name or f"pin_{index}",
                    "body_i": joint.i,
                    "body_j": joint.j,
                    "point_i": _vector(joint.ri),
                    "point_j": _vector(joint.rj),
                }
                for index, joint in enumerate(mbody.rev_joints)
            ],
            "prismatic": [
                {
                    "index": index,
                    "name": joint.name or f"slider_{index}",
                    "rail_body": joint.i,
                    "slider_body": joint.j,
                    "point_rail": _vector(joint.ri),
                    "point_slider": _vector(joint.rj),
                    "axis": _vector(np.array([joint.hi[1], -joint.hi[0]])),
                    "normal": _vector(joint.hi),
                }
                for index, joint in enumerate(mbody.prism_joints)
            ],
        },
        "user_constraints": user_constraints,
        "forces": {
            "gravity": _vector(mbody.g),
            "springs": [
                _spring_to_dict(mbody, index, spring)
                for index, spring in enumerate(all_springs)
            ],
            "loads": [_json_ready(load.to_dict()) for load in loads],
            "custom": [],
        },
    }


def mechanism_to_json(
    mechanism: PlanarMechanism | MBody,
    path: str | Path,
    *,
    indent: int = 2,
    springs: Iterable[Spring] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a planar mechanism export JSON file and return its path."""
    return _write_json(
        mechanism_to_dict(mechanism, springs=springs, metadata=metadata),
        path,
        indent=indent,
    )


def result_to_dict(
    mechanism: PlanarMechanism,
    result: KinematicResult | DynamicsResult,
    *,
    include_diagnostics: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready trajectory/result export."""
    mechanism._validate_result(result)
    payload: dict[str, Any] = {
        "schema": "mbsd.planar.result",
        "schema_version": SCHEMA_VERSION,
        "mbsd_version": _package_version(),
        "result_type": _result_type(result),
        "dimension": 2,
        "units": _units(),
        "conventions": _conventions(),
        "metadata": _metadata(metadata),
        "time": _vector(result.t),
        "coordinates": {
            "q": _matrix(result.q),
            "v": _matrix(result.v),
        },
        "body_poses": _body_poses(mechanism.model, result),
    }
    if hasattr(result, "a"):
        payload["coordinates"]["a"] = _matrix(result.a)
    if hasattr(result, "provenance"):
        payload["provenance"] = _metadata(dict(result.provenance))
    multipliers = getattr(result, "multipliers", None)
    if multipliers is not None:
        payload["coordinates"]["multipliers"] = _matrix(multipliers)
    if include_diagnostics:
        payload["diagnostics"] = mechanism.diagnostics(result).as_dict()
    return payload


def result_to_json(
    mechanism: PlanarMechanism,
    result: KinematicResult | DynamicsResult,
    path: str | Path,
    *,
    indent: int = 2,
    include_diagnostics: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a planar result export JSON file and return its path."""
    return _write_json(
        result_to_dict(
            mechanism,
            result,
            include_diagnostics=include_diagnostics,
            metadata=metadata,
        ),
        path,
        indent=indent,
    )


def result_to_csv(
    mechanism: PlanarMechanism,
    result: KinematicResult | DynamicsResult,
    path: str | Path,
) -> Path:
    """Write a wide trajectory CSV file and return its path."""
    mechanism._validate_result(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_acceleration = hasattr(result, "a")
    headers = ["time_s"]
    for index, body in enumerate(mechanism.model.bodies):
        prefix = f"body{index}_{_slug(body.name)}"
        headers.extend([f"{prefix}_x_m", f"{prefix}_y_m", f"{prefix}_theta_rad"])
        headers.extend(
            [
                f"{prefix}_vx_m_per_s",
                f"{prefix}_vy_m_per_s",
                f"{prefix}_omega_rad_per_s",
            ]
        )
        if has_acceleration:
            headers.extend(
                [
                    f"{prefix}_ax_m_per_s2",
                    f"{prefix}_ay_m_per_s2",
                    f"{prefix}_alpha_rad_per_s2",
                ]
            )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for step, ti in enumerate(result.t):
            row: list[float] = [float(ti)]
            for index in range(mechanism.model.nb):
                start = 3 * index
                row.extend(float(value) for value in result.q[start : start + 3, step])
                row.extend(float(value) for value in result.v[start : start + 3, step])
                if has_acceleration:
                    row.extend(float(value) for value in result.a[start : start + 3, step])
            writer.writerow(row)
    return path


def point_trace_to_dict(
    mechanism: PlanarMechanism,
    result: KinematicResult | DynamicsResult,
    body: int | BodyHandle,
    point: Iterable[float] = (0.0, 0.0),
    *,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return position, velocity, and optional acceleration for a body-local point."""
    mechanism._validate_result(result)
    body_index = _body_index(mechanism.model, body)
    local_point = _finite_vector(point, "point", 2)
    positions: list[list[float]] = []
    velocities: list[list[float]] = []
    accelerations: list[list[float]] = []
    for step in range(len(result.t)):
        start = 3 * body_index
        qi = result.q[start : start + 3, step]
        vi = result.v[start : start + 3, step]
        positions.append(_vector(point_position(qi, local_point)))
        velocities.append(_vector(point_velocity(qi, vi, local_point)))
        if hasattr(result, "a"):
            ai = result.a[start : start + 3, step]
            accelerations.append(_vector(point_acceleration(qi, vi, ai, local_point)))

    payload: dict[str, Any] = {
        "schema": "mbsd.planar.point_trace",
        "schema_version": SCHEMA_VERSION,
        "mbsd_version": _package_version(),
        "dimension": 2,
        "units": _units(),
        "conventions": _conventions(),
        "metadata": _metadata(metadata),
        "point": {
            "name": name or f"{mechanism.model.bodies[body_index].name}_point",
            "body_index": body_index,
            "body_name": mechanism.model.bodies[body_index].name,
            "local_position": _vector(local_point),
        },
        "time": _vector(result.t),
        "position": {"x": [row[0] for row in positions], "y": [row[1] for row in positions]},
        "velocity": {"x": [row[0] for row in velocities], "y": [row[1] for row in velocities]},
    }
    if accelerations:
        payload["acceleration"] = {
            "x": [row[0] for row in accelerations],
            "y": [row[1] for row in accelerations],
        }
    return payload


def point_trace_to_json(
    mechanism: PlanarMechanism,
    result: KinematicResult | DynamicsResult,
    body: int | BodyHandle,
    point: Iterable[float],
    path: str | Path,
    *,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    indent: int = 2,
) -> Path:
    """Write a body-local point trace as JSON and return its path."""
    return _write_json(
        point_trace_to_dict(
            mechanism,
            result,
            body,
            point,
            name=name,
            metadata=metadata,
        ),
        path,
        indent=indent,
    )


def point_trace_to_csv(
    mechanism: PlanarMechanism,
    result: KinematicResult | DynamicsResult,
    body: int | BodyHandle,
    point: Iterable[float],
    path: str | Path,
    *,
    name: str | None = None,
) -> Path:
    """Write a body-local point trace as an SI-unit CSV and return its path."""
    payload = point_trace_to_dict(mechanism, result, body, point, name=name)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_acceleration = "acceleration" in payload
    headers = ["time_s", "x_m", "y_m", "vx_m_per_s", "vy_m_per_s"]
    if has_acceleration:
        headers.extend(["ax_m_per_s2", "ay_m_per_s2"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for step, ti in enumerate(payload["time"]):
            row = [
                ti,
                payload["position"]["x"][step],
                payload["position"]["y"][step],
                payload["velocity"]["x"][step],
                payload["velocity"]["y"][step],
            ]
            if has_acceleration:
                row.extend(
                    [
                        payload["acceleration"]["x"][step],
                        payload["acceleration"]["y"][step],
                    ]
                )
            writer.writerow(row)
    return path


def _as_mbody(mechanism: PlanarMechanism | MBody) -> MBody:
    return mechanism.model if isinstance(mechanism, PlanarMechanism) else mechanism


def _body_poses(
    mbody: MBody,
    result: KinematicResult | DynamicsResult,
) -> list[dict[str, Any]]:
    body_poses = []
    for index, body in enumerate(mbody.bodies):
        start = 3 * index
        body_poses.append(
            {
                "index": index,
                "name": body.name,
                "x": _vector(result.q[start, :]),
                "y": _vector(result.q[start + 1, :]),
                "theta": _vector(result.q[start + 2, :]),
                "vx": _vector(result.v[start, :]),
                "vy": _vector(result.v[start + 1, :]),
                "omega": _vector(result.v[start + 2, :]),
            }
        )
        if hasattr(result, "a"):
            body_poses[-1].update(
                {
                    "ax": _vector(result.a[start, :]),
                    "ay": _vector(result.a[start + 1, :]),
                    "alpha": _vector(result.a[start + 2, :]),
                }
            )
    return body_poses


def _user_constraint_to_dict(index: int, constraint: Any) -> dict[str, Any]:
    metadata = getattr(constraint, "metadata", None) or {}
    payload = {
        "index": index,
        "kind": metadata.get("kind", "custom"),
        "count": int(constraint.count),
    }
    payload.update(_json_ready(metadata))
    payload.setdefault("portable", False)
    if not payload["portable"]:
        payload["opaque"] = True
    return payload


def _spring_to_dict(mbody: MBody, index: int, spring: Spring) -> dict[str, Any]:
    if not isinstance(spring, Spring):
        raise TypeError("springs must contain Spring instances")
    body_i = _body_index(mbody, spring.i)
    body_j = _body_index(mbody, spring.j)
    stiffness = _finite_scalar(spring.k, "spring stiffness")
    damping = _finite_scalar(spring.c, "spring damping")
    if stiffness <= 0.0:
        raise ValueError("spring stiffness must be positive")
    if damping < 0.0:
        raise ValueError("spring damping must be non-negative")
    if spring.l0 is None:
        raise ValueError("spring natural length must be explicit for portable export")
    natural_length = _finite_scalar(spring.l0, "spring natural length")
    if natural_length < 0.0:
        raise ValueError("spring natural length must be non-negative")
    payload = {
        "index": index,
        "kind": "linear_spring_damper",
        "body_i": body_i,
        "body_j": body_j,
        "point_i": _vector(_finite_vector(spring.ri, "spring point_i", 2)),
        "point_j": _vector(_finite_vector(spring.rj, "spring point_j", 2)),
        "stiffness": stiffness,
        "damping": damping,
        "natural_length": natural_length,
    }
    name = getattr(spring, "name", None)
    if name:
        payload["name"] = name
    return payload


def _body_index(mbody: MBody, body: Any) -> int:
    value = body.index if hasattr(body, "index") else body
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("body must be an integer index or BodyHandle")
    index = int(value)
    if index < 0 or index >= mbody.nb:
        raise IndexError(f"body index {index} is out of range")
    return index


def _metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    return _json_ready(metadata)


def _write_json(payload: dict[str, Any], path: str | Path, *, indent: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _vector(values: Any) -> list[float]:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("export arrays must contain only finite values")
    return [float(value) for value in vector]


def _matrix(values: Any) -> list[list[float]]:
    matrix = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("export arrays must contain only finite values")
    return [[float(value) for value in row] for row in matrix]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _vector(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value
    raise TypeError(f"metadata value of type {type(value).__name__} is not JSON serializable")


def _finite_vector(values: Any, name: str, size: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _finite_scalar(value: Any, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _package_version() -> str:
    try:
        return version("mbsd")
    except PackageNotFoundError:
        return "0+local"


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    slug = "".join(chars).strip("_")
    return slug or "body"


def _result_type(result: Any) -> str:
    if isinstance(result, InverseDynamicsResult):
        return "inverse_dynamic"
    return "kinematic" if hasattr(result, "a") else "dynamic"


def joint_loads_to_dict(
    mechanism: PlanarMechanism,
    result: InverseDynamicsResult | DynamicsResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a versioned joint-load history: the FEM / sizing handoff payload."""
    loads = mechanism.joint_loads(result)
    bodies = mechanism.model.bodies
    joints = []
    for name, record in loads.joints.items():
        joints.append(
            {
                "name": name,
                "kind": record["kind"],
                "body_i": {"index": record["body_i"], "name": bodies[record["body_i"]].name},
                "body_j": {"index": record["body_j"], "name": bodies[record["body_j"]].name},
                "point": {"x": _vector(record["point"][0]), "y": _vector(record["point"][1])},
                "on_body_i": {
                    "fx": _vector(record["force_on_i"][0]),
                    "fy": _vector(record["force_on_i"][1]),
                    "moment": _vector(record["moment_on_i"]),
                },
                "on_body_j": {
                    "fx": _vector(record["force_on_j"][0]),
                    "fy": _vector(record["force_on_j"][1]),
                    "moment": _vector(record["moment_on_j"]),
                },
            }
        )
    drives = []
    for name, record in loads.drives.items():
        coordinate = record["coordinate"]
        drives.append(
            {
                "name": name,
                "body": {"index": record["body"], "name": bodies[record["body"]].name},
                "coordinate": coordinate,
                "effort_unit": "N*m" if coordinate == "theta" else "N",
                "effort": _vector(record["effort"]),
                "power": _vector(record["power"]),
            }
        )
    return {
        "schema": "mbsd.planar.joint_loads",
        "schema_version": JOINT_LOADS_SCHEMA_VERSION,
        "mbsd_version": _package_version(),
        "dimension": 2,
        "result_type": _result_type(result),
        "units": {**_units(), "power": "W"},
        "conventions": {
            **_conventions(),
            "force_frame": "world",
            "joint_point": "body_j_attachment_point_world",
            "moments": "about_joint_point_counterclockwise_positive",
            "frame_moment": "about_world_origin",
            "drive_effort": "generalized_force_applied_by_drive_to_coordinate",
            "drive_reaction": "actuators_react_against_ground",
        },
        "metadata": _metadata(metadata),
        "time": _vector(loads.t),
        "joints": joints,
        "drives": drives,
        "frame": {
            "fx": _vector(loads.frame["force"][0]),
            "fy": _vector(loads.frame["force"][1]),
            "moment": _vector(loads.frame["moment"]),
        },
        "diagnostics": {"max_action_reaction_residual": loads.max_action_reaction_residual},
    }


def joint_loads_to_json(
    mechanism: PlanarMechanism,
    result: InverseDynamicsResult | DynamicsResult,
    path: str | Path,
    *,
    indent: int = 2,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write the joint-load history payload as JSON and return its path."""
    return _write_json(
        joint_loads_to_dict(mechanism, result, metadata=metadata), path, indent=indent
    )


def mechanism_from_dict(payload: dict[str, Any]) -> PlanarMechanism:
    """Rebuild a planar mechanism from a ``mbsd.planar.mechanism`` v1 payload.

    Rejects opaque entries (Python-callable drives, callback loads, raw user
    constraints) with ``NotPortableError``.
    """
    from ..errors import MechanismSolveError, NotPortableError
    from .laws import law_from_dict
    from .loads import load_from_dict

    if payload.get("schema") != "mbsd.planar.mechanism":
        raise MechanismSolveError(
            f"Not a planar mechanism payload: schema={payload.get('schema')!r}.",
            code="schema_mismatch",
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise MechanismSolveError(
            f"Unsupported planar mechanism schema_version={payload.get('schema_version')!r}.",
            code="unsupported_schema_version",
            context={"readable": [SCHEMA_VERSION]},
        )
    mechanism = PlanarMechanism(gravity=payload["gravity"])
    bodies = sorted(payload["bodies"], key=lambda body: body["index"])
    for body in bodies:
        if body["index"] == 0:
            mechanism.ground(body["name"])
        else:
            mechanism.body(
                body["name"],
                mass=body["mass"],
                inertia=body["inertia"],
                center_of_mass=body["center_of_mass"],
            )
    joints = payload.get("joints", {})
    for joint in joints.get("revolute", []):
        mechanism.pin(
            joint["body_i"],
            joint["body_j"],
            point_a=joint["point_i"],
            point_b=joint["point_j"],
            name=joint.get("name"),
        )
    for joint in joints.get("prismatic", []):
        mechanism.slider(
            joint["rail_body"],
            joint["slider_body"],
            axis=joint["axis"],
            point_rail=joint["point_rail"],
            point_slider=joint["point_slider"],
            name=joint.get("name"),
        )
    for drive in payload.get("user_constraints", []):
        kind = drive.get("kind")
        name = drive.get("name")
        if kind == "motor":
            mechanism.motor(drive["body"], omega=drive["omega"], phase=drive["phase"], name=name)
        elif kind == "coordinate_drive" and "law" in drive:
            mechanism.coordinate_drive(
                drive["body"], drive["coordinate"], law=law_from_dict(drive["law"]), name=name
            )
        else:
            raise NotPortableError(
                f"User constraint {name or drive.get('index')!r} ({kind}) has no portable "
                "descriptor and cannot be imported.",
                context={"user_constraint": drive.get("index"), "kind": kind},
                remediation="Rebuild it with motor() or coordinate_drive(law=...).",
            )
    forces = payload.get("forces", {})
    for spring in forces.get("springs", []):
        mechanism.spring(
            spring["body_i"],
            spring["body_j"],
            stiffness=spring["stiffness"],
            natural_length=spring["natural_length"],
            damping=spring["damping"],
            point_a=spring["point_i"],
            point_b=spring["point_j"],
            name=spring.get("name"),
        )
    for load in forces.get("loads", []):
        descriptor = load_from_dict(load)
        mechanism._idx(descriptor.body)
        mechanism._claim_name(descriptor.name, "load")
        mechanism.loads.append(descriptor)
    return mechanism


def mechanism_from_json(path: str | Path) -> PlanarMechanism:
    """Read a planar mechanism JSON file written by ``to_json``."""
    return mechanism_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
