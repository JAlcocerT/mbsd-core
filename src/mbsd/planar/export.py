"""Portable export helpers for planar MBSD mechanisms and results."""

from __future__ import annotations

import csv
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .builder import DynamicsResult, KinematicResult, PlanarMechanism
from .model import MBody


SCHEMA_VERSION = 1


def mechanism_to_dict(mechanism: PlanarMechanism | MBody) -> dict[str, Any]:
    """Return a JSON-ready description of a planar mechanism."""
    mbody = _as_mbody(mechanism)
    return {
        "schema": "mbsd.planar.mechanism",
        "schema_version": SCHEMA_VERSION,
        "mbsd_version": _package_version(),
        "dimension": 2,
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
        "user_constraints": [
            _user_constraint_to_dict(index, constraint)
            for index, constraint in enumerate(mbody.user_constraints)
        ],
    }


def mechanism_to_json(
    mechanism: PlanarMechanism | MBody,
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """Write a planar mechanism export JSON file and return its path."""
    return _write_json(mechanism_to_dict(mechanism), path, indent=indent)


def result_to_dict(
    mechanism: PlanarMechanism,
    result: KinematicResult | DynamicsResult,
    *,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    """Return a JSON-ready trajectory/result export."""
    mechanism._validate_result(result)
    payload: dict[str, Any] = {
        "schema": "mbsd.planar.result",
        "schema_version": SCHEMA_VERSION,
        "mbsd_version": _package_version(),
        "result_type": "kinematic" if hasattr(result, "a") else "dynamic",
        "dimension": 2,
        "time": _vector(result.t),
        "coordinates": {
            "q": _matrix(result.q),
            "v": _matrix(result.v),
        },
        "body_poses": _body_poses(mechanism.model, result),
    }
    if hasattr(result, "a"):
        payload["coordinates"]["a"] = _matrix(result.a)
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
) -> Path:
    """Write a planar result export JSON file and return its path."""
    return _write_json(
        result_to_dict(mechanism, result, include_diagnostics=include_diagnostics),
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
    headers = ["time"]
    for index, body in enumerate(mechanism.model.bodies):
        prefix = f"body{index}_{_slug(body.name)}"
        headers.extend([f"{prefix}_x", f"{prefix}_y", f"{prefix}_theta"])
        headers.extend([f"{prefix}_vx", f"{prefix}_vy", f"{prefix}_omega"])
        if has_acceleration:
            headers.extend([f"{prefix}_ax", f"{prefix}_ay", f"{prefix}_alpha"])

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
    return payload


def _write_json(payload: dict[str, Any], path: str | Path, *, indent: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _vector(values: Any) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def _matrix(values: Any) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(values, dtype=float)]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _vector(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _package_version() -> str:
    try:
        return version("mbsd")
    except PackageNotFoundError:
        return "0+local"


def _slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value]
    slug = "".join(chars).strip("_")
    return slug or "body"
