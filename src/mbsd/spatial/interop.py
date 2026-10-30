"""Engine-neutral spatial transform handoff for reference adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


HANDOFF_SCHEMA = "mbsd.interop.transform_tracks"
HANDOFF_SCHEMA_VERSION = 1
_TARGETS = {
    "freecad": {"length_unit": "mm", "length_scale": 1000.0, "quaternion_order": "wxyz"},
    "blender": {"length_unit": "m", "length_scale": 1.0, "quaternion_order": "wxyz"},
    "browser": {"length_unit": "m", "length_scale": 1.0, "quaternion_order": "wxyz"},
    "generic": {"length_unit": "m", "length_scale": 1.0, "quaternion_order": "wxyz"},
}


def build_transform_handoff(
    result_payload: Mapping[str, Any], *, target: str = "generic",
    axis_matrix: Any | None = None, length_scale: float | None = None,
    geometry_references: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Convert a validated spatial result into portable transform tracks."""
    payload = _validated_spatial_result(result_payload)
    if target not in _TARGETS:
        raise ValueError(f"target must be one of: {', '.join(sorted(_TARGETS))}")
    target_contract = dict(_TARGETS[target])
    scale = target_contract["length_scale"] if length_scale is None else float(length_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("length_scale must be positive and finite")
    axes = np.eye(3) if axis_matrix is None else np.asarray(axis_matrix, dtype=float)
    if axes.shape != (3, 3) or not np.all(np.isfinite(axes)) or not np.allclose(axes @ axes.T, np.eye(3), atol=1e-12) or not np.isclose(np.linalg.det(axes), 1.0, atol=1e-12):
        raise ValueError("axis_matrix must be a finite proper orthogonal 3x3 matrix")
    times = np.asarray(payload["time"], dtype=float)
    bodies = payload["bodies"] if "bodies" in payload else [payload["body"]]
    geometry = {} if geometry_references is None else dict(geometry_references)
    tracks = []
    for index, body in enumerate(bodies):
        samples = []
        for time, pose in zip(times, body["poses"]):
            translation = scale * (axes @ np.asarray(pose["translation"], dtype=float))
            quaternion = pose["rotation"]
            source_q = np.asarray([quaternion[key] for key in ("w", "x", "y", "z")], dtype=float)
            target_q = _matrix_to_quaternion(axes @ _quaternion_matrix(source_q) @ axes.T)
            samples.append({
                "time_s": float(time), "translation": translation.tolist(),
                "rotation_wxyz": target_q.tolist(),
            })
        name = body["name"]
        body_id = body.get("id", f"body:{name}")
        track = {"id": body_id, "index": index, "name": name, "samples": samples}
        if body_id in geometry:
            track["geometry_reference"] = _geometry_reference(geometry.pop(body_id))
        tracks.append(track)
    if geometry:
        raise ValueError(
            "geometry_references contain unknown body IDs: "
            + ", ".join(sorted(geometry))
        )
    source = {
        "schema": payload["schema"], "schema_version": payload["schema_version"],
        "model_id": payload.get("model_id"), "excitation_id": payload.get("excitation_id"),
        "simulation_id": payload.get("simulation_id"),
        "solver_kind": payload.get("solver_kind", "spatial_kinematics"),
    }
    result = {
        "schema": HANDOFF_SCHEMA, "schema_version": HANDOFF_SCHEMA_VERSION,
        "source": source,
        "target": {
            "engine": target, "coordinate_handedness": "right_handed",
            "axis_matrix_source_to_target": axes.tolist(),
            "length_unit": target_contract["length_unit"] if length_scale is None else "caller_defined",
            "length_scale_from_m": scale, "time_unit": "s",
            "quaternion_order": target_contract["quaternion_order"],
            "rotation": "active_body_to_world",
        },
        "capabilities": {"transform_tracks": True, "geometry_loading": False, "force_import": False},
        "tracks": tracks,
    }
    result["artifact_id"] = _identity(result)
    return validate_transform_handoff(result)


def validate_transform_handoff(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or payload.get("schema") != HANDOFF_SCHEMA or payload.get("schema_version") != 1:
        raise ValueError("unsupported transform-handoff schema")
    source, target = payload.get("source"), payload.get("target")
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise ValueError("handoff source and target must be mappings")
    if not isinstance(source.get("schema"), str) or not isinstance(source.get("schema_version"), int):
        raise ValueError("handoff source identity is invalid")
    expected_capabilities = {
        "transform_tracks": True,
        "geometry_loading": False,
        "force_import": False,
    }
    if payload.get("capabilities") != expected_capabilities:
        raise ValueError("handoff capabilities are invalid or unsupported")
    if target.get("coordinate_handedness") != "right_handed" or target.get("quaternion_order") != "wxyz" or target.get("rotation") != "active_body_to_world":
        raise ValueError("handoff target conventions are invalid")
    axes = np.asarray(target.get("axis_matrix_source_to_target"), dtype=float)
    if axes.shape != (3, 3) or not np.allclose(axes @ axes.T, np.eye(3), atol=1e-12) or not np.isclose(np.linalg.det(axes), 1.0, atol=1e-12):
        raise ValueError("handoff target axis matrix is invalid")
    scale = target.get("length_scale_from_m")
    if not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("handoff target length scale is invalid")
    tracks = payload.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("handoff tracks must be a non-empty list")
    ids = set()
    reference_times = None
    for index, track in enumerate(tracks):
        if not isinstance(track, Mapping) or track.get("index") != index or not isinstance(track.get("id"), str) or not track["id"] or track["id"] in ids:
            raise ValueError("handoff track identity is invalid")
        ids.add(track["id"])
        if not isinstance(track.get("name"), str) or not track["name"]:
            raise ValueError("handoff track name is invalid")
        if "geometry_reference" in track:
            _geometry_reference(track["geometry_reference"])
        samples = track.get("samples")
        if not isinstance(samples, list) or len(samples) < 2:
            raise ValueError("handoff track requires at least two samples")
        times = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise ValueError("handoff sample must be a mapping")
            time = sample.get("time_s")
            translation = np.asarray(sample.get("translation"), dtype=float)
            quaternion = np.asarray(sample.get("rotation_wxyz"), dtype=float)
            if not isinstance(time, (int, float)) or not np.isfinite(time) or translation.shape != (3,) or not np.all(np.isfinite(translation)) or quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)) or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-10):
                raise ValueError("handoff sample is invalid")
            times.append(float(time))
        if np.any(np.diff(times) <= 0):
            raise ValueError("handoff sample times must be strictly increasing")
        if reference_times is None:
            reference_times = times
        elif times != reference_times:
            raise ValueError("handoff tracks must share one time grid")
    detached = json.loads(json.dumps(payload, allow_nan=False))
    artifact_id = detached.pop("artifact_id", None)
    if artifact_id != _identity(detached):
        raise ValueError("handoff artifact_id does not match its content")
    detached["artifact_id"] = artifact_id
    return detached


def transform_handoff_to_json(payload: Mapping[str, Any], path: str | Path, *, indent: int = 2) -> Path:
    validated = validate_transform_handoff(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, indent=indent, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _validated_spatial_result(payload):
    if not isinstance(payload, Mapping):
        raise TypeError("result_payload must be a mapping")
    schema = payload.get("schema")
    if schema == "mbsd.spatial.constrained_rigid_body_dynamic_result":
        from .constrained import validate_constrained_result_payload
        return validate_constrained_result_payload(payload)
    if schema == "mbsd.spatial.kinematic_result":
        from .builder import validate_spatial_kinematic_result_payload
        return validate_spatial_kinematic_result_payload(payload)
    if schema == "mbsd.spatial.coupled_rigid_body_dynamic_result":
        from .coupled import validate_coupled_result_payload
        return validate_coupled_result_payload(payload)
    if schema == "mbsd.spatial.free_rigid_body_dynamic_result":
        from .rotation import validate_rigid_body_result_payload
        return validate_rigid_body_result_payload(payload)
    raise ValueError("result schema is not supported by the transform handoff")


def _quaternion_matrix(value):
    w, x, y, z = value / np.linalg.norm(value)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _geometry_reference(value):
    if not isinstance(value, Mapping):
        raise ValueError("geometry reference must be a mapping")
    allowed = {"uri", "object", "media_type", "sha256"}
    if not set(value) <= allowed or not isinstance(value.get("uri"), str) or not value["uri"]:
        raise ValueError("geometry reference requires a non-empty uri and known fields")
    result = dict(value)
    for name in set(result) - {"uri"}:
        if not isinstance(result[name], str) or not result[name]:
            raise ValueError(f"geometry reference {name} must be a non-empty string")
    if "sha256" in result and (
        len(result["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in result["sha256"].lower())
    ):
        raise ValueError("geometry reference sha256 must contain 64 hexadecimal characters")
    return result


def _matrix_to_quaternion(matrix):
    trace = float(np.trace(matrix))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        values = np.array([0.25*s, (matrix[2,1]-matrix[1,2])/s, (matrix[0,2]-matrix[2,0])/s, (matrix[1,0]-matrix[0,1])/s])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = np.sqrt(1.0 + matrix[0,0] - matrix[1,1] - matrix[2,2]) * 2.0
            values = np.array([(matrix[2,1]-matrix[1,2])/s, 0.25*s, (matrix[0,1]+matrix[1,0])/s, (matrix[0,2]+matrix[2,0])/s])
        elif index == 1:
            s = np.sqrt(1.0 + matrix[1,1] - matrix[0,0] - matrix[2,2]) * 2.0
            values = np.array([(matrix[0,2]-matrix[2,0])/s, (matrix[0,1]+matrix[1,0])/s, 0.25*s, (matrix[1,2]+matrix[2,1])/s])
        else:
            s = np.sqrt(1.0 + matrix[2,2] - matrix[0,0] - matrix[1,1]) * 2.0
            values = np.array([(matrix[1,0]-matrix[0,1])/s, (matrix[0,2]+matrix[2,0])/s, (matrix[1,2]+matrix[2,1])/s, 0.25*s])
    values /= np.linalg.norm(values)
    if values[0] < 0:
        values = -values
    return values


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
