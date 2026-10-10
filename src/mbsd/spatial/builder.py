"""Experimental spatial mechanism builder and constrained kinematic solver."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from scipy.optimize import least_squares

from ..errors import MechanismSolveError
from .kinematics import joint_residual_jacobian
from .vocabulary import (
    FixedJoint3D,
    Frame3D,
    Pose3D,
    Quaternion,
    SpatialBody,
    SpatialModel,
    SphericalJoint3D,
)


ArrayLike3 = Iterable[float]
RESULT_SCHEMA = "mbsd.spatial.kinematic_result"
RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SpatialBodyHandle:
    """Stable public reference to a body owned by a spatial mechanism."""

    index: int
    name: str

    def __int__(self) -> int:
        return self.index


@dataclass(frozen=True)
class SpatialFrameHandle:
    """Stable public reference to a named spatial frame."""

    index: int
    name: str

    def __int__(self) -> int:
        return self.index


@dataclass(frozen=True)
class SpatialCoordinateDrive:
    """Prescribed world translation coordinate and its time derivative."""

    body: int
    coordinate: str
    value: Callable[[float], float]
    velocity: Callable[[float], float]

    def residual(self, poses: tuple[Pose3D, ...], t: float) -> float:
        axis = {"x": 0, "y": 1, "z": 2}[self.coordinate]
        return float(poses[self.body].translation[axis] - self.value(t))

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "coordinate_drive",
            "body": self.body,
            "coordinate": self.coordinate,
            "value": getattr(self.value, "__name__", "callable"),
            "velocity": getattr(self.velocity, "__name__", "callable"),
        }


@dataclass(frozen=True)
class SpatialKinematicResult:
    """Sampled poses and world-expressed spatial velocities."""

    t: np.ndarray
    poses: tuple[tuple[Pose3D, ...], ...]
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    residuals: np.ndarray
    success: bool
    message: str
    model_id: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Mapping[str, Any] = field(default_factory=dict)

    def body_poses(self, body: int | SpatialBodyHandle) -> tuple[Pose3D, ...]:
        index = body.index if isinstance(body, SpatialBodyHandle) else int(body)
        if index < 0 or not self.poses or index >= len(self.poses[0]):
            raise IndexError(f"body index {index} is out of range")
        return tuple(step[index] for step in self.poses)


@dataclass(frozen=True)
class SpatialModelDiagnostics:
    """Constraint count, rank, and degree-of-freedom summary."""

    coordinates: int
    constraints: int
    jacobian_rank: int
    degrees_of_freedom: int
    classification: str
    rank_deficient: bool
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "coordinates": self.coordinates,
            "constraints": self.constraints,
            "jacobian_rank": self.jacobian_rank,
            "degrees_of_freedom": self.degrees_of_freedom,
            "classification": self.classification,
            "rank_deficient": self.rank_deficient,
            "finite": self.finite,
        }


@dataclass(frozen=True)
class SpatialResultDiagnostics:
    """Residual and status summary for a spatial kinematic result."""

    steps: int
    max_position_residual: float
    max_velocity_residual: float
    finite: bool
    success: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "max_position_residual": self.max_position_residual,
            "max_velocity_residual": self.max_velocity_residual,
            "finite": self.finite,
            "success": self.success,
        }


class SpatialMechanism:
    """Experimental builder for small constrained spatial kinematic models."""

    def __init__(self) -> None:
        self._bodies: list[SpatialBody] = []
        self._frames: list[Frame3D] = []
        self._joints: list[SphericalJoint3D | FixedJoint3D] = []
        self._drives: list[SpatialCoordinateDrive] = []
        self._ground: int | None = None

    @property
    def nbody(self) -> int:
        return len(self._bodies)

    @property
    def ncoord(self) -> int:
        return 6 * self.nbody

    @property
    def capabilities(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "position_solve": True,
                "velocity_solve": True,
                "spatial_dynamics": False,
                "joint_types": ("spherical", "fixed"),
                "velocity_frame": "world",
            }
        )

    def ground(self, name: str = "ground") -> SpatialBodyHandle:
        if self._ground is None:
            handle = self.body(name, mass=1.0, inertia=(1.0, 1.0, 1.0))
            self._ground = handle.index
        return SpatialBodyHandle(self._ground, self._bodies[self._ground].name)

    def body(
        self,
        name: str,
        *,
        mass: float = 1.0,
        inertia: ArrayLike3 = (1.0, 1.0, 1.0),
        center_of_mass: ArrayLike3 = (0.0, 0.0, 0.0),
        pose: Pose3D | None = None,
    ) -> SpatialBodyHandle:
        body = SpatialBody(
            name=name,
            mass=mass,
            inertia=np.asarray(inertia, dtype=float),
            center_of_mass=np.asarray(center_of_mass, dtype=float),
            pose=Pose3D.identity() if pose is None else pose,
        )
        if any(existing.name == name for existing in self._bodies):
            raise ValueError(f"body name {name!r} is already in use")
        self._bodies.append(body)
        return SpatialBodyHandle(len(self._bodies) - 1, name)

    def frame(
        self,
        name: str,
        *,
        parent_body: int | SpatialBodyHandle | None = None,
        pose: Pose3D | None = None,
    ) -> SpatialFrameHandle:
        if any(existing.name == name for existing in self._frames):
            raise ValueError(f"frame name {name!r} is already in use")
        parent = self._idx(parent_body) if parent_body is not None else None
        self._frames.append(Frame3D(name, Pose3D.identity() if pose is None else pose, parent))
        return SpatialFrameHandle(len(self._frames) - 1, name)

    def spherical(
        self,
        body_i: int | SpatialBodyHandle | None,
        body_j: int | SpatialBodyHandle | None,
        *,
        point_i: ArrayLike3 = (0.0, 0.0, 0.0),
        point_j: ArrayLike3 = (0.0, 0.0, 0.0),
        name: str | None = None,
    ) -> SphericalJoint3D:
        joint = SphericalJoint3D(
            name or f"spherical-{len(self._joints)}",
            self._idx_optional(body_i),
            self._idx_optional(body_j),
            point_i,
            point_j,
        )
        self._add_joint(joint)
        return joint

    def fixed(
        self,
        body_i: int | SpatialBodyHandle | None,
        body_j: int | SpatialBodyHandle | None,
        *,
        frame_i: Pose3D | None = None,
        frame_j: Pose3D | None = None,
        name: str | None = None,
    ) -> FixedJoint3D:
        joint = FixedJoint3D(
            name or f"fixed-{len(self._joints)}",
            self._idx_optional(body_i),
            self._idx_optional(body_j),
            Pose3D.identity() if frame_i is None else frame_i,
            Pose3D.identity() if frame_j is None else frame_j,
        )
        self._add_joint(joint)
        return joint

    def coordinate_drive(
        self,
        body: int | SpatialBodyHandle,
        coordinate: str,
        *,
        value: Callable[[float], float],
        velocity: Callable[[float], float],
    ) -> SpatialCoordinateDrive:
        if coordinate not in {"x", "y", "z"}:
            raise ValueError("coordinate must be one of: x, y, z")
        if not callable(value) or not callable(velocity):
            raise TypeError("value and velocity must be callable")
        drive = SpatialCoordinateDrive(self._idx(body), coordinate, value, velocity)
        self._drives.append(drive)
        return drive

    def solve_position(
        self,
        poses: Iterable[Pose3D] | None = None,
        t: float = 0.0,
        *,
        tol: float = 1e-9,
        max_nfev: int = 1000,
    ) -> tuple[Pose3D, ...]:
        self._ensure_solvable()
        poses = self._initial_poses() if poses is None else self._validate_poses(poses)
        t = _finite_scalar(t, "t")
        tol = _positive_scalar(tol, "tol")

        def residual(increments: np.ndarray) -> np.ndarray:
            return self.constraint_residual(self._apply_increments(poses, increments), t)

        solver_tol = max(np.finfo(float).eps * 32.0, tol * 0.01)
        solution = least_squares(
            residual,
            np.zeros(self.ncoord),
            xtol=solver_tol,
            ftol=solver_tol,
            gtol=solver_tol,
            max_nfev=max_nfev,
        )
        solved = self._apply_increments(poses, solution.x)
        residual_norm = _infinity_norm(self.constraint_residual(solved, t))
        if not solution.success or residual_norm > tol:
            raise MechanismSolveError(
                "Spatial position solve failed: "
                f"{solution.message}; residual {residual_norm:.3e} exceeds {tol:.3e}."
            )
        return solved

    def solve_velocity(
        self,
        poses: Iterable[Pose3D],
        t: float = 0.0,
        *,
        tol: float = 1e-9,
    ) -> np.ndarray:
        poses = self._validate_poses(poses)
        t = _finite_scalar(t, "t")
        tol = _positive_scalar(tol, "tol")
        jacobian = self.constraint_jacobian(poses, t)
        time_derivative = self.constraint_time_derivative(poses, t)
        velocity, *_ = np.linalg.lstsq(jacobian, -time_derivative, rcond=None)
        residual_norm = _infinity_norm(jacobian @ velocity + time_derivative)
        if residual_norm > tol:
            raise MechanismSolveError(
                "Spatial velocity solve failed: velocity-level residual "
                f"{residual_norm:.3e} exceeds {tol:.3e}."
            )
        return velocity.reshape(self.nbody, 6)

    def solve_kinematics(
        self,
        t: np.ndarray,
        poses: Iterable[Pose3D] | None = None,
        *,
        tol: float = 1e-9,
    ) -> SpatialKinematicResult:
        times = _time_array(t)
        current = self._initial_poses() if poses is None else self._validate_poses(poses)
        pose_history: list[tuple[Pose3D, ...]] = []
        velocities = np.zeros((len(times), self.nbody, 6))
        residuals: list[np.ndarray] = []
        for index, time in enumerate(times):
            current = self.solve_position(current, float(time), tol=tol)
            pose_history.append(current)
            velocities[index] = self.solve_velocity(current, float(time), tol=tol)
            residuals.append(self.constraint_residual(current, float(time)))
        residual_matrix = np.column_stack(residuals)
        provenance = MappingProxyType(
            {
                "mbsd_version": _package_version(),
                "solver": "scipy.optimize.least_squares",
                "velocity_solver": "numpy.linalg.lstsq",
                "tolerance": tol,
                "pose_increment_frame": "world",
            }
        )
        return SpatialKinematicResult(
            t=np.array(times, copy=True),
            poses=tuple(pose_history),
            linear_velocity=np.array(velocities[:, :, :3], copy=True),
            angular_velocity=np.array(velocities[:, :, 3:], copy=True),
            residuals=residual_matrix,
            success=True,
            message="converged",
            model_id=self.model_id(),
            provenance=provenance,
            capabilities=self.capabilities,
        )

    def constraint_residual(
        self, poses: Iterable[Pose3D], t: float = 0.0
    ) -> np.ndarray:
        poses = self._validate_poses(poses)
        values: list[float] = []
        if self._ground is not None:
            ground = poses[self._ground]
            values.extend(ground.translation)
            values.extend(_rotation_vector(ground.rotation))
        for joint in self._joints:
            values.extend(joint.residual(poses))
        for drive in self._drives:
            values.append(drive.residual(poses, t))
        return np.asarray(values, dtype=float)

    def constraint_jacobian(
        self, poses: Iterable[Pose3D], t: float = 0.0
    ) -> np.ndarray:
        poses = self._validate_poses(poses)
        rows: list[np.ndarray] = []
        if self._ground is not None:
            block = np.zeros((6, self.ncoord))
            start = 6 * self._ground
            block[:, start : start + 6] = np.eye(6)
            rows.append(block)
        rows.extend(joint_residual_jacobian(joint, poses) for joint in self._joints)
        for drive in self._drives:
            row = np.zeros((1, self.ncoord))
            axis = {"x": 0, "y": 1, "z": 2}[drive.coordinate]
            row[0, 6 * drive.body + axis] = 1.0
            rows.append(row)
        return np.vstack(rows) if rows else np.zeros((0, self.ncoord))

    def constraint_time_derivative(
        self, poses: Iterable[Pose3D], t: float = 0.0
    ) -> np.ndarray:
        poses = self._validate_poses(poses)
        values = np.zeros(len(self.constraint_residual(poses, t)))
        offset = (6 if self._ground is not None else 0) + sum(
            len(joint.residual(poses)) for joint in self._joints
        )
        for index, drive in enumerate(self._drives):
            values[offset + index] = -_finite_scalar(drive.velocity(t), "drive velocity")
        return values

    def model_diagnostics(
        self, poses: Iterable[Pose3D] | None = None, t: float = 0.0
    ) -> SpatialModelDiagnostics:
        self._ensure_solvable()
        poses = self._initial_poses() if poses is None else self._validate_poses(poses)
        residual = self.constraint_residual(poses, t)
        jacobian = self.constraint_jacobian(poses, t)
        rank = int(np.linalg.matrix_rank(jacobian)) if jacobian.size else 0
        constraints = len(residual)
        rank_deficient = rank < min(jacobian.shape) if jacobian.size else False
        if constraints > self.ncoord:
            classification = "overconstrained"
        elif rank_deficient:
            classification = "rank_deficient"
        elif constraints < self.ncoord:
            classification = "underconstrained"
        else:
            classification = "fully_constrained"
        return SpatialModelDiagnostics(
            coordinates=self.ncoord,
            constraints=constraints,
            jacobian_rank=rank,
            degrees_of_freedom=self.ncoord - rank,
            classification=classification,
            rank_deficient=rank_deficient,
            finite=bool(np.all(np.isfinite(residual)) and np.all(np.isfinite(jacobian))),
        )

    def result_diagnostics(self, result: SpatialKinematicResult) -> SpatialResultDiagnostics:
        self._validate_result(result)
        max_velocity = 0.0
        finite = True
        for index, poses in enumerate(result.poses):
            velocity = np.concatenate(
                (result.linear_velocity[index], result.angular_velocity[index]), axis=1
            ).reshape(-1)
            velocity_residual = (
                self.constraint_jacobian(poses, float(result.t[index])) @ velocity
                + self.constraint_time_derivative(poses, float(result.t[index]))
            )
            max_velocity = max(max_velocity, _infinity_norm(velocity_residual))
            finite = finite and bool(np.all(np.isfinite(velocity_residual)))
        return SpatialResultDiagnostics(
            steps=len(result.t),
            max_position_residual=_infinity_norm(result.residuals),
            max_velocity_residual=max_velocity,
            finite=finite
            and bool(np.all(np.isfinite(result.linear_velocity)))
            and bool(np.all(np.isfinite(result.angular_velocity))),
            success=result.success,
        )

    def to_spatial_model(self) -> SpatialModel:
        return SpatialModel(
            bodies=tuple(self._bodies),
            frames=tuple(self._frames),
            joints=tuple(self._joints),
            metadata={
                "builder": "Mechanism.spatial",
                "ground_body": self._ground,
                "drives": [drive.as_dict() for drive in self._drives],
            },
        )

    @classmethod
    def from_spatial_model(cls, model: SpatialModel) -> "SpatialMechanism":
        if not isinstance(model, SpatialModel):
            raise TypeError("model must be a SpatialModel")
        if model.metadata.get("drives"):
            raise ValueError(
                "portable coordinate-drive descriptors cannot reconstruct Python callbacks"
            )
        mechanism = cls()
        mechanism._bodies = list(model.bodies)
        mechanism._frames = list(model.frames)
        mechanism._joints = list(model.joints)
        ground = model.metadata.get("ground_body")
        mechanism._ground = None if ground is None else int(ground)
        return mechanism

    def model_id(self) -> str:
        payload = self.to_spatial_model().as_dict()
        payload.pop("mbsd_version", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_spatial_model().as_dict()
        payload["model_id"] = self.model_id()
        payload["capabilities"] = _json_ready(self.capabilities)
        return payload

    def result_to_dict(self, result: SpatialKinematicResult) -> dict[str, Any]:
        self._validate_result(result)
        diagnostics = self.result_diagnostics(result)
        return {
            "schema": RESULT_SCHEMA,
            "schema_version": RESULT_SCHEMA_VERSION,
            "mbsd_version": _package_version(),
            "status": "experimental",
            "dimension": 3,
            "model_id": result.model_id,
            "units": {
                "length": "m",
                "angle": "rad",
                "time": "s",
                "linear_velocity": "m/s",
                "angular_velocity": "rad/s",
            },
            "conventions": {
                "world_frame": "right_handed_xyz",
                "rotation": "active_body_to_world",
                "quaternion_order": ["w", "x", "y", "z"],
                "velocity_frame": "world",
                "array_layout": "time_body_component",
            },
            "capabilities": _json_ready(result.capabilities),
            "solver_status": {"success": result.success, "message": result.message},
            "provenance": _json_ready(result.provenance),
            "diagnostics": diagnostics.as_dict(),
            "time": _vector(result.t),
            "residuals": np.asarray(result.residuals, dtype=float).tolist(),
            "bodies": [
                {
                    "index": body_index,
                    "name": body.name,
                    "poses": [step[body_index].as_dict() for step in result.poses],
                    "linear_velocity": result.linear_velocity[:, body_index, :].tolist(),
                    "angular_velocity": result.angular_velocity[:, body_index, :].tolist(),
                }
                for body_index, body in enumerate(self._bodies)
            ],
        }

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        return _write_json(self.to_dict(), path, indent)

    def result_to_json(
        self, result: SpatialKinematicResult, path: str | Path, *, indent: int = 2
    ) -> Path:
        return _write_json(self.result_to_dict(result), path, indent)

    def _add_joint(self, joint: SphericalJoint3D | FixedJoint3D) -> None:
        if any(existing.name == joint.name for existing in self._joints):
            raise ValueError(f"joint name {joint.name!r} is already in use")
        self._joints.append(joint)

    def _idx(self, body: int | SpatialBodyHandle) -> int:
        index = body.index if isinstance(body, SpatialBodyHandle) else int(body)
        if index < 0 or index >= self.nbody:
            raise IndexError(f"body index {index} is out of range")
        return index

    def _idx_optional(self, body: int | SpatialBodyHandle | None) -> int | None:
        return None if body is None else self._idx(body)

    def _initial_poses(self) -> tuple[Pose3D, ...]:
        return tuple(body.pose for body in self._bodies)

    def _validate_poses(self, poses: Iterable[Pose3D]) -> tuple[Pose3D, ...]:
        values = tuple(poses)
        if len(values) != self.nbody:
            raise ValueError(f"poses must contain {self.nbody} Pose3D values")
        for index, pose in enumerate(values):
            if not isinstance(pose, Pose3D):
                raise TypeError(f"poses[{index}] must be a Pose3D")
        return values

    def _apply_increments(
        self, poses: tuple[Pose3D, ...], increments: np.ndarray
    ) -> tuple[Pose3D, ...]:
        increments = np.asarray(increments, dtype=float)
        if increments.shape != (self.ncoord,) or not np.all(np.isfinite(increments)):
            raise ValueError(f"increments must be a finite vector with shape ({self.ncoord},)")
        updated = []
        for index, pose in enumerate(poses):
            delta = increments[6 * index : 6 * index + 6]
            rotation = _rotation_from_vector(delta[3:]).compose(pose.rotation)
            updated.append(Pose3D(pose.translation + delta[:3], rotation))
        return tuple(updated)

    def _ensure_solvable(self) -> None:
        if not self._bodies:
            raise ValueError("spatial mechanism must contain at least one body")
        if len(self.constraint_residual(self._initial_poses())) == 0:
            raise MechanismSolveError("spatial mechanism has no constraints")

    def _validate_result(self, result: SpatialKinematicResult) -> None:
        if not isinstance(result, SpatialKinematicResult):
            raise TypeError("result must be a SpatialKinematicResult")
        if result.model_id != self.model_id():
            raise ValueError("result model_id does not match this mechanism")
        expected = (len(result.t), self.nbody, 3)
        if result.linear_velocity.shape != expected or result.angular_velocity.shape != expected:
            raise ValueError(f"result velocity arrays must have shape {expected}")


def validate_spatial_kinematic_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach a spatial kinematic-result interchange payload."""
    if not isinstance(payload, Mapping):
        raise TypeError("spatial kinematic result payload must be a mapping")
    if payload.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"schema must be {RESULT_SCHEMA!r}")
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported spatial kinematic result schema_version "
            f"{payload.get('schema_version')!r}; supported version is {RESULT_SCHEMA_VERSION}"
        )
    for field_name in (
        "model_id",
        "units",
        "conventions",
        "capabilities",
        "solver_status",
        "provenance",
        "diagnostics",
        "time",
        "residuals",
        "bodies",
    ):
        if field_name not in payload:
            raise ValueError(f"spatial kinematic result missing {field_name!r}")
    if not isinstance(payload["model_id"], str) or not payload["model_id"]:
        raise ValueError("model_id must be a non-empty string")
    for field_name in (
        "units",
        "conventions",
        "capabilities",
        "solver_status",
        "provenance",
        "diagnostics",
    ):
        if not isinstance(payload[field_name], Mapping):
            raise ValueError(f"{field_name} must be a mapping")
    if not isinstance(payload["solver_status"].get("success"), bool):
        raise ValueError("solver_status.success must be a boolean")
    times = np.asarray(payload["time"], dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("time must be a non-empty finite one-dimensional array")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("time must be strictly increasing")
    residuals = np.asarray(payload["residuals"], dtype=float)
    if residuals.ndim != 2 or residuals.shape[1] != len(times):
        raise ValueError("residuals must use constraint_by_time layout")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals must contain only finite values")
    bodies = payload["bodies"]
    if not isinstance(bodies, list) or not bodies:
        raise ValueError("bodies must be a non-empty list")
    names: list[str] = []
    for index, body in enumerate(bodies):
        if not isinstance(body, Mapping):
            raise ValueError(f"bodies[{index}] must be a mapping")
        if body.get("index") != index or not isinstance(body.get("name"), str):
            raise ValueError(f"bodies[{index}] has invalid identity fields")
        names.append(body["name"])
        if len(body.get("poses", [])) != len(times):
            raise ValueError(f"bodies[{index}].poses must match time length")
        for velocity_name in ("linear_velocity", "angular_velocity"):
            velocity = np.asarray(body.get(velocity_name), dtype=float)
            if velocity.shape != (len(times), 3) or not np.all(np.isfinite(velocity)):
                raise ValueError(
                    f"bodies[{index}].{velocity_name} must have shape ({len(times)}, 3)"
                )
    if len(names) != len(set(names)):
        raise ValueError("body names must be unique")
    return json.loads(json.dumps(payload))


def load_spatial_kinematic_result_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return validate_spatial_kinematic_result_payload(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _rotation_from_vector(vector: np.ndarray) -> Quaternion:
    angle = float(np.linalg.norm(vector))
    return Quaternion.identity() if angle == 0.0 else Quaternion.from_axis_angle(vector, angle)


def _rotation_vector(rotation: Quaternion) -> np.ndarray:
    quaternion = rotation.normalized().as_array()
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm == 0.0:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vector_norm, quaternion[0])
    return quaternion[1:] * (angle / vector_norm)


def _finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_scalar(value: float, name: str) -> float:
    value = _finite_scalar(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _time_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("t must be a non-empty finite one-dimensional array")
    if len(array) > 1 and np.any(np.diff(array) <= 0.0):
        raise ValueError("t must be strictly increasing")
    return array


def _infinity_norm(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=float)
    return float(np.linalg.norm(array, ord=np.inf)) if array.size else 0.0


def _vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _write_json(payload: dict[str, Any], path: str | Path, indent: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _package_version() -> str:
    try:
        return version("mbsd")
    except PackageNotFoundError:
        return "0+local"
