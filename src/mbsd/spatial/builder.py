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
from ..errors import MechanismSolveError
from .kinematics import joint_residual_jacobian
from .dynamics import (
    SpatialDynamicsResult,
    SpatialForce,
    dynamic_result_to_dict,
    dynamic_result_to_json,
    dynamics_diagnostics,
    simulate_fixed_orientation,
)
from .rotation import (
    RigidBodyLoad,
    SpatialRigidBodyDynamicsResult,
    rigid_body_diagnostics,
    rigid_body_result_to_dict,
    rigid_body_result_to_json,
    simulate_free_body,
)
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
DRIVE_DERIVATIVE_ATOL = 1e-7
DRIVE_DERIVATIVE_RTOL = 1e-5


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
    """Prescribed world translation coordinate and its time derivatives.

    ``acceleration`` is optional for kinematics.  Fixed-orientation dynamics
    uses it when supplied and otherwise derives it adaptively from ``velocity``.
    """

    body: int
    coordinate: str
    value: Callable[[float], float]
    velocity: Callable[[float], float]
    acceleration: Callable[[float], float] | None = None
    name: str = "coordinate-drive"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("coordinate drive name must be non-empty")

    def residual(self, poses: tuple[Pose3D, ...], t: float) -> float:
        axis = {"x": 0, "y": 1, "z": 2}[self.coordinate]
        return float(poses[self.body].translation[axis] - self.value(t))

    def validate_derivative(self, t: float) -> str:
        """Validate velocity with a converged, scale-aware derivative estimate."""
        t = _finite_scalar(t, "t")
        analytic = _finite_scalar(self.velocity(t), "drive velocity")
        numeric, uncertainty, converged = _adaptive_derivative(
            self.value, t, "drive value"
        )
        tolerance = DRIVE_DERIVATIVE_ATOL + DRIVE_DERIVATIVE_RTOL * max(
            abs(numeric), abs(analytic)
        )
        difference = abs(numeric - analytic)
        if converged and difference > max(tolerance, 8.0 * uncertainty):
            raise ValueError(
                f"coordinate drive {self.coordinate!r} velocity is inconsistent with "
                f"the derivative of its value law at t={t:g}: "
                f"reported {analytic:.6g}, numerical derivative {numeric:.6g}, "
                f"tolerance {tolerance:.3e}"
            )
        return "validated" if converged else "inconclusive"

    def acceleration_value(self, t: float) -> float:
        """Return supplied acceleration or an adaptive derivative of velocity."""
        t = _finite_scalar(t, "t")
        if self.acceleration is None:
            estimate, _, _ = _adaptive_derivative(
                self.velocity, t, "drive velocity"
            )
            return estimate
        analytic = _finite_scalar(self.acceleration(t), "drive acceleration")
        numeric, uncertainty, converged = _adaptive_derivative(
            self.velocity, t, "drive velocity"
        )
        tolerance = DRIVE_DERIVATIVE_ATOL + DRIVE_DERIVATIVE_RTOL * max(
            abs(numeric), abs(analytic)
        )
        if converged and abs(numeric - analytic) > max(tolerance, 8.0 * uncertainty):
            raise ValueError(
                f"coordinate drive {self.coordinate!r} acceleration is inconsistent with "
                f"the derivative of its velocity law at t={t:g}: reported "
                f"{analytic:.6g}, numerical derivative {numeric:.6g}, "
                f"tolerance {tolerance:.3e}"
            )
        return analytic

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "coordinate_drive",
            "body": self.body,
            "coordinate": self.coordinate,
            "value": getattr(self.value, "__name__", "callable"),
            "velocity": getattr(self.velocity, "__name__", "callable"),
            "acceleration": (
                None
                if self.acceleration is None
                else getattr(self.acceleration, "__name__", "callable")
            ),
            "name": self.name,
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
    excitation_id: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _readonly_array(self.t))
        object.__setattr__(self, "linear_velocity", _readonly_array(self.linear_velocity))
        object.__setattr__(self, "angular_velocity", _readonly_array(self.angular_velocity))
        object.__setattr__(self, "residuals", _readonly_array(self.residuals))

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
    max_translation_residual: float
    max_rotation_residual: float
    max_linear_velocity_residual: float
    max_angular_velocity_residual: float
    max_trajectory_linear_velocity_error: float
    max_trajectory_angular_velocity_error: float
    trajectory_velocity_checked: bool
    finite: bool
    success: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "max_position_residual": self.max_position_residual,
            "max_velocity_residual": self.max_velocity_residual,
            "max_translation_residual_m": self.max_translation_residual,
            "max_rotation_residual_rad": self.max_rotation_residual,
            "max_linear_velocity_residual_m_per_s": self.max_linear_velocity_residual,
            "max_angular_velocity_residual_rad_per_s": self.max_angular_velocity_residual,
            "max_trajectory_linear_velocity_error": (
                self.max_trajectory_linear_velocity_error
            ),
            "max_trajectory_angular_velocity_error": (
                self.max_trajectory_angular_velocity_error
            ),
            "trajectory_velocity_checked": self.trajectory_velocity_checked,
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
                "fixed_orientation_translational_dynamics": True,
                "free_body_rotational_dynamics": True,
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
        acceleration: Callable[[float], float] | None = None,
        name: str | None = None,
    ) -> SpatialCoordinateDrive:
        if coordinate not in {"x", "y", "z"}:
            raise ValueError("coordinate must be one of: x, y, z")
        if not callable(value) or not callable(velocity):
            raise TypeError("value and velocity must be callable")
        if acceleration is not None and not callable(acceleration):
            raise TypeError("acceleration must be callable when supplied")
        drive_name = name or f"drive-{len(self._drives)}"
        if any(existing.name == drive_name for existing in self._drives):
            raise ValueError(f"coordinate drive name {drive_name!r} is already in use")
        drive = SpatialCoordinateDrive(
            self._idx(body), coordinate, value, velocity, acceleration,
            drive_name,
        )
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

        if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev <= 0:
            raise ValueError("max_nfev must be a positive integer")
        current = poses
        evaluations = 0
        for _ in range(max_nfev):
            residual = self.constraint_residual(current, t)
            evaluations += 1
            residual_norm = _infinity_norm(residual)
            if residual_norm <= tol:
                return current
            jacobian = self.constraint_jacobian(current, t)
            rank = int(np.linalg.matrix_rank(jacobian)) if jacobian.size else 0
            increment, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
            if not np.all(np.isfinite(increment)):
                break
            accepted = False
            scale = 1.0
            for _ in range(14):
                candidate = self._apply_increments(current, scale * increment)
                candidate_norm = _infinity_norm(self.constraint_residual(candidate, t))
                evaluations += 1
                if candidate_norm < residual_norm:
                    current = candidate
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                break
        residual_norm = _infinity_norm(self.constraint_residual(current, t))
        jacobian = self.constraint_jacobian(current, t)
        rank = int(np.linalg.matrix_rank(jacobian)) if jacobian.size else 0
        raise MechanismSolveError(
            "Spatial position solve failed after rank-aware minimum-norm projection: "
            f"residual {residual_norm:.3e} exceeds {tol:.3e}; rank {rank}, "
            f"degrees of freedom {self.ncoord - rank}, evaluations {evaluations}."
        )

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
        for drive in self._drives:
            drive.validate_derivative(t)
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
                "solver": "rank_aware_minimum_norm_projection",
                "velocity_solver": "numpy.linalg.lstsq",
                "tolerance": tol,
                "pose_increment_frame": "world",
                "drive_derivative": "caller_supplied_checked_by_adaptive_richardson",
                "drive_derivative_atol": DRIVE_DERIVATIVE_ATOL,
                "drive_derivative_rtol": DRIVE_DERIVATIVE_RTOL,
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
            excitation_id=self.excitation_id(times),
            provenance=provenance,
            capabilities=self.capabilities,
        )

    def force(
        self,
        body: int | SpatialBodyHandle,
        vector: np.ndarray | Callable[[float, tuple[Pose3D, ...], np.ndarray], ArrayLike3],
        *,
        name: str = "force",
        frame: str = "world",
        application_point: str = "center_of_mass",
    ) -> SpatialForce:
        """Create a validated force for fixed-orientation dynamics."""
        return SpatialForce(
            body=self._idx(body),
            vector=vector,
            name=name,
            frame=frame,
            application_point=application_point,
        )

    def simulate_fixed_orientation(
        self,
        t: np.ndarray,
        *,
        poses0: Iterable[Pose3D] | None = None,
        linear_velocity0: np.ndarray | None = None,
        forces: Iterable[SpatialForce] = (),
        method: str = "DOP853",
        rtol: float = 1e-9,
        atol: float = 1e-11,
        constraint_tol: float = 1e-8,
        project_samples: bool = True,
    ) -> SpatialDynamicsResult:
        """Run the restricted translational solver with constant orientations."""
        return simulate_fixed_orientation(
            self,
            t,
            poses0=poses0,
            linear_velocity0=linear_velocity0,
            forces=forces,
            method=method,
            rtol=rtol,
            atol=atol,
            constraint_tol=constraint_tol,
            project_samples=project_samples,
        )

    def dynamics_diagnostics(self, result: SpatialDynamicsResult):
        return dynamics_diagnostics(self, result)

    def dynamics_result_to_dict(self, result: SpatialDynamicsResult) -> dict[str, Any]:
        return dynamic_result_to_dict(self, result)

    def dynamics_result_to_json(
        self, result: SpatialDynamicsResult, path: str | Path, *, indent: int = 2
    ) -> Path:
        return dynamic_result_to_json(self, result, path, indent=indent)

    def dynamics_constraint_groups(self, result: SpatialDynamicsResult):
        """Group residual and multiplier histories by named constraint."""
        from .dynamics import constraint_groups

        return constraint_groups(self, result)

    def rigid_body_load(
        self,
        body: int | SpatialBodyHandle,
        *,
        force=(0.0, 0.0, 0.0),
        torque=(0.0, 0.0, 0.0),
        application_point_from_com_body=(0.0, 0.0, 0.0),
        force_frame: str = "world",
        torque_frame: str = "body",
        name: str = "load",
    ) -> RigidBodyLoad:
        """Create a force/torque descriptor for free rigid-body dynamics."""
        return RigidBodyLoad(
            body=self._idx(body), force=force, torque=torque,
            application_point_from_com_body=application_point_from_com_body,
            force_frame=force_frame, torque_frame=torque_frame, name=name,
        )

    def simulate_free_body(
        self,
        t: np.ndarray,
        *,
        linear_velocity0=(0.0, 0.0, 0.0),
        angular_velocity0_body=(0.0, 0.0, 0.0),
        loads=(),
        inertia_tensor_body: np.ndarray | None = None,
        method: str = "DOP853",
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> SpatialRigidBodyDynamicsResult:
        """Integrate one unconstrained body with translation and rotation."""
        return simulate_free_body(
            self, t, linear_velocity0=linear_velocity0,
            angular_velocity0_body=angular_velocity0_body, loads=loads,
            inertia_tensor_body=inertia_tensor_body,
            method=method, rtol=rtol, atol=atol,
        )

    def rigid_body_diagnostics(self, result: SpatialRigidBodyDynamicsResult):
        return rigid_body_diagnostics(self, result)

    def rigid_body_result_to_dict(self, result: SpatialRigidBodyDynamicsResult):
        return rigid_body_result_to_dict(self, result)

    def rigid_body_result_to_json(self, result, path: str | Path, *, indent: int = 2):
        return rigid_body_result_to_json(self, result, path, indent=indent)

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
        velocity_residuals = self._velocity_residuals(result)
        layout = self._residual_layout()
        max_translation, max_rotation = _unit_separated_maxima(result.residuals, layout)
        max_linear, max_angular = _unit_separated_maxima(velocity_residuals, layout)
        linear_error, angular_error, trajectory_checked = _trajectory_velocity_error(result)
        return SpatialResultDiagnostics(
            steps=len(result.t),
            max_position_residual=_infinity_norm(result.residuals),
            max_velocity_residual=_infinity_norm(velocity_residuals),
            max_translation_residual=max_translation,
            max_rotation_residual=max_rotation,
            max_linear_velocity_residual=max_linear,
            max_angular_velocity_residual=max_angular,
            max_trajectory_linear_velocity_error=linear_error,
            max_trajectory_angular_velocity_error=angular_error,
            trajectory_velocity_checked=trajectory_checked,
            finite=True,
            success=bool(result.success),
        )

    def _velocity_residuals(self, result: SpatialKinematicResult) -> np.ndarray:
        columns = []
        for index, poses in enumerate(result.poses):
            velocity = np.concatenate(
                (result.linear_velocity[index], result.angular_velocity[index]), axis=1
            ).reshape(-1)
            columns.append(
                self.constraint_jacobian(poses, float(result.t[index])) @ velocity
                + self.constraint_time_derivative(poses, float(result.t[index]))
            )
        return np.column_stack(columns)

    def _residual_layout(self) -> list[str]:
        layout: list[str] = []
        if self._ground is not None:
            layout.extend(["translation"] * 3 + ["rotation"] * 3)
        for joint in self._joints:
            layout.extend(["translation"] * 3)
            if isinstance(joint, FixedJoint3D):
                layout.extend(["rotation"] * 3)
        layout.extend(["translation"] * len(self._drives))
        return layout

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
        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict):
            metadata.pop("drives", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

    def excitation_id(self, times: Iterable[float]) -> str:
        """Identify sampled drive behavior separately from model structure."""
        sampled_times = _time_array(np.asarray(tuple(times), dtype=float))
        payload = {
            "model_id": self.model_id(),
            "time": _vector(sampled_times),
            "drives": [
                {
                    "body": drive.body,
                    "coordinate": drive.coordinate,
                    "value": [
                        _finite_scalar(drive.value(float(t)), "drive value")
                        for t in sampled_times
                    ],
                    "velocity": [
                        _finite_scalar(drive.velocity(float(t)), "drive velocity")
                        for t in sampled_times
                    ],
                }
                for drive in self._drives
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_spatial_model().as_dict()
        payload["model_id"] = self.model_id()
        payload["model_id_scope"] = "structure_excluding_runtime_drive_callbacks"
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
            "excitation_id": result.excitation_id,
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
            "residual_layout": self._residual_layout(),
            "velocity_residuals": self._velocity_residuals(result).tolist(),
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
        if result.excitation_id != self.excitation_id(result.t):
            raise ValueError("result excitation_id does not match this mechanism and time grid")
        expected = (len(result.t), self.nbody, 3)
        if result.linear_velocity.shape != expected or result.angular_velocity.shape != expected:
            raise ValueError(f"result velocity arrays must have shape {expected}")
        expected_residuals = (len(self._residual_layout()), len(result.t))
        if result.residuals.shape != expected_residuals:
            raise ValueError(f"result residuals must have shape {expected_residuals}")
        if len(result.poses) != len(result.t) or any(
            len(step) != self.nbody for step in result.poses
        ):
            raise ValueError("result pose history must match time and body counts")
        arrays = [result.t, result.linear_velocity, result.angular_velocity, result.residuals]
        arrays.extend(pose.translation for step in result.poses for pose in step)
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("result arrays and pose translations must contain only finite values")


def validate_spatial_kinematic_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Structurally and semantically validate a result interchange payload."""
    if not isinstance(payload, Mapping):
        raise TypeError("spatial kinematic result payload must be a mapping")
    if payload.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"schema must be {RESULT_SCHEMA!r}")
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported spatial kinematic result schema_version "
            f"{payload.get('schema_version')!r}; supported version is {RESULT_SCHEMA_VERSION}"
        )
    if not isinstance(payload.get("mbsd_version"), str) or not payload["mbsd_version"]:
        raise ValueError("mbsd_version must be a non-empty string")
    if payload.get("status") != "experimental" or payload.get("dimension") != 3:
        raise ValueError("spatial kinematic results require experimental status and dimension 3")
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
    excitation_id = payload.get("excitation_id")
    if excitation_id is not None and (
        not isinstance(excitation_id, str) or not excitation_id
    ):
        raise ValueError("excitation_id must be a non-empty string when present")
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
    if not isinstance(payload["solver_status"].get("message"), str):
        raise ValueError("solver_status.message must be a string")
    expected_units = {
        "length": "m",
        "angle": "rad",
        "time": "s",
        "linear_velocity": "m/s",
        "angular_velocity": "rad/s",
    }
    if payload["units"] != expected_units:
        raise ValueError(f"units must be {expected_units!r}")
    expected_conventions = {
        "world_frame": "right_handed_xyz",
        "rotation": "active_body_to_world",
        "quaternion_order": ["w", "x", "y", "z"],
        "velocity_frame": "world",
        "array_layout": "time_body_component",
    }
    if payload["conventions"] != expected_conventions:
        raise ValueError("conventions must match the spatial kinematic-result contract")
    _validate_result_capabilities(payload["capabilities"])
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
    residual_layout = payload.get("residual_layout")
    velocity_residuals = payload.get("velocity_residuals")
    if (residual_layout is None) != (velocity_residuals is None):
        raise ValueError("residual_layout and velocity_residuals must be provided together")
    if residual_layout is not None:
        if (
            not isinstance(residual_layout, list)
            or len(residual_layout) != residuals.shape[0]
            or any(kind not in {"translation", "rotation"} for kind in residual_layout)
        ):
            raise ValueError("residual_layout must classify every residual row")
        velocity_residuals = np.asarray(velocity_residuals, dtype=float)
        if velocity_residuals.shape != residuals.shape or not np.all(
            np.isfinite(velocity_residuals)
        ):
            raise ValueError("velocity_residuals must match residuals and be finite")
    bodies = payload["bodies"]
    if not isinstance(bodies, list) or not bodies:
        raise ValueError("bodies must be a non-empty list")
    names: list[str] = []
    pose_histories: list[tuple[Pose3D, ...]] = []
    linear_histories: list[np.ndarray] = []
    angular_histories: list[np.ndarray] = []
    for index, body in enumerate(bodies):
        if not isinstance(body, Mapping):
            raise ValueError(f"bodies[{index}] must be a mapping")
        if body.get("index") != index or not isinstance(body.get("name"), str):
            raise ValueError(f"bodies[{index}] has invalid identity fields")
        names.append(body["name"])
        if not body["name"]:
            raise ValueError(f"bodies[{index}].name must be non-empty")
        poses = body.get("poses")
        if not isinstance(poses, list) or len(poses) != len(times):
            raise ValueError(f"bodies[{index}].poses must match time length")
        parsed_poses: list[Pose3D] = []
        for pose_index, pose in enumerate(poses):
            try:
                parsed_poses.append(_pose_from_result_payload(pose))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"bodies[{index}].poses[{pose_index}] is invalid: {error}"
                ) from error
        pose_histories.append(tuple(parsed_poses))
        for velocity_name in ("linear_velocity", "angular_velocity"):
            velocity = np.asarray(body.get(velocity_name), dtype=float)
            if velocity.shape != (len(times), 3) or not np.all(np.isfinite(velocity)):
                raise ValueError(
                    f"bodies[{index}].{velocity_name} must have shape ({len(times)}, 3)"
                )
            if velocity_name == "linear_velocity":
                linear_histories.append(velocity)
            else:
                angular_histories.append(velocity)
    if len(names) != len(set(names)):
        raise ValueError("body names must be unique")
    diagnostics = payload["diagnostics"]
    expected_steps = diagnostics.get("steps")
    if isinstance(expected_steps, bool) or expected_steps != len(times):
        raise ValueError("diagnostics.steps must match time length")
    for name in ("max_position_residual", "max_velocity_residual"):
        value = diagnostics.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"diagnostics.{name} must be a non-negative finite number")
    if not np.isclose(
        diagnostics["max_position_residual"], _infinity_norm(residuals), atol=1e-15, rtol=1e-12
    ):
        raise ValueError("diagnostics.max_position_residual does not match residuals")
    for name in ("finite", "success"):
        if not isinstance(diagnostics.get(name), bool):
            raise ValueError(f"diagnostics.{name} must be a boolean")
    if diagnostics["success"] is not payload["solver_status"]["success"]:
        raise ValueError("diagnostics.success must match solver_status.success")
    if diagnostics["finite"] is not True:
        raise ValueError("diagnostics.finite must be true when all result arrays are finite")
    tolerance = payload["provenance"].get("tolerance")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("provenance.tolerance must be a positive finite number")
    if payload["solver_status"]["success"] and (
        diagnostics["max_position_residual"] > tolerance
        or diagnostics["max_velocity_residual"] > tolerance
    ):
        raise ValueError(
            "successful result residuals must not exceed provenance.tolerance"
        )
    for name in (
        "max_trajectory_linear_velocity_error",
        "max_trajectory_angular_velocity_error",
    ):
        if name in diagnostics:
            value = diagnostics[name]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"diagnostics.{name} must be a non-negative finite number")
    if residual_layout is not None:
        expected_values = {
            "max_translation_residual_m": _unit_separated_maxima(
                residuals, residual_layout
            )[0],
            "max_rotation_residual_rad": _unit_separated_maxima(
                residuals, residual_layout
            )[1],
            "max_linear_velocity_residual_m_per_s": _unit_separated_maxima(
                velocity_residuals, residual_layout
            )[0],
            "max_angular_velocity_residual_rad_per_s": _unit_separated_maxima(
                velocity_residuals, residual_layout
            )[1],
        }
        for name, expected in expected_values.items():
            value = diagnostics.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0.0
                or not np.isclose(value, expected, atol=1e-15, rtol=1e-12)
            ):
                raise ValueError(f"diagnostics.{name} does not match residual arrays")
        if not np.isclose(
            diagnostics["max_velocity_residual"],
            _infinity_norm(velocity_residuals),
            atol=1e-15,
            rtol=1e-12,
        ):
            raise ValueError("diagnostics.max_velocity_residual does not match velocity_residuals")
    if "trajectory_velocity_checked" in diagnostics and not isinstance(
        diagnostics["trajectory_velocity_checked"], bool
    ):
        raise ValueError("diagnostics.trajectory_velocity_checked must be a boolean")
    if "trajectory_velocity_checked" in diagnostics:
        step_poses = tuple(
            tuple(pose_histories[body][step] for body in range(len(bodies)))
            for step in range(len(times))
        )
        expected_linear, expected_angular, expected_checked = (
            _trajectory_velocity_error_arrays(
                times,
                step_poses,
                np.stack(linear_histories, axis=1),
                np.stack(angular_histories, axis=1),
            )
        )
        for name, expected in (
            ("max_trajectory_linear_velocity_error", expected_linear),
            ("max_trajectory_angular_velocity_error", expected_angular),
        ):
            if not np.isclose(
                diagnostics.get(name, np.nan), expected, atol=1e-12, rtol=1e-9
            ):
                raise ValueError(f"diagnostics.{name} does not match pose trajectory")
        if diagnostics["trajectory_velocity_checked"] is not expected_checked:
            raise ValueError(
                "diagnostics.trajectory_velocity_checked does not match time length"
            )
    try:
        return json.loads(json.dumps(payload, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("spatial kinematic result must contain finite JSON values") from error


def _validate_result_capabilities(capabilities: Mapping[str, Any]) -> None:
    expected = {
        "position_solve": True,
        "velocity_solve": True,
        "spatial_dynamics": False,
        "velocity_frame": "world",
    }
    if any(capabilities.get(name) != value for name, value in expected.items()):
        raise ValueError("capabilities do not match the spatial kinematics contract")
    if capabilities.get("joint_types") not in (("spherical", "fixed"), ["spherical", "fixed"]):
        raise ValueError("capabilities.joint_types must contain spherical and fixed")


def _pose_from_result_payload(payload: Any) -> Pose3D:
    if not isinstance(payload, Mapping):
        raise ValueError("pose must be a mapping")
    translation = payload.get("translation")
    rotation = payload.get("rotation")
    if not isinstance(rotation, Mapping):
        raise ValueError("pose rotation must be a mapping")
    try:
        components = np.asarray(
            [rotation[name] for name in ("w", "x", "y", "z")], dtype=float
        )
    except KeyError as error:
        raise ValueError(f"pose rotation missing {error.args[0]!r}") from error
    if not np.all(np.isfinite(components)):
        raise ValueError("pose rotation components must be finite")
    norm = float(np.linalg.norm(components))
    if not np.isclose(norm, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"pose quaternion norm must be 1, got {norm:.12g}")
    quaternion = Quaternion(*components)
    return Pose3D(translation=translation, rotation=quaternion)


def load_spatial_kinematic_result_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return validate_spatial_kinematic_result_payload(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _trajectory_velocity_error(
    result: SpatialKinematicResult,
) -> tuple[float, float, bool]:
    """Compare reported velocities with centered pose differences at interior steps."""
    return _trajectory_velocity_error_arrays(
        result.t, result.poses, result.linear_velocity, result.angular_velocity
    )


def _trajectory_velocity_error_arrays(
    times: np.ndarray,
    poses: tuple[tuple[Pose3D, ...], ...],
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
) -> tuple[float, float, bool]:
    if len(times) < 3:
        return 0.0, 0.0, False
    max_linear = 0.0
    max_angular = 0.0
    for index in range(1, len(times) - 1):
        duration = float(times[index + 1] - times[index - 1])
        for body_index in range(linear_velocity.shape[1]):
            previous = poses[index - 1][body_index]
            following = poses[index + 1][body_index]
            linear = (following.translation - previous.translation) / duration
            relative = following.rotation.compose(previous.rotation.inverse())
            angular = _rotation_vector(relative) / duration
            max_linear = max(
                max_linear,
                _infinity_norm(linear - linear_velocity[index, body_index]),
            )
            max_angular = max(
                max_angular,
                _infinity_norm(angular - angular_velocity[index, body_index]),
            )
    return max_linear, max_angular, True


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


def _adaptive_derivative(
    function: Callable[[float], float], t: float, value_name: str
) -> tuple[float, float, bool]:
    """Estimate a first derivative without tying the step to absolute time.

    Central differences are refined and Richardson-extrapolated.  The lower
    step bound follows floating-point spacing at ``t``; the nominal step uses
    the cube root of machine epsilon, appropriate for a first derivative.
    """
    spacing = abs(float(np.spacing(t)))
    base = max(float(np.cbrt(np.finfo(float).eps)), 16.0 * spacing)
    # Start wide enough to survive representable-time quantization at large
    # absolute times, then refine far enough for high-frequency laws.
    steps = [base * (2.0**power) for power in range(10, -4, -1)]
    estimates: list[tuple[float, float]] = []
    for step in steps:
        if t + step == t or t - step == t:
            continue
        forward_time = t + step
        backward_time = t - step
        forward = _finite_scalar(function(forward_time), value_name)
        backward = _finite_scalar(function(backward_time), value_name)
        actual_span = forward_time - backward_time
        derivative = (forward - backward) / actual_span
        roundoff = (
            8.0
            * np.finfo(float).eps
            * max(1.0, abs(forward), abs(backward))
            / actual_span
        )
        estimates.append((derivative, roundoff))
    if len(estimates) < 2:
        raise ValueError(f"numerical derivative check for {value_name} is inconclusive")
    candidates: list[tuple[float, float]] = []
    for (coarse, _), (fine, roundoff) in zip(estimates, estimates[1:]):
        truncation = abs(fine - coarse) / 3.0
        richardson = fine + (fine - coarse) / 3.0
        candidates.append((richardson, truncation + roundoff))
    usable = candidates
    for index in range(2, len(candidates)):
        if candidates[index][1] > 4.0 * candidates[index - 1][1]:
            usable = candidates[:index]
            break
    numeric, uncertainty = min(usable, key=lambda item: item[1])
    scale = max(1.0, abs(numeric))
    converged = uncertainty <= max(DRIVE_DERIVATIVE_ATOL, DRIVE_DERIVATIVE_RTOL * scale)
    return float(numeric), float(uncertainty), bool(converged)


def _readonly_array(value: Any) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


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
    return float(np.max(np.abs(array))) if array.size else 0.0


def _unit_separated_maxima(
    residuals: np.ndarray, layout: list[str] | tuple[str, ...]
) -> tuple[float, float]:
    array = np.asarray(residuals, dtype=float)
    if array.ndim != 2 or array.shape[0] != len(layout):
        raise ValueError("residual layout must match the residual row count")
    translation_rows = [index for index, kind in enumerate(layout) if kind == "translation"]
    rotation_rows = [index for index, kind in enumerate(layout) if kind == "rotation"]
    translation = _infinity_norm(array[translation_rows]) if translation_rows else 0.0
    rotation = _infinity_norm(array[rotation_rows]) if rotation_rows else 0.0
    return translation, rotation


def _vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _write_json(payload: dict[str, Any], path: str | Path, indent: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=indent, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _package_version() -> str:
    try:
        return version("mbsd")
    except PackageNotFoundError:
        return "0+local"
