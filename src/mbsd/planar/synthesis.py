"""Small planar mechanism synthesis helpers.

The initial synthesis surface is intentionally narrow: four-bar geometry,
assembly, and three-precision-point function generation. These helpers are
pure NumPy and complement the dynamic mechanism builder without adding a new
modeling layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


PrecisionPoints = Iterable[tuple[float, float]]


@dataclass(frozen=True)
class FourBar:
    """Planar four-bar link lengths.

    Link convention:
    - ``ground`` is the distance between fixed pivots A and D.
    - ``crank`` is AB.
    - ``coupler`` is BC.
    - ``rocker`` is DC.
    """

    ground: float
    crank: float
    coupler: float
    rocker: float

    def __post_init__(self) -> None:
        for name, value in {
            "ground": self.ground,
            "crank": self.crank,
            "coupler": self.coupler,
            "rocker": self.rocker,
        }.items():
            value = float(value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} length must be finite and positive")
            object.__setattr__(self, name, value)

    @property
    def lengths(self) -> tuple[float, float, float, float]:
        return (self.ground, self.crank, self.coupler, self.rocker)

    def grashof_class(self) -> str:
        return grashof_class(*self.lengths)

    def assemble(self, theta: float, branch: int = 1) -> "FourBarPose":
        return assemble_four_bar(self, theta, branch=branch)


@dataclass(frozen=True)
class FourBarPose:
    """Assembled four-bar joint positions and output angles."""

    theta: float
    branch: int
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    coupler_angle: float
    rocker_angle: float


@dataclass(frozen=True)
class AffineFit:
    """Closed-form affine fit ``y_fit = scale * x + offset``."""

    scale: float
    offset: float
    rms_error: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(x, dtype=float) + self.offset


def grashof_class(ground: float, crank: float, coupler: float, rocker: float) -> str:
    """Classify a four-bar by the Grashof length condition."""
    links = np.asarray([ground, crank, coupler, rocker], dtype=float)
    if links.shape != (4,) or not np.all(np.isfinite(links)) or np.any(links <= 0.0):
        raise ValueError("four-bar lengths must be finite and positive")

    shortest = float(np.min(links))
    longest = float(np.max(links))
    middle_sum = float(np.sum(links) - shortest - longest)
    margin = shortest + longest - middle_sum

    if abs(margin) < 1e-12:
        return "change-point"
    if margin > 0.0:
        return "non-Grashof"
    if shortest == crank:
        return "crank-rocker"
    if shortest == ground:
        return "double-crank"
    return "double-rocker"


def assemble_four_bar(four_bar: FourBar, theta: float, branch: int = 1) -> FourBarPose:
    """Assemble a four-bar at crank angle ``theta``.

    ``branch`` selects one of the two circle-intersection assembly modes. Use
    ``+1`` for one side of the directed segment B-to-D and ``-1`` for the
    reflected configuration.
    """
    if branch not in (-1, 1):
        raise ValueError("branch must be +1 or -1")
    theta = _finite_scalar(theta, "theta")

    ground, crank, coupler, rocker = four_bar.lengths
    a = np.array([0.0, 0.0])
    d = np.array([ground, 0.0])
    b = np.array([crank * np.cos(theta), crank * np.sin(theta)])

    db = d - b
    distance = float(np.linalg.norm(db))
    if distance > coupler + rocker + 1e-12:
        raise ValueError("four-bar cannot assemble: pivots are too far apart")
    if distance < abs(coupler - rocker) - 1e-12:
        raise ValueError("four-bar cannot assemble: one link is contained by the other")
    if distance < 1e-12:
        raise ValueError("four-bar cannot assemble: moving pivots are coincident")

    along = (coupler**2 - rocker**2 + distance**2) / (2.0 * distance)
    height_sq = coupler**2 - along**2
    if height_sq < -1e-12:
        raise ValueError("four-bar cannot assemble: no real circle intersection")

    height = np.sqrt(max(height_sq, 0.0))
    midpoint = b + along * db / distance
    normal = np.array([-db[1], db[0]]) / distance
    c = midpoint + branch * height * normal

    coupler_angle = float(np.arctan2(c[1] - b[1], c[0] - b[0]))
    rocker_angle = float(np.arctan2(c[1] - d[1], c[0] - d[0]))
    return FourBarPose(
        theta=theta,
        branch=branch,
        a=a,
        b=b,
        c=c,
        d=d,
        coupler_angle=coupler_angle,
        rocker_angle=rocker_angle,
    )


def rocker_angles(
    four_bar: FourBar,
    theta: np.ndarray,
    branch: int = 1,
    unwrap: bool = True,
) -> np.ndarray:
    """Return rocker output angles for a crank-angle grid."""
    theta_arr = _as_1d_array(theta, "theta")
    angles = np.array(
        [assemble_four_bar(four_bar, float(item), branch=branch).rocker_angle for item in theta_arr]
    )
    if unwrap:
        return np.unwrap(angles)
    return angles


def affine_fit(x: np.ndarray, y: np.ndarray) -> AffineFit:
    """Fit ``y ~= scale * x + offset`` by least squares."""
    x_arr = _as_1d_array(x, "x")
    y_arr = _as_1d_array(y, "y")
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    design = np.column_stack([x_arr, np.ones_like(x_arr)])
    coeffs, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    scale = float(coeffs[0])
    offset = float(coeffs[1])
    residual = scale * x_arr + offset - y_arr
    rms_error = float(np.sqrt(np.mean(residual**2)))
    return AffineFit(scale=scale, offset=offset, rms_error=rms_error)


def freudenstein_3pt(
    precision_points: PrecisionPoints,
    ground: float = 1.0,
) -> FourBar:
    """Synthesize a four-bar from three input/output angle precision points.

    The precision points are ``(crank_angle, rocker_angle)`` pairs in radians.
    The implementation solves Freudenstein's equation in closed form and
    returns a ``FourBar`` with the requested ground length.
    """
    ground = _finite_scalar(ground, "ground")
    if ground <= 0.0:
        raise ValueError("ground length must be positive")

    points = list(precision_points)
    if len(points) != 3:
        raise ValueError("freudenstein_3pt requires exactly three precision points")

    matrix = np.zeros((3, 3))
    rhs = np.zeros(3)
    for row, (phi, psi) in enumerate(points):
        phi = _finite_scalar(phi, "crank angle")
        psi = _finite_scalar(psi, "rocker angle")
        matrix[row, 0] = np.cos(psi)
        matrix[row, 1] = -np.cos(phi)
        matrix[row, 2] = 1.0
        rhs[row] = np.cos(psi - phi)

    try:
        k1, k2, k3 = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("precision points produce a singular synthesis system") from exc

    if abs(k1) < 1e-12 or abs(k2) < 1e-12:
        raise ValueError("precision points produce non-physical link ratios")

    crank = ground / k1
    rocker = ground / k2
    if crank <= 0.0 or rocker <= 0.0:
        raise ValueError("precision points produce non-positive link lengths")

    coupler_sq = crank**2 + rocker**2 + ground**2 - 2.0 * k3 * crank * rocker
    if coupler_sq <= 0.0:
        raise ValueError("precision points produce a non-positive coupler length")

    return FourBar(
        ground=ground,
        crank=float(crank),
        coupler=float(np.sqrt(coupler_sq)),
        rocker=float(rocker),
    )


def _finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _as_1d_array(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr
