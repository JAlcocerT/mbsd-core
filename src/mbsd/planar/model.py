"""
Mechanism data structure for 2D multibody systems.

Replaces the global MBody struct used throughout the MATLAB code.
Each mechanism is defined by creating an MBody instance with bodies and joints.

Migrated from the BielaManivela.m pattern and FuncionesMBody/*.m
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable


@dataclass
class Body:
    name: str
    mass: float = 0.0
    inertia: float = 0.0
    rG: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class RevJoint:
    """Revolute (pin) joint between body i and body j.
    ri, rj are the local attachment points on each body."""
    i: int
    j: int
    ri: np.ndarray
    rj: np.ndarray


@dataclass
class PrismJoint:
    """Prismatic (slider) joint between body i and body j.
    hi is the sliding direction in body i's local frame."""
    i: int
    j: int
    ri: np.ndarray
    rj: np.ndarray
    hi: np.ndarray


@dataclass
class PuntoLineaJoint:
    """Point-on-Line joint: point on body i stays on line on body j.
    ri: local point on body i
    rj: reference point on body j
    hi: line direction in body i's local frame (defines the line)
    """
    i: int
    j: int
    ri: np.ndarray
    rj: np.ndarray
    hi: np.ndarray


@dataclass
class LevaJoint:
    """CAM (Leva) joint: two parameterized surfaces in contact.
    geom_i: Surface object for body i (defines the CAM surface in body i's frame)
    geom_j: Surface object for body j (defines the follower surface in body j's frame)
    ri: reference point on body i (origin of surface i's local frame)
    rj: reference point on body j (origin of surface j's local frame)
    """
    i: int
    j: int
    geom_i: object  # Surface instance
    geom_j: object  # Surface instance
    ri: np.ndarray = field(default_factory=lambda: np.zeros(2))
    rj: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class GearJoint:
    """Gear joint: two rotating bodies with fixed transmission ratio.
    tau: transmission ratio (R_j / R_i or tooth_j / tooth_i)
    Constraint: theta_i - tau * theta_j = 0
    """
    i: int
    j: int
    tau: float  # transmission ratio


@dataclass
class UserConstraint:
    """User-defined constraint with its functions.

    constraint_fn(mbody, q, t) -> array of constraint values
    jacobian_fn(mbody, q, t) -> 2D array (nrus x ncoord)
    dt_jacobian_fn(mbody, q, v, t) -> 2D array (nrus x ncoord)
    dt_constraint_fn(mbody, q, t) -> array (partial dC/dt)
    dtdt_constraint_fn(mbody, q, v, t) -> array (d²C/dt²)
    """
    constraint_fn: Callable
    jacobian_fn: Callable
    dt_jacobian_fn: Callable
    dt_constraint_fn: Callable
    dtdt_constraint_fn: Callable
    count: int = 1


class MBody:
    def __init__(self):
        self.dim = 2
        self.bodies: List[Body] = []
        self.rev_joints: List[RevJoint] = []
        self.prism_joints: List[PrismJoint] = []
        self.punto_linea_joints: List[PuntoLineaJoint] = []
        self.leva_joints: List[LevaJoint] = []
        self.gear_joints: List[GearJoint] = []
        self.user_constraints: List[UserConstraint] = []

        self.g = np.array([0.0, -9.81])
        self.w = 0.0
        self.t = 0.0

    @property
    def nb(self):
        return len(self.bodies)

    @property
    def nr(self):
        return len(self.rev_joints)

    @property
    def np_(self):
        return len(self.prism_joints)

    @property
    def nl(self):
        return len(self.leva_joints)

    @property
    def npl(self):
        return len(self.punto_linea_joints)

    @property
    def ng(self):
        return len(self.gear_joints)

    @property
    def nrus(self):
        return sum(uc.count for uc in self.user_constraints)

    @property
    def ncoord(self):
        return 3 * self.nb + 2 * self.nl  # Include 2 surface parameters per CAM joint

    @property
    def nrestr(self):
        return 3 + 2 * self.nr + 2 * self.np_ + 3 * self.nl + self.npl + self.ng + self.nrus

    def get_qi(self, q, body_idx):
        """Extract coordinates [x, y, theta] for body body_idx (0-based)."""
        s = 3 * body_idx
        return q[s:s + 3]

    def get_vi(self, v, body_idx):
        """Extract velocities [vx, vy, dtheta] for body body_idx (0-based)."""
        s = 3 * body_idx
        return v[s:s + 3]
