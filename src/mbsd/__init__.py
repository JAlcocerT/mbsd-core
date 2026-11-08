"""Python-native multibody-system dynamics tools."""

from importlib.metadata import version

from .errors import MechanismSolveError, NotPortableError
from .operations import capabilities, operation_manifest
from .planar.builder import Mechanism
from .planar.forces import Spring
from .planar.laws import ConstantSpeed, Polynomial, Tabulated
from .stability import api_surface

__version__ = version("mbsd")

__all__ = [
    "ConstantSpeed",
    "Mechanism",
    "MechanismSolveError",
    "NotPortableError",
    "Polynomial",
    "Spring",
    "Tabulated",
    "api_surface",
    "capabilities",
    "operation_manifest",
]
