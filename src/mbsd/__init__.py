"""Python-native multibody-system dynamics tools."""

from importlib.metadata import version

from .errors import MechanismSolveError
from .planar.builder import Mechanism
from .planar.forces import Spring

__version__ = version("mbsd")

__all__ = [
    "Mechanism",
    "MechanismSolveError",
    "Spring",
]
