"""Public exceptions raised by MBSD."""


class MechanismSolveError(RuntimeError):
    """Raised when a mechanism solve is singular, inconsistent, or divergent."""
