"""Public exceptions raised by MBSD."""

from __future__ import annotations

from typing import Any, Mapping


class MechanismSolveError(RuntimeError):
    """Raised when a mechanism solve is singular, inconsistent, or divergent.

    ``code`` is a stable machine-readable category, ``context`` carries the
    offending values, and ``remediation`` suggests a fix. All three are
    optional so plain ``MechanismSolveError(message)`` keeps working.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "solve_failed",
        context: Mapping[str, Any] | None = None,
        remediation: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = dict(context or {})
        self.remediation = remediation

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "context": dict(self.context),
            "remediation": self.remediation,
        }


class NotPortableError(MechanismSolveError):
    """Raised when a model with Python callbacks is exported for import."""

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "not_portable")
        super().__init__(message, **kwargs)
