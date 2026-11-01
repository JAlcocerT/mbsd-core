"""Structured guardrail reports shared by the planar and spatial builders.

``validate`` returns a ``ValidationReport`` before any solve; ``check``
returns a ``CheckReport`` after one. Both are plain data with ``as_dict``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import MechanismSolveError


DEFAULT_MAX_STEPS = 100_000


@dataclass(frozen=True)
class Issue:
    """One structured finding from ``validate`` with a stable ``code``."""

    code: str
    severity: str
    message: str
    remediation: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Pre-solve model report: ``ok`` is false when any issue is an error."""

    ok: bool
    portable: bool
    classification: str | None
    degrees_of_freedom: int | None
    issues: tuple[Issue, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "portable": self.portable,
            "classification": self.classification,
            "degrees_of_freedom": self.degrees_of_freedom,
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class Check:
    """One post-solve check: ``passed`` is ``None`` when the check was skipped."""

    name: str
    value: float | None
    threshold: float | None
    passed: bool | None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "note": self.note,
        }


@dataclass(frozen=True)
class CheckReport:
    """Post-solve verdict with the numbers and thresholds behind it."""

    passed: bool
    result_type: str
    checks: tuple[Check, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "result_type": self.result_type,
            "checks": [check.as_dict() for check in self.checks],
        }


def check_steps(t, max_steps: int) -> None:
    """Raise a ``resource_limit`` error when a time grid exceeds ``max_steps``."""
    if len(t) > max_steps:
        raise MechanismSolveError(
            f"{len(t)} time samples exceed max_steps={max_steps}.",
            code="resource_limit",
            context={"steps": len(t), "max_steps": max_steps},
            remediation="Use fewer samples or raise max_steps deliberately.",
        )
