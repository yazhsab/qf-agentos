"""Typed exception hierarchy for QF-AgentOS.

Every failure mode the platform can surface has a specific exception so callers
(the CLI, the REST API, the SDK) can react precisely and map to the right exit
code / HTTP status. All inherit from :class:`QFAgentOSError`.
"""

from __future__ import annotations


class QFAgentOSError(Exception):
    """Base class for all QF-AgentOS errors.

    Attributes:
        message: Human-readable description.
        exit_code: Suggested process exit code for CLI use.
    """

    exit_code: int = 1

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SpecError(QFAgentOSError):
    """The problem specification is missing, malformed, or fails validation."""

    exit_code = 2


class ConfigurationError(QFAgentOSError):
    """Invalid runtime configuration (settings, environment, credentials)."""

    exit_code = 3


class InfeasibleProblemError(QFAgentOSError):
    """The problem, as stated, admits no feasible solution.

    This is a *legitimate* modelling outcome, not a crash: it is raised only when
    a caller explicitly requests strict feasibility. The pipeline itself records
    infeasibility as evidence rather than raising.
    """

    exit_code = 4


class BackendError(QFAgentOSError):
    """A solver backend failed to execute."""

    exit_code = 5


class BackendUnavailableError(BackendError):
    """A backend was requested but its dependency or credentials are missing.

    Carries the pip extra / environment hint needed to enable it.
    """

    exit_code = 6

    def __init__(self, backend: str, hint: str) -> None:
        super().__init__(f"Backend '{backend}' is unavailable: {hint}")
        self.backend = backend
        self.hint = hint


class PolicyViolationError(QFAgentOSError):
    """An action was blocked by the autonomy/budget policy engine.

    Raised only when a caller forces an action the policy denies. Normal flow
    records the denial and abstains instead.
    """

    exit_code = 7

    def __init__(self, message: str, *, needs_human_approval: bool = False) -> None:
        super().__init__(message)
        self.needs_human_approval = needs_human_approval


class VerificationError(QFAgentOSError):
    """Deterministic verification detected an inconsistency (e.g. a solver's
    reported objective disagrees with independent recomputation)."""

    exit_code = 8


__all__ = [
    "BackendError",
    "BackendUnavailableError",
    "ConfigurationError",
    "InfeasibleProblemError",
    "PolicyViolationError",
    "QFAgentOSError",
    "SpecError",
    "VerificationError",
]
