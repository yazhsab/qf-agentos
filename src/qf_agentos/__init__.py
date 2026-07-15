"""QF-AgentOS — an agentic operating system for quantum finance.

Give it a financial optimisation problem in structured form. A team of
deterministic agents will formulate it, build strong classical baselines,
identify a quantum-compatible sub-problem, run gate-model simulation (and,
where authorised, real QPUs), *verify* feasibility and quantum contribution,
and honestly report whether quantum technology should be used.

The design principle: an agent that knows *when not* to use quantum computing
is more valuable than one that always produces a circuit.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .core.errors import (
    BackendError,
    BackendUnavailableError,
    ConfigurationError,
    InfeasibleProblemError,
    PolicyViolationError,
    QFAgentOSError,
    SpecError,
    VerificationError,
)
from .core.ir import (
    AutonomyLevel,
    Constraints,
    ExecutionPolicy,
    Objective,
    ProblemSpec,
    Security,
    load_spec,
    parse_spec,
)
from .core.workflow import RunContext, Workflow
from .pipeline import build_default_pipeline, solve

try:
    __version__ = _version("qf-agentos")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "AutonomyLevel",
    "BackendError",
    "BackendUnavailableError",
    "ConfigurationError",
    "Constraints",
    "ExecutionPolicy",
    "InfeasibleProblemError",
    "Objective",
    "PolicyViolationError",
    "ProblemSpec",
    "QFAgentOSError",
    "RunContext",
    "Security",
    "SpecError",
    "VerificationError",
    "Workflow",
    "__version__",
    "build_default_pipeline",
    "load_spec",
    "parse_spec",
    "solve",
]
