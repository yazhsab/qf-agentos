"""The agent team.

Each agent is a deterministic workflow step: it reads/writes the shared
RunContext and returns a one-line summary. Together they realise the reference
architecture — Requirements, Formulation, Classical Baseline, Hardware Planner,
Quantum Algorithm, Execution, Verification, Quantum-Advantage Auditor, and
Governance.
"""

from .auditor import auditor_agent
from .classical import classical_baseline_agent
from .formulation import formulation_agent
from .governance import governance_agent
from .planner import hardware_planner_agent
from .quantum_agent import execution_agent, quantum_algorithm_agent
from .requirements import requirements_agent
from .verification import verification_agent

__all__ = [
    "auditor_agent",
    "classical_baseline_agent",
    "execution_agent",
    "formulation_agent",
    "governance_agent",
    "hardware_planner_agent",
    "quantum_algorithm_agent",
    "requirements_agent",
    "verification_agent",
]
