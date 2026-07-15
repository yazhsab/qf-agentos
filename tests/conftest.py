"""Pytest fixtures for the QF-AgentOS test suite.

Shared helpers live in ``qf_test_utils`` (on the pytest ``pythonpath``) to avoid
a name clash with a stray top-level ``tests`` package shipped by a dependency.
"""

from __future__ import annotations

import pytest

from qf_agentos.core.ir import ProblemSpec, load_spec
from qf_test_utils import EXAMPLE, make_spec


@pytest.fixture
def spec_factory():
    return make_spec


@pytest.fixture
def small_spec() -> ProblemSpec:
    return make_spec()


@pytest.fixture(scope="session")
def example_spec() -> ProblemSpec:
    return load_spec(EXAMPLE)
