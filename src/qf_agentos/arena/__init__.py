"""QF-Arena — a reproducible benchmark across every problem family and backend.

Runs each benchmark problem through the full agent pipeline and records the honest
verdict (does quantum help?), the classical and quantum scores, and the
quantum-contribution signal — then renders a leaderboard. The whole point is an
at-a-glance, reproducible answer to "does quantum help anywhere here?" — which, on
the shipped suite, is a resounding *no* (parity at best), exactly as the platform
is designed to report.
"""

from __future__ import annotations

from .report import render_leaderboard
from .runner import ArenaEntry, ArenaResult, load_example_specs, run_arena

__all__ = [
    "ArenaEntry",
    "ArenaResult",
    "load_example_specs",
    "render_leaderboard",
    "run_arena",
]
