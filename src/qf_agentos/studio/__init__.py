"""QF-Studio — a self-contained web UI for QF-AgentOS, served by the REST API.

A single static page (inline CSS/JS, no build step, no external requests) that
submits a problem spec to the async job queue, polls it, and renders the honest
decision + evidence report. Served at ``GET /`` when the ``server`` extra is
installed.
"""

from __future__ import annotations

from .assets import list_example_specs, read_index_html

__all__ = ["list_example_specs", "read_index_html"]
