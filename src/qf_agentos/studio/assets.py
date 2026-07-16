"""Static-asset access for QF-Studio: the page HTML and the example spec presets."""

from __future__ import annotations

import contextlib
import os
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


def read_index_html() -> str:
    """Return the Studio single-page HTML (packaged with the wheel)."""
    return (files("qf_agentos.studio") / "index.html").read_text(encoding="utf-8")


def _examples_dir() -> Path | None:
    """Locate the example specs — packaged copy, the repo tree, or an override."""
    candidates: list[Path] = []
    override = os.environ.get("QF_EXAMPLES_DIR")
    if override:
        candidates.append(Path(override))
    with contextlib.suppress(ModuleNotFoundError, FileNotFoundError):  # pragma: no cover
        candidates.append(Path(str(files("qf_agentos.studio") / "examples")))
    # Dev / sdist: the repo-root examples/ dir (src/qf_agentos/studio -> repo root).
    candidates.append(Path(__file__).resolve().parents[3] / "examples")
    candidates.append(Path.cwd() / "examples")
    for c in candidates:
        if c.is_dir():
            return c
    return None


def list_example_specs() -> list[dict[str, Any]]:
    """Return the bundled example specs as ``{name, problem, yaml}`` for the UI."""
    directory = _examples_dir()
    if directory is None:
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(text)
            problem = data.get("problem", "?") if isinstance(data, dict) else "?"
        except yaml.YAMLError:
            problem = "?"
        out.append({"name": path.stem, "problem": str(problem), "yaml": text})
    return out


__all__ = ["list_example_specs", "read_index_html"]
