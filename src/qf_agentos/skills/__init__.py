"""Quantum Skills — a plugin ecosystem for finance-native quantum methods.

A skill is a directory containing a `skill.yaml` manifest that declares its
inputs, its classical and quantum methods, and the verification checks that must
pass. This registry discovers built-in skills (shipped with the package) and any
user skills under an external directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_BUILTIN_DIR = Path(__file__).parent


def load_skills(extra_dir: str | Path | None = None) -> list[dict[str, Any]]:
    skills: list[dict[str, Any]] = []
    search_dirs = [_BUILTIN_DIR]
    if extra_dir is not None:
        search_dirs.append(Path(extra_dir))

    for base in search_dirs:
        if not base.exists():
            continue
        for manifest in sorted(base.glob("*/skill.yaml")):
            try:
                data = yaml.safe_load(manifest.read_text())
                data["_path"] = str(manifest.parent)
                data["_builtin"] = base == _BUILTIN_DIR
                skills.append(data)
            except Exception as exc:  # pragma: no cover - defensive
                skills.append({"name": manifest.parent.name, "error": str(exc)})
    return skills


__all__ = ["load_skills"]
