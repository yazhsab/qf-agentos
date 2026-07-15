"""Quantum Skills registry discovery."""

from __future__ import annotations

from qf_agentos.skills import load_skills


def test_builtin_collateral_skill_discovered():
    skills = load_skills()
    names = {s.get("name") for s in skills}
    assert "collateral-optimizer" in names
    skill = next(s for s in skills if s.get("name") == "collateral-optimizer")
    assert skill["problem"] == "collateral_allocation"
    assert skill["_builtin"] is True


def test_external_dir_and_malformed_manifest(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    (good / "skill.yaml").write_text(
        "name: my-skill\nversion: 1.0\nproblem: collateral_allocation\n"
    )

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "skill.yaml").write_text("name: [broken\n:\n  - unterminated")

    skills = load_skills(extra_dir=tmp_path)
    names = {s.get("name") for s in skills}
    assert "my-skill" in names  # external skill discovered
    assert "collateral-optimizer" in names  # builtin still present
    # A malformed manifest is reported as an error entry, not a crash.
    assert any("error" in s for s in skills if s.get("name") == "bad")


def test_missing_external_dir_is_ignored(tmp_path):
    skills = load_skills(extra_dir=tmp_path / "does-not-exist")
    assert any(s.get("name") == "collateral-optimizer" for s in skills)
