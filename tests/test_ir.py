"""Finance IR validation and robust spec loading."""

from __future__ import annotations

import pytest

from qf_agentos.core.errors import SpecError
from qf_agentos.core.ir import Security, load_spec, parse_spec
from qf_test_utils import EXAMPLE


def test_security_derived_values():
    s = Security(id="X", issuer="I", market_value=1_000_000, haircut=0.1, cost_bps=10)
    assert s.coverage == pytest.approx(900_000)
    assert s.cost == pytest.approx(1_000)


def test_example_loads_and_validates():
    spec = load_spec(EXAMPLE)
    assert spec.problem == "collateral_allocation"
    assert len(spec.inventory) == 10
    assert spec.total_available_coverage > spec.constraints.required_collateral


def test_rejects_unknown_problem():
    with pytest.raises(SpecError):
        parse_spec({"problem": "portfolio", "constraints": {"required_collateral": 1}})


def test_rejects_duplicate_ids():
    inv = [
        {"id": "A", "issuer": "I", "market_value": 1, "haircut": 0.0, "cost_bps": 1},
        {"id": "A", "issuer": "I", "market_value": 1, "haircut": 0.0, "cost_bps": 1},
    ]
    with pytest.raises(SpecError):
        parse_spec(
            {
                "problem": "collateral_allocation",
                "constraints": {"required_collateral": 1},
                "inventory": inv,
            }
        )


def _spec_with_concentration(concentration: dict) -> dict:
    return {
        "problem": "collateral_allocation",
        "constraints": {"required_collateral": 1_000_000, "concentration": concentration},
    }


def test_rejects_invalid_concentration_attribute():
    with pytest.raises(SpecError) as exc:
        parse_spec(_spec_with_concentration({"sector": 0.3}))
    assert "not a supported grouping attribute" in str(exc.value)


def test_rejects_out_of_range_concentration():
    with pytest.raises(SpecError):
        parse_spec(_spec_with_concentration({"issuer": 1.5}))


def test_rejects_excessive_max_effective_qubits():
    # Guards against an O(n^2) QUBO built before the planner's budget check.
    with pytest.raises(SpecError):
        parse_spec(
            {
                "problem": "collateral_allocation",
                "constraints": {"required_collateral": 1_000_000},
                "execution_policy": {"max_effective_qubits": 100_000},
            }
        )


def test_rejects_bad_haircut():
    with pytest.raises(SpecError):
        parse_spec(
            {
                "problem": "collateral_allocation",
                "constraints": {"required_collateral": 1},
                "inventory": [
                    {"id": "A", "issuer": "I", "market_value": 1, "haircut": 1.0, "cost_bps": 1}
                ],
            }
        )


def test_extra_field_forbidden():
    with pytest.raises(SpecError):
        parse_spec(
            {
                "problem": "collateral_allocation",
                "constraints": {"required_collateral": 1},
                "nonsense": 42,
            }
        )


def test_load_missing_file():
    with pytest.raises(SpecError) as exc:
        load_spec("/nonexistent/spec.yaml")
    assert "not found" in str(exc.value)


def test_load_invalid_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("problem: [unclosed")
    with pytest.raises(SpecError) as exc:
        load_spec(p)
    assert "Invalid YAML" in str(exc.value)


def test_load_empty_file(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    with pytest.raises(SpecError):
        load_spec(p)


def test_load_non_mapping(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(SpecError):
        load_spec(p)


def test_validation_error_does_not_leak_full_input():
    # The message should reference the field, not dump every security value.
    with pytest.raises(SpecError) as exc:
        parse_spec(_spec_with_concentration({"issuer": 5.0}))
    assert "concentration" in str(exc.value)
