"""Unit tests for RuleRegistry.configure_calibrations.

Verifies that settings.calibrations enables exactly the named calibrations,
never a physics law or an unknown name, and that state does not leak across
calls (the global-singleton reset).
"""
from __future__ import annotations

import pytest

from bees.rules import Reference, Rule, RuleRegistry


def _rule(name: str, kind: str, enabled: bool = True) -> Rule:
    return Rule(
        name=name,
        description="test",
        reference=Reference(authors=("Test",), title="t", year="2024"),
        reference_type="theoretical",
        kind=kind,
        enabled=enabled,
    )


def _registry() -> RuleRegistry:
    reg = RuleRegistry()
    reg.register(_rule("a_law", "law", enabled=True))
    reg.register(_rule("cal_one", "calibration", enabled=False))
    reg.register(_rule("cal_two", "calibration", enabled=False))
    return reg


def _enabled(reg):
    return {r.name: r.enabled for r in reg}


def test_enables_exactly_named_calibration():
    reg = _registry()
    reg.configure_calibrations(["cal_one"])
    assert _enabled(reg) == {"a_law": True, "cal_one": True, "cal_two": False}


def test_empty_disables_all_calibrations_law_untouched():
    reg = _registry()
    reg.configure_calibrations([])
    assert _enabled(reg) == {"a_law": True, "cal_one": False, "cal_two": False}


def test_none_is_treated_as_empty():
    reg = _registry()
    reg.configure_calibrations(None)
    assert all(not r.enabled for r in reg if r.kind == "calibration")


def test_law_name_rejected():
    reg = _registry()
    with pytest.raises(ValueError, match="always-on physics law"):
        reg.configure_calibrations(["a_law"])


def test_unknown_name_rejected_lists_available():
    reg = _registry()
    with pytest.raises(ValueError, match="Unknown calibration"):
        reg.configure_calibrations(["nope"])


def test_no_leak_across_calls():
    reg = _registry()
    reg.configure_calibrations(["cal_one", "cal_two"])
    assert reg.by_name("cal_one").enabled and reg.by_name("cal_two").enabled
    # A subsequent run that lists nothing must turn them back off.
    reg.configure_calibrations([])
    assert not reg.by_name("cal_one").enabled
    assert not reg.by_name("cal_two").enabled


def test_validation_failure_leaves_state_unchanged():
    # An invalid request should raise before flipping any enabled flags.
    reg = _registry()
    reg.configure_calibrations(["cal_one"])
    with pytest.raises(ValueError):
        reg.configure_calibrations(["cal_two", "a_law"])
    # cal_one still enabled, cal_two still disabled (no partial application).
    assert reg.by_name("cal_one").enabled
    assert not reg.by_name("cal_two").enabled
