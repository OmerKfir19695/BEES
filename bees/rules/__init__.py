from __future__ import annotations

from typing import Iterable

from bees.rules.base import (
    REFERENCE_TYPES,
    RULES,
    Reference,
    Rule,
    RuleRegistry,
)


def load_rules() -> RuleRegistry:
    """Return the global registry populated with physics laws AND calibrations.

    Triggers the side-effect imports of `bees.rules.physics_rules` (laws) and
    `bees.rules.calibrations` (fitted, opt-in corrections), whose module bodies
    register each Rule on the global `RULES`. The imports are kept *inside* this
    function to avoid a load-time import cycle (the rule modules import from
    `bees.rules.base`, re-exported here). Idempotent: Python caches imports.

    Calibrations register with `enabled=False`; a run turns specific ones on via
    `configure_calibrations` (driven by `settings.calibrations`).
    """
    try:
        import bees.rules.physics_rules  # noqa: F401  (import for side-effects)
        import bees.rules.calibrations  # noqa: F401  (import for side-effects)
    except ImportError:
        pass
    return RULES


# Back-compat alias — the registry now carries calibrations too, so `load_rules`
# is the clearer name, but existing callers import `load_physics_rules`.
load_physics_rules = load_rules


def configure_calibrations(registry: RuleRegistry, names: Iterable[str]) -> None:
    """Enable exactly the named calibrations on `registry`; disable the rest.

    For every `kind=="calibration"` rule, set `.enabled = (rule.name in names)`.
    This is what makes a run reproducible from its declared inputs and closes the
    global-singleton leak (a calibration enabled by a prior run in the same
    process is turned back off here unless the current run lists it). Physics laws
    are never touched.

    Raises ValueError with an explicit message if a requested name is unknown or
    names a physics law (which is always-on and cannot be toggled this way).
    """
    requested = list(names or [])
    by_name = {r.name: r for r in registry}
    calibration_names = sorted(r.name for r in registry if getattr(r, "kind", "law") == "calibration")
    for name in requested:
        rule = by_name.get(name)
        if rule is None:
            raise ValueError(
                f"Unknown calibration {name!r}. Available calibrations: {calibration_names}"
            )
        if getattr(rule, "kind", "law") != "calibration":
            raise ValueError(
                f"{name!r} is an always-on physics law, not a calibration; it cannot be "
                f"toggled via settings.calibrations. Available calibrations: {calibration_names}"
            )
    requested_set = set(requested)
    for rule in registry:
        if getattr(rule, "kind", "law") == "calibration":
            rule.enabled = rule.name in requested_set


__all__ = [
    "REFERENCE_TYPES",
    "RULES",
    "Reference",
    "Rule",
    "RuleRegistry",
    "load_rules",
    "load_physics_rules",
    "configure_calibrations",
]
