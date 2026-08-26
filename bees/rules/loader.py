"""Load and populate the global rule registry."""

from __future__ import annotations

from bees.rules.base import RULES, RuleRegistry


def load_rules() -> RuleRegistry:
    """Return the global registry populated with physics laws and calibrations.

    Side-effect imports register each Rule on ``RULES``. Imports stay inside this
    function to avoid load-time cycles with ``bees.rules.base``. Idempotent.
    """
    try:
        import bees.rules.physics_rules  # noqa: F401
        import bees.rules.calibrations  # noqa: F401
    except ImportError:
        pass
    return RULES
