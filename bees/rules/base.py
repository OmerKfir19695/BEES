"""Tier-1 physics rule layer for BEES."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional, Sequence

logger = logging.getLogger("BEES.rules")

# Snapshot/restore these so multiplicative rules do not compound across iterations.
_MUTABLE_KINETICS_FIELDS: tuple[str, ...] = ("kcat", "km", "km_per_substrate")
_MUTABLE_THERMO_FIELDS: tuple[str, ...] = (
    "dgr_prime_kJmol",
    "sigma_kJmol",
    "keq",
    "kcat_rev",
    "irreversible",
    "source",
)

def _snapshot_reaction(reaction: Any) -> dict[str, dict[str, Any]]:
    """Copy the rule-mutable fields of a reaction's kinetics/thermo into a dict."""
    snap: dict[str, dict[str, Any]] = {}
    kin = getattr(reaction, "kinetics", None)
    if kin is not None:
        snap["kinetics"] = {
            f: copy.deepcopy(getattr(kin, f, None)) for f in _MUTABLE_KINETICS_FIELDS
        }
    thermo = getattr(reaction, "thermo", None)
    if thermo is not None:
        snap["thermo"] = {
            f: copy.deepcopy(getattr(thermo, f, None)) for f in _MUTABLE_THERMO_FIELDS
        }
    return snap

def _restore_reaction(reaction: Any, snap: dict[str, dict[str, Any]]) -> None:
    """Write the snapshotted kinetics/thermo fields back onto a reaction."""
    kin = getattr(reaction, "kinetics", None)
    if kin is not None and "kinetics" in snap:
        for f, v in snap["kinetics"].items():
            setattr(kin, f, copy.deepcopy(v))
    thermo = getattr(reaction, "thermo", None)
    if thermo is not None and "thermo" in snap:
        for f, v in snap["thermo"].items():
            setattr(thermo, f, copy.deepcopy(v))

def _reaction_key(reaction: Any) -> Any:
    """Stable hashable key. Lazy import avoids exporter cycle; falls back to id()."""
    try:
        from bees.exporter import reaction_signature
        return reaction_signature(reaction)
    except Exception:
        return id(reaction)

@dataclass(frozen=True)
class Reference:
    authors: tuple[str, ...]
    title: str
    year: str  # str, not int — supports "2003a", "in press"
    journal: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None

REFERENCE_TYPES = frozenset({"theoretical", "experimental", "review", "textbook"})

@dataclass
class Rule:
    name: str
    description: str
    reference: Reference
    reference_type: str
    params: dict[str, object] = field(default_factory=dict)
    enabled: bool = True
    kind: str = "law"  # "law" always on; "calibration" opt-in via settings.calibrations

    def __post_init__(self) -> None:
        if self.reference_type not in REFERENCE_TYPES:
            raise ValueError(
                f"reference_type={self.reference_type!r} not in {sorted(REFERENCE_TYPES)}"
            )

    def applies_to(self, reaction: object) -> bool:
        """Override in subclass. Default: always fires."""
        return True

    def apply(self, reaction: object) -> None:
        """Mutate reaction.kinetics / reaction.thermo in place."""
        raise NotImplementedError(f"{type(self).__name__}.apply not implemented")

class RuleRegistry:
    """Ordered collection of rules; insertion-order priority (no rank)."""

    def __init__(self) -> None:
        self._rules: list[Rule] = []
        self._baselines: dict[Any, dict[str, dict[str, Any]]] = {}

    def register(self, rule: Rule, *, before: Optional[str] = None) -> Rule:
        if any(r.name == rule.name for r in self._rules):
            raise ValueError(f"rule {rule.name!r} already registered")
        # `before` slots a later-imported rule ahead of a named anchor (order ≠ import order).
        if before is not None:
            for i, r in enumerate(self._rules):
                if r.name == before:
                    self._rules.insert(i, rule)
                    return rule
        self._rules.append(rule)
        return rule

    def __iter__(self) -> Iterator[Rule]:
        return iter(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    def by_name(self, name: str) -> Rule:
        for r in self._rules:
            if r.name == name:
                return r
        raise KeyError(name)

    def enabled_rules(self) -> Sequence[Rule]:
        return [r for r in self._rules if r.enabled]

    def apply_all(self, reactions: Iterable[object]) -> None:
        """Restore baselines then apply enabled rules."""
        reactions = list(reactions)
        current_keys = {_reaction_key(r) for r in reactions}

        self._baselines = {
            k: v for k, v in self._baselines.items() if k in current_keys
        }

        for rxn in reactions:
            key = _reaction_key(rxn)
            if key not in self._baselines:
                self._baselines[key] = _snapshot_reaction(rxn)
            _restore_reaction(rxn, self._baselines[key])

        for rule in self.enabled_rules():
            for rxn in reactions:
                if rule.applies_to(rxn):
                    rule.apply(rxn)
                    logger.debug("applied %s to %s", rule.name, rxn)

    def configure_calibrations(self, names: Optional[Iterable[str]]) -> None:
        """Enable exactly the named calibrations; disable the rest.

        Physics laws are never touched. Clears prior-run leakage on the global
        registry when called each enlarger iteration with ``settings.calibrations``.
        """
        requested = list(names or [])
        by_name = {r.name: r for r in self._rules}
        calibration_names = sorted(
            r.name for r in self._rules if getattr(r, "kind", "law") == "calibration"
        )
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
        for rule in self._rules:
            if getattr(rule, "kind", "law") == "calibration":
                rule.enabled = rule.name in requested_set

RULES = RuleRegistry()
