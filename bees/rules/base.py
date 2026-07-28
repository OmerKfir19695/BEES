"""Tier-1 physics rule layer for BEES.

A Rule is a small, citable, A/B-toggleable transformation of a reaction's
kinetics or thermodynamics, applied after catpred / equilibrator have run.

Borrows RMG-Py's *metadata* conventions: `reference`, `reference_type`, and
the rank concept (omitted for now, see Rule). The execution model differs:
RMG's Entry is pure data consumed by a per-family engine in family.py;
BEES's Rule is a Strategy object with its own apply() method. This is
deliberate — Tier-1 rules each have distinct logic (hydrophobic Km scaling,
decarboxylation flag, kcat ceiling), so a shared engine would just dispatch
on rule.kind. `designed_from` is a BEES-specific provenance field with no
RMG equivalent; it powers the held-out generalization check in
`bees validate-rule`.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional, Sequence

logger = logging.getLogger("BEES.rules")


# Mutable fields the rule layer may modify. The snapshot/restore machinery in
# RuleRegistry copies these out on first sight of a reaction and writes them
# back at the start of every apply_all() call so multiplicative rules do not
# compound across iterations.
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
    """Stable, hashable key for a reaction.

    Imported lazily to avoid a load-time dependency cycle (bees.exporter pulls
    in many BEES modules). Falls back to id(reaction) if the canonical
    signature helper is unavailable (e.g. during unit tests with hand-built
    objects that don't quack like a GeneratedReaction).
    """
    try:
        from bees.exporter import reaction_signature
        return reaction_signature(reaction)
    except Exception:
        return id(reaction)


# ---------------------------------------------------------------------------
# Structured citation (matches RMG's Reference object shape)
# ---------------------------------------------------------------------------

# NOTE: RMG splits this across a class hierarchy (Reference base +
# Article/Book/Thesis subclasses) so journal/volume/pages only exist on
# Article. We collapse it into one flat dataclass: BEES has few enough
# citations that a union-of-fields record is more readable than a hierarchy.
# Field names and types (authors=list[str], year=str) match RMG so a future
# bibliography renderer can consume both.
@dataclass(frozen=True)
class Reference:
    authors: tuple[str, ...]                # ("Tanford, C.",) or ("Ruppe, A.", "Fox, J.")
    title: str
    year: str                               # str, not int — supports "2003a", "in press"
    journal: Optional[str] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None

    def short(self) -> str:
        first_author = self.authors[0] if self.authors else "Anon."
        return f"{first_author} ({self.year})"


# RMG uses "theoretical" / "experimental" / "review". "textbook" is a BEES
# addition: many universal biology rules cite canonical references
# (Tanford's *Hydrophobic Effect*, Lehninger, etc.) that are neither
# single experiments nor meta-reviews.
REFERENCE_TYPES = frozenset({"theoretical", "experimental", "review", "textbook"})


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    name: str
    description: str
    reference: Reference
    reference_type: str
    # Heterogeneous on purpose: rules may carry floats (energies), ints
    # (counts), strings (identifiers), or bools (flags).
    params: dict[str, object] = field(default_factory=dict)
    designed_from: tuple[str, ...] = ()
    enabled: bool = True
    # "law"   — universal physical-chemistry correction, always on (engine).
    # "calibration" — fitted/empirical correction for a specific system; off
    #   unless a model input declares it via settings.calibrations. The tag is
    #   what keeps a fit from silently masquerading as a law.
    kind: str = "law"

    # `rank` (RMG-style trust scale) is intentionally omitted from this cut.
    # Add it here and sort in RuleRegistry if cross-rule priority is ever
    # needed.

    def __post_init__(self) -> None:
        if self.reference_type not in REFERENCE_TYPES:
            raise ValueError(
                f"reference_type={self.reference_type!r} not in {sorted(REFERENCE_TYPES)}"
            )

    def applies_to(self, reaction: object) -> bool:
        """Override in subclass. Default: always fires."""
        return True

    def apply(self, reaction: object) -> None:
        """Mutate reaction.kinetics / reaction.thermo in place.

        Subclasses must override. RuleRegistry calls applies_to() first, so
        defensive re-checking inside apply() is unnecessary.
        """
        raise NotImplementedError(f"{type(self).__name__}.apply not implemented")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class RuleRegistry:
    """Ordered collection of rules; applies them to every reaction.

    Today: insertion-order priority (no rank). If two rules ever overwrite
    the same field, the second one wins — that is a design smell and the
    signal to add a rank attribute and sort.
    """

    def __init__(self) -> None:
        self._rules: list[Rule] = []
        # baseline snapshot keyed by reaction_signature; lifecycle in apply_all
        self._baselines: dict[Any, dict[str, dict[str, Any]]] = {}

    # -- registration --------------------------------------------------------

    def register(self, rule: Rule, *, before: Optional[str] = None) -> Rule:
        if any(r.name == rule.name for r in self._rules):
            raise ValueError(f"rule {rule.name!r} already registered")
        # `before` lets a rule registered later (e.g. a calibration in a separate
        # module imported after physics_rules) still slot ahead of a named rule,
        # so application order is independent of import order. Appends if the
        # named anchor isn't present yet.
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

    # -- application ---------------------------------------------------------

    def apply_all(self, reactions: Iterable[object]) -> None:
        """Idempotent rule application.

        Lifecycle, per call:
          1. Drop baselines for reactions no longer present (no memory leak
             when reactions get pruned across enlarger iterations).
          2. Snapshot each reaction the first time we see it.
          3. Restore every reaction from its baseline so any prior mutation
             by this rule layer is wiped before this pass.
          4. Apply enabled rules in registration order.

        Step 3 is what makes multiplicative rules (e.g. hydrophobic Km
        scaling) safe under the enlarger loop's repeated calls.
        """
        # Materialize once so multi-rule iteration doesn't exhaust a generator.
        reactions = list(reactions)
        current_keys = {_reaction_key(r) for r in reactions}

        # 1. drop stale baselines
        self._baselines = {
            k: v for k, v in self._baselines.items() if k in current_keys
        }

        for rxn in reactions:
            key = _reaction_key(rxn)
            # 2. snapshot on first sight
            if key not in self._baselines:
                self._baselines[key] = _snapshot_reaction(rxn)
            # 3. restore from baseline before applying rules
            _restore_reaction(rxn, self._baselines[key])

        # 4. apply enabled rules in registration order over the restored set
        for rule in self.enabled_rules():
            for rxn in reactions:
                if rule.applies_to(rxn):
                    rule.apply(rxn)
                    logger.debug("applied %s to %s", rule.name, rxn)

    # -- A/B validation support ---------------------------------------------

    def with_rule_disabled(self, name: str) -> "RuleRegistry":
        """Return a copy of the registry with one rule disabled.

        Used by `bees validate-rule <name>`: run all benchmarks against this
        registry vs the live one, then compute per-enzyme deltas. Benchmarks
        in `rule.designed_from` are in-sample (sanity); the rest are
        held-out (the actual test of generalization).
        """
        if not any(r.name == name for r in self._rules):
            raise KeyError(name)
        clone = RuleRegistry()
        for r in self._rules:
            clone._rules.append(
                dataclasses.replace(r, enabled=(r.enabled and r.name != name))
            )
        return clone


# Global singleton. Rules self-register at import time via load_physics_rules().
RULES = RuleRegistry()
